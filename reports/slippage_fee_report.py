
import json, time, csv
import ccxt
from pathlib import Path

CONFIG_PATH = "/opt/multi-strat-engine/config.binance_futures_live.json"
OUT_CSV = "/opt/multi-strat-engine/reports/slippage_fees.csv"
PAIRS = ["ETH/USDT","SOL/USDT","LTC/USDT","DOGE/USDT","AVAX/USDT","NEAR/USDT","INJ/USDT","LINK/USDT","BNB/USDT"]

def main(days=7):
    cfg = json.loads(Path(CONFIG_PATH).read_text())
    ex = ccxt.binance({
        "apiKey": cfg["exchange"]["key"],
        "secret": cfg["exchange"]["secret"],
        "enableRateLimit": True,
        "options": {"defaultType":"future"},
    })
    end_ms = int(time.time()*1000)
    start_ms = end_ms - days*24*3600*1000
    rows = []
    for sym in PAIRS:
        trades = ex.fetch_my_trades(sym, since=start_ms)
        # prefetch 1m candles for slippage estimation
        ohlcv = ex.fetch_ohlcv(sym, timeframe='1m', since=start_ms, limit=1000)
        # map minute -> close
        closes = {c[0]: c[4] for c in ohlcv}
        for t in trades:
            ts = t['timestamp']
            minute = ts - (ts % 60000)
            close = closes.get(minute)
            if close is None: continue
            price = t['price']
            side = t['side']
            # slippage bps: positive means worse execution
            if side == 'buy':
                slip = (price - close) / close * 10000
            else:
                slip = (close - price) / close * 10000
            fee = (t.get('fee') or {}).get('cost', 0)
            rows.append([t['datetime'], sym, side, price, close, round(slip,3), fee])
    with open(OUT_CSV,'w',newline='') as f:
        w=csv.writer(f)
        w.writerow(["datetime","symbol","side","trade_price","candle_close","slippage_bps","fee"])
        w.writerows(rows)
    print(OUT_CSV)

if __name__ == '__main__':
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=7)
    args=ap.parse_args()
    main(args.days)
