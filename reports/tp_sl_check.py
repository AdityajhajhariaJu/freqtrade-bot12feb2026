import json, time
import ccxt
from pathlib import Path

CONFIG_PATH = "/opt/multi-strat-engine/config.binance_futures_live.json"

# Best-effort: Binance API may hide conditional orders; treat empty reduce-only as missing

def main():
    cfg = json.loads(Path(CONFIG_PATH).read_text())
    ex = ccxt.binance({
        "apiKey": cfg["exchange"]["key"],
        "secret": cfg["exchange"]["secret"],
        "enableRateLimit": True,
        "options": {"defaultType":"future"},
    })
    missing = []
    positions = ex.fetch_positions()
    for p in positions:
        amt = float(p.get('contracts') or p.get('contractSize') or 0)
        if amt == 0:
            continue
        sym = p.get('info',{}).get('symbol') or p.get('symbol')
        try:
            orders = ex.fetch_open_orders(symbol=sym)
            ro = [o for o in orders if o.get('reduceOnly')]
            if len(ro) < 2:
                missing.append(sym)
        except Exception as e:
            missing.append(f"{sym} (check error: {e})")
    if missing:
        print("MISSING:" + ", ".join(missing))
    else:
        print("OK")

if __name__ == "__main__":
    main()
