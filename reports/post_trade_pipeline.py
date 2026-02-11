import json, time, csv
import ccxt
from pathlib import Path
from datetime import datetime, timezone

CONFIG_PATH = "/opt/multi-strat-engine/config.binance_futures_live.json"
STATE_PATH = "/opt/multi-strat-engine/reports/.post_trade_state.json"
OUT_CSV = "/opt/multi-strat-engine/reports/post_trade_events.csv"

# Tracks REALIZED_PNL + COMMISSION + FUNDING_FEE events since last run

def load_state():
    if Path(STATE_PATH).exists():
        return json.loads(Path(STATE_PATH).read_text())
    return {"last_ms": 0}

def save_state(state):
    Path(STATE_PATH).write_text(json.dumps(state))

def load_entry_history():
    events = Path('/opt/multi-strat-engine/trade_events.csv')
    hist = {}
    if not events.exists():
        return hist
    import csv
    with events.open() as f:
        r=csv.DictReader(f)
        for row in r:
            if row.get('event') != 'ENTRY':
                continue
            pair = row.get('pair')
            ts = int(float(row.get('ts',0))) * 1000
            hist.setdefault(pair, []).append({
                'ts': ts,
                'strategy_id': row.get('strategy_id') or 'unknown',
                'strategy_name': row.get('strategy_name') or 'unknown',
                'side': row.get('side') or ''
            })
    for p in hist:
        hist[p] = sorted(hist[p], key=lambda x: x['ts'])
    return hist


def map_strategy(entry_hist, sym, t_ms):
    lst = entry_hist.get(sym, [])
    # latest entry before income time
    best = None
    for e in lst:
        if e['ts'] <= t_ms:
            best = e
        else:
            break
    return best or {'strategy_id':'unknown','strategy_name':'unknown','side':''}


def main():
    cfg = json.loads(Path(CONFIG_PATH).read_text())
    entry_hist = load_entry_history()
    ex = ccxt.binance({
        "apiKey": cfg["exchange"]["key"],
        "secret": cfg["exchange"]["secret"],
        "enableRateLimit": True,
        "options": {"defaultType":"future"},
    })
    state = load_state()
    last_ms = int(state.get("last_ms", 0))
    now_ms = int(time.time()*1000)
    events = []
    for it in ["REALIZED_PNL","COMMISSION","FUNDING_FEE"]:
        data = ex.fapiPrivateGetIncome({"incomeType": it, "startTime": last_ms or (now_ms-6*3600*1000), "endTime": now_ms, "limit": 1000})
        for row in data:
            sym = row.get("symbol")
            t_ms = int(row.get("time",0))
            strat = map_strategy(entry_hist, sym, t_ms)
            events.append({
                "time": t_ms,
                "symbol": sym,
                "incomeType": it,
                "income": float(row.get("income",0)),
                "asset": row.get("asset"),
                "strategy_id": strat.get('strategy_id','unknown'),
                "strategy_name": strat.get('strategy_name','unknown'),
                "side": strat.get('side',''),
            })
    events = sorted({(e["time"], e["symbol"], e["incomeType"], e["income"]): e for e in events}.values(), key=lambda x: x["time"])

    if events:
        fpath = Path(OUT_CSV)
        exists = fpath.exists()
        with fpath.open('a', newline='') as f:
            w = csv.writer(f)
            if not exists:
                w.writerow(["datetime_utc","symbol","income_type","income","asset","strategy_id","strategy_name","side"])
            for e in events:
                dt = datetime.fromtimestamp(e["time"]/1000, tz=timezone.utc).isoformat()
                w.writerow([dt, e["symbol"], e["incomeType"], e["income"], e["asset"], e.get('strategy_id','unknown'), e.get('strategy_name','unknown'), e.get('side','')])
        last_ms = max(e["time"] for e in events)
        state["last_ms"] = last_ms
        save_state(state)
        print(f"Wrote {len(events)} events. last_ms={last_ms}")
    else:
        print("No new events")

if __name__ == "__main__":
    main()
