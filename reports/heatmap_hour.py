
import json, time, csv
import ccxt
from pathlib import Path
from datetime import datetime, timezone, timedelta

CONFIG_PATH = "/opt/multi-strat-engine/config.binance_futures_live.json"
OUT_CSV = "/opt/multi-strat-engine/reports/heatmap_hourly.csv"

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
    data = ex.fapiPrivateGetIncome({"incomeType": "REALIZED_PNL", "startTime": start_ms, "endTime": end_ms, "limit": 1000})
    buckets = {}
    ist = timezone(timedelta(hours=5, minutes=30))
    for row in data:
        ts = int(row.get('time',0))
        dt = datetime.fromtimestamp(ts/1000, tz=ist)
        key = dt.strftime('%Y-%m-%d %H:00')
        buckets[key] = buckets.get(key, 0.0) + float(row.get('income',0))
    with open(OUT_CSV, 'w', newline='') as f:
        w=csv.writer(f)
        w.writerow(["hour_start_ist","hour_ist","realized_pnl","last_updated_ist"])
        last_updated = datetime.now(ist).isoformat()
        for key in sorted(buckets.keys()):
            hour = key.split(' ')[1].split(':')[0]
            w.writerow([key, hour, buckets[key], last_updated])
    print(OUT_CSV)

if __name__ == '__main__':
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=7)
    args=ap.parse_args()
    main(args.days)
