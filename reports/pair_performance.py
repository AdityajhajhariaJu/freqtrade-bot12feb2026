
import json, time, csv
import ccxt
from pathlib import Path

CONFIG_PATH = "/opt/multi-strat-engine/config.binance_futures_live.json"
OUT_CSV = "/opt/multi-strat-engine/reports/pair_performance.csv"

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
    income_types = ["REALIZED_PNL","FUNDING_FEE","COMMISSION"]
    stats = {}
    for it in income_types:
        data = ex.fapiPrivateGetIncome({"incomeType": it, "startTime": start_ms, "endTime": end_ms, "limit": 1000})
        for row in data:
            sym = row.get("symbol") or "UNKNOWN"
            stats.setdefault(sym, {"realized":0.0, "funding":0.0, "fees":0.0, "total":0.0})
            val = float(row.get("income",0))
            if it == "REALIZED_PNL": stats[sym]["realized"] += val
            if it == "FUNDING_FEE": stats[sym]["funding"] += val
            if it == "COMMISSION": stats[sym]["fees"] += val
    for sym, v in stats.items():
        v["total"] = v["realized"] + v["funding"] + v["fees"]
    with open(OUT_CSV, "w", newline="") as f:
        w=csv.writer(f)
        w.writerow(["symbol","realized_pnl","funding","fees","total_pnl"])
        for sym, v in sorted(stats.items(), key=lambda x: x[1]["total"], reverse=True):
            w.writerow([sym, v["realized"], v["funding"], v["fees"], v["total"]])
    print(OUT_CSV)

if __name__ == "__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=7)
    args=ap.parse_args()
    main(args.days)
