import json, time, csv, argparse
import ccxt
from datetime import datetime, timezone

CONFIG_PATH = "/opt/multi-strat-engine/config.binance_futures_live.json"
OUT_CSV = "/opt/multi-strat-engine/pnl_daily.csv"

# incomeType: REALIZED_PNL, FUNDING_FEE, COMMISSION (fees)

def ms(dt):
    return int(dt.timestamp() * 1000)


def fetch_income(ex, income_type, start_ms, end_ms):
    out = []
    params = {"incomeType": income_type, "startTime": start_ms, "endTime": end_ms, "limit": 1000}
    data = ex.fapiPrivateGetIncome(params)
    for row in data:
        out.append(row)
    return out


def main(days=7):
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    ex = ccxt.binance({
        "apiKey": cfg["exchange"]["key"],
        "secret": cfg["exchange"]["secret"],
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    })
    now = datetime.now(timezone.utc)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0) - (days-1) * (now - now.replace(hour=0, minute=0, second=0, microsecond=0))
    # build list of day starts
    day_starts = [now.replace(hour=0, minute=0, second=0, microsecond=0) - i* (now - now.replace(hour=0, minute=0, second=0, microsecond=0)) for i in range(days-1, -1, -1)]

    rows = []
    for d in day_starts:
        d_start = d
        d_end = d_start + (now - now.replace(hour=0, minute=0, second=0, microsecond=0))
        start_ms = ms(d_start)
        end_ms = ms(d_end)
        realized = sum(float(x["income"]) for x in fetch_income(ex, "REALIZED_PNL", start_ms, end_ms))
        funding = sum(float(x["income"]) for x in fetch_income(ex, "FUNDING_FEE", start_ms, end_ms))
        fees = sum(float(x["income"]) for x in fetch_income(ex, "COMMISSION", start_ms, end_ms))
        total = realized + funding + fees
        rows.append([d_start.date().isoformat(), realized, funding, fees, total])

    # write csv
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date_utc", "realized_pnl", "funding", "fees", "total_pnl"])
        for r in rows:
            w.writerow(r)

    # compute rolling avg
    totals = [r[4] for r in rows]
    avg = sum(totals) / len(totals) if totals else 0.0
    print(OUT_CSV)
    print(f"rolling_avg_{days}d: {avg:.4f} USDT")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    args = ap.parse_args()
    main(args.days)
