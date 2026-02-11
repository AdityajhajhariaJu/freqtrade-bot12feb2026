import csv
from pathlib import Path
from datetime import datetime, timezone, timedelta

POST_TRADE = Path("/opt/multi-strat-engine/reports/post_trade_events.csv")

if not POST_TRADE.exists():
    print("No post_trade_events.csv found yet. Run post_trade_pipeline first.")
    raise SystemExit(0)

# last 24h
cutoff = datetime.now(timezone.utc) - timedelta(days=1)

rows = []
with POST_TRADE.open() as f:
    r = csv.DictReader(f)
    for row in r:
        try:
            dt = datetime.fromisoformat(row["datetime_utc"])
        except Exception:
            continue
        if dt >= cutoff:
            rows.append(row)

if not rows:
    print("No PnL events in last 24h.")
    raise SystemExit(0)

pnl = sum(float(r.get("income",0) or 0) for r in rows)
by_type = {}
by_symbol = {}
for r in rows:
    it = r.get("income_type","unknown")
    by_type[it] = by_type.get(it,0)+float(r.get("income",0) or 0)
    sym = r.get("symbol","unknown")
    by_symbol[sym] = by_symbol.get(sym,0)+float(r.get("income",0) or 0)

print(f"24h PnL: {pnl:.4f}")
print("By type:")
for k,v in sorted(by_type.items(), key=lambda x: -abs(x[1])):
    print(f"  {k}: {v:.4f}")
print("Top symbols:")
for k,v in sorted(by_symbol.items(), key=lambda x: -abs(x[1]))[:5]:
    print(f"  {k}: {v:.4f}")
