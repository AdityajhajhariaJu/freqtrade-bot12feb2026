from trade_utils import load_trade_events

rows = load_trade_events()
if not rows:
    print("No trade events found.")
    raise SystemExit(0)

entries = [r for r in rows if r.get("event") == "ENTRY"]
closes = [r for r in rows if r.get("event","").startswith("CLOSE")]

print("Eval Harness (basic):")
print(f"Total events: {len(rows)}")
print(f"Entries: {len(entries)}")
print(f"Closes: {len(closes)}")

# per strategy counts
from collections import Counter
cnt = Counter()
for r in entries:
    strat = r.get("strategy_name") or r.get("strategy_id") or "unknown"
    cnt[strat] += 1
print("Top strategies by entries:")
for strat,n in cnt.most_common(10):
    print(f"  {strat}: {n}")
