from collections import Counter
from trade_utils import load_trade_events

rows = load_trade_events()
entries = [r for r in rows if r.get("event") == "ENTRY"]
if not entries:
    print("No ENTRY events found.")
    raise SystemExit(0)

cnt = Counter()
for r in entries:
    pair = r.get("pair","?")
    strat = r.get("strategy_name") or r.get("strategy_id") or "unknown"
    cnt[(strat,pair)] += 1

print("Strategy ↔ Pair (entry counts):")
for (strat,pair),n in cnt.most_common(30):
    print(f"{strat} | {pair}: {n}")
