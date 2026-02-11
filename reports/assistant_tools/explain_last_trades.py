import sys
from trade_utils import load_trade_events

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
rows = load_trade_events(limit=N)
if not rows:
    print("No trade events found.")
    raise SystemExit(0)

print(f"Last {len(rows)} trade events:")
for r in rows:
    ts = r.get("ts","?")
    ev = r.get("event","?")
    pair = r.get("pair","?")
    side = r.get("side","?")
    strat = r.get("strategy_name") or r.get("strategy_id") or "unknown"
    engine = r.get("engine") or ""
    notes = r.get("notes") or ""
    print(f"- {ts} | {ev} | {pair} {side} | {strat} {('('+engine+')') if engine else ''} | {notes}")
