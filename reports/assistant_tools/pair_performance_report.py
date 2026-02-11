import csv
from pathlib import Path

CSV = Path("/opt/multi-strat-engine/reports/pair_performance.csv")
if not CSV.exists():
    print("pair_performance.csv not found. Run pair_performance.py first.")
    raise SystemExit(0)

with CSV.open() as f:
    print(f.read().strip())
