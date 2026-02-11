import sys
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: kb_search.py <keyword>")
    raise SystemExit(1)

kw = sys.argv[1].lower()
paths = ["/opt/multi-strat-engine", "/opt/freqtrade/user_data"]

hits = 0
for base in paths:
    for p in Path(base).rglob("*.py"):
        try:
            txt = p.read_text(errors='ignore')
        except Exception:
            continue
        if kw in txt.lower():
            hits += 1
            print(f"\n== {p} ==")
            # show small context
            for line in txt.splitlines():
                if kw in line.lower():
                    print(line[:300])
    for p in Path(base).rglob("*.md"):
        try:
            txt = p.read_text(errors='ignore')
        except Exception:
            continue
        if kw in txt.lower():
            hits += 1
            print(f"\n== {p} ==")
            for line in txt.splitlines():
                if kw in line.lower():
                    print(line[:300])

if hits == 0:
    print("No matches.")
