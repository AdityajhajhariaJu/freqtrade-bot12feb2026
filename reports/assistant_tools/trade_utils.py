import csv
from pathlib import Path

TRADE_EVENTS = Path("/opt/multi-strat-engine/trade_events.csv")

COLUMNS = ["ts","event","pair","side","price","qty","strategy_id","strategy_name","engine","notes"]

def load_trade_events(limit=None):
    if not TRADE_EVENTS.exists():
        return []
    rows = []
    with TRADE_EVENTS.open() as f:
        r = csv.reader(f)
        for row in r:
            if not row:
                continue
            # pad to 10 cols
            if len(row) < len(COLUMNS):
                row = row + [""]*(len(COLUMNS)-len(row))
            d = dict(zip(COLUMNS, row[:len(COLUMNS)]))
            rows.append(d)
    if limit:
        rows = rows[-limit:]
    return rows
