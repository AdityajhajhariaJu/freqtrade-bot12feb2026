
import csv
from collections import defaultdict
from pathlib import Path

EVENTS = Path('/opt/multi-strat-engine/reports/post_trade_events.csv')
OUT = Path('/opt/multi-strat-engine/reports/strategy_performance.csv')

# Uses realized PnL events mapped to last entry strategy (best-effort).

def main():
    if not EVENTS.exists():
        print('trade_events.csv not found')
        return
    stats = defaultdict(lambda: {"wins":0, "losses":0, "pnl":0.0})
    with EVENTS.open() as f:
        r=csv.DictReader(f)
        for row in r:
            if row.get('income_type') != 'REALIZED_PNL':
                continue
            sid = row.get('strategy_id') or 'unknown'
            pnl = float(row.get('income',0))
            stats[sid]["pnl"] += pnl
            if pnl > 0: stats[sid]["wins"] += 1
            elif pnl < 0: stats[sid]["losses"] += 1
    with OUT.open('w', newline='') as f:
        w=csv.writer(f)
        w.writerow(["strategy_id","wins","losses","win_rate","pnl"])
        for sid, v in stats.items():
            total = v['wins'] + v['losses']
            win_rate = (v['wins']/total) if total else 0
            w.writerow([sid, v['wins'], v['losses'], round(win_rate,4), round(v['pnl'],6)])
    print(OUT)

if __name__ == '__main__':
    main()
