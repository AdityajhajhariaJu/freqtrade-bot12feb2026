import json, time, subprocess
import ccxt
from pathlib import Path

CONFIG_PATH = "/opt/multi-strat-engine/config.binance_futures_live.json"
STATE_PATH = "/opt/multi-strat-engine/reports/.pipeline_health_state.json"

# Read-only health checks: latency, data fetch, balance, open positions, tp/sl visibility

def load_state():
    if Path(STATE_PATH).exists():
        return json.loads(Path(STATE_PATH).read_text())
    return {"last_run": 0}


def save_state(state):
    Path(STATE_PATH).write_text(json.dumps(state))


def main():
    state = load_state()
    state["last_run"] = int(time.time())

    cfg = json.loads(Path(CONFIG_PATH).read_text())
    ex = ccxt.binance({
        "apiKey": cfg["exchange"]["key"],
        "secret": cfg["exchange"]["secret"],
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    })
    report = {"timestamp": int(time.time()), "errors": []}

    # API latency
    try:
        t0 = time.time()
        ex.fetch_time()
        report["api_latency_ms"] = int((time.time() - t0) * 1000)
    except Exception as e:
        report["errors"].append(f"time_fetch: {e}")

    # Balance
    try:
        bal = ex.fetch_balance()
        report["usdt_free"] = bal.get('USDT',{}).get('free',0)
        report["usdt_total"] = bal.get('USDT',{}).get('total',0)
    except Exception as e:
        report["errors"].append(f"balance: {e}")

    # Positions + TP/SL visibility
    try:
        positions = ex.fetch_positions()
        open_pos = []
        missing = []
        for p in positions:
            amt = float(p.get('contracts') or p.get('contractSize') or 0)
            if amt == 0: continue
            sym = p.get('info',{}).get('symbol') or p.get('symbol')
            side = p.get('side') or p.get('info',{}).get('positionSide')
            open_pos.append({"symbol": sym, "side": side, "amt": amt})
            try:
                orders = ex.fetch_open_orders(symbol=sym)
                ro = [o for o in orders if o.get('reduceOnly')]
                if len(ro) < 2:
                    missing.append(sym)
            except Exception as e:
                missing.append(f"{sym} (order check err)")
        report["open_positions"] = open_pos
        # User request: suppress tp_sl_missing alerts (keep for info only)
        report["tp_sl_missing_ignored"] = missing
        report["tp_sl_missing"] = []
    except Exception as e:
        report["errors"].append(f"positions: {e}")

    # Service status
    try:
        out = subprocess.check_output(['systemctl','--user','is-active','multistrat.service'], text=True).strip()
        report["service_active"] = (out == 'active')
    except Exception as e:
        report["errors"].append(f"service_status: {e}")

    print(json.dumps(report, indent=2))
    save_state(state)

if __name__ == "__main__":
    main()
