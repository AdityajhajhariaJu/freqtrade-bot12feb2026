
import json, time
import ccxt
from pathlib import Path
import subprocess

CONFIG_PATH = "/opt/multi-strat-engine/config.binance_futures_live.json"
PAIRS = ["ETHUSDT","BNBUSDT","SOLUSDT","LINKUSDT","LTCUSDT","DOGEUSDT"]

def is_bot_running():
    try:
        out = subprocess.check_output(["pgrep","-f","trade_loop.py"], text=True).strip()
        return bool(out)
    except subprocess.CalledProcessError:
        return False

def main():
    cfg = json.loads(Path(CONFIG_PATH).read_text())
    ex = ccxt.binance({
        "apiKey": cfg["exchange"]["key"],
        "secret": cfg["exchange"]["secret"],
        "enableRateLimit": True,
        "options": {"defaultType":"future"},
    })
    status = {
        "timestamp": int(time.time()),
        "bot_running": is_bot_running(),
        "positions": [],
        "tp_sl_missing": [],
    }
    try:
        bal = ex.fetch_balance()
        status["usdt_free"] = bal.get("USDT",{}).get("free",0)
    except Exception as e:
        status["balance_error"] = str(e)
    try:
        positions = ex.fetch_positions()
        for p in positions:
            amt = float(p.get('contracts') or p.get('contractSize') or 0)
            if amt == 0:
                continue
            sym = p.get('info',{}).get('symbol') or p.get('symbol')
            side = p.get('side') or p.get('info',{}).get('positionSide')
            status["positions"].append({"symbol": sym, "side": side, "amt": amt})
            # best-effort TP/SL check (API may hide conditional orders)
            try:
                orders = ex.fetch_open_orders(symbol=sym)
                ro = [o for o in orders if o.get('reduceOnly')]
                if len(ro) == 0:
                    status["tp_sl_missing"].append(sym)
            except Exception as e:
                status.setdefault("open_orders_error", {})[sym] = str(e)
    except Exception as e:
        status["positions_error"] = str(e)
    print(json.dumps(status, indent=2))

if __name__ == "__main__":
    main()
