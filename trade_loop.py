#!/usr/bin/env python3
"""
Live trade loop for the multi-strategy engine (Binance USDT-M futures).

FIXES APPLIED:
  BUG-001: 4H scan now uses market_data_1h (1h candles) instead of market_data (1m candles)
  BUG-004: Engine tag in place_orders() now detects 4H strategies (was only 2H)
  BUG-005: Imported rsi from strategies for should_momentum_exit()
  BUG-008: Same as BUG-001 — data source consistency for 4H scan
  BUG-009: Engine tag in seed_position_meta() now detects 4H strategies
  BUG-010: Added _pending_entries set to prevent duplicate ENTRY events per pair
"""
import asyncio
import json
import time
import ccxt.async_support as ccxt
from strategies import (
    Candle, ActiveTrade, TradeSignal, run_signal_scan,
    CONFIG as STRAT_CONFIG, CONFIG_2H, CONFIG_4H,
    set_pair_cooldown, load_2h_strategies, calculate_trade_economics,
    apply_cooldown_tiered, correlation_filter,
    is_2h_strategy, is_4h_strategy,
    rsi,  # BUG-005 FIX: import rsi for should_momentum_exit()
)
from news_bias import get_news_bias
from new_strategies_4h import scan_4h_strategies, compute_12h_bias, should_allow_signal, STRATEGIES_4H

CONFIG_PATH = "/opt/multi-strat-engine/config.binance_futures_live.json"
load_2h_strategies()
PAIRS = STRAT_CONFIG["pairs"]

# User request: disable 12H bias gate
ENABLE_12H_BIAS = False
FETCH_LIMIT = 1500
TIMEFRAME = "1m"
TIMEFRAME_5M = "5m"
TIMEFRAME_15M = "15m"
TIMEFRAME_1H = "1h"
LOOP_SEC = 75
DEFAULT_MAX_AGE = 25 * 60
from new_strategies_2h import NEW_MAX_AGE
MAX_AGE = {
    "ema_scalp": 40 * 60, "bb_squeeze": 40 * 60, "macd_flip": 40 * 60,
    "atr_breakout": 40 * 60, "triple_ema": 40 * 60,
    "rsi_snap": 15 * 60, "vwap_bounce": 15 * 60, "stoch_cross": 15 * 60,
    "obv_divergence": 15 * 60, "engulfing_sr": 35 * 60,
}
MAX_AGE.update(NEW_MAX_AGE)
MAX_AGE.update({
    "weekly_vwap_trend_4h": 16 * 3600, "ichimoku_breakout_4h": 24 * 3600,
    "bb_rsi_reversion_4h": 12 * 3600, "structure_break_ob_4h": 20 * 3600,
})
DEFAULT_TPSL = {"bb_squeeze": (0.012, 0.0065)}
position_meta = {}
META_PATH = "/opt/multi-strat-engine/reports/position_meta.json"
TRADE_LOG_PATH = "/opt/multi-strat-engine/reports/trade_entry_log.csv"
TRADE_EXIT_LOG_PATH = "/opt/multi-strat-engine/reports/trade_exit_log.csv"

# BUG-010 FIX: Track pending entries to prevent duplicate ENTRY events
_pending_entries: set = set()

def load_position_meta():
    import json; from pathlib import Path
    p = Path(META_PATH)
    if not p.exists(): return {}
    try: return json.loads(p.read_text())
    except Exception: return {}

def save_position_meta():
    import json; from pathlib import Path
    try: Path(META_PATH).write_text(json.dumps(position_meta))
    except Exception: pass

_LAST_SWAP_TS = 0
_LAST_TPSL_FIX_TS = 0
_LAST_HTF_SCAN = 0

# BUG-004/009 FIX: Centralized engine tier detection
def _detect_engine(strategy_id: str) -> str:
    if is_4h_strategy(strategy_id): return "4h"
    if is_2h_strategy(strategy_id): return "2h"
    return "1m"

def log_filtered(sig, reason):
    from pathlib import Path; import csv, time
    fpath = Path('/opt/multi-strat-engine/reports/filtered_signals.csv')
    exists = fpath.exists()
    with fpath.open('a', newline='') as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["ts","pair","strategy","side","price","reason","outcome","pnl_pct","exit_price","horizon_min"])
        w.writerow([time.time(), sig.pair, sig.strategy_name, sig.side, sig.entry_price, reason, "", "", "", ""])

def log_high_conf_drop(sig, reason):
    from pathlib import Path; import csv, time
    # Capture why high-confidence candidates still didn't get executed.
    fpath = Path('/opt/multi-strat-engine/reports/high_conf_drops.csv')
    exists = fpath.exists()
    with fpath.open('a', newline='') as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["ts","pair","strategy_id","strategy_name","side","confidence","price","reason"])
        w.writerow([
            int(time.time()), sig.pair, sig.strategy_id, sig.strategy_name, sig.side,
            getattr(sig, 'confidence', ''), getattr(sig, 'entry_price', ''), reason
        ])

def normalize_trade_events_file():
    from pathlib import Path; import csv
    fpath = Path('/opt/multi-strat-engine/trade_events.csv')
    if not fpath.exists(): return
    try:
        with fpath.open() as f: first = f.readline().strip()
        if first.startswith('ts,event,'): return
        rows = []
        with fpath.open() as f:
            r = csv.reader(f)
            for row in r:
                if not row: continue
                if len(row) >= 2 and row[0] == 'ts' and row[1] == 'event': continue
                rows.append(row)
        header = ["ts","event","pair","side","qty","price","strategy_id","strategy_name","pnl","note","engine"]
        with fpath.open('w', newline='') as f:
            w = csv.writer(f); w.writerow(header)
            for row in rows:
                if len(row) < len(header): row = row + ['']*(len(header)-len(row))
                w.writerow(row[:len(header)])
    except Exception: return

def normalize_pair(p):
    if not p: return ''
    return (p.replace('/USDT:USDT','USDT').replace('/USDT','USDT'))

def _meta_key(pair):
    return normalize_pair(pair)

def _meta_get(pair):
    k = _meta_key(pair)
    return position_meta.get(k) or position_meta.get(pair) or {}

def log_event(event, pair, side, qty=None, price=None, strategy_id="", strategy_name="", pnl=None, note="", engine=""):
    from pathlib import Path; import csv, time
    fpath = Path('/opt/multi-strat-engine/trade_events.csv')
    normalize_trade_events_file()
    exists = fpath.exists(); header = None
    if exists:
        try:
            with fpath.open() as f: header = f.readline().strip().split(',')
        except Exception: header = None
    with fpath.open('a', newline='') as f:
        w=csv.writer(f)
        if not exists or not header or header[:2] != ["ts","event"]:
            w.writerow(["ts","event","pair","side","qty","price","strategy_id","strategy_name","pnl","note","engine"])
            header = ["ts","event","pair","side","qty","price","strategy_id","strategy_name","pnl","note","engine"]
        if "engine" not in header:
            note = (note + f" | engine={engine}").strip()
            w.writerow([int(time.time()), event, normalize_pair(pair), side, qty or "", price or "", strategy_id, strategy_name, pnl or "", note])
        else:
            w.writerow([int(time.time()), event, normalize_pair(pair), side, qty or "", price or "", strategy_id, strategy_name, pnl or "", note, engine])

# One-row-per-trade log (updates PnL on exit)

IST_OFFSET_SEC = 5.5 * 3600

def _ts_ist(ts):
    return int(ts + IST_OFFSET_SEC)

def _fmt_ist(ts):
    import datetime
    return datetime.datetime.utcfromtimestamp(_ts_ist(ts)).strftime('%Y-%m-%d %H:%M:%S IST')

def _ensure_trade_log_header():
    from pathlib import Path; import csv
    fpath = Path(TRADE_LOG_PATH)
    if fpath.exists():
        return
    fpath.parent.mkdir(parents=True, exist_ok=True)
    with fpath.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["trade_id","entry_ts","entry_ts_ist","pair","side","strategy_id","strategy_name","strategy_tier","confidence","entry_price","qty","note","exit_ts","exit_ts_ist","exit_price","realized_pnl"])

def _sort_trade_log_desc():
    import csv
    from pathlib import Path
    fpath = Path(TRADE_LOG_PATH)
    if not fpath.exists():
        return
    with fpath.open() as f:
        r = csv.DictReader(f)
        rows = list(r)
        fields = r.fieldnames
    rows.sort(key=lambda x: int(float(x.get('entry_ts') or 0)), reverse=True)
    with fpath.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)

def rebuild_trade_exit_log():
    import csv
    from pathlib import Path
    src = Path(TRADE_LOG_PATH)
    dst = Path(TRADE_EXIT_LOG_PATH)
    if not src.exists():
        return
    with src.open() as f:
        r = csv.DictReader(f)
        fields = r.fieldnames
        rows = list(r)
    closed = []
    for row in rows:
        if (row.get('exit_ts') or '').strip() or (row.get('realized_pnl') or '').strip():
            closed.append(row)
    closed.sort(key=lambda x: int(float(x.get('exit_ts') or x.get('entry_ts') or 0)), reverse=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in closed:
            w.writerow(row)


def log_trade_entry(trade_id, ts, pair, side, strategy_id, strategy_name, tier, confidence, entry_price, qty, note):
    import csv
    _ensure_trade_log_header()
    with open(TRADE_LOG_PATH, 'a', newline='') as f:
        w = csv.writer(f)
        w.writerow([
            trade_id, int(ts), _fmt_ist(ts), normalize_pair(pair), side,
            strategy_id, strategy_name, tier, f"{confidence:.6f}",
            entry_price or "", qty or "", note or "", "", "", "", ""
        ])
    _sort_trade_log_desc()
    rebuild_trade_exit_log()


def update_trade_exit(trade_id, ts, exit_price, realized_pnl):
    import csv
    from pathlib import Path
    fpath = Path(TRADE_LOG_PATH)
    if not fpath.exists():
        return
    rows=[]
    with fpath.open() as f:
        r=csv.DictReader(f)
        rows=list(r)
        fields=r.fieldnames
    updated=False
    for row in rows:
        if row.get('trade_id') == trade_id:
            row['exit_ts'] = str(int(ts))
            row['exit_ts_ist'] = _fmt_ist(ts)
            row['exit_price'] = str(exit_price or "")
            row['realized_pnl'] = str(realized_pnl or "")
            updated=True
            break
    if not updated:
        return
    with fpath.open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    _sort_trade_log_desc()
    rebuild_trade_exit_log()

def load_keys():
    with open(CONFIG_PATH, "r") as f: cfg = json.load(f)
    api_key = cfg.get("exchange", {}).get("key"); secret = cfg.get("exchange", {}).get("secret")
    if not api_key or not secret: raise RuntimeError("API keys not found in freqtrade config")
    return api_key, secret

async def fetch_candles(exchange, pair, timeframe=TIMEFRAME, limit=None):
    try:
        ohlcv = await exchange.fetch_ohlcv(pair, timeframe=timeframe, limit=limit or FETCH_LIMIT)
        candles = [Candle(open=o[1], high=o[2], low=o[3], close=o[4], volume=o[5], timestamp=o[0]) for o in ohlcv]
        return pair, candles
    except Exception as e:
        print(f"Fetch error {pair} {timeframe}: {e}"); return pair, None

async def fetch_positions(exchange):
    try: positions = await exchange.fetch_positions()
    except Exception as e: print(f"Fetch positions error: {e}"); return []
    active = []
    for p in positions:
        amt = float(p.get('contracts') or p.get('contractSize') or 0)
        if amt == 0: continue
        pair = p.get('symbol') or p.get('info', {}).get('symbol') or p.get('id')
        side_field = p.get('side') or p.get('info', {}).get('positionSide') or ''
        side = 'LONG' if str(side_field).lower() == 'long' else 'SHORT'
        strat_id = _meta_get(pair).get('strategy_id', '')
        active.append(ActiveTrade(pair=pair, strategy_id=strat_id, side=side))
    return active

async def seed_position_meta(exchange):
    position_meta.update(load_position_meta() or {})
    try: positions = await exchange.fetch_positions()
    except Exception as e: print(f"Seed positions error: {e}"); return
    now = time.time()
    def last_entry_strategy(pair):
        from pathlib import Path; import csv
        fpath = Path('/opt/multi-strat-engine/trade_events.csv')
        if not fpath.exists(): return '', ''
        last = ('','')
        try:
            with fpath.open() as f:
                r=csv.DictReader(f)
                for row in r:
                    if row.get('event') in ('ENTRY','ENTRY_RECOVER') and row.get('pair') == normalize_pair(pair):
                        last = (row.get('strategy_id',''), row.get('strategy_name',''))
        except Exception: return '', ''
        return last
    def has_entry(pair):
        from pathlib import Path; import csv
        fpath = Path('/opt/multi-strat-engine/trade_events.csv')
        if not fpath.exists(): return False
        try:
            with fpath.open() as f:
                r=csv.DictReader(f)
                for row in r:
                    if row.get('event') in ('ENTRY','ENTRY_RECOVER') and row.get('pair') == normalize_pair(pair):
                        return True
        except Exception: return False
        return False
    for p in positions:
        amt = float(p.get('contracts') or p.get('contractSize') or 0)
        if amt == 0: continue
        pair = p.get('symbol') or p.get('info', {}).get('symbol') or p.get('id')
        mk = _meta_key(pair)
        upd_ms = float(p.get('info', {}).get('updateTime') or p.get('timestamp') or now * 1000)
        opened_at = upd_ms / 1000.0
        if mk not in position_meta:
            position_meta[mk] = {"strategy_id": "", "opened_at": opened_at}
            print(f"Seeded meta for {mk} opened_at={opened_at}")
        if not position_meta.get(mk, {}).get('strategy_id'):
            sid, sname = last_entry_strategy(pair)
            if sid: position_meta[mk]["strategy_id"] = sid
        if not has_entry(pair):
            sid = position_meta.get(mk, {}).get('strategy_id','')
            sname = last_entry_strategy(pair)[1] if sid else ''
            # BUG-009 FIX: Use centralized engine detection (was only checking 2H)
            eng = _detect_engine(sid) if sid else "1m"
            log_event("ENTRY_RECOVER", pair, "", qty=amt, price=None, strategy_id=sid, strategy_name=sname, note="Recovered open position", engine=eng)
        # BUG-010 FIX: Mark existing positions as pending
        _pending_entries.add(mk)
    save_position_meta()

async def set_oneway(exchange):
    try: await exchange.set_position_mode(False)
    except Exception as e: print(f"set_position_mode warning: {e}")

async def ensure_leverage(exchange, pair, lev):
    try: await exchange.set_leverage(lev, pair)
    except Exception as e: print(f"set_leverage warning {pair} {lev}x: {e}")

async def cancel_all_open_orders(exchange):
    try: exchange.options['warnOnFetchOpenOrdersWithoutSymbol'] = False
    except Exception: pass
    for sym in PAIRS:
        try: await exchange.cancel_all_orders(sym); print(f"cancel_all_orders called for {sym}")
        except Exception as e: print(f"cancel_all_orders error {sym}: {e}")

async def ensure_tp_sl_missing(exchange, positions):
    global _LAST_TPSL_FIX_TS
    now = time.time()
    if now - _LAST_TPSL_FIX_TS < 300: return
    _LAST_TPSL_FIX_TS = now
    for p in positions:
        amt = float(p.get('contracts') or p.get('contractSize') or 0)
        if amt == 0: continue
        pair = p.get('symbol') or p.get('info', {}).get('symbol') or p.get('id')
        side_field = p.get('side') or p.get('info', {}).get('positionSide') or ''
        side = 'LONG' if str(side_field).lower() == 'long' else 'SHORT'
        try: orders = await exchange.fetch_open_orders(pair)
        except Exception: orders = []
        has_tp = False; has_sl = False
        for o in orders:
            otype = (o.get('type') or '').lower(); info_type = (o.get('info', {}).get('type') or '').lower()
            reduce_only = o.get('reduceOnly') or o.get('info', {}).get('reduceOnly')
            if not reduce_only: continue
            if 'take_profit' in otype or 'take_profit' in info_type: has_tp = True
            if 'stop' in otype or 'stop' in info_type: has_sl = True
        if has_tp and has_sl: continue
        strat_id = position_meta.get(pair, {}).get('strategy_id')
        tp_sl = DEFAULT_TPSL.get(strat_id)
        if not tp_sl: continue
        entry = float(p.get('entryPrice') or p.get('info', {}).get('entryPrice') or 0)
        if not entry: continue
        tp_pct, sl_pct = tp_sl
        if side == 'LONG': tp = entry * (1 + tp_pct); sl = entry * (1 - sl_pct); close_side = 'sell'
        else: tp = entry * (1 - tp_pct); sl = entry * (1 + sl_pct); close_side = 'buy'
        amount = abs(amt); reduce_params = {"reduceOnly": True}
        try:
            if not has_tp:
                await exchange.create_order(pair, 'take_profit_market', close_side, amount, None, {**reduce_params, "stopPrice": tp})
                print(f"TP fixed {pair} {side} @ {tp:.6f}")
            if not has_sl:
                await exchange.create_order(pair, 'stop_market', close_side, amount, None, {**reduce_params, "stopPrice": sl})
                print(f"SL fixed {pair} {side} @ {sl:.6f}")
        except Exception as e: print(f"TP/SL fix error {pair}: {e}")

async def place_orders(exchange, signal):
    # BUG-010 FIX: Skip if pair already has a pending/active entry
    sig_mk = _meta_key(signal.pair)
    if sig_mk in _pending_entries:
        print(f"SKIP {signal.pair}: already has pending entry (BUG-010 guard)")
        return
    side = 'buy' if signal.side == 'LONG' else 'sell'
    notional = signal.trade_size * signal.leverage
    amount = notional / signal.entry_price
    params = {"type": "MARKET"}
    try: await exchange.create_order(signal.pair, 'market', side, amount, None, params)
    except Exception as e: print(f"Entry error {signal.pair} {side}: {e}"); return
    _pending_entries.add(sig_mk)  # BUG-010 FIX
    reduce_params = {"reduceOnly": True}
    try: await exchange.create_order(signal.pair, 'take_profit_market', 'sell' if signal.side == 'LONG' else 'buy', amount, None, {**reduce_params, "stopPrice": signal.tp_price})
    except Exception as e: print(f"TP error {signal.pair}: {e}")
    try: await exchange.create_order(signal.pair, 'stop_market', 'sell' if signal.side == 'LONG' else 'buy', amount, None, {**reduce_params, "stopPrice": signal.sl_price})
    except Exception as e: print(f"SL error {signal.pair}: {e}")
    ts = time.time()
    trade_id = f"{int(ts)}-{normalize_pair(signal.pair)}"
    position_meta[sig_mk] = {"strategy_id": signal.strategy_id, "opened_at": ts, "confidence": signal.confidence, "trade_id": trade_id}
    save_position_meta()
    # Set cooldown only after successful ENTRY (user request)
    try:
        set_pair_cooldown(signal.pair)
    except Exception:
        pass
    # BUG-004 FIX: Use centralized engine detection (was only checking 2H)
    engine = _detect_engine(signal.strategy_id)
    note = f"{signal.reason} | conf={signal.confidence:.4f}"
    log_event("ENTRY", signal.pair, signal.side, qty=amount, price=signal.entry_price, strategy_id=signal.strategy_id, strategy_name=signal.strategy_name, note=note, engine=engine)
    # one-row trade log (IST + confidence)
    try:
        log_trade_entry(trade_id, ts, signal.pair, signal.side, signal.strategy_id, signal.strategy_name, engine, signal.confidence, signal.entry_price, amount, signal.reason)
    except Exception:
        pass
    print(f"EXECUTED {signal.pair} {signal.side} size=${signal.trade_size:.2f} lev={signal.leverage} entry~{signal.entry_price} tp={signal.tp_price} sl={signal.sl_price} strat={signal.strategy_name} reason={signal.reason}")

async def close_position(exchange, pair, side, amount):
    close_side = 'sell' if side == 'LONG' else 'buy'
    mk = _meta_key(pair)
    entry_price = None
    try:
        positions = await exchange.fetch_positions()
        for p in positions:
            sym = p.get('symbol') or p.get('info', {}).get('symbol') or p.get('id')
            if sym == pair: entry_price = float(p.get('entryPrice') or p.get('info', {}).get('entryPrice') or 0) or None; break
    except Exception: entry_price = None
    try:
        await exchange.create_order(pair, 'market', close_side, amount, None, {"reduceOnly": True})
        close_price = None
        try:
            t = await exchange.fetch_ticker(pair)
            close_price = float(t.get('last') or t.get('close') or 0) or None
        except Exception: close_price = None
        pnl = None
        if entry_price and close_price:
            pnl = (close_price - entry_price) * amount if side == 'LONG' else (entry_price - close_price) * amount
        try:
            opened_at = _meta_get(pair).get('opened_at')
            if opened_at:
                data = await exchange.fapiPrivateGetIncome({"incomeType": "REALIZED_PNL", "startTime": int(opened_at * 1000), "endTime": int(time.time() * 1000), "limit": 1000})
                realized = sum(float(r.get('income', 0) or 0) for r in data if r.get('symbol') in (pair, pair.replace('/USDT:USDT','').replace('/USDT','')+'USDT'))
                if realized != 0: pnl = realized
        except Exception: pass
        log_event("CLOSE_MAX_AGE", pair, side, qty=amount, price=close_price, pnl=pnl, note="max_age")
        try:
            trade_id = _meta_get(pair).get('trade_id')
            if trade_id:
                update_trade_exit(trade_id, time.time(), close_price, pnl)
        except Exception:
            pass
        _pending_entries.discard(mk)  # BUG-010 FIX
        print(f"Closed {pair} {side} due to max age")
    except Exception as e: print(f"Close error {pair}: {e}")

# BUG-005 FIX: rsi is now imported from strategies at the top
def should_momentum_exit(candles, side):
    if len(candles) < 15: return False
    closes = [c.close for c in candles]
    r = rsi(closes, 14)
    last3 = candles[-3:]
    vols = [c.volume for c in last3]
    vol_down = vols[0] > vols[1] > vols[2]
    if side == "LONG":
        green = all(c.close > c.open for c in last3)
        return r >= 75 and green and vol_down
    else:
        red = all(c.close < c.open for c in last3)
        return r <= 25 and red and vol_down

def position_age_seconds(pair, p_obj):
    mk = _meta_key(pair)
    meta = _meta_get(pair)
    strat_id = meta.get('strategy_id') if meta else None
    opened_at = meta.get('opened_at') if meta else None
    if not opened_at:
        upd_ms = float(p_obj.get('info', {}).get('updateTime') or p_obj.get('timestamp') or time.time()*1000)
        opened_at = upd_ms / 1000.0
        position_meta[mk] = {"strategy_id": strat_id or "", "opened_at": opened_at}
    return strat_id, time.time() - opened_at

async def loop():
    normalize_trade_events_file()
    api_key, secret = load_keys()
    exchange = ccxt.binance({"apiKey": api_key, "secret": secret, "enableRateLimit": True, "options": {"defaultType": "future"}})
    await cancel_all_open_orders(exchange)
    await set_oneway(exchange)
    await seed_position_meta(exchange)
    try:
        while True:
            start = time.time()
            tasks_1m = [fetch_candles(exchange, p, TIMEFRAME, limit=FETCH_LIMIT) for p in PAIRS]
            tasks_5m = [fetch_candles(exchange, p, TIMEFRAME_5M, limit=FETCH_LIMIT) for p in PAIRS]
            tasks_15m = [fetch_candles(exchange, p, TIMEFRAME_15M, limit=FETCH_LIMIT) for p in PAIRS]
            tasks_1h = [fetch_candles(exchange, p, TIMEFRAME_1H, limit=500) for p in PAIRS]
            results_1m, results_5m, results_15m, results_1h = await asyncio.gather(
                asyncio.gather(*tasks_1m), asyncio.gather(*tasks_5m),
                asyncio.gather(*tasks_15m), asyncio.gather(*tasks_1h))
            market_data = {p: c for p, c in results_1m if c}
            market_data_5m = {p: c for p, c in results_5m if c}
            market_data_15m = {p: c for p, c in results_15m if c}
            market_data_1h = {p: c for p, c in results_1h if c}
            active_trades = await fetch_positions(exchange)

            # BUG-010 FIX: Sync _pending_entries with actual positions
            current_position_pairs = {_meta_key(t.pair) for t in active_trades}
            stale = _pending_entries - current_position_pairs
            for p in stale: _pending_entries.discard(p)
            for p in current_position_pairs: _pending_entries.add(p)

            try:
                positions = await exchange.fetch_positions()
                await ensure_tp_sl_missing(exchange, positions)
                current_pairs = set()
                for p in positions:
                    amt = float(p.get('contracts') or p.get('contractSize') or 0)
                    if amt == 0: continue
                    pair = p.get('symbol') or p.get('info', {}).get('symbol') or p.get('id')
                    mk = _meta_key(pair)
                    current_pairs.add(mk)
                    strat_id, age_sec = position_age_seconds(pair, p)
                    max_age = MAX_AGE.get(strat_id) if strat_id else DEFAULT_MAX_AGE
                    if max_age and age_sec > max_age:
                        side_field = p.get('side') or p.get('info', {}).get('positionSide') or ''
                        side = 'LONG' if str(side_field).lower() == 'long' else 'SHORT'
                        await close_position(exchange, pair, side, amt)
                        position_meta.pop(mk, None); save_position_meta()
                post_close = STRAT_CONFIG.get("correlation", {}).get("post_close_cooldown_sec", 0)
                cooldown = STRAT_CONFIG.get("correlation", {}).get("cooldown_sec", 0)
                extra = max(0, post_close - cooldown)
                for pair in list(position_meta.keys()):
                    if pair not in current_pairs:
                        ts = time.time() + extra if extra else time.time()
                        set_pair_cooldown(pair, ts)
                        position_meta.pop(pair, None); save_position_meta()
                        _pending_entries.discard(pair)  # BUG-010 FIX
                        print(f"Post-close cooldown set for {pair} ({post_close}s)")
            except Exception as e: print(f"Age check error: {e}")
            try:
                balance = await exchange.fetch_balance()
                total_usdt = balance.get('USDT', {}).get('total', 0)
                free_usdt = balance.get('USDT', {}).get('free', 0)
            except Exception as e: print(f"Balance fetch error: {e}"); free_usdt = 100.0
            funding = {}
            try:
                for sym in PAIRS:
                    data = await exchange.fapiPublicGetPremiumIndex({"symbol": sym})
                    funding[sym] = float(data.get("lastFundingRate", 0.0))
            except Exception as e: print(f"Funding fetch error: {e}")
            try: news_bias = get_news_bias(PAIRS, pos_bias=0.02, neg_bias=0.04, half_life_hours=2.0)
            except Exception as e: print(f"News bias error: {e}"); news_bias = {}
            equity_for_drawdown = total_usdt or free_usdt
            bias_cache = {}
            if ENABLE_12H_BIAS:
                for pair, c1h in market_data_1h.items():
                    try: bias_cache[pair] = compute_12h_bias(c1h, pair)
                    except Exception as e: print(f"12H bias error {pair}: {e}")
            res = run_signal_scan(market_data, active_trades, equity_for_drawdown, funding, news_bias=news_bias, market_data_5m=market_data_5m, market_data_15m=market_data_15m)
            kept = []
            for sig in res.signals:
                tier = _detect_engine(sig.strategy_id)  # BUG-004 FIX
                bias = bias_cache.get(sig.pair)
                if ENABLE_12H_BIAS and bias and not should_allow_signal(bias, sig.side, tier):
                    sig._filtered = True; sig._filter_reason = f"Blocked by 12H bias ({bias.direction})"; res.filtered.append(sig)
                else: kept.append(sig)
            res.signals = kept

            # 4H scan every 5 minutes
            global _LAST_HTF_SCAN
            now = time.time(); htf_signals = []
            if now - _LAST_HTF_SCAN > 300:
                _LAST_HTF_SCAN = now
                try: positions = await exchange.fetch_positions()
                except Exception: positions = []
                active_pairs_4h = set(); active_4h = 0
                for p in positions:
                    amt = float(p.get('contracts') or p.get('contractSize') or 0)
                    if amt == 0: continue
                    pair = p.get('symbol') or p.get('info', {}).get('symbol') or p.get('id')
                    strat_id = _meta_get(pair).get('strategy_id')
                    if strat_id in STRATEGIES_4H: active_pairs_4h.add(pair); active_4h += 1

                # BUG-001/008 FIX: Use market_data_1h for 4H scan
                # Was market_data (1m candles) — 1500/240 = only 6 four-hour candles
                # Now 500 1h candles / 4 = 125 four-hour candles (satisfies MIN_BARS 25-80)
                for pair, candles_1h in market_data_1h.items():
                    bias = bias_cache.get(pair) if ENABLE_12H_BIAS else None
                    results = scan_4h_strategies(
                        candles_1m=candles_1h,  # BUG-001 FIX: 1h candles (auto-detected by _aggregate_4h)
                        pair=pair, bias=bias,
                        funding_rate=funding.get(pair, 0.0),
                        active_4h_count=active_4h,
                        max_4h_trades=STRAT_CONFIG.get("max_4h_trades", 2),
                        active_pairs_4h=active_pairs_4h,
                        confidence_threshold=CONFIG_4H.get("confidence_threshold", 0.60),
                    )
                    if results:
                        best_id, best_sig = max(results, key=lambda x: x[1].confidence)
                        candles_1m_pair = market_data.get(pair)
                        price = candles_1m_pair[-1].close if candles_1m_pair else candles_1h[-1].close
                        tp_price = price * (1 + best_sig.tp_percent) if best_sig.side == "LONG" else price * (1 - best_sig.tp_percent)
                        sl_price = price * (1 - best_sig.sl_percent) if best_sig.side == "LONG" else price * (1 + best_sig.sl_percent)
                        base_size = max(STRAT_CONFIG["min_trade_size"], min(equity_for_drawdown * STRAT_CONFIG["risk_per_trade"], equity_for_drawdown / STRAT_CONFIG["max_concurrent_trades"]))
                        trade_size = max(STRAT_CONFIG["min_trade_size"], base_size)
                        econ = calculate_trade_economics(price, tp_price, sl_price, best_sig.side, trade_size, min(best_sig.leverage, CONFIG_4H.get("max_leverage", best_sig.leverage)))
                        if not econ.is_profitable: continue
                        strat_cat = "trend_4h" if best_id in STRAT_CONFIG.get("strategy_categories", {}).get("trend_4h", []) else "reversion_4h" if best_id in STRAT_CONFIG.get("strategy_categories", {}).get("reversion_4h", []) else "structural_4h"
                        htf_signals.append((best_id, best_sig, pair, strat_cat, price, tp_price, sl_price, trade_size, econ))
            for best_id, best_sig, pair, strat_cat, price, tp_price, sl_price, trade_size, econ in htf_signals:
                res.signals.append(TradeSignal(pair=pair, strategy_id=best_id, strategy_name=best_id, strategy_category=strat_cat, side=best_sig.side, confidence=best_sig.confidence, entry_price=price, tp_price=tp_price, sl_price=sl_price, leverage=min(best_sig.leverage, CONFIG_4H.get("max_leverage", best_sig.leverage)), trade_size=trade_size, reason=best_sig.reason, economics=econ))
            combined = res.signals
            by_pair = {}
            for s in combined:
                if s.pair not in by_pair or s.confidence > by_pair[s.pair].confidence: by_pair[s.pair] = s
            combined = list(by_pair.values())
            combined.sort(key=lambda s: s.confidence, reverse=True)
            after_cooldown = apply_cooldown_tiered(combined)
            after_flood = correlation_filter.apply_same_side_flood_filter(after_cooldown)
            after_category = correlation_filter.apply_category_filter(after_flood, active_trades)
            slots_available = STRAT_CONFIG["max_concurrent_trades"] - len(active_trades)
            res.signals = after_category[:max(0, slots_available)]
            # record all filtered (cooldown/flood/category)
            res.filtered.extend([s for s in combined if s._filtered])
            # log high-confidence drops from post-filter gates
            for s in combined:
                try:
                    if s._filtered and float(getattr(s,'confidence',0) or 0) >= 0.66:
                        log_high_conf_drop(s, s._filter_reason or "post_filter_drop")
                except Exception:
                    pass
            # log high-confidence drops due to slot limits
            if slots_available < len(after_category):
                for s in after_category[slots_available:]:
                    try:
                        if float(getattr(s,'confidence',0) or 0) >= 0.66:
                            log_high_conf_drop(s, "slot_limit")
                    except Exception:
                        pass
            if res.drawdown and res.drawdown.paused:
                print(res.drawdown.message or "Drawdown breaker active; skipping")
            else:
                global _LAST_SWAP_TS
                if res.signals:
                    slots_available = STRAT_CONFIG["max_concurrent_trades"] - len(active_trades)
                    if slots_available <= 0 and time.time() - _LAST_SWAP_TS > 3600:
                        best = max(res.signals, key=lambda s: s.confidence)
                        try:
                            positions = await exchange.fetch_positions(); candidates = []
                            for p in positions:
                                amt = float(p.get('contracts') or p.get('contractSize') or 0)
                                if amt == 0: continue
                                pair = p.get('symbol') or p.get('info', {}).get('symbol') or p.get('id')
                                side_field = p.get('side') or p.get('info', {}).get('positionSide') or ''
                                side = 'LONG' if str(side_field).lower() == 'long' else 'SHORT'
                                pnl = float(p.get('unrealizedPnl') or 0)
                                meta = _meta_get(pair); conf = float(meta.get('confidence') or 0)
                                if pnl <= 0: candidates.append((conf, pnl, pair, side, amt))
                            if candidates:
                                weakest = min(candidates, key=lambda x: x[0])
                                if best.confidence >= weakest[0] + 0.10:
                                    _, _, pair, side, amt = weakest
                                    await close_position(exchange, pair, side, amt)
                                    position_meta.pop(pair, None); _LAST_SWAP_TS = time.time()
                        except Exception as e: print(f"Swap check error: {e}")
                for sig in res.signals:
                    await ensure_leverage(exchange, sig.pair, sig.leverage)
                    await place_orders(exchange, sig)
            for fs in res.filtered:
                try:
                    log_filtered(fs, fs._filter_reason)
                    if float(getattr(fs, 'confidence', 0) or 0) >= 0.80:
                        log_high_conf_drop(fs, fs._filter_reason)
                except Exception:
                    pass
            d = res.diagnostics
            try:
                from pathlib import Path; import csv
                fpath = Path('/opt/multi-strat-engine/reports/cycle_diag_log.csv')
                exists = fpath.exists()
                with fpath.open('a', newline='') as f:
                    w = csv.writer(f)
                    if not exists: w.writerow(["ts","raw","after_cooldown","after_flood","after_category","final"])
                    w.writerow([int(time.time()), d.raw_count, d.after_cooldown, d.after_flood, d.after_category, d.final])
            except Exception: pass
            print(f"Cycle diag: Raw {d.raw_count} -> Cooldown {d.after_cooldown} -> Flood {d.after_flood} -> Cat {d.after_category} -> Final {d.final}")
            elapsed = time.time() - start
            await asyncio.sleep(max(5, LOOP_SEC - elapsed))
    finally:
        try: await exchange.close()
        except Exception: pass

if __name__ == "__main__":
    asyncio.run(loop())
