#!/usr/bin/env python3
"""
Live trade loop for the multi-strategy engine (Binance USDT-M futures).
- Uses freqtrade config keys.
- Pairs: ETHUSDT, BNBUSDT, SOLUSDT, LINKUSDT, LTCUSDT, DOGEUSDT
- 1m scans, market entries, TP/SL reduce-only.
- One-way mode assumed.
- Max position age per strategy applied (upper bounds), default 25m if unknown.
- On startup: cancels ALL open orders on the configured pairs (does NOT close positions).
- On startup: seeds position metadata from existing positions (updateTime) to enforce max-age across restarts.
- Persists strategy_id on entry for max-age tracking.
"""
import asyncio
import json
import time
import ccxt.async_support as ccxt
from strategies import Candle, ActiveTrade, TradeSignal, run_signal_scan, CONFIG as STRAT_CONFIG, CONFIG_2H, CONFIG_4H, set_pair_cooldown, load_2h_strategies, calculate_trade_economics, apply_cooldown_tiered, correlation_filter
from news_bias import get_news_bias
from new_strategies_4h import scan_4h_strategies, compute_12h_bias, should_allow_signal, STRATEGIES_4H

CONFIG_PATH = "/opt/multi-strat-engine/config.binance_futures_live.json"
load_2h_strategies()
PAIRS = STRAT_CONFIG["pairs"]
FETCH_LIMIT = 1500
TIMEFRAME = "1m"
TIMEFRAME_5M = "5m"
TIMEFRAME_15M = "15m"
TIMEFRAME_1H = "1h"
LOOP_SEC = 75
DEFAULT_MAX_AGE = 25 * 60  # 25 minutes
from new_strategies_2h import NEW_MAX_AGE
# Max ages (seconds) per strategy (upper bound minutes)
MAX_AGE = {
    # Trend/Breakout -> 40m
    "ema_scalp": 40 * 60,
    "bb_squeeze": 40 * 60,
    "macd_flip": 40 * 60,
    "atr_breakout": 40 * 60,
    "triple_ema": 40 * 60,
    # Reversion -> 15m
    "rsi_snap": 15 * 60,
    "vwap_bounce": 15 * 60,
    "stoch_cross": 15 * 60,
    "obv_divergence": 15 * 60,
    # Structural -> 35m
    "engulfing_sr": 35 * 60,
}
# extend with 2h max ages
MAX_AGE.update(NEW_MAX_AGE)
# 4H max ages
MAX_AGE.update({
    "weekly_vwap_trend_4h": 16 * 3600,
    "ichimoku_breakout_4h": 24 * 3600,
    "bb_rsi_reversion_4h": 12 * 3600,
    "structure_break_ob_4h": 20 * 3600,
})

# Default TP/SL % by strategy (fallback for repairs)
DEFAULT_TPSL = {
    "bb_squeeze": (0.012, 0.0065),
}

# In-memory tracker: pair -> {strategy_id, opened_at}
position_meta = {}
META_PATH = "/opt/multi-strat-engine/reports/position_meta.json"


def load_position_meta():
    import json
    from pathlib import Path
    p = Path(META_PATH)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def save_position_meta():
    import json
    from pathlib import Path
    try:
        Path(META_PATH).write_text(json.dumps(position_meta))
    except Exception:
        pass
_LAST_SWAP_TS = 0
_LAST_TPSL_FIX_TS = 0
_LAST_HTF_SCAN = 0




def log_filtered(sig, reason):
    from pathlib import Path
    import csv, time
    fpath = Path('/opt/multi-strat-engine/reports/filtered_signals.csv')
    exists = fpath.exists()
    with fpath.open('a', newline='') as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["ts","pair","strategy","side","price","reason","outcome","pnl_pct","exit_price","horizon_min"])
        w.writerow([time.time(), sig.pair, sig.strategy_name, sig.side, sig.entry_price, reason, "", "", "", ""])


def normalize_trade_events_file():
    from pathlib import Path
    import csv
    fpath = Path('/opt/multi-strat-engine/trade_events.csv')
    if not fpath.exists():
        return
    try:
        with fpath.open() as f:
            first = f.readline().strip()
        if first.startswith('ts,event,'):
            return
        # rewrite with single header at top, remove embedded headers
        rows = []
        with fpath.open() as f:
            r = csv.reader(f)
            for row in r:
                if not row:
                    continue
                if len(row) >= 2 and row[0] == 'ts' and row[1] == 'event':
                    continue
                rows.append(row)
        header = ["ts","event","pair","side","qty","price","strategy_id","strategy_name","pnl","note","engine"]
        with fpath.open('w', newline='') as f:
            w = csv.writer(f)
            w.writerow(header)
            for row in rows:
                if len(row) < len(header):
                    row = row + ['']*(len(header)-len(row))
                w.writerow(row[:len(header)])
    except Exception:
        return


def normalize_pair(p):
    if not p:
        return ''
    return (p.replace('/USDT:USDT','USDT')
             .replace('/USDT','USDT'))
def log_event(event, pair, side, qty=None, price=None, strategy_id="", strategy_name="", pnl=None, note="", engine=""):
    from pathlib import Path
    import csv, time
    fpath = Path('/opt/multi-strat-engine/trade_events.csv')
    normalize_trade_events_file()
    exists = fpath.exists()
    header = None
    if exists:
        try:
            with fpath.open() as f:
                header = f.readline().strip().split(',')
        except Exception:
            header = None
    with fpath.open('a', newline='') as f:
        w=csv.writer(f)
        if not exists or not header or header[:2] != ["ts","event"]:
            w.writerow(["ts","event","pair","side","qty","price","strategy_id","strategy_name","pnl","note","engine"])
            header = ["ts","event","pair","side","qty","price","strategy_id","strategy_name","pnl","note","engine"]
        # if old header (no engine), append engine into note
        if "engine" not in header:
            note = (note + f" | engine={engine}").strip()
            w.writerow([int(time.time()), event, normalize_pair(pair), side, qty or "", price or "", strategy_id, strategy_name, pnl or "", note])
        else:
            w.writerow([int(time.time()), event, normalize_pair(pair), side, qty or "", price or "", strategy_id, strategy_name, pnl or "", note, engine])

def load_keys():
    with open(CONFIG_PATH, "r") as f:
        cfg = json.load(f)
    api_key = cfg.get("exchange", {}).get("key")
    secret = cfg.get("exchange", {}).get("secret")
    if not api_key or not secret:
        raise RuntimeError("API keys not found in freqtrade config")
    return api_key, secret


async def fetch_candles(exchange, pair, timeframe=TIMEFRAME, limit=None):
    try:
        ohlcv = await exchange.fetch_ohlcv(pair, timeframe=timeframe, limit=limit or FETCH_LIMIT)
        candles = [Candle(open=o[1], high=o[2], low=o[3], close=o[4], volume=o[5], timestamp=o[0]) for o in ohlcv]
        return pair, candles
    except Exception as e:
        print(f"Fetch error {pair} {timeframe}: {e}")
        return pair, None


async def fetch_positions(exchange):
    try:
        positions = await exchange.fetch_positions()
    except Exception as e:
        print(f"Fetch positions error: {e}")
        return []
    active = []
    for p in positions:
        amt = float(p.get('contracts') or p.get('contractSize') or 0)
        if amt == 0:
            continue
        pair = p.get('symbol') or p.get('info', {}).get('symbol') or p.get('id')
        side_field = p.get('side') or p.get('info', {}).get('positionSide') or ''
        side = 'LONG' if str(side_field).lower() == 'long' else 'SHORT'
        strat_id = position_meta.get(pair, {}).get('strategy_id', '') if pair in position_meta else ''
        active.append(ActiveTrade(pair=pair, strategy_id=strat_id, side=side))
    return active


async def seed_position_meta(exchange):
    # load persisted meta first
    position_meta.update(load_position_meta() or {})
    try:
        positions = await exchange.fetch_positions()
    except Exception as e:
        print(f"Seed positions error: {e}")
        return
    now = time.time()


    def last_entry_strategy(pair):
        from pathlib import Path
        import csv
        fpath = Path('/opt/multi-strat-engine/trade_events.csv')
        if not fpath.exists():
            return '', ''
        last = ('','')
        try:
            with fpath.open() as f:
                r=csv.DictReader(f)
                for row in r:
                    if row.get('event') in ('ENTRY','ENTRY_RECOVER') and row.get('pair') == normalize_pair(pair):
                        last = (row.get('strategy_id',''), row.get('strategy_name',''))
        except Exception:
            return '', ''
        return last

    def has_entry(pair):
        from pathlib import Path
        import csv
        fpath = Path('/opt/multi-strat-engine/trade_events.csv')
        if not fpath.exists():
            return False
        try:
            with fpath.open() as f:
                r=csv.DictReader(f)
                for row in r:
                    if row.get('event') in ('ENTRY','ENTRY_RECOVER') and row.get('pair') == normalize_pair(pair):
                        return True
        except Exception:
            return False
        return False

    for p in positions:
        amt = float(p.get('contracts') or p.get('contractSize') or 0)
        if amt == 0:
            continue
        pair = p.get('symbol') or p.get('info', {}).get('symbol') or p.get('id')
        upd_ms = float(p.get('info', {}).get('updateTime') or p.get('timestamp') or now * 1000)
        opened_at = upd_ms / 1000.0
        if pair not in position_meta:
            position_meta[pair] = {"strategy_id": "", "opened_at": opened_at}
            print(f"Seeded meta for {pair} opened_at={opened_at}")
        # backfill strategy_id/name from last entry if missing
        if not position_meta.get(pair, {}).get('strategy_id'):
            sid, sname = last_entry_strategy(pair)
            if sid:
                position_meta[pair]["strategy_id"] = sid
        # ensure entry is mapped even after restart
        if not has_entry(pair):
            sid = position_meta.get(pair, {}).get('strategy_id','')
            sname = last_entry_strategy(pair)[1] if sid else ''
            eng = "2h" if sid and (sid in STRAT_CONFIG.get('strategy_categories', {}).get('trend_2h', []) or sid in STRAT_CONFIG.get('strategy_categories', {}).get('reversion_2h', []) or sid in STRAT_CONFIG.get('strategy_categories', {}).get('structural_2h', [])) else "1m"
            log_event("ENTRY_RECOVER", pair, "", qty=amt, price=None, strategy_id=sid, strategy_name=sname, note="Recovered open position", engine=eng)
    save_position_meta()


async def set_oneway(exchange):
    try:
        await exchange.set_position_mode(False)  # one-way
    except Exception as e:
        print(f"set_position_mode warning: {e}")


async def ensure_leverage(exchange, pair, lev):
    try:
        await exchange.set_leverage(lev, pair)
    except Exception as e:
        print(f"set_leverage warning {pair} {lev}x: {e}")


async def cancel_all_open_orders(exchange):
    try:
        exchange.options['warnOnFetchOpenOrdersWithoutSymbol'] = False
    except Exception:
        pass
    for sym in PAIRS:
        try:
            await exchange.cancel_all_orders(sym)
            print(f"cancel_all_orders called for {sym}")
        except Exception as e:
            print(f"cancel_all_orders error {sym}: {e}")


async def ensure_tp_sl_missing(exchange, positions):
    global _LAST_TPSL_FIX_TS
    now = time.time()
    if now - _LAST_TPSL_FIX_TS < 300:
        return
    _LAST_TPSL_FIX_TS = now
    for p in positions:
        amt = float(p.get('contracts') or p.get('contractSize') or 0)
        if amt == 0:
            continue
        pair = p.get('symbol') or p.get('info', {}).get('symbol') or p.get('id')
        side_field = p.get('side') or p.get('info', {}).get('positionSide') or ''
        side = 'LONG' if str(side_field).lower() == 'long' else 'SHORT'
        try:
            orders = await exchange.fetch_open_orders(pair)
        except Exception:
            orders = []
        has_tp = False
        has_sl = False
        for o in orders:
            otype = (o.get('type') or '').lower()
            info_type = (o.get('info', {}).get('type') or '').lower()
            reduce_only = o.get('reduceOnly') or o.get('info', {}).get('reduceOnly')
            if not reduce_only:
                continue
            if 'take_profit' in otype or 'take_profit' in info_type:
                has_tp = True
            if 'stop' in otype or 'stop' in info_type:
                has_sl = True
        if has_tp and has_sl:
            continue
        strat_id = position_meta.get(pair, {}).get('strategy_id')
        tp_sl = DEFAULT_TPSL.get(strat_id)
        if not tp_sl:
            continue
        entry = float(p.get('entryPrice') or p.get('info', {}).get('entryPrice') or 0)
        if not entry:
            continue
        tp_pct, sl_pct = tp_sl
        if side == 'LONG':
            tp = entry * (1 + tp_pct)
            sl = entry * (1 - sl_pct)
            close_side = 'sell'
        else:
            tp = entry * (1 - tp_pct)
            sl = entry * (1 + sl_pct)
            close_side = 'buy'
        amount = abs(amt)
        reduce_params = {"reduceOnly": True}
        try:
            if not has_tp:
                await exchange.create_order(pair, 'take_profit_market', close_side, amount, None, {**reduce_params, "stopPrice": tp})
                print(f"TP fixed {pair} {side} @ {tp:.6f}")
            if not has_sl:
                await exchange.create_order(pair, 'stop_market', close_side, amount, None, {**reduce_params, "stopPrice": sl})
                print(f"SL fixed {pair} {side} @ {sl:.6f}")
        except Exception as e:
            print(f"TP/SL fix error {pair}: {e}")


async def place_orders(exchange, signal):
    side = 'buy' if signal.side == 'LONG' else 'sell'
    notional = signal.trade_size * signal.leverage
    amount = notional / signal.entry_price
    params = {"type": "MARKET"}
    try:
        await exchange.create_order(signal.pair, 'market', side, amount, None, params)
    except Exception as e:
        print(f"Entry error {signal.pair} {side}: {e}")
        return
    reduce_params = {"reduceOnly": True}
    try:
        await exchange.create_order(signal.pair, 'take_profit_market', 'sell' if signal.side == 'LONG' else 'buy', amount, None, {**reduce_params, "stopPrice": signal.tp_price})
    except Exception as e:
        print(f"TP error {signal.pair}: {e}")
    try:
        await exchange.create_order(signal.pair, 'stop_market', 'sell' if signal.side == 'LONG' else 'buy', amount, None, {**reduce_params, "stopPrice": signal.sl_price})
    except Exception as e:
        print(f"SL error {signal.pair}: {e}")
    position_meta[signal.pair] = {"strategy_id": signal.strategy_id, "opened_at": time.time(), "confidence": signal.confidence}
    save_position_meta()
    engine = "2h" if signal.strategy_id in STRAT_CONFIG.get("strategy_categories", {}).get("trend_2h", []) or signal.strategy_id in STRAT_CONFIG.get("strategy_categories", {}).get("reversion_2h", []) or signal.strategy_id in STRAT_CONFIG.get("strategy_categories", {}).get("structural_2h", []) else "1m"
    log_event("ENTRY", signal.pair, signal.side, qty=amount, price=signal.entry_price, strategy_id=signal.strategy_id, strategy_name=signal.strategy_name, note=signal.reason, engine=engine)
    print(f"EXECUTED {signal.pair} {signal.side} size=${signal.trade_size:.2f} lev={signal.leverage} entry~{signal.entry_price} tp={signal.tp_price} sl={signal.sl_price} strat={signal.strategy_name} reason={signal.reason}")


async def close_position(exchange, pair, side, amount):
    close_side = 'sell' if side == 'LONG' else 'buy'
    entry_price = None
    try:
        positions = await exchange.fetch_positions()
        for p in positions:
            sym = p.get('symbol') or p.get('info', {}).get('symbol') or p.get('id')
            if sym == pair:
                entry_price = float(p.get('entryPrice') or p.get('info', {}).get('entryPrice') or 0) or None
                break
    except Exception:
        entry_price = None
    try:
        await exchange.create_order(pair, 'market', close_side, amount, None, {"reduceOnly": True})
        close_price = None
        try:
            t = await exchange.fetch_ticker(pair)
            close_price = float(t.get('last') or t.get('close') or 0) or None
        except Exception:
            close_price = None
        pnl = None
        if entry_price and close_price:
            pnl = (close_price - entry_price) * amount if side == 'LONG' else (entry_price - close_price) * amount
        # exact realized pnl from income history (best-effort)
        try:
            opened_at = position_meta.get(pair, {}).get('opened_at')
            if opened_at:
                data = await exchange.fapiPrivateGetIncome({
                    "incomeType": "REALIZED_PNL",
                    "startTime": int(opened_at * 1000),
                    "endTime": int(time.time() * 1000),
                    "limit": 1000,
                })
                realized = sum(float(r.get('income', 0) or 0) for r in data if r.get('symbol') in (pair, pair.replace('/USDT:USDT','').replace('/USDT','')+'USDT'))
                if realized != 0:
                    pnl = realized
        except Exception:
            pass
        log_event("CLOSE_MAX_AGE", pair, side, qty=amount, price=close_price, pnl=pnl, note="max_age")
        print(f"Closed {pair} {side} due to max age")
    except Exception as e:
        print(f"Close error {pair}: {e}")




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
    meta = position_meta.get(pair)
    strat_id = meta.get('strategy_id') if meta else None
    opened_at = meta.get('opened_at') if meta else None
    if not opened_at:
        upd_ms = float(p_obj.get('info', {}).get('updateTime') or p_obj.get('timestamp') or time.time()*1000)
        opened_at = upd_ms / 1000.0
        position_meta[pair] = {"strategy_id": strat_id or "", "opened_at": opened_at}
    return strat_id, time.time() - opened_at

async def loop():
    normalize_trade_events_file()
    api_key, secret = load_keys()
    exchange = ccxt.binance({
        "apiKey": api_key,
        "secret": secret,
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    })
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
                asyncio.gather(*tasks_1m),
                asyncio.gather(*tasks_5m),
                asyncio.gather(*tasks_15m),
                asyncio.gather(*tasks_1h)
            )
            market_data = {p: c for p, c in results_1m if c}
            market_data_5m = {p: c for p, c in results_5m if c}
            market_data_15m = {p: c for p, c in results_15m if c}
            market_data_1h = {p: c for p, c in results_1h if c}
            active_trades = await fetch_positions(exchange)
            try:
                positions = await exchange.fetch_positions()
                await ensure_tp_sl_missing(exchange, positions)
                current_pairs = set()
                for p in positions:
                    amt = float(p.get('contracts') or p.get('contractSize') or 0)
                    if amt == 0:
                        continue
                    pair = p.get('symbol') or p.get('info', {}).get('symbol') or p.get('id')
                    current_pairs.add(pair)
                    strat_id, age_sec = position_age_seconds(pair, p)
                    max_age = MAX_AGE.get(strat_id) if strat_id else DEFAULT_MAX_AGE
                    if max_age and age_sec > max_age:
                        side_field = p.get('side') or p.get('info', {}).get('positionSide') or ''
                        side = 'LONG' if str(side_field).lower() == 'long' else 'SHORT'
                        await close_position(exchange, pair, side, amt)
                        position_meta.pop(pair, None)
                        save_position_meta()
                # apply post-close cooldowns for pairs that closed via TP/SL
                post_close = STRAT_CONFIG.get("correlation", {}).get("post_close_cooldown_sec", 0)
                cooldown = STRAT_CONFIG.get("correlation", {}).get("cooldown_sec", 0)
                extra = max(0, post_close - cooldown)
                for pair in list(position_meta.keys()):
                    if pair not in current_pairs:
                        ts = time.time() + extra if extra else time.time()
                        set_pair_cooldown(pair, ts)
                        position_meta.pop(pair, None)
                        save_position_meta()
                        print(f"Post-close cooldown set for {pair} ({post_close}s)")
            except Exception as e:
                print(f"Age check error: {e}")
            try:
                balance = await exchange.fetch_balance()
                # use total equity for drawdown checks
                total_usdt = balance.get('USDT', {}).get('total', 0)
                free_usdt = balance.get('USDT', {}).get('free', 0)
            except Exception as e:
                print(f"Balance fetch error: {e}")
                free_usdt = 100.0
            # fetch funding rates
            funding = {}
            try:
                for sym in PAIRS:
                    data = await exchange.fapiPublicGetPremiumIndex({"symbol": sym})
                    funding[sym] = float(data.get("lastFundingRate", 0.0))
            except Exception as e:
                print(f"Funding fetch error: {e}")
            try:
                news_bias = get_news_bias(PAIRS, pos_bias=0.02, neg_bias=0.04, half_life_hours=2.0)
            except Exception as e:
                print(f"News bias error: {e}")
                news_bias = {}
            equity_for_drawdown = total_usdt or free_usdt
            # compute 12H bias per pair (from 1h candles)
            bias_cache = {}
            for pair, c1h in market_data_1h.items():
                try:
                    bias_cache[pair] = compute_12h_bias(c1h, pair)
                except Exception as e:
                    print(f"12H bias error {pair}: {e}")
            res = run_signal_scan(market_data, active_trades, equity_for_drawdown, funding, news_bias=news_bias, market_data_5m=market_data_5m, market_data_15m=market_data_15m)
            # apply 12H bias filter to 1m/2h signals
            kept = []
            for sig in res.signals:
                tier = "2h" if sig.strategy_id in STRAT_CONFIG.get("strategy_categories", {}).get("trend_2h", []) or sig.strategy_id in STRAT_CONFIG.get("strategy_categories", {}).get("reversion_2h", []) or sig.strategy_id in STRAT_CONFIG.get("strategy_categories", {}).get("structural_2h", []) else "1m"
                bias = bias_cache.get(sig.pair)
                if bias and not should_allow_signal(bias, sig.side, tier):
                    sig._filtered = True
                    sig._filter_reason = f"Blocked by 12H bias ({bias.direction})"
                    res.filtered.append(sig)
                else:
                    kept.append(sig)
            res.signals = kept

            # 4H scan every 5 minutes
            global _LAST_HTF_SCAN
            now = time.time()
            htf_signals = []
            if now - _LAST_HTF_SCAN > 300:
                _LAST_HTF_SCAN = now
                try:
                    positions = await exchange.fetch_positions()
                except Exception:
                    positions = []
                active_pairs_4h = set()
                active_4h = 0
                for p in positions:
                    amt = float(p.get('contracts') or p.get('contractSize') or 0)
                    if amt == 0:
                        continue
                    pair = p.get('symbol') or p.get('info', {}).get('symbol') or p.get('id')
                    strat_id = position_meta.get(pair, {}).get('strategy_id')
                    if strat_id in STRATEGIES_4H:
                        active_pairs_4h.add(pair)
                        active_4h += 1
                for pair, candles_1m in market_data.items():
                    bias = bias_cache.get(pair)
                    results = scan_4h_strategies(
                        candles_1m=candles_1m,
                        pair=pair,
                        bias=bias,
                        funding_rate=funding.get(pair, 0.0),
                        active_4h_count=active_4h,
                        max_4h_trades=STRAT_CONFIG.get("max_4h_trades", 2),
                        active_pairs_4h=active_pairs_4h,
                        confidence_threshold=CONFIG_4H.get("confidence_threshold", 0.60),
                    )
                    if results:
                        best_id, best_sig = max(results, key=lambda x: x[1].confidence)
                        price = candles_1m[-1].close
                        tp_price = price * (1 + best_sig.tp_percent) if best_sig.side == "LONG" else price * (1 - best_sig.tp_percent)
                        sl_price = price * (1 - best_sig.sl_percent) if best_sig.side == "LONG" else price * (1 + best_sig.sl_percent)
                        base_size = max(STRAT_CONFIG["min_trade_size"], min(equity_for_drawdown * STRAT_CONFIG["risk_per_trade"], equity_for_drawdown / STRAT_CONFIG["max_concurrent_trades"]))
                        trade_size = max(STRAT_CONFIG["min_trade_size"], base_size)
                        econ = calculate_trade_economics(price, tp_price, sl_price, best_sig.side, trade_size, min(best_sig.leverage, CONFIG_4H.get("max_leverage", best_sig.leverage)))
                        if not econ.is_profitable:
                            continue
                        strat_cat = "trend_4h" if best_id in STRAT_CONFIG.get("strategy_categories", {}).get("trend_4h", []) else "reversion_4h" if best_id in STRAT_CONFIG.get("strategy_categories", {}).get("reversion_4h", []) else "structural_4h"
                        htf_signals.append((best_id, best_sig, pair, strat_cat, price, tp_price, sl_price, trade_size, econ))
            # merge 4h signals
            for best_id, best_sig, pair, strat_cat, price, tp_price, sl_price, trade_size, econ in htf_signals:
                res.signals.append(TradeSignal(
                    pair=pair,
                    strategy_id=best_id,
                    strategy_name=best_id,
                    strategy_category=strat_cat,
                    side=best_sig.side,
                    confidence=best_sig.confidence,
                    entry_price=price,
                    tp_price=tp_price,
                    sl_price=sl_price,
                    leverage=min(best_sig.leverage, CONFIG_4H.get("max_leverage", best_sig.leverage)),
                    trade_size=trade_size,
                    reason=best_sig.reason,
                    economics=econ,
                ))

            # re-apply filters on combined signals (1m/2h + 4h)
            combined = res.signals
            # de-dup by pair (keep highest confidence)
            by_pair = {}
            for s in combined:
                if s.pair not in by_pair or s.confidence > by_pair[s.pair].confidence:
                    by_pair[s.pair] = s
            combined = list(by_pair.values())
            combined.sort(key=lambda s: s.confidence, reverse=True)
            after_cooldown = apply_cooldown_tiered(combined)
            after_flood = correlation_filter.apply_same_side_flood_filter(after_cooldown)
            after_category = correlation_filter.apply_category_filter(after_flood, active_trades)
            slots_available = STRAT_CONFIG["max_concurrent_trades"] - len(active_trades)
            res.signals = after_category[:max(0, slots_available)]
            res.filtered.extend([s for s in combined if s._filtered])

            if res.drawdown and res.drawdown.paused:
                print(res.drawdown.message or "Drawdown breaker active; skipping")
            else:
                # opportunity cost swap (fee-aware)
                global _LAST_SWAP_TS
                if res.signals:
                    slots_available = STRAT_CONFIG["max_concurrent_trades"] - len(active_trades)
                    if slots_available <= 0 and time.time() - _LAST_SWAP_TS > 3600:
                        best = max(res.signals, key=lambda s: s.confidence)
                        try:
                            positions = await exchange.fetch_positions()
                            candidates = []
                            for p in positions:
                                amt = float(p.get('contracts') or p.get('contractSize') or 0)
                                if amt == 0: continue
                                pair = p.get('symbol') or p.get('info', {}).get('symbol') or p.get('id')
                                side_field = p.get('side') or p.get('info', {}).get('positionSide') or ''
                                side = 'LONG' if str(side_field).lower() == 'long' else 'SHORT'
                                pnl = float(p.get('unrealizedPnl') or 0)
                                meta = position_meta.get(pair, {})
                                conf = float(meta.get('confidence') or 0)
                                if pnl <= 0:
                                    candidates.append((conf, pnl, pair, side, amt))
                            if candidates:
                                weakest = min(candidates, key=lambda x: x[0])
                                if best.confidence >= weakest[0] + 0.10:
                                    _, _, pair, side, amt = weakest
                                    await close_position(exchange, pair, side, amt)
                                    position_meta.pop(pair, None)
                                    _LAST_SWAP_TS = time.time()
                        except Exception as e:
                            print(f"Swap check error: {e}")
                for sig in res.signals:
                    await ensure_leverage(exchange, sig.pair, sig.leverage)
                    await place_orders(exchange, sig)
            for fs in res.filtered:
                try:
                    log_filtered(fs, fs._filter_reason)
                except Exception:
                    pass
            d = res.diagnostics
            # log cycle diagnostics with timestamp for charts
            try:
                from pathlib import Path
                import csv
                fpath = Path('/opt/multi-strat-engine/reports/cycle_diag_log.csv')
                exists = fpath.exists()
                with fpath.open('a', newline='') as f:
                    w = csv.writer(f)
                    if not exists:
                        w.writerow(["ts","raw","after_cooldown","after_flood","after_category","final"])
                    w.writerow([int(time.time()), d.raw_count, d.after_cooldown, d.after_flood, d.after_category, d.final])
            except Exception:
                pass
            print(f"Cycle diag: Raw {d.raw_count} -> Cooldown {d.after_cooldown} -> Flood {d.after_flood} -> Cat {d.after_category} -> Final {d.final}")
            elapsed = time.time() - start
            await asyncio.sleep(max(5, LOOP_SEC - elapsed))
    finally:
        try:
            await exchange.close()
        except Exception:
            pass

if __name__ == "__main__":
    asyncio.run(loop())
