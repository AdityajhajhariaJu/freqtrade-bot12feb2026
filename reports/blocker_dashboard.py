#!/opt/multi-strat-engine/.venv/bin/python
import asyncio
import csv
import json
import time
from pathlib import Path

import ccxt.async_support as ccxt
import sys
sys.path.insert(0, '/opt/multi-strat-engine')
import strategies
from news_bias import get_news_bias

strategies.load_2h_strategies()
cfg = strategies.CONFIG
cfg2h = strategies.CONFIG_2H
PAIRS = cfg['pairs']

async def main():
    ex = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'future'}})
    reasons = {}
    passed = []
    try:
        C = strategies.Candle
        md1, md5, md15 = {}, {}, {}
        for p in PAIRS:
            md1[p] = [C(open=x[1], high=x[2], low=x[3], close=x[4], volume=x[5], timestamp=x[0]) for x in await ex.fetch_ohlcv(p, '1m', limit=500)]
            md5[p] = [C(open=x[1], high=x[2], low=x[3], close=x[4], volume=x[5], timestamp=x[0]) for x in await ex.fetch_ohlcv(p, '5m', limit=500)]
            md15[p] = [C(open=x[1], high=x[2], low=x[3], close=x[4], volume=x[5], timestamp=x[0]) for x in await ex.fetch_ohlcv(p, '15m', limit=500)]

        funding = {}
        for p in PAIRS:
            try:
                d = await ex.fapiPublicGetPremiumIndex({'symbol': p})
                funding[p] = float(d.get('lastFundingRate', 0.0))
            except Exception:
                funding[p] = 0.0

        try:
            news = get_news_bias(PAIRS, pos_bias=0.02, neg_bias=0.04, half_life_hours=2.0)
        except Exception:
            news = {}

        def bump(k):
            reasons[k] = reasons.get(k, 0) + 1

        for pair in PAIRS:
            candles = md1[pair]
            closes = [c.close for c in candles]
            price = candles[-1].close
            ap = strategies.get_adaptive_params(pair, candles) if cfg.get('adaptive_params', {}).get('enabled', False) else None
            af = ap.get('ema_fast') if ap else cfg.get('trend_ema_fast', 50)
            aslow = ap.get('ema_slow') if ap else cfg.get('trend_ema_slow', 200)
            ema_fast = strategies.ema(closes, af) if af else None
            ema_slow = strategies.ema(closes, aslow) if aslow else None
            trend_strength = abs(ema_fast - ema_slow) / price if (ema_fast and ema_slow and price) else 0

            for s in strategies.ALL_STRATEGIES:
                is2h = strategies.is_2h_strategy(s.id)
                s_cfg = cfg2h if is2h else {}
                conf_th = s_cfg.get('confidence_threshold', cfg.get('confidence_threshold', 0.66))
                confirm = s_cfg.get('confirm_signal', cfg.get('confirm_signal', True))
                min_vol = s_cfg.get('min_volatility_pct', cfg.get('min_volatility_pct', 0.0))
                skip_trend = s_cfg.get('skip_trend_ema_filter', False)
                skip_regime = s_cfg.get('skip_regime_filter', False)
                max_funding_long = s_cfg.get('max_funding_long', cfg.get('max_funding_long', 0.0))
                max_funding_short = s_cfg.get('max_funding_short', cfg.get('max_funding_short', 0.0))

                use = md5[pair] if (s.id == 'bb_squeeze' or is2h) else md1[pair]
                p_used = use[-1].close

                if min_vol and len(use) >= 20:
                    hi = max(c.high for c in use[-20:])
                    lo = min(c.low for c in use[-20:])
                    vol_pct = (hi - lo) / p_used if p_used else 0
                    if vol_pct < min_vol:
                        bump('volatility_filter')
                        continue

                try:
                    r = s.evaluate(use, funding_rate=funding.get(pair, 0.0)) if getattr(s, 'needs_funding', False) else s.evaluate(use)
                except Exception:
                    bump('evaluate_exception')
                    continue
                if r is None:
                    bump('no_pattern_trigger')
                    continue

                pf = cfg.get('pair_filters', {})
                if pair in pf and s.id not in pf[pair]:
                    bump('pair_filter_block')
                    continue

                adj = max(0.0, min(1.0, r.confidence + news.get(pair, 0.0)))
                if adj < conf_th:
                    bump('confidence_below_threshold')
                    continue

                if confirm and not is2h:
                    prev = s.evaluate(use[:-1]) if len(use) > 51 else None
                    if (not prev) or prev.side != r.side or prev.confidence < conf_th:
                        bump('confirm_signal_failed')
                        continue

                if (not skip_regime) and cfg.get('regime_min_trend_pct', 0.0) and trend_strength < cfg.get('regime_min_trend_pct', 0.0):
                    if s.id in cfg.get('strategy_categories', {}).get('trend', []):
                        bump('regime_filter_block')
                        continue

                if s.id == 'bb_squeeze':
                    c15 = md15[pair]
                    if not c15 or len(c15) < 120:
                        bump('bb_15m_insufficient')
                        continue
                    cl15 = [c.close for c in c15]
                    ema50 = strategies.ema(cl15, 50)
                    ema100 = strategies.ema(cl15, 100)
                    if ema50 is None or ema100 is None:
                        bump('bb_15m_ema_missing')
                        continue
                    p15 = cl15[-1]
                    if r.side == 'LONG' and not (p15 > ema50 and ema50 > ema100):
                        bump('trend_alignment_block')
                        continue
                    if r.side == 'SHORT' and not (p15 < ema50 and ema50 < ema100):
                        bump('trend_alignment_block')
                        continue
                else:
                    if (not skip_trend) and ema_fast is not None and ema_slow is not None:
                        if r.side == 'LONG' and not (price > ema_fast and ema_fast > ema_slow):
                            bump('trend_alignment_block')
                            continue
                        if r.side == 'SHORT' and not (price < ema_fast and ema_fast < ema_slow):
                            bump('trend_alignment_block')
                            continue

                fr = funding.get(pair, 0.0)
                if r.side == 'LONG' and max_funding_long and fr > max_funding_long:
                    bump('funding_block')
                    continue
                if r.side == 'SHORT' and max_funding_short and fr < -max_funding_short:
                    bump('funding_block')
                    continue

                atr_val = strategies.atr(use, cfg.get('atr_period', 14))
                atr_pct = (atr_val / p_used) if p_used else 0
                base = max(cfg['min_trade_size'], min(1000 * cfg['risk_per_trade'], 1000 / cfg['max_concurrent_trades']))
                if cfg.get('vol_target_pct', 0.0) and atr_pct > 0:
                    scale = cfg['vol_target_pct'] / atr_pct
                    size = max(cfg['min_trade_size'], min(base * scale, base * 2))
                else:
                    size = base

                tp_mult = s_cfg.get('tp_multiplier', cfg.get('tp_multiplier', 1.0))
                min_tp = s_cfg.get('min_tp_percent', cfg.get('min_tp_percent', 0.0))
                min_sl = s_cfg.get('min_sl_percent', cfg.get('min_sl_percent', 0.0))
                tp_pct = max(r.tp_percent * tp_mult, min_tp)
                sl_pct = max(r.sl_percent, min_sl)
                tp = p_used * (1 + tp_pct) if r.side == 'LONG' else p_used * (1 - tp_pct)
                sl = p_used * (1 - sl_pct) if r.side == 'LONG' else p_used * (1 + sl_pct)
                econ = strategies.calculate_trade_economics(p_used, tp, sl, r.side, size, strategies.clamp_leverage(r.leverage, cfg))
                if not econ.is_profitable:
                    bump('economics_not_profitable')
                    continue

                min_rr = s_cfg.get('min_risk_reward', cfg.get('min_risk_reward', 1.3))
                if min_rr is not None and econ.risk_reward < min_rr:
                    bump('risk_reward_block')
                    continue

                passed.append((pair, s.id, round(adj, 4), r.side))

    finally:
        await ex.close()

    ts = int(time.time())
    out_dir = Path('/opt/multi-strat-engine/reports')
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / 'blocker_dashboard.csv'
    json_path = out_dir / 'blocker_latest.json'

    row = {
        'ts': ts,
        'pass_after_pre_filters': len(passed),
        'no_pattern_trigger': reasons.get('no_pattern_trigger', 0),
        'confidence_below_threshold': reasons.get('confidence_below_threshold', 0),
        'confirm_signal_failed': reasons.get('confirm_signal_failed', 0),
        'trend_alignment_block': reasons.get('trend_alignment_block', 0),
        'pair_filter_block': reasons.get('pair_filter_block', 0),
        'volatility_filter': reasons.get('volatility_filter', 0),
        'regime_filter_block': reasons.get('regime_filter_block', 0),
        'funding_block': reasons.get('funding_block', 0),
        'economics_not_profitable': reasons.get('economics_not_profitable', 0),
        'risk_reward_block': reasons.get('risk_reward_block', 0),
        'evaluate_exception': reasons.get('evaluate_exception', 0),
    }

    new_file = not csv_path.exists()
    with csv_path.open('a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            w.writeheader()
        w.writerow(row)

    json_path.write_text(json.dumps({'ts': ts, 'reasons': reasons, 'passed': passed[:20]}, indent=2))
    print('ok', row)

if __name__ == '__main__':
    asyncio.run(main())
