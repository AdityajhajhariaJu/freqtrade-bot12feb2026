# Multi-Strategy Futures Engine (1m / 2h / 4h)

This repo is a full snapshot of the live multi-strategy engine from `/opt/multi-strat-engine`.

## What’s included
- **Core loop**: `trade_loop.py`
- **1m strategies**: `strategies.py`
- **2h strategies**: `new_strategies_2h.py`
- **4h strategies**: `new_strategies_4h.py`
- **News bias**: `news_bias.py`
- **Reports & tools**: `reports/` (health checks, post-trade pipeline, assistant tools, etc.)
- **Config**: `config.binance_futures_live.json` (contains live keys; handle securely)

## Runtime notes
- **Pairs**: `ETHUSDT, LTCUSDT, DOGEUSDT, AVAXUSDT, NEARUSDT, LINKUSDT, BNBUSDT`
- **Timeframes**:
  - 1m scalps
  - 2h swing (via aggregated candles)
  - 4h position tier (with 12h directional filter)
- **Max leverage**: 10 (global cap)
- **4h max trades**: 2
- **Confidence threshold**: 0.66 (global)
- **Drawdown breaker**: disabled (`max_drawdown_pause=1.0`)

## Strategy tiers (active)
**1m**: ema_scalp, rsi_snap, macd_flip, vwap_bounce, stoch_cross, atr_breakout, triple_ema, engulfing_sr, obv_divergence

**2h**: ichimoku_cloud, keltner_reversion, donchian_breakout, supertrend_flip, vp_poc_reversion, pivot_bounce, funding_fade, bb_kc_squeeze

**4h**: weekly_vwap_trend_4h, ichimoku_breakout_4h, bb_rsi_reversion_4h, structure_break_ob_4h

## Services
The bot runs via systemd user service:
- `~/.config/systemd/user/multistrat.service`

Logs:
- `/home/ubuntu/.openclaw/workspace/logs/multistrat.log`
- `/home/ubuntu/.openclaw/workspace/logs/multistrat.err`

## Mapping & Reports
- Trade log: `/opt/multi-strat-engine/trade_events.csv`
- Mapping state: `/opt/multi-strat-engine/reports/position_meta.json`
- Hourly backups: `/opt/multi-strat-engine/reports/backups/`

## Notable recent fixes
- 4h scan now uses **1m candles** (1h only for 12h bias) and passes full filters.
- Active trades now keep `strategy_id` to enforce caps.
- `clamp_leverage` bug fixed.
- 4h cooldown tier added.

## Security
This repo contains live config/keys. If sharing, **remove secrets** and rotate keys.
