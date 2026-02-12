"""
Strategy Activation Policy (Market-Regime Aware)

SECTION 1) Criteria for ACTIVATING strategies
- Volatility bands (ATR%):
  - Low-vol floor: atr_pct >= 0.0008 (0.08%)
  - Extreme-vol cap: atr_pct <= 0.0600 (6.00%)
- One-sided / trend market indicators:
  - one_sided=True when directional move is persistent over lookback
  - trend_strength from EMA(20)-EMA(50) gap on 15m
- Volume engagement:
  - volume_ratio = latest_vol / avg_vol_20
  - minimum engagement >= 0.35 (global)
- Strategy category activation:
  - trend: one_sided OR trend_strength >= 0.0025, volume_ratio >= 0.8
  - breakout: volatility_expanding=True AND volume_ratio >= 1.05
  - reversion: NOT one_sided, moderate volatility, volume_ratio >= 0.7
  - scalp: non-extreme volatility, volume_ratio >= 0.6

SECTION 2) Criteria for DEACTIVATING strategies
- Low market engagement: volume_ratio < 0.35
- Dead market / too flat: atr_pct < 0.0008
- Extreme chaotic market: atr_pct > 0.0600
- Category-specific deactivation:
  - trend disabled in flat/choppy regimes (weak trend)
  - reversion disabled in strong one-sided trend
  - breakout disabled when volatility contraction + weak volume

SECTION 3) Strategy list and intended conditions
- trend:
  - mtf_ema_ribbon, ema_cross_rsi, macd_hist_flip, adx_di_cross
  - best in one-sided / directional conditions
- breakout:
  - bb_squeeze, bb_kc_squeeze, atr_breakout, donchian_breakout
  - best in volatility expansion + strong participation
- reversion:
  - rsi_snapback, keltner_reversion, cmf_divergence, vwap_bounce
  - best in balanced/choppy markets with mean-reversion behavior
- event/momentum:
  - liquidation_cascade
  - best in impulse spikes with elevated activity

SECTION 4) Profit vs loss summary (expected impact)
- Expected profit impact:
  - Better alignment of strategy to market regime can reduce low-quality entries.
  - Fewer false breakouts in dead markets and fewer mean-reversion fades in one-way trends.
- Expected loss impact:
  - Reduced overtrading in poor conditions lowers repeated small losses/fees.
- Risk note:
  - Thresholds are intentionally moderate (not too strict) to avoid killing signal flow.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal


Category = Literal["trend", "breakout", "reversion", "scalp", "event", "unknown"]


@dataclass
class MarketRegime:
    atr_pct: float
    atr_pct_15m: float
    volume_ratio: float
    trend_strength: float
    one_sided: bool
    volatility_expanding: bool


def _ema(values: list[float], period: int):
    if len(values) < period:
        return None
    alpha = 2 / (period + 1)
    e = values[0]
    for v in values[1:]:
        e = alpha * v + (1 - alpha) * e
    return e


def _atr_pct(candles, period=14):
    if not candles or len(candles) < period + 1:
        return 0.0
    trs = []
    for i in range(1, len(candles)):
        h = candles[i].high
        l = candles[i].low
        pc = candles[i - 1].close
        tr = max(h - l, abs(h - pc), abs(l - pc))
        trs.append(tr)
    atr = sum(trs[-period:]) / period if len(trs) >= period else 0.0
    px = candles[-1].close or 1.0
    return max(0.0, atr / px)


def _volume_ratio(candles, lookback=20):
    if not candles:
        return 0.0
    if len(candles) < lookback + 1:
        return 1.0
    latest = candles[-1].volume
    avg = sum(c.volume for c in candles[-lookback - 1:-1]) / lookback
    if avg <= 0:
        return 1.0
    return latest / avg


def _one_sided(candles_15m, lookback=12, min_dir_ratio=0.67):
    if not candles_15m or len(candles_15m) < lookback + 1:
        return False
    closes = [c.close for c in candles_15m[-(lookback + 1):]]
    ups = 0
    dns = 0
    for i in range(1, len(closes)):
        if closes[i] > closes[i - 1]:
            ups += 1
        elif closes[i] < closes[i - 1]:
            dns += 1
    total = max(1, ups + dns)
    return (ups / total) >= min_dir_ratio or (dns / total) >= min_dir_ratio


def _trend_strength(candles_15m):
    if not candles_15m or len(candles_15m) < 60:
        return 0.0
    closes = [c.close for c in candles_15m]
    e20 = _ema(closes[-80:], 20)
    e50 = _ema(closes[-120:], 50)
    px = closes[-1] or 1.0
    if e20 is None or e50 is None:
        return 0.0
    return abs(e20 - e50) / px


def detect_market_regime(candles_1m, candles_5m=None, candles_15m=None) -> MarketRegime:
    candles_5m = candles_5m or candles_1m
    candles_15m = candles_15m or candles_5m

    atr1 = _atr_pct(candles_1m, 14)
    atr15 = _atr_pct(candles_15m, 14)
    vol = _volume_ratio(candles_1m, 20)
    tr = _trend_strength(candles_15m)
    one_way = _one_sided(candles_15m, lookback=12, min_dir_ratio=0.67)

    # Expansion when short-term vol exceeds medium-term baseline meaningfully
    vol_exp = atr1 > (atr15 * 1.10) if atr15 > 0 else atr1 > 0.002

    return MarketRegime(
        atr_pct=atr1,
        atr_pct_15m=atr15,
        volume_ratio=vol,
        trend_strength=tr,
        one_sided=one_way,
        volatility_expanding=vol_exp,
    )


STRATEGY_CATEGORY_MAP: dict[str, Category] = {
    # trend
    "mtf_ema_ribbon": "trend",
    "ema_cross_rsi": "trend",
    "macd_hist_flip": "trend",
    "adx_di_cross": "trend",

    # breakout
    "bb_squeeze": "breakout",
    "bb_kc_squeeze": "breakout",
    "atr_breakout": "breakout",
    "donchian_breakout": "breakout",

    # reversion
    "rsi_snapback": "reversion",
    "keltner_reversion": "reversion",
    "cmf_divergence": "reversion",
    "vwap_bounce": "reversion",

    # event / momentum
    "liquidation_cascade": "event",
}


def infer_category(strategy_id: str, strategy_category: str = "") -> Category:
    sid = (strategy_id or "").strip().lower()
    if sid in STRATEGY_CATEGORY_MAP:
        return STRATEGY_CATEGORY_MAP[sid]

    sc = (strategy_category or "").strip().lower()
    if sc in ("trend", "reversion", "breakout"):
        return sc  # type: ignore[return-value]

    # Heuristic fallback (safe, moderate)
    if "ema" in sid or "adx" in sid or "trend" in sid:
        return "trend"
    if "squeeze" in sid or "breakout" in sid:
        return "breakout"
    if "rsi" in sid or "reversion" in sid or "vwap" in sid or "cmf" in sid:
        return "reversion"
    if "liq" in sid or "cascade" in sid:
        return "event"
    return "unknown"


def should_activate_strategy(strategy_id: str, strategy_category: str, regime: MarketRegime):
    cat = infer_category(strategy_id, strategy_category)

    # Global deactivation guards (kept moderate)
    if regime.volume_ratio < 0.35:
        return False, f"low_engagement(vol_ratio={regime.volume_ratio:.2f})"
    if regime.atr_pct < 0.0008:
        return False, f"dead_market(atr={regime.atr_pct:.4f})"
    if regime.atr_pct > 0.0600:
        return False, f"extreme_volatility(atr={regime.atr_pct:.4f})"

    if cat == "trend":
        if regime.one_sided or regime.trend_strength >= 0.0025:
            if regime.volume_ratio >= 0.8:
                return True, "trend_ok"
            return False, f"trend_low_volume(vol_ratio={regime.volume_ratio:.2f})"
        return False, f"trend_not_directional(trend={regime.trend_strength:.4f})"

    if cat == "breakout":
        if regime.volatility_expanding and regime.volume_ratio >= 1.05:
            return True, "breakout_ok"
        return False, f"breakout_no_expansion(expand={regime.volatility_expanding},vol={regime.volume_ratio:.2f})"

    if cat == "reversion":
        if regime.one_sided and regime.trend_strength >= 0.0035:
            return False, "reversion_blocked_one_sided"
        if regime.volume_ratio < 0.7:
            return False, f"reversion_low_volume(vol_ratio={regime.volume_ratio:.2f})"
        if regime.atr_pct > 0.0300:
            return False, f"reversion_too_volatile(atr={regime.atr_pct:.4f})"
        return True, "reversion_ok"

    if cat == "event":
        if regime.volume_ratio >= 1.0 and regime.atr_pct >= 0.0015:
            return True, "event_ok"
        return False, f"event_insufficient_impulse(vol={regime.volume_ratio:.2f},atr={regime.atr_pct:.4f})"

    # unknown/scalp fallback: permissive, only block extremes
    return True, "fallback_ok"
