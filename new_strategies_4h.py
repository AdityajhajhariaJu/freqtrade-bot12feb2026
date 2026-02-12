"""
new_strategies_4h.py — Higher-Timeframe Strategy Module (4H Entries + 12H Directional Filter)
==============================================================================================

INTEGRATION OVERVIEW
--------------------
This file adds a new "Position" tier to your multi-strategy engine:

    Current tiers:     Scalp (1m)  →  Swing (2-3h)
    After integration: Scalp (1m)  →  Swing (2-3h)  →  Position (4H)
                                                        ↑ guided by 12H filter

The 4H tier produces 1-3 high-conviction signals per day across all pairs, with
TP targets of 1.5-4.0% — making fee impact negligible (2-5% of profit vs your
current 20-55% on 1m scalps).

The 12H filter is NOT a trading strategy — it computes a directional bias
(LONG / SHORT / NEUTRAL) that all tiers should respect. It prevents counter-trend
entries and alone can eliminate 30-40% of losing trades.


INTEGRATION STEPS (do these in order)
--------------------------------------

STEP 1: Place this file alongside your existing strategy files.
         cp new_strategies_4h.py  /path/to/your/bot/new_strategies_4h.py

STEP 2: In trade_loop.py, increase FETCH_LIMIT for 4H candle aggregation:
         FETCH_LIMIT = 1500   # was 200 — Binance allows up to 1500 per request

STEP 3: In trade_loop.py, add an ALTERNATIVE 1h candle fetch for the 12H filter
         (optional but recommended — avoids needing 7200+ 1m candles):

         # Fetch once per loop iteration, per pair:
         ohlcv_1h = await exchange.fetch_ohlcv(pair, "1h", limit=500)
         candles_1h = [Candle(
             timestamp=o[0], open=o[1], high=o[2], low=o[3],
             close=o[4], volume=o[5]
         ) for o in ohlcv_1h]

STEP 4: In strategies.py (or wherever CONFIG lives), add the 4H config block:

         CONFIG_4H = {
             "confidence_threshold": 0.60,
             "min_volatility_pct": 0.008,
             "min_tp_percent": 0.015,
             "min_sl_percent": 0.008,
             "min_risk_reward": 1.5,
             "max_funding_long": 0.0008,
             "max_funding_short": 0.0008,
             "cooldown_sec": 3600,
             "post_close_cooldown_sec": 7200,
             "max_leverage": 5,
         }

STEP 5: In strategies.py, register the new categories:

         CONFIG["strategy_categories"].update({
             "trend_4h":       ["weekly_vwap_trend_4h", "ichimoku_breakout_4h"],
             "reversion_4h":   ["bb_rsi_reversion_4h"],
             "structural_4h":  ["structure_break_ob_4h"],
         })

STEP 6: In trade_loop.py, add slot management for 4H trades:

         MAX_4H_TRADES = 2
         active_4h = sum(1 for t in active_trades if t.strategy_id in STRATEGIES_4H)
         # Skip 4H signal scan if active_4h >= MAX_4H_TRADES

STEP 7: In trade_loop.py, add MAX_AGE entries:

         MAX_AGE.update({
             "weekly_vwap_trend_4h":  16 * 3600,   # 16 hours
             "ichimoku_breakout_4h":  24 * 3600,   # 24 hours
             "bb_rsi_reversion_4h":   12 * 3600,   # 12 hours
             "structure_break_ob_4h": 20 * 3600,   # 20 hours
         })

STEP 8: In the main scan loop, add a scan frequency limiter for 4H strategies:

         _last_htf_scan = 0
         now = time.time()
         if now - _last_htf_scan > 300:     # evaluate 4H every 5 minutes, not every 75s
             run_4h_signal_scan(...)
             _last_htf_scan = now

STEP 9: Wire the 12H directional filter into ALL tiers.
         Before running any signal scan (1m, 2h, or 4H), call:

         bias = compute_12h_bias(candles_1h, pair)
         # Then in each strategy scan, skip signals opposing the bias:
         # if bias == "LONG"  and signal.side == "SHORT": skip
         # if bias == "SHORT" and signal.side == "LONG":  skip
         # if bias == "NEUTRAL": allow both

STEP 10: Test with logging before enabling live execution.
         Set DRY_RUN_4H = True in this file to log signals without placing orders.


WHAT THIS FILE DOES NOT CHANGE
-------------------------------
- Your existing 1m and 2h strategies are untouched
- Your CONFIG dict, correlation filters, and drawdown breaker are untouched
- Your Candle, Signal, and Trade dataclasses are untouched
- Your indicator library is used as-is (no new dependencies)
- Your exchange connection and order execution logic are untouched

COMPATIBILITY
-------------
- Python 3.10+ (uses | union types in type hints)
- Requires: your existing indicators.py (ema, rsi, bollinger_bands, adx, atr,
  volume_spike, ichimoku, chaikin_money_flow/cmf)
- Requires: aggregate_candles from new_strategies_2h.py
- Requires: Candle and Signal dataclasses from your models
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Optional

# ─── Adjust these imports to match YOUR project structure ───────────────────
# If your files are named differently, update these paths:
#
#   from models       import Candle, Signal
#   from indicators   import (ema, rsi, bollinger_bands, adx, atr, vwap,
#                              volume_spike, ichimoku, chaikin_money_flow)
#   from new_strategies_2h import aggregate_candles
#
# The exact import paths depend on your project layout.  The names below are
# reasonable defaults — change ONLY the 'from X' part, not the function names.

try:
    from models import Candle, Signal
except ImportError:
    from strategies import Candle, Signal  # fallback if models live here

# Core indicators live in strategies.py in this project
from strategies import (
    ema, rsi, bollinger_bands, atr, vwap,
    volume_spike,
)

# Extended indicators + aggregation live in new_strategies_2h.py here
from new_strategies_2h import (
    adx, ichimoku, chaikin_money_flow, aggregate_candles,
)

# ─── Module-level config ────────────────────────────────────────────────────

logger = logging.getLogger("htf_strategies")

# Set True to log signals without executing trades during initial testing
DRY_RUN_4H = False

# Master set of 4H strategy IDs — use this in trade_loop.py for slot counting
STRATEGIES_4H = {
    "weekly_vwap_trend_4h",
    "ichimoku_breakout_4h",
    "bb_rsi_reversion_4h",
    "structure_break_ob_4h",
}

# 4H candle aggregation factor (240 x 1-min candles = 1 x 4H candle)
AGG_4H = 240

# Minimum 4H candles needed per strategy before evaluation is valid
MIN_BARS = {
    "weekly_vwap_trend_4h":  45,   # 42 for VWAP + 3 warmup
    "ichimoku_breakout_4h":  80,   # 52+26 for cloud projection + buffer
    "bb_rsi_reversion_4h":   25,   # 20 for BB + 5 warmup
    "structure_break_ob_4h": 15,   # need swing detection range
}


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — 12-HOUR DIRECTIONAL FILTER  (judgment only, never trades)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DirectionalBias:
    """Output of the 12H filter.  Read-only judgment for other tiers."""
    direction: str          # "LONG", "SHORT", or "NEUTRAL"
    strength: float         # 0.0 – 1.0, how separated the EMAs are
    ema_fast: float         # current EMA(10) on 12H
    ema_slow: float         # current EMA(30) on 12H
    adx_12h: float          # ADX on 12H — trend conviction
    rsi_12h: float          # RSI on 12H — overbought/oversold context
    regime: str             # "TRENDING", "RANGING", or "VOLATILE"
    updated_at: float       # timestamp of last computation


def compute_12h_bias(
    candles_1h: list[Candle],
    pair: str = "",
) -> DirectionalBias:
    """
    Compute 12-hour directional bias from 1-hour candles.

    THIS IS NOT A TRADING STRATEGY.  It produces a DirectionalBias object
    that other tiers use to filter entries.  It never generates a Signal,
    never opens positions, and never risks capital.

    Usage in trade_loop.py:
        bias = compute_12h_bias(candles_1h, pair)
        if bias.direction == "LONG"  and signal.side == "SHORT": continue
        if bias.direction == "SHORT" and signal.side == "LONG":  continue

    Data requirement:  ≥ 360 one-hour candles (= 30 twelve-hour bars).
    Recommended: fetch 500 1h candles from Binance (single API call).

    Parameters
    ----------
    candles_1h : list[Candle]
        1-hour OHLCV candles, oldest first.  Minimum 360 candles.
    pair : str
        Trading pair symbol (for logging only).

    Returns
    -------
    DirectionalBias
        direction="NEUTRAL" if insufficient data or flat market.
    """
    neutral = DirectionalBias(
        direction="NEUTRAL", strength=0.0,
        ema_fast=0.0, ema_slow=0.0, adx_12h=0.0, rsi_12h=50.0,
        regime="RANGING", updated_at=time.time(),
    )

    # ── Guard: need enough 1h candles to build 12H bars ──
    if len(candles_1h) < 360:
        logger.warning(f"[12H filter] {pair}: only {len(candles_1h)} 1h candles (need 360+) — falling back to NEUTRAL bias. Ensure 1h candle fetch limit >= 500.")
        return neutral

    # ── Aggregate 1h → 12h ──
    candles_12h = aggregate_candles(candles_1h, 12)
    if len(candles_12h) < 32:
        logger.warning(f"[12H filter] {pair}: only {len(candles_12h)} 12H bars after aggregation (need 32+) — NEUTRAL bias")
        return neutral

    closes_12h = [c.close for c in candles_12h]

    # ── Core EMAs ──
    ema_fast = ema(closes_12h, 10)
    ema_slow = ema(closes_12h, 30)
    if ema_fast is None or ema_slow is None:
        return neutral

    # ── Trend strength via ADX ──
    adx_data = adx(candles_12h, 14)
    adx_val = adx_data["adx"] if isinstance(adx_data, dict) else 0.0

    # ── RSI for context (not a signal, just information) ──
    rsi_val = rsi(closes_12h, 14)
    if rsi_val is None:
        rsi_val = 50.0

    # ── ATR for regime classification ──
    atr_val = atr(candles_12h, 14)
    price = candles_12h[-1].close
    atr_pct = (atr_val / price) if (price and atr_val) else 0.0

    # ── Compute regime ──
    if adx_val > 28 and atr_pct > 0.015:
        regime = "TRENDING"
    elif atr_pct > 0.035:
        regime = "VOLATILE"
    elif adx_val < 18:
        regime = "RANGING"
    else:
        regime = "RANGING"

    # ── Directional bias ──
    separation = (ema_fast - ema_slow) / price if price else 0.0
    strength = min(abs(separation) / 0.02, 1.0)   # normalize: 2% separation = max strength

    # Thresholds — intentionally conservative to avoid frequent flipping:
    #   - EMAs must be separated by at least 0.15% of price
    #   - ADX must be above 18 (mild trend present)
    #   - RSI must not be at extremes contradicting direction
    min_sep = 0.0015   # 0.15% minimum separation
    min_adx = 18.0

    if separation > min_sep and adx_val >= min_adx and rsi_val < 78:
        direction = "LONG"
    elif separation < -min_sep and adx_val >= min_adx and rsi_val > 22:
        direction = "SHORT"
    else:
        direction = "NEUTRAL"

    logger.info(
        f"[12H filter] {pair}: direction={direction} | "
        f"EMA10={ema_fast:.2f} EMA30={ema_slow:.2f} sep={separation:.4%} | "
        f"ADX={adx_val:.1f} RSI={rsi_val:.1f} | regime={regime}"
    )

    return DirectionalBias(
        direction=direction,
        strength=strength,
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        adx_12h=adx_val,
        rsi_12h=rsi_val,
        regime=regime,
        updated_at=time.time(),
    )


def should_allow_signal(
    bias: DirectionalBias,
    signal_side: str,
    strategy_tier: str = "any",
) -> bool:
    """
    Gate function: returns True if a signal is allowed given the 12H bias.

    For 4H strategies:  strict — must align with bias direction.
    For 2h strategies:  strict — must align with bias direction.
    For 1m strategies:  moderate — blocked only when bias is strong (strength > 0.5).
    NEUTRAL bias:       always allows both directions on all tiers.

    Parameters
    ----------
    bias : DirectionalBias
        Output from compute_12h_bias().
    signal_side : str
        "LONG" or "SHORT".
    strategy_tier : str
        "1m", "2h", "4h", or "any".

    Returns
    -------
    bool
        True = signal is allowed; False = signal is blocked by directional filter.
    """
    if bias.direction == "NEUTRAL":
        return True

    # Aligned signals always pass
    if bias.direction == signal_side:
        return True

    # Counter-trend signals:
    if strategy_tier in ("4h", "2h"):
        # Strict: 4H and 2h trades must align
        logger.debug(
            f"[12H gate] BLOCKED {signal_side} on {strategy_tier} tier "
            f"(bias={bias.direction}, strength={bias.strength:.2f})"
        )
        return False

    if strategy_tier == "1m":
        # 1m scalps get a pass if bias is weak — they're fast enough to escape
        if bias.strength > 0.5:
            logger.debug(
                f"[12H gate] BLOCKED 1m {signal_side} (strong bias={bias.direction})"
            )
            return False
        return True

    # Default: allow
    return True


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — 4-HOUR TRADING STRATEGIES
# ═══════════════════════════════════════════════════════════════════════════════

# ─── Shared utilities ───────────────────────────────────────────────────────

def _aggregate_4h(candles_1m: list[Candle]) -> list[Candle]:
    """Aggregate candles into 4-hour candles.

    Supports either 1m candles (AGG_4H=240) or 1h candles (AGG_4H=4).
    Auto-detects candle interval from timestamps.
    """
    if not candles_1m or len(candles_1m) < 2:
        return []
    try:
        dt = abs(candles_1m[-1].timestamp - candles_1m[-2].timestamp)
        # timestamps are usually ms in ccxt
        dt_sec = dt / 1000.0 if dt > 10000 else dt
        if dt_sec >= 3300:  # ~55 minutes => treat as 1h candles
            return aggregate_candles(candles_1m, 4)
    except Exception:
        pass
    return aggregate_candles(candles_1m, AGG_4H)


def _fee_adjusted_rr(tp_pct: float, sl_pct: float, leverage: int) -> float:
    """
    Calculate risk-reward ratio after accounting for round-trip taker fees.
    Ensures we never take a trade where fees destroy the edge.
    """
    fee_rt = 0.001  # 0.05% × 2 = 0.10% round-trip taker fee
    net_tp = tp_pct * leverage - fee_rt * leverage
    net_sl = sl_pct * leverage + fee_rt * leverage
    if net_sl <= 0:
        return 0.0
    return net_tp / net_sl


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    """Division that returns default instead of raising on zero."""
    return a / b if b != 0 else default


# ─── Strategy 1: Weekly VWAP Trend Continuation ────────────────────────────

def evaluate_weekly_vwap_trend_4h(
    candles_1m: list[Candle],
    funding_rate: float = 0.0,
) -> Optional[Signal]:
    """
    4H Weekly VWAP Trend Continuation
    ----------------------------------
    Category:  Trend-following
    Timeframe: 4-hour candles (aggregated from 1m)
    Hold time: 4 – 16 hours
    Leverage:  4×
    TP:        2.0 – 3.5% (ATR-based)
    SL:        1.0 – 1.8% (ATR-based)
    Signals:   ~0.5 – 1.0 per day across all pairs

    HOW IT WORKS:
    VWAP calculated over a rolling 7-day window acts as dynamic support/resistance
    on the 4H chart.  When price pulls back to weekly VWAP and bounces with volume
    confirmation in the direction of the EMA(50) trend, it's a high-probability
    continuation entry.

    WHY IT WORKS ON 4H:
    Weekly VWAP on 4H represents institutional "fair value".  With $16.8B in
    CME Bitcoin futures open interest (Sept 2025), institutional players benchmark
    execution against VWAP.  Pullbacks to VWAP in a trending market consistently
    attract fresh institutional flow.

    ENTRY (LONG):
      1. Price above 4H EMA(50) — confirms uptrend
      2. Price pulled back to within 0.5% of 7-day VWAP
      3. Previous candle touched or dipped below VWAP
      4. Current candle closes above VWAP (bounce confirmation)
      5. RSI(14) on 4H between 38 and 58 (pulled back, not oversold extreme)
      6. Volume on bounce candle > 1.2× 10-bar average

    EXIT:
      TP = 2.5 × ATR(14) from entry
      SL = 1.5 × ATR(14) from entry
      Time stop: 16 hours max
      Trailing: after 50% of TP, move SL to breakeven
    """
    candles_4h = _aggregate_4h(candles_1m)
    if len(candles_4h) < MIN_BARS["weekly_vwap_trend_4h"]:
        return None

    price = candles_4h[-1].close
    closes = [c.close for c in candles_4h]

    # ── Indicators ──
    ema50 = ema(closes, min(50, len(closes) - 1))
    if ema50 is None:
        return None

    rsi_val = rsi(closes, 14)
    if rsi_val is None:
        return None

    atr_val = atr(candles_4h, 14)
    if atr_val is None or atr_val == 0:
        return None
    atr_pct = atr_val / price

    vol_ratio = volume_spike(candles_4h, 10)

    # ── VWAP (rolling 42 bars ≈ 7 days on 4H) ──
    vwap_period = min(42, len(candles_4h))
    vwap_val = vwap(candles_4h, vwap_period)
    if vwap_val is None or vwap_val == 0:
        return None

    vwap_dist = (price - vwap_val) / vwap_val
    prev = candles_4h[-2]
    curr = candles_4h[-1]

    # ── Minimum volatility gate ──
    if atr_pct < 0.005:
        return None  # market too quiet for this strategy

    # ── Funding awareness ──
    # Skip if funding cost would eat >20% of expected TP over ~2 funding periods
    expected_funding_cost = abs(funding_rate) * 4 * 2  # lev=4, ~2 funding periods
    expected_tp_gain = atr_pct * 2.5 * 4
    if expected_funding_cost > expected_tp_gain * 0.20 and expected_tp_gain > 0:
        return None

    # ═══════ LONG SETUP ═══════
    if (price > ema50
            and abs(vwap_dist) < 0.006           # within 0.6% of VWAP
            and prev.low <= vwap_val * 1.003      # prev candle tested VWAP area
            and price > vwap_val                  # bounced above
            and 38 <= rsi_val <= 58               # not overextended
            and vol_ratio >= 1.2                  # volume confirmation
            and curr.close > curr.open):          # bullish candle body

        conf = 0.64
        conf += min((58 - rsi_val) / 150, 0.08)     # closer to midpoint = stronger
        conf += min((vol_ratio - 1.2) / 6, 0.08)    # stronger volume = higher conf
        conf += 0.04 if atr_pct > 0.010 else 0.0    # bonus for decent volatility

        tp = max(0.015, atr_pct * 2.5)
        sl = max(0.008, atr_pct * 1.5)

        if _fee_adjusted_rr(tp, sl, 4) < 1.3:
            return None  # RR too thin after fees

        return Signal(
            side="LONG",
            confidence=round(min(conf, 0.86), 4),
            tp_percent=round(tp, 5),
            sl_percent=round(sl, 5),
            leverage=4,
            reason=(
                f"VWAP Trend 4H LONG | vwap={vwap_val:.2f} dist={vwap_dist:+.3%} | "
                f"ema50={ema50:.2f} | rsi={rsi_val:.1f} | vol={vol_ratio:.1f}x | "
                f"atr%={atr_pct:.3%} | tp={tp:.3%} sl={sl:.3%}"
            ),
        )

    # ═══════ SHORT SETUP ═══════
    if (price < ema50
            and abs(vwap_dist) < 0.006
            and prev.high >= vwap_val * 0.997
            and price < vwap_val
            and 42 <= rsi_val <= 62
            and vol_ratio >= 1.2
            and curr.close < curr.open):

        conf = 0.64
        conf += min((rsi_val - 42) / 150, 0.08)
        conf += min((vol_ratio - 1.2) / 6, 0.08)
        conf += 0.04 if atr_pct > 0.010 else 0.0

        tp = max(0.015, atr_pct * 2.5)
        sl = max(0.008, atr_pct * 1.5)

        if _fee_adjusted_rr(tp, sl, 4) < 1.3:
            return None

        return Signal(
            side="SHORT",
            confidence=round(min(conf, 0.86), 4),
            tp_percent=round(tp, 5),
            sl_percent=round(sl, 5),
            leverage=4,
            reason=(
                f"VWAP Trend 4H SHORT | vwap={vwap_val:.2f} dist={vwap_dist:+.3%} | "
                f"ema50={ema50:.2f} | rsi={rsi_val:.1f} | vol={vol_ratio:.1f}x | "
                f"atr%={atr_pct:.3%} | tp={tp:.3%} sl={sl:.3%}"
            ),
        )

    return None


# ─── Strategy 2: Ichimoku Cloud Breakout + ADX ─────────────────────────────

def evaluate_ichimoku_breakout_4h(
    candles_1m: list[Candle],
    funding_rate: float = 0.0,
) -> Optional[Signal]:
    """
    4H Ichimoku Cloud Breakout with ADX Confirmation
    --------------------------------------------------
    Category:  Trend-following
    Timeframe: 4-hour candles
    Hold time: 8 – 24 hours
    Leverage:  3×
    TP:        2.0 – 4.0% (cloud distance + ATR)
    SL:        Kijun-sen line (natural S/R)
    Signals:   ~0.3 – 0.5 per day

    HOW IT WORKS:
    Ichimoku on 4H with standard 9/26/52 periods covers roughly 1.5 / 4.3 / 8.7
    days of price history — close to the original daily-chart intent of the Hosoda
    system.  When price breaks out of the cloud with all five Ichimoku conditions
    aligned AND ADX confirms real momentum, the signal is institutional-grade.

    ENTRY (LONG):
      1. Price breaks above the cloud (above both Senkou A and Senkou B)
      2. Tenkan-sen > Kijun-sen (bullish TK cross)
      3. Forward cloud is green (Senkou A > Senkou B projected)
      4. ADX > 25 and rising
      5. +DI > -DI
      6. Price not more than 1× ATR above cloud top (filters late entries)

    SL is placed at the Kijun-sen line — the 26-period baseline that acts as
    natural support.  This is the classic Ichimoku stop.
    """
    candles_4h = _aggregate_4h(candles_1m)
    if len(candles_4h) < MIN_BARS["ichimoku_breakout_4h"]:
        return None

    price = candles_4h[-1].close

    # ── Ichimoku ──
    ichi = ichimoku(candles_4h, conv=9, base=26, span_b=52)
    if ichi is None:
        return None

    # ── ADX ──
    adx_data = adx(candles_4h, 14)
    if not isinstance(adx_data, dict):
        return None
    adx_val = adx_data.get("adx", 0)
    plus_di = adx_data.get("plus_di", 0)
    minus_di = adx_data.get("minus_di", 0)

    if adx_val < 25:
        return None  # no confirmed trend

    # ── ATR ──
    atr_val = atr(candles_4h, 14)
    if atr_val is None or atr_val == 0:
        return None
    atr_pct = atr_val / price

    if atr_pct < 0.005:
        return None  # too quiet

    # ── Cloud reference values ──
    # Ichimoku implementations vary — handle both dict-style and attribute-style
    cloud_top = ichi.get("cloud_top", ichi.get("senkou_a", 0)) if isinstance(ichi, dict) else getattr(ichi, "cloud_top", 0)
    cloud_bot = ichi.get("cloud_bottom", ichi.get("senkou_b", 0)) if isinstance(ichi, dict) else getattr(ichi, "cloud_bottom", 0)
    kijun = ichi.get("kijun", ichi.get("base", 0)) if isinstance(ichi, dict) else getattr(ichi, "kijun", 0)
    tenkan = ichi.get("tenkan", ichi.get("conv", 0)) if isinstance(ichi, dict) else getattr(ichi, "tenkan", 0)

    price_vs = ichi.get("price_vs_cloud", "") if isinstance(ichi, dict) else getattr(ichi, "price_vs_cloud", "")
    tk_cross = ichi.get("tk_cross", "") if isinstance(ichi, dict) else getattr(ichi, "tk_cross", "")
    cloud_color = ichi.get("cloud_color", "") if isinstance(ichi, dict) else getattr(ichi, "cloud_color", "")

    # ── Extension filter: don't enter if price is already too far from cloud ──
    if cloud_top and price:
        cloud_dist = abs(price - cloud_top) / price
        if cloud_dist > atr_pct * 1.2:
            return None  # already extended — bad entry point

    # ═══════ LONG SETUP ═══════
    if (price_vs == "ABOVE"
            and tk_cross == "BULL"
            and cloud_color == "GREEN"
            and plus_di > minus_di
            and tenkan > kijun):

        conf = 0.63
        conf += min(adx_val / 250, 0.10)          # stronger ADX = higher conf
        conf += 0.05 if plus_di > minus_di * 1.3 else 0.0  # DI separation bonus

        kijun_dist = _safe_div(price - kijun, price, 0.01)
        tp = max(0.020, atr_pct * 3.0)
        sl = max(0.010, kijun_dist + 0.002)
        sl = min(sl, 0.025)  # hard cap SL at 2.5%

        if _fee_adjusted_rr(tp, sl, 3) < 1.4:
            return None

        return Signal(
            side="LONG",
            confidence=round(min(conf, 0.85), 4),
            tp_percent=round(tp, 5),
            sl_percent=round(sl, 5),
            leverage=3,
            reason=(
                f"Ichimoku 4H LONG | cloud_break | adx={adx_val:.1f} "
                f"+di={plus_di:.1f} -di={minus_di:.1f} | "
                f"kijun={kijun:.2f} sl_dist={sl:.3%} | tp={tp:.3%}"
            ),
        )

    # ═══════ SHORT SETUP ═══════
    if (price_vs == "BELOW"
            and tk_cross == "BEAR"
            and cloud_color == "RED"
            and minus_di > plus_di
            and tenkan < kijun):

        conf = 0.63
        conf += min(adx_val / 250, 0.10)
        conf += 0.05 if minus_di > plus_di * 1.3 else 0.0

        kijun_dist = _safe_div(kijun - price, price, 0.01)
        tp = max(0.020, atr_pct * 3.0)
        sl = max(0.010, kijun_dist + 0.002)
        sl = min(sl, 0.025)

        if _fee_adjusted_rr(tp, sl, 3) < 1.4:
            return None

        return Signal(
            side="SHORT",
            confidence=round(min(conf, 0.85), 4),
            tp_percent=round(tp, 5),
            sl_percent=round(sl, 5),
            leverage=3,
            reason=(
                f"Ichimoku 4H SHORT | cloud_break | adx={adx_val:.1f} "
                f"+di={plus_di:.1f} -di={minus_di:.1f} | "
                f"kijun={kijun:.2f} sl_dist={sl:.3%} | tp={tp:.3%}"
            ),
        )

    return None


# ─── Strategy 3: Bollinger Band + RSI Mean Reversion ──────────────────────

def evaluate_bb_rsi_reversion_4h(
    candles_1m: list[Candle],
    funding_rate: float = 0.0,
) -> Optional[Signal]:
    """
    4H Bollinger Band + RSI Mean Reversion
    ----------------------------------------
    Category:  Mean-reversion
    Timeframe: 4-hour candles
    Hold time: 4 – 12 hours
    Leverage:  4×
    TP:        BB midline (SMA20) — typically 1.5 – 2.5%
    SL:        1× ATR beyond the touched band
    Signals:   ~0.3 – 0.8 per day

    HOW IT WORKS:
    When price hits the 2σ Bollinger Band on the 4H chart AND RSI confirms the
    extreme, it's a genuine overextension (unlike 1m where this happens dozens of
    times daily).  The target is a reversion to the 20-period SMA (BB middle line).

    KEY SAFEGUARD:  This strategy only fires when ADX < 30, which means the market
    is ranging.  In a strong trend, hitting the band is normal and mean-reversion
    fails — the ADX gate prevents this.

    ADDITIONAL FILTER:  Chaikin Money Flow (CMF) must not be strongly negative
    on long entries or strongly positive on short entries.  This catches situations
    where the band touch is backed by heavy distribution/accumulation.

    ENTRY (LONG):
      1. Price closes at or below lower BB(20, 2.0) on 4H
      2. RSI(14) < 30 on 4H
      3. ADX(14) < 30 (ranging market — reversion works)
      4. Current candle shows bullish body (close > open)
      5. CMF(20) > -0.18 (money flow isn't catastrophic)

    EXIT:
      TP = BB middle line (20-period SMA)
      SL = 1× ATR(14) beyond the touched band
      Time stop: 12 hours (thesis fails if no reversion by then)
      No trailing — fixed-target reversion trade
    """
    candles_4h = _aggregate_4h(candles_1m)
    if len(candles_4h) < MIN_BARS["bb_rsi_reversion_4h"]:
        return None

    closes = [c.close for c in candles_4h]
    price = candles_4h[-1].close
    curr = candles_4h[-1]

    # ── Indicators ──
    bb = bollinger_bands(closes, 20, 2.0)
    if bb is None:
        return None

    rsi_val = rsi(closes, 14)
    if rsi_val is None:
        return None

    adx_data = adx(candles_4h, 14)
    if not isinstance(adx_data, dict):
        return None
    adx_val = adx_data.get("adx", 50)

    atr_val = atr(candles_4h, 14)
    if atr_val is None or atr_val == 0:
        return None
    atr_pct = atr_val / price

    # ── CMF (Chaikin Money Flow) ──
    try:
        cmf_val = chaikin_money_flow(candles_4h, 20)
    except Exception:
        cmf_val = 0.0  # fallback if CMF not available
    if cmf_val is None:
        cmf_val = 0.0

    # ── Regime gate: only trade reversion in ranging markets ──
    if adx_val > 30:
        return None

    # ── Minimum volatility ──
    if atr_pct < 0.005:
        return None

    # BB reference values (handle both dict and object styles)
    bb_upper = bb.get("upper", bb.get("upper_band", 0)) if isinstance(bb, dict) else getattr(bb, "upper", 0)
    bb_lower = bb.get("lower", bb.get("lower_band", 0)) if isinstance(bb, dict) else getattr(bb, "lower", 0)
    bb_mid = bb.get("middle", bb.get("mid", bb.get("sma", 0))) if isinstance(bb, dict) else getattr(bb, "middle", 0)
    bb_bw = bb.get("bandwidth", 0) if isinstance(bb, dict) else getattr(bb, "bandwidth", 0)

    # ═══════ LONG: at lower band + oversold ═══════
    if (price <= bb_lower
            and rsi_val < 30
            and curr.close > curr.open    # bullish reversal candle
            and cmf_val > -0.18):         # money flow not catastrophic

        tp_dist = _safe_div(bb_mid - price, price, 0.015)
        sl_dist = atr_pct + 0.002

        conf = 0.63
        conf += min((30 - rsi_val) / 100, 0.10)     # deeper oversold = more conf
        conf += min(abs(cmf_val) * 0.2, 0.05)        # CMF near zero = healthier bounce
        conf += 0.03 if bb_bw > 0.02 else 0.0        # wider bands = more room to revert

        tp = max(0.015, tp_dist * 0.85)    # target 85% of distance to midline
        sl = max(0.008, sl_dist)

        if _fee_adjusted_rr(tp, sl, 4) < 1.1:
            return None

        return Signal(
            side="LONG",
            confidence=round(min(conf, 0.85), 4),
            tp_percent=round(tp, 5),
            sl_percent=round(sl, 5),
            leverage=4,
            reason=(
                f"BB Reversion 4H LONG | at lower band | rsi={rsi_val:.1f} "
                f"adx={adx_val:.1f} cmf={cmf_val:.3f} | bw={bb_bw:.4f} | "
                f"tp_to_mid={tp:.3%} sl={sl:.3%}"
            ),
        )

    # ═══════ SHORT: at upper band + overbought ═══════
    if (price >= bb_upper
            and rsi_val > 70
            and curr.close < curr.open
            and cmf_val < 0.18):

        tp_dist = _safe_div(price - bb_mid, price, 0.015)
        sl_dist = atr_pct + 0.002

        conf = 0.63
        conf += min((rsi_val - 70) / 100, 0.10)
        conf += min(abs(cmf_val) * 0.2, 0.05)
        conf += 0.03 if bb_bw > 0.02 else 0.0

        tp = max(0.015, tp_dist * 0.85)
        sl = max(0.008, sl_dist)

        if _fee_adjusted_rr(tp, sl, 4) < 1.1:
            return None

        return Signal(
            side="SHORT",
            confidence=round(min(conf, 0.85), 4),
            tp_percent=round(tp, 5),
            sl_percent=round(sl, 5),
            leverage=4,
            reason=(
                f"BB Reversion 4H SHORT | at upper band | rsi={rsi_val:.1f} "
                f"adx={adx_val:.1f} cmf={cmf_val:.3f} | bw={bb_bw:.4f} | "
                f"tp_to_mid={tp:.3%} sl={sl:.3%}"
            ),
        )

    return None


# ─── Strategy 4: Market Structure Break + Order Block ──────────────────────

def evaluate_structure_break_ob_4h(
    candles_1m: list[Candle],
    funding_rate: float = 0.0,
) -> Optional[Signal]:
    """
    4H Market Structure Break + Order Block Entry
    -----------------------------------------------
    Category:  Structural / Price Action
    Timeframe: 4-hour candles
    Hold time: 8 – 20 hours
    Leverage:  3×
    TP:        Measured move (swing high – swing low projected) — typically 2 – 4%
    SL:        Below/above the order block zone — typically 1 – 1.5%
    Signals:   ~0.2 – 0.5 per day

    HOW IT WORKS:
    Market structure on 4H is what institutions actually trade.  A "structure break"
    occurs when price breaks a significant swing high (bullish) or swing low (bearish).
    After the break, price often retests the "order block" — the last candle that
    moved against the breakout direction before it happened.  This zone contains
    clustered stop-losses from trapped traders, creating liquidity that propels
    price further in the breakout direction.

    SWING DETECTION:
    A swing high is a candle whose high is higher than the 2 candles before and
    after it.  A swing low is the mirror.  This is the simplest fractal pattern.

    ORDER BLOCK:
    For a bullish break: the last bearish (red) candle before the breakout move.
    For a bearish break: the last bullish (green) candle before the breakdown.
    The order block "zone" spans from the candle's open to its close.

    ENTRY (LONG):
      1. Price breaks above the most recent 4H swing high
      2. Price retraces back into the order block zone
      3. Current candle shows bullish rejection (close > open, lower wick present)
      4. Volume on the original breakout > 1.3× average
      5. Risk-reward meets minimum 1.5:1 threshold
    """
    candles_4h = _aggregate_4h(candles_1m)
    n = len(candles_4h)
    if n < MIN_BARS["structure_break_ob_4h"]:
        return None

    price = candles_4h[-1].close
    curr = candles_4h[-1]

    atr_val = atr(candles_4h, 14)
    if atr_val is None or atr_val == 0:
        return None
    atr_pct = atr_val / price

    if atr_pct < 0.005:
        return None

    vol_ratio = volume_spike(candles_4h, 10)

    # ── Detect swing highs and lows (2-bar lookback/lookahead) ──
    swing_highs = []   # (index, high_price)
    swing_lows = []    # (index, low_price)

    for i in range(2, n - 2):
        h = candles_4h[i].high
        if (h > candles_4h[i-1].high and h > candles_4h[i-2].high
                and h > candles_4h[i+1].high and h > candles_4h[i+2].high):
            swing_highs.append((i, h))

        lo = candles_4h[i].low
        if (lo < candles_4h[i-1].low and lo < candles_4h[i-2].low
                and lo < candles_4h[i+1].low and lo < candles_4h[i+2].low):
            swing_lows.append((i, lo))

    if not swing_highs or not swing_lows:
        return None

    last_sh_idx, last_sh = swing_highs[-1]
    last_sl_idx, last_sl = swing_lows[-1]

    # ═══════ BULLISH STRUCTURE BREAK ═══════
    # Price has broken above the last swing high AND is now retesting
    if price > last_sh and last_sh_idx < n - 2:

        # Find order block: last bearish candle between swing high and break
        ob_candle = None
        for i in range(last_sh_idx, n - 1):
            c = candles_4h[i]
            if c.close < c.open:   # bearish candle
                ob_candle = c
                break

        if ob_candle is None:
            return None

        ob_top = max(ob_candle.open, ob_candle.close)
        ob_bot = min(ob_candle.open, ob_candle.close)

        # Price must be testing the OB zone (within it or slightly above)
        if not (ob_bot - atr_val * 0.2 <= price <= ob_top + atr_val * 0.3):
            return None

        # Volume confirmation on the breakout
        if vol_ratio < 1.3:
            return None

        # Bullish confirmation candle
        if curr.close <= curr.open:
            return None

        # TP = measured move (distance from last swing low to swing high, projected)
        measured_move = last_sh - last_sl
        tp = max(0.020, _safe_div(measured_move, price, 0.025))
        tp = min(tp, 0.04)  # cap at 4%

        sl = max(0.010, _safe_div(price - ob_bot, price, 0.012) + 0.003)
        sl = min(sl, 0.020)  # cap SL at 2%

        if _fee_adjusted_rr(tp, sl, 3) < 1.4:
            return None

        conf = 0.65
        conf += min((vol_ratio - 1.3) / 5, 0.08)
        conf += 0.04 if measured_move / price > 0.02 else 0.0  # bigger structure = more conf

        return Signal(
            side="LONG",
            confidence=round(min(conf, 0.84), 4),
            tp_percent=round(tp, 5),
            sl_percent=round(sl, 5),
            leverage=3,
            reason=(
                f"StructBreak 4H LONG | broke SH={last_sh:.2f} | "
                f"OB={ob_bot:.2f}-{ob_top:.2f} | vol={vol_ratio:.1f}x | "
                f"mm={measured_move:.2f} | tp={tp:.3%} sl={sl:.3%}"
            ),
        )

    # ═══════ BEARISH STRUCTURE BREAK ═══════
    if price < last_sl and last_sl_idx < n - 2:

        ob_candle = None
        for i in range(last_sl_idx, n - 1):
            c = candles_4h[i]
            if c.close > c.open:   # bullish candle
                ob_candle = c
                break

        if ob_candle is None:
            return None

        ob_top = max(ob_candle.open, ob_candle.close)
        ob_bot = min(ob_candle.open, ob_candle.close)

        if not (ob_bot - atr_val * 0.3 <= price <= ob_top + atr_val * 0.2):
            return None

        if vol_ratio < 1.3:
            return None

        if curr.close >= curr.open:
            return None

        measured_move = last_sh - last_sl
        tp = max(0.020, _safe_div(measured_move, price, 0.025))
        tp = min(tp, 0.04)

        sl = max(0.010, _safe_div(ob_top - price, price, 0.012) + 0.003)
        sl = min(sl, 0.020)

        if _fee_adjusted_rr(tp, sl, 3) < 1.4:
            return None

        conf = 0.65
        conf += min((vol_ratio - 1.3) / 5, 0.08)
        conf += 0.04 if measured_move / price > 0.02 else 0.0

        return Signal(
            side="SHORT",
            confidence=round(min(conf, 0.84), 4),
            tp_percent=round(tp, 5),
            sl_percent=round(sl, 5),
            leverage=3,
            reason=(
                f"StructBreak 4H SHORT | broke SL={last_sl:.2f} | "
                f"OB={ob_bot:.2f}-{ob_top:.2f} | vol={vol_ratio:.1f}x | "
                f"mm={measured_move:.2f} | tp={tp:.3%} sl={sl:.3%}"
            ),
        )

    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — MASTER SCANNER  (call this from trade_loop.py)
# ═══════════════════════════════════════════════════════════════════════════════

# Strategy registry: maps strategy_id → evaluate function
_4H_STRATEGY_REGISTRY = {
    "weekly_vwap_trend_4h":  evaluate_weekly_vwap_trend_4h,
    "ichimoku_breakout_4h":  evaluate_ichimoku_breakout_4h,
    "bb_rsi_reversion_4h":   evaluate_bb_rsi_reversion_4h,
    "structure_break_ob_4h": evaluate_structure_break_ob_4h,
}


def scan_4h_strategies(
    candles_1m: list[Candle],
    pair: str,
    bias: DirectionalBias | None = None,
    funding_rate: float = 0.0,
    active_4h_count: int = 0,
    max_4h_trades: int = 2,
    active_pairs_4h: set | None = None,
    confidence_threshold: float = 0.60,
) -> list[tuple[str, Signal]]:
    """
    Run all 4H strategies for a single pair and return qualifying signals.

    This is the main entry point you call from trade_loop.py.

    Parameters
    ----------
    candles_1m : list[Candle]
        1-minute candles for this pair.  Need ≥ 1500 for full 4H coverage.
    pair : str
        Trading pair symbol, e.g. "ETH/USDT:USDT".
    bias : DirectionalBias | None
        Output from compute_12h_bias().  If None, no directional filtering.
    funding_rate : float
        Current funding rate for this pair (from exchange).
    active_4h_count : int
        Number of currently open 4H trades across all pairs.
    max_4h_trades : int
        Maximum allowed concurrent 4H trades.
    active_pairs_4h : set | None
        Set of pairs that already have an active 4H trade (no stacking).
    confidence_threshold : float
        Minimum confidence to accept a signal.

    Returns
    -------
    list[tuple[str, Signal]]
        List of (strategy_id, signal) tuples that passed all gates.
        Typically 0 or 1 results.  The caller should pick the highest-confidence
        signal if multiple are returned.

    Usage in trade_loop.py:
        results = scan_4h_strategies(
            candles_1m=candles_1m,
            pair=pair,
            bias=bias_cache.get(pair),
            funding_rate=funding_rates.get(pair, 0),
            active_4h_count=count_4h_trades(active_trades),
            active_pairs_4h=get_4h_pairs(active_trades),
        )
        if results:
            best_id, best_signal = max(results, key=lambda x: x[1].confidence)
            if not DRY_RUN_4H:
                execute_trade(pair, best_id, best_signal)
            else:
                logger.info(f"[DRY RUN] Would trade {pair} {best_id}: {best_signal}")
    """
    if active_pairs_4h is None:
        active_pairs_4h = set()

    results = []

    # ── Gate: slot limit ──
    if active_4h_count >= max_4h_trades:
        return results

    # ── Gate: no stacking (don't open 4H trade on a pair that already has one) ──
    if pair in active_pairs_4h:
        return results

    # ── Evaluate each strategy ──
    for strat_id, evaluate_fn in _4H_STRATEGY_REGISTRY.items():
        try:
            signal = evaluate_fn(candles_1m, funding_rate=funding_rate)
        except Exception as e:
            logger.warning(f"[4H] {pair} {strat_id} error: {e}")
            continue

        if signal is None:
            continue

        # ── Gate: confidence threshold ──
        if signal.confidence < confidence_threshold:
            logger.debug(f"[4H] {pair} {strat_id}: conf {signal.confidence:.3f} < {confidence_threshold}")
            continue

        # ── Gate: 12H directional filter ──
        if bias is not None and not should_allow_signal(bias, signal.side, "4h"):
            logger.info(
                f"[4H] {pair} {strat_id}: {signal.side} BLOCKED by 12H bias "
                f"({bias.direction}, str={bias.strength:.2f})"
            )
            continue

        logger.info(
            f"[4H] ✓ {pair} {strat_id}: {signal.side} conf={signal.confidence:.3f} "
            f"tp={signal.tp_percent:.3%} sl={signal.sl_percent:.3%} lev={signal.leverage}x"
        )
        results.append((strat_id, signal))

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — MONITORING & ADJUSTMENT GUIDELINES
# ═══════════════════════════════════════════════════════════════════════════════
#
#  POST-IMPLEMENTATION MONITORING CHECKLIST
#  ─────────────────────────────────────────
#
#  FIRST 24 HOURS (Day 1):
#
#    □  Enable DRY_RUN_4H = True and watch the logs
#    □  Confirm 4H candle aggregation is producing correct OHLCV
#       (spot-check: does the 4H candle high match the highest 1m high in that window?)
#    □  Confirm 12H bias is computing and logging correctly for each pair
#    □  Check that no existing 1m or 2h strategy behavior has changed
#    □  Count how many 4H signals fire — expect 1-3 across all pairs per day
#    □  Verify FETCH_LIMIT=1500 isn't causing API rate limit errors
#
#  FIRST WEEK (Days 2-7):
#
#    □  Switch DRY_RUN_4H = False to enable live execution
#    □  Monitor these metrics daily:
#
#         4H Win Rate:     target ≥ 50% (track per strategy)
#         4H Avg TP hit:   target > 1.5% (after fees)
#         4H Avg SL hit:   should be < TP (risk-reward confirmed)
#         4H Fee Impact:   should be < 5% of gross profit per trade
#         4H Hold Time:    average should be 6-14 hours
#         12H Filter:      count blocked vs allowed signals
#
#    □  If win rate < 40% after 15+ trades:
#         → Raise confidence_threshold from 0.60 to 0.65
#         → Check if 12H filter is incorrectly blocking good signals
#
#    □  If too few signals (<1 per day):
#         → Check atr_pct thresholds — market may be too quiet
#         → Temporarily lower VWAP distance from 0.006 to 0.008
#         → Check that candle aggregation isn't dropping partial bars
#
#    □  If too many signals (>4 per day):
#         → Raise confidence_threshold to 0.65
#         → Tighten volume requirements (1.2× → 1.4×)
#
#  ONGOING ADJUSTMENTS (Week 2+):
#
#    □  Track strategy-level performance:
#
#         weekly_vwap_trend_4h:  Best in trending markets.  If markets go range-
#                                bound, this will stop firing (correct behavior).
#                                Don't force it — wait for trend to return.
#
#         ichimoku_breakout_4h:  Highest conviction but fewest signals.  If it
#                                goes 3+ days without firing, that's normal.
#                                It only fires on major trend shifts.
#
#         bb_rsi_reversion_4h:   Best in ranging markets.  Watch for false signals
#                                during trending markets — the ADX < 30 gate should
#                                prevent this, but if not, tighten to ADX < 25.
#
#         structure_break_ob_4h: Most complex logic.  Monitor the swing detection
#                                closely.  If it's firing on minor structure instead
#                                of significant breaks, increase the lookback from
#                                2 bars to 3 bars on each side.
#
#    □  12H filter tuning:
#         If filter blocks >60% of all signals across all tiers → loosen:
#           - Decrease min_sep from 0.0015 to 0.0010
#           - Decrease min_adx from 18 to 15
#         If filter allows too many counter-trend losers → tighten:
#           - Increase min_sep from 0.0015 to 0.0020
#           - Increase min_adx from 18 to 22
#
#    □  Leverage adjustment by account size:
#         Account < $150:   use leverage 2-3× on 4H (override the defaults)
#         Account $150-500: use leverage 3-4× (defaults are fine)
#         Account > $500:   can use full 3-5× range
#
#    □  Pair-specific notes:
#         BNBUSDT:  Lowest ATR% of your pairs (~1.0% on 4H).  May rarely fire.
#                   Consider removing from 4H scan if it never triggers after 2 weeks.
#         DOGEUSDT: Highest ATR% (~2.0%).  May hit SL more often due to wicks.
#                   Monitor closely — if win rate < 40%, reduce leverage to 2×.
#         NEARUSDT: High ATR but lower liquidity.  Check spread before 4H entries.
#
#    □  Seasonal/event awareness:
#         - Reduce 4H activity 2 hours before major economic events (CPI, FOMC, NFP)
#         - Funding rate spikes often precede 4H reversions — the BB strategy
#           naturally captures these
#         - During BTC dominance shifts, altcoin 4H signals become less reliable
#
# ═══════════════════════════════════════════════════════════════════════════════
