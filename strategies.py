"""
Multi-strategy crypto futures engine — Python port
Provided by user (file_238). Kept as-is.
"""

from __future__ import annotations
import time
import math
import logging
from dataclasses import dataclass, field
from typing import Optional, Literal

logger = logging.getLogger(__name__)
VERBOSE_FILTER_LOG = True
_FUNDING_CTX = {}


@dataclass
class Candle:
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: float = 0.0

@dataclass
class Signal:
    side: Literal["LONG", "SHORT"]
    confidence: float
    tp_percent: float
    sl_percent: float
    leverage: int
    reason: str

@dataclass
class TradeEconomics:
    total_fees: float
    net_tp: float
    net_sl: float
    tp_roi: float
    sl_roi: float
    risk_reward: float
    breakeven_move: float
    is_profitable: bool

@dataclass
class TradeSignal:
    pair: str
    strategy_id: str
    strategy_name: str
    strategy_category: str
    side: Literal["LONG", "SHORT"]
    confidence: float
    entry_price: float
    tp_price: float
    sl_price: float
    leverage: int
    trade_size: float
    reason: str
    economics: TradeEconomics
    timestamp: float = field(default_factory=time.time)
    _filtered: bool = field(default=False, repr=False)
    _filter_reason: str = field(default="", repr=False)

@dataclass
class ActiveTrade:
    pair: str
    strategy_id: str = ""
    side: Literal["LONG", "SHORT"] = "LONG"

@dataclass
class BTCMacroStatus:
    is_macro_move: bool
    direction: Optional[Literal["LONG", "SHORT"]]
    magnitude: float
    btc_price: float = 0.0

@dataclass
class DrawdownStatus:
    paused: bool
    drawdown: float
    peak: float = 0.0
    threshold: float = 0.0
    message: Optional[str] = None

@dataclass
class ScanDiagnostics:
    raw_count: int = 0
    after_cooldown: int = 0
    after_flood: int = 0
    after_category: int = 0
    final: int = 0
    reason: str = ""

@dataclass
class ScanResult:
    signals: list[TradeSignal] = field(default_factory=list)
    filtered: list[TradeSignal] = field(default_factory=list)
    btc_macro: Optional[BTCMacroStatus] = None
    drawdown: Optional[DrawdownStatus] = None
    diagnostics: ScanDiagnostics = field(default_factory=ScanDiagnostics)

CONFIG = {
    "min_trade_size": 10,
    "max_concurrent_trades": 8,
    "risk_per_trade": 0.02,
    "confidence_threshold": 0.66,
    "confirm_signal": True,
    "min_volatility_pct": 0.002,  # 0.2%
    "trend_ema_fast": 50,
    "trend_ema_slow": 200,
    "regime_min_trend_pct": 0.0015,
    "atr_period": 14,
    "vol_target_pct": 0.004,
    "tp_multiplier": 1.20,
    "min_tp_percent": 0.0018,
    "min_risk_reward": 1.3,
    "max_leverage": 10,
    "divergence_boost": {"enabled": True, "pairs_min": 3, "boost": 0.04, "lookback": 20},
    "oi_boost": {"enabled": True, "boost": 0.06},
    "anti_signal": {"enabled": True, "threshold": 0.20, "cooldown_sec": 21600},
    "structural_quality": {"enabled": True, "vol_mult": 2.5, "sr_dist": 0.003, "boost": 0.04},
    "agreement_boost": {"enabled": True, "min_strategies": 2, "boost": 0.03},
    "adaptive_params": {
        "enabled": True,
        "update_sec": 1800,
        "regime_atr_pct": {"low": 0.0015, "high": 0.0045},
        "rsi_period": {"low": 18, "mid": 14, "high": 12},
        "rsi_oversold": {"low": 32, "mid": 30, "high": 28},
        "rsi_overbought": {"low": 68, "mid": 70, "high": 72},
        "ema_fast": {"low": 30, "mid": 20, "high": 15},
        "ema_slow": {"low": 80, "mid": 50, "high": 35}
    },
    "max_funding_long": 0.0005,
    "max_funding_short": 0.0005,
    "fees": {"maker_rate": 0.0002, "taker_rate": 0.0005},
    "correlation": {
        "max_same_side_signals": 2,
        "max_per_category": 1,
        "same_side_window_sec": 120,
        "btc_move_lookback": 5,
        "btc_move_threshold": 0.008,
        "cooldown_sec": 450,
        "post_close_cooldown_sec": 600,
        "max_drawdown_pause": 1.0,
    },
        "pair_filters": {
        "LINKUSDT": ["rsi_snap", "stoch_cross", "obv_divergence"],
        "BNBUSDT": ["rsi_snap", "vwap_bounce", "ichimoku_cloud", "keltner_reversion", "donchian_breakout", "supertrend_flip", "adx_di_cross", "fib_pullback", "cmf_divergence", "vp_poc_reversion", "pivot_bounce", "vwap_sd_reversion", "mtf_ema_ribbon", "funding_fade", "bb_kc_squeeze"],
    },
    "strategy_categories": {
        "trend": ["ema_scalp", "triple_ema", "macd_flip", "atr_breakout"],
        "reversion": ["rsi_snap", "stoch_cross", "obv_divergence", "funding_fade"],
        "structural": ["vwap_bounce", "engulfing_sr"],
        "trend_4h": ["weekly_vwap_trend_4h", "ichimoku_breakout_4h"],
        "reversion_4h": ["bb_rsi_reversion_4h"],
        "structural_4h": ["structure_break_ob_4h"],
    },
    "pairs": [
        "ETHUSDT", "LTCUSDT", "DOGEUSDT", "AVAXUSDT", "NEARUSDT", "LINKUSDT", "BNBUSDT",
    ],
    "timeframes": ["1m", "5m"],
    "max_1m_trades": 4,
    "max_2h_trades": 4,
    "max_4h_trades": 2,
}

# Separate config for 2-3h strategies
CONFIG_2H = {
    "confidence_threshold": 0.63,
    "confirm_signal": False,
    "min_volatility_pct": 0.006,
    "tp_multiplier": 1.10,
    "min_tp_percent": 0.005,
    "min_sl_percent": 0.003,
    "min_risk_reward": 1.5,
    "max_funding_long": 0.0005,
    "max_funding_short": 0.0005,
    "cooldown_sec": 1200,
    "post_close_cooldown_sec": 1800,
    "skip_trend_ema_filter": True,
    "skip_regime_filter": True,
    "skip_htf_alignment": True,
}

# Separate config for 4h strategies
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

STRATEGIES_2H = {
    "ichimoku_cloud", "keltner_reversion", "donchian_breakout", "supertrend_flip",
    "vp_poc_reversion", "pivot_bounce", "funding_fade", "bb_kc_squeeze",
}

def is_2h_strategy(strategy_id: str) -> bool:
    return strategy_id in STRATEGIES_2H

def is_4h_strategy(strategy_id: str) -> bool:
    return strategy_id in CONFIG.get("strategy_categories", {}).get("trend_4h", []) \
        or strategy_id in CONFIG.get("strategy_categories", {}).get("reversion_4h", []) \
        or strategy_id in CONFIG.get("strategy_categories", {}).get("structural_4h", [])


def set_funding_context(funding: dict):
    global _FUNDING_CTX
    _FUNDING_CTX = funding or {}



def clamp_leverage(lv, CONFIG):
    try:
        return min(int(lv), int(CONFIG.get("max_leverage", lv)))
    except Exception:
        return lv

def calculate_trade_economics(entry_price: float, tp_price: float, sl_price: float, side: str, trade_size: float, leverage: int) -> TradeEconomics:
    notional = trade_size * leverage
    qty = notional / entry_price
    taker = CONFIG["fees"]["taker_rate"]
    entry_fee = notional * taker
    exit_fee = notional * taker
    total_fees = entry_fee + exit_fee
    if side == "LONG":
        tp_pnl = (tp_price - entry_price) * qty
        sl_pnl = (sl_price - entry_price) * qty
    else:
        tp_pnl = (entry_price - tp_price) * qty
        sl_pnl = (entry_price - sl_price) * qty
    net_tp = tp_pnl - total_fees
    net_sl = sl_pnl - total_fees
    tp_roi = (net_tp / trade_size) * 100
    sl_roi = (net_sl / trade_size) * 100
    risk_reward = abs(net_tp) / abs(net_sl) if abs(net_sl) > 0 else 0
    breakeven_move = (total_fees / notional) * 100
    return TradeEconomics(total_fees=total_fees, net_tp=net_tp, net_sl=net_sl, tp_roi=tp_roi, sl_roi=sl_roi, risk_reward=risk_reward, breakeven_move=breakeven_move, is_profitable=net_tp > 0 and risk_reward > 1.0)

def sma(data: list[float], period: int) -> Optional[float]:
    if len(data) < period:
        return None
    return sum(data[-period:]) / period

def ema(data: list[float], period: int) -> Optional[float]:
    if len(data) < period:
        return None
    k = 2 / (period + 1)
    val = sma(data[:period], period)
    for i in range(period, len(data)):
        val = data[i] * k + val * (1 - k)
    return val

def rsi(closes: list[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    gains = 0.0
    losses = 0.0
    for i in range(len(closes) - period, len(closes)):
        diff = closes[i] - closes[i - 1]
        if diff > 0:
            gains += diff
        else:
            losses -= diff
    if losses == 0:
        return 100.0
    rs = gains / losses
    return 100.0 - (100.0 / (1.0 + rs))

def macd(closes: list[float]) -> dict:
    fast = ema(closes, 12)
    slow = ema(closes, 26)
    if fast is None or slow is None:
        return {"line": 0.0, "signal": 0.0, "histogram": 0.0}
    macd_history = []
    for i in range(30, len(closes) + 1):
        f = ema(closes[:i], 12)
        s = ema(closes[:i], 26)
        if f is not None and s is not None:
            macd_history.append(f - s)
    macd_line = fast - slow
    signal_line = ema(macd_history, 9) if len(macd_history) >= 9 else 0.0
    if signal_line is None:
        signal_line = 0.0
    return {"line": macd_line, "signal": signal_line, "histogram": macd_line - signal_line}

def bollinger_bands(closes: list[float], period: int = 20, mult: float = 2.0) -> Optional[dict]:
    if len(closes) < period:
        return None
    slc = closes[-period:]
    mean = sum(slc) / period
    variance = sum((v - mean) ** 2 for v in slc) / period
    std = math.sqrt(variance)
    upper = mean + mult * std
    lower = mean - mult * std
    bandwidth = (upper - lower) / mean if mean != 0 else 0
    return {"upper": upper, "middle": mean, "lower": lower, "std": std, "bandwidth": bandwidth}

def atr(candles: list[Candle], period: int = 14) -> float:
    if len(candles) < period + 1:
        return 0.0
    total = 0.0
    for i in range(len(candles) - period, len(candles)):
        tr = max(candles[i].high - candles[i].low, abs(candles[i].high - candles[i - 1].close), abs(candles[i].low - candles[i - 1].close))
        total += tr
    return total / period

# Adaptive parameter cache (per pair)
_ADAPT_CACHE = {}
_OI_CACHE = {}

def get_adaptive_params(pair: str, candles: list[Candle]):
    cfg = CONFIG.get("adaptive_params", {})
    if not cfg.get("enabled", False):
        return None
    now = time.time()
    cache = _ADAPT_CACHE.get(pair)
    if cache and (now - cache.get("ts", 0) < cfg.get("update_sec", 7200)):
        return cache.get("params")
    price = candles[-1].close if candles else 0
    atr_val = atr(candles, cfg.get("atr_period", 14)) if candles else 0
    atr_pct = (atr_val / price) if price else 0
    low = cfg.get("regime_atr_pct", {}).get("low", 0.0015)
    high = cfg.get("regime_atr_pct", {}).get("high", 0.0045)
    if atr_pct < low:
        regime = "low"
    elif atr_pct > high:
        regime = "high"
    else:
        regime = "mid"
    params = {
        "rsi_period": cfg.get("rsi_period", {}).get(regime, 14),
        "rsi_oversold": cfg.get("rsi_oversold", {}).get(regime, 30),
        "rsi_overbought": cfg.get("rsi_overbought", {}).get(regime, 70),
        "ema_fast": cfg.get("ema_fast", {}).get(regime, 20),
        "ema_slow": cfg.get("ema_slow", {}).get(regime, 50),
    }
    _ADAPT_CACHE[pair] = {"ts": now, "params": params}
    return params


def compute_market_divergence(market_data: dict[str, list[Candle]], lookback: int = 20):
    bull = 0; bear = 0
    for pair, candles in market_data.items():
        if len(candles) < lookback + 2: continue
        closes = [c.close for c in candles]
        recent = closes[-lookback:]
        p_low = min(recent); p_prev = min(recent[:-5]) if len(recent) > 5 else min(recent)
        r_recent = rsi(closes[-(lookback+1):], 14)
        r_prev = rsi(closes[-(lookback*2):-(lookback)], 14) if len(closes) > lookback*2 else r_recent
        if p_low < p_prev and r_recent > r_prev: bull += 1
        if p_low > p_prev and r_recent < r_prev: bear += 1
    if bull > bear: return 1, bull
    if bear > bull: return -1, bear
    return 0, max(bull, bear)


def compute_oi_context(pair: str, price: float, oi: float):
    if oi is None or oi == 0: return 0
    prev = _OI_CACHE.get(pair)
    _OI_CACHE[pair] = {"price": price, "oi": oi}
    if not prev: return 0
    dp = price - prev["price"]
    doi = oi - prev["oi"]
    if dp > 0 and doi > 0: return 1  # trend confirm long
    if dp < 0 and doi < 0: return 2  # liquidation fade long
    return 0


def structural_quality_boost(candles, side, cfg):
    if len(candles) < 30: return 0.0
    closes = [c.close for c in candles]
    price = closes[-1]
    vol = candles[-1].volume
    avg_vol = sum(c.volume for c in candles[-20:]) / 20
    vol_ratio = vol / avg_vol if avg_vol else 1.0
    # simple SR: recent high/low
    recent_high = max(c.high for c in candles[-30:])
    recent_low = min(c.low for c in candles[-30:])
    dist_high = abs(recent_high - price) / price
    dist_low = abs(price - recent_low) / price
    near_sr = (dist_high < cfg.get('sr_dist', 0.003)) or (dist_low < cfg.get('sr_dist', 0.003))
    boost = 0.0
    if vol_ratio >= cfg.get('vol_mult', 2.5):
        boost += cfg.get('boost', 0.04) * 0.6
    if near_sr:
        boost += cfg.get('boost', 0.04) * 0.4
    return boost


def stochastic(candles: list[Candle], k_period: int = 14) -> dict:
    if len(candles) < k_period:
        return {"k": 50.0, "d": 50.0}
    slc = candles[-k_period:]
    high = max(c.high for c in slc)
    low = min(c.low for c in slc)
    if high == low:
        return {"k": 50.0, "d": 50.0}
    k_val = ((candles[-1].close - low) / (high - low)) * 100
    k_values = []
    for i in range(max(0, len(candles) - 3), len(candles)):
        s = candles[max(0, i - k_period + 1): i + 1]
        h = max(c.high for c in s)
        l = min(c.low for c in s)
        k_values.append(50.0 if h == l else ((candles[i].close - l) / (h - l)) * 100)
    d_val = sum(k_values) / len(k_values) if k_values else 50.0
    return {"k": k_val, "d": d_val}

def vwap(candles: list[Candle], period: int = 20) -> Optional[float]:
    slc = candles[-period:]
    cum_tp_v = 0.0
    cum_v = 0.0
    for c in slc:
        tp = (c.high + c.low + c.close) / 3
        cum_tp_v += tp * c.volume
        cum_v += c.volume
    return cum_tp_v / cum_v if cum_v > 0 else None

def obv(candles: list[Candle]) -> float:
    val = 0.0
    for i in range(1, len(candles)):
        if candles[i].close > candles[i - 1].close:
            val += candles[i].volume
        elif candles[i].close < candles[i - 1].close:
            val -= candles[i].volume
    return val

def volume_spike(candles: list[Candle], lookback: int = 20) -> float:
    if len(candles) < lookback + 1:
        return 1.0
    vols = [c.volume for c in candles[-(lookback + 1):-1]]
    avg_vol = sma(vols, lookback)
    if avg_vol is None or avg_vol == 0:
        return 1.0
    return candles[-1].volume / avg_vol

def candle_body_ratio(c: Candle) -> float:
    rng = max(1e-9, c.high - c.low)
    body = abs(c.close - c.open)
    return body / rng


class BaseStrategy:
    id: str = ""
    name: str = ""
    timeframe: str = "1m"
    leverage: int = 10
    avg_signals_per_hour: float = 0.0
    def evaluate(self, candles: list[Candle]) -> Optional[Signal]:
        raise NotImplementedError

class EMAScalpStrategy(BaseStrategy):
    id = "ema_scalp"
    name = "EMA 3/8 Scalp Crossover"
    timeframe = "1m"
    leverage = 15
    avg_signals_per_hour = 1.5
    def evaluate(self, candles: list[Candle]) -> Optional[Signal]:
        if len(candles) < 30: return None
        closes = [c.close for c in candles]
        ema3_now = ema(closes, 3); ema8_now = ema(closes, 8)
        ema3_prev = ema(closes[:-1], 3); ema8_prev = ema(closes[:-1], 8)
        if any(v is None for v in [ema3_now, ema8_now, ema3_prev, ema8_prev]): return None
        rsi_val = rsi(closes, 7); vol_ratio = volume_spike(candles, 15)
        if ema3_prev <= ema8_prev and ema3_now > ema8_now and rsi_val < 68 and vol_ratio > 0.8:
            conf = 0.60 + min((68 - rsi_val) / 150, 0.15) + (0.05 if vol_ratio > 1.3 else 0)
            return Signal(side="LONG", confidence=conf, tp_percent=0.004, sl_percent=0.003, leverage=15, reason=f"EMA3 crossed above EMA8 | RSI(7)={rsi_val:.1f} | Vol={vol_ratio:.1f}x")
        if ema3_prev >= ema8_prev and ema3_now < ema8_now and rsi_val > 32 and vol_ratio > 0.8:
            conf = 0.60 + min((rsi_val - 32) / 150, 0.15) + (0.05 if vol_ratio > 1.3 else 0)
            return Signal(side="SHORT", confidence=conf, tp_percent=0.004, sl_percent=0.003, leverage=15, reason=f"EMA3 crossed below EMA8 | RSI(7)={rsi_val:.1f} | Vol={vol_ratio:.1f}x")
        return None

class RSISnapStrategy(BaseStrategy):
    id = "rsi_snap"; name = "RSI Snap Reversal"; timeframe = "1m"; leverage = 12; avg_signals_per_hour = 1.2
    def evaluate(self, candles: list[Candle]) -> Optional[Signal]:
        if len(candles) < 30: return None
        closes = [c.close for c in candles]
        params = get_adaptive_params("RSI_SNAP", candles) or {}
        rsi_p = int(params.get("rsi_period", 14))
        rsi_os = float(params.get("rsi_oversold", 30))
        rsi_ob = float(params.get("rsi_overbought", 70))
        rsi7 = rsi(closes, 7); rsiP = rsi(closes, rsi_p); vol_ratio = volume_spike(candles, 20)
        if rsi7 < (rsi_os - 10) and rsiP < rsi_os and vol_ratio > 1.3:
            conf = 0.62 + min((rsi_os - rsi7) / 80, 0.15) + min((vol_ratio - 1.3) / 5, 0.08)
            return Signal(side="LONG", confidence=conf, tp_percent=0.005, sl_percent=0.004, leverage=12, reason=f"RSI(7)={rsi7:.1f} oversold snap | RSI({rsi_p})={rsiP:.1f} | Vol={vol_ratio:.1f}x")
        if rsi7 > (rsi_ob + 10) and rsiP > rsi_ob and vol_ratio > 1.3:
            conf = 0.62 + min((rsi7 - rsi_ob) / 80, 0.15) + min((vol_ratio - 1.3) / 5, 0.08)
            return Signal(side="SHORT", confidence=conf, tp_percent=0.005, sl_percent=0.004, leverage=12, reason=f"RSI(7)={rsi7:.1f} overbought snap | RSI({rsi_p})={rsiP:.1f} | Vol={vol_ratio:.1f}x")
        return None

class BBSqueezeStrategy(BaseStrategy):
    id = "bb_squeeze"; name = "Bollinger Squeeze Breakout"; timeframe = "5m"; leverage = 10; avg_signals_per_hour = 0.5
    def evaluate(self, candles: list[Candle]) -> Optional[Signal]:
        if len(candles) < 40: return None
        closes = [c.close for c in candles]; price = closes[-1]
        bb = bollinger_bands(closes, 20, 2); bb_prev = bollinger_bands(closes[:-1], 20, 2)
        if bb is None or bb_prev is None: return None
        atr_val = atr(candles, 14); atr_pct = atr_val / price if price else 0
        was_tight = bb_prev["bandwidth"] < 0.012
        is_expanding = bb["bandwidth"] > bb_prev["bandwidth"] * 1.20
        # volume filter: current volume > 1.5x avg(20)
        vols = [c.volume for c in candles[-20:]] if len(candles) >= 20 else [c.volume for c in candles]
        avg_vol = sum(vols) / max(1, len(vols))
        vol_ok = candles[-1].volume > (avg_vol * 1.5) if avg_vol else False
        # trade only tight -> expanding with volume confirmation
        if not (was_tight and is_expanding and vol_ok):
            return None
        ema20 = ema(closes, 20)
        prev_close = closes[-2] if len(closes) >= 2 else price
        close_outside = (price > bb["upper"]) or (price < bb["lower"])
        pullback_long = prev_close > bb["upper"] and price >= min(bb["middle"], ema20)
        pullback_short = prev_close < bb["lower"] and price <= max(bb["middle"], ema20)
        bonus = 0.08
        if price > bb["upper"] or pullback_long:
            conf = 0.62 + bonus
            if CONFIG.get('structural_quality', {}).get('enabled', False):
                conf += structural_quality_boost(candles, 'LONG', CONFIG.get('structural_quality', {}))
            return Signal(side="LONG", confidence=conf, tp_percent=0.012, sl_percent=0.0065, leverage=clamp_leverage(10, CONFIG), reason=f"BB squeeze breakout UP | BW={bb['bandwidth'] * 100:.2f}% | ATR={atr_pct * 100:.3f}% | Vol>1.5x")
        if price < bb["lower"] or pullback_short:
            conf = 0.62 + bonus
            if CONFIG.get('structural_quality', {}).get('enabled', False):
                conf += structural_quality_boost(candles, 'SHORT', CONFIG.get('structural_quality', {}))
            return Signal(side="SHORT", confidence=conf, tp_percent=0.012, sl_percent=0.0065, leverage=clamp_leverage(10, CONFIG), reason=f"BB squeeze breakout DOWN | BW={bb['bandwidth'] * 100:.2f}% | ATR={atr_pct * 100:.3f}% | Vol>1.5x")
        return None

class MACDFlipStrategy(BaseStrategy):
    id = "macd_flip"; name = "MACD Histogram Flip"; timeframe = "1m"; leverage = 10; avg_signals_per_hour = 0.9
    def evaluate(self, candles: list[Candle]) -> Optional[Signal]:
        if len(candles) < 40: return None
        closes = [c.close for c in candles]; price = closes[-1]
        m = macd(closes); m_prev = macd(closes[:-1]); ema20 = ema(closes, 20)
        if ema20 is None: return None
        threshold = price * 0.0002
        if m_prev["histogram"] < 0 and m["histogram"] > threshold:
            trend_bonus = 0.06 if price > ema20 else 0
            conf = 0.57 + trend_bonus + min(abs(m["histogram"]) / (price * 0.001), 0.1)
            trend = "UP" if price > ema20 else "DOWN"
            return Signal(side="LONG", confidence=conf, tp_percent=0.006, sl_percent=0.004, leverage=clamp_leverage(10, CONFIG), reason=f"MACD histogram flipped bullish | H={m['histogram']:.4f} | Trend={trend}")
        if m_prev["histogram"] > 0 and m["histogram"] < -threshold:
            trend_bonus = 0.06 if price < ema20 else 0
            conf = 0.57 + trend_bonus + min(abs(m["histogram"]) / (price * 0.001), 0.1)
            trend = "DOWN" if price < ema20 else "UP"
            return Signal(side="SHORT", confidence=conf, tp_percent=0.006, sl_percent=0.004, leverage=clamp_leverage(10, CONFIG), reason=f"MACD histogram flipped bearish | H={m['histogram']:.4f} | Trend={trend}")
        return None



class FundingFadeStrategy(BaseStrategy):
    id = "funding_fade"; name = "Funding Fade Reversion"; timeframe = "1m"; leverage = 10; avg_signals_per_hour = 0.3
    def evaluate(self, candles: list[Candle]) -> Optional[Signal]:
        if len(candles) < 30: return None
        # use funding context
        pair = None
        # infer pair from last candle meta not available; will be set in run_signal_scan by wrapping eval (see below)
        return None

class LiquidationCascadeStrategy(BaseStrategy):
    id = "liquidation_cascade"; name = "Liquidation Cascade (Proxy)"; timeframe = "1m"; leverage = 10; avg_signals_per_hour = 0.4
    def evaluate(self, candles: list[Candle]) -> Optional[Signal]:
        if len(candles) < 30: return None
        closes = [c.close for c in candles]; price = closes[-1]
        atr_val = atr(candles, 14); atr_pct = atr_val / price if price else 0
        last = candles[-1]
        rng = (last.high - last.low) if last else 0
        vol_ratio = volume_spike(candles, 20)
        if atr_val == 0: return None
        impulse = rng / atr_val
        # strong impulse + volume spike = liquidation proxy
        if impulse > 2.0 and vol_ratio > 2.0:
            if last.close < last.open and (last.close - last.low) / max(1e-9, rng) < 0.3:
                conf = 0.60 + min((impulse - 2.0) * 0.05, 0.10)
                return Signal(side="SHORT", confidence=conf, tp_percent=0.006, sl_percent=0.004, leverage=clamp_leverage(10, CONFIG), reason=f"Liquidation cascade proxy DOWN | Impulse={impulse:.2f}x | Vol={vol_ratio:.1f}x")
            if last.close > last.open and (last.high - last.close) / max(1e-9, rng) < 0.3:
                conf = 0.60 + min((impulse - 2.0) * 0.05, 0.10)
                return Signal(side="LONG", confidence=conf, tp_percent=0.006, sl_percent=0.004, leverage=clamp_leverage(10, CONFIG), reason=f"Liquidation cascade proxy UP | Impulse={impulse:.2f}x | Vol={vol_ratio:.1f}x")
        return None
class VWAPBounceStrategy(BaseStrategy):
    id = "vwap_bounce"; name = "VWAP Bounce Scalp"; timeframe = "1m"; leverage = 10; avg_signals_per_hour = 0.8
    def evaluate(self, candles: list[Candle]) -> Optional[Signal]:
        if len(candles) < 40: return None
        closes = [c.close for c in candles]; price = closes[-1]
        vwap_val = vwap(candles, 30)
        vwap_prev = vwap(candles[:-1], 30)
        if vwap_val is None or vwap_prev is None: return None
        dist = (price - vwap_val) / vwap_val; rsi_val = rsi(closes, 14); prev = candles[-2]
        # filters: not too close, some volatility, volume confirmation
        min_dist = 0.006; max_dist = 0.02
        atr_val = atr(candles, 14); atr_pct = atr_val / price if price else 0
        vols = [c.volume for c in candles[-20:]] if len(candles) >= 20 else [c.volume for c in candles]
        avg_vol = sum(vols) / max(1, len(vols))
        vol_ok = candles[-1].volume > (avg_vol * 1.5) if avg_vol else False
        if not (min_dist <= abs(dist) <= max_dist):
            return None
        if atr_pct < 0.0025:
            return None
        if not vol_ok:
            return None
        vwap_rising = vwap_val > vwap_prev
        vwap_falling = vwap_val < vwap_prev
        if prev.low <= vwap_val * 1.001 and price > vwap_val and 42 < rsi_val < 65 and vwap_rising:
            conf = 0.60 + min((65 - rsi_val) / 200, 0.08)
            if CONFIG.get('structural_quality', {}).get('enabled', False):
                conf += structural_quality_boost(candles, 'LONG', CONFIG.get('structural_quality', {}))
            return Signal(side="LONG", confidence=conf, tp_percent=0.005, sl_percent=0.003, leverage=clamp_leverage(10, CONFIG), reason=f"VWAP bounce long | Dist={dist * 100:.3f}% | RSI={rsi_val:.1f} | ATR%={atr_pct*100:.3f}% | Vol>1.5x")
        if prev.high >= vwap_val * 0.999 and price < vwap_val and 35 < rsi_val < 58 and vwap_falling:
            conf = 0.60 + min((rsi_val - 35) / 200, 0.08)
            if CONFIG.get('structural_quality', {}).get('enabled', False):
                conf += structural_quality_boost(candles, 'SHORT', CONFIG.get('structural_quality', {}))
            return Signal(side="SHORT", confidence=conf, tp_percent=0.005, sl_percent=0.003, leverage=clamp_leverage(10, CONFIG), reason=f"VWAP rejection short | Dist={dist * 100:.3f}% | RSI={rsi_val:.1f} | ATR%={atr_pct*100:.3f}% | Vol>1.5x")
        return None

class StochCrossStrategy(BaseStrategy):
    id = "stoch_cross"; name = "Stochastic Zone Crossover"; timeframe = "1m"; leverage = 10; avg_signals_per_hour = 0.8
    def evaluate(self, candles: list[Candle]) -> Optional[Signal]:
        if len(candles) < 20: return None
        closes = [c.close for c in candles]; stoch_now = stochastic(candles, 14); stoch_prev = stochastic(candles[:-1], 14); rsi_val = rsi(closes, 14)
        if stoch_now["k"] < 25 and stoch_prev["k"] < stoch_prev["d"] and stoch_now["k"] > stoch_now["d"] and rsi_val < 40:
            conf = 0.59 + min((25 - stoch_now["k"]) / 100, 0.1)
            return Signal(side="LONG", confidence=conf, tp_percent=0.006, sl_percent=0.004, leverage=clamp_leverage(10, CONFIG), reason=f"Stoch bullish cross in OS | K={stoch_now['k']:.1f} D={stoch_now['d']:.1f} | RSI={rsi_val:.1f}")
        if stoch_now["k"] > 75 and stoch_prev["k"] > stoch_prev["d"] and stoch_now["k"] < stoch_now["d"] and rsi_val > 60:
            conf = 0.59 + min((stoch_now["k"] - 75) / 100, 0.1)
            return Signal(side="SHORT", confidence=conf, tp_percent=0.006, sl_percent=0.004, leverage=clamp_leverage(10, CONFIG), reason=f"Stoch bearish cross in OB | K={stoch_now['k']:.1f} D={stoch_now['d']:.1f} | RSI={rsi_val:.1f}")
        return None

class ATRBreakoutStrategy(BaseStrategy):
    id = "atr_breakout"; name = "ATR Volatility Breakout"; timeframe = "1m"; leverage = 8; avg_signals_per_hour = 0.6
    def evaluate(self, candles: list[Candle]) -> Optional[Signal]:
        if len(candles) < 30: return None
        closes = [c.close for c in candles]; price = closes[-1]
        atr_val = atr(candles, 14)
        if atr_val == 0: return None
        prev = candles[-2]; ema20 = ema(closes, 20)
        if ema20 is None: return None
        vol_ratio = volume_spike(candles, 15)
        if price > prev.high + atr_val * 0.5 and price > ema20 and vol_ratio > 1.0:
            conf = 0.60 + (0.08 if vol_ratio > 1.5 else 0) + (0.04 if price > ema20 else 0)
            return Signal(side="LONG", confidence=conf, tp_percent=0.010, sl_percent=0.005, leverage=8, reason=f"ATR breakout UP | Break={((price - prev.high) / atr_val):.2f}x ATR | Vol={vol_ratio:.1f}x")
        if price < prev.low - atr_val * 0.5 and price < ema20 and vol_ratio > 1.0:
            conf = 0.60 + (0.08 if vol_ratio > 1.5 else 0) + (0.04 if price < ema20 else 0)
            return Signal(side="SHORT", confidence=conf, tp_percent=0.010, sl_percent=0.005, leverage=8, reason=f"ATR breakout DOWN | Break={((prev.low - price) / atr_val):.2f}x ATR | Vol={vol_ratio:.1f}x")
        return None

class TripleEMAStrategy(BaseStrategy):
    id = "triple_ema"; name = "Triple EMA Ribbon Entry"; timeframe = "1m"; leverage = 12; avg_signals_per_hour = 0.8
    def evaluate(self, candles: list[Candle]) -> Optional[Signal]:
        if len(candles) < 30: return None
        closes = [c.close for c in candles]; ema5 = ema(closes, 5); ema13 = ema(closes, 13); ema21 = ema(closes, 21)
        if any(v is None for v in [ema5, ema13, ema21]): return None
        p_ema5 = ema(closes[:-1], 5); p_ema13 = ema(closes[:-1], 13)
        if p_ema5 is None or p_ema13 is None: return None
        rsi_val = rsi(closes, 14)
        bullish_now = ema5 > ema13 and ema13 > ema21; wasnt_bullish = p_ema5 <= p_ema13
        if bullish_now and wasnt_bullish and 45 < rsi_val < 72:
            conf = 0.61 + min((72 - rsi_val) / 200, 0.08)
            return Signal(side="LONG", confidence=conf, tp_percent=0.006, sl_percent=0.004, leverage=12, reason=f"Triple EMA aligned bullish | 5>{ema5:.2f} 13>{ema13:.2f} 21>{ema21:.2f}")
        bearish_now = ema5 < ema13 and ema13 < ema21; wasnt_bearish = p_ema5 >= p_ema13
        if bearish_now and wasnt_bearish and 28 < rsi_val < 55:
            conf = 0.61 + min((rsi_val - 28) / 200, 0.08)
            return Signal(side="SHORT", confidence=conf, tp_percent=0.006, sl_percent=0.004, leverage=12, reason=f"Triple EMA aligned bearish | 5<{ema5:.2f} 13<{ema13:.2f} 21<{ema21:.2f}")
        return None

class EngulfingSRStrategy(BaseStrategy):
    id = "engulfing_sr"; name = "Engulfing at Support/Resistance"; timeframe = "1m"; leverage = 10; avg_signals_per_hour = 0.6
    def evaluate(self, candles: list[Candle]) -> Optional[Signal]:
        if len(candles) < 40: return None
        last = candles[-1]; prev = candles[-2]
        bullish = (prev.close < prev.open and last.close > last.open and last.open <= prev.close and last.close >= prev.open)
        bearish = (prev.close > prev.open and last.close < last.open and last.open >= prev.close and last.close <= prev.open)
        if not bullish and not bearish: return None
        recent = candles[-40:]
        sorted_lows = sorted(c.low for c in recent); sorted_highs = sorted((c.high for c in recent), reverse=True)
        support = sorted_lows[3]; resistance = sorted_highs[3]
        near_support = abs(last.low - support) / support < 0.004 if support else False
        near_resistance = abs(last.high - resistance) / resistance < 0.004 if resistance else False
        if bullish and near_support:
            return Signal(side="LONG", confidence=0.63, tp_percent=0.006, sl_percent=0.003, leverage=clamp_leverage(10, CONFIG), reason=f"Bullish engulfing at support {support:.2f}")
        if bearish and near_resistance:
            return Signal(side="SHORT", confidence=0.63, tp_percent=0.006, sl_percent=0.003, leverage=clamp_leverage(10, CONFIG), reason=f"Bearish engulfing at resistance {resistance:.2f}")
        return None

class OBVDivergenceStrategy(BaseStrategy):
    id = "obv_divergence"; name = "OBV Divergence Reversal"; timeframe = "1m"; leverage = 8; avg_signals_per_hour = 0.5
    def evaluate(self, candles: list[Candle]) -> Optional[Signal]:
        if len(candles) < 40: return None
        closes = [c.close for c in candles]
        recent_candles = candles[-10:]; prior_candles = candles[-20:-10]
        recent_high = max(c.high for c in recent_candles); prior_high = max(c.high for c in prior_candles)
        recent_low = min(c.low for c in recent_candles); prior_low = min(c.low for c in prior_candles)
        obv_recent = obv(recent_candles); obv_prior = obv(prior_candles); rsi_val = rsi(closes, 14)
        if recent_high > prior_high and obv_recent < obv_prior and rsi_val > 55:
            conf = 0.60 + min((rsi_val - 55) / 200, 0.08)
            return Signal(side="SHORT", confidence=conf, tp_percent=0.007, sl_percent=0.004, leverage=8, reason=f"OBV bearish divergence | Price HH but OBV LH | RSI={rsi_val:.1f}")
        if recent_low < prior_low and obv_recent > obv_prior and rsi_val < 45:
            conf = 0.60 + min((45 - rsi_val) / 200, 0.08)
            return Signal(side="LONG", confidence=conf, tp_percent=0.007, sl_percent=0.004, leverage=8, reason=f"OBV bullish divergence | Price LL but OBV HL | RSI={rsi_val:.1f}")
        return None

ALL_STRATEGIES = [
    EMAScalpStrategy(), RSISnapStrategy(), MACDFlipStrategy(), VWAPBounceStrategy(), StochCrossStrategy(), ATRBreakoutStrategy(), TripleEMAStrategy(), EngulfingSRStrategy(), OBVDivergenceStrategy()
]



def load_2h_strategies():
    try:
        from new_strategies_2h import NEW_2H_STRATEGIES, NEW_CATEGORIES
    except Exception as e:
        logger.warning(f"Failed to load 2h strategies: {e}")
        return
    existing = {s.id for s in ALL_STRATEGIES}
    for s in NEW_2H_STRATEGIES:
        if s.id not in existing:
            ALL_STRATEGIES.append(s)
    CONFIG["strategy_categories"].update(NEW_CATEGORIES)

class CorrelationFilter:
    def __init__(self):
        self.pair_cooldowns: dict[str, float] = {}
        self.peak_balance: float = 0.0
        self.recent_signal_log: list[dict] = []
    def reset(self):
        self.pair_cooldowns.clear(); self.peak_balance = 0.0; self.recent_signal_log.clear()
    @staticmethod
    def get_strategy_category(strategy_id: str) -> str:
        for category, ids in CONFIG["strategy_categories"].items():
            if strategy_id in ids: return category
        return "unknown"
    def get_state(self) -> dict:
        now = time.time(); cooldown_sec = CONFIG["correlation"]["cooldown_sec"]; window_sec = CONFIG["correlation"]["same_side_window_sec"]
        return {
            "active_cooldowns": [{"pair": pair, "expires_in": cooldown_sec - (now - ts)} for pair, ts in self.pair_cooldowns.items() if now - ts < cooldown_sec],
            "recent_signal_count": sum(1 for s in self.recent_signal_log if now - s["timestamp"] < window_sec),
            "peak_balance": self.peak_balance,
        }
    @staticmethod
    def detect_btc_macro_move(market_data: dict[str, list[Candle]]) -> BTCMacroStatus:
        btc_candles = market_data.get("BTCUSDT"); lookback = CONFIG["correlation"]["btc_move_lookback"]
        if not btc_candles or len(btc_candles) < lookback + 1:
            return BTCMacroStatus(is_macro_move=False, direction=None, magnitude=0.0)
        recent = btc_candles[-lookback:]; price_now = recent[-1].close; price_then = recent[0].open
        move = (price_now - price_then) / price_then if price_then else 0
        return BTCMacroStatus(is_macro_move=abs(move) >= CONFIG["correlation"]["btc_move_threshold"], direction="LONG" if move > 0 else "SHORT", magnitude=abs(move), btc_price=price_now)
    def apply_same_side_flood_filter(self, signals: list[TradeSignal]) -> list[TradeSignal]:
        now = time.time(); window = CONFIG["correlation"]["same_side_window_sec"]; max_same = CONFIG["correlation"]["max_same_side_signals"]
        self.recent_signal_log = [s for s in self.recent_signal_log if now - s["timestamp"] < window]
        side_counts = {"LONG": 0, "SHORT": 0}
        for s in self.recent_signal_log:
            side_counts[s["side"]] = side_counts.get(s["side"], 0) + 1
        passed = []
        for sig in signals:
            if side_counts.get(sig.side, 0) >= max_same:
                sig._filtered = True; sig._filter_reason = f"Same-side flood: {side_counts[sig.side]} {sig.side}s already in window"; continue
            side_counts[sig.side] = side_counts.get(sig.side, 0) + 1; passed.append(sig)
        return passed
    def apply_category_filter(self, signals: list[TradeSignal], active_trades: list[ActiveTrade]) -> list[TradeSignal]:
        max_per_cat = CONFIG["correlation"]["max_per_category"]
        active_cat_counts: dict[str, int] = {}
        for trade in active_trades:
            cat = self.get_strategy_category(trade.strategy_id); active_cat_counts[cat] = active_cat_counts.get(cat, 0) + 1
        pending_cat_counts: dict[str, int] = {}; passed = []
        for sig in signals:
            cat = sig.strategy_category; total = active_cat_counts.get(cat, 0) + pending_cat_counts.get(cat, 0)
            if total >= max_per_cat:
                sig._filtered = True; sig._filter_reason = f"Category cap: {cat} has {total}/{max_per_cat} slots"; continue
            pending_cat_counts[cat] = pending_cat_counts.get(cat, 0) + 1; passed.append(sig)
        return passed
    def apply_cooldown_filter(self, signals: list[TradeSignal]) -> list[TradeSignal]:
        now = time.time(); cooldown = CONFIG["correlation"]["cooldown_sec"]; passed = []
        for sig in signals:
            last = self.pair_cooldowns.get(sig.pair)
            if last and now - last < cooldown:
                sig._filtered = True; sig._filter_reason = f"Cooldown: {sig.pair} signaled {now - last:.0f}s ago"; continue
            passed.append(sig)
        return passed
    def check_drawdown_breaker(self, balance: float) -> DrawdownStatus:
        if balance > self.peak_balance: self.peak_balance = balance
        peak = self.peak_balance
        if peak == 0: return DrawdownStatus(paused=False, drawdown=0.0)
        drawdown = (peak - balance) / peak; threshold = CONFIG["correlation"]["max_drawdown_pause"]
        paused = drawdown >= threshold
        return DrawdownStatus(paused=paused, drawdown=drawdown, peak=peak, threshold=threshold, message=(f"CIRCUIT BREAKER: {drawdown * 100:.2f}% drawdown from peak ${peak:.2f}. New entries paused." if paused else None))
    def record_signals(self, signals: list[TradeSignal]):
        now = time.time()
        for sig in signals:
            self.pair_cooldowns[sig.pair] = now
            self.recent_signal_log.append({"side": sig.side, "timestamp": now})
    def set_pair_cooldown(self, pair: str, ts: float | None = None):
        self.pair_cooldowns[pair] = ts or time.time()

correlation_filter = CorrelationFilter()

def set_pair_cooldown(pair: str, ts: float | None = None):
    correlation_filter.set_pair_cooldown(pair, ts)

def run_signal_scan(market_data: dict[str, list[Candle]], active_trades: list[ActiveTrade], balance: float, funding: dict[str, float] | None = None, open_interest: dict[str, float] | None = None, spread_map: dict[str, tuple[float,float]] | None = None, strategies: list[BaseStrategy] | None = None, news_bias: dict[str, float] | None = None, market_data_5m: dict[str, list[Candle]] | None = None, market_data_15m: dict[str, list[Candle]] | None = None) -> ScanResult:
    if strategies is None: strategies = ALL_STRATEGIES
    result = ScanResult(); cfg = CONFIG
    if open_interest is None:
        open_interest = {}
    if spread_map is None:
        spread_map = {}
    if funding is None:
        funding = {}
    if news_bias is None:
        news_bias = {}
    if market_data_5m is None:
        market_data_5m = {}
    if market_data_15m is None:
        market_data_15m = {}

    drawdown_check = correlation_filter.check_drawdown_breaker(balance); result.drawdown = drawdown_check
    if drawdown_check.paused:
        result.diagnostics = ScanDiagnostics(raw_count=0, final=0, reason="CIRCUIT_BREAKER"); return result
    btc_macro = CorrelationFilter.detect_btc_macro_move(market_data); result.btc_macro = btc_macro
    open_pairs = {t.pair for t in active_trades}
    slots_available = cfg["max_concurrent_trades"] - len(active_trades)
    max_1m = cfg.get("max_1m_trades", cfg["max_concurrent_trades"])
    max_2h = cfg.get("max_2h_trades", cfg["max_concurrent_trades"])
    active_1m = sum(1 for t in active_trades if not is_2h_strategy(t.strategy_id))
    active_2h = sum(1 for t in active_trades if is_2h_strategy(t.strategy_id))
    if slots_available <= 0:
        result.diagnostics = ScanDiagnostics(raw_count=0, final=0, reason="ALL_SLOTS_FULL"); return result
    raw_signals: list[TradeSignal] = []
    base_conf_th = cfg.get("confidence_threshold", 0.55)
    base_confirm = cfg.get("confirm_signal", False)
    base_min_vol = cfg.get("min_volatility_pct", 0.0)
    ema_fast_n = cfg.get("trend_ema_fast", 0)
    ema_slow_n = cfg.get("trend_ema_slow", 0)
    regime_min_trend = cfg.get("regime_min_trend_pct", 0.0)
    atr_period = cfg.get("atr_period", 14)
    vol_target = cfg.get("vol_target_pct", 0.0)
    max_funding_long = cfg.get("max_funding_long", 0.0)
    max_funding_short = cfg.get("max_funding_short", 0.0)
    for pair in cfg["pairs"]:
        candles = market_data.get(pair)
        if not candles or len(candles) < 50: continue
        candles_5m = market_data_5m.get(pair)
        candles_15m = market_data_15m.get(pair)
        if pair in open_pairs: continue
        if btc_macro.is_macro_move and pair != "BTCUSDT": continue
        price = candles[-1].close
        # Trend filter (EMA)
        closes = [c.close for c in candles]
        ap = get_adaptive_params(pair, candles) if cfg.get("adaptive_params", {}).get("enabled", False) else None
        af = ap.get("ema_fast") if ap else ema_fast_n
        aslow = ap.get("ema_slow") if ap else ema_slow_n
        ema_fast = ema(closes, af) if af else None
        ema_slow = ema(closes, aslow) if aslow else None
        # Regime filter: if trend strength small, only allow reversion/structural
        trend_strength = 0.0
        if ema_fast is not None and ema_slow is not None and price:
            trend_strength = abs(ema_fast - ema_slow) / price
        for strategy in strategies:
            is2h = is_2h_strategy(strategy.id)
            # bb_squeeze uses 5m candles and 15m EMA50/100 trend filter
            use_candles = candles_5m if strategy.id == 'bb_squeeze' and candles_5m else candles
            if is2h and active_2h >= max_2h: continue
            if (not is2h) and active_1m >= max_1m: continue
            s_cfg = CONFIG_2H if is2h else {}
            conf_th = s_cfg.get("confidence_threshold", base_conf_th)
            confirm = s_cfg.get("confirm_signal", base_confirm)
            min_vol = s_cfg.get("min_volatility_pct", base_min_vol)
            max_funding_long = s_cfg.get("max_funding_long", cfg.get("max_funding_long", 0.0))
            max_funding_short = s_cfg.get("max_funding_short", cfg.get("max_funding_short", 0.0))
            skip_trend = s_cfg.get("skip_trend_ema_filter", False)
            skip_regime = s_cfg.get("skip_regime_filter", False)
            price_used = use_candles[-1].close if use_candles else price
            # volatility filter (tier-specific)
            if min_vol and len(use_candles) >= 20:
                hi = max(c.high for c in use_candles[-20:])
                lo = min(c.low for c in use_candles[-20:])
                vol_pct = (hi - lo) / price_used if price_used else 0
                if vol_pct < min_vol:
                    continue
            if btc_macro.is_macro_move and pair == "BTCUSDT":
                cat = CorrelationFilter.get_strategy_category(strategy.id)
                if cat != "reversion": continue
            if hasattr(strategy, "needs_funding") and strategy.needs_funding:
                eval_result = strategy.evaluate(use_candles, funding_rate=funding.get(pair, 0.0))
            else:
                eval_result = strategy.evaluate(use_candles)
            if eval_result is None: continue
            # pair-specific strategy allowlist
            pf = cfg.get("pair_filters", {})
            if pair in pf and strategy.id not in pf[pair]:
                continue
            bias = news_bias.get(pair, 0.0)
            adj_conf = max(0.0, min(1.0, eval_result.confidence + bias))
            if adj_conf < conf_th: continue
            if confirm and not is2h:
                prev_eval = strategy.evaluate(use_candles[:-1]) if len(use_candles) > 51 else None
                if not prev_eval or prev_eval.side != eval_result.side or prev_eval.confidence < conf_th:
                    continue
            # Regime filter
            if (not skip_regime) and regime_min_trend and trend_strength < regime_min_trend:
                if strategy.id in CONFIG["strategy_categories"].get("trend", []):
                    continue
            # Trend alignment (override for bb_squeeze using 15m EMA50/100)
            if strategy.id == 'bb_squeeze':
                if not candles_15m or len(candles_15m) < 120:
                    continue
                closes_15m = [c.close for c in candles_15m]
                ema50 = ema(closes_15m, 50)
                ema100 = ema(closes_15m, 100)
                if ema50 is None or ema100 is None:
                    continue
                price_15m = closes_15m[-1]
                if eval_result.side == "LONG" and not (price_15m > ema50 and ema50 > ema100):
                    continue
                if eval_result.side == "SHORT" and not (price_15m < ema50 and ema50 < ema100):
                    continue
            else:
                # Trend alignment (default)
                if (not skip_trend) and ema_fast is not None and ema_slow is not None:
                    if eval_result.side == "LONG" and not (price > ema_fast and ema_fast > ema_slow):
                        continue
                    if eval_result.side == "SHORT" and not (price < ema_fast and ema_fast < ema_slow):
                        continue
            # Funding filter
            if funding:
                fr = funding.get(pair, 0.0)
                if eval_result.side == "LONG" and max_funding_long and fr > max_funding_long:
                    continue
                if eval_result.side == "SHORT" and max_funding_short and fr < -max_funding_short:
                    continue
            # Volatility-targeted sizing (ATR)
            atr_val = atr(use_candles, atr_period) if atr_period else 0.0
            atr_pct = (atr_val / price_used) if price_used else 0.0
            base_size = max(cfg["min_trade_size"], min(balance * cfg["risk_per_trade"], balance / cfg["max_concurrent_trades"]))
            if vol_target and atr_pct > 0:
                scale = vol_target / atr_pct
                trade_size = max(cfg["min_trade_size"], min(base_size * scale, base_size * 2))
            else:
                trade_size = base_size
            if trade_size < cfg["min_trade_size"] or balance < trade_size: continue
            tp_mult = s_cfg.get("tp_multiplier", cfg.get("tp_multiplier", 1.0))
            min_tp = s_cfg.get("min_tp_percent", cfg.get("min_tp_percent", 0.0))
            min_sl = s_cfg.get("min_sl_percent", cfg.get("min_sl_percent", 0.0))
            tp_pct = max(eval_result.tp_percent * tp_mult, min_tp)
            tp_price = price_used * (1 + tp_pct) if eval_result.side == "LONG" else price_used * (1 - tp_pct)
            sl_pct = max(eval_result.sl_percent, min_sl)
            sl_price = price_used * (1 - sl_pct) if eval_result.side == "LONG" else price_used * (1 + sl_pct)
            economics = calculate_trade_economics(price_used, tp_price, sl_price, eval_result.side, trade_size, clamp_leverage(eval_result.leverage, CONFIG))
            if not economics.is_profitable: continue
            # min risk-reward guard
            min_rr = s_cfg.get("min_risk_reward", cfg.get("min_risk_reward", None))
            if min_rr is not None and economics.risk_reward < min_rr:
                continue
            reason = eval_result.reason
            if bias:
                reason = f"{reason} | news_bias={bias:+.3f}"
            raw_signals.append(TradeSignal(pair=pair, strategy_id=strategy.id, strategy_name=strategy.name, strategy_category=CorrelationFilter.get_strategy_category(strategy.id), side=eval_result.side, confidence=adj_conf, entry_price=price, tp_price=tp_price, sl_price=sl_price, leverage=clamp_leverage(eval_result.leverage, CONFIG), trade_size=trade_size, reason=reason, economics=economics))
    # apply market divergence boost (confidence only)
    div_cfg = cfg.get("divergence_boost", {})
    div_dir, div_count = (0,0)
    if div_cfg.get("enabled", False):
        div_dir, div_count = compute_market_divergence(market_data, div_cfg.get("lookback", 20))
    if div_dir != 0 and div_count >= div_cfg.get("pairs_min", 3):
        boost = div_cfg.get("boost", 0.04)
        for s in raw_signals:
            if (div_dir == 1 and s.side == "LONG") or (div_dir == -1 and s.side == "SHORT"):
                s.confidence += boost
    # apply OI context boost (confidence only)
    oi_cfg = cfg.get("oi_boost", {})
    if oi_cfg.get("enabled", False) and open_interest:
        for s in raw_signals:
            oi = open_interest.get(s.pair, 0.0)
            ctx = compute_oi_context(s.pair, s.entry_price, oi)
            if ctx in (1,2) and s.side == "LONG":
                s.confidence += oi_cfg.get("boost", 0.06)
    # apply agreement boost (same pair + same side)
    ab = cfg.get("agreement_boost", {})
    if ab.get("enabled", False):
        from collections import defaultdict
        counts = defaultdict(int)
        for s in raw_signals:
            counts[(s.pair, s.side)] += 1
        for s in raw_signals:
            if counts[(s.pair, s.side)] >= ab.get("min_strategies", 2):
                s.confidence += ab.get("boost", 0.05)
    raw_signals.sort(key=lambda s: s.confidence, reverse=True)



    seen_pairs: set[str] = set(); deduped: list[TradeSignal] = []
    for sig in raw_signals:
        if sig.pair in seen_pairs: continue
        seen_pairs.add(sig.pair); deduped.append(sig)
    diag = ScanDiagnostics(raw_count=len(deduped))
    after_cooldown = apply_cooldown_tiered(deduped); diag.after_cooldown = len(after_cooldown)
    after_flood = correlation_filter.apply_same_side_flood_filter(after_cooldown); diag.after_flood = len(after_flood)
    after_category = correlation_filter.apply_category_filter(after_flood, active_trades); diag.after_category = len(after_category)
    final_signals = after_category[:slots_available]; diag.final = len(final_signals)
    correlation_filter.record_signals(final_signals)
    all_filtered = [s for s in deduped + after_cooldown + after_flood + after_category if s._filtered]
    result.signals = final_signals; result.filtered = all_filtered; result.diagnostics = diag
    return result



def apply_cooldown_tiered(signals: list[TradeSignal]) -> list[TradeSignal]:
    now = time.time()
    passed = []
    for sig in signals:
        if is_4h_strategy(sig.strategy_id):
            cooldown = CONFIG_4H.get("cooldown_sec", CONFIG["correlation"]["cooldown_sec"])
        elif is_2h_strategy(sig.strategy_id):
            cooldown = CONFIG_2H.get("cooldown_sec", CONFIG["correlation"]["cooldown_sec"])
        else:
            cooldown = CONFIG["correlation"]["cooldown_sec"]
        last = correlation_filter.pair_cooldowns.get(sig.pair)
        if last and now - last < cooldown:
            sig._filtered = True; sig._filter_reason = f"Cooldown: {sig.pair} {now - last:.0f}s/{cooldown}s"; continue
        passed.append(sig)
    return passed

def get_correlation_state() -> dict:
    return correlation_filter.get_state()

def reset_correlation_state():
    correlation_filter.reset()

if __name__ == "__main__":
    print("Multi-Strategy Crypto Futures Engine loaded.")
    print(f"Strategies: {len(ALL_STRATEGIES)}"); print(f"Pairs: {len(CONFIG['pairs'])}"); print(f"Target: ~8.7 signals/hour"); print(f"Max concurrent: {CONFIG['max_concurrent_trades']}"); print(f"Min trade: ${CONFIG['min_trade_size']}")
    for s in ALL_STRATEGIES:
        print(f" {s.id:<20} {s.name:<35} ~{s.avg_signals_per_hour}/hr Lev {s.leverage}x")
