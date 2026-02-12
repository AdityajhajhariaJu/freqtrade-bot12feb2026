# Strategy Activation / Deactivation Criteria

## 1) Criteria for activating strategies

### Core market-state metrics
- **ATR volatility (1m)**: `atr_pct`
- **ATR volatility (15m baseline)**: `atr_pct_15m`
- **Volume engagement ratio**: `volume_ratio = latest_vol / avg_vol_20`
- **Trend strength**: EMA20 vs EMA50 distance on 15m (`trend_strength`)
- **One-sided market detector**: directional persistence over last 12x 15m closes
- **Volatility expansion**: short-term ATR meaningfully above medium-term baseline

### Activation rules (moderate, not over-strict)
- **Trend strategies**
  - Activate when: `one_sided = true` **or** `trend_strength >= 0.25%`
  - And `volume_ratio >= 0.8`
- **Breakout strategies**
  - Activate when: `volatility_expanding = true`
  - And `volume_ratio >= 1.05`
- **Reversion strategies**
  - Activate when: market is not strongly one-sided
  - `volume_ratio >= 0.7`
  - `atr_pct <= 3.0%`
- **Event/momentum strategies**
  - Activate when: `volume_ratio >= 1.0` and `atr_pct >= 0.15%`

## 2) Criteria for deactivating strategies

### Global deactivation guards
- **Low engagement**: `volume_ratio < 0.35`
- **Dead market**: `atr_pct < 0.08%`
- **Extreme volatility chaos**: `atr_pct > 6.0%`

### Category-specific deactivation
- **Trend**: disabled in weak/flat trend conditions
- **Breakout**: disabled when no vol expansion or weak participation
- **Reversion**: disabled in strongly one-sided trend and high-vol impulse
- **Event/momentum**: disabled when no impulse (low vol + low activity)

## 3) Strategies included and intended conditions

### Trend-oriented
- `mtf_ema_ribbon`
- `ema_cross_rsi`
- `macd_hist_flip`
- `adx_di_cross`

### Breakout-oriented
- `bb_squeeze`
- `bb_kc_squeeze`
- `atr_breakout`
- `donchian_breakout`

### Reversion-oriented
- `rsi_snapback`
- `keltner_reversion`
- `cmf_divergence`
- `vwap_bounce`

### Event / momentum
- `liquidation_cascade`

## 4) Summary of potential profits vs losses (under new criteria)

### Potential profit improvements
- Better regime fit for strategy type (trend in trend, reversion in balanced markets)
- Higher quality breakout participation (with volume + expansion confirmation)
- Reduced false entries in low-engagement periods

### Potential loss reduction
- Lower overtrading in dead/choppy zones
- Fewer mean-reversion attempts during one-way moves
- Fewer low-conviction breakout attempts without participation

### Practical caution
- Criteria are intentionally **moderate** to avoid over-filtering.
- If signal count drops too much, first relax:
  - breakout volume ratio (`1.05 -> 1.0`)
  - trend strength (`0.25% -> 0.20%`)
  - reversion volume floor (`0.7 -> 0.6`)
