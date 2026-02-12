#!/usr/bin/env python3
"""
BUG-012 FIX: Patch new_strategies_2h.py to export all 13 strategies.
Run: python3 patch_new_strategies_2h.py

Previously, 5 fully-implemented strategies were dead code because they
weren't in the NEW_2H_STRATEGIES list, NEW_CATEGORIES, or NEW_MAX_AGE:
  - ADXDICrossStrategy (adx_di_cross)
  - FibonacciPullbackStrategy (fib_pullback)  
  - CMFDivergenceStrategy (cmf_divergence)
  - VWAPBandReversionStrategy (vwap_sd_reversion)
  - MultiTFEMARibbonStrategy (mtf_ema_ribbon)
"""

import os, sys

TARGET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "new_strategies_2h.py")
if not os.path.exists(TARGET):
    # Try the engine directory
    TARGET = "/opt/multi-strat-engine/new_strategies_2h.py"

if not os.path.exists(TARGET):
    print(f"ERROR: Cannot find new_strategies_2h.py")
    sys.exit(1)

with open(TARGET, 'r') as f:
    content = f.read()

changes = 0

# ── PATCH 1: NEW_2H_STRATEGIES list ──
OLD_LIST = """NEW_2H_STRATEGIES = [
    IchimokuCloudStrategy(),        # S1:  Trend breakout (5m)
    KeltnerReversionStrategy(),     # S2:  Mean reversion in ranges (5m)
    DonchianBreakoutStrategy(),     # S3:  Turtle breakout (5m)
    SupertrendFlipStrategy(),       # S4:  Trend flip (5m)
    VolumeProfilePOCStrategy(),     # S8:  POC reversion (15m)
    PivotPointBounceStrategy(),     # S9:  Key level reaction (5m)
    FundingRateFadeStrategy(),      # S12: Fade overleveraged crowd (needs funding data)
    BBKeltnerSqueezeStrategy(),     # S13: Volatility squeeze explosion (5m)
]"""

NEW_LIST = """# ── BUG-012 FIX: All 13 strategies now exported (was 8, 5 were dead code) ──
NEW_2H_STRATEGIES = [
    IchimokuCloudStrategy(),        # S1:  Trend breakout (5m)
    KeltnerReversionStrategy(),     # S2:  Mean reversion in ranges (5m)
    DonchianBreakoutStrategy(),     # S3:  Turtle breakout (5m)
    SupertrendFlipStrategy(),       # S4:  Trend flip (5m)
    ADXDICrossStrategy(),           # S5:  Trend start DI cross (5m)
    FibonacciPullbackStrategy(),    # S6:  Fibonacci pullback (5m)
    CMFDivergenceStrategy(),        # S7:  Smart money divergence (5m)
    VolumeProfilePOCStrategy(),     # S8:  POC reversion (15m)
    PivotPointBounceStrategy(),     # S9:  Key level reaction (5m)
    VWAPBandReversionStrategy(),    # S10: VWAP SD reversion (5m)
    MultiTFEMARibbonStrategy(),     # S11: Multi-TF EMA ribbon (5m+15m)
    FundingRateFadeStrategy(),      # S12: Fade overleveraged crowd (needs funding data)
    BBKeltnerSqueezeStrategy(),     # S13: Volatility squeeze explosion (5m)
]"""

if OLD_LIST in content:
    content = content.replace(OLD_LIST, NEW_LIST)
    changes += 1
    print("✓ Patched NEW_2H_STRATEGIES (8 → 13 strategies)")
else:
    print("⚠ NEW_2H_STRATEGIES already patched or not found")

# ── PATCH 2: NEW_CATEGORIES dict ──
OLD_CAT = """NEW_CATEGORIES = {
    "trend_2h": [
        "ichimoku_cloud", "donchian_breakout", "supertrend_flip",
    ],
    "reversion_2h": [
        "keltner_reversion", "vp_poc_reversion", "pivot_bounce",
    ],
    "structural_2h": [
        "bb_kc_squeeze", "funding_fade",
    ],
}"""

NEW_CAT = """# ── BUG-012 FIX: Categories now include all 13 strategy IDs ──
NEW_CATEGORIES = {
    "trend_2h": [
        "ichimoku_cloud", "donchian_breakout", "supertrend_flip",
        "adx_di_cross", "mtf_ema_ribbon",
    ],
    "reversion_2h": [
        "keltner_reversion", "vp_poc_reversion", "pivot_bounce",
        "fib_pullback", "cmf_divergence", "vwap_sd_reversion",
    ],
    "structural_2h": [
        "bb_kc_squeeze", "funding_fade",
    ],
}"""

if OLD_CAT in content:
    content = content.replace(OLD_CAT, NEW_CAT)
    changes += 1
    print("✓ Patched NEW_CATEGORIES (added 5 missing IDs)")
else:
    print("⚠ NEW_CATEGORIES already patched or not found")

# ── PATCH 3: NEW_MAX_AGE dict ──
OLD_AGE = """NEW_MAX_AGE = {
    # These trades hold 1-3 hours, so max_age should be ~4 hours
    "ichimoku_cloud":    14400,  # 4h
    "keltner_reversion": 10800,  # 3h
    "donchian_breakout": 14400,  # 4h
    "supertrend_flip":   10800,  # 3h
    "vp_poc_reversion":  10800,  # 3h
    "pivot_bounce":      7200,   # 2h
    "funding_fade":      14400,  # 4h
    "bb_kc_squeeze":     14400,  # 4h
}"""

NEW_AGE = """# ── BUG-012 FIX: Max ages for all 13 strategies ──
NEW_MAX_AGE = {
    # These trades hold 1-3 hours, so max_age should be ~4 hours
    "ichimoku_cloud":    14400,  # 4h
    "keltner_reversion": 10800,  # 3h
    "donchian_breakout": 14400,  # 4h
    "supertrend_flip":   10800,  # 3h
    "adx_di_cross":      10800,  # 3h
    "fib_pullback":      10800,  # 3h
    "cmf_divergence":     9000,  # 2.5h
    "vp_poc_reversion":  10800,  # 3h
    "pivot_bounce":      7200,   # 2h
    "vwap_sd_reversion":  7200,  # 2h
    "mtf_ema_ribbon":    10800,  # 3h
    "funding_fade":      14400,  # 4h
    "bb_kc_squeeze":     14400,  # 4h
}"""

if OLD_AGE in content:
    content = content.replace(OLD_AGE, NEW_AGE)
    changes += 1
    print("✓ Patched NEW_MAX_AGE (added 5 missing entries)")
else:
    print("⚠ NEW_MAX_AGE already patched or not found")

# Write back
with open(TARGET, 'w') as f:
    f.write(content)

print(f"\nDone: {changes}/3 patches applied to {TARGET}")
