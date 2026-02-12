#!/usr/bin/env python3
"""
BUG-002 FIX: Patch new_strategies_4h.py to log WARNING (not debug) when
compute_12h_bias() falls back to NEUTRAL due to insufficient 1h candles.

This was a critical silent failure — the 12H directional filter was returning
NEUTRAL (allowing all directions) instead of actively filtering, because
insufficient 1h candles were being fetched. The debug-level log was invisible
in production logs.

Run: python3 patch_new_strategies_4h.py
"""

import os, sys

TARGET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "new_strategies_4h.py")
if not os.path.exists(TARGET):
    TARGET = "/opt/multi-strat-engine/new_strategies_4h.py"

if not os.path.exists(TARGET):
    print(f"ERROR: Cannot find new_strategies_4h.py")
    sys.exit(1)

with open(TARGET, 'r') as f:
    content = f.read()

changes = 0

# ── PATCH 1: Change debug → warning for insufficient candles in compute_12h_bias ──
OLD_LOG = '''    if len(candles_1h) < 360:
        logger.debug(f"[12H filter] {pair}: only {len(candles_1h)} 1h candles, need 360+")
        return neutral'''

NEW_LOG = '''    if len(candles_1h) < 360:
        logger.warning(f"[12H filter] {pair}: only {len(candles_1h)} 1h candles (need 360+) — falling back to NEUTRAL bias. Ensure 1h candle fetch limit >= 500.")
        return neutral'''

if OLD_LOG in content:
    content = content.replace(OLD_LOG, NEW_LOG)
    changes += 1
    print("✓ Patched compute_12h_bias() insufficient candles: debug → warning")
else:
    print("⚠ compute_12h_bias() log already patched or not found")

# ── PATCH 2: Add warning for insufficient 12H bars after aggregation ──
OLD_12H_CHECK = '''    candles_12h = aggregate_candles(candles_1h, 12)
    if len(candles_12h) < 32:
        return neutral'''

NEW_12H_CHECK = '''    candles_12h = aggregate_candles(candles_1h, 12)
    if len(candles_12h) < 32:
        logger.warning(f"[12H filter] {pair}: only {len(candles_12h)} 12H bars after aggregation (need 32+) — NEUTRAL bias")
        return neutral'''

if OLD_12H_CHECK in content:
    content = content.replace(OLD_12H_CHECK, NEW_12H_CHECK)
    changes += 1
    print("✓ Patched 12H bar count check: added warning log")
else:
    print("⚠ 12H bar check already patched or not found")

with open(TARGET, 'w') as f:
    f.write(content)

print(f"\nDone: {changes}/2 patches applied to {TARGET}")
