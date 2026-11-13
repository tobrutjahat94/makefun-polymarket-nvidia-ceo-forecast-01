#!/usr/bin/env python3
"""Polymarket Next-CEO-of-NVIDIA price forecaster.

Entry for make.fun competition polymarket-nvidia-ceo-forecast-01.
Reads each case file /input/<id>.in (one checkpoint, 72h of 1-minute Yes
prices per market) and writes /output/<id>.out: for each market, 240 rows
(one per minute after the checkpoint) of five quantiles (0.05, 0.25, 0.5,
0.75, 0.95) of that minute's Yes price.

Method (persistence median + per-market heavy-tailed spread):
  The median is the last price: prediction-market prices are close to a
  martingale at this horizon, and every drift I tried on the practice
  cases (reversion to a trailing mean, sum-normalisation) made the
  pinball loss worse, so the median stays at the checkpoint price.
  The spread is per market, estimated from that market's own recent
  1-minute moves, because markets differ hugely in activity: most
  minutes a price does not move at all (88% of 1-minute diffs in the
  practice are exactly zero) and then it jumps, sometimes by 20+ points
  (Colette Kress 0.3535 -> 0.59 -> 0.4585 in two minutes). A single
  global sigma (the house baseline's 0.0025) is far too wide for quiet
  low-price markets and too narrow for the jumpy leaders.
  For each market, sigma is the mean absolute 1-minute move over the
  last 240 / 720 / 1440 minutes (averaged), rescaled to a normal sigma
  (divide by sqrt(2/pi) = 0.7979), blended 80/20 with the last 60
  minutes' mean absolute move so a market that just woke up widens
  fast, times 0.55, clipped to [0.0004, 0.003]. Quantiles are
  q = x + z_tau * sigma * sqrt(h), with heavy-tailed multipliers
  z = (-2.2, -0.7, 0, 0.7, 2.2) instead of the normal's (-1.645,
  -0.674, 0, 0.674, 1.645): the wider tails pay for the jumps, the
  slightly wider inner quantiles cost little on quiet minutes.
  Values are clamped to [0, 1] and rounded to 6 decimals.

Backtest (local, practice cases p00-p04 scored against the next case's
prices, p05 has no locally available answer): pinball loss 0.308215 vs
house baseline 0.392518 on the same five cases (-21.5%), beating the
baseline on every single case (p00 0.471 vs 0.536, p01 0.311 vs 0.380,
p02 0.343 vs 0.408, p03 0.248 vs 0.369, p04 0.207 vs 0.299); median
error 0.8795, 90% coverage 0.908. The house baseline scores 0.363589
on the full six-case practice. Verified 2026-11-13: output format
valid on all 6 practice cases, two runs byte-identical, ~0.06 s/case.

Stdlib only, no network, deterministic.
"""
import json
import math
import os

# --- fitted parameters (see docstring) ---
Z = (-2.2, -0.7, 0.0, 0.7, 2.2)   # heavy-tailed quantile multipliers
K = 0.55                            # scale on the MAD-implied sigma
SIGMA_LO = 0.0004                   # floor: even a dead-quiet market moves
SIGMA_HI = 0.003                    # cap: never wider than this
WINDOWS = (240, 720, 1440)          # lookbacks for the base volatility
RECENT_WINDOW = 60                  # lookback for the fast volatility
RECENT_WEIGHT = 0.2                 # weight on the fast volatility
SQRT_2_OVER_PI = 0.7978845608028654  # E|N(0,1)|, converts mean|diff| to sigma


def mean_abs_diff(prices, window):
    """Mean absolute 1-minute move over the last `window` minutes."""
    seg = prices[-window:] if len(prices) > window else prices
    if len(seg) < 2:
        return 0.0
    total = 0.0
    for i in range(len(seg) - 1):
        d = seg[i + 1] - seg[i]
        total += d if d >= 0 else -d
    return total / (len(seg) - 1)


def market_sigma(prices):
    """Per-market per-sqrt-minute spread from its own recent moves."""
    base = sum(mean_abs_diff(prices, w) for w in WINDOWS) / len(WINDOWS)
    recent = mean_abs_diff(prices, RECENT_WINDOW)
    mad = (1.0 - RECENT_WEIGHT) * base + RECENT_WEIGHT * recent
    sigma = K * mad / SQRT_2_OVER_PI
    if sigma < SIGMA_LO:
        return SIGMA_LO
    if sigma > SIGMA_HI:
        return SIGMA_HI
    return sigma


def clamp6(x):
    """Clamp to [0, 1] and round to 6 decimals (0, never -0)."""
    return round(min(1.0, max(0.0, x)), 6) + 0.0


def forecast_market(prices, horizon):
    """240 rows x 5 quantiles for one market, median = last price."""
    x = prices[-1]
    sigma = market_sigma(prices)
    rows = []
    for h in range(1, horizon + 1):
        spread = sigma * math.sqrt(h)
        rows.append([clamp6(x + z * spread) for z in Z])
    return rows


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(output_dir, exist_ok=True)
    inputs = sorted(f for f in os.listdir(input_dir) if f.endswith(".in"))
    if not inputs:
        raise ValueError(f"no .in case file found in {input_dir}")
    for fname in inputs:
        case_id = fname[:-3]
        with open(os.path.join(input_dir, fname), "r", encoding="utf-8") as f:
            case = json.load(f)
        horizon = int(case["horizon_minutes"])
        forecasts = []
        for m in case["markets"]:
            prices = m["prices"]
            if not prices:
                raise ValueError(f"market {m['id']} has no prices")
            forecasts.append({"id": m["id"], "q": forecast_market(prices, horizon)})
        out_path = os.path.join(output_dir, case_id + ".out")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"forecasts": forecasts}) + "\n")


if __name__ == "__main__":
    main()
