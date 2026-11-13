#!/usr/bin/env python3
"""Polymarket Next-CEO-of-NVIDIA price forecaster.

Entry for make.fun competition polymarket-nvidia-ceo-forecast-01.
Reads each case file /input/<id>.in (one checkpoint, 72h of 1-minute Yes
prices per market) and writes /output/<id>.out: for each market, 240 rows
(one per minute after the checkpoint) of five quantiles (0.05, 0.25, 0.5,
0.75, 0.95) of that minute's Yes price.

Method (persistence median + per-market heavy-tailed spread, inner
  quantiles ramped with horizon):
  The median is the last price: prediction-market prices are close to a
  martingale at this horizon, and every drift I tried on the practice
  cases (reversion to a trailing mean, trend following, sum-normalisation)
  made the pinball loss worse, so the median stays at the checkpoint price.
  The spread is per market, estimated from that market's own recent
  1-minute moves, because markets differ hugely in activity: most
  minutes a price does not move at all (88% of 1-minute diffs in the
  practice are exactly zero) and then it jumps, sometimes by 20+ points
  (Colette Kress 0.3535 -> 0.59 -> 0.4585 in two minutes). A single
  global sigma (the house baseline's 0.0025) is far too wide for quiet
  low-price markets and too narrow for the jumpy leaders.
  For each market, sigma is the mean absolute 1-minute move over the
  last 240 / 720 / 1440 minutes (averaged), rescaled to a normal sigma
  (divide by sqrt(2/pi) = 0.7979), blended 60/40 with the last 60
  minutes' mean absolute move so a market that just woke up widens
  fast, times 0.6, clipped to [0.0001, 0.004]. Quantiles are
  q = x + z_tau(h) * sigma * sqrt(h), with heavy tails
  z = (-2.4, -zi(h), 0, zi(h), 2.4) where the inner multiplier ramps
  with horizon, zi(h) = 0.7 * h / (h + 10): at h=1 the price almost
  surely has not moved, so the 25%/75% quantiles sit almost at the last
  price; by h=240 they reach 0.7. The tails stay wide at every horizon
  because a jump can land at any minute. (A fixed inner multiplier was
  too wide at short horizons: the move distribution has a big atom at
  zero, so the inner quantiles must start near zero and grow in.)
  Values are clamped to [0, 1] and rounded to 6 decimals.
  Tried and rejected on the practice (each made the 6-case mean worse
  than 0.266): empirical quantiles per market/horizon from recent moves
  (0.37-0.38; the 72h history is far more volatile than the forecast
  windows); RMS (second-moment) sigma (0.44-0.56: one old jump inflates
  it forever); level-capped sigma (0.28-0.31); a jump-regime sigma boost
  after a recent big move (0.28-0.33: the MAD already prices persistent
  volatility, and the extra width is pure cost on quiet minutes);
  asymmetric tails; horizon exponents other than 0.5; trend drift.

Backtest (local, practice cases p00-p05 scored against the next case's
prices for p00-p04 and against Polymarket's public 1-minute history
02:01-06:00Z 13 Nov for p05; item-weighted mean over all cases,
markets, minutes and quantiles): pinball loss 0.266129 vs house
baseline 0.363589 on the same six cases (-26.8%), beating the baseline
on every single case (p00 0.473 vs 0.536, p01 0.307 vs 0.380, p02 0.349
vs 0.408, p03 0.232 vs 0.369, p04 0.204 vs 0.299, p05 0.099 vs 0.238);
5-case (p00-p04) mean 0.304655 vs 0.392518 for the baseline; median
error 0.7675, 90% coverage 0.872. The previous version scored 0.271465
on the six cases (0.308215 on five), so this revision gains 0.0053, mostly on the calm cases (p03 0.248 -> 0.232, p05 0.112 ->
0.099), while p00 (0.471 -> 0.473) and p02 (0.343 -> 0.349) degrade
slightly. This does NOT reach the ~0.255 the dev leaderboard's top 100
needs: the remaining loss is dominated by unpredictable jumps (an
in-sample per-market oracle sigma only reaches 0.2227). Verified
2026-11-13: output format valid on all 6 practice cases, two runs
byte-identical, ~0.06 s/case.

Stdlib only, no network, deterministic.
"""
import json
import math
import os

# --- fitted parameters (see docstring) ---
Z_TAIL = 2.4                        # heavy-tailed outer multiplier
Z_INNER_INF = 0.7                   # inner multiplier at long horizons
Z_INNER_H0 = 10.0                   # inner multiplier ramps as h/(h+H0)
K = 0.6                             # scale on the MAD-implied sigma
SIGMA_LO = 0.0001                   # floor: even a dead-quiet market moves
SIGMA_HI = 0.004                    # cap: never wider than this
WINDOWS = (240, 720, 1440)          # lookbacks for the base volatility
RECENT_WINDOW = 60                  # lookback for the fast volatility
RECENT_WEIGHT = 0.4                 # weight on the fast volatility
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
    """240 rows x 5 quantiles for one market, median = last price.

    The inner multipliers ramp with horizon: at h=1 the price almost
    surely has not moved (88% of 1-minute diffs are exactly zero), so
    the 25%/75% quantiles sit almost at the last price; by h=240 they
    reach their long-horizon value. The tails stay wide at every
    horizon because a jump can land at any minute.
    """
    x = prices[-1]
    sigma = market_sigma(prices)
    rows = []
    for h in range(1, horizon + 1):
        spread = sigma * math.sqrt(h)
        zi = Z_INNER_INF * h / (h + Z_INNER_H0)
        z_row = (-Z_TAIL, -zi, 0.0, zi, Z_TAIL)
        rows.append([clamp6(x + z * spread) for z in z_row])
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
