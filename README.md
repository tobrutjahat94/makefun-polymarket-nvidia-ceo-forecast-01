# polymarket-nvidia-ceo-forecast-01

Forecast the Yes price of every trading market of Polymarket's
"Next CEO of NVIDIA?" event, 4 hours ahead, minute by minute, as five
quantiles (0.05 / 0.25 / 0.5 / 0.75 / 0.95) — for the make.fun
competition `polymarket-nvidia-ceo-forecast-01` ($250, 100 places).

## Approach

- **Median = last price.** Prediction-market prices are close to a
  martingale at this horizon. Every drift tried on the practice cases
  (reversion to a 1h/4h/24h trailing mean, sum-normalisation across the
  24 markets) made the pinball loss *worse*, so the median stays at the
  checkpoint price.
- **Per-market spread from its own recent moves.** 88% of 1-minute
  diffs in the practice are exactly zero, then the price jumps —
  sometimes 20+ points in a minute (Colette Kress 0.3535 → 0.59 →
  0.4585 in two minutes). A single global sigma (house baseline 0.0025)
  is far too wide for quiet low-price markets and too narrow for the
  jumpy leaders. Sigma = 0.55 × mean |1-min move| over the last
  240/720/1440 min (averaged), blended 80/20 with the last 60 min so a
  market that just woke up widens fast, rescaled by 1/0.7979,
  clipped to [0.0004, 0.003], growing as sqrt(h).
- **Heavy tails.** Quantile multipliers (−2.2, −0.7, 0, 0.7, 2.2)
  instead of the normal's (−1.645, −0.674, 0, 0.674, 1.645): the wider
  tails pay for the jumps, the slightly wider inner quantiles cost
  little on quiet minutes.

Stdlib only, no network, deterministic.

## Backtest (practice, local)

Cases p00–p04 scored against the next case's prices (p05 has no
locally available answer):

| case | this entry | house baseline |
|------|-----------|----------------|
| p00  | 0.471 | 0.536 |
| p01  | 0.311 | 0.380 |
| p02  | 0.343 | 0.408 |
| p03  | 0.248 | 0.369 |
| p04  | 0.207 | 0.299 |
| **mean** | **0.308215** | **0.392518** |

Median error 0.8795, 90% coverage 0.908. House baseline scores
0.363589 on the full six-case practice.

## Run

```sh
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Writes `out/<id>.out` per `<id>.in` case. Verified: format valid on
all 6 practice cases, two runs byte-identical, ~0.06 s per case
(well under the 8 s cap).

## Files

- `main.py` — the entry (Python 3.13, stdlib only)
- `LICENSE` — MIT
