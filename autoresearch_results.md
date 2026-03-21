# Autoresearch Optimization Log

## Goal
Maximize annualized return and Sharpe ratio while keeping max drawdown < 20%.

## Baseline (Iteration #0)
| Metric | Value |
|---|---|
| Total Return | 14.14% |
| Annualized Return | 2.45% |
| Sharpe Ratio | 0.18 |
| Sortino Ratio | 0.22 |
| Max Drawdown | -23.18% |
| Win Rate | 41.84% |
| Profit Factor | 1.05 |
| Total Costs | $87,308 |
| Turnover | 132.63x |

## Current Best
| Metric | Value | Delta from Baseline |
|---|---|---|
| Total Return | 47.89% | +33.75% |
| Annualized Return | 7.41% | +4.96% |
| Sharpe Ratio | 0.49 | +0.31 |
| Sortino Ratio | 0.63 | +0.41 |
| Max Drawdown | -19.65% | +3.53% (under 20% target!) |
| Win Rate | 42.71% | +0.87% |
| Profit Factor | 1.11 | +0.06 |
| Total Costs | $90,009 | +$2,701 |
| Turnover | 145.76x | +13.13x |

## Iterations

| # | Change | Ann. Return | Sharpe | Max DD | Result |
|---|--------|-------------|--------|--------|--------|
| 1 | Black-Litterman + tighter DD | -3.29% | -0.52 | -20.21% | DISCARD |
| 2 | Lower risk aversion (1.0) + higher vol (20%) | 2.95% | 0.17 | -25.25% | DISCARD |
| 3 | Trend strength filter on TS momentum | 0.96% | 0.07 | -23.16% | DISCARD |
| 4 | **Continuous z-score momentum signals** | **3.59%** | **0.25** | **-22.39%** | **KEEP** |
| 5 | Multi-horizon XS momentum (12-1, 6-1, 3-1) | -1.39% | -0.10 | -24.46% | DISCARD |
| 6 | Reduce rebalance to 15 days | 1.87% | 0.12 | -17.90% | DISCARD |
| 7 | **Correlation crash filter for vol brake** | **4.30%** | **0.29** | **-21.25%** | **KEEP** |
| 8 | **Shorter OU lookback (126d)** | **5.60%** | **0.38** | **-22.63%** | **KEEP** |
| 9 | Faster ML retraining (3x vs 5x) | 2.45% | 0.17 | -22.49% | DISCARD |
| 10 | Max position 25% + lower turnover penalty | 1.86% | 0.13 | -25.04% | DISCARD |
| 11 | Exp-decay IC in ensemble | -2.25% | -0.20 | -28.83% | DISCARD |
| 12 | Net exposure 0.8 | 1.62% | 0.12 | -24.79% | DISCARD |
| 13 | Sector-neutral pairs only | 3.49% | 0.24 | -24.11% | DISCARD |
| 14 | **Lower risk aversion to 1.5** | **7.41%** | **0.49** | **-19.65%** | **KEEP** |
| 15 | Faster momentum slow (42d) | 1.92% | 0.13 | -24.44% | DISCARD |
| 16 | Shorter cov halflife (42d) | 5.87% | 0.39 | -18.92% | DISCARD |
| 17 | Gross leverage 2.0x | 7.00% | 0.46 | -19.92% | DISCARD |
| 18 | Min IC floor (0.02) | 1.68% | 0.12 | -23.86% | DISCARD |

## Kept Changes (Applied Cumulatively)
1. **Continuous z-score momentum** (iteration 4) — replaced binary sign() with vol-normalized z-scores
2. **Correlation crash filter** (iteration 7) — reduces momentum exposure during correlation spikes
3. **Shorter OU lookback** (iteration 8) — 126d vs 252d for faster regime adaptation
4. **Lower risk aversion** (iteration 14) — 1.5 vs 2.0 for more alpha-seeking optimization
