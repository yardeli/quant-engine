# Quant Engine Optimization Log

## Baseline (2026-03-11)
| Metric | Value |
|---|---|
| Total Return | 8.08% |
| Annualized Return | 1.43% |
| Sharpe Ratio | 0.17 |
| Sortino Ratio | 0.20 |
| Max Drawdown | -12.77% |
| Win Rate | 42.02% |
| Profit Factor | 1.04 |
| Ann. Volatility | 8.58% |
| Trading Days | 1378 |
| Total Costs | $42,678 |
| Turnover | 75.33x |

**Key observations:**
- Progressive DD de-risking starts at 7.5% (50% of 15% max_dd) — engine spends ~60% of time in de-risking mode
- Net exposure capped at 0.3 — can barely take directional bets
- VaR limit 2% triggers constantly, scaling positions down
- Pairs trading gets 56% weight (inverse_vol ensemble rewards stable but weak signals)
- $42k in costs on $1M = 4.3% drag

---

## Experiments

### Experiment 1: Relax Risk Constraints
**Hypothesis:** Safety layers compound multiplicatively, crushing returns. Relax DD, VaR, net exposure, and vol target.
**Changes:**
- max_drawdown: 0.15 → 0.25 (DD de-risking starts at 12.5% instead of 7.5%)
- max_var: 0.02 → 0.035
- max_net_exposure: 0.3 → 0.6
- vol_target: 0.10 → 0.15

**Result:** KEEP
| Metric | Before | After | Delta |
|---|---|---|---|
| Total Return | 8.08% | 18.08% | +10.00% |
| Ann. Return | 1.43% | 3.09% | +1.66% |
| Sharpe | 0.17 | 0.23 | +0.06 |
| Max Drawdown | -12.77% | -19.58% | -6.81% |
| Win Rate | 42.02% | 42.31% | +0.29% |
| Total Costs | $42,678 | $74,076 | +$31,398 |
| Turnover | 75.33x | 110.44x | +35.11x |

---

### Experiment 2: Reduce Turnover
**Hypothesis:** Rebalance every 10 days instead of 5 to cut costs ($74k → ~$40k). Also increase turnover penalty.
**Changes (on top of Exp 1):**
- rebalance_frequency: 5 → 10 (both portfolio and backtest configs)
- turnover_penalty: 0.002 → 0.005

**Result:** KEEP
| Metric | Before | After | Delta |
|---|---|---|---|
| Total Return | 18.08% | 23.71% | +5.63% |
| Sharpe | 0.23 | 0.29 | +0.06 |
| Max Drawdown | -19.58% | -19.38% | +0.20% |
| Total Costs | $74,076 | $49,099 | -$24,977 |
| Turnover | 110.44x | 69.88x | -40.56x |

---

### Experiment 3: Performance-Weighted Ensemble
**Hypothesis:** inverse_vol rewards signal stability, not quality. performance_weighted (IC-based) should favor models that actually predict returns.
**Changes:** ensemble_method: "inverse_vol" → "performance_weighted"

**Result:** KEEP
| Metric | Before | After | Delta |
|---|---|---|---|
| Total Return | 23.71% | 30.98% | +7.27% |
| Sharpe | 0.29 | 0.38 | +0.09 |
| Max Drawdown | -19.38% | -21.80% | -2.42% |
| Total Costs | $49,099 | $85,192 | +$36,093 |
| Turnover | 69.88x | 122.10x | +52.22x |

---

### Experiment 4: Tune Alpha Parameters
**Hypothesis:** Faster momentum (10d/63d) + tighter mean-reversion (entry 1.0z) + more aggressive ML retraining.
**Changes:**
- momentum_fast: 21 → 10
- momentum_slow: 126 → 63
- mean_rev_entry_z: 1.5 → 1.0
- stat_arb_entry_z: 2.0 → 1.5
- ml_retrain_frequency: 63 → 42
- ml_forward_return_horizon: 5 → 10

**Result:** KEEP
| Metric | Before | After | Delta |
|---|---|---|---|
| Total Return | 30.98% | 37.06% | +6.08% |
| Sharpe | 0.38 | 0.43 | +0.05 |
| Sortino | 0.47 | 0.57 | +0.10 |
| Max Drawdown | -21.80% | -16.29% | +5.51% |
| Total Costs | $85,192 | $116,820 | +$31,628 |
| Turnover | 122.10x | 151.25x | +29.15x |

---

### Experiment 5: Control Turnover + Bump Turnover Penalty
**Hypothesis:** Higher turnover penalty will reduce costs without killing the alpha edge from Exp 3+4.
**Changes:**
- turnover_penalty: 0.005 → 0.010 (100bps)

**Result:** DISCARD (hurt returns more than it saved in costs: 37.06% → 34.52%, only saved $3k)
Reverted turnover_penalty back to 0.005.

---

### Experiment 6: Mean-Variance Portfolio Optimization
**Hypothesis:** risk_parity ignores signal magnitude — it allocates equal risk regardless of conviction. mean_variance uses signals directly in optimization, putting more capital into high-conviction bets.
**Changes:**
- portfolio method: "risk_parity" → "mean_variance"
- max_position_size: 0.15 → 0.20 (allow bigger bets on high-conviction)

**Result:** *(pending)*

