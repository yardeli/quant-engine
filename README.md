# Quant Engine

An institutional-grade quantitative trading engine with 6 alpha models, adaptive ensemble weighting, portfolio optimization, and risk management.

## Performance

| Metric | Value |
|---|---|
| **Annualized Return** | **7.41%** |
| **Sharpe Ratio** | **0.49** |
| **Sortino Ratio** | **0.63** |
| **Max Drawdown** | **-19.65%** |
| Total Return (5yr) | 47.89% |
| Win Rate | 42.71% |
| Profit Factor | 1.11 |
| Calmar Ratio | 0.38 |

Backtested on 25 assets (equities, bonds, commodities, mega-cap tech, financials, energy) over 5 years with realistic transaction costs.

## Architecture

```
yfinance (free data)
    |
DataFeed --> FeatureEngine (20+ features, cross-sectional z-scored)
    |
    +-- TimeSeriesMomentum      (Moskowitz-style trend following)
    +-- CrossSectionalMomentum  (Jegadeesh-Titman 12-1 momentum)
    +-- MomentumWithVolBreak    (CTA de-risking on vol spikes + correlation crash filter)
    +-- OUMeanReversion         (Ornstein-Uhlenbeck with adaptive half-life)
    +-- PairsTrading            (Engle-Granger cointegration, top 5 pairs)
    +-- MLAlpha                 (Gradient boosting, walk-forward retrained)
    |
SignalAggregator (performance-weighted by realized IC)
    |
PortfolioOptimizer (mean-variance with turnover penalty)
    |
RiskManager (VaR, vol targeting, drawdown de-risking)
    |
ExecutionEngine (commission + spread + square-root market impact)
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run backtest (terminal dashboard)
python main.py

# Run backtest (headless, prints results)
python main.py --no-ui

# Web dashboard
python server.py
# Open http://localhost:5000

# CLI options
python main.py --tickers AAPL MSFT GOOGL --years 3
python main.py --method risk_parity --ensemble performance_weighted
python main.py --vol-target 0.12
```

## Alpha Models

### 1. Time-Series Momentum (`ts_momentum`)
Moskowitz-style trend following. Uses continuous z-score signals (not binary sign) with volatility scaling. Fast (10d) + slow (63d) blend.

### 2. Cross-Sectional Momentum (`xs_momentum`)
Jegadeesh-Titman 12-1 momentum: 12-month return minus last month (skip-month effect). Cross-sectionally rank-normalized.

### 3. Momentum with Vol Brake (`momentum_vol_break`)
Time-series momentum scaled down during volatility spikes AND correlation regime changes. Avoids momentum crashes.

### 4. OU Mean Reversion (`ou_mean_reversion`)
Ornstein-Uhlenbeck process with adaptive half-life estimation. Only trades assets where half-life is 2-63 days. Uses 126-day lookback for faster regime adaptation.

### 5. Pairs Trading (`pairs_trading`)
Engle-Granger cointegration on all pairs, trades top 5 by R-squared. Z-score entry/exit with stop-loss.

### 6. ML Alpha (`ml_alpha`)
Gradient boosting regressor predicting 10-day forward returns from 20+ features. Walk-forward retrained every ~125 days. Shallow trees (depth 4) + min_samples_leaf 20 to prevent overfitting.

## Configuration

All parameters in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `momentum_fast` | 10 | Fast momentum lookback (days) |
| `momentum_slow` | 63 | Slow momentum lookback (days) |
| `mean_rev_entry_z` | 1.0 | Z-score to enter mean reversion |
| `ensemble_method` | `performance_weighted` | IC-based adaptive weighting |
| `method` | `mean_variance` | Portfolio optimization |
| `risk_aversion` | 1.5 | Mean-variance lambda |
| `max_position_size` | 0.20 | 20% max single name |
| `max_gross_leverage` | 1.5 | 150% gross |
| `vol_target` | 0.15 | 15% annualized vol target |
| `max_drawdown` | 0.25 | Hard stop at 25% drawdown |
| `rebalance_frequency` | 10 | Days between rebalances |

## Optimization History

22 iterations of systematic optimization (see `autoresearch_results.md`):

**4 kept improvements:**
1. **Continuous z-score momentum** — replaced binary sign() with vol-normalized z-scores
2. **Correlation crash filter** — reduces momentum exposure during correlation spikes
3. **Shorter OU lookback (126d)** — faster regime adaptation vs 252d
4. **Lower risk aversion (1.5)** — more alpha-seeking optimization vs 2.0

## Project Structure

```
quant-engine-local/
├── alpha/
│   ├── base.py            # Abstract AlphaModel
│   ├── momentum.py        # TS, XS, Vol Brake momentum
│   ├── mean_reversion.py  # OU mean reversion + pairs trading
│   └── ml_alpha.py        # Gradient boosting
├── backtest/
│   └── engine.py          # Walk-forward backtester
├── data/
│   └── feed.py            # yfinance data loader
├── ensemble/
│   └── aggregator.py      # Signal combination (equal/inv-vol/IC-weighted)
├── execution/
│   └── engine.py          # Transaction cost modeling
├── features/
│   └── engine.py          # 20+ technical features
├── portfolio/
│   └── optimizer.py       # Mean-variance, risk parity, Black-Litterman
├── risk/
│   └── manager.py         # VaR, vol targeting, drawdown limits
├── ui/
│   └── dashboard.py       # Rich terminal dashboard
├── config.py              # All tunable parameters
├── main.py                # CLI entry point
├── server.py              # Flask web server + REST API
└── dashboard.html         # Web dashboard
```

## Integration with Trading Bot

The engine exports a `brain_export.json` that maps model weights to the trading bot's strategy system. Run `python server.py` and the trading bot dashboard can fetch signals via REST API.

## Dependencies

- Python 3.12+
- pandas, numpy, scipy, scikit-learn
- yfinance (free market data)
- flask, flask-cors (web server)
- rich (terminal dashboard)
