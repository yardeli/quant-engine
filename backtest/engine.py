"""
Walk-Forward Backtesting Engine.

Simulates the full trading pipeline on historical data:
    Data → Features → Alpha → Ensemble → Portfolio → Risk → Execution

Key Design Principles:
    - Walk-forward only: no future information leaks.
    - Features, signals, and portfolio weights are computed
      using only data available at each point in time.
    - Transaction costs are modeled realistically.
    - Performance is measured with proper risk-adjusted metrics.

Hedge Fund Usage:
    - Walk-forward backtesting is the minimum standard.
    - In-sample / out-of-sample splits prevent overfitting.
    - Monte Carlo permutation tests verify statistical significance.
    - Realistic transaction costs separate paper alpha from real alpha.
"""
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from alpha.base import AlphaModel
from config import SystemConfig
from data.feed import DataFeed
from ensemble.aggregator import SignalAggregator
from execution.engine import ExecutionEngine
from features.engine import FeatureEngine
from portfolio.optimizer import PortfolioOptimizer
from risk.manager import RiskManager

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Container for backtest results."""
    equity_curve: pd.Series
    returns: pd.Series
    weights_history: pd.DataFrame
    trade_log: list[dict]
    risk_metrics_history: list[dict]
    execution_summary: dict
    performance_metrics: dict


class BacktestEngine:
    """
    Walk-forward backtesting engine.

    Runs the complete pipeline day by day on historical data,
    respecting the temporal ordering to prevent look-ahead bias.
    """

    def __init__(self, config: SystemConfig):
        self.config = config
        self.feature_engine = FeatureEngine(config.features)
        self.signal_aggregator = SignalAggregator(
            method=config.alpha.ensemble_method,
            ic_lookback=63,
        )
        self.portfolio_optimizer = PortfolioOptimizer(config.portfolio)
        self.risk_manager = RiskManager(config.risk)
        self.execution_engine = ExecutionEngine(config.execution)

    def run(
        self,
        data: DataFeed,
        alpha_models: list[AlphaModel],
        initial_capital: float = 1_000_000,
        dashboard=None,
    ) -> BacktestResult:
        """
        Run a full walk-forward backtest.

        Args:
            data: Historical market data.
            alpha_models: List of alpha models to run.
            initial_capital: Starting capital.
            dashboard: Optional TerminalDashboard for live UI updates.

        Returns:
            BacktestResult with equity curve, metrics, etc.
        """
        prices = data.prices
        returns = data.returns
        dates = prices.index
        assets = prices.columns.tolist()

        # Initialize execution engine
        self.execution_engine.initialize(initial_capital, assets)

        # Compute features once (they only use past data internally)
        logger.info("Computing features...")
        features = self.feature_engine.generate(data)

        # Generate signals from each alpha model
        logger.info("Generating alpha signals...")
        all_signals = {}
        for model in alpha_models:
            logger.info(f"  Running {model.name}...")
            sig = model.generate_signals(data, features)
            all_signals[model.name] = sig

        # Determine rebalance dates
        warmup = self.config.backtest.warmup_period
        rebal_freq = self.config.backtest.rebalance_frequency
        rebalance_dates = dates[warmup::rebal_freq]

        logger.info(
            f"Backtesting {len(rebalance_dates)} rebalance dates "
            f"({dates[warmup]} to {dates[-1]})"
        )

        # Walk-forward simulation
        equity_values = []
        weights_history = []
        risk_metrics_history = []

        current_weights = pd.Series(0.0, index=assets)

        for i, date in enumerate(dates):
            current_prices = prices.loc[date]

            # Record equity
            port_value = self.execution_engine.get_portfolio_value(current_prices)
            equity_values.append({"date": date, "equity": port_value})

            is_rebalance = date in rebalance_dates.values
            trade_info = None

            # Rebalance if scheduled
            if is_rebalance:
                # Get signals for this date (use data up to this date only)
                date_signals = {}
                for name, sig_df in all_signals.items():
                    if date in sig_df.index:
                        date_signals[name] = sig_df.loc[:date]

                if date_signals:
                    # Aggregate signals
                    try:
                        current_signals = {}
                        for name, sig_df in date_signals.items():
                            current_signals[name] = sig_df

                        combined = self.signal_aggregator.aggregate(
                            current_signals, returns.loc[:date]
                        )
                        signal_row = combined.iloc[-1]
                    except Exception as e:
                        logger.debug(f"Signal aggregation failed at {date}: {e}")
                        signal_row = None

                    if signal_row is not None:
                        # Estimate covariance from recent returns
                        lookback_returns = returns.loc[:date].iloc[-252:]
                        if len(lookback_returns) >= 60:
                            cov_matrix = PortfolioOptimizer.estimate_covariance(
                                lookback_returns,
                                method="exponential",
                                halflife=63,
                            )

                            # Optimize portfolio
                            try:
                                target_weights = self.portfolio_optimizer.optimize(
                                    signal_row, cov_matrix, current_weights
                                )
                            except Exception as e:
                                logger.debug(f"Optimization failed at {date}: {e}")
                                target_weights = None

                            if target_weights is not None:
                                # Risk checks
                                equity_curve_so_far = pd.Series(
                                    {e["date"]: e["equity"] for e in equity_values}
                                )
                                target_weights = self.risk_manager.check_and_adjust(
                                    target_weights, lookback_returns, equity_curve_so_far
                                )

                                # Execute trades
                                volumes = data.volume.loc[date] if data.volume is not None else None
                                self.execution_engine.execute_rebalance(
                                    target_weights, current_prices, volumes, date
                                )

                                current_weights = self.execution_engine.get_weights(current_prices)

                                # Record
                                weights_history.append({"date": date, **current_weights.to_dict()})
                                risk_metrics_history.append(
                                    {"date": date, **self.risk_manager.get_risk_report()}
                                )

                                if self.execution_engine.trade_log:
                                    trade_info = self.execution_engine.trade_log[-1]

            # Dashboard update (once per day, after any rebalance)
            if dashboard is not None:
                day_kwargs = {"date": date, "equity": port_value}
                if is_rebalance and current_weights is not None:
                    day_kwargs["weights"] = {
                        k: v for k, v in current_weights.items() if abs(v) > 0.001
                    }
                if self.risk_manager.risk_metrics:
                    day_kwargs["risk_metrics"] = self.risk_manager.get_risk_report()
                if self.signal_aggregator.model_weights:
                    day_kwargs["model_weights"] = self.signal_aggregator.model_weights
                if trade_info is not None:
                    day_kwargs["trade_info"] = trade_info
                dashboard.on_day_update(**day_kwargs)

        # Notify dashboard
        if dashboard is not None:
            dashboard.on_backtest_complete()

        # Build results
        equity_df = pd.DataFrame(equity_values).set_index("date")["equity"]
        port_returns = equity_df.pct_change().dropna()

        weights_df = pd.DataFrame(weights_history)
        if not weights_df.empty:
            weights_df = weights_df.set_index("date")

        performance = self._compute_performance_metrics(
            equity_df, port_returns, initial_capital
        )

        result = BacktestResult(
            equity_curve=equity_df,
            returns=port_returns,
            weights_history=weights_df,
            trade_log=self.execution_engine.trade_log,
            risk_metrics_history=risk_metrics_history,
            execution_summary=self.execution_engine.get_execution_summary(),
            performance_metrics=performance,
        )

        # Show final report in dashboard or print to console
        if dashboard is not None:
            dashboard.show_final_report(result)
        else:
            self._print_summary(result)

        return result

    def _compute_performance_metrics(
        self,
        equity: pd.Series,
        returns: pd.Series,
        initial_capital: float,
    ) -> dict:
        """Compute comprehensive performance statistics."""
        if returns.empty or len(returns) < 2:
            return {}

        total_return = (equity.iloc[-1] / initial_capital) - 1
        n_years = len(returns) / 252
        ann_return = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1
        ann_vol = returns.std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

        # Sortino ratio (downside deviation only)
        downside = returns[returns < 0]
        downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else ann_vol
        sortino = ann_return / downside_vol if downside_vol > 0 else 0.0

        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()

        # Calmar ratio
        calmar = ann_return / abs(max_dd) if max_dd != 0 else 0.0

        # Win rate
        win_rate = (returns > 0).mean()

        # Profit factor
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        profit_factor = gains / losses if losses > 0 else float("inf")

        # Skewness and kurtosis
        skew = returns.skew()
        kurt = returns.kurtosis()

        return {
            "total_return": total_return,
            "annualized_return": ann_return,
            "annualized_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_dd,
            "calmar_ratio": calmar,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "skewness": skew,
            "kurtosis": kurt,
            "n_trading_days": len(returns),
            "final_equity": equity.iloc[-1],
        }

    @staticmethod
    def _print_summary(result: BacktestResult) -> None:
        """Print a formatted performance summary."""
        m = result.performance_metrics
        if not m:
            logger.warning("No performance metrics to display")
            return

        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)
        print(f"  Total Return:        {m['total_return']:>10.2%}")
        print(f"  Annualized Return:   {m['annualized_return']:>10.2%}")
        print(f"  Annualized Vol:      {m['annualized_volatility']:>10.2%}")
        print(f"  Sharpe Ratio:        {m['sharpe_ratio']:>10.2f}")
        print(f"  Sortino Ratio:       {m['sortino_ratio']:>10.2f}")
        print(f"  Max Drawdown:        {m['max_drawdown']:>10.2%}")
        print(f"  Calmar Ratio:        {m['calmar_ratio']:>10.2f}")
        print(f"  Win Rate:            {m['win_rate']:>10.2%}")
        print(f"  Profit Factor:       {m['profit_factor']:>10.2f}")
        print(f"  Skewness:            {m['skewness']:>10.2f}")
        print(f"  Kurtosis:            {m['kurtosis']:>10.2f}")
        print(f"  Trading Days:        {m['n_trading_days']:>10d}")
        print(f"  Final Equity:        ${m['final_equity']:>12,.2f}")
        print("-" * 60)

        ex = result.execution_summary
        print(f"  Total Costs:         ${ex['total_costs']:>12,.2f}")
        print(f"  Total Turnover:      {ex['total_turnover']:>10.2f}x")
        print(f"  Rebalances:          {ex['n_rebalances']:>10d}")
        print("=" * 60 + "\n")
