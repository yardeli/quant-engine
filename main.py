"""
Institutional Quantitative Trading Engine — Main Orchestrator.

Ties together all modules:
    Data -> Features -> Alpha Models -> Ensemble -> Portfolio -> Risk -> Execution

Usage:
    python main.py                     # Run with defaults (terminal dashboard)
    python main.py --tickers AAPL MSFT GOOGL --years 5
    python main.py --method risk_parity --ensemble performance_weighted
    python main.py --no-ui             # Headless mode (no dashboard)

The system downloads data via yfinance, computes features, generates
alpha signals from multiple models, combines them, constructs an
optimal portfolio, applies risk limits, and simulates execution
with realistic transaction costs.
"""
import argparse
import logging
import sys

import numpy as np

from alpha.mean_reversion import OUMeanReversion, PairsTrading
from alpha.ml_alpha import MLAlpha
from alpha.momentum import (
    CrossSectionalMomentum,
    MomentumWithVolBreak,
    TimeSeriesMomentum,
)
from backtest.engine import BacktestEngine
from config import SystemConfig
from data.feed import DataFeed


def build_alpha_models(config: SystemConfig) -> list:
    """Instantiate all alpha models."""
    models = []

    alpha_cfg = config.alpha

    # Momentum family
    if "ts_momentum" in alpha_cfg.enabled_models:
        models.append(TimeSeriesMomentum(alpha_cfg))
    if "xs_momentum" in alpha_cfg.enabled_models:
        models.append(CrossSectionalMomentum(alpha_cfg))
    if "momentum_vol_break" in alpha_cfg.enabled_models:
        models.append(MomentumWithVolBreak(alpha_cfg))

    # Mean reversion family
    if "ou_mean_reversion" in alpha_cfg.enabled_models:
        models.append(OUMeanReversion(alpha_cfg))
    if "pairs_trading" in alpha_cfg.enabled_models:
        models.append(PairsTrading(alpha_cfg))

    # ML
    if "ml_alpha" in alpha_cfg.enabled_models:
        models.append(MLAlpha(alpha_cfg))

    if not models:
        models = [
            TimeSeriesMomentum(alpha_cfg),
            CrossSectionalMomentum(alpha_cfg),
            MomentumWithVolBreak(alpha_cfg),
            OUMeanReversion(alpha_cfg),
            PairsTrading(alpha_cfg),
            MLAlpha(alpha_cfg),
        ]

    return models


def main():
    parser = argparse.ArgumentParser(
        description="Institutional Quantitative Trading Engine"
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="List of tickers (default: diversified universe)",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=None,
        help="Years of historical data (default: 5)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=1_000_000,
        help="Initial capital (default: 1,000,000)",
    )
    parser.add_argument(
        "--method",
        choices=["mean_variance", "risk_parity", "black_litterman"],
        default=None,
        help="Portfolio optimization method",
    )
    parser.add_argument(
        "--ensemble",
        choices=["equal_weight", "inverse_vol", "performance_weighted"],
        default=None,
        help="Signal ensemble method",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Alpha models to enable (e.g. ts_momentum xs_momentum ml_alpha)",
    )
    parser.add_argument(
        "--vol-target",
        type=float,
        default=None,
        help="Annualized volatility target (e.g. 0.15 for 15%%)",
    )
    parser.add_argument(
        "--no-ui",
        action="store_true",
        help="Disable terminal dashboard (headless mode)",
    )

    args = parser.parse_args()

    # Build configuration
    config = SystemConfig()

    if args.tickers:
        config.data.tickers = args.tickers
    if args.years:
        config.data.years = args.years
    if args.method:
        config.portfolio.method = args.method
    if args.ensemble:
        config.alpha.ensemble_method = args.ensemble
    if args.models:
        config.alpha.enabled_models = args.models
    if args.vol_target:
        config.risk.vol_target = args.vol_target

    # Set up logging — suppress during UI mode to avoid clobbering the dashboard
    use_ui = not args.no_ui
    if use_ui:
        logging.basicConfig(level=logging.WARNING, format="%(message)s")
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    logger = logging.getLogger(__name__)

    # Dashboard setup
    dashboard = None
    if use_ui:
        try:
            from ui.dashboard import TerminalDashboard
            dashboard = TerminalDashboard()
        except ImportError:
            print("Install 'rich' for the terminal dashboard: pip install rich")
            print("Falling back to headless mode.\n")
            use_ui = False
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
                force=True,
            )

    # Load data
    if dashboard:
        from rich.console import Console
        console = Console()
        with console.status("[bold blue]Downloading market data...", spinner="dots"):
            data = DataFeed(config.data)
            data.load()
        with console.status("[bold blue]Building alpha models...", spinner="dots"):
            models = build_alpha_models(config)
        console.print(
            f"[green]Loaded {len(data.prices)} days x "
            f"{len(data.prices.columns)} assets[/]"
        )
        console.print(
            f"[green]Active models: {[m.name for m in models]}[/]\n"
        )
    else:
        logger.info("Downloading market data...")
        data = DataFeed(config.data)
        data.load()
        logger.info(
            f"Loaded {len(data.prices)} days x {len(data.prices.columns)} assets"
        )
        models = build_alpha_models(config)
        logger.info(f"Active models: {[m.name for m in models]}")

    # Start dashboard
    if dashboard:
        config_info = {
            "n_assets": len(data.prices.columns),
            "years": config.data.years,
            "method": config.portfolio.method,
            "ensemble": config.alpha.ensemble_method,
            "vol_target": config.risk.vol_target,
        }
        dashboard.on_backtest_start(
            total_days=len(data.prices),
            config_info=config_info,
            initial_capital=args.capital,
        )

    # Run backtest
    engine = BacktestEngine(config)
    result = engine.run(
        data, models,
        initial_capital=args.capital,
        dashboard=dashboard,
    )

    # Export brain for paper-trader-v4
    try:
        brain_data = engine.export_brain(result)
        logger.info(
            f"Brain exported: {len(brain_data['strategies'])} strategies, "
            f"regime={brain_data['regime']['type']}"
        )
    except Exception as e:
        logger.warning(f"Brain export failed: {e}")

    # Save chart if matplotlib is available
    if not use_ui:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

            result.equity_curve.plot(ax=axes[0], linewidth=1.5)
            axes[0].set_title("Portfolio Equity Curve")
            axes[0].set_ylabel("Equity ($)")
            axes[0].grid(True, alpha=0.3)

            cumret = result.equity_curve / result.equity_curve.iloc[0]
            running_max = cumret.cummax()
            drawdown = (cumret - running_max) / running_max
            drawdown.plot(ax=axes[1], linewidth=1, color="red")
            axes[1].fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color="red")
            axes[1].set_title("Drawdown")
            axes[1].set_ylabel("Drawdown (%)")
            axes[1].grid(True, alpha=0.3)

            if len(result.returns) > 63:
                rolling_sharpe = (
                    result.returns.rolling(63).mean()
                    / result.returns.rolling(63).std()
                    * np.sqrt(252)
                )
                rolling_sharpe.plot(ax=axes[2], linewidth=1, color="green")
                axes[2].axhline(y=0, color="black", linewidth=0.5)
                axes[2].set_title("Rolling 63-Day Sharpe Ratio")
                axes[2].set_ylabel("Sharpe")
                axes[2].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig("backtest_results.png", dpi=150, bbox_inches="tight")
            logger.info("Chart saved to backtest_results.png")
            try:
                plt.show(block=False)
                plt.pause(0.5)
                plt.close("all")
            except Exception:
                plt.close("all")
        except ImportError:
            pass

    return result


if __name__ == "__main__":
    main()
