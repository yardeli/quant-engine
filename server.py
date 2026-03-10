"""
Quant Engine Web Server.

Provides a Flask API that runs the quantitative backtesting engine
and serves results to the web dashboard.

Usage:
    python server.py                    # Start server on port 5000
    python server.py --port 8080        # Custom port
"""
import argparse
import json
import logging
import sys
import threading
import time
from datetime import datetime

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return str(obj)[:10]
        return super().default(obj)


app.json.encoder = NumpyEncoder  # type: ignore
# Also override json provider for Flask 3.x
import flask.json.provider as _fjp
_orig_dumps = _fjp.DefaultJSONProvider.dumps
def _patched_dumps(self, obj, **kwargs):
    kwargs.setdefault('cls', NumpyEncoder)
    return json.dumps(obj, **kwargs)
_fjp.DefaultJSONProvider.dumps = _patched_dumps

# ── Global state ─────────────────────────────────────────────────
engine_state = {
    "status": "idle",       # idle, loading, running, complete, error
    "progress": 0,          # 0-100
    "progress_msg": "",
    "result": None,
    "error": None,
    "config": None,
    "start_time": None,
    "elapsed": 0,
    # Live streaming data during backtest
    "equity_history": [],
    "date_history": [],
    "current_day": 0,
    "total_days": 0,
    "current_weights": {},
    "risk_metrics": {},
    "model_weights": {},
    "recent_trades": [],
    "running_metrics": {
        "total_return": 0, "ann_return": 0, "ann_vol": 0,
        "sharpe": 0, "max_dd": 0, "current_dd": 0, "win_rate": 0,
    },
}
engine_lock = threading.Lock()


class WebDashboardCallback:
    """Callback adapter: feeds backtest progress to the web API state."""

    def __init__(self):
        self.peak_equity = 0
        self.initial_capital = 1_000_000
        self.return_history = []

    def on_backtest_start(self, total_days, config_info, initial_capital):
        with engine_lock:
            engine_state["total_days"] = total_days
            engine_state["current_day"] = 0
            engine_state["equity_history"] = []
            engine_state["date_history"] = []
            engine_state["status"] = "running"
            engine_state["progress_msg"] = "Backtesting..."
        self.initial_capital = initial_capital
        self.peak_equity = initial_capital

    def on_day_update(self, date, equity, weights=None, risk_metrics=None,
                      model_weights=None, trade_info=None):
        with engine_lock:
            engine_state["current_day"] += 1
            engine_state["equity_history"].append(round(equity, 2))
            date_str = str(date)[:10]
            engine_state["date_history"].append(date_str)

            pct = engine_state["current_day"] / max(engine_state["total_days"], 1) * 100
            engine_state["progress"] = round(pct, 1)

            if weights is not None:
                engine_state["current_weights"] = {
                    k: round(v, 4) for k, v in weights.items() if abs(v) > 0.001
                }
            if risk_metrics is not None:
                engine_state["risk_metrics"] = {
                    k: round(v, 6) if isinstance(v, float) else v
                    for k, v in risk_metrics.items()
                }
            if model_weights is not None:
                engine_state["model_weights"] = {
                    k: round(v, 4) if v == v else 0
                    for k, v in model_weights.items()
                }
            if trade_info is not None:
                engine_state["recent_trades"].append({
                    k: (round(v, 4) if isinstance(v, float) else str(v)[:10] if hasattr(v, 'strftime') else v)
                    for k, v in trade_info.items()
                })
                if len(engine_state["recent_trades"]) > 20:
                    engine_state["recent_trades"] = engine_state["recent_trades"][-20:]

            # Running metrics
            eq = engine_state["equity_history"]
            n = len(eq)
            if n > 1:
                ret = (eq[-1] - eq[-2]) / eq[-2] if eq[-2] != 0 else 0
                self.return_history.append(ret)

            self.peak_equity = max(self.peak_equity, equity)
            total_ret = (equity / self.initial_capital) - 1
            dd = (equity - self.peak_equity) / self.peak_equity if self.peak_equity > 0 else 0
            n_years = n / 252

            ann_ret = 0
            ann_vol = 0
            sharpe = 0
            win_rate = 0
            if n_years > 0.05:
                ann_ret = (1 + total_ret) ** (1 / n_years) - 1
            if len(self.return_history) > 20:
                rets = np.array(self.return_history[-252:])
                ann_vol = float(np.std(rets) * np.sqrt(252))
                if ann_vol > 0:
                    sharpe = ann_ret / ann_vol
                win_rate = float(np.mean(np.array(rets) > 0))

            engine_state["running_metrics"] = {
                "total_return": round(total_ret, 6),
                "ann_return": round(ann_ret, 6),
                "ann_vol": round(ann_vol, 6),
                "sharpe": round(sharpe, 4),
                "max_dd": round(min([dd] + [d for d in [engine_state["running_metrics"].get("max_dd", 0)]]), 6),
                "current_dd": round(dd, 6),
                "win_rate": round(win_rate, 4),
            }

    def on_backtest_complete(self):
        with engine_lock:
            engine_state["progress"] = 100
            engine_state["progress_msg"] = "Complete"

    def show_final_report(self, result):
        pass  # We handle this in the run thread


def _run_backtest(config_overrides: dict):
    """Run the full backtest pipeline in a background thread."""
    try:
        with engine_lock:
            engine_state["status"] = "loading"
            engine_state["progress"] = 0
            engine_state["progress_msg"] = "Initializing..."
            engine_state["error"] = None
            engine_state["result"] = None
            engine_state["equity_history"] = []
            engine_state["date_history"] = []
            engine_state["recent_trades"] = []
            engine_state["start_time"] = time.time()

        config = SystemConfig()

        # Apply overrides
        if "tickers" in config_overrides and config_overrides["tickers"]:
            config.data.tickers = config_overrides["tickers"]
        if "years" in config_overrides:
            config.data.years = int(config_overrides["years"])
        if "capital" in config_overrides:
            capital = float(config_overrides["capital"])
        else:
            capital = 1_000_000
        if "method" in config_overrides and config_overrides["method"]:
            config.portfolio.method = config_overrides["method"]
        if "ensemble" in config_overrides and config_overrides["ensemble"]:
            config.alpha.ensemble_method = config_overrides["ensemble"]
        if "vol_target" in config_overrides:
            config.risk.vol_target = float(config_overrides["vol_target"])
        if "models" in config_overrides and config_overrides["models"]:
            config.alpha.enabled_models = config_overrides["models"]

        with engine_lock:
            engine_state["config"] = {
                "tickers": config.data.tickers,
                "years": config.data.years,
                "capital": capital,
                "method": config.portfolio.method,
                "ensemble": config.alpha.ensemble_method,
                "vol_target": config.risk.vol_target,
                "models": config.alpha.enabled_models,
            }

        # Load data
        with engine_lock:
            engine_state["progress_msg"] = "Downloading market data..."
            engine_state["progress"] = 5

        feed = DataFeed(config.data)
        feed.fetch()
        logger.info(f"Data loaded: {feed.prices.shape}")

        with engine_lock:
            engine_state["progress_msg"] = "Building alpha models..."
            engine_state["progress"] = 15

        # Build models
        models = []
        enabled = config.alpha.enabled_models
        if "ts_momentum" in enabled:
            models.append(TimeSeriesMomentum(config.alpha))
        if "xs_momentum" in enabled:
            models.append(CrossSectionalMomentum(config.alpha))
        if "momentum_vol_break" in enabled:
            models.append(MomentumWithVolBreak(config.alpha))
        if "ou_mean_reversion" in enabled:
            models.append(OUMeanReversion(config.alpha))
        if "pairs_trading" in enabled:
            models.append(PairsTrading(config.alpha))
        if "ml_alpha" in enabled:
            models.append(MLAlpha(config.alpha))

        if not models:
            models = [
                TimeSeriesMomentum(config.alpha),
                CrossSectionalMomentum(config.alpha),
                MomentumWithVolBreak(config.alpha),
                OUMeanReversion(config.alpha),
                PairsTrading(config.alpha),
                MLAlpha(config.alpha),
            ]

        logger.info(f"Active models: {[m.name for m in models]}")

        with engine_lock:
            engine_state["progress_msg"] = "Running backtest..."
            engine_state["progress"] = 20

        # Create callback and initialize it
        callback = WebDashboardCallback()
        callback.on_backtest_start(
            total_days=len(feed.prices),
            config_info={
                "n_assets": len(feed.prices.columns),
                "years": config.data.years,
                "method": config.portfolio.method,
                "ensemble": config.alpha.ensemble_method,
                "vol_target": config.risk.vol_target,
            },
            initial_capital=capital,
        )

        # Run backtest
        engine = BacktestEngine(config)
        result = engine.run(
            feed, models,
            initial_capital=capital,
            dashboard=callback,
        )

        # Package results
        perf = result.performance_metrics
        exec_summary = result.execution_summary

        # Equity curve (downsample for large datasets)
        eq_vals = result.equity_curve.values.tolist()
        eq_dates = [str(d)[:10] for d in result.equity_curve.index]

        # Weights history
        weights_hist = {}
        if not result.weights_history.empty:
            for col in result.weights_history.columns:
                weights_hist[col] = [
                    round(v, 4) if not np.isnan(v) else 0
                    for v in result.weights_history[col].values
                ]
            weights_hist["_dates"] = [str(d)[:10] for d in result.weights_history.index]

        # Returns for distribution
        returns_list = result.returns.values.tolist()

        # Drawdown series
        cumret = result.equity_curve / result.equity_curve.iloc[0]
        running_max = cumret.cummax()
        drawdown = ((cumret - running_max) / running_max).values.tolist()

        # Rolling Sharpe
        rolling_sharpe = []
        if len(result.returns) > 63:
            rs = (
                result.returns.rolling(63).mean()
                / result.returns.rolling(63).std()
                * np.sqrt(252)
            )
            rolling_sharpe = [round(v, 4) if not np.isnan(v) else 0 for v in rs.values]

        final_result = {
            "performance": {
                k: round(v, 6) if isinstance(v, float) else v
                for k, v in perf.items()
            },
            "execution": {
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in exec_summary.items()
            },
            "equity_curve": {
                "dates": eq_dates,
                "values": [round(v, 2) for v in eq_vals],
            },
            "drawdown": [round(v, 6) for v in drawdown],
            "returns": [round(v, 6) for v in returns_list],
            "rolling_sharpe": rolling_sharpe,
            "weights_history": weights_hist,
            "trade_log": [
                {k: (round(v, 4) if isinstance(v, float) else str(v)[:10] if hasattr(v, 'strftime') else v)
                 for k, v in t.items()}
                for t in result.trade_log[-50:]
            ],
            "risk_history": result.risk_metrics_history[-50:] if result.risk_metrics_history else [],
        }

        with engine_lock:
            engine_state["status"] = "complete"
            engine_state["progress"] = 100
            engine_state["progress_msg"] = "Complete"
            engine_state["result"] = final_result
            engine_state["elapsed"] = round(time.time() - engine_state["start_time"], 1)

        logger.info(f"Backtest complete in {engine_state['elapsed']}s")

    except Exception as e:
        logger.exception("Backtest failed")
        with engine_lock:
            engine_state["status"] = "error"
            engine_state["error"] = str(e)
            engine_state["progress_msg"] = f"Error: {e}"


# ── Routes ───────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the dashboard HTML."""
    return send_file("dashboard.html")


@app.route("/api/status")
def api_status():
    """Get current engine status and live metrics."""
    with engine_lock:
        # Downsample equity history for live updates (max 500 points)
        eq = engine_state["equity_history"]
        dates = engine_state["date_history"]
        if len(eq) > 500:
            step = len(eq) // 500
            eq = eq[::step]
            dates = dates[::step]

        return jsonify({
            "status": engine_state["status"],
            "progress": engine_state["progress"],
            "progress_msg": engine_state["progress_msg"],
            "error": engine_state["error"],
            "config": engine_state["config"],
            "elapsed": round(time.time() - engine_state["start_time"], 1) if engine_state["start_time"] else 0,
            "current_day": engine_state["current_day"],
            "total_days": engine_state["total_days"],
            "equity_history": eq,
            "date_history": dates,
            "current_weights": engine_state["current_weights"],
            "risk_metrics": engine_state["risk_metrics"],
            "model_weights": engine_state["model_weights"],
            "recent_trades": engine_state["recent_trades"][-10:],
            "running_metrics": engine_state["running_metrics"],
        })


@app.route("/api/result")
def api_result():
    """Get final backtest results."""
    with engine_lock:
        if engine_state["result"] is None:
            return jsonify({"error": "No results available"}), 404
        return jsonify(engine_state["result"])


@app.route("/api/run", methods=["POST"])
def api_run():
    """Start a new backtest run."""
    with engine_lock:
        if engine_state["status"] == "running" or engine_state["status"] == "loading":
            return jsonify({"error": "Backtest already running"}), 409

    config_overrides = request.json or {}
    thread = threading.Thread(target=_run_backtest, args=(config_overrides,), daemon=True)
    thread.start()

    return jsonify({"status": "started"})


@app.route("/api/defaults")
def api_defaults():
    """Return default configuration values."""
    config = SystemConfig()
    return jsonify({
        "tickers": config.data.tickers,
        "years": config.data.years,
        "capital": 1_000_000,
        "method": config.portfolio.method,
        "methods": ["mean_variance", "risk_parity", "black_litterman"],
        "ensemble": config.alpha.ensemble_method,
        "ensembles": ["equal_weight", "inverse_vol", "performance_weighted"],
        "vol_target": config.risk.vol_target,
        "models": config.alpha.enabled_models,
        "all_models": [
            "ts_momentum", "xs_momentum", "momentum_vol_break",
            "ou_mean_reversion", "pairs_trading", "ml_alpha",
        ],
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quant Engine Web Server")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    print(f"\n  Quant Engine Dashboard: http://{args.host}:{args.port}\n")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
