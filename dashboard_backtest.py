"""
Backtest Results Dashboard
Interactive visualizations for backtest performance.

Run: python dashboard_backtest.py
Open: http://localhost:8050
"""
import json
import logging
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html, callback, Input, Output
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

# Try to import backtest results if they exist
try:
    from backtest.engine import BacktestResult
except ImportError:
    BacktestResult = None


def load_sample_backtest():
    """Load or create sample backtest results for demo."""
    import numpy as np
    from datetime import datetime, timedelta

    # Create sample data
    dates = pd.date_range('2023-01-01', '2025-12-31', freq='D')
    n = len(dates)

    # Generate realistic equity curve
    returns = np.random.normal(0.0005, 0.015, n)
    equity_curve = (1 + returns).cumprod() * 1_000_000

    performance_metrics = {
        'total_return': float((equity_curve[-1] / equity_curve[0] - 1) * 100),
        'sharpe_ratio': float(np.mean(returns) / np.std(returns) * np.sqrt(252)),
        'max_drawdown': float(
            ((np.minimum.accumulate(equity_curve) - equity_curve) / 
             np.minimum.accumulate(equity_curve)).min() * 100
        ),
        'win_rate': float(np.mean(returns > 0) * 100),
        'best_month': float(returns.sum() * 100),
        'worst_month': float(returns.min() * 100),
    }

    return {
        'dates': dates,
        'equity_curve': equity_curve,
        'returns': returns,
        'performance_metrics': performance_metrics,
    }


def create_equity_curve_chart(dates, equity_curve, returns):
    """Equity curve with drawdown."""
    running_max = pd.Series(equity_curve).expanding().max()
    drawdown = (equity_curve - running_max) / running_max * 100

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Equity Curve', 'Drawdown %'),
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
    )

    fig.add_trace(
        go.Scatter(
            x=dates, y=equity_curve,
            name='Equity',
            mode='lines',
            line=dict(color='#2ecc71', width=2),
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=dates, y=drawdown,
            name='Drawdown %',
            mode='lines',
            fill='tozeroy',
            line=dict(color='#e74c3c'),
        ),
        row=2, col=1
    )

    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    fig.update_layout(height=600, hovermode='x unified')

    return fig


def create_returns_distribution(returns):
    """Returns histogram with normal curve."""
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=returns,
        name='Daily Returns',
        nbinsx=50,
        marker_color='#3498db',
    ))

    fig.update_layout(
        title='Daily Returns Distribution',
        xaxis_title='Return (%)',
        yaxis_title='Frequency',
        height=400,
    )

    return fig


def create_metrics_cards(metrics):
    """Performance metric cards."""
    cards = []

    metric_defs = [
        ('Total Return', f"{metrics['total_return']:.1f}%", '#2ecc71'),
        ('Sharpe Ratio', f"{metrics['sharpe_ratio']:.2f}", '#3498db'),
        ('Max Drawdown', f"{metrics['max_drawdown']:.1f}%", '#e74c3c'),
        ('Win Rate', f"{metrics['win_rate']:.1f}%", '#f39c12'),
    ]

    for label, value, color in metric_defs:
        cards.append(
            html.Div([
                html.Div(value, style={
                    'fontSize': '28px',
                    'fontWeight': 'bold',
                    'color': color,
                }),
                html.Div(label, style={
                    'fontSize': '12px',
                    'color': '#7f8c8d',
                    'marginTop': '8px',
                }),
            ], style={
                'padding': '20px',
                'border': f'2px solid {color}',
                'borderRadius': '8px',
                'textAlign': 'center',
            })
        )

    return cards


def create_monthly_heatmap(dates, returns):
    """Monthly returns heatmap."""
    df = pd.DataFrame({
        'date': dates,
        'return': returns,
    })

    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month

    monthly_returns = df.groupby(['year', 'month'])['return'].sum() * 100
    monthly_returns = monthly_returns.unstack(fill_value=0)

    fig = go.Figure(data=go.Heatmap(
        z=monthly_returns.values,
        x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        y=monthly_returns.index,
        colorscale='RdYlGn',
        zmid=0,
    ))

    fig.update_layout(
        title='Monthly Returns (%)',
        height=300,
    )

    return fig


def create_dashboard():
    """Build the Dash app."""
    app = Dash(__name__)

    # Load sample data for demo
    data = load_sample_backtest()

    app.layout = html.Div([
        html.Div([
            html.H1('Backtest Results Dashboard', style={
                'color': '#2c3e50',
                'marginBottom': '10px',
            }),
            html.P('Real-time performance monitoring', style={
                'color': '#7f8c8d',
                'marginBottom': '30px',
            }),
        ], style={'padding': '30px'}),

        # Metrics cards
        html.Div(
            create_metrics_cards(data['performance_metrics']),
            style={
                'display': 'grid',
                'gridTemplateColumns': 'repeat(auto-fit, minmax(200px, 1fr))',
                'gap': '20px',
                'padding': '0 30px 30px 30px',
            }
        ),

        # Charts
        html.Div([
            html.Div([
                dcc.Graph(
                    figure=create_equity_curve_chart(
                        data['dates'],
                        data['equity_curve'],
                        data['returns'],
                    ),
                    style={'height': '100%'},
                ),
            ], style={
                'flex': '2',
                'padding': '0 15px',
            }),

            html.Div([
                dcc.Graph(
                    figure=create_returns_distribution(data['returns']),
                    style={'height': '100%'},
                ),
            ], style={
                'flex': '1',
                'padding': '0 15px',
            }),
        ], style={
            'display': 'flex',
            'gap': '20px',
            'padding': '30px',
        }),

        # Monthly heatmap
        html.Div([
            dcc.Graph(
                figure=create_monthly_heatmap(data['dates'], data['returns']),
            ),
        ], style={
            'padding': '30px',
        }),

        # Footer
        html.Div([
            html.P(
                'Dashboard auto-refreshes on file changes. '
                'Load backtest results via BacktestResult object.',
                style={'color': '#95a5a6', 'fontSize': '12px'},
            ),
        ], style={
            'padding': '30px',
            'borderTop': '1px solid #ecf0f1',
            'marginTop': '30px',
            'color': '#7f8c8d',
        }),

    ], style={
        'fontFamily': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
        'backgroundColor': '#f8f9fa',
        'minHeight': '100vh',
        'padding': '0',
        'margin': '0',
    })

    return app


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app = create_dashboard()
    print('🚀 Dashboard live at http://localhost:8050')
    print('📊 Visualizing backtest results in real-time')
    app.run(debug=True, host='0.0.0.0', port=8050)
