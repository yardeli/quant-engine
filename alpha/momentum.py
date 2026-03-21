"""
Momentum Alpha Models.

Implements time-series and cross-sectional momentum strategies
as described in Moskowitz, Ooi & Pedersen (2012) "Time Series Momentum"
and Jegadeesh & Titman (1993) "Returns to Buying Winners and Selling Losers".

Mathematical Intuition:
    - Time-series momentum: assets that have gone up tend to keep going up
      (and vice versa). Signal = sign(past return) * vol-scaled position.
    - Cross-sectional momentum: long top decile, short bottom decile of
      past returns. Captures relative outperformance.

When it works: Trending markets, macro regime shifts, slow information diffusion.
When it fails: Momentum crashes (sharp reversals), choppy/mean-reverting markets.
"""
import numpy as np
import pandas as pd

from alpha.base import AlphaModel
from config import AlphaConfig
from data.feed import DataFeed


class TimeSeriesMomentum(AlphaModel):
    """
    Moskowitz-style time-series momentum.

    For each asset independently:
        signal_i = sign(r_{t-slow, t}) * (vol_target / realized_vol_i)

    This is the "trend following" signal used by CTAs and managed futures funds.
    """

    def __init__(self, config: AlphaConfig):
        super().__init__(config, name="ts_momentum")

    def generate_signals(self, data: DataFeed, features: dict[str, pd.DataFrame]) -> pd.DataFrame:
        prices = data.prices
        returns = data.returns

        # Compute lookback returns
        fast_ret = prices.pct_change(self.config.momentum_fast)
        slow_ret = prices.pct_change(self.config.momentum_slow)

        # Blend fast and slow signals using continuous returns (not sign)
        # Normalize by rolling std to get z-score-like signals
        fast_vol = returns.rolling(self.config.momentum_fast).std() * np.sqrt(self.config.momentum_fast)
        slow_vol = returns.rolling(self.config.momentum_slow).std() * np.sqrt(self.config.momentum_slow)
        fast_z = fast_ret / (fast_vol + 1e-8)
        slow_z = slow_ret / (slow_vol + 1e-8)
        blended = 0.5 * fast_z.clip(-2, 2) + 0.5 * slow_z.clip(-2, 2)

        # Volatility scaling: target a specific vol per asset
        realized_vol = returns.rolling(63).std() * np.sqrt(252)
        vol_scalar = self.config.momentum_vol_target / (realized_vol + 1e-8)
        vol_scalar = vol_scalar.clip(0.1, 3.0)  # Safety bounds

        signals = blended * vol_scalar

        return self._clip_signals(signals)


class CrossSectionalMomentum(AlphaModel):
    """
    Jegadeesh-Titman cross-sectional momentum.

    Rank assets by past 12-month return (skipping most recent month
    to avoid short-term reversal). Go long top quintile, short bottom quintile.

    The skip-month is critical — without it, the strategy picks up
    short-term reversal instead of momentum.
    """

    def __init__(self, config: AlphaConfig):
        super().__init__(config, name="xs_momentum")

    def generate_signals(self, data: DataFeed, features: dict[str, pd.DataFrame]) -> pd.DataFrame:
        prices = data.prices

        # Multi-horizon momentum: blend 12-1, 6-1, and 3-1 month lookbacks
        ret_1m = prices.pct_change(21)

        ret_12m = prices.pct_change(252)
        mom_12_1 = ret_12m - ret_1m  # 12-1 momentum

        ret_6m = prices.pct_change(126)
        mom_6_1 = ret_6m - ret_1m    # 6-1 momentum

        ret_3m = prices.pct_change(63)
        mom_3_1 = ret_3m - ret_1m    # 3-1 momentum

        # Blend with higher weight on medium-term (most robust empirically)
        momentum_signal = 0.25 * self._rank_normalize(mom_12_1) + \
                         0.50 * self._rank_normalize(mom_6_1) + \
                         0.25 * self._rank_normalize(mom_3_1)

        signals = self._rank_normalize(momentum_signal)

        return self._clip_signals(signals)


class MomentumWithVolBreak(AlphaModel):
    """
    Enhanced momentum that reduces exposure during volatility spikes.

    Intuition: momentum strategies crash during volatility regime changes.
    By scaling down when vol is elevated, we avoid the worst drawdowns.

    This is how most sophisticated CTAs manage their trend-following books.
    """

    def __init__(self, config: AlphaConfig):
        super().__init__(config, name="momentum_vol_break")
        self._ts_mom = TimeSeriesMomentum(config)

    def generate_signals(self, data: DataFeed, features: dict[str, pd.DataFrame]) -> pd.DataFrame:
        # Base momentum signal
        base_signal = self._ts_mom.generate_signals(data, features)

        # Compute volatility regime indicator
        returns = data.returns
        short_vol = returns.rolling(10).std() * np.sqrt(252)
        long_vol = returns.rolling(63).std() * np.sqrt(252)

        # Vol ratio > 1 means vol is elevated relative to recent history
        vol_ratio = short_vol / (long_vol + 1e-8)

        # Scale down when vol is elevated (smooth scaling, not binary)
        vol_brake = (1.0 / vol_ratio).clip(0.2, 1.0)

        signals = base_signal * vol_brake

        return self._clip_signals(signals)
