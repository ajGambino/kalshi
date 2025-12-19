"""
Realized volatility calculations.

Phase 1: Simple historical volatility from recent returns.
Phase 2: Can extend to GARCH, exponential weighting, etc.
"""

import sys
from pathlib import Path

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import config


def calculate_realized_volatility(
    returns: np.ndarray,
    return_period_minutes: float
) -> float:
    """
    Calculate annualized realized volatility from log returns.

    Uses the actual observed period between returns (e.g., 5 minutes)
    to correctly scale volatility.

    Formula:
        σ_period = std(returns)
        periods_per_year = (365.25 * 24 * 60) / return_period_minutes
        σ_annual = σ_period * sqrt(periods_per_year)

    Args:
        returns: Array of log returns
        return_period_minutes: Time period between returns in minutes

    Returns:
        Annualized volatility (as decimal, not percentage)
    """
    if len(returns) < 2:
        raise ValueError("Need at least 2 returns to calculate volatility")

    if return_period_minutes <= 0:
        raise ValueError("Return period must be positive")

    # Standard deviation of returns (using ddof=1 for sample std)
    period_vol = np.std(returns, ddof=1)

    # Calculate periods per year based on actual data frequency
    minutes_per_year = 365.25 * 24 * 60
    periods_per_year = minutes_per_year / return_period_minutes

    # Annualize using square-root-of-time rule
    annual_vol = period_vol * np.sqrt(periods_per_year)

    return annual_vol


def scale_volatility_to_horizon(
    annual_vol: float,
    horizon_hours: float
) -> float:
    """
    Scale annualized volatility to a specific forecast horizon.

    Uses square-root-of-time scaling:
        σ_horizon = σ_annual * sqrt(horizon_hours / hours_per_year)

    Args:
        annual_vol: Annualized volatility (decimal)
        horizon_hours: Forecast horizon in hours

    Returns:
        Volatility scaled to the horizon (decimal)
    """
    hours_per_year = 365.25 * 24
    time_fraction = horizon_hours / hours_per_year
    horizon_vol = annual_vol * np.sqrt(time_fraction)

    return horizon_vol


class VolatilityEstimator:
    """
    Encapsulates volatility estimation logic.

    Phase 1: Uses simple realized volatility.
    Phase 2: Can swap in GARCH, exponential weighting, regime models, etc.
    """

    def __init__(
        self,
        lookback_hours: float = config.LOOKBACK_HOURS
    ):
        """
        Initialize volatility estimator.

        Args:
            lookback_hours: Window for historical volatility calculation
        """
        self.lookback_hours = lookback_hours

    def estimate_volatility(
        self,
        returns: np.ndarray,
        return_period_minutes: float
    ) -> float:
        """
        Estimate annualized volatility from returns.

        Args:
            returns: Historical log returns
            return_period_minutes: Time period between returns in minutes

        Returns:
            Annualized volatility (decimal)
        """
        return calculate_realized_volatility(returns, return_period_minutes)

    def get_horizon_volatility(
        self,
        returns: np.ndarray,
        return_period_minutes: float,
        horizon_hours: float
    ) -> tuple[float, float]:
        """
        Estimate volatility and scale to forecast horizon.

        Args:
            returns: Historical log returns
            return_period_minutes: Time period between returns in minutes
            horizon_hours: Forecast horizon in hours

        Returns:
            Tuple of (annualized_vol, horizon_vol)
        """
        annual_vol = self.estimate_volatility(returns, return_period_minutes)
        horizon_vol = scale_volatility_to_horizon(annual_vol, horizon_hours)

        return annual_vol, horizon_vol
