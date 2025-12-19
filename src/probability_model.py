"""
Probability models for BTC price outcomes.

Phase 1: Gaussian (normal) distribution with zero drift.
Phase 2: Student-t, empirical distributions, etc.
"""

import sys
from pathlib import Path

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from scipy import stats
from dataclasses import dataclass
import config


@dataclass
class ProbabilityResult:
    """Result of probability calculation for a single strike."""

    strike: float
    current_price: float
    distance_dollars: float
    distance_percent: float
    annual_volatility: float
    horizon_volatility: float
    probability_above: float

    @property
    def probability_below(self) -> float:
        """Probability of being below strike."""
        return 1.0 - self.probability_above


class GaussianModel:
    """
    Gaussian (log-normal) price model with zero drift.

    Assumptions:
        - Log returns are normally distributed
        - Zero mean (no directional bias)
        - Constant volatility over horizon
        - Continuous time approximation

    Model:
        log(S_T / S_0) ~ N(0, σ²T)

        where:
            S_T = price at time T
            S_0 = current price
            σ = volatility (annualized)
            T = time horizon (in years)

    Probability calculation:
        P(S_T > K) = P(log(S_T / S_0) > log(K / S_0))
                   = P(Z > log(K / S_0) / (σ√T))
                   = 1 - Φ(log(K / S_0) / (σ√T))

        where Φ is the standard normal CDF
    """

    def __init__(self, drift: float = config.DRIFT_MEAN):
        """
        Initialize Gaussian model.

        Args:
            drift: Mean of log returns (Phase 1: 0.0)
        """
        self.drift = drift

    def probability_above_strike(
        self,
        current_price: float,
        strike: float,
        horizon_volatility: float
    ) -> float:
        """
        Calculate probability that price is above strike at horizon.

        Args:
            current_price: Current spot price (S_0)
            strike: Strike price (K)
            horizon_volatility: Volatility scaled to horizon (σ√T)

        Returns:
            Probability that S_T > K
        """
        if horizon_volatility <= 0:
            raise ValueError("Horizon volatility must be positive")

        if current_price <= 0 or strike <= 0:
            raise ValueError("Prices must be positive")

        # Log-moneyness: log(K / S_0)
        log_moneyness = np.log(strike / current_price)

        # Adjusted for drift (Phase 1: drift = 0)
        # d = (drift * T - log_moneyness) / (σ√T)
        # For zero drift: d = -log_moneyness / (σ√T)
        d = (self.drift - log_moneyness) / horizon_volatility

        # P(S_T > K) = Φ(d)
        prob_above = stats.norm.cdf(d)

        return prob_above

    def calculate_strike_probability(
        self,
        current_price: float,
        strike: float,
        annual_volatility: float,
        horizon_volatility: float
    ) -> ProbabilityResult:
        """
        Calculate full probability result for a strike.

        Args:
            current_price: Current spot price
            strike: Strike price
            annual_volatility: Annualized volatility (for display)
            horizon_volatility: Volatility scaled to horizon (for calculation)

        Returns:
            ProbabilityResult with all relevant information
        """
        prob_above = self.probability_above_strike(
            current_price,
            strike,
            horizon_volatility
        )

        distance_dollars = strike - current_price
        distance_percent = (distance_dollars / current_price) * 100

        return ProbabilityResult(
            strike=strike,
            current_price=current_price,
            distance_dollars=distance_dollars,
            distance_percent=distance_percent,
            annual_volatility=annual_volatility,
            horizon_volatility=horizon_volatility,
            probability_above=prob_above
        )


def create_model(distribution: str = config.RETURN_DISTRIBUTION):
    """
    Factory function to create probability models.

    Args:
        distribution: Model type ('gaussian', 'student_t', etc.)

    Returns:
        Probability model instance

    Raises:
        ValueError: If distribution type is not supported
    """
    if distribution == 'gaussian':
        return GaussianModel()
    else:
        raise ValueError(
            f"Unsupported distribution: {distribution}. "
            f"Phase 1 only supports 'gaussian'"
        )
