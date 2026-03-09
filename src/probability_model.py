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


class StudentTModel:
    """
    Student-t price model with fat tails and zero drift.

    Assumptions:
        - Log returns follow a Student-t distribution (heavier tails than normal)
        - Zero mean (no directional bias)
        - Degrees of freedom parameter controls tail thickness
        - Variance is matched to Gaussian model (only tail shape differs)

    Model:
        log(S_T / S_0) ~ t(df) scaled to match horizon volatility

        where:
            S_T = price at time T
            S_0 = current price
            df = degrees of freedom (lower = fatter tails)
            Variance matched to Gaussian via scale adjustment

    Why Student-t?
        - Captures fat tails observed in crypto returns
        - More realistic probability for extreme moves
        - Reduces to Gaussian as df → ∞
        - Common in quantitative finance for modeling rare events

    Degrees of freedom (df) interpretation:
        - df = 3: Very fat tails (high kurtosis)
        - df = 5: Moderately fat tails (default for crypto)
        - df = 10: Light fat tails
        - df → ∞: Converges to Gaussian

    Variance scaling:
        Student-t variance = df / (df - 2) for df > 2
        To match Gaussian variance σ², we scale:
            t_scale = σ√T × sqrt((df - 2) / df)
        This ensures only tail behavior differs, not overall dispersion.

    Probability calculation:
        P(S_T > K) = P(log(S_T / S_0) > log(K / S_0))
                   = P(T > log(K / S_0) / t_scale)
                   = 1 - F_t(log(K / S_0) / t_scale; df)

        where F_t is the Student-t CDF with df degrees of freedom
    """

    def __init__(self, drift: float = config.DRIFT_MEAN, df: float = config.STUDENT_T_DF):
        """
        Initialize Student-t model.

        Args:
            drift: Mean of log returns (Phase 1: 0.0)
            df: Degrees of freedom (must be > 2 for finite variance)

        Raises:
            ValueError: If df <= 2
        """
        if df <= 2:
            raise ValueError(
                f"Student-t degrees of freedom must be > 2 for finite variance, got {df}"
            )
        self.drift = drift
        self.df = df

    def probability_above_strike(
        self,
        current_price: float,
        strike: float,
        horizon_volatility: float
    ) -> float:
        """
        Calculate probability that price is above strike at horizon.

        Uses Student-t distribution with variance-matched scaling.

        Args:
            current_price: Current spot price (S_0)
            strike: Strike price (K)
            horizon_volatility: Volatility scaled to horizon (σ√T)

        Returns:
            Probability that S_T > K

        Raises:
            ValueError: If volatility or prices are invalid
        """
        if horizon_volatility <= 0:
            raise ValueError("Horizon volatility must be positive")

        if current_price <= 0 or strike <= 0:
            raise ValueError("Prices must be positive")

        # Log-moneyness: log(K / S_0)
        log_moneyness = np.log(strike / current_price)

        # Scale adjustment to match Gaussian variance
        # Student-t with df has variance df/(df-2), so we scale down
        t_scale = horizon_volatility * np.sqrt((self.df - 2) / self.df)

        # Standardized value for Student-t CDF
        # d = (drift * T - log_moneyness) / t_scale
        # For zero drift: d = -log_moneyness / t_scale
        d = (self.drift - log_moneyness) / t_scale

        # P(S_T > K) = F_t(d; df)
        prob_above = stats.t.cdf(d, df=self.df)

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


def fit_student_t_params(returns: np.ndarray) -> dict:
    """
    Estimate Student-t parameters from historical log returns.

    This is a research/calibration tool. It fits a Student-t distribution
    to observed returns to estimate optimal degrees of freedom.

    Use this to:
        - Calibrate STUDENT_T_DF in config.py
        - Compare fitted vs assumed parameters
        - Validate zero-drift assumption (check if loc ≈ 0)

    Args:
        returns: Array of log returns (e.g., log(S_t / S_{t-1}))

    Returns:
        Dictionary with fitted parameters:
            - df: Degrees of freedom (tail thickness)
            - loc: Location parameter (estimated mean)
            - scale: Scale parameter (estimated std dev)

    Example:
        >>> log_returns = np.diff(np.log(prices))
        >>> params = fit_student_t_params(log_returns)
        >>> print(f"Fitted df: {params['df']:.2f}")
        >>> print(f"Fitted mean: {params['loc']:.6f} (should be ≈ 0)")
        >>> print(f"Fitted scale: {params['scale']:.6f}")
    """
    if len(returns) < 10:
        raise ValueError("Need at least 10 returns to fit Student-t distribution")

    # scipy.stats.t.fit() returns (df, loc, scale)
    df, loc, scale = stats.t.fit(returns)

    return {
        "df": df,
        "loc": loc,
        "scale": scale
    }


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
    elif distribution == 'student_t':
        return StudentTModel()
    else:
        raise ValueError(
            f"Unsupported distribution: {distribution}. "
            f"Supported: 'gaussian', 'student_t'"
        )
