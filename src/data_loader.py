"""
Load and validate historical BTC price data.
Compute log returns for volatility estimation.
"""

import pandas as pd
import numpy as np
from pathlib import Path


class PriceData:
    """Container for price data and derived quantities."""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize with price data.

        Args:
            df: DataFrame with columns ['timestamp', 'price']
                timestamp should be datetime-like
                price should be numeric
        """
        self.df = df.copy()
        self._validate()
        self._compute_log_returns()

    def _validate(self):
        """Validate data quality."""
        required_cols = ['timestamp', 'price']
        if not all(col in self.df.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")

        # Ensure timestamp is datetime
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])

        # Sort by timestamp
        self.df = self.df.sort_values('timestamp').reset_index(drop=True)

        # Check for NaN prices
        if self.df['price'].isna().any():
            raise ValueError("Price data contains NaN values")

        # Check for non-positive prices
        if (self.df['price'] <= 0).any():
            raise ValueError("Price data contains non-positive values")

    def _compute_log_returns(self):
        """Compute log returns: log(P_t / P_{t-1})"""
        self.df['log_return'] = np.log(self.df['price'] / self.df['price'].shift(1))

    def get_recent_returns(self, lookback_hours: float) -> np.ndarray:
        """
        Get log returns from the most recent lookback_hours.

        Args:
            lookback_hours: Number of hours to look back

        Returns:
            Array of log returns (excluding NaN from first observation)
        """
        if len(self.df) == 0:
            raise ValueError("No price data available")

        latest_time = self.df['timestamp'].iloc[-1]
        cutoff_time = latest_time - pd.Timedelta(hours=lookback_hours)

        recent_data = self.df[self.df['timestamp'] >= cutoff_time]
        returns = recent_data['log_return'].dropna().values

        if len(returns) == 0:
            raise ValueError(f"No data found in lookback window of {lookback_hours} hours")

        return returns

    def get_current_price(self) -> float:
        """Get the most recent price."""
        return float(self.df['price'].iloc[-1])

    def get_current_timestamp(self) -> pd.Timestamp:
        """Get the most recent timestamp."""
        return self.df['timestamp'].iloc[-1]

    def get_return_period_minutes(self) -> float:
        """
        Detect the actual time interval between observations.

        Returns:
            Average time period in minutes between consecutive observations

        Note:
            Uses median to be robust to occasional gaps in data
        """
        if len(self.df) < 2:
            raise ValueError("Need at least 2 observations to detect period")

        # Calculate time differences between consecutive rows
        time_diffs = self.df['timestamp'].diff().dropna()
        time_diffs = pd.to_timedelta(time_diffs)

        period_minutes = time_diffs.dt.total_seconds().median() / 60.0


        return period_minutes


def load_price_data(filepath: str | Path) -> PriceData:
    """
    Load BTC price data from CSV file.

    Expected format:
        - CSV with header
        - Columns: timestamp, price
        - timestamp: ISO format or parseable datetime string
        - price: numeric (USD)

    Args:
        filepath: Path to CSV file

    Returns:
        PriceData object with validated data and computed returns
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    df = pd.read_csv(filepath)
    return PriceData(df)
