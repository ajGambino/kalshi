"""
Spot price provider for BTC probability model.

Kalshi BTC hourly contracts settle on:
    CF Benchmarks Bitcoin Real-Time Index (BRTI)
    60-second average of constituent exchanges

This module provides spot price proxies that approximate BRTI.

Supported sources:
    1. Manual input (CLI/override)
    2. Coinbase BTC-USD public ticker (no authentication required)

Note: These are PROXIES for BRTI, not the official settlement reference.
Coinbase typically has < 0.1% basis vs BRTI in normal market conditions.

Future: Could add Kraken, Binance, or other BRTI-correlated sources.
"""

import sys
from pathlib import Path
from typing import Tuple
from abc import ABC, abstractmethod

# Add parent directory to path for requests import
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import requests
except ImportError:
    print("Warning: 'requests' library not found. Install with: pip install requests")
    requests = None


class SpotProvider(ABC):
    """Abstract base class for BTC spot price providers."""

    @abstractmethod
    def get_spot_price(self) -> Tuple[float, str]:
        """
        Fetch current BTC spot price.

        Returns:
            Tuple of (price, source_description)
            Example: (85432.50, "Coinbase BTC-USD public ticker")

        Raises:
            Exception: If price cannot be fetched
        """
        pass


class ManualSpotProvider(SpotProvider):
    """
    User-provided spot price (manual input or CLI override).

    Use this for:
        - Testing specific scenarios
        - Backtesting with known historical prices
        - When market data is unavailable
    """

    def __init__(self, price: float):
        """
        Initialize with manual price.

        Args:
            price: BTC spot price in USD

        Raises:
            ValueError: If price is not positive
        """
        if price <= 0:
            raise ValueError(f"Price must be positive, got {price}")
        self.price = price

    def get_spot_price(self) -> Tuple[float, str]:
        """Return the manually-specified price."""
        return (self.price, "Manual input")


class CoinbaseSpotProvider(SpotProvider):
    """
    Fetch Coinbase BTC-USD spot price via public API.

    Uses Coinbase's public ticker endpoint (no authentication required):
        GET https://api.coinbase.com/v2/prices/BTC-USD/spot

    Response format:
        {
            "data": {
                "base": "BTC",
                "currency": "USD",
                "amount": "43250.00"
            }
        }

    Note:
        - This is a PROXY for BRTI, not the official settlement reference
        - Typically < 0.1% basis vs BRTI in normal conditions
        - Good for production use as BRTI approximation
        - Free, no rate limits for reasonable usage
    """

    API_URL = "https://api.coinbase.com/v2/prices/BTC-USD/spot"
    TIMEOUT_SECONDS = 10

    def get_spot_price(self) -> Tuple[float, str]:
        """
        Fetch current BTC-USD spot price from Coinbase.

        Returns:
            Tuple of (price, source_description)

        Raises:
            RuntimeError: If requests library not installed
            Exception: If API call fails or returns invalid data
        """
        if requests is None:
            raise RuntimeError(
                "requests library required for Coinbase spot provider. "
                "Install with: pip install requests"
            )

        try:
            # Fetch from Coinbase public API
            response = requests.get(
                self.API_URL,
                timeout=self.TIMEOUT_SECONDS,
                headers={"User-Agent": "BTC-Probability-Model/1.0"}
            )

            # Check for HTTP errors
            response.raise_for_status()

            # Parse JSON response
            data = response.json()

            # Extract price from nested structure
            if "data" not in data or "amount" not in data["data"]:
                raise ValueError(f"Unexpected Coinbase API response format: {data}")

            price_str = data["data"]["amount"]
            price = float(price_str)

            if price <= 0:
                raise ValueError(f"Invalid price from Coinbase: {price}")

            return (price, "Coinbase BTC-USD public ticker")

        except requests.exceptions.Timeout:
            raise Exception(
                f"Coinbase API request timed out after {self.TIMEOUT_SECONDS} seconds"
            )
        except requests.exceptions.ConnectionError:
            raise Exception("Failed to connect to Coinbase API. Check internet connection.")
        except requests.exceptions.HTTPError as e:
            raise Exception(f"Coinbase API HTTP error: {e}")
        except (KeyError, ValueError, TypeError) as e:
            raise Exception(f"Failed to parse Coinbase API response: {e}")


def create_spot_provider(
    source: str = "coinbase",
    manual_price: float | None = None
) -> SpotProvider:
    """
    Factory function to create spot price provider.

    Args:
        source: Provider type ('coinbase' or 'manual')
        manual_price: Required if source='manual'

    Returns:
        SpotProvider instance

    Raises:
        ValueError: If invalid source or missing manual_price

    Example:
        >>> provider = create_spot_provider(source='coinbase')
        >>> price, source_name = provider.get_spot_price()
        >>> print(f"BTC: ${price:,.2f} from {source_name}")
        BTC: $85,432.50 from Coinbase BTC-USD public ticker
    """
    source = source.lower()

    if source == "manual":
        if manual_price is None:
            raise ValueError("manual_price required when source='manual'")
        return ManualSpotProvider(manual_price)

    elif source == "coinbase":
        return CoinbaseSpotProvider()

    else:
        raise ValueError(
            f"Unknown spot source: '{source}'. "
            f"Supported: 'coinbase', 'manual'"
        )
