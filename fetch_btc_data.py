"""
Fetch BTC-USD candle data from Coinbase Exchange and save to CSV.

This script uses Coinbase Exchange public API instead of Yahoo Finance to:
- Eliminate data delays (real-time exchange data)
- Ensure proper UTC timestamp handling (all timestamps are timezone-aware)
- Avoid time confusion by explicitly defining "now" as datetime.now(timezone.utc)

The script fetches OHLCV candles and extracts the close price for volatility estimation.
All timestamps in the output CSV are UTC-aware datetime objects.

Usage:
    python fetch_btc_data.py [--days DAYS] [--interval INTERVAL] [--output OUTPUT]

Examples:
    # Fetch last 7 days of 1-minute data (default)
    python fetch_btc_data.py

    # Fetch last 30 days of 1-minute data
    python fetch_btc_data.py --days 30

    # Fetch last 7 days of 5-minute data
    python fetch_btc_data.py --interval 5m

    # Save to custom location
    python fetch_btc_data.py --output data/my_btc_data.csv

Freshness Diagnostics:
    After fetching, the script displays:
    - Latest candle timestamp (UTC)
    - Current system time (UTC)
    - Candle lag in minutes

    This helps verify data freshness and detect any delays.
"""

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import requests


def fetch_btc_data(
    days: int = 7,
    interval: str = '1m'
) -> pd.DataFrame:
    """
    Fetch BTC-USD candle data from Coinbase Exchange public API.

    This function uses Coinbase's real-time exchange data instead of Yahoo Finance
    to eliminate data delays and ensure proper UTC timestamp handling.

    Args:
        days: Number of days of historical data to fetch
        interval: Data interval ('1m' or '5m')

    Returns:
        DataFrame with timezone-aware UTC timestamp and price (close) columns

    Notes:
        - Uses Coinbase Exchange public API (no authentication required)
        - All timestamps are UTC-aware datetime objects
        - Coinbase limits responses to 300 candles; multiple requests are batched automatically
        - "now" is always defined as datetime.now(timezone.utc) to avoid time confusion
    """
    # Map interval strings to Coinbase granularity (in seconds)
    INTERVAL_MAP = {
        '1m': 60,
        '5m': 300,
    }

    if interval not in INTERVAL_MAP:
        raise ValueError(
            f"Unsupported interval '{interval}'. "
            f"Supported intervals: {', '.join(INTERVAL_MAP.keys())}"
        )

    granularity = INTERVAL_MAP[interval]

    print(f"Fetching {days} days of {interval} BTC-USD data from Coinbase Exchange...")

    # Define "now" explicitly as UTC-aware datetime
    # This is the authoritative current time for freshness calculations
    now_utc = datetime.now(timezone.utc)
    start_utc = now_utc - timedelta(days=days)

    # Fetch candles from Coinbase (handles batching if needed)
    candles = _fetch_coinbase_candles(start_utc, now_utc, granularity)

    if not candles:
        raise ValueError(
            "No data returned from Coinbase Exchange API. "
            "Check network connection and API availability."
        )

    # Convert candles to DataFrame
    # Coinbase returns: [unix_time, low, high, open, close, volume]
    df = pd.DataFrame(candles, columns=['unix_time', 'low', 'high', 'open', 'close', 'volume'])

    # Convert Unix timestamps to UTC-aware datetime objects
    # This ensures all timestamps are explicitly timezone-aware
    df['timestamp'] = df['unix_time'].apply(
        lambda ts: datetime.fromtimestamp(ts, tz=timezone.utc)
    )

    # Use close price as the price
    df['price'] = df['close']

    # Keep only timestamp and price columns
    df = df[['timestamp', 'price']].copy()

    # Deduplicate candles (Coinbase can return overlapping candles between batches)
    df = df.drop_duplicates(subset='timestamp', keep='first')

    # Sort by timestamp (should already be sorted, but ensure it)
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Granularity safety check: verify candles have expected spacing
    expected_delta = pd.Timedelta(seconds=granularity)
    actual_deltas = df['timestamp'].diff().dropna()

    if len(actual_deltas) > 0:
        mode_delta = actual_deltas.mode()
        if len(mode_delta) > 0 and mode_delta[0] != expected_delta:
            print(f"WARNING: Candle spacing irregular. Expected {expected_delta}, most common: {mode_delta[0]}")

    # Basic data summary
    print(f"Fetched {len(df)} records")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Price range: ${df['price'].min():.2f} to ${df['price'].max():.2f}")

    # Freshness diagnostics
    # Compare latest candle timestamp with current system time (both UTC)
    latest_candle_utc = df['timestamp'].max()
    lag_seconds = (now_utc - latest_candle_utc).total_seconds()
    lag_minutes = lag_seconds / 60.0

    print("\n--- Freshness Diagnostics ---")
    print(f"Latest candle UTC: {latest_candle_utc}")
    print(f"System time UTC:   {now_utc}")
    print(f"Candle lag:        {lag_minutes:.1f} minutes")
    print("-----------------------------\n")

    return df


def _fetch_coinbase_candles(
    start_utc: datetime,
    end_utc: datetime,
    granularity: int
) -> List[List]:
    """
    Fetch candles from Coinbase Exchange API with automatic batching.

    Coinbase limits responses to 300 candles per request. This function
    automatically batches requests if the time range requires more candles.

    Args:
        start_utc: Start time (UTC-aware datetime)
        end_utc: End time (UTC-aware datetime)
        granularity: Candle size in seconds (60 for 1m, 300 for 5m)

    Returns:
        List of candles, each candle is [unix_time, low, high, open, close, volume]
    """
    COINBASE_API_URL = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
    MAX_CANDLES_PER_REQUEST = 300

    # Calculate total candles needed
    total_seconds = (end_utc - start_utc).total_seconds()
    total_candles_needed = int(total_seconds / granularity)

    all_candles = []

    # If we need more than 300 candles, batch the requests
    if total_candles_needed > MAX_CANDLES_PER_REQUEST:
        # Calculate how many requests we need
        num_requests = (total_candles_needed // MAX_CANDLES_PER_REQUEST) + 1

        # Split time range into chunks
        current_start = start_utc
        chunk_duration = timedelta(seconds=granularity * MAX_CANDLES_PER_REQUEST)

        for i in range(num_requests):
            current_end = min(current_start + chunk_duration, end_utc)

            candles = _fetch_single_coinbase_request(
                current_start, current_end, granularity, COINBASE_API_URL
            )
            all_candles.extend(candles)

            current_start = current_end

            if current_start >= end_utc:
                break
    else:
        # Single request is sufficient
        all_candles = _fetch_single_coinbase_request(
            start_utc, end_utc, granularity, COINBASE_API_URL
        )

    return all_candles


def _fetch_single_coinbase_request(
    start_utc: datetime,
    end_utc: datetime,
    granularity: int,
    api_url: str
) -> List[List]:
    """
    Make a single request to Coinbase Exchange API.

    Args:
        start_utc: Start time (UTC-aware datetime)
        end_utc: End time (UTC-aware datetime)
        granularity: Candle size in seconds
        api_url: Coinbase API endpoint URL

    Returns:
        List of candles from the API response
    """
    # Convert datetimes to ISO 8601 format (required by Coinbase API)
    start_iso = start_utc.isoformat()
    end_iso = end_utc.isoformat()

    params = {
        'start': start_iso,
        'end': end_iso,
        'granularity': granularity
    }

    try:
        response = requests.get(api_url, params=params, timeout=30)
        response.raise_for_status()  # Raise exception for 4xx/5xx errors

        candles = response.json()

        if not isinstance(candles, list):
            raise ValueError(f"Unexpected API response format: {type(candles)}")

        return candles

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to fetch data from Coinbase API: {e}") from e


def save_to_csv(df: pd.DataFrame, filepath: str | Path):
    """
    Save DataFrame to CSV.

    Args:
        df: DataFrame with timestamp and price columns
        filepath: Output CSV path
    """
    filepath = Path(filepath)

    # Create directory if it doesn't exist
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    df.to_csv(filepath, index=False)
    print(f"\nSaved to: {filepath}")
    print(f"File size: {filepath.stat().st_size / 1024:.1f} KB")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Fetch BTC-USD candle data from Coinbase Exchange and save to CSV',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fetch_btc_data.py
  python fetch_btc_data.py --days 30
  python fetch_btc_data.py --days 7 --interval 1m
  python fetch_btc_data.py --output data/my_btc_data.csv

Supported intervals:
  1m, 5m
  (Coinbase Exchange public API)
        """
    )

    parser.add_argument(
        '--days',
        type=int,
        default=7,
        help='Number of days of historical data to fetch (default: 7)'
    )

    parser.add_argument(
        '--interval',
        type=str,
        default='1m',
        choices=['1m', '5m'],
        help='Data interval (default: 1m)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/btc_prices.csv',
        help='Output CSV file path (default: data/btc_prices.csv)'
    )

    args = parser.parse_args()

    try:
        # Fetch data from Coinbase Exchange
        df = fetch_btc_data(
            days=args.days,
            interval=args.interval
        )

        # Save to CSV
        save_to_csv(df, args.output)

        print("\nSuccess! You can now run the probability model with:")
        print(f"  python src/main.py")

    except Exception as e:
        print(f"\nError: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
