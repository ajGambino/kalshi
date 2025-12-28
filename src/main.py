"""
Main entry point for BTC probability model.

Loads data, estimates volatility, outputs probability tables,
and optionally logs trades to CSV.
"""

import sys
import time
from pathlib import Path
from typing import List, Tuple
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.data_loader import load_price_data
from src.volatility import VolatilityEstimator
from src.probability_model import create_model, ProbabilityResult
from src.spot_provider import create_spot_provider
from src.trade_logger import log_trade

import numpy as np


# ------------------------
# Formatting helpers
# ------------------------

def format_currency(v: float) -> str:
    return f"${v:,.2f}"


def format_percent(v: float) -> str:
    return f"{v * 100:.2f}%"


# ------------------------
# Strike ladder
# ------------------------

def generate_strike_ladder(
    spot_price: float,
    increment: float,
    steps_above: int,
    steps_below: int
) -> List[float]:
    center = round(spot_price / increment) * increment
    return [
        center + i * increment
        for i in range(-steps_below, steps_above + 1)
        if center + i * increment > 0
    ]


# ------------------------
# Settlement time logic
# ------------------------

def choose_effective_settlement_mode(now_utc: datetime) -> str:
    """
    Kalshi availability rule:
    - 00:00–07:59 ET: only daily 5pm market available
    - 08:00+ ET: hourly markets available again (9am opens at 8am ET)
    """
    est = ZoneInfo("America/New_York")
    now_et = now_utc.astimezone(est)

    if 0 <= now_et.hour < 8:
        return "daily_5pm"
    return "hourly"


def get_next_hourly_settlement(now_utc: datetime) -> Tuple[datetime, float]:
    hour = now_utc.replace(minute=0, second=0, microsecond=0)
    settlement = hour + timedelta(hours=1)
    horizon = (settlement - now_utc).total_seconds() / 3600
    return settlement, horizon


def get_next_daily_5pm_settlement(now_utc: datetime) -> Tuple[datetime, float]:
    est = ZoneInfo("America/New_York")
    now_est = now_utc.astimezone(est)

    today_5pm = now_est.replace(hour=17, minute=0, second=0, microsecond=0)
    if now_est >= today_5pm:
        today_5pm += timedelta(days=1)

    settlement_utc = today_5pm.astimezone(timezone.utc)
    horizon = (settlement_utc - now_utc).total_seconds() / 3600
    return settlement_utc, horizon


# ------------------------
# Output table
# ------------------------

def print_results_table(
    results: List[ProbabilityResult],
    now_utc: datetime,
    settlement_utc: datetime,
    horizon_hours: float,
    spot_source: str,
    effective_mode: str
):
    est = ZoneInfo("America/New_York")

    print("\n" + "=" * 60)
    print(f"Settlement Reference:  {config.SETTLEMENT_REFERENCE}")
    print(f"Settlement Mode (config): {config.SETTLEMENT_MODE}")
    print(f"Settlement Mode (used):   {effective_mode}")
    print(f"Spot Proxy Used:       {spot_source}")
    print(f"Current Time:          {now_utc.astimezone(est).strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"Settlement Time:       {settlement_utc.astimezone(est).strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"Forecast Horizon:      {horizon_hours:.2f} hours")
    print("=" * 60)

    cp = results[0].current_price
    av = results[0].annual_volatility
    hv = results[0].horizon_volatility

    print(f"\nCurrent BTC Price: {format_currency(cp)}")
    print(f"Annual Volatility: {format_percent(av)}")
    print(f"Horizon Vol:       {format_percent(hv)}\n")

    print(f"{'Strike':<12}{'Linear %':<12}{'Log Dist':<10}{'Probability':<12}")
    print("-" * 46)

    for r in results:
        log_dist = np.log(r.strike / cp)

        print(
            f"{format_currency(r.strike):<12}"
            f"{r.distance_percent:+.2f}%{' ' * 3}"
            f"{log_dist:+.4f}{' ' * 3}"
            f"{format_percent(r.probability_above):<12}"
        )

    print()


# ------------------------
# Core model runner
# ------------------------

def run_model(
    data_file: str,
    strikes: List[float],
    spot_price: float,
    spot_source: str
):
    # Authoritative current time
    now_utc = datetime.now(timezone.utc)

    # Load historical candles
    price_data = load_price_data(data_file)
    last_candle_time = price_data.get_current_timestamp()
    last_candle_close = price_data.get_current_price()

    # Candle freshness diagnostics
    lag_sec = (now_utc - last_candle_time).total_seconds()
    print(f"\nSystem time UTC:     {now_utc}")
    print(f"Last candle UTC:     {last_candle_time}")
    print(f"Candle lag (sec):    {lag_sec:.1f}")

    # Spot–candle gap check
    gap_pct = abs(spot_price - last_candle_close) / last_candle_close * 100
    if gap_pct > config.MAX_SPOT_CANDLE_GAP_PCT:
        print("\nWARNING: Spot moved materially since last candle")
        print(f"  Spot:   {format_currency(spot_price)}")
        print(f"  Candle: {format_currency(last_candle_close)}")
        print(f"  Gap:    {gap_pct:.2f}%\n")

    # Settlement selection
    effective_mode = choose_effective_settlement_mode(now_utc)

    if effective_mode == "hourly":
        settlement, horizon = get_next_hourly_settlement(now_utc)
    else:
        settlement, horizon = get_next_daily_5pm_settlement(now_utc)


    # Volatility estimation
    returns = price_data.get_recent_returns(config.LOOKBACK_HOURS)
    rp_minutes = price_data.get_return_period_minutes()

    vol = VolatilityEstimator(config.LOOKBACK_HOURS)
    annual_vol, horizon_vol = vol.get_horizon_volatility(
        returns, rp_minutes, horizon
    )

    # Probability model
    model = create_model(config.RETURN_DISTRIBUTION)
    results = [
        model.calculate_strike_probability(
            current_price=spot_price,
            strike=k,
            annual_volatility=annual_vol,
            horizon_volatility=horizon_vol
        )
        for k in strikes
    ]

    print_results_table(results, now_utc, settlement, horizon, spot_source, effective_mode)


    # ------------------------
    # Optional trade logging
    # ------------------------

    results_by_strike = {r.strike: r for r in results}

    action = input("Log trade? (OPEN / skip): ").strip().upper()
    if action != "OPEN":
        return

    market = input("Market ID (e.g. BTC-HOURLY-2025-12-17-16): ").strip()
    strike = float(input("Strike: "))
    side = input("Side (YES / NO): ").strip().upper()
    cost = float(input("Total cost paid ($): "))
    max_payout = float(input("Max payout if correct ($): "))

    size = int(round(max_payout))
    price = cost / size
    if abs(size - max_payout) > 0.01:
        print("WARNING: max payout not integer — check Kalshi UI")

    notes = input("Notes (optional): ")

    r = results_by_strike[strike]
    model_prob = r.probability_above if side == "YES" else 1 - r.probability_above

    log_trade(
        trade_id=f"T{int(time.time())}",
        market=market,
        side=side,
        strike=strike,
        price=price,
        size=size,
        model_probability=model_prob,
        spot_price=spot_price,
        last_candle_close=last_candle_close,
        spot_candle_gap_pct=gap_pct,
        annual_vol=annual_vol,
        horizon_hours=horizon,
        settlement_time_utc=settlement,
        settlement_mode=effective_mode,
        event=action,
        notes=notes
    )


# ------------------------
# CLI entry
# ------------------------

def main():
    data_file = "data/btc_prices.csv"

    if not Path(data_file).exists():
        print("ERROR: data/btc_prices.csv not found")
        sys.exit(1)

    spot_provider = create_spot_provider(config.SPOT_SOURCE)
    spot_price, spot_source = spot_provider.get_spot_price()

    strikes = generate_strike_ladder(
        spot_price,
        config.STRIKE_INCREMENT,
        config.STRIKE_STEPS_ABOVE,
        config.STRIKE_STEPS_BELOW
    )

    run_model(
        data_file=data_file,
        strikes=strikes,
        spot_price=spot_price,
        spot_source=spot_source
    )


if __name__ == "__main__":
    main()
