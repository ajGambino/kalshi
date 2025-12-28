"""
Trade logging utilities for Kalshi BTC markets.

Logs trades to CSV with:
- Model probability
- Entry price
- Expected value (EV) at entry
- Realized PnL on close
- Realized vs Expected EV comparison
"""

from pathlib import Path
from datetime import datetime, timezone
import csv
from typing import Optional, Dict

from src.csv_schema import FIELDNAMES, validate_csv_header, validate_row_required_fields

# -------------------------------------------------
# Configuration
# -------------------------------------------------

TRADE_LOG_PATH = Path("trades/trade_log.csv")

# -------------------------------------------------
# OPEN / CLOSE logging
# -------------------------------------------------

def log_trade(
    *,
    trade_id: str,
    market: str,
    side: str,
    strike: float,
    price: float,
    size: int,
    model_probability: float,
    spot_price: float,
    last_candle_close: float,
    spot_candle_gap_pct: float,
    annual_vol: float,
    horizon_hours: float,
    settlement_time_utc,
    settlement_mode: str,
    event: str,
    notes: str = "",
    realized_pnl: Optional[float] = None,
):
    """
    Log a trade OPEN or CLOSE event.

    - OPEN: computes EV fields automatically
    - CLOSE: realized_pnl should be supplied, EV fields left blank
    """

    timestamp = datetime.now(timezone.utc).isoformat()

    # EV calculations (OPEN only)
    if event == "OPEN":
        ev_per_contract = model_probability - price
        ev_dollars = ev_per_contract * size
        edge_pct = ev_per_contract / price if price > 0 else 0.0
    else:
        ev_per_contract = None
        ev_dollars = None
        edge_pct = None

    row = {
        "trade_id": trade_id,
        "timestamp_utc": timestamp,
        "event": event,
        "market": market,
        "side": side,
        "strike": strike,
        "price": price,
        "outcome_price": "",  # Always blank on OPEN (exit price filled on CLOSE)
        "size": size,
        "model_probability": round(model_probability, 6) if event == "OPEN" else "",
        "ev_per_contract": round(ev_per_contract, 4) if ev_per_contract is not None else "",
        "ev_dollars": round(ev_dollars, 2) if ev_dollars is not None else "",
        "edge_pct": round(edge_pct, 4) if edge_pct is not None else "",
        "realized_pnl": round(realized_pnl, 2) if realized_pnl is not None else "",
        "spot_price": round(spot_price, 2) if event == "OPEN" else "",
        "last_candle_close": round(last_candle_close, 2) if event == "OPEN" else "",
        "spot_candle_gap_pct": round(spot_candle_gap_pct, 4) if event == "OPEN" else "",
        "annual_vol": round(annual_vol, 4) if event == "OPEN" else "",
        "horizon_hours": round(horizon_hours, 3) if event == "OPEN" else "",
        "settlement_time_utc": settlement_time_utc.isoformat(),
        "settlement_mode": settlement_mode,
        "notes": notes,
    }

    # Validate required fields before writing
    validate_row_required_fields(row, event)

    write_header = not TRADE_LOG_PATH.exists()
    TRADE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Validate header matches schema (prevent drift)
    validate_csv_header(TRADE_LOG_PATH)

    with TRADE_LOG_PATH.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="raise")
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"Trade {event} logged successfully.")


# -------------------------------------------------
# Post-trade analytics
# -------------------------------------------------

def compute_realized_vs_expected() -> Dict[str, float]:
    """
    Compute aggregate realized vs expected EV across all completed trades.

    Returns:
        dict with totals and ratios
    """

    if not TRADE_LOG_PATH.exists():
        raise FileNotFoundError("trade_log.csv not found")

    with TRADE_LOG_PATH.open(newline="") as f:
        rows = list(csv.DictReader(f))

    opens = {r["trade_id"]: r for r in rows if r["event"] == "OPEN"}
    closes = [r for r in rows if r["event"] == "CLOSE"]

    total_expected_ev = 0.0
    total_realized_pnl = 0.0
    completed_trades = 0

    for c in closes:
        tid = c["trade_id"]
        if tid not in opens:
            continue

        o = opens[tid]

        expected_ev = float(o["ev_dollars"])
        realized_pnl = float(c["realized_pnl"])

        total_expected_ev += expected_ev
        total_realized_pnl += realized_pnl
        completed_trades += 1

    return {
        "completed_trades": completed_trades,
        "total_expected_ev": round(total_expected_ev, 2),
        "total_realized_pnl": round(total_realized_pnl, 2),
        "realized_minus_expected": round(total_realized_pnl - total_expected_ev, 2),
    }
