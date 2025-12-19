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

# -------------------------------------------------
# Configuration
# -------------------------------------------------

TRADE_LOG_PATH = Path("trades/trade_log.csv")

FIELDNAMES = [
    "trade_id",
    "timestamp_utc",
    "event",                   # OPEN or CLOSE
    "market",
    "side",                    # YES / NO
    "strike",
    "price",                   # entry price (OPEN) or outcome price (CLOSE)
    "size",                    # number of contracts
    "model_probability",
    "ev_per_contract",
    "ev_dollars",
    "edge_pct",
    "realized_pnl",            # populated on CLOSE
    "spot_price",
    "last_candle_close",
    "spot_candle_gap_pct",
    "annual_vol",
    "horizon_hours",
    "settlement_time_utc",
    "settlement_mode",
    "notes",
]

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
        "size": size,
        "model_probability": round(model_probability, 6) if event == "OPEN" else "",
        "ev_per_contract": round(ev_per_contract, 4) if ev_per_contract is not None else None,
        "ev_dollars": round(ev_dollars, 2) if ev_dollars is not None else None,
        "edge_pct": round(edge_pct, 4) if edge_pct is not None else None,
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

    write_header = not TRADE_LOG_PATH.exists()
    TRADE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    with TRADE_LOG_PATH.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"Trade {event} logged successfully.")


# -------------------------------------------------
# CLOSE helper (minimal inputs)
# -------------------------------------------------

def log_close(
    trade_id: str,
    outcome_price: float,
    notes: str = ""
):
    """
    Log a CLOSE event for an existing trade.

    Args:
        trade_id: Existing trade_id from OPEN row
        outcome_price: 1.0 if ITM, 0.0 if OTM, 0.5 if refund
        notes: Optional notes
    """

    if not TRADE_LOG_PATH.exists():
        raise FileNotFoundError("trade_log.csv not found")

    # Load all trades
    with TRADE_LOG_PATH.open(newline="") as f:
        rows = list(csv.DictReader(f))

    open_trade = next(
        (r for r in rows if r["trade_id"] == trade_id and r["event"] == "OPEN"),
        None
    )

    if open_trade is None:
        raise ValueError(f"No OPEN trade found for trade_id={trade_id}")

    size = int(open_trade["size"])
    entry_price = float(open_trade["price"])

    # Kalshi PnL
    realized_pnl = size * (outcome_price - entry_price)

    log_trade(
        trade_id=trade_id,
        market=open_trade["market"],
        side=open_trade["side"],
        strike=float(open_trade["strike"]),
        price=outcome_price,
        size=size,
        model_probability=0.0,        # unused on CLOSE
        spot_price=0.0,
        last_candle_close=0.0,
        spot_candle_gap_pct=0.0,
        annual_vol=0.0,
        horizon_hours=0.0,
        settlement_time_utc=datetime.fromisoformat(open_trade["settlement_time_utc"]),
        settlement_mode=open_trade["settlement_mode"],
        event="CLOSE",
        realized_pnl=realized_pnl,
        notes=notes,
    )

    print(f"Trade {trade_id} CLOSED | PnL: {realized_pnl:+.2f}")


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
