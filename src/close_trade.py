"""
CLI utility to close an existing Kalshi trade.

- Auto-fills CLOSE row from OPEN row
- Computes realized PnL
- Uses the same CSV schema as trade_logger.py
"""

from pathlib import Path
from datetime import datetime, timezone
import csv
from typing import Any, Dict

TRADE_LOG_PATH = Path("trades/trade_log.csv")

FIELDNAMES = [
    "trade_id",
    "timestamp_utc",
    "event",
    "market",
    "side",
    "strike",
    "price",
    "size",
    "model_probability",
    "ev_per_contract",
    "ev_dollars",
    "edge_pct",
    "realized_pnl",
    "spot_price",
    "last_candle_close",
    "spot_candle_gap_pct",
    "annual_vol",
    "horizon_hours",
    "settlement_time_utc",
    "settlement_mode",
    "notes",
]


def main():
    if not TRADE_LOG_PATH.exists():
        raise FileNotFoundError("trade_log.csv not found")

    trade_id = input("Trade ID: ").strip()

    outcome_raw = input("Outcome (WIN / LOSS / REFUND): ").strip().upper()
    if outcome_raw not in {"WIN", "LOSS", "REFUND"}:
        raise ValueError("Outcome must be WIN, LOSS, or REFUND")

    outcome_price = {
        "WIN": 1.0,
        "LOSS": 0.0,
        "REFUND": 0.5,
    }[outcome_raw]

    notes = input("Notes (optional): ").strip()

    # Load trades
    with TRADE_LOG_PATH.open(newline="") as f:
        rows = list(csv.DictReader(f))

    open_trade = next(
        (r for r in rows if r["trade_id"] == trade_id and r["event"] == "OPEN"),
        None
    )

    if open_trade is None:
        raise ValueError(f"No OPEN trade found for trade_id={trade_id}")

    # Prevent duplicate close
    if any(r for r in rows if r["trade_id"] == trade_id and r["event"] == "CLOSE"):
        raise ValueError(f"Trade {trade_id} already CLOSED")

    size = int(open_trade["size"])
    entry_price = float(open_trade["price"])

    realized_pnl = size * (outcome_price - entry_price)

    # Build CLOSE row with full schema
    close_row: Dict[str, Any] = {field: "" for field in FIELDNAMES}

    # Copy shared fields from OPEN
    for field in FIELDNAMES:
        if field in open_trade:
            close_row[field] = open_trade[field]

    # Override CLOSE-specific fields
    close_row["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    close_row["event"] = "CLOSE"
    close_row["price"] = outcome_price
    close_row["realized_pnl"] = round(realized_pnl, 2)
    close_row["notes"] = f"{notes} | PnL: {realized_pnl:+.2f}".strip()

    with TRADE_LOG_PATH.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerow(close_row)

    print(f"Trade {trade_id} CLOSED | PnL: {realized_pnl:+.2f}")


if __name__ == "__main__":
    main()
