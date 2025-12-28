"""
CLI utility to close an existing Kalshi trade.

Now supports:
- WIN / LOSS (settlement close)
- REFUND (PnL forced to 0 by setting outcome_price = entry_price)
- CASHOUT (early exit at a user-provided exit price in [0, 1])

Notes:
- Outcome is relative to YOUR position (YES or NO).
  WIN => payout price 1.0, LOSS => 0.0.
- CASHOUT uses a market exit price (0..1) as the outcome_price.
- This script still writes outcome_price into the existing CLOSE.price column.
  (Your updated analyze_trades now self-validates and can later prefer an explicit
   outcome_price column if you add one.)
"""
import sys
from pathlib import Path
from datetime import datetime, timezone
import csv
from typing import Any, Dict, Optional
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.csv_schema import FIELDNAMES, validate_csv_header, validate_row_required_fields

TRADE_LOG_PATH = Path("trades/trade_log.csv")


def _prompt_choice(prompt: str, allowed: set[str]) -> str:
    val = input(prompt).strip().upper()
    if val not in allowed:
        raise ValueError(f"Invalid choice '{val}'. Allowed: {', '.join(sorted(allowed))}")
    return val


def _prompt_float(prompt: str, min_v: Optional[float] = None, max_v: Optional[float] = None) -> float:
    raw = input(prompt).strip()
    try:
        val = float(raw)
    except ValueError:
        raise ValueError(f"Expected a number, got '{raw}'")

    if min_v is not None and val < min_v:
        raise ValueError(f"Value must be >= {min_v}, got {val}")
    if max_v is not None and val > max_v:
        raise ValueError(f"Value must be <= {max_v}, got {val}")
    return val


def main():
    if not TRADE_LOG_PATH.exists():
        raise FileNotFoundError("trade_log.csv not found")

    trade_id = input("Trade ID: ").strip()
    if not trade_id:
        raise ValueError("Trade ID cannot be empty")

    close_type = _prompt_choice(
        "Close Type (WIN / LOSS / REFUND / CASHOUT): ",
        {"WIN", "LOSS", "REFUND", "CASHOUT"}
    )

    notes = input("Notes (optional): ").strip()

    # Load trades
    with TRADE_LOG_PATH.open(newline="") as f:
        rows = list(csv.DictReader(f))

    open_trade = next(
        (r for r in rows if r.get("trade_id") == trade_id and r.get("event") == "OPEN"),
        None
    )
    if open_trade is None:
        raise ValueError(f"No OPEN trade found for trade_id={trade_id}")

    # Prevent duplicate close (full-close model)
    if any(r for r in rows if r.get("trade_id") == trade_id and r.get("event") == "CLOSE"):
        raise ValueError(f"Trade {trade_id} already CLOSED")

    # Core fields
    try:
        size = int(float(open_trade["size"]))
    except Exception:
        raise ValueError(f"Invalid size on OPEN row for {trade_id}: {open_trade.get('size')}")

    try:
        entry_price = float(open_trade["price"])
    except Exception:
        raise ValueError(f"Invalid entry price on OPEN row for {trade_id}: {open_trade.get('price')}")

    if size <= 0:
        raise ValueError(f"OPEN size must be > 0 (got {size})")

    # Determine outcome_price (exit price)
    if close_type == "WIN":
        outcome_price = 1.0
    elif close_type == "LOSS":
        outcome_price = 0.0
    elif close_type == "REFUND":
        # Refund should produce 0 PnL regardless of entry price
        outcome_price = entry_price
    else:  # CASHOUT
        # This is the price you can sell your position at (0..1)
        outcome_price = _prompt_float("Cashout exit price (0 to 1): ", min_v=0.0, max_v=1.0)

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

    # Schema convention: price always = entry, outcome_price always = exit
    close_row["price"] = f"{entry_price:.6f}"  # preserve entry price from OPEN
    close_row["outcome_price"] = f"{outcome_price:.6f}"  # exit/settlement price

    close_row["realized_pnl"] = f"{realized_pnl:.2f}"

    # Notes: include close type + pnl for easy parsing/debugging
    # Keep your prior "| PnL:" token so older parsers still work.
    note_bits = []
    if notes:
        note_bits.append(notes)
    note_bits.append(f"close_type={close_type}")
    if close_type == "CASHOUT":
        note_bits.append(f"exit_price={outcome_price:.6f}")
    note_bits.append(f"PnL: {realized_pnl:+.2f}")
    close_row["notes"] = " | ".join(note_bits)

    # Validate required fields before writing
    validate_row_required_fields(close_row, "CLOSE")

    # Validate header matches schema (prevent drift)
    validate_csv_header(TRADE_LOG_PATH)

    with TRADE_LOG_PATH.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="raise")
        writer.writerow(close_row)

    print(
        f"Trade {trade_id} CLOSED ({close_type}) | "
        f"Entry: {entry_price:.4f} | Exit: {outcome_price:.4f} | "
        f"Size: {size} | PnL: {realized_pnl:+.2f}"
    )


if __name__ == "__main__":
    main()
