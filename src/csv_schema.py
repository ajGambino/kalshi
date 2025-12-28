"""
Shared CSV schema and validation for trade log.

Single source of truth for:
- Column names and order (FIELDNAMES)
- Header validation (prevent schema drift)
- Required field validation (prevent incomplete rows)
"""

from pathlib import Path
from typing import Dict, Any
import csv


# -------------------------------------------------
# Schema Definition (Single Source of Truth)
# -------------------------------------------------

FIELDNAMES = [
    "trade_id",
    "timestamp_utc",
    "event",                   # OPEN or CLOSE
    "market",
    "side",                    # YES / NO
    "strike",
    "price",                   # entry price (always, both OPEN and CLOSE)
    "outcome_price",           # exit price (blank on OPEN, filled on CLOSE)
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
# Validation Functions
# -------------------------------------------------

def validate_csv_header(csv_path: Path) -> None:
    """
    Validate that existing CSV header matches expected FIELDNAMES exactly.

    Raises:
        RuntimeError: If header doesn't match (schema drift detected)
    """
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        # File doesn't exist or is empty - no header to validate
        return

    with csv_path.open(newline="") as f:
        reader = csv.reader(f)
        try:
            existing_header = next(reader)
        except StopIteration:
            # Empty file
            return

    if existing_header != FIELDNAMES:
        # Find differences for helpful error message
        missing = set(FIELDNAMES) - set(existing_header)
        extra = set(existing_header) - set(FIELDNAMES)

        msg_parts = [
            "Schema mismatch detected in trade_log.csv!",
            "",
            "This prevents silent column shifts and data corruption.",
            "",
        ]

        if missing:
            msg_parts.append(f"Missing columns: {', '.join(sorted(missing))}")
        if extra:
            msg_parts.append(f"Extra columns: {', '.join(sorted(extra))}")

        msg_parts.extend([
            "",
            "Action required:",
            "1. Archive/backup the existing trade_log.csv",
            "2. Migrate data to new schema if needed",
            "3. Delete or rename the old file",
            "4. Re-run to create a fresh log with correct schema",
        ])

        raise RuntimeError("\n".join(msg_parts))


def validate_row_required_fields(row: Dict[str, Any], event: str) -> None:
    """
    Validate that required fields are present and non-empty.

    Args:
        row: Row dict to validate
        event: "OPEN" or "CLOSE"

    Raises:
        ValueError: If required fields are missing or empty
    """
    # Always required (both OPEN and CLOSE)
    always_required = [
        "trade_id",
        "timestamp_utc",
        "event",
        "market",
        "side",
        "strike",
        "size",
        "settlement_time_utc",
        "settlement_mode",
    ]

    # Event-specific requirements
    if event == "OPEN":
        event_required = ["price", "model_probability"]
    elif event == "CLOSE":
        event_required = ["outcome_price", "realized_pnl"]
    else:
        raise ValueError(f"Invalid event type: {event}")

    required_fields = always_required + event_required

    missing = []
    for field in required_fields:
        value = row.get(field)
        # Check for None, empty string, or missing key
        if value is None or value == "":
            missing.append(field)

    if missing:
        raise ValueError(
            f"Required fields missing or empty for {event} row: {', '.join(missing)}\n"
            f"Row trade_id: {row.get('trade_id', 'UNKNOWN')}"
        )
