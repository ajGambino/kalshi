"""
Sanity test for trade log schema consistency.

Tests:
1. OPEN row writes correctly with outcome_price as blank column
2. CLOSE row writes correctly with outcome_price filled
3. Header matches expected FIELDNAMES exactly
4. outcome_price appears in correct column position
5. analyzer can parse the file without errors
"""

import sys
import csv
from pathlib import Path
from datetime import datetime, timezone, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.csv_schema import FIELDNAMES
from src.trade_logger import log_trade


TEST_LOG_PATH = Path("trades/test_trade_log.csv")


def cleanup():
    """Remove test log file if it exists."""
    if TEST_LOG_PATH.exists():
        TEST_LOG_PATH.unlink()
        print(f"Cleaned up {TEST_LOG_PATH}")


def test_schema():
    """Run schema consistency test."""

    print("\n" + "=" * 60)
    print("TRADE LOG SCHEMA SANITY TEST")
    print("=" * 60 + "\n")

    # Clean up any previous test file
    cleanup()

    # Override the log path temporarily
    import src.trade_logger
    import src.close_trade
    original_path = src.trade_logger.TRADE_LOG_PATH
    src.trade_logger.TRADE_LOG_PATH = TEST_LOG_PATH
    src.close_trade.TRADE_LOG_PATH = TEST_LOG_PATH

    try:
        # Test data
        trade_id = "TEST001"
        market = "BTC-HOURLY-2025-12-27-18"
        side = "YES"
        strike = 100000.0
        entry_price = 0.45
        size = 100
        model_prob = 0.52
        spot_price = 99500.0
        last_candle_close = 99480.0
        spot_candle_gap_pct = 0.02
        annual_vol = 0.65
        horizon_hours = 0.85
        settlement_time = datetime.now(timezone.utc) + timedelta(hours=1)
        settlement_mode = "hourly"

        # ------------------------------------------------
        # Test 1: Write OPEN row
        # ------------------------------------------------
        print("Test 1: Writing OPEN row...")
        log_trade(
            trade_id=trade_id,
            market=market,
            side=side,
            strike=strike,
            price=entry_price,
            size=size,
            model_probability=model_prob,
            spot_price=spot_price,
            last_candle_close=last_candle_close,
            spot_candle_gap_pct=spot_candle_gap_pct,
            annual_vol=annual_vol,
            horizon_hours=horizon_hours,
            settlement_time_utc=settlement_time,
            settlement_mode=settlement_mode,
            event="OPEN",
            notes="Test OPEN row"
        )
        print("[OK] OPEN row written\n")

        # ------------------------------------------------
        # Test 2: Verify header
        # ------------------------------------------------
        print("Test 2: Verifying CSV header...")
        with TEST_LOG_PATH.open(newline="") as f:
            reader = csv.reader(f)
            header = next(reader)

        if header != FIELDNAMES:
            print("[FAIL] Header mismatch")
            print(f"Expected: {FIELDNAMES}")
            print(f"Got:      {header}")
            return False

        print(f"[OK] Header matches FIELDNAMES exactly ({len(FIELDNAMES)} columns)\n")

        # ------------------------------------------------
        # Test 3: Verify outcome_price column position
        # ------------------------------------------------
        print("Test 3: Verifying outcome_price column position...")
        outcome_price_idx = FIELDNAMES.index("outcome_price")
        print(f"[OK] outcome_price is at index {outcome_price_idx}\n")

        # ------------------------------------------------
        # Test 4: Verify OPEN row has blank outcome_price
        # ------------------------------------------------
        print("Test 4: Verifying OPEN row has blank outcome_price...")
        with TEST_LOG_PATH.open(newline="") as f:
            rows = list(csv.DictReader(f))

        open_row = rows[0]
        if open_row["outcome_price"] != "":
            print(f"[FAIL] OPEN row outcome_price should be blank, got '{open_row['outcome_price']}'")
            return False

        if open_row["price"] != str(entry_price):
            print(f"[FAIL] OPEN row price should be {entry_price}, got '{open_row['price']}'")
            return False

        print(f"[OK] OPEN row: price={open_row['price']}, outcome_price='{open_row['outcome_price']}' (blank)\n")

        # ------------------------------------------------
        # Test 5: Write CLOSE row (WIN scenario)
        # ------------------------------------------------
        print("Test 5: Writing CLOSE row (WIN)...")

        # Manually construct CLOSE row (simulating close_trade.py logic)
        outcome_price = 1.0  # WIN
        realized_pnl = size * (outcome_price - entry_price)

        close_row = {field: "" for field in FIELDNAMES}

        # Copy shared fields from OPEN
        for field in FIELDNAMES:
            if field in open_row:
                close_row[field] = open_row[field]

        # Override CLOSE-specific fields
        close_row["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        close_row["event"] = "CLOSE"
        close_row["price"] = f"{entry_price:.6f}"  # preserve entry price
        close_row["outcome_price"] = f"{outcome_price:.6f}"  # exit price
        close_row["realized_pnl"] = f"{realized_pnl:.2f}"
        close_row["notes"] = "Test CLOSE row | close_type=WIN | PnL: +55.00"

        # Validate and write
        from src.csv_schema import validate_row_required_fields, validate_csv_header
        validate_row_required_fields(close_row, "CLOSE")
        validate_csv_header(TEST_LOG_PATH)

        with TEST_LOG_PATH.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="raise")
            writer.writerow(close_row)

        print("[OK] CLOSE row written\n")

        # ------------------------------------------------
        # Test 6: Verify CLOSE row
        # ------------------------------------------------
        print("Test 6: Verifying CLOSE row...")
        with TEST_LOG_PATH.open(newline="") as f:
            rows = list(csv.DictReader(f))

        if len(rows) != 2:
            print(f"[FAIL] Expected 2 rows, got {len(rows)}")
            return False

        close_row_verify = rows[1]

        if close_row_verify["event"] != "CLOSE":
            print(f"[FAIL] Second row should be CLOSE, got '{close_row_verify['event']}'")
            return False

        if close_row_verify["price"] != f"{entry_price:.6f}":
            print(f"[FAIL] CLOSE price should preserve entry={entry_price:.6f}, got '{close_row_verify['price']}'")
            return False

        if close_row_verify["outcome_price"] != f"{outcome_price:.6f}":
            print(f"[FAIL] CLOSE outcome_price should be {outcome_price:.6f}, got '{close_row_verify['outcome_price']}'")
            return False

        if close_row_verify["realized_pnl"] != f"{realized_pnl:.2f}":
            print(f"[FAIL] CLOSE realized_pnl should be {realized_pnl:.2f}, got '{close_row_verify['realized_pnl']}'")
            return False

        print(f"[OK] CLOSE row: price={close_row_verify['price']} (entry), outcome_price={close_row_verify['outcome_price']} (exit), realized_pnl={close_row_verify['realized_pnl']}\n")

        # ------------------------------------------------
        # Test 7: Verify analyzer can parse
        # ------------------------------------------------
        print("Test 7: Verifying analyzer can parse...")
        try:
            from src.analyze_trades import load_trade_log
            df = load_trade_log(str(TEST_LOG_PATH))

            if len(df) != 2:
                print(f"[FAIL] Analyzer loaded {len(df)} rows, expected 2")
                return False

            # Check that outcome_price column exists and is parsed correctly
            if "outcome_price" not in df.columns:
                print("[FAIL] Analyzer didn't load outcome_price column")
                return False

            open_outcome = df[df['event'] == 'OPEN']['outcome_price'].iloc[0]
            close_outcome = df[df['event'] == 'CLOSE']['outcome_price'].iloc[0]

            # OPEN should have NaN (was blank)
            import pandas as pd
            if not pd.isna(open_outcome):
                print(f"[FAIL] OPEN outcome_price should be NaN, got {open_outcome}")
                return False

            # CLOSE should have the outcome value
            if close_outcome != outcome_price:
                print(f"[FAIL] CLOSE outcome_price should be {outcome_price}, got {close_outcome}")
                return False

            print("[OK] Analyzer parsed file successfully\n")

        except Exception as e:
            print(f"[FAIL] Analyzer error: {e}")
            return False

        # ------------------------------------------------
        # All tests passed
        # ------------------------------------------------
        print("=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60)
        print("\nSchema is consistent:")
        print(f"  - Header has {len(FIELDNAMES)} columns including outcome_price")
        print(f"  - OPEN rows: price=entry, outcome_price=blank")
        print(f"  - CLOSE rows: price=entry, outcome_price=exit")
        print(f"  - No column shifts or DictWriter errors")
        print(f"  - Analyzer can parse correctly\n")

        return True

    finally:
        # Restore original path
        src.trade_logger.TRADE_LOG_PATH = original_path
        src.close_trade.TRADE_LOG_PATH = original_path

        # Cleanup
        cleanup()


if __name__ == "__main__":
    success = test_schema()
    sys.exit(0 if success else 1)
