"""
Trade Log Backtest & Analytics Script

Reads trades/trade_log.csv and produces:
- Summary statistics (win rate, PnL, EV vs realized, etc.)
- Per-trade table with detailed metrics
- Edge bucket analysis
- Probability calibration (decile buckets + Brier + log loss)
- CSV export of backtest results (+ optional calibration export)

Backward compatible with older log formats that may be missing:
- realized_pnl (parsed from notes)
- ev_dollars, ev_per_contract, edge_pct (computed from model_probability, price, size)
"""

import sys
import argparse
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import pandas as pd


# ============================================================================
# Constants
# ============================================================================

DEFAULT_LOG_PATH = "trades/trade_log.csv"
DEFAULT_EXPORT_PATH = "trades/backtest_report.csv"
DEFAULT_CALIBRATION_EXPORT_PATH = "trades/calibration_report.csv"


# ============================================================================
# Data Loading & Backward Compatibility
# ============================================================================

def parse_pnl_from_notes(notes: str) -> Optional[float]:
    """
    Extract PnL from CLOSE notes like:
    - "| PnL: +10.61"
    - "settled win | PnL: +19.14"
    - "| PnL: -4.80"

    Returns None if pattern not found.
    """
    if not isinstance(notes, str):
        return None

    match = re.search(r'PnL:\s*([+-]?\d+\.?\d*)', notes)
    if match:
        return float(match.group(1))
    return None


def compute_ev_at_open(row: pd.Series) -> Tuple[float, float, float, float]:
    """
    Compute EV metrics for OPEN row if not already present.

    Returns: (model_prob_used, ev_per_contract, ev_dollars, edge_pct)
    """
    side = row['side']
    model_prob = row['model_probability']
    price = row['price']
    size = row['size']

    # Determine probability based on side
    if side == 'YES':
        p = model_prob
    elif side == 'NO':
        p = 1.0 - model_prob
    else:
        raise ValueError(f"Unknown side: {side}")

    # EV calculations
    ev_per_contract = p - price
    ev_dollars = ev_per_contract * size
    edge_pct = ev_per_contract / price if price > 0 else 0.0

    return p, ev_per_contract, ev_dollars, edge_pct


def load_trade_log(path: str) -> pd.DataFrame:
    """
    Load trade log CSV with backward compatibility.

    Ensures all required columns exist:
    - realized_pnl (parsed from notes if missing)
    - ev_dollars, ev_per_contract, edge_pct (computed if missing)
    """
    df = pd.read_csv(path)

    # Parse timestamps
    df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True, errors='coerce')

    # Parse settlement_time_utc if it exists (coerce errors to NaT for safety)
    if 'settlement_time_utc' in df.columns:
        df['settlement_time_utc'] = pd.to_datetime(df['settlement_time_utc'], utc=True, errors='coerce')

    # Coerce numeric columns (prevents silent failures from string contamination)
    # Common failure mode: CSV edited in Excel, manual edits, inconsistent quoting
    numeric_columns = [
        'price', 'size', 'strike',
        'model_probability', 'ev_per_contract', 'ev_dollars', 'edge_pct', 'realized_pnl',
        'spot_price', 'last_candle_close', 'spot_candle_gap_pct',
        'annual_vol', 'horizon_hours',
        'outcome_price',  # future-proof if/when you add it
    ]

    for col in numeric_columns:
        if col in df.columns:
            # Track original NaN count before coercion
            original_na_count = df[col].isna().sum()
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Warn only if coercion created NEW NaNs (indicates string contamination)
            new_na_count = df[col].isna().sum()
            if new_na_count > original_na_count:
                invalid_count = new_na_count - original_na_count
                print(f"WARNING: Coerced {invalid_count} non-numeric values to NaN in column '{col}'")

    # Ensure realized_pnl exists (CLOSE rows)
    if 'realized_pnl' not in df.columns:
        df['realized_pnl'] = None

    # Ensure notes exists (older logs)
    if 'notes' not in df.columns:
        df['notes'] = ""

    # Parse PnL from notes for CLOSE rows if realized_pnl is missing
    close_mask = df['event'] == 'CLOSE'
    missing_pnl = close_mask & df['realized_pnl'].isna()

    if missing_pnl.any():
        df.loc[missing_pnl, 'realized_pnl'] = df.loc[missing_pnl, 'notes'].apply(parse_pnl_from_notes)

    # Ensure EV columns exist (OPEN rows)
    for col in ['ev_per_contract', 'ev_dollars', 'edge_pct']:
        if col not in df.columns:
            df[col] = None

    # Compute EV for OPEN rows if missing
    open_mask = df['event'] == 'OPEN'
    missing_ev = open_mask & df['ev_dollars'].isna()

    if missing_ev.any():
        for idx in df[missing_ev].index:
            row = df.loc[idx]
            # Skip if core fields are missing
            if pd.isna(row.get('price')) or pd.isna(row.get('size')) or pd.isna(row.get('model_probability')):
                continue
            p, ev_per, ev_total, edge = compute_ev_at_open(row)
            df.loc[idx, 'ev_per_contract'] = ev_per
            df.loc[idx, 'ev_dollars'] = ev_total
            df.loc[idx, 'edge_pct'] = edge

    # Add model_prob_used column for consistency (used in reporting/calibration)
    df['model_prob_used'] = None
    for idx in df[open_mask].index:
        row = df.loc[idx]
        if pd.isna(row.get('model_probability')):
            continue
        if row['side'] == 'YES':
            df.loc[idx, 'model_prob_used'] = row['model_probability']
        else:
            df.loc[idx, 'model_prob_used'] = 1.0 - row['model_probability']

    return df


# ============================================================================
# Trade Pairing
# ============================================================================

def pair_trades(df: pd.DataFrame) -> Tuple[List[Dict], List[Dict]]:
    """
    Pair OPEN and CLOSE rows by trade_id.

    Returns:
        - completed_trades: list of dicts with both open_row and close_row
        - incomplete_trades: list of dicts with only open_row

    Handles:
    - Missing CLOSE rows
    - Duplicate CLOSE rows (warns and takes first by timestamp)
    """
    opens = df[df['event'] == 'OPEN'].copy()
    closes = df[df['event'] == 'CLOSE'].copy()

    # Group by trade_id
    opens_by_id = {row['trade_id']: row for _, row in opens.iterrows()}
    closes_by_id = {}

    for _, row in closes.iterrows():
        tid = row['trade_id']
        if tid not in closes_by_id:
            closes_by_id[tid] = row
        else:
            # Duplicate CLOSE - warn and keep earliest
            existing = closes_by_id[tid]
            if row['timestamp_utc'] < existing['timestamp_utc']:
                print(f"WARNING: Duplicate CLOSE for {tid}, using earlier timestamp")
                closes_by_id[tid] = row
            else:
                print(f"WARNING: Duplicate CLOSE for {tid}, ignoring later timestamp")

    completed = []
    incomplete = []

    for tid, open_row in opens_by_id.items():
        if tid in closes_by_id:
            completed.append({
                'trade_id': tid,
                'open': open_row,
                'close': closes_by_id[tid]
            })
        else:
            incomplete.append({
                'trade_id': tid,
                'open': open_row
            })

    return completed, incomplete


# ============================================================================
# Metrics Calculation
# ============================================================================

def calculate_summary_metrics(completed_trades: List[Dict]) -> Dict:
    """
    Calculate aggregate summary statistics across all completed trades.
    """
    if not completed_trades:
        return {
            'completed_trades': 0,
            'win_rate': 0.0,
            'total_cost': 0.0,
            'total_expected_ev': 0.0,
            'total_realized_pnl': 0.0,
            'realized_minus_expected': 0.0,
            'realized_over_expected': 0.0,
            'average_edge_pct': 0.0,
            'average_hold_minutes': 0.0,
        }

    wins = 0
    total_cost = 0.0
    total_expected_ev = 0.0
    total_realized_pnl = 0.0
    total_edge_pct = 0.0
    total_hold_minutes = 0.0

    for trade in completed_trades:
        open_row = trade['open']
        close_row = trade['close']

        # Cost at entry
        cost = float(open_row['price']) * float(open_row['size'])
        total_cost += cost

        # Expected EV
        total_expected_ev += float(open_row['ev_dollars'])

        # Realized PnL
        pnl = float(close_row['realized_pnl'])
        total_realized_pnl += pnl

        # Win/loss
        if pnl > 0:
            wins += 1

        # Edge
        total_edge_pct += float(open_row['edge_pct'])

        # Hold time (to settlement, not to CLOSE log time)
        settle = open_row.get('settlement_time_utc')
        if pd.notna(settle) and pd.notna(open_row.get('timestamp_utc')):
            hold_seconds = (settle - open_row['timestamp_utc']).total_seconds()
        else:
            hold_seconds = (close_row['timestamp_utc'] - open_row['timestamp_utc']).total_seconds()
        total_hold_minutes += hold_seconds / 60.0

    n = len(completed_trades)

    realized_minus_expected = total_realized_pnl - total_expected_ev
    realized_over_expected = (
        total_realized_pnl / total_expected_ev
        if total_expected_ev != 0
        else 0.0
    )

    return {
        'completed_trades': n,
        'win_rate': wins / n,
        'total_cost': total_cost,
        'total_expected_ev': total_expected_ev,
        'total_realized_pnl': total_realized_pnl,
        'realized_minus_expected': realized_minus_expected,
        'realized_over_expected': realized_over_expected,
        'average_edge_pct': total_edge_pct / n,
        'average_hold_minutes': total_hold_minutes / n,
    }


def build_per_trade_table(completed_trades: List[Dict], include_log_lag: bool = True) -> pd.DataFrame:
    """
    Build detailed per-trade table for export and display.

    Args:
        completed_trades: List of completed trade dicts
        include_log_lag: Whether to include log_lag_minutes column (default: True)
    """
    rows = []

    for trade in completed_trades:
        open_row = trade['open']
        close_row = trade['close']

        # Determine outcome price from CLOSE row
        # Prefer explicit outcome_price column (future), fallback to CLOSE.price (current)
        outcome_price = None

        # close_row may be a Series; use .get for safety
        if 'outcome_price' in close_row and pd.notna(close_row.get('outcome_price')):
            outcome_price = float(close_row.get('outcome_price'))
        elif pd.notna(close_row.get('price')):
            outcome_price = float(close_row.get('price'))

        # Validate outcome_price presence
        if outcome_price is None or pd.isna(outcome_price):
            print(f"ERROR: Trade {trade['trade_id']} missing outcome price in CLOSE row; defaulting to 0.0")
            outcome_price = 0.0

        # Validate outcome_price range (settlement should be ~0 or ~1; cashout should be within [0,1])
        if not (0.0 <= outcome_price <= 1.0):
            print(
                f"WARNING: Trade {trade['trade_id']} has suspicious outcome_price={outcome_price:.4f} "
                f"(expected 0/1 or cashout in [0,1])"
            )

        # Consistency check vs realized_pnl (self-correct if CLOSE.price is wrong but PnL is right)
        entry_price = open_row.get('price')
        size = open_row.get('size')
        realized = close_row.get('realized_pnl')

        if pd.notna(entry_price) and pd.notna(size) and float(size) != 0 and pd.notna(realized):
            expected_pnl = float(size) * (float(outcome_price) - float(entry_price))
            pnl_diff = abs(float(realized) - expected_pnl)

            # Allow small rounding tolerance
            if pnl_diff > 0.01:
                print(
                    f"WARNING: Trade {trade['trade_id']} outcome_price={outcome_price:.4f} inconsistent with "
                    f"realized_pnl={float(realized):+.2f} (diff={pnl_diff:.2f}). Deriving outcome_price from PnL."
                )
                outcome_price = (float(realized) / float(size)) + float(entry_price)

                # Re-validate derived outcome_price
                if not (0.0 <= outcome_price <= 1.0):
                    print(
                        f"WARNING: Trade {trade['trade_id']} derived outcome_price={outcome_price:.4f} is outside [0,1]. "
                        f"Check entry_price/size/realized_pnl inputs."
                    )

        # Hold time in minutes (to settlement, not to CLOSE log time)
        settle = open_row.get('settlement_time_utc')
        if pd.notna(settle) and pd.notna(open_row.get('timestamp_utc')):
            hold_seconds = (settle - open_row['timestamp_utc']).total_seconds()
            # Calculate log lag (how late the CLOSE was logged after settlement)
            log_lag_seconds = (close_row['timestamp_utc'] - settle).total_seconds()
            log_lag_minutes = log_lag_seconds / 60.0

            # Warn about negative log lag (CLOSE logged before settlement - data issue)
            if log_lag_minutes < 0:
                print(
                    f"WARNING: Trade {trade['trade_id']} has negative log lag ({log_lag_minutes:.1f} min) "
                    f"- CLOSE logged before settlement"
                )
        else:
            hold_seconds = (close_row['timestamp_utc'] - open_row['timestamp_utc']).total_seconds()
            log_lag_minutes = None

        hold_minutes = hold_seconds / 60.0

        rows.append({
            'trade_id': trade['trade_id'],
            'open_time': open_row['timestamp_utc'],
            'market': open_row.get('market', ''),
            'side': open_row.get('side', ''),
            'strike': open_row.get('strike', None),
            'entry_price': open_row.get('price', None),
            'size': open_row.get('size', None),
            'model_prob_used': open_row.get('model_prob_used', None),
            'expected_ev_dollars': open_row.get('ev_dollars', None),
            'outcome_price': outcome_price,
            'realized_pnl': close_row.get('realized_pnl', None),
            'hold_minutes': hold_minutes,
            'log_lag_minutes': log_lag_minutes,
            'edge_pct': open_row.get('edge_pct', None),
            'notes_close': close_row.get('notes', ''),
        })

    df = pd.DataFrame(rows)

    # Sort by open_time
    df = df.sort_values('open_time').reset_index(drop=True)

    # Optionally drop log_lag_minutes for export (always keep in terminal display)
    if not include_log_lag and 'log_lag_minutes' in df.columns:
        df = df.drop(columns=['log_lag_minutes'])

    return df


def edge_bucket_analysis(completed_trades: List[Dict]) -> pd.DataFrame:
    """
    Group trades by edge_pct buckets and compute stats per bucket.

    Buckets: (≤0%), (0-2%), (2-5%), (5-10%), (10%+)
    """
    buckets = []

    for trade in completed_trades:
        edge = trade['open']['edge_pct']

        if edge <= 0:
            bucket = '≤0%'
        elif edge <= 0.02:
            bucket = '0-2%'
        elif edge <= 0.05:
            bucket = '2-5%'
        elif edge <= 0.10:
            bucket = '5-10%'
        else:
            bucket = '10%+'

        buckets.append({
            'bucket': bucket,
            'expected_ev': trade['open']['ev_dollars'],
            'realized_pnl': trade['close']['realized_pnl'],
            'win': 1 if trade['close']['realized_pnl'] > 0 else 0,
        })

    df = pd.DataFrame(buckets)

    # Aggregate by bucket
    agg = df.groupby('bucket').agg({
        'expected_ev': ['count', 'mean'],
        'realized_pnl': 'mean',
        'win': 'mean',
    }).reset_index()

    agg.columns = ['bucket', 'count', 'avg_expected_ev', 'avg_realized_pnl', 'win_rate']

    # Order buckets
    bucket_order = ['≤0%', '0-2%', '2-5%', '5-10%', '10%+']
    agg['bucket'] = pd.Categorical(agg['bucket'], categories=bucket_order, ordered=True)
    agg = agg.sort_values('bucket').reset_index(drop=True)

    return agg


# ============================================================================
# Calibration
# ============================================================================

def calibration_report(per_trade_df: pd.DataFrame, n_bins: int = 10) -> Tuple[pd.DataFrame, float, float]:
    """
    Build a probability calibration report using model_prob_used vs realized outcome.

    - Bins are equal-width probability buckets [0,1] (default: deciles)
    - Realized outcome is defined as win=1 if realized_pnl > 0 else 0

    Returns:
        (calibration_df, brier_score, log_loss)
    """
    if per_trade_df.empty:
        cal = pd.DataFrame(columns=[
            'bucket', 'count', 'avg_model_prob', 'win_rate', 'gap', 'avg_expected_ev', 'avg_realized_pnl'
        ])
        return cal, float('nan'), float('nan')

    df = per_trade_df.copy()

    # Ensure numeric
    df['model_prob_used'] = pd.to_numeric(df['model_prob_used'], errors='coerce')
    df['realized_pnl'] = pd.to_numeric(df['realized_pnl'], errors='coerce')
    df['expected_ev_dollars'] = pd.to_numeric(df['expected_ev_dollars'], errors='coerce')

    df = df.dropna(subset=['model_prob_used', 'realized_pnl'])

    # Outcome label
    df['win'] = (df['realized_pnl'] > 0).astype(int)

    # Scores
    p = df['model_prob_used'].clip(0.0, 1.0)
    y = df['win']

    brier = float(((p - y) ** 2).mean()) if len(df) else float('nan')

    eps = 1e-12
    p_clip = p.clip(eps, 1 - eps)
    logloss = float((-(y * (p_clip.apply(lambda x: __import__("math").log(x))) +
                     (1 - y) * ((1 - p_clip).apply(lambda x: __import__("math").log(x))))).mean()) if len(df) else float('nan')

    # Binning
    bins = [i / n_bins for i in range(n_bins + 1)]
    labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(n_bins)]
    df['bucket'] = pd.cut(p, bins=bins, labels=labels, include_lowest=True, right=True)

    cal = df.groupby('bucket', observed=False).agg(
        count=('bucket', 'count'),
        avg_model_prob=('model_prob_used', 'mean'),
        win_rate=('win', 'mean'),
        avg_expected_ev=('expected_ev_dollars', 'mean'),
        avg_realized_pnl=('realized_pnl', 'mean'),
    ).reset_index()

    cal['gap'] = cal['win_rate'] - cal['avg_model_prob']
    return cal, brier, logloss


def print_calibration(cal_df: pd.DataFrame, brier: float, logloss: float):
    """Print probability calibration table to terminal."""
    print("\n" + "=" * 80)
    print("PROBABILITY CALIBRATION (by model_prob_used deciles)")
    print("=" * 80)

    if cal_df.empty:
        print("No data available for calibration (need completed trades with model_prob_used and realized_pnl).")
        print("=" * 80 + "\n")
        return

    display = cal_df.copy()
    display['avg_model_prob'] = display['avg_model_prob'].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
    display['win_rate'] = display['win_rate'].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
    display['gap'] = display['gap'].apply(lambda x: f"{x:+.2%}" if pd.notna(x) else "N/A")
    display['avg_expected_ev'] = display['avg_expected_ev'].apply(lambda x: f"${x:+.2f}" if pd.notna(x) else "N/A")
    display['avg_realized_pnl'] = display['avg_realized_pnl'].apply(lambda x: f"${x:+.2f}" if pd.notna(x) else "N/A")

    print(display.to_string(index=False))
    print("-" * 80)
    if pd.notna(brier):
        print(f"Brier Score: {brier:.4f}  (lower is better)")
    else:
        print("Brier Score: N/A")
    if pd.notna(logloss):
        print(f"Log Loss:    {logloss:.4f}  (lower is better)")
    else:
        print("Log Loss:    N/A")
    print("=" * 80 + "\n")


# ============================================================================
# Output Formatting
# ============================================================================

def print_summary(metrics: Dict):
    """Print summary metrics to terminal."""
    print("\n" + "=" * 70)
    print("TRADE BACKTEST SUMMARY")
    print("=" * 70)
    print(f"Completed Trades:         {metrics['completed_trades']}")
    print(f"Win Rate:                 {metrics['win_rate']:.2%}")
    print(f"Total Cost (Entry):       ${metrics['total_cost']:.2f}")
    print(f"Total Expected EV:        ${metrics['total_expected_ev']:.2f}")
    print(f"Total Realized PnL:       ${metrics['total_realized_pnl']:.2f}")
    print(f"Realized - Expected:      ${metrics['realized_minus_expected']:.2f}")

    # Ratio with inf handling
    if metrics['total_expected_ev'] != 0:
        print(f"Realized / Expected:      {metrics['realized_over_expected']:.2f}x")
    else:
        print(f"Realized / Expected:      N/A (expected EV = 0)")

    print(f"Average Edge %:           {metrics['average_edge_pct']:.2%}")
    print(f"Average Hold Time:        {metrics['average_hold_minutes']:.1f} minutes")
    print("=" * 70 + "\n")


def print_per_trade_table(df: pd.DataFrame):
    """Print per-trade table to terminal."""
    print("\n" + "=" * 140)
    print("PER-TRADE BREAKDOWN")
    print("=" * 140)

    # Format for display
    display_df = df.copy()
    display_df['open_time'] = pd.to_datetime(display_df['open_time'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M')
    display_df['entry_price'] = display_df['entry_price'].apply(lambda x: f"{x:.4f}")
    display_df['model_prob_used'] = display_df['model_prob_used'].apply(lambda x: f"{x:.4f}")
    display_df['expected_ev_dollars'] = display_df['expected_ev_dollars'].apply(lambda x: f"{x:+.2f}")
    display_df['outcome_price'] = display_df['outcome_price'].apply(lambda x: f"{x:.3f}")
    display_df['realized_pnl'] = display_df['realized_pnl'].apply(lambda x: f"{x:+.2f}")
    display_df['hold_minutes'] = display_df['hold_minutes'].apply(lambda x: f"{x:.1f}")
    display_df['log_lag_minutes'] = display_df['log_lag_minutes'].apply(
        lambda x: f"{x:+.1f}" if pd.notna(x) else "N/A"  # Use +/- sign to flag negatives
    )
    display_df['edge_pct'] = display_df['edge_pct'].apply(lambda x: f"{x:.2%}")

    # Select columns for terminal display
    cols = [
        'trade_id', 'open_time', 'side', 'strike', 'entry_price', 'size',
        'edge_pct', 'expected_ev_dollars', 'outcome_price', 'realized_pnl', 'hold_minutes', 'log_lag_minutes'
    ]

    print(display_df[cols].to_string(index=False))
    print("=" * 140 + "\n")


def print_edge_buckets(bucket_df: pd.DataFrame):
    """Print edge bucket analysis to terminal."""
    print("\n" + "=" * 80)
    print("EDGE BUCKET ANALYSIS")
    print("=" * 80)

    display_df = bucket_df.copy()
    display_df['avg_expected_ev'] = display_df['avg_expected_ev'].apply(lambda x: f"${x:+.2f}")
    display_df['avg_realized_pnl'] = display_df['avg_realized_pnl'].apply(lambda x: f"${x:+.2f}")
    display_df['win_rate'] = display_df['win_rate'].apply(lambda x: f"{x:.2%}")

    print(display_df.to_string(index=False))
    print("=" * 80 + "\n")


def print_incomplete_trades(incomplete: List[Dict]):
    """Print list of trades that are OPEN but not CLOSED."""
    if not incomplete:
        print("\nNo incomplete trades (all OPEN trades have been CLOSED).\n")
        return

    print("\n" + "=" * 80)
    print(f"INCOMPLETE TRADES ({len(incomplete)} OPEN without CLOSE)")
    print("=" * 80)

    for trade in incomplete:
        open_row = trade['open']
        print(f"  {trade['trade_id']} | {open_row.get('market','')} | {open_row.get('side','')} @ {open_row.get('strike','')}")

    print("=" * 80 + "\n")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze trade log and generate backtest + calibration reports"
    )
    parser.add_argument(
        '--path',
        default=DEFAULT_LOG_PATH,
        help=f"Path to trade log CSV (default: {DEFAULT_LOG_PATH})"
    )
    parser.add_argument(
        '--export',
        default=DEFAULT_EXPORT_PATH,
        help=f"Path to export per-trade backtest CSV (default: {DEFAULT_EXPORT_PATH})"
    )
    parser.add_argument(
        '--export-calibration',
        default=DEFAULT_CALIBRATION_EXPORT_PATH,
        help=f"Path to export calibration CSV (default: {DEFAULT_CALIBRATION_EXPORT_PATH})"
    )
    parser.add_argument(
        '--show-open-only',
        action='store_true',
        help="Show only incomplete trades (OPEN without CLOSE)"
    )
    parser.add_argument(
        '--no-log-lag',
        dest='include_log_lag',
        action='store_false',
        help="Exclude log_lag_minutes diagnostic column from CSV export (default: include it)"
    )
    parser.add_argument(
        '--no-calibration',
        dest='run_calibration',
        action='store_false',
        help="Skip calibration output and CSV export"
    )
    parser.set_defaults(include_log_lag=True, run_calibration=True)

    args = parser.parse_args()

    # Validate input file
    if not Path(args.path).exists():
        print(f"ERROR: Trade log not found at {args.path}")
        sys.exit(1)

    # Load data
    print(f"Loading trade log from {args.path}...")
    df = load_trade_log(args.path)
    print(f"Loaded {len(df)} rows.\n")

    # Pair trades
    completed, incomplete = pair_trades(df)
    print(f"Found {len(completed)} completed trades, {len(incomplete)} incomplete.\n")

    # Handle --show-open-only flag
    if args.show_open_only:
        print_incomplete_trades(incomplete)
        return

    # Skip analysis if no completed trades
    if not completed:
        print("No completed trades to analyze.")
        return

    # Calculate metrics
    summary = calculate_summary_metrics(completed)
    per_trade_df_display = build_per_trade_table(completed, include_log_lag=True)  # Always show in terminal
    bucket_df = edge_bucket_analysis(completed)

    # Print outputs
    print_summary(summary)
    print_per_trade_table(per_trade_df_display)
    print_edge_buckets(bucket_df)
    print_incomplete_trades(incomplete)

    # Calibration (print + export)
    if args.run_calibration:
        cal_df, brier, logloss = calibration_report(per_trade_df_display, n_bins=10)
        print_calibration(cal_df, brier, logloss)

        cal_path = Path(args.export_calibration)
        cal_path.parent.mkdir(parents=True, exist_ok=True)
        cal_df.to_csv(cal_path, index=False)
        print(f"Exported calibration report to {args.export_calibration}\n")

    # Export per-trade CSV (optionally exclude log_lag_minutes)
    export_path = Path(args.export)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    per_trade_df_export = build_per_trade_table(completed, include_log_lag=args.include_log_lag)
    per_trade_df_export.to_csv(export_path, index=False)
    print(f"Exported per-trade table to {args.export}\n")


if __name__ == "__main__":
    main()
