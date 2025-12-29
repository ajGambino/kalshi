"""
Configuration parameters for BTC probability model.

All assumptions are explicit and configurable here.
"""

# ============================================================================
# Data Parameters
# ============================================================================

# Expected granularity of price data (minutes)
PRICE_DATA_INTERVAL_MIN = 5

# ============================================================================
# Volatility Estimation
# ============================================================================

# Lookback window for realized volatility calculation (hours)
LOOKBACK_HOURS = 24

# Note: The actual data frequency (e.g., 5-minute bars) is auto-detected
# from the timestamp column in the data. No manual configuration needed.

# ============================================================================
# Forecast Parameters
# ============================================================================

# Target Kalshi settlements automatically
# When True, uses SETTLEMENT_MODE to determine next settlement
# When False, uses fixed FORECAST_HORIZON_HOURS
USE_NEXT_HOUR_SETTLEMENT = True

# Settlement mode for Kalshi contracts
# Options:
#   'hourly'     - Target next top-of-hour (14:00, 15:00, 16:00, 17:00, ...)
#                  Use for intraday contracts that settle every hour
#   'daily_5pm'  - Target next 17:00 EST only (overnight contracts)
#                  Use when markets pause overnight until next 5pm Eastern Time
#   'auto'       - Auto-select based on Kalshi availability (hourly after 8am ET, daily_5pm before 8am ET)
SETTLEMENT_MODE = 'daily_5pm'  # Options: 'hourly', 'daily_5pm', or 'auto'

# Default forecast horizon (hours) - only used if USE_NEXT_HOUR_SETTLEMENT = False
FORECAST_HORIZON_HOURS = 1.0

# ============================================================================
# Spot Price Provider (Settlement Reference)
# ============================================================================

# Kalshi BTC hourly contracts settle on CF Benchmarks Bitcoin Real-Time Index (BRTI)
# BRTI is a 60-second average of constituent exchanges
SETTLEMENT_REFERENCE = "CF Benchmarks BRTI (60s average)"

# Default spot source for S₀ anchor
# Options:
#   'coinbase' - Fetch from Coinbase BTC-USD public ticker (good BRTI proxy)
#   'manual'   - User-provided price (for testing/backtesting)
SPOT_SOURCE = 'coinbase'

# Spot-candle gap sanity check threshold
# Warn if live spot has moved from last candle close by more than this percentage
# This detects intrabar drift that may indicate volatility regime changes
MAX_SPOT_CANDLE_GAP_PCT = 0.3  # percent (0.3% = 30 basis points)

# ============================================================================
# Strike Grid Parameters (Kalshi Alignment)
# ============================================================================

# Kalshi uses fixed increments for BTC binary options
# Standard is $250 for most BTC contracts
STRIKE_INCREMENT = 250

# Number of strikes to generate above and below spot price
# Example: steps_above=6, steps_below=6 → 13 total strikes (including center)
STRIKE_STEPS_ABOVE = 4
STRIKE_STEPS_BELOW = 4

# ============================================================================
# Model Assumptions (Phase 1)
# ============================================================================

# Drift assumption for log returns
# Phase 1: Zero drift (no directional bias)
DRIFT_MEAN = 0.0

# Distribution assumption
# Phase 1: 'gaussian'
# Phase 2 candidates: 'student_t', 'empirical'
RETURN_DISTRIBUTION = 'gaussian'
