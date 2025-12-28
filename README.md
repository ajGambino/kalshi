# BTC Probability Model

Stochastic model for short horizon bitcoin price outcomes.

## Purpose

Estimate the probability that BTC will be above a given strike price at an exact future timestamp to compare against market implied probabilities from binary markets like Kalshi.

## Phase 1: Core Probability Model

### Model Assumptions

- **Zero drift**: Log returns have mean = 0 (no directional bias)
- **Gaussian distribution**: Log returns are normally distributed (symmetric, light tails)
- **Constant volatility**: Volatility is constant over the forecast horizon
- **Realized volatility**: Estimated from recent historical returns

### Mathematical Framework

Given:

- Current price S₀
- Strike price K
- Forecast horizon T (in hours)
- Historical volatility σ (annualized)

We assume:

```
log(S_T / S₀) ~ N(0, σ²T)
```

Therefore:

```
P(S_T > K) = Φ(−log(K/S₀) / (σ√T))
```

where Φ is the standard normal cumulative distribution function.

### Customizing Parameters

**config.py**:

```python
LOOKBACK_HOURS = 24 # Volatility estimation window
FORECAST_HORIZON_HOURS = 1 # Prediction horizon
PRICE_DATA_INTERVAL_MIN = 5 # Expected data granularity
```

## Phase 2: Trading System & Analytics Layer

Full research & trading framework for evaluating model performance over time.

### Trade Logging Engine

All trades are recorded in central ledger: `trades/trade_log.csv`

**Logged fields:**

- Entry details (market, side, strike, size, price)
- Model diagnostics (probability, volatility, horizon, spot source)
- Execution diagnostics (spot–candle gap, settlement mode, timestamps)
- Expected value metrics & realized PnL
- Settlement metadata

### Backtesting & Performance Analytics

`src/analyze_trades.py` produces full backtest reports:

**Portfolio metrics:**

- Total EV vs realized PnL
- Win rate, ROI, average edge
- Average hold time

**Per-trade breakdown:**

- Entry/outcome prices, expected EV, realized PnL
- Edge % and hold time diagnostics
- Edge bucket analysis

### Probability Calibration

Exported to `trades/calibration_report.csv`:

- Calibration by decile
- Brier score & log loss

### Data Integrity

- Automatic numeric coercion (protects against CSV corruption)
- Backward compatibility with older logs
- Robust outcome handling (supports early exits, validates ranges)
- Self healing for corrupted rows

### Research Workflow

Complete quantitative cycle:

1. Generate probabilities
2. Log trades with diagnostics
3. Track real outcomes
4. Analyze performance vs expectations
5. Measure calibration & improve assumptions iteratively

### Example Output

```

============================================================
Settlement Reference: CF Benchmarks BRTI (60s average)
Settlement Mode: hourly
Spot Proxy Used: Coinbase BTC-USD public ticker
Current Time: 2025-12-18 21:13:12 EST
Settlement Time: 2025-12-18 22:00:00 EST
Forecast Horizon: 0.78 hours
============================================================

Current BTC Price: $85,235.29
Annual Volatility: 55.69%
Horizon Vol: 0.53%

Strike Linear % Log Dist Probability

$83,750.00 -1.74% -0.0176 99.96%
$84,000.00 -1.45% -0.0146 99.73%
$84,250.00 -1.16% -0.0116 98.66%
$84,500.00 -0.86% -0.0087 95.05%
$84,750.00 -0.57% -0.0057 86.15%
$85,000.00 -0.28% -0.0028 70.06%
$85,250.00 +0.02% +0.0002 48.69%
$85,500.00 +0.31% +0.0031 27.75%
$85,750.00 +0.60% +0.0060 12.59%
$86,000.00 +0.90% +0.0089 4.45%
$86,250.00 +1.19% +0.0118 1.21%
$86,500.00 +1.48% +0.0147 0.25%
$86,750.00 +1.78% +0.0176 0.04%

Log trade? (OPEN / skip):

```

## Module Documentation

### **data_loader.py**

- Loads and validates historical price data
- Computes log returns: `log(P_t / P_{t-1})`
- Provides interface to extract recent returns

### **volatility.py**

- Calculates realized volatility from log returns
- Annualizes volatility using square-root-of-time rule
- Scales volatility to forecast horizon

### **probability_model.py**

- Implements Gaussian (log-normal) price model
- Calculates `P(S_T > K)` using normal CDF
- Structured for easy extension to other distributions

### **main.py**

- Entry point for running the model
- Formats and displays probability tables
- Example usage and configuration

### Using with Binary Markets

Compare model probabilities to market implied probabilities:

1. Get Kalshi (or other market) probability for strike K at horizon T
2. Run this model with same K and T
3. If model probability > market probability → market may be underpricing
4. If model probability < market probability → market may be overpricing

**Important**: This is just one input. Consider:

- Model assumptions may be wrong (fat tails, volatility clustering, etc.)
- Markets may have information you don't
- Transaction costs and position sizing
- Most edges are small and fleeting

## Project Roadmap

### Phase 3 — Model Refinement

1. **Fat-tailed return distributions**

   - Replace Gaussian with Student-t and empirical return distributions
   - Improve tail risk estimation and extreme move pricing

2. **Conditional volatility**

   - GARCH / EWMA style volatility models
   - Regime aware volatility scaling

3. **Time-of-day effects**
   - Learn volatility & liquidity profiles by hour
   - Adjust horizon risk estimates accordingly

---

### Phase 4 — Market Regime Modeling

Introduce explicit **market regime classification** to adapt strategies and risk.

Planned regime dimensions:

- **Trend vs Range**
- **Low volatility vs High volatility**
- **Expansion vs Contraction**
- **Momentum vs Mean Reversion dominance**

Initial implementation will use only **price & volatility statistics**:

- Realized volatility
- Trend strength
- Range compression / expansion
- Return autocorrelation

Each trade will be tagged with its regime context, enabling:

- Performance analysis by regime
- Strategy selection by environment
- Adaptive position sizing
- Regime aware calibration of probabilities

---

### Phase 5 — Strategy & Risk Layer

- Regime conditioned strategy selection
- Dynamic risk scaling by regime
- Portfolio level risk management
- Long term calibration monitoring

## Philosophy & Scope

This project is a **probability calculator**, not a prediction engine.

It estimates fair odds under explicit assumptions. Most situations produce no actionable edge.  
Correctness and clarity matter more than sophistication.

This model does **not**:

- Predict price direction
- Use technical indicators
