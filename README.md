# Quantitative Trading Pipeline

A sophisticated algorithmic trading system implementing multi-factor signal processing with dynamic leverage optimization and liquidity-aware position sizing, designed for automated portfolio management across multiple brokers.

> **Project Focus**: This project was built to understand how quantitative signals behave once realistic execution, liquidity and risk constraints are applied, with a strong focus on risk management rather than pure alpha maximization.

---

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [System Architecture](#system-architecture)
4. [Signal Processing](#signal-processing)
5. [Risk Management](#risk-management)
6. [Execution System](#execution-system)
7. [Installation](#installation)
8. [Configuration](#configuration)
9. [Usage](#usage)
10. [Stress Testing](#stress-testing)
11. [Database Schema](#database-schema)
12. [Design Trade-offs](#design-trade-offs)
13. [Performance Metrics](#performance-metrics)
14. [Limitations & Risks](#limitations--risks)
15. [Future Development](#future-development)

---

## Overview

This pipeline implements a **hybrid multi-factor trading system** that combines:

- **5 Alpha Factors**: Volatility, Momentum, Tail Risk, Beta, Carry
- **Multi-Timeframe Fusion**: 70% daily signals + 30% hourly tactical timing
- **Real Benchmark Data**: Actual SPY returns for accurate beta/momentum calculations
- **Liquidity Constraints**: 10% ADV cap to prevent market impact
- **Dual-Broker Architecture**: Alpaca for data, Interactive Brokers for execution
- **Comprehensive Risk Management**: Drawdown limits, leverage control, tail hedging

### System Metrics

| Metric | Value |
|--------|-------|
| Total Lines of Code | ~4,200 |
| Alpha Factors | 5 |
| Historical Data | 10 years daily + 2 years hourly |
| Stress Scenarios | 6 historical crises |
| Database Tables | 6 |
| Rebalance Frequency | Hourly |

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Real SPY Benchmark** | Uses actual SPY returns instead of approximations for accurate signal calculation |
| **Liquidity Constraints** | Maximum 10% of 20-day ADV per position to prevent market impact |
| **Dual-Broker Architecture** | Alpaca for superior data API, IBKR for execution (no PDT restrictions) |
| **Stateful Tail Hedge** | Convex protection with state machine (activation at -5% DD) |
| **Adaptive Position Decay** | Exponential decay (10-day halflife) with 15% buffer zone |
| **Slippage Estimation** | Square-root market impact model for pre-trade cost analysis |
| **Dynamic Correlation Weighting** | SLSQP optimization to maximize IC while penalizing correlated signals |
| **Health Score System** | Path-dependent signal validation with 5 components |
| **Multi-Timeframe Fusion** | 70% daily + 30% hourly signal composition |
| **Comprehensive Stress Testing** | 6 historical crisis scenarios (1987-2020) |

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TRADING SYSTEM                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │   DATA LAYER    │    │  SIGNAL LAYER   │    │   RISK LAYER    │         │
│  ├─────────────────┤    ├─────────────────┤    ├─────────────────┤         │
│  │ DualBrokerAdapter│   │ EnhancedPipeline │   │  RiskMonitor    │         │
│  │ IBKRAdapter     │    │ HybridPipeline  │    │ StatefulTailHedge│        │
│  │ Alpaca API      │    │ QuantSignal ×5  │    │ PositionSizer   │         │
│  └────────┬────────┘    └────────┬────────┘    └────────┬────────┘         │
│           │                      │                      │                   │
│           └──────────────────────┼──────────────────────┘                   │
│                                  │                                          │
│                    ┌─────────────▼─────────────┐                            │
│                    │    EXECUTION LAYER        │                            │
│                    ├───────────────────────────┤                            │
│                    │   PaperTradingEngine      │                            │
│                    │   AlertManager            │                            │
│                    │   TradingDatabase         │                            │
│                    └───────────────────────────┘                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Component Overview

| Layer | Component | Responsibility |
|-------|-----------|----------------|
| **Data** | DualBrokerAdapter | Unified interface for Alpaca data + IBKR execution |
| **Data** | IBKRAdapter | Interactive Brokers API wrapper |
| **Signal** | EnhancedProductionPipeline | Factor computation, weight optimization, backtesting |
| **Signal** | HybridWeightedPipeline | Multi-timeframe fusion (70% daily + 30% hourly) |
| **Signal** | QuantSignal (5 subclasses) | Individual alpha factor implementations |
| **Risk** | RiskMonitor | Real-time drawdown, leverage, daily loss monitoring |
| **Risk** | StatefulTailHedge | Convex protection with state machine |
| **Risk** | AdaptivePositionSizer | Position decay and buffer zone management |
| **Execution** | PaperTradingEngine | Main loop, rebalancing, order execution |
| **Execution** | AlertManager | Email and Slack notifications |
| **Storage** | TradingDatabase | SQLite persistence for all trading data |

### File Structure

```
quantitative-trading-pipeline/
├── paper_trading.py           # 1,700 lines - Main trading engine
├── enhanced_pipeline.py       # 1,690 lines - Signal processing & backtesting
├── hybrid_pipeline_weighted.py # 200 lines - Multi-timeframe fusion
├── dual_broker_adapter.py     # 175 lines - Unified broker interface
├── ibkr_adapter.py            # 418 lines - IBKR integration
├── trading_config.json        # Runtime configuration
├── paper_trading_ibkr.db      # SQLite database
└── paper_trading_ibkr.log     # Execution logs
```

### Dual-Broker Architecture

| Broker | Functions | Rationale |
|--------|-----------|-----------|
| **Alpaca** | `get_bars()`, `get_latest_bar()`, `get_clock()` | Excellent data API, 10+ years history, free tier |
| **IBKR** | `submit_order()`, `list_positions()`, `get_account()` | No PDT restrictions, better execution, cash account |

This architecture leverages Alpaca's superior data API while avoiding Pattern Day Trader restrictions on small accounts.

---

## Signal Processing

### Real Benchmark Integration

A critical improvement over typical academic implementations:

```python
# Fetch real SPY returns for benchmark
spy_bars = api.get_bars('SPY', TimeFrame.Day, limit=500)
spy_returns = spy_bars['close'].pct_change()

# Align with stock returns
market_returns = spy_returns.reindex(stock_returns.index).ffill()

# Pass to signal calculation
signals = calculate_daily_signals_weighted(returns, market_returns=market_returns)
```

**Benefits**:
- Accurate excess momentum calculation
- Proper beta estimation against real market
- Realistic signal correlation

### Alpha Factors

#### 1. Volatility Signal (Mean Reversion)

```python
realized_vol = returns.rolling(20).std() * np.sqrt(252)
vol_of_vol = realized_vol.rolling(20).std()
vol_zscore = (realized_vol - realized_vol.rolling(252).mean()) / realized_vol.rolling(252).std()
signal = -vol_zscore * (1 + vol_of_vol)
```

**Rationale**: Short volatility spikes (mean reversion), amplified by vol-of-vol for regime changes.

#### 2. Momentum Signal (Excess Returns)

```python
momentum_fast = returns.rolling(20).sum()
benchmark_momentum = spy_returns.rolling(20).sum()
excess_momentum = momentum_fast - benchmark_momentum
momentum_diff = momentum_fast - returns.rolling(60).sum()
signal = 0.7 * excess_momentum + 0.3 * momentum_diff
```

**Rationale**: Combines relative strength vs benchmark with time-series momentum.

#### 3. Tail Risk Signal (CVaR-based)

```python
expected_shortfall = returns.rolling(60).quantile(0.05)
left_tail = returns.rolling(60).quantile(0.05)
right_tail = returns.rolling(60).quantile(0.95)
tail_asymmetry = (right_tail + left_tail) / (right_tail - left_tail)
signal = -expected_shortfall * tail_asymmetry
```

**Rationale**: Positions against extreme downside risk, adjusted by tail asymmetry.

#### 4. Beta Signal (Market Sensitivity)

```python
rolling_cov = returns.rolling(60).cov(spy_returns)
market_var = spy_returns.rolling(60).var()
beta = rolling_cov / market_var
beta_vol = beta.rolling(60).std()
signal = -beta * (1 + beta_vol)
```

**Rationale**: Short high and unstable market sensitivity.

#### 5. Carry Signal (Composite)

```python
# VRP Proxy (50%)
implied_vol_proxy = returns.rolling(5).std() * np.sqrt(252) * 1.2
vrp = implied_vol_proxy - realized_vol

# Mean Reversion (30%)
ma_252 = prices.rolling(252).mean()
mean_reversion = -(prices - ma_252) / ma_252

# Seasonality (20%)
day_of_year = prices.index.dayofyear
seasonality = np.sin(2 * np.pi * day_of_year / 365)

signal = 0.5 * vrp + 0.3 * mean_reversion + 0.2 * seasonality
```

### Signal Validation

Each signal is continuously validated:

| Metric | Formula | Purpose |
|--------|---------|---------|
| Rolling IC | `corr(signal, returns[t+1])` | Predictive power |
| Rolling Hit Rate | `% correct direction` | Directional accuracy |
| Turnover-Adjusted IC | `IC / (1 + turnover×100)` | Net of trading costs |

### Health Score

Path-dependent health score (5 components):

| Component | Weight | Description |
|-----------|--------|-------------|
| IC Level Score | 35% | Sigmoid transformation of recent IC level |
| Stability Score | 20% | Inverse of IC standard deviation |
| Turnover Score | 15% | Turnover-adjusted IC quality |
| Trend Score | 20% | IC velocity (recent vs past) |
| Acceleration Score | 10% | IC second derivative |

**Signals with health score < 0.3 are excluded from the composite.**

### Weight Optimization

```python
# 1. Base Weights
composite_score = 0.5 * health_score + 0.5 * sigmoid(pnl_sharpe)

# 2. Risk Budgeting
risk_budgets = {
    'Momentum': 0.30,
    'Carry': 0.25,
    'Volatility': 0.20,
    'TailRisk': 0.15,
    'Beta': 0.10
}

# 3. Correlation Adjustment (SLSQP optimization)
objective = minimize(-IC + 0.5 * avg_correlation)

# 4. Normalize to sum to 1.0
```

### Hybrid Multi-Timeframe Fusion

| Timeframe | Weight | Data | Signals |
|-----------|--------|------|---------|
| Daily | 70% | 10 years | All 5 factors |
| Hourly | 30% | 2 years | Momentum, Volatility, RSI, Price Action |

```python
final_signal = 0.7 * daily_signal + 0.3 * hourly_signal
confidence = calculate_agreement(daily_signal, hourly_signal)
```

---

## Risk Management

### Position Sizing with Liquidity Constraints

```python
def calculate_target_positions(signals, equity, prices):
    for symbol in symbols:
        # 1. Signal strength [-1, +1]
        signal_strength = np.tanh(composite_signal * 2)
        
        # 2. Confidence adjustment
        adjusted_strength = signal_strength * confidence
        
        # 3. Dynamic leverage
        leverage = min_leverage + (max_leverage - min_leverage) * abs(adjusted_strength)
        
        # 4. Target notional
        target_notional = equity * np.sign(adjusted_strength) * leverage * max_position_size
        
        # 5. Shares calculation
        shares = int(target_notional / price)
        
        # 6. LIQUIDITY CONSTRAINT (NEW)
        adv_20 = volume.rolling(20).mean()
        max_shares_liquidity = int(adv_20 * 0.10)  # 10% of ADV
        
        # 7. Portfolio constraint
        max_shares_portfolio = int(equity * max_position_size * max_leverage / price)
        
        # 8. Apply both constraints
        final_shares = np.clip(shares, 
                               -min(max_shares_portfolio, max_shares_liquidity),
                               +min(max_shares_portfolio, max_shares_liquidity))
```

### Risk Limits

| Parameter | Value | Action |
|-----------|-------|--------|
| Maximum Position Size | 20% | Per-asset limit |
| Maximum Gross Leverage | 3.0x | Total exposure / equity |
| Maximum Net Leverage | 1.5x | (Long - Short) / equity |
| Liquidity Cap (NEW) | 10% ADV | Market impact prevention |
| Daily Loss Warning | -5% | Alert, reduce new positions |
| Daily Loss Limit | -10% | Halt new positions |
| Maximum Drawdown | -15% | Flatten 50% of portfolio |
| Stop Loss | -20% | Full liquidation |
| Position Buffer Zone | 15% | Minimum change to trigger rebalance |

### Tail Hedging Mechanism

State machine with two states:

| State | Condition | Behavior |
|-------|-----------|----------|
| **INACTIVE** | DD > -5% | No hedge cost, no protection |
| **ACTIVE** | DD < -5% | Convex payoff active, ~2 bps/day cost |

**Transitions**:
- INACTIVE → ACTIVE: When drawdown breaches -5%
- ACTIVE → INACTIVE: When drawdown recovers above -2.5%

**Payoff Formula**:
```python
downside = max(0, -market_return)
hedge_pnl = hedge_size * (downside ** 1.5) * 3  # Convex payoff
hedge_cost = 0.0002 if is_active else 0  # 2 bps daily
net_return = portfolio_return + hedge_pnl - hedge_cost
```

### Adaptive Position Decay

```python
decay_halflife = 10  # days
decay_factor = 0.5 ** (position_age / decay_halflife)
decayed_position = raw_position * decay_factor

# Buffer zone prevents excessive trading
buffer_zone = 0.15
if abs(new_position - current_position) / current_position < buffer_zone:
    new_position = current_position  # No change
```

### Slippage Estimation

Square-root market impact model:

```python
def estimate_slippage(shares, adv, volatility):
    participation = abs(shares) / adv
    base_spread = 10.0  # bps (large cap)
    impact = 10.0 * np.sqrt(participation) * (volatility / 0.20)
    total_slippage = base_spread/2 + impact
    return min(total_slippage, 100.0)  # Cap at 1%
```

### Leverage Management

```python
# Pre-rebalance check
if current_leverage > max_leverage:
    scale_factor = max_leverage / current_leverage
    for symbol in positions:
        target[symbol] = int(target[symbol] * scale_factor)

# Buying power check
if total_notional_needed > buying_power:
    scale_factor = (buying_power / total_notional_needed) * 0.95
    # Scale down all orders
```

---

## Execution System

### Order Flow

| Transition | Execution Steps |
|------------|-----------------|
| FLAT → LONG | Single BUY order |
| FLAT → SHORT | Single SELL order |
| LONG → FLAT | Single SELL order (close) |
| SHORT → FLAT | Single BUY order (cover) |
| LONG → SHORT | Step 1: SELL to close, Step 2: SELL to open (2s delay) |
| SHORT → LONG | Step 1: BUY to cover, Step 2: BUY to open (2s delay) |

### Order Parameters

| Parameter | Value |
|-----------|-------|
| Order Type | Market (execution certainty) |
| TIF (Market Open) | DAY |
| TIF (Market Closed) | GTC |
| Minimum Trade Size | 1 share |
| Fill Timeout | 30 seconds |

### IBKR Integration

| Feature | Implementation |
|---------|----------------|
| Connection | TWS/Gateway on 127.0.0.1:7497 |
| Daily Bars | Duration in YEARS for >365 days |
| Hourly Bars | Chunked requests (24h max), concatenated |
| Market Hours | US Eastern, 9:30 AM - 4:00 PM |

### Rebalancing Schedule

| Parameter | Value |
|-----------|-------|
| Rebalance Frequency | Every 60 minutes |
| Position Reconciliation | Every 5 minutes |
| Portfolio Status Log | Every 10 iterations |
| Market Hours Check | Every cycle |

---

## Installation

### Prerequisites

- Python 3.9+
- Interactive Brokers TWS or Gateway
- Alpaca account (free tier works)

### Install Dependencies

```bash
pip install pandas numpy scipy alpaca-trade-api ib_insync requests
```

### Configure Brokers

1. **Alpaca**: Get API keys from [alpaca.markets](https://alpaca.markets)
2. **IBKR**: Enable API in TWS (File → Global Configuration → API → Settings)

### Setup Configuration

```bash
cp trading_config.example.json trading_config.json
# Edit with your API keys
```

---

## Configuration

### trading_config.json

```json
{
    "alpaca_api_key": "YOUR_KEY",
    "alpaca_api_secret": "YOUR_SECRET",
    "alpaca_base_url": "https://paper-api.alpaca.markets",
    "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
    "rebalance_frequency_minutes": 60,
    "min_trade_size": 1,
    "max_position_size": 0.20,
    "max_daily_loss": -0.05,
    "max_drawdown": -0.15,
    "min_leverage": 1.0,
    "max_leverage": 3.0,
    "check_interval_seconds": 30,
    "reconciliation_interval_minutes": 5,
    "db_path": "paper_trading_ibkr.db",
    "log_file": "paper_trading_ibkr.log"
}
```

### Pipeline Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `initial_capital` | 1,000,000 | Backtest starting capital |
| `max_gross_leverage` | 3.0 | Maximum gross leverage |
| `max_net_leverage` | 1.5 | Maximum net leverage |
| `max_adv_pct` | 0.10 | 10% ADV liquidity cap |
| `hedge_threshold` | -0.05 | Tail hedge activation |
| `hedge_size` | 0.10 | Hedge notional size |
| `volatility_window` | 20 | Vol signal lookback |
| `momentum_fast` | 20 | Fast momentum window |
| `momentum_slow` | 60 | Slow momentum window |
| `validation_window` | 252 | Rolling validation window |
| `costs_bps` | 5 | Transaction cost assumption |

---

## Usage

### Run Paper Trading

```bash
python paper_trading.py
```

### Run Backtest Only

```python
from enhanced_pipeline import EnhancedProductionPipeline

pipeline = EnhancedProductionPipeline()
pipeline.add_signal(VolatilitySignal)
pipeline.add_signal(MomentumSignal)
pipeline.add_signal(TailRiskSignal)
pipeline.add_signal(BetaSignal)
pipeline.add_signal(CarrySignal)

results = pipeline.run_full_pipeline_vectorized(prices, returns, volume)
metrics = pipeline.get_performance_metrics(results)
```

### Run Stress Tests

```python
from enhanced_pipeline import HistoricalVaRStressTester

tester = HistoricalVaRStressTester(config, logger)
results = tester.run_historical_stress_test(pipeline, prices, returns, volume)
```

---

## Stress Testing

### Historical Crisis Scenarios

| Scenario | Crash | Days | Vol × | Characteristics |
|----------|-------|------|-------|-----------------|
| Black Monday 1987 | -20% | 1 | 5× | Single-day flash crash |
| LTCM Crisis 1998 | -15% | 5 | 3× | Liquidity crisis |
| Dot-com Crash 2000 | -30% | 20 | 2.5× | Bubble burst |
| Financial Crisis 2008 | -40% | 60 | 4× | Systemic crisis |
| Flash Crash 2010 | -10% | 1 | 10× | Algo-driven crash |
| COVID Crash 2020 | -35% | 15 | 6× | Pandemic shock |

### Correlation Breakdown Testing

| Scenario | Correlation | Expected Impact |
|----------|-------------|-----------------|
| Base Case | ρ ≈ 0 | Full diversification |
| Moderate Stress | ρ = 0.5 | Reduced diversification |
| High Stress | ρ = 0.8 | Significant loss |
| Extreme Crisis | ρ = 0.95 | Near-total correlation |

---

## Database Schema

### Tables

| Table | Purpose |
|-------|---------|
| `market_data` | OHLCV price data cache |
| `signals` | Historical signal values |
| `positions` | Position snapshots with P&L |
| `orders` | Complete order history |
| `portfolio` | Portfolio snapshots |
| `alerts` | System notifications |

### Key Schemas

```sql
-- market_data
CREATE TABLE market_data (
    timestamp TEXT,
    symbol TEXT,
    open REAL, high REAL, low REAL, close REAL,
    volume INTEGER,
    PRIMARY KEY (timestamp, symbol)
);

-- orders
CREATE TABLE orders (
    order_id TEXT PRIMARY KEY,
    timestamp TEXT,
    symbol TEXT,
    side TEXT,
    quantity INTEGER,
    order_type TEXT,
    status TEXT,
    filled_qty INTEGER,
    filled_avg_price REAL,
    commission REAL
);
```

---

## Design Trade-offs

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Database** | SQLite | Lightweight, zero-config. Production: PostgreSQL |
| **Order Type** | Market | Execution certainty. Limit orders for less liquid names |
| **Liquidity Cap** | 10% ADV | Conservative. Could relax to 15-20% for liquid ETFs |
| **Tail Hedge** | Synthetic proxy | Simulates options. Real impl: actual put spreads |
| **Rebalance Freq** | Hourly | Balances freshness vs costs |
| **Benchmark** | Real SPY | More accurate than approximation |
| **Position Sizing** | Tanh | Bounded, smooth. Alternatives: sigmoid, linear |
| **Weight Optim** | SLSQP | Fast, handles constraints. Alt: convex optim |

---

## Performance Metrics

> **Note**: Performance targets are indicative and primarily used to evaluate risk-adjusted behavior rather than absolute returns.

| Metric | Target | Notes |
|--------|--------|-------|
| Sharpe Ratio | > 1.0 | Risk-adjusted returns |
| Maximum Drawdown | < 15% | Capital preservation |
| Win Rate | > 52% | Slight edge sufficient |
| Profit Factor | > 1.2 | Gross profit / gross loss |
| Average Trade | > 5 bps | Net of costs |
| Recovery Factor | > 2.0 | Total return / max DD |

---

## Limitations & Risks

### Known Limitations

| Limitation | Impact & Mitigation |
|------------|---------------------|
| Capacity Constraints | May not scale beyond $10M AUM. Expand universe. |
| Data Dependency | Requires reliable API. Fallback to cached data. |
| Model Risk | Historical patterns may not repeat. Rolling validation. |
| Technology Risk | System failures possible. Automated alerts. |
| Single Broker Execution | IBKR dependency. Could add backup broker. |
| No Options Trading | Synthetic hedge only. Real options better. |

### Key Risks

| Risk Type | Description | Mitigation |
|-----------|-------------|------------|
| Market Risk | Adverse price movements | Drawdown limits, hedging |
| Liquidity Risk | Poor exit prices | 10% ADV constraint |
| Execution Risk | Failed orders | Market orders, monitoring |
| Model Risk | Signal degradation | Health scores, validation |
| Operational Risk | System failures | Alerts, logging |

---

## Future Development

- **Machine Learning**: Gradient-boosted features for regime detection
- **Options Overlay**: Systematic premium collection, real tail hedging
- **Alternative Data**: Sentiment analysis, news processing
- **Multi-Asset**: Futures, FX, cryptocurrency integration
- **Execution Algos**: TWAP, VWAP implementation

---

## Recent Updates (v2.0)

1. ✅ Real SPY benchmark integration (no more approximations)
2. ✅ Liquidity constraints (10% ADV cap)
3. ✅ Enhanced logging with ADV and constraint details
4. ✅ Slippage estimation with square-root model
5. ✅ Improved position transition handling

---

## Author

**Anonym_** - ESSCA School of Management  
MSc Finance & Data Analytics Candidate

## License

© 2026 Anonym_.  
This project is provided for educational and personal use only.
Commercial use requires explicit permission from the author.


Educational and personal use only. Not financial advice.

---

*Last updated: January 2026*
