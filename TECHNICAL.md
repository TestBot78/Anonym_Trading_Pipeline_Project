# Technical Documentation

## System Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PAPER TRADING ENGINE                         │
│                        (paper_trading.py)                           │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │  TradingConfig  │  │  TradingDatabase │  │    AlertManager    │  │
│  │  - API keys     │  │  - SQLite3       │  │  - Email/Slack     │  │
│  │  - Risk limits  │  │  - Market data   │  │  - Log levels      │  │
│  │  - Symbols      │  │  - Positions     │  │                    │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                    PaperTradingEngine                           ││
│  │  - calculate_signals()      → Uses real SPY benchmark           ││
│  │  - calculate_target_positions() → Liquidity constraints         ││
│  │  - execute_rebalance()                                          ││
│  │  - check_and_reduce_leverage()                                  ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     DUAL BROKER ADAPTER                             │
│                   (dual_broker_adapter.py)                          │
├─────────────────────────────────────────────────────────────────────┤
│  ┌───────────────────────┐    ┌───────────────────────────────────┐ │
│  │     ALPACA (Data)     │    │        IBKR (Execution)           │ │
│  │  - get_bars()         │    │  - submit_order()                 │ │
│  │  - get_latest_bar()   │    │  - list_positions()               │ │
│  │  - get_clock()        │    │  - get_account()                  │ │
│  │  - SPY benchmark      │    │  - NO PDT RESTRICTIONS            │ │
│  └───────────────────────┘    └───────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    SIGNAL PIPELINE                                  │
│               (enhanced_pipeline.py)                                │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              EnhancedProductionPipeline                     │    │
│  │  - run_full_pipeline_vectorized()                           │    │
│  │  - get_performance_metrics()                                │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                      │
│       ┌──────────────────────┼──────────────────────┐               │
│       ▼                      ▼                      ▼               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐   │
│  │Volatility│  │ Momentum │  │ TailRisk │  │   Beta   │  │Carry │   │
│  │  Signal  │  │  Signal  │  │  Signal  │  │  Signal  │  │Signal│   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

## Signal Computation Details

### Real Benchmark Integration

The system now uses actual SPY returns instead of approximations:

```python
def calculate_signals(self) -> Dict:
    # Fetch real SPY benchmark
    try:
        spy_bars = self.api.get_bars('SPY', tradeapi.TimeFrame.Day, limit=500).df
        spy_returns = spy_bars['close'].pct_change()
        self.logger.info(f'Using real SPY benchmark ({len(spy_returns)} bars)')
    except Exception as e:
        self.logger.warning(f'Failed to get SPY: {e}, using approximation')
        spy_returns = None
    
    for symbol in self.config.symbols:
        hybrid_result = self.hybrid_weighted.calculate_hybrid_weighted_signal(
            symbol, 
            market_returns=spy_returns  # Pass real benchmark
        )
```

### 1. Volatility Signal

```python
def compute(data):
    returns = data['returns']
    window = 20
    
    # Realized volatility (annualized)
    realized_vol = returns.rolling(window).std() * sqrt(252)
    
    # Volatility of volatility
    vol_of_vol = realized_vol.rolling(window).std()
    
    # Z-score normalization
    vol_zscore = (realized_vol - realized_vol.rolling(252).mean()) / 
                 (realized_vol.rolling(252).std() + 1e-8)
    
    # Signal: short vol spikes (mean reversion)
    signal = -vol_zscore * (1 + vol_of_vol)
    
    return signal
```

### 2. Momentum Signal (with Real Benchmark)

```python
def compute(data):
    returns = data['returns']
    market_returns = data['market_returns']  # Real SPY returns
    fast, slow = 20, 60
    
    momentum_fast = returns.rolling(fast).sum()
    momentum_slow = returns.rolling(slow).sum()
    
    # Excess vs REAL benchmark (not approximation)
    benchmark_momentum = market_returns.rolling(fast).sum()
    excess_momentum = momentum_fast - benchmark_momentum
    
    # Combined signal
    signal = 0.7 * excess_momentum + 0.3 * (momentum_fast - momentum_slow)
    
    return signal
```

### 3. Tail Risk Signal

```python
def compute(data):
    returns = data['returns']
    window = 60
    
    # Expected Shortfall (CVaR at 5%)
    es = returns.rolling(window).quantile(0.05)
    
    # Tail asymmetry
    left_tail = returns.rolling(window).quantile(0.05)
    right_tail = returns.rolling(window).quantile(0.95)
    tail_asymmetry = (right_tail + left_tail) / (right_tail - left_tail)
    
    # Signal: short tail risk
    signal = -es * tail_asymmetry
    
    return signal
```

### 4. Beta Signal (with Real Benchmark)

```python
def compute(data):
    returns = data['returns']
    market_returns = data['market_returns']  # Real SPY returns
    window = 60
    
    # Rolling beta against REAL market
    rolling_cov = returns.rolling(window).cov(market_returns)
    market_var = market_returns.rolling(window).var()
    beta = rolling_cov / (market_var + 1e-8)
    
    # Beta volatility
    beta_vol = beta.rolling(window).std()
    
    # Signal: short high beta
    signal = -beta * (1 + beta_vol)
    
    return signal
```

### 5. Carry Signal

```python
def compute(data):
    returns = data['returns']
    price = data['close']
    window = 20
    
    # Volatility Risk Premium proxy
    realized_vol = returns.rolling(window).std()
    implied_vol_proxy = returns.rolling(window * 2).std()
    vrp = implied_vol_proxy - realized_vol
    
    # Mean reversion
    price_ma = price.rolling(252).mean()
    distance = (price - price_ma) / price_ma
    mean_reversion = -distance
    
    # Seasonality
    seasonality = sin(2π × day_of_year / 365)
    
    # Combined
    signal = 0.5 * vrp + 0.3 * mean_reversion + 0.2 * seasonality
    
    return signal
```

## Position Sizing with Liquidity Constraints

### New Liquidity-Aware Algorithm

```python
def calculate_target_positions(self, signals: Dict) -> Dict[str, int]:
    for symbol, signal_dict in signals.items():
        # Get price AND volume data
        bars = self.api.get_bars(symbol, TimeFrame.Day, limit=20).df
        price = bars['close'].iloc[-1]
        
        # Calculate 20-day Average Daily Volume
        adv_20 = bars['volume'].mean()
        
        # Max shares from liquidity (10% of ADV)
        max_shares_liquidity = int(adv_20 * 0.10)
        
        # Max shares from portfolio constraints
        max_shares_portfolio = int(
            equity * max_position_size * max_leverage / price
        )
        
        # Signal-based target
        signal_strength = tanh(composite * 2)
        adjusted_strength = signal_strength * confidence
        leverage = min_leverage + (max_leverage - min_leverage) * abs(adjusted_strength)
        target_notional = equity * leverage * max_position_size
        shares_from_signal = int(target_notional / price)
        
        # Apply BOTH constraints
        final_shares = clip(
            shares_from_signal,
            -min(max_shares_portfolio, max_shares_liquidity),
            min(max_shares_portfolio, max_shares_liquidity)
        )
        
        # Log if liquidity constrained
        if abs(shares_from_signal) > max_shares_liquidity:
            logger.warning(f'{symbol}: Liquidity constrained! '
                          f'Signal wants {shares_from_signal} but ADV allows {max_shares_liquidity}')
```

### Constraint Priority

1. **Liquidity constraint** (10% ADV) - Prevents market impact
2. **Portfolio constraint** (20% position size × leverage) - Diversification
3. **Leverage constraint** (max 3x) - Risk management

## Hybrid Pipeline with Real Benchmark

### Updated Data Flow

```python
class HybridWeightedPipeline:
    def calculate_daily_signals_weighted(self, bars, market_returns=None):
        prices = bars['close']
        returns = prices.pct_change()
        
        # Use real market returns if provided
        if market_returns is not None and len(market_returns) > 0:
            # Align market returns with stock returns
            if len(market_returns) >= len(returns):
                market_returns_aligned = market_returns.iloc[-len(returns):]
                market_returns_aligned.index = returns.index
            else:
                market_returns_aligned = returns * 0.6  # Fallback
            
            self.logger.info(f'Using REAL SPY benchmark ({len(market_returns_aligned)} bars)')
        else:
            market_returns_aligned = returns * 0.6  # Approximation
        
        # Pass to signals
        data = {
            'returns': returns, 
            'close': prices, 
            'market_returns': market_returns_aligned
        }
```

## Database Schema

### Tables

```sql
-- Market data cache
CREATE TABLE market_data (
    timestamp TEXT,
    symbol TEXT,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume INTEGER,
    PRIMARY KEY (timestamp, symbol)
);

-- Signal history
CREATE TABLE signals (
    timestamp TEXT PRIMARY KEY,
    composite_signal REAL,
    volatility_signal REAL,
    momentum_signal REAL,
    tail_risk_signal REAL,
    beta_signal REAL,
    carry_signal REAL
);

-- Position snapshots
CREATE TABLE positions (
    timestamp TEXT,
    symbol TEXT,
    quantity INTEGER,
    avg_entry_price REAL,
    current_price REAL,
    market_value REAL,
    unrealized_pnl REAL,
    PRIMARY KEY (timestamp, symbol)
);

-- Order history
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

-- Portfolio snapshots
CREATE TABLE portfolio (
    timestamp TEXT PRIMARY KEY,
    cash REAL,
    portfolio_value REAL,
    equity REAL,
    buying_power REAL,
    daily_pnl REAL,
    total_pnl REAL
);
```

## Configuration Parameters

### trading_config.json

```json
{
  "alpaca_api_key": "...",
  "alpaca_api_secret": "...",
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
  "log_file": "paper_trading_ibkr.log",
  "log_level": "INFO"
}
```

### Pipeline Configuration

```python
PipelineConfig = {
    # Capital & Risk
    'initial_capital': 1_000_000,
    'max_gross_leverage': 3.0,
    'max_net_leverage': 1.5,
    
    # Liquidity Constraints
    'max_adv_pct': 0.10,  # 10% of ADV
    'max_daily_turnover': 0.50,
    
    # Risk Budgets
    'risk_budgets': {
        'Momentum': 0.30,
        'Carry': 0.25,
        'Volatility': 0.20,
        'TailRisk': 0.15,
        'Beta': 0.10
    },
    
    # Signal Parameters
    'volatility_window': 20,
    'momentum_fast': 20,
    'momentum_slow': 60,
    'tail_window': 60,
    'beta_window': 60,
    'carry_window': 20,
}
```

## Error Handling

### Design Trade-offs

> **Why these choices?**
> - **SQLite**: Lightweight, zero-config, sufficient for single-user paper trading. Production would use PostgreSQL/TimescaleDB.
> - **Market orders**: Prioritizes execution certainty over price improvement. Acceptable for liquid large-caps; limit orders would be better for less liquid names.
> - **10% ADV cap**: Conservative threshold to ensure minimal market impact. Could be relaxed to 15-20% for highly liquid ETFs.
> - **Proxy tail hedge**: Uses synthetic convex payoff instead of actual options due to complexity of options pricing/execution. Real implementation would use put spreads.
> - **Hourly rebalancing**: Balances signal freshness with transaction costs. Higher frequency would increase costs without proportional alpha improvement.

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `insufficient day trading buying power` | PDT rule on Alpaca | Use IBKR or cash account |
| `Liquidity constrained` | Position > 10% ADV | Normal - system auto-caps |
| `Failed to get SPY benchmark` | API timeout | Falls back to approximation |
| `Market closed` | Trading outside hours | Wait for market open |

### Logging Output Example

```
INFO - Using real SPY benchmark (500 bars)
INFO - HYBRID WEIGHTED CALCULATION: AAPL
INFO - Running FULL PIPELINE on daily data...
INFO - Daily composite (weighted): 0.1234
INFO - Hourly composite: 0.0567
INFO - HYBRID SUMMARY: Trend=BULLISH, Timing=BUY, Signal=0.1034, Confidence=75.00%

INFO - AAPL:
INFO -    Signal:           +0.1034
INFO -    Confidence:       75.00%
INFO -    Leverage Used:    2.15x
INFO -    ADV (20d):        45,000,000 shares
INFO -    Max (liquidity):  4,500,000 shares
INFO -    Max (portfolio):  850 shares
INFO -    Target Shares:    850 shares
```
---

*Disclaimer: This is an educational project. Past performance does not guarantee future results. Trading involves risk of loss.*

---

**Author**: Anonym_  
**Institution**: ESSCA School of Management  
**Program**: MSc Finance & Data Analytics candidate
**Date**: January 2026

### Intellectual Property Notice

This project was developed independently for educational and
career development purposes. All intellectual property remains
with the author.
