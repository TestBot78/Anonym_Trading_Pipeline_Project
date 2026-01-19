# Technical Documentation

## Quantitative Trading Pipeline - Deep Technical Reference

> This document provides comprehensive technical specifications for developers, quants, and technical reviewers. For strategic overview, see STRATEGY.md.

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [Data Layer](#2-data-layer)
3. [Signal Computation](#3-signal-computation)
4. [Weight Optimization](#4-weight-optimization)
5. [Position Sizing](#5-position-sizing)
6. [Risk Management](#6-risk-management)
7. [Execution Engine](#7-execution-engine)
8. [Database Schema](#8-database-schema)
9. [Configuration Reference](#9-configuration-reference)
10. [Error Handling](#10-error-handling)
11. [Design Trade-offs](#11-design-trade-offs)

---

## 1. System Architecture

### 1.1 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TRADING SYSTEM v2.0                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                           DATA LAYER                                   │  │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐               │  │
│  │  │ Alpaca API  │    │  IBKR TWS   │    │ SPY Bench   │               │  │
│  │  │ (Data)      │    │ (Execution) │    │ (Real)      │               │  │
│  │  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘               │  │
│  │         └──────────────────┼──────────────────┘                       │  │
│  │                   ┌────────▼────────┐                                 │  │
│  │                   │DualBrokerAdapter│                                 │  │
│  │                   └────────┬────────┘                                 │  │
│  └────────────────────────────┼──────────────────────────────────────────┘  │
│                               │                                              │
│  ┌────────────────────────────▼──────────────────────────────────────────┐  │
│  │                          SIGNAL LAYER                                  │  │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐        │  │
│  │  │Volatility│ │Momentum │ │TailRisk │ │  Beta   │ │  Carry  │        │  │
│  │  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘        │  │
│  │       └───────────┼───────────┼───────────┼───────────┘              │  │
│  │           ┌───────▼───────────▼───────────▼───────┐                  │  │
│  │           │      EnhancedProductionPipeline       │                  │  │
│  │           │  • Validation • Health • Correlation  │                  │  │
│  │           └───────────────────┬───────────────────┘                  │  │
│  │                   ┌───────────▼───────────┐                          │  │
│  │                   │ HybridWeightedPipeline│                          │  │
│  │                   │   70% Daily + 30% H   │                          │  │
│  │                   └───────────┬───────────┘                          │  │
│  └───────────────────────────────┼───────────────────────────────────────┘  │
│                                  │                                          │
│  ┌───────────────────────────────▼───────────────────────────────────────┐  │
│  │                           RISK LAYER                                   │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                   │  │
│  │  │ RiskMonitor │  │StatefulTail │  │  Adaptive   │                   │  │
│  │  │ DD/Leverage │  │   Hedge     │  │ PositionSizer│                  │  │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                   │  │
│  │         └────────────────┼────────────────┘                           │  │
│  └──────────────────────────┼────────────────────────────────────────────┘  │
│                             │                                              │
│  ┌──────────────────────────▼────────────────────────────────────────────┐  │
│  │                       EXECUTION LAYER                                  │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │  │
│  │  │                  PaperTradingEngine                              │ │  │
│  │  │  • Main Loop • Rebalancing • Slippage • Orders                  │ │  │
│  │  └─────────────────────────────────────────────────────────────────┘ │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                   │  │
│  │  │AlertManager │  │TradingDB    │  │  Logger     │                   │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                   │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Code Metrics

| File | Lines | Classes | Functions |
|------|-------|---------|-----------|
| `paper_trading.py` | 1,700 | 5 | 45 |
| `enhanced_pipeline.py` | 1,690 | 12 | 60 |
| `hybrid_pipeline_weighted.py` | 200 | 1 | 8 |
| `dual_broker_adapter.py` | 175 | 1 | 12 |
| `ibkr_adapter.py` | 418 | 1 | 15 |
| **Total** | **~4,200** | **20** | **140** |

---

## 2. Data Layer

### 2.1 DualBrokerAdapter

```python
class DualBrokerAdapter:
    """
    Unified interface:
    - Alpaca: Market data (10Y daily, 2Y hourly)
    - IBKR: Order execution (no PDT)
    """
    
    def __init__(self, alpaca_key, alpaca_secret, alpaca_base_url,
                 ibkr_host, ibkr_port, ibkr_client_id):
        self.alpaca = tradeapi.REST(alpaca_key, alpaca_secret, alpaca_base_url)
        self.ibkr = IBKRAdapter(ibkr_host, ibkr_port, ibkr_client_id)
    
    # Data → Alpaca
    def get_bars(self, symbol, timeframe, **kwargs):
        return self.alpaca.get_bars(symbol, timeframe, **kwargs)
    
    def get_latest_bar(self, symbol):
        return self.alpaca.get_latest_bar(symbol)
    
    # Execution → IBKR
    def submit_order(self, **kwargs):
        return self.ibkr.submit_order(**kwargs)
    
    def list_positions(self):
        return self.ibkr.list_positions()
    
    def get_account(self):
        return self.ibkr.get_account()
```

### 2.2 Real Benchmark Integration

```python
def calculate_signals(self):
    """Fetch real SPY for benchmark"""
    
    try:
        spy_bars = self.api.get_bars('SPY', TimeFrame.Day, limit=500).df
        spy_returns = spy_bars['close'].pct_change()
        self.logger.info(f"Using REAL SPY benchmark ({len(spy_bars)} bars)")
    except Exception as e:
        self.logger.warning(f"SPY fetch failed, using approximation")
        spy_returns = None
    
    for symbol in self.config.symbols:
        bars = self.api.get_bars(symbol, TimeFrame.Day, limit=500).df
        returns = bars['close'].pct_change()
        
        if spy_returns is not None:
            market_returns = spy_returns.reindex(returns.index).ffill()
        else:
            market_returns = returns * 0.6  # Fallback
        
        signals = calculate_daily_signals_weighted(
            returns, market_returns=market_returns
        )
```

### 2.3 IBKR Data Handling

```python
def get_bars(self, symbol, timeframe, limit=None):
    """IBKR-specific duration requirements"""
    
    contract = Stock(symbol, 'SMART', 'USD')
    self.ib.qualifyContracts(contract)
    
    if 'Day' in str(timeframe):
        # CRITICAL: Use YEARS for >365 days
        if limit and limit > 365:
            years = int(limit / 252) + 1
            duration = f'{years} Y'
        else:
            duration = f'{limit} D'
        barsize = '1 day'
    
    elif 'Hour' in str(timeframe):
        # Max 24h per request - need chunking
        if limit and limit > 24:
            all_bars = []
            end_dt = datetime.now()
            remaining = limit
            
            while remaining > 0:
                chunk_size = min(24, remaining)
                bars_chunk = self.ib.reqHistoricalData(
                    contract,
                    endDateTime=end_dt.strftime('%Y%m%d %H:%M:%S'),
                    durationStr=f'{chunk_size * 3600} S',
                    barSizeSetting='1 hour',
                    whatToShow='TRADES',
                    useRTH=True
                )
                all_bars.extend(bars_chunk)
                end_dt = bars_chunk[0].date
                remaining -= chunk_size
                time.sleep(1)  # Rate limit
            
            return pd.DataFrame(all_bars)
```

---

## 3. Signal Computation

### 3.1 Base Signal Class

```python
class QuantSignal:
    """Base class for all alpha signals"""
    
    def __init__(self, config, logger):
        self.name = self.__class__.__name__.replace('Signal', '')
        self.config = config
        self.logger = logger
        self.metrics = {}
        self.health_score = 1.0
        self.pnl_attribution = {}
    
    def compute(self, data: Dict[str, pd.Series]) -> pd.Series:
        raise NotImplementedError
    
    def validate_rolling_vectorized(self, signal, returns, window=252):
        """Rolling validation metrics"""
        
        # Information Coefficient
        rolling_ic = signal.rolling(window).corr(returns.shift(-1))
        
        # Hit Rate
        signal_dir = (signal > 0).astype(int)
        returns_dir = (returns.shift(-1) > 0).astype(int)
        rolling_hit_rate = (signal_dir == returns_dir).rolling(window).mean()
        
        # Turnover-adjusted IC
        turnover = signal.diff().abs().rolling(window).mean()
        rolling_turnover_adj_ic = rolling_ic / (1 + turnover * 100)
        
        self.metrics['rolling_ic'] = rolling_ic
        self.metrics['rolling_hit_rate'] = rolling_hit_rate
        self.metrics['rolling_turnover_adj_ic'] = rolling_turnover_adj_ic
    
    def compute_health_score(self, lookback=63):
        """Path-dependent health score (5 components)"""
        
        ic_series = self.metrics['rolling_ic'].dropna()
        
        recent_ic = ic_series.iloc[-lookback:].mean()
        recent_ic_std = ic_series.iloc[-lookback:].std()
        recent_turnover_ic = self.metrics['rolling_turnover_adj_ic'].iloc[-lookback:].mean()
        
        # Trend (velocity)
        past_ic = ic_series.iloc[-lookback*2:-lookback].mean()
        ic_trend = recent_ic - past_ic
        
        # Acceleration
        very_past_ic = ic_series.iloc[-lookback*3:-lookback*2].mean()
        ic_acceleration = (recent_ic - past_ic) - (past_ic - very_past_ic)
        
        # Component scores (sigmoid)
        ic_level_score = expit((recent_ic - 0.00) / 0.015)
        stability_score = expit(-recent_ic_std / 0.02)
        turnover_score = expit((recent_turnover_ic - 0.00) / 0.01)
        trend_score = expit(ic_trend / 0.01)
        accel_score = expit(ic_acceleration / 0.005)
        
        # Weighted composite
        self.health_score = (
            0.35 * ic_level_score +
            0.20 * stability_score +
            0.15 * turnover_score +
            0.20 * trend_score +
            0.10 * accel_score
        )
        
        return self.health_score
```

### 3.2 Signal Implementations

#### Volatility Signal

```python
class VolatilitySignal(QuantSignal):
    """Mean reversion on realized volatility"""
    
    def compute(self, data):
        returns = data['returns']
        window = self.config['volatility_window']  # 20
        
        realized_vol = returns.rolling(window).std() * np.sqrt(252)
        vol_of_vol = realized_vol.rolling(window).std()
        
        vol_zscore = (
            (realized_vol - realized_vol.rolling(252).mean()) / 
            (realized_vol.rolling(252).std() + 1e-8)
        )
        
        # Short vol spikes (mean reversion)
        signal = -vol_zscore * (1 + vol_of_vol)
        
        return signal.fillna(0)
```

#### Momentum Signal

```python
class MomentumSignal(QuantSignal):
    """Excess returns vs real SPY benchmark"""
    
    def compute(self, data):
        returns = data['returns']
        fast = self.config['momentum_fast']  # 20
        slow = self.config['momentum_slow']  # 60
        
        momentum_fast = returns.rolling(fast).sum()
        momentum_slow = returns.rolling(slow).sum()
        
        # Excess vs REAL SPY benchmark
        market_returns = data.get('market_returns', None)
        if market_returns is not None:
            benchmark_momentum = market_returns.rolling(fast).sum()
            excess_momentum = momentum_fast - benchmark_momentum
        else:
            excess_momentum = momentum_fast
        
        momentum_diff = momentum_fast - momentum_slow
        signal = 0.7 * excess_momentum + 0.3 * momentum_diff
        
        return signal.fillna(0)
```

#### Tail Risk Signal

```python
class TailRiskSignal(QuantSignal):
    """CVaR-based with asymmetry adjustment"""
    
    def compute(self, data):
        returns = data['returns']
        window = self.config['tail_window']  # 60
        
        expected_shortfall = returns.rolling(window).quantile(0.05)
        left_tail = returns.rolling(window).quantile(0.05)
        right_tail = returns.rolling(window).quantile(0.95)
        tail_asymmetry = (right_tail + left_tail) / (right_tail - left_tail + 1e-8)
        
        signal = -expected_shortfall * tail_asymmetry
        
        return signal.fillna(0)
```

#### Beta Signal

```python
class BetaSignal(QuantSignal):
    """Rolling beta vs real SPY"""
    
    def compute(self, data):
        returns = data['returns']
        window = self.config['beta_window']  # 60
        market_returns = data.get('market_returns', returns * 0.6)
        
        rolling_cov = returns.rolling(window).cov(market_returns)
        market_var = market_returns.rolling(window).var()
        
        beta = rolling_cov / (market_var + 1e-8)
        beta_vol = beta.rolling(window).std()
        
        # Short high and unstable beta
        signal = -beta * (1 + beta_vol)
        
        return signal.fillna(0)
```

#### Carry Signal

```python
class CarrySignal(QuantSignal):
    """VRP + Mean Reversion + Seasonality"""
    
    def compute(self, data):
        returns = data['returns']
        prices = data['close']
        
        # VRP Proxy (50%)
        realized_vol = returns.rolling(20).std() * np.sqrt(252)
        implied_vol_proxy = returns.rolling(5).std() * np.sqrt(252) * 1.2
        vrp = implied_vol_proxy - realized_vol
        
        # Mean Reversion (30%)
        ma_252 = prices.rolling(252).mean()
        mean_reversion = -(prices - ma_252) / (ma_252 + 1e-8)
        
        # Seasonality (20%)
        day_of_year = pd.Series(prices.index.dayofyear, index=prices.index)
        seasonality = np.sin(2 * np.pi * day_of_year / 365)
        
        signal = 0.5 * vrp + 0.3 * mean_reversion + 0.2 * seasonality
        
        return signal.fillna(0)
```

### 3.3 Hybrid Multi-Timeframe Pipeline

```python
class HybridWeightedPipeline:
    """70% daily + 30% hourly"""
    
    def calculate_hourly_signals(self, hourly_data):
        returns = hourly_data['close'].pct_change()
        
        # Hourly momentum (20 bars)
        momentum = returns.rolling(20).sum()
        
        # Hourly volatility
        volatility = -returns.rolling(20).std()
        
        # RSI (14 bars)
        delta = returns.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + gain / (loss + 1e-8)))
        rsi_signal = (rsi - 50) / 50
        
        # Price action
        high_20 = hourly_data['high'].rolling(20).max()
        low_20 = hourly_data['low'].rolling(20).min()
        price_action = (hourly_data['close'] - low_20) / (high_20 - low_20 + 1e-8) - 0.5
        
        return 0.25 * (momentum + volatility + rsi_signal + price_action)
    
    def combine_signals(self, daily_signal, hourly_signal):
        hourly_daily = hourly_signal.resample('D').last()
        composite = 0.7 * daily_signal + 0.3 * hourly_daily
        
        same_direction = np.sign(daily_signal) == np.sign(hourly_daily)
        confidence = np.where(same_direction, 0.7 + 0.3 * abs(hourly_daily), 0.3)
        
        return composite, confidence
```

---

## 4. Weight Optimization

### 4.1 Correlation-Aware Weighting

```python
class DynamicCorrelationWeighter:
    """SLSQP: maximize IC, penalize correlation"""
    
    def compute_dynamic_weights(self, signals_df, returns, base_weights):
        signal_corr = signals_df.corr()
        
        signal_ics = {}
        for col in signals_df.columns:
            ic = signals_df[col].corr(returns.shift(-1))
            signal_ics[col] = ic if not np.isnan(ic) else 0
        
        def objective(weights):
            weighted_ic = sum(w * signal_ics.get(name, 0) 
                            for name, w in zip(signals_df.columns, weights))
            
            avg_corr = 0
            n_pairs = 0
            for i, name_i in enumerate(signals_df.columns):
                for j, name_j in enumerate(signals_df.columns):
                    if i < j:
                        corr_ij = signal_corr.loc[name_i, name_j]
                        avg_corr += weights[i] * weights[j] * abs(corr_ij)
                        n_pairs += 1
            
            if n_pairs > 0:
                avg_corr /= n_pairs
            
            return -weighted_ic + 0.5 * avg_corr
        
        n_signals = len(signals_df.columns)
        constraints = [{'type': 'eq', 'fun': lambda w: sum(w) - 1.0}]
        bounds = [(0.05, 0.5) for _ in range(n_signals)]
        
        x0 = [base_weights.get(col, 1/n_signals) for col in signals_df.columns]
        result = minimize(objective, x0, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        return {col: w for col, w in zip(signals_df.columns, result.x)}
```

### 4.2 Risk Budget Allocator

```python
class RiskBudgetAllocator:
    """Category-level risk budgets"""
    
    risk_budgets = {
        'Momentum': 0.30,
        'Carry': 0.25,
        'Volatility': 0.20,
        'TailRisk': 0.15,
        'Beta': 0.10
    }
    
    def apply_risk_budgets(self, weights, signal_categories):
        category_weights = {}
        for signal, weight in weights.items():
            category = signal_categories.get(signal, signal)
            category_weights[category] = category_weights.get(category, 0) + weight
        
        modified_weights = weights.copy()
        
        for category, total_weight in category_weights.items():
            budget = self.risk_budgets.get(category, 1.0)
            
            if total_weight > budget:
                scale_factor = budget / total_weight
                for signal, weight in weights.items():
                    if signal_categories.get(signal, signal) == category:
                        modified_weights[signal] = weight * scale_factor
        
        total = sum(modified_weights.values())
        return {k: v/total for k, v in modified_weights.items()}
```

---

## 5. Position Sizing

### 5.1 Liquidity-Aware Algorithm

```python
def calculate_target_positions(self, signals):
    """Dual constraints: portfolio + liquidity"""
    
    account = self.api.get_account()
    equity = float(account.equity)
    
    targets = {}
    target_exposures = {}
    
    for symbol, composite_signal in signals.items():
        bars = self.api.get_bars(symbol, TimeFrame.Day, limit=20).df
        price = bars['close'].iloc[-1]
        adv_20 = bars['volume'].mean()
        
        # 1. Signal strength [-1, +1]
        signal_strength = np.tanh(composite_signal * 2)
        
        # 2. Confidence adjustment
        confidence = signals.get(f'{symbol}_confidence', 1.0)
        adjusted_strength = signal_strength * confidence
        
        # 3. Dynamic leverage
        leverage = (self.config.min_leverage + 
                   (self.config.max_leverage - self.config.min_leverage) * 
                   abs(adjusted_strength))
        signed_leverage = np.sign(adjusted_strength) * leverage
        
        # 4. Target notional
        target_notional = equity * signed_leverage * self.config.max_position_size
        
        # 5. Raw shares
        raw_shares = int(target_notional / price)
        
        # 6. LIQUIDITY CONSTRAINT: 10% of ADV
        max_shares_liquidity = int(adv_20 * 0.10)
        
        # 7. PORTFOLIO CONSTRAINT
        max_shares_portfolio = int(
            equity * self.config.max_position_size * 
            self.config.max_leverage / price
        )
        
        # 8. Apply both
        max_shares = min(max_shares_portfolio, max_shares_liquidity)
        final_shares = np.clip(raw_shares, -max_shares, max_shares)
        
        if abs(raw_shares) > max_shares_liquidity:
            self.logger.warning(
                f'{symbol}: Liquidity constraint binding! '
                f'Raw={raw_shares}, Cap={max_shares_liquidity}'
            )
        
        targets[symbol] = final_shares
        target_exposures[symbol] = abs(final_shares * price)
    
    # Scale if leverage exceeded
    total_exposure = sum(target_exposures.values())
    target_leverage = total_exposure / equity
    
    if target_leverage > self.config.max_leverage:
        scale_factor = self.config.max_leverage / target_leverage
        for symbol in targets:
            targets[symbol] = int(targets[symbol] * scale_factor)
    
    return targets
```

### 5.2 Adaptive Position Decay

```python
class AdaptivePositionSizer:
    """Exponential decay with buffer zone"""
    
    decay_halflife = 10  # days
    buffer_zone = 0.15   # 15%
    
    def compute_positions_with_decay(self, signal, returns):
        positions = pd.Series(0.0, index=signal.index)
        position_ages = pd.Series(0, index=signal.index)
        
        current_position = 0
        position_age = 0
        
        for i, (date, sig) in enumerate(signal.items()):
            if i == 0:
                continue
            
            raw_position = np.sign(sig) * min(abs(sig), 1.0)
            decay_factor = 0.5 ** (position_age / self.decay_halflife)
            decayed_position = raw_position * decay_factor
            
            if current_position != 0:
                pct_change = abs(decayed_position - current_position) / abs(current_position)
                if pct_change < self.buffer_zone:
                    decayed_position = current_position
                    position_age += 1
                else:
                    position_age = 0
            else:
                position_age = 0
            
            current_position = decayed_position
            positions.iloc[i] = decayed_position
            position_ages.iloc[i] = position_age
        
        return positions, position_ages
```

---

## 6. Risk Management

### 6.1 Stateful Tail Hedge

```python
class StatefulTailHedge:
    """State machine for tail protection"""
    
    INACTIVE = 'INACTIVE'
    ACTIVE = 'ACTIVE'
    
    def __init__(self, config, logger):
        self.state = self.INACTIVE
        self.activation_level = None
        self.hedge_size = config['hedge_size']  # 0.10
        self.cost_per_hedge = config['cost_per_hedge']  # 0.0002
        self.activation_threshold = config['hedge_threshold']  # -0.05
        self.deactivation_threshold = -0.025
    
    def apply_tail_hedge_stateful(self, returns, market_returns, drawdowns):
        hedged_returns = returns.copy()
        
        for i, (date, dd) in enumerate(drawdowns.items()):
            mkt_ret = market_returns.iloc[i] if i < len(market_returns) else 0
            
            # State transitions
            if self.state == self.INACTIVE and dd < self.activation_threshold:
                self.state = self.ACTIVE
                self.activation_level = dd
                self.logger.info(f"Tail hedge ACTIVATED at DD={dd:.2%}")
            
            elif self.state == self.ACTIVE and dd > self.deactivation_threshold:
                self.state = self.INACTIVE
                self.logger.info(f"Tail hedge DEACTIVATED at DD={dd:.2%}")
            
            # Apply hedge if active
            if self.state == self.ACTIVE:
                downside = max(0, -mkt_ret)
                hedge_pnl = self.hedge_size * (downside ** 1.5) * 3  # Convex
                hedge_cost = self.cost_per_hedge
                hedged_returns.iloc[i] = returns.iloc[i] + hedge_pnl - hedge_cost
        
        return hedged_returns
```

### 6.2 Risk Monitor

```python
class RiskMonitor:
    """Real-time risk monitoring"""
    
    def __init__(self, config, alert_manager):
        self.config = config
        self.alert_manager = alert_manager
        self.daily_peak = None
        self.overall_peak = None
    
    def check_risk_limits(self, account, positions):
        portfolio_value = float(account['portfolio_value'])
        
        if self.daily_peak is None:
            self.daily_peak = portfolio_value
        if self.overall_peak is None:
            self.overall_peak = portfolio_value
        
        self.overall_peak = max(self.overall_peak, portfolio_value)
        
        # Daily P&L check
        daily_pnl = float(account.get('daily_pnl', 0))
        daily_pnl_pct = daily_pnl / self.daily_peak
        
        if daily_pnl_pct < self.config.max_daily_loss:
            self.alert_manager.alert(
                "DAILY LOSS LIMIT BREACHED",
                f"Daily P&L: {daily_pnl_pct:.2%}",
                level="CRITICAL"
            )
            return False
        
        # Drawdown check
        drawdown = (portfolio_value - self.overall_peak) / self.overall_peak
        
        if drawdown < self.config.max_drawdown:
            self.alert_manager.alert(
                "MAX DRAWDOWN BREACHED",
                f"Drawdown: {drawdown:.2%}",
                level="CRITICAL"
            )
            return False
        
        # Leverage check
        equity = float(account['equity'])
        total_exposure = sum(abs(float(p.get('market_value', 0))) 
                           for p in positions.values())
        leverage = total_exposure / equity if equity > 0 else 0
        
        if leverage > self.config.max_leverage:
            self.alert_manager.alert(
                "LEVERAGE EXCEEDED",
                f"Leverage: {leverage:.2f}x",
                level="WARNING"
            )
            return False
        
        return True
```

### 6.3 Slippage Estimation

```python
def estimate_slippage(self, symbol, shares, adv, volatility):
    """Square-root market impact model"""
    
    if adv == 0:
        return 100.0  # 1% if no volume
    
    participation = abs(shares) / adv
    base_spread_bps = 10.0  # Large cap assumption
    
    # Impact ∝ sqrt(participation) * volatility
    impact_bps = 10.0 * np.sqrt(participation) * (volatility / 0.20)
    
    total_slippage_bps = base_spread_bps/2 + impact_bps
    return min(total_slippage_bps, 100.0)  # Cap at 1%

def log_estimated_costs(self, targets):
    """Pre-trade cost analysis"""
    
    total_cost_usd = 0
    
    for symbol, target_shares in targets.items():
        if target_shares == 0:
            continue
        
        bars = self.api.get_bars(symbol, TimeFrame.Day, limit=20).df
        price = bars['close'].iloc[-1]
        adv = bars['volume'].mean()
        volatility = bars['close'].pct_change().std() * np.sqrt(252)
        
        current_qty = self.get_current_position(symbol)
        delta = target_shares - current_qty
        
        if delta == 0:
            continue
        
        slippage_bps = self.estimate_slippage(symbol, delta, adv, volatility)
        notional = abs(delta) * price
        cost_usd = notional * (slippage_bps / 10000)
        
        total_cost_usd += cost_usd
        
        self.logger.info(
            f'{symbol}: {abs(delta):,} shares @ ${price:.2f}\n'
            f'   Participation: {abs(delta)/adv:.2%} of ADV\n'
            f'   Est. Slippage: {slippage_bps:.1f} bps\n'
            f'   Est. Cost: ${cost_usd:.2f}'
        )
    
    self.logger.info(f'TOTAL ESTIMATED COST: ${total_cost_usd:.2f}')
```

---

## 7. Execution Engine

### 7.1 Order Flow

```python
def execute_rebalance(self, target_positions):
    """Execute with position transitions"""
    
    # Pre-checks
    self.check_and_reduce_leverage()
    self.log_estimated_costs(target_positions)
    
    # Buying power check
    account = self.api.get_account()
    buying_power = float(account.buying_power)
    
    total_notional_needed = 0
    for symbol, target_qty in target_positions.items():
        current_qty = self.get_current_position(symbol)
        delta = target_qty - current_qty
        if abs(delta) >= self.config.min_trade_size:
            price = self.get_price(symbol)
            total_notional_needed += abs(delta * price)
    
    # Scale if insufficient
    if total_notional_needed > buying_power:
        scale_factor = (buying_power / total_notional_needed) * 0.95
        self.logger.warning(f"Scaling down by {scale_factor:.2%}")
        target_positions = {
            s: int(q * scale_factor) for s, q in target_positions.items()
        }
    
    # Execute
    for symbol, target_qty in target_positions.items():
        current_qty = self.get_current_position(symbol)
        delta = target_qty - current_qty
        
        if abs(delta) < self.config.min_trade_size:
            continue
        
        # Handle transitions
        if current_qty < 0 and target_qty > 0:
            # SHORT → LONG
            self.submit_order(symbol, abs(current_qty), 'buy')  # Cover
            time.sleep(2)
            self.submit_order(symbol, target_qty, 'buy')  # Open long
        
        elif current_qty > 0 and target_qty < 0:
            # LONG → SHORT
            self.submit_order(symbol, current_qty, 'sell')  # Close
            time.sleep(2)
            self.submit_order(symbol, abs(target_qty), 'sell')  # Open short
        
        else:
            # Simple adjustment
            side = 'buy' if delta > 0 else 'sell'
            self.submit_order(symbol, abs(delta), side)
    
    # Post-check
    self.check_and_reduce_leverage()
```

### 7.2 Order Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Order Type | Market | Execution certainty |
| TIF (Open) | DAY | Standard |
| TIF (Closed) | GTC | Queue for open |
| Min Trade | 1 share | Configurable |
| Fill Timeout | 30 sec | Reasonable wait |

---

## 8. Database Schema

### 8.1 Tables

```sql
-- market_data: OHLCV cache
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

-- signals: Historical signals
CREATE TABLE signals (
    timestamp TEXT PRIMARY KEY,
    composite_signal REAL,
    volatility_signal REAL,
    momentum_signal REAL,
    tail_risk_signal REAL,
    beta_signal REAL,
    carry_signal REAL
);

-- positions: Snapshots
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

-- orders: History
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

-- portfolio: Snapshots
CREATE TABLE portfolio (
    timestamp TEXT PRIMARY KEY,
    cash REAL,
    portfolio_value REAL,
    equity REAL,
    buying_power REAL,
    daily_pnl REAL,
    total_pnl REAL
);

-- alerts: Notifications
CREATE TABLE alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    level TEXT,
    message TEXT
);
```

---

## 9. Configuration Reference

### 9.1 trading_config.json

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `alpaca_api_key` | string | - | Alpaca API key |
| `alpaca_api_secret` | string | - | Alpaca secret |
| `alpaca_base_url` | string | paper-api | Endpoint |
| `symbols` | array | ["AAPL",...] | Universe |
| `rebalance_frequency_minutes` | int | 60 | Rebalance interval |
| `min_trade_size` | int | 1 | Min shares |
| `max_position_size` | float | 0.20 | 20% max |
| `max_daily_loss` | float | -0.05 | -5% limit |
| `max_drawdown` | float | -0.15 | -15% limit |
| `min_leverage` | float | 1.0 | Min leverage |
| `max_leverage` | float | 3.0 | Max leverage |

### 9.2 Pipeline Config

| Parameter | Default | Description |
|-----------|---------|-------------|
| `initial_capital` | 1,000,000 | Backtest capital |
| `max_adv_pct` | 0.10 | 10% ADV cap |
| `hedge_threshold` | -0.05 | Tail hedge activation |
| `hedge_size` | 0.10 | Hedge notional |
| `cost_per_hedge` | 0.0002 | 2 bps daily |
| `volatility_window` | 20 | Vol lookback |
| `momentum_fast` | 20 | Fast window |
| `momentum_slow` | 60 | Slow window |
| `validation_window` | 252 | Validation window |
| `health_lookback` | 63 | Health score lookback |
| `costs_bps` | 5 | Cost assumption |

---

## 10. Error Handling

### 10.1 Common Errors

| Error | Cause | Resolution |
|-------|-------|------------|
| `insufficient buying power` | Orders exceed margin | Scale down positions |
| `symbol not found` | Invalid ticker | Verify symbol exists |
| `market closed` | Trading outside hours | Use GTC orders |
| `connection timeout` | IBKR disconnect | Reconnect with retry |
| `rate limit exceeded` | Too many API calls | Add delays |

### 10.2 Recovery Strategy

```python
def safe_api_call(self, func, *args, max_retries=3, **kwargs):
    """Retry wrapper for API calls"""
    
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self.logger.warning(f"API call failed (attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                self.logger.error(f"API call failed after {max_retries} attempts")
                raise
```

---

## 11. Design Trade-offs

| Decision | Choice | Rationale | Alternative |
|----------|--------|-----------|-------------|
| **Database** | SQLite | Zero-config, lightweight | PostgreSQL for production |
| **Order Type** | Market | Execution certainty | Limit for less liquid |
| **Liquidity Cap** | 10% ADV | Conservative impact | 15-20% for liquid ETFs |
| **Tail Hedge** | Synthetic | Simulates options | Real put spreads |
| **Rebalance** | Hourly | Freshness vs costs | Daily for lower costs |
| **Benchmark** | Real SPY | Accurate signals | Approximation faster |
| **Position Sizing** | Tanh | Bounded, smooth | Sigmoid, linear |
| **Optimization** | SLSQP | Fast, constrained | Convex, genetic |

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
