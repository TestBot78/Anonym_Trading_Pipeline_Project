# Strategy Documentation

## Quantitative Trading Pipeline - Investment Strategy & Risk Framework

> This document provides the strategic rationale, investment philosophy, and risk framework for the trading system. For technical implementation, see TECHNICAL.md.

---

## Table of Contents

1. [Executive Overview](#1-executive-overview)
2. [Investment Philosophy](#2-investment-philosophy)
3. [Alpha Generation](#3-alpha-generation)
4. [Signal Processing Pipeline](#4-signal-processing-pipeline)
5. [Risk Framework](#5-risk-framework)
6. [Position Management](#6-position-management)
7. [Execution Strategy](#7-execution-strategy)
8. [Stress Testing & Scenarios](#8-stress-testing--scenarios)
9. [Expected Performance](#9-expected-performance)
10. [Competitive Advantages](#10-competitive-advantages)
11. [Limitations & Risks](#11-limitations--risks)
12. [Use Cases](#12-use-cases)
13. [Future Development](#13-future-development)

---

## 1. Executive Overview

### 1.1 Strategy Summary

This quantitative trading pipeline implements a **multi-factor systematic equity strategy** designed for risk-adjusted returns across varying market conditions. The system combines five orthogonal alpha factors with dynamic risk management and liquidity-aware position sizing.

### 1.2 Core Principles

| Principle | Implementation |
|-----------|----------------|
| **Risk First** | Maximum drawdown controls, leverage limits, tail hedging |
| **Real Benchmark Accuracy** | Actual SPY returns for beta/momentum calculations |
| **Liquidity Awareness** | 10% ADV constraint prevents market impact |
| **Signal Robustness** | Rolling validation, health scores, correlation adjustment |
| **Execution Realism** | Slippage estimation, dual-broker architecture |

### 1.3 Key Metrics

| Metric | Value |
|--------|-------|
| Alpha Factors | 5 |
| Rebalance Frequency | Hourly |
| Maximum Leverage | 3.0x |
| Maximum Position | 20% |
| Liquidity Cap | 10% ADV |
| Stress Scenarios | 6 |

---

## 2. Investment Philosophy

### 2.1 Philosophy Statement

> **"Understand signal behavior under realistic constraints, with risk management as the primary objective rather than pure alpha maximization."**

This system is designed to explore how quantitative signals perform when subjected to:
- Realistic execution constraints
- Liquidity limitations
- Transaction costs
- Leverage restrictions
- Drawdown controls

### 2.2 Guiding Principles

#### 2.2.1 Diversification Through Orthogonal Factors

The five alpha factors are selected for low correlation:

| Factor Pair | Expected Correlation | Rationale |
|-------------|---------------------|-----------|
| Momentum vs Volatility | Low negative | Momentum struggles in high-vol regimes |
| Carry vs Beta | Low | Different economic drivers |
| Tail Risk vs Momentum | Low | Defensive vs trend-following |
| Volatility vs Beta | Moderate | Both market-sensitive |

#### 2.2.2 Adaptive Behavior

The system adapts to changing conditions:

- **Signal Health Monitoring**: Signals with degraded IC are automatically downweighted
- **Correlation-Aware Weighting**: Reduces allocation to correlated signals
- **Dynamic Leverage**: Scales with signal confidence
- **Tail Hedge Activation**: Triggers during drawdowns

#### 2.2.3 Risk Budgeting

Capital is allocated based on factor risk budgets:

| Factor | Risk Budget | Rationale |
|--------|-------------|-----------|
| Momentum | 30% | Highest capacity, well-documented |
| Carry | 25% | Steady returns, lower turnover |
| Volatility | 20% | Mean reversion potential |
| Tail Risk | 15% | Defensive, crisis alpha |
| Beta | 10% | Market timing, lowest conviction |

#### 2.2.4 Transaction Cost Awareness

Every decision considers implementation costs:

- Slippage estimation before trading
- Buffer zones prevent excessive turnover
- Position decay reduces stale positions gradually
- Minimum trade sizes filter noise

---

## 3. Alpha Generation

### 3.1 Factor Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                      ALPHA FACTOR UNIVERSE                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │
│  │ VOLATILITY  │  │  MOMENTUM   │  │  TAIL RISK  │                 │
│  │ Mean Revert │  │ Excess vs   │  │ CVaR-based  │                 │
│  │ Vol Spikes  │  │ Real SPY    │  │ Protection  │                 │
│  │   20-day    │  │  20/60 day  │  │   60-day    │                 │
│  │ Budget: 20% │  │ Budget: 30% │  │ Budget: 15% │                 │
│  └─────────────┘  └─────────────┘  └─────────────┘                 │
│                                                                      │
│  ┌─────────────┐  ┌─────────────┐                                   │
│  │    BETA     │  │    CARRY    │                                   │
│  │ Rolling vs  │  │ VRP + Mean  │                                   │
│  │ Real SPY    │  │ Reversion + │                                   │
│  │   60-day    │  │ Seasonality │                                   │
│  │ Budget: 10% │  │ Budget: 25% │                                   │
│  └─────────────┘  └─────────────┘                                   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Factor Details

#### 3.2.1 Volatility Factor (20% Budget)

**Hypothesis**: Realized volatility mean-reverts after spikes.

| Component | Description |
|-----------|-------------|
| Signal | Short vol z-score, amplified by vol-of-vol |
| Window | 20 days realized, 252 days for z-score |
| Rationale | Vol clusters but mean-reverts; vol-of-vol indicates regime |

**When it works**: Post-crisis normalization, vol selling premium
**When it fails**: Sustained volatility regimes, trending vol

#### 3.2.2 Momentum Factor (30% Budget)

**Hypothesis**: Excess returns vs benchmark persist over medium term.

| Component | Description |
|-----------|-------------|
| Signal | 70% excess momentum (vs real SPY) + 30% TSMOM |
| Windows | 20-day fast, 60-day slow |
| Benchmark | REAL SPY returns (not approximation) |

**When it works**: Trending markets, sector rotation
**When it fails**: Reversals, choppy markets, crisis

#### 3.2.3 Tail Risk Factor (15% Budget)

**Hypothesis**: Left-tail risk is compensated; asymmetric tails predict returns.

| Component | Description |
|-----------|-------------|
| Signal | -CVaR × tail asymmetry |
| Window | 60 days |
| Quantile | 5th percentile |

**When it works**: Crisis alpha, defensive positioning
**When it fails**: Sustained bull markets

#### 3.2.4 Beta Factor (10% Budget)

**Hypothesis**: High and unstable beta is not compensated.

| Component | Description |
|-----------|-------------|
| Signal | -beta × (1 + beta volatility) |
| Window | 60 days |
| Benchmark | REAL SPY returns |

**When it works**: Risk-off environments, beta compression
**When it fails**: Strong bull markets, beta expansion

#### 3.2.5 Carry Factor (25% Budget)

**Hypothesis**: Volatility risk premium, mean reversion, and seasonality provide steady returns.

| Component | Weight | Description |
|-----------|--------|-------------|
| VRP Proxy | 50% | Implied vol proxy - realized vol |
| Mean Reversion | 30% | Distance from 252-day MA |
| Seasonality | 20% | Annual sine wave |

**When it works**: Range-bound markets, normal conditions
**When it fails**: Trending markets, regime changes

### 3.3 Factor Validation

Each factor undergoes continuous validation:

| Metric | Threshold | Action if Breached |
|--------|-----------|-------------------|
| Rolling IC | < 0.00 | Reduce weight |
| Health Score | < 0.30 | Exclude from composite |
| Hit Rate | < 0.48 | Flag for review |
| Turnover-Adj IC | < -0.01 | Reduce weight |

---

## 4. Signal Processing Pipeline

### 4.1 Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                     SIGNAL PROCESSING PIPELINE                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. DATA ACQUISITION                                                 │
│     ┌─────────────────────────────────────────────────────────┐    │
│     │ Alpaca: 10Y daily + 2Y hourly + Real SPY benchmark      │    │
│     └──────────────────────────┬──────────────────────────────┘    │
│                                │                                     │
│  2. SIGNAL COMPUTATION         ▼                                     │
│     ┌─────────────────────────────────────────────────────────┐    │
│     │ 5 Alpha Factors (Vol, Mom, Tail, Beta, Carry)           │    │
│     │ Each with real market_returns parameter                 │    │
│     └──────────────────────────┬──────────────────────────────┘    │
│                                │                                     │
│  3. SIGNAL VALIDATION          ▼                                     │
│     ┌─────────────────────────────────────────────────────────┐    │
│     │ Rolling IC, Hit Rate, Turnover-Adjusted IC              │    │
│     │ Health Score (5 components)                              │    │
│     │ PnL Attribution                                          │    │
│     └──────────────────────────┬──────────────────────────────┘    │
│                                │                                     │
│  4. WEIGHT OPTIMIZATION        ▼                                     │
│     ┌─────────────────────────────────────────────────────────┐    │
│     │ Base: 50% health + 50% PnL Sharpe                       │    │
│     │ Risk Budgets: Category limits applied                    │    │
│     │ Correlation: SLSQP minimization                          │    │
│     └──────────────────────────┬──────────────────────────────┘    │
│                                │                                     │
│  5. MULTI-TIMEFRAME FUSION     ▼                                     │
│     ┌─────────────────────────────────────────────────────────┐    │
│     │ 70% Daily signals + 30% Hourly signals                  │    │
│     │ Confidence based on agreement                            │    │
│     └──────────────────────────┬──────────────────────────────┘    │
│                                │                                     │
│  6. COMPOSITE SIGNAL           ▼                                     │
│     ┌─────────────────────────────────────────────────────────┐    │
│     │ Rank transform → Center → Position sizing               │    │
│     └─────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 Multi-Timeframe Fusion

| Timeframe | Weight | Data | Purpose |
|-----------|--------|------|---------|
| Daily | 70% | 10 years | Structural factors |
| Hourly | 30% | 2 years | Tactical timing |

**Hourly Signals**:
- Momentum (20 bars)
- Volatility (20 bars)
- RSI (14 bars)
- Price Action (channel position)

**Confidence Calculation**:
- High (70-100%): Daily and hourly agree
- Low (0-30%): Daily and hourly disagree

---

## 5. Risk Framework

### 5.1 Risk Hierarchy

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RISK CONTROL HIERARCHY                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  LEVEL 1: POSITION LIMITS                                           │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ • Max 20% per position                                       │   │
│  │ • Max 10% of 20-day ADV (liquidity)                         │   │
│  │ • Signal-based sizing with confidence adjustment             │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  LEVEL 2: PORTFOLIO LIMITS                                          │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ • Max 3.0x gross leverage                                    │   │
│  │ • Max 1.5x net leverage                                      │   │
│  │ • Auto-scaling if exceeded                                   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  LEVEL 3: DRAWDOWN CONTROLS                                         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ • -5% daily loss: Warning, reduce new positions             │   │
│  │ • -10% daily loss: Halt new positions                       │   │
│  │ • -15% drawdown: Flatten 50%                                │   │
│  │ • -20% drawdown: Full liquidation                           │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  LEVEL 4: TAIL HEDGE                                                │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ • Activates at -5% drawdown                                  │   │
│  │ • Convex payoff: downside^1.5 × 3                           │   │
│  │ • Cost: 2 bps/day when active                               │   │
│  │ • Deactivates at -2.5% drawdown                             │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 Leverage Dynamics

| Signal Confidence | Leverage | Rationale |
|-------------------|----------|-----------|
| 0% (no signal) | 1.0x | Minimum exposure |
| 50% (moderate) | 2.0x | Proportional scaling |
| 100% (high conviction) | 3.0x | Maximum allowed |

**Leverage Formula**:
```
leverage = min_leverage + (max_leverage - min_leverage) × |confidence|
```

### 5.3 Position Sizing Formula

```
target_notional = equity × signed_leverage × max_position_size
raw_shares = target_notional / price
liquidity_cap = ADV_20 × 0.10
portfolio_cap = equity × max_position_size × max_leverage / price
final_shares = clip(raw_shares, -min(caps), +min(caps))
```

### 5.4 Drawdown Control Matrix

| Drawdown Level | Immediate Action | Position Reduction | New Trades |
|----------------|------------------|-------------------|------------|
| 0% to -5% | None | None | Allowed |
| -5% to -10% | Warning alert | None | Reduced size |
| -10% to -15% | Critical alert | Progressive reduction | Halted |
| -15% to -20% | Emergency | 50% flatten | Halted |
| Below -20% | Stop loss | Full liquidation | Halted |

### 5.5 Tail Hedge Mechanics

**State Machine**:

| State | Entry Condition | Exit Condition | Behavior |
|-------|-----------------|----------------|----------|
| INACTIVE | Default | DD > -5% | No cost, no protection |
| ACTIVE | DD < -5% | DD > -2.5% | Convex payoff, 2 bps/day |

**Payoff Profile**:

| Market Return | Hedge PnL (10% size) |
|---------------|---------------------|
| +1% | $0 |
| 0% | $0 |
| -1% | +$300 |
| -2% | +$850 |
| -3% | +$1,560 |
| -5% | +$3,350 |
| -10% | +$9,490 |

*Convex payoff provides increasing protection during severe declines.*

---

## 6. Position Management

### 6.1 Position Lifecycle

```
┌─────────────────────────────────────────────────────────────────────┐
│                      POSITION LIFECYCLE                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. ENTRY                                                            │
│     • Signal strength > threshold                                    │
│     • Liquidity check (10% ADV)                                     │
│     • Slippage estimation                                           │
│     • Order submission                                               │
│                                                                      │
│  2. MONITORING                                                       │
│     • Hourly signal recalculation                                   │
│     • Position aging tracked                                         │
│     • Decay factor applied (halflife = 10 days)                     │
│     • Buffer zone check (15%)                                       │
│                                                                      │
│  3. ADJUSTMENT                                                       │
│     • If signal change > buffer: rebalance                          │
│     • If signal change < buffer: hold                               │
│     • Leverage scaling if needed                                     │
│                                                                      │
│  4. EXIT                                                             │
│     • Signal reversal                                                │
│     • Risk limit breach                                              │
│     • Position decay to near-zero                                   │
│     • Stop loss trigger                                              │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 Position Decay

| Position Age | Decay Factor | Effective Size |
|--------------|--------------|----------------|
| 0 days | 100% | Full |
| 5 days | 71% | Reduced |
| 10 days | 50% | Half |
| 20 days | 25% | Quarter |
| 30 days | 12.5% | Minimal |

**Purpose**: Ensure portfolio freshness, reduce stale positions, adapt to changing signals.

### 6.3 Buffer Zone

Positions must change by >15% to trigger rebalancing.

**Benefits**:
- Reduces unnecessary turnover
- Lowers transaction costs
- Prevents overtrading on noise

---

## 7. Execution Strategy

### 7.1 Execution Principles

| Principle | Implementation |
|-----------|----------------|
| **Execution Certainty** | Market orders (not limit) |
| **Impact Minimization** | 10% ADV constraint |
| **Cost Awareness** | Pre-trade slippage estimation |
| **No PDT** | IBKR execution (no pattern day trader rules) |

### 7.2 Order Types

| Scenario | Order Type | TIF | Rationale |
|----------|------------|-----|-----------|
| Market open | Market | DAY | Immediate execution |
| Market closed | Market | GTC | Queue for open |
| Position transition | Market | DAY | 2-second delay between legs |

### 7.3 Slippage Model

Square-root market impact:

```
participation = |shares| / ADV
base_spread = 10 bps (large cap)
impact = 10 × √participation × (volatility / 0.20)
total_slippage = base_spread/2 + impact
```

| Participation | Volatility | Est. Slippage |
|---------------|------------|---------------|
| 1% of ADV | 20% | 6 bps |
| 5% of ADV | 20% | 12 bps |
| 10% of ADV | 20% | 17 bps |
| 10% of ADV | 40% | 27 bps |

---

## 8. Stress Testing & Scenarios

### 8.1 Historical Crisis Scenarios

| Scenario | Market Crash | Duration | Vol Multiplier | Key Stress |
|----------|-------------|----------|----------------|------------|
| Black Monday 1987 | -20% | 1 day | 5× | Flash crash |
| LTCM 1998 | -15% | 5 days | 3× | Liquidity crisis |
| Dot-com 2000 | -30% | 20 days | 2.5× | Bubble burst |
| GFC 2008 | -40% | 60 days | 4× | Systemic crisis |
| Flash Crash 2010 | -10% | 1 day | 10× | Algo-driven |
| COVID 2020 | -35% | 15 days | 6× | Pandemic shock |

### 8.2 Stress Test Results (Expected)

| Scenario | Max DD | Recovery Time | Tail Hedge Benefit |
|----------|--------|---------------|-------------------|
| Black Monday | -12% | 15 days | +3% protection |
| LTCM | -10% | 20 days | +2% protection |
| Dot-com | -18% | 45 days | +5% protection |
| GFC | -22% | 90 days | +8% protection |
| Flash Crash | -8% | 3 days | +2% protection |
| COVID | -20% | 30 days | +6% protection |

### 8.3 Correlation Breakdown

| Correlation Level | Diversification | Expected Sharpe Impact |
|-------------------|-----------------|----------------------|
| Base (ρ ≈ 0) | Full | Baseline |
| Moderate (ρ = 0.5) | Partial | -15% |
| High (ρ = 0.8) | Limited | -35% |
| Extreme (ρ = 0.95) | Minimal | -50% |

---

## 9. Expected Performance

> **Note**: Performance targets are indicative and primarily used to evaluate risk-adjusted behavior rather than absolute returns.

### 9.1 Target Metrics

| Metric | Target | Notes |
|--------|--------|-------|
| Sharpe Ratio | > 1.0 | Risk-adjusted returns |
| Maximum Drawdown | < 15% | With tail hedge active |
| Win Rate | > 52% | Slight edge sufficient |
| Profit Factor | > 1.2 | Gross profit / loss |
| Average Trade | > 5 bps | Net of costs |
| Recovery Factor | > 2.0 | Return / max DD |
| Annual Turnover | < 20x | Cost control |

### 9.2 Return Attribution (Expected)

| Source | Contribution |
|--------|--------------|
| Momentum | 35% |
| Carry | 25% |
| Volatility | 20% |
| Tail Risk | 12% |
| Beta | 8% |

---

## 10. Competitive Advantages

### 10.1 Key Differentiators

| Advantage | Description |
|-----------|-------------|
| **1. Real Benchmark Data** | Actual SPY returns, not approximations |
| **2. Liquidity Constraints** | 10% ADV cap prevents market impact |
| **3. Dual-Broker Architecture** | Best of Alpaca (data) + IBKR (execution) |
| **4. No PDT Restrictions** | IBKR cash account bypasses day trading rules |
| **5. Correlation-Aware Weighting** | SLSQP optimization reduces redundancy |
| **6. Stateful Tail Hedge** | Convex protection with state machine |
| **7. Integrated Stress Testing** | 6 historical crisis scenarios |
| **8. Pre-Trade Cost Analysis** | Square-root slippage model |
| **9. Adaptive Position Decay** | Ensures portfolio freshness |
| **10. Health Score System** | 5-component signal validation |

### 10.2 vs. Typical Academic Implementations

| Aspect | Academic | This System |
|--------|----------|-------------|
| Benchmark | Approximated | Real SPY |
| Liquidity | Ignored | 10% ADV cap |
| Costs | Ignored | Estimated |
| Execution | Assumed | Dual-broker |
| Validation | In-sample | Rolling out-of-sample |
| Risk | Basic | Multi-layer framework |

---

## 11. Limitations & Risks

### 11.1 Known Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| Capacity | ~$10M max | Expand universe |
| Single asset class | Equities only | Add futures, FX |
| Synthetic hedge | Not real options | Use actual puts |
| Historical signals | May not persist | Rolling validation |
| Technology | System failures | Alerts, monitoring |

### 11.2 Key Risks

| Risk | Description | Mitigation |
|------|-------------|------------|
| Market | Adverse price moves | DD controls, hedging |
| Liquidity | Poor execution | 10% ADV constraint |
| Model | Signal decay | Health scores |
| Execution | Order failures | Monitoring, retries |
| Operational | System bugs | Logging, alerts |
| Counterparty | Broker failure | Regulated brokers |

---

## 12. Use Cases

### 12.1 Primary Use Case

**Risk Management Career Applications**
- Demonstrates understanding of realistic constraints
- Shows focus on risk over alpha
- Suitable for systematic risk roles

### 12.2 Secondary Use Cases

| Use Case | Suitability |
|----------|-------------|
| Quant research | Signal validation methodology |
| Sales & Trading | Risk framework understanding |
| Asset Management | Factor investing principles |
| Compliance | Risk limit implementation |
| Technology | System architecture |

---

## 13. Future Development

### 13.1 Planned Enhancements

| Enhancement | Priority | Description |
|-------------|----------|-------------|
| ML Signals | High | Gradient-boosted regime detection |
| Options Overlay | High | Real put spreads for tail hedge |
| Alternative Data | Medium | Sentiment, news processing |
| Multi-Asset | Medium | Futures, FX, crypto |
| Execution Algos | Medium | TWAP, VWAP |
| Web Dashboard | Low | Real-time monitoring UI |

### 13.2 Research Agenda

- Factor timing models
- Regime-switching allocation
- Cross-asset correlation
- Intraday signal decay
- Options-based signals

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
