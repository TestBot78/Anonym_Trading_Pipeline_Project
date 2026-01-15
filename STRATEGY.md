# Strategy Summary

## Executive Overview

This document presents a quantitative trading system designed for systematic equity portfolio management. The system combines multiple alpha factors with dynamic risk management and liquidity-aware position sizing to generate risk-adjusted returns.

## Investment Philosophy

The strategy is built on four core principles:

1. **Multi-factor diversification**: No single signal dominates; returns come from uncorrelated sources
2. **Real benchmark accuracy**: Uses actual SPY returns for precise signal calculation
3. **Liquidity awareness**: Position sizes respect market depth to minimize impact
4. **Adaptive risk management**: Leverage scales with signal confidence and market conditions

## Alpha Generation

### Signal Sources

| Factor | Weight | Rationale |
|--------|--------|-----------|
| Momentum | 30% | Strongest historical predictive power |
| Carry | 25% | Stable premium, low correlation |
| Volatility | 20% | Mean reversion reliability |
| Tail Risk | 15% | Crash protection value |
| Beta | 10% | Market timing component |

### Signal Processing

The system processes market data through a multi-stage pipeline:

```
Raw Data → Real SPY Benchmark → Feature Engineering → Signal Generation → Weight Optimization → Liquidity Check → Position Sizing
```

**Multi-timeframe fusion**: Daily signals (70% weight) capture structural trends while hourly signals (30% weight) optimize entry timing.

**Real benchmark integration**: Unlike approximation-based systems, this pipeline fetches actual SPY returns to ensure:
- Accurate excess momentum calculation
- Proper beta estimation
- Realistic signal correlation

## Risk Framework

### Position Sizing

Positions are constrained by TWO factors:

| Constraint | Value | Purpose |
|------------|-------|---------|
| Portfolio limit | 20% × leverage | Diversification |
| Liquidity limit | 10% of 20-day ADV | Market impact prevention |

Final position = MIN(portfolio_limit, liquidity_limit)

### Leverage Dynamics

| Signal Confidence | Leverage Used |
|-------------------|---------------|
| Low (< 30%) | 1.0x (minimum) |
| Medium (30-60%) | 1.5x - 2.0x |
| High (> 60%) | 2.0x - 3.0x (maximum) |

### Drawdown Controls

| Threshold | Action |
|-----------|--------|
| -5% daily | Warning, reduce new positions |
| -10% daily | Halt new positions |
| -15% total | Flatten 50% of portfolio |
| -20% total | Full liquidation |

### Tail Hedging

A dynamic tail hedge activates during market stress:

- **Trigger**: Portfolio drawdown exceeds -5%
- **Payoff**: Convex protection (downside^1.5)
- **Cost**: ~2 bps per day when active
- **Deactivation**: Recovery above -2.5%

## Expected Performance

Based on backtesting across 6 years of market data (2018-2023):

> **Note**: Performance targets are indicative and primarily used to evaluate risk-adjusted behavior rather than absolute returns.

| Metric | Target | Notes |
|--------|--------|-------|
| Annual Return | 8-15% | Varies by regime |
| Sharpe Ratio | > 1.0 | Risk-adjusted |
| Max Drawdown | < -15% | With tail hedge |
| Win Rate | > 52% | Daily signals |

### Stress Test Results

The system was tested against 6 historical crisis scenarios:

| Crisis | Duration | Market Drop | Strategy Response |
|--------|----------|-------------|-------------------|
| Black Monday (1987) | 1 day | -20% | Hedge activated, contained |
| LTCM Crisis (1998) | 5 days | -15% | Moderate drawdown |
| Dot-com Crash (2000) | 20 days | -30% | Survived, slow recovery |
| Financial Crisis (2008) | 60 days | -40% | Max DD approached |
| Flash Crash (2010) | 1 day | -10% | Minimal impact |
| COVID Crash (2020) | 15 days | -35% | Hedge activated |

## Implementation Details

### Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| Data Source | Alpaca API (10y daily, 2y hourly) |
| Benchmark | Real SPY returns |
| Execution | Interactive Brokers TWS |
| Database | SQLite |
| Optimization | SciPy (SLSQP) |

### Trading Universe

Current implementation focuses on liquid US equities:

**ETFs**: SPY, QQQ, IWM, XLK

**Large-caps**: AAPL, MSFT, GOOGL, AMZN, NVDA

### Execution Parameters

| Parameter | Value |
|-----------|-------|
| Rebalance frequency | Every 60 minutes |
| Signal calculation | Hourly |
| Order type | Market orders |
| Time in force | Day / GTC |
| Reconciliation | Every 5 minutes |

## Competitive Advantages

1. **Real benchmark data**: Accurate SPY returns for proper signal calculation (not approximations)

2. **Liquidity constraints**: 10% ADV cap ensures executable trades without market impact

3. **Dual-broker architecture**: Separates data quality (Alpaca) from execution efficiency (IBKR)

4. **No PDT restrictions**: IBKR execution avoids pattern day trader limits

5. **Correlation-aware weighting**: Optimizes signal combination dynamically

6. **Integrated stress testing**: Built-in scenario analysis for 6 historical crises

## Limitations & Risks

### Known Limitations

- **Capacity constraints**: Strategy may not scale beyond $10M AUM due to liquidity caps
- **Data dependency**: Requires reliable API connections for real-time signals
- **Model risk**: Historical patterns may not repeat in future regimes
- **Technology risk**: System failures during volatile markets

### Key Risks

| Risk Type | Mitigation |
|-----------|------------|
| Market Risk | Drawdown limits, tail hedging |
| Liquidity Risk | 10% ADV constraint |
| Execution Risk | Market orders, dual brokers |
| Model Risk | Rolling validation, multiple factors |
| Operational Risk | Automated monitoring, alerts |

## Use Cases

### For Individual Investors

- Systematic exposure to equity markets
- Disciplined risk management
- Emotion-free trading decisions

### For Academic Purposes

- Demonstrates quantitative finance concepts
- Factor investing implementation
- Portfolio optimization techniques
- Risk management frameworks

### For Career Development

- Shows proficiency in Python and finance
- Demonstrates understanding of:
  - Alpha factor construction
  - Signal processing
  - Risk budgeting
  - Execution systems
  - Database design

## Future Development

### Planned Enhancements

1. **Machine learning signals**: Gradient-boosted features for regime detection
2. **Options overlay**: Systematic premium collection strategies
3. **Alternative data**: Sentiment analysis, web scraping
4. **Multi-asset expansion**: Futures, FX, cryptocurrency

### Research Agenda

- Factor timing models based on macro indicators
- Regime detection algorithms (HMM, clustering)
- Transaction cost optimization (TWAP, VWAP)
- Portfolio construction improvements (hierarchical risk parity)

## Conclusion

This quantitative trading system represents a systematic approach to equity portfolio management. By combining multiple uncorrelated alpha sources with rigorous risk management and liquidity awareness, the strategy aims to deliver consistent risk-adjusted returns across market conditions.

**Key differentiators**:
- Real SPY benchmark (not approximation)
- Liquidity-constrained position sizing
- Multi-timeframe signal fusion
- Comprehensive stress testing
- Production-ready codebase (~4,200 lines)

The system demonstrates proficiency in quantitative finance, software engineering, and risk management — skills directly applicable to roles in asset management, hedge funds, and fintech.

---

*Disclaimer: This is an educational project. Past performance does not guarantee future results. Trading involves risk of loss.*

---

**Author**: Anonym_  
**Institution**: ESSCA School of Management  
**Program**: MSc Finance & Data Analytics  
**Date**: January 2026

### Intellectual Property Notice

This project was developed independently for educational and
career development purposes. All intellectual property remains
with the author.
