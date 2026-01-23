# ============================================================
# Copyright (c) 2026 Anonym_
# All rights reserved.
#
# This code is provided for educational and personal use only.
# Unauthorized commercial use, redistribution, or modification
# without explicit permission is prohibited.
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime, timedelta

# ============================================
# RANDOM SEED FOR REPRODUCIBILITY
# ============================================
np.random.seed(42)

# ============================================
# CONFIG
# ============================================
SYMBOLS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',  # Tech
    'JPM', 'BAC', 'GS', 'MS',                  # Finance
    'XOM', 'CVX',                               # Energy
    'LMT', 'RTX', 'NOC', 'GD', 'BA',           # Defense
    'JNJ', 'UNH', 'PFE', 'MRK', 'ABBV'         # Healthcare
]
YEARS = 10
INITIAL_CAPITAL = 1_000_000

os.makedirs('figures', exist_ok=True)

# ============================================
# IMPORTS
# ============================================
print("Loading pipeline...")
try:
    from enhanced_pipeline import (
        EnhancedProductionPipeline,
        VolatilitySignal,
        MomentumSignal,
        TailRiskSignal,
        BetaSignal,
        CarrySignal,
        setup_logger
    )
    import yfinance as yf
    import logging
    print("✅ Imports OK")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("   Run: pip install yfinance")
    sys.exit(1)

# ============================================
# FETCH DATA (Yahoo Finance - Fixed dates)
# ============================================
print(f"\nFetching data from 2016-01-21 to 2026-01-21...")

start_date = '2016-01-21'
end_date = '2026-01-21'

# Fetch all symbols
all_data = {}
for symbol in SYMBOLS + ['SPY']:
    print(f"   {symbol}...", end=" ")
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, auto_adjust=True)
        
        if len(df) > 252:
            # Rename columns to match our format
            df.columns = [c.lower() for c in df.columns]
            all_data[symbol] = df
            print(f"✅ {len(df)} bars")
        else:
            print(f"⚠️ Only {len(df)} bars, skipping")
    except Exception as e:
        print(f"❌ {e}")

# Find common dates
available_symbols = [s for s in SYMBOLS if s in all_data]
print(f"\n✅ {len(available_symbols)}/{len(SYMBOLS)} symbols loaded")

common_dates = None
for symbol in available_symbols + ['SPY']:
    if symbol in all_data:
        dates = set(all_data[symbol].index)
        common_dates = dates if common_dates is None else common_dates.intersection(dates)

common_dates = sorted(list(common_dates))
print(f"✅ {len(common_dates)} common trading days")

# SPY for benchmark
spy_returns = all_data['SPY']['close'].loc[common_dates].pct_change().fillna(0)

# ============================================
# RUN PIPELINE ON EACH SYMBOL
# ============================================
print(f"\nRunning pipeline on {len(available_symbols)} symbols...")

logger = setup_logger('backtest')
logger.setLevel(logging.ERROR)

# Store results per symbol
symbol_results = {}

for symbol in available_symbols:
    print(f"   {symbol}...", end=" ")
    try:
        pipeline = EnhancedProductionPipeline()
        pipeline.logger.setLevel(logging.ERROR)
        
        pipeline.add_signal(VolatilitySignal)
        pipeline.add_signal(MomentumSignal)
        pipeline.add_signal(TailRiskSignal)
        pipeline.add_signal(BetaSignal)
        pipeline.add_signal(CarrySignal)
        
        df = all_data[symbol]
        prices = df['close'].loc[common_dates]
        returns = prices.pct_change().fillna(0)
        volumes = df['volume'].loc[common_dates]
        
        # Pass real SPY returns as market_returns
        results = pipeline.run_full_pipeline_vectorized(prices, returns, volumes, spy_returns)
        
        if results is not None:
            symbol_results[symbol] = {
            'position': results['final_position'].values,   # or scaled_positions if you expose it
            'returns': returns.values
        }
            print(f"✅")
        else:
            print(f"❌ failed")
    except Exception as e:
        print(f"❌ {e}")

print(f"\n✅ {len(symbol_results)}/{len(available_symbols)} symbols processed")

# ============================================
# COMBINE INTO PORTFOLIO (Signal-Weighted like paper_trading.py)
# ============================================
print("\nCombining into position-weighted portfolio...")

dates = pd.DatetimeIndex(common_dates)

pos_df = pd.DataFrame({s: symbol_results[s]['position'] for s in symbol_results}, index=dates).fillna(0.0)
ret_df = pd.DataFrame({s: symbol_results[s]['returns']  for s in symbol_results}, index=dates).fillna(0.0)

# Shift positions so today’s return uses yesterday’s position
pos_df = pos_df.shift(1).fillna(0.0)

# PREVENT FLAT-LINES: if book goes fully flat (gross==0), carry last non-zero book
gross = pos_df.abs().sum(axis=1)
flat_days = gross == 0

# forward-fill last non-zero positions, but keep zeros before first non-zero
pos_ffill = pos_df.replace(0.0, np.nan).ffill().fillna(0.0)

# optional: re-enter with small exposure during flat periods (so equity isn't perfectly flat)
REENTRY_FLOOR = 0.10  # 10% of whatever the last book was
pos_df = pos_df.where(~flat_days, pos_ffill * REENTRY_FLOOR)


# FORCE POSITIONS TO BE SIGNAL WEIGHTS (REMOVE HIDDEN LEVERAGE)
pos_df = pos_df.clip(-1.0, 1.0)

# CROSS-SECTIONAL NORMALIZATION (KEY FOR SHARPE)
gross = pos_df.abs().sum(axis=1)
pos_df = pos_df.div(gross.replace(0, np.nan), axis=0).fillna(0.0)


MAX_GROSS = 1.5
gross = pos_df.abs().sum(axis=1)
scale = (MAX_GROSS / gross).clip(upper=1.5)   # cap only
pos_df = pos_df.mul(scale, axis=0)

pos_df = pos_df.clip(-0.15, 0.15)
gross = pos_df.abs().sum(axis=1)
scale = (MAX_GROSS / gross).clip(upper=1.5)
pos_df = pos_df.mul(scale, axis=0)

gross = pos_df.abs().sum(axis=1)


# Portfolio returns (return-space)
portfolio_returns = (pos_df * ret_df).sum(axis=1)

# PORTFOLIO DRAWDOWN BRAKE (smooth de-leveraging)
equity_tmp = (1 + portfolio_returns.fillna(0.0)).cumprod()
dd = (equity_tmp - equity_tmp.cummax()) / equity_tmp.cummax()

brake = (1 + dd / 0.25).clip(0.3, 1.0)   # starts reducing after -25%
portfolio_returns = portfolio_returns * brake


equity = (1 + portfolio_returns).cumprod() * INITIAL_CAPITAL
drawdown = (equity - equity.cummax()) / equity.cummax()

results = pd.DataFrame({
    'strategy_returns': portfolio_returns,
    'equity': equity,
    'drawdown': drawdown
}, index=dates)

print(f"✅ Portfolio built: {pos_df.shape[1]} symbols (position-weighted, max gross {MAX_GROSS}x)")


# ============================================
# EXTRACT METRICS
# ============================================
equity = results['equity']
drawdown = results['drawdown']
strategy_returns = results['strategy_returns']

# Strategy metrics
total_return = (equity.iloc[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL
years = len(dates) / 252
annual_return = (1 + total_return) ** (1/years) - 1
annual_vol = strategy_returns.std() * np.sqrt(252)
sharpe = annual_return / annual_vol if annual_vol > 0 else 0
max_dd = drawdown.min()

# SPY metrics
spy_equity = INITIAL_CAPITAL * (1 + spy_returns).cumprod()
spy_total_return = (spy_equity.iloc[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL
spy_annual_return = (1 + spy_total_return) ** (1/years) - 1
spy_vol = spy_returns.std() * np.sqrt(252)
spy_sharpe = spy_annual_return / spy_vol if spy_vol > 0 else 0
spy_dd = ((spy_equity - spy_equity.cummax()) / spy_equity.cummax()).min()


years = len(dates) / 252
cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1/years) - 1

print(f"\n{'='*50}")
print("RESULTS")
print(f"{'='*50}")
print(f"Portfolio ({pos_df.shape[1]} symbols):")
print(f"   Return:  {total_return:+.1%}")
print("CAGR:", f"{cagr:.2%}")
print(f"   Sharpe:  {sharpe:.2f}")
print(f"   Max DD:  {max_dd:.1%}")
print(f"\nSPY Buy & Hold:")
print(f"   Return:  {spy_total_return:+.1%}")
print(f"   Sharpe:  {spy_sharpe:.2f}")
print(f"   Max DD:  {spy_dd:.1%}")
print("Avg gross:", pos_df.abs().sum(axis=1).mean())
print("Max gross:", pos_df.abs().sum(axis=1).max())

# sanity: daily portfolio returns shouldn't be absurd
print("99.9% daily abs return:", portfolio_returns.abs().quantile(0.999))
print("Pct days gross==0:", (gross == 0).mean())

yearly = results['strategy_returns'].groupby(results.index.year).sum()
print("\nYearly returns:")
print((1 + yearly).cumprod() - 1)

# ============================================
# GENERATE CHARTS
# ============================================
print("\nGenerating charts...")

plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {'strategy': '#2E86AB', 'spy': '#666666'}

# 1. Equity Curve
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(dates, equity / 1e6, label=f'Strategy (Sharpe: {sharpe:.2f})', 
        color=COLORS['strategy'], linewidth=2.5)
ax.plot(dates, spy_equity / 1e6, label=f'SPY Buy & Hold (Sharpe: {spy_sharpe:.2f})',
        color=COLORS['spy'], linewidth=2, linestyle='--')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Portfolio Value ($M)', fontsize=12)
ax.set_title(f'Multi-Factor Strategy vs SPY Benchmark\n{pos_df.shape[1]
} Symbols Portfolio | 5 Alpha Signals', fontsize=14)
ax.legend(loc='upper left', fontsize=11)
ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/equity_curve.png', dpi=150, bbox_inches='tight')
print("   ✅ figures/equity_curve.png")
plt.close()

# 2. Drawdown
fig, ax = plt.subplots(figsize=(14, 5))
ax.fill_between(dates, drawdown * 100, 0, alpha=0.7, color=COLORS['strategy'], label='Strategy')
spy_drawdown = (spy_equity - spy_equity.cummax()) / spy_equity.cummax()
ax.plot(dates, spy_drawdown * 100, color=COLORS['spy'], linewidth=1.5, linestyle='--', label='SPY')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Drawdown (%)', fontsize=12)
ax.set_title('Underwater Curve', fontsize=14)
ax.legend(loc='lower left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/drawdown.png', dpi=150, bbox_inches='tight')
print("   ✅ figures/drawdown.png")
plt.close()

# 3. Rolling Sharpe
fig, ax = plt.subplots(figsize=(14, 5))
rolling_ret = strategy_returns.rolling(252).mean() * 252
rolling_vol = strategy_returns.rolling(252).std() * np.sqrt(252)
rolling_sharpe = rolling_ret / rolling_vol
ax.plot(dates, rolling_sharpe, color=COLORS['strategy'], linewidth=2)
ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
ax.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Sharpe = 1')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Rolling Sharpe (1Y)', fontsize=12)
ax.set_title('Rolling 1-Year Sharpe Ratio', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/rolling_sharpe.png', dpi=150, bbox_inches='tight')
print("   ✅ figures/rolling_sharpe.png")
plt.close()

# 4. Performance Summary Bars
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# Return
ax = axes[0]
bars = ax.bar(['Strategy', 'SPY'], [total_return*100, spy_total_return*100],
              color=[COLORS['strategy'], COLORS['spy']], edgecolor='black')
for bar in bars:
    h = bar.get_height()
    ax.annotate(f'{h:.1f}%', xy=(bar.get_x() + bar.get_width()/2, h),
                xytext=(0, 3), textcoords='offset points', ha='center', fontweight='bold')
ax.set_ylabel('Total Return (%)')
ax.set_title('Total Return')
ax.axhline(y=0, color='black', linewidth=0.5)

# Sharpe
ax = axes[1]
bars = ax.bar(['Strategy', 'SPY'], [sharpe, spy_sharpe],
              color=[COLORS['strategy'], COLORS['spy']], edgecolor='black')
for bar in bars:
    h = bar.get_height()
    ax.annotate(f'{h:.2f}', xy=(bar.get_x() + bar.get_width()/2, h),
                xytext=(0, 3), textcoords='offset points', ha='center', fontweight='bold')
ax.set_ylabel('Sharpe Ratio')
ax.set_title('Sharpe Ratio')
ax.axhline(y=1, color='green', linestyle='--', alpha=0.5)

# Max DD
ax = axes[2]
bars = ax.bar(['Strategy', 'SPY'], [max_dd*100, spy_dd*100],
              color=[COLORS['strategy'], COLORS['spy']], edgecolor='black')
for bar in bars:
    h = bar.get_height()
    ax.annotate(f'{h:.1f}%', xy=(bar.get_x() + bar.get_width()/2, h),
                xytext=(0, -15), textcoords='offset points', ha='center', fontweight='bold')
ax.set_ylabel('Max Drawdown (%)')
ax.set_title('Max Drawdown')

plt.tight_layout()
plt.savefig('figures/performance_summary.png', dpi=150, bbox_inches='tight')
print("   ✅ figures/performance_summary.png")
plt.close()

# 5. Monthly Returns Heatmap
try:
    monthly = strategy_returns.resample('M').apply(lambda x: (1+x).prod()-1)
    monthly_df = pd.DataFrame({
        'Year': monthly.index.year,
        'Month': monthly.index.month,
        'Return': monthly.values
    })
    pivot = monthly_df.pivot(index='Year', columns='Month', values='Return')
    
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(pivot.values * 100, cmap='RdYlGn', aspect='auto', vmin=-10, vmax=10)
    ax.set_xticks(range(12))
    ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    plt.colorbar(im, ax=ax, label='Monthly Return (%)')
    ax.set_title('Monthly Returns Heatmap', fontsize=14)
    plt.tight_layout()
    plt.savefig('figures/monthly_returns.png', dpi=150, bbox_inches='tight')
    print("   ✅ figures/monthly_returns.png")
    plt.close()
except Exception as e:
    print(f"   ⚠️ Monthly heatmap skipped: {e}")

# 6. Gross Exposure Through Time
fig, ax = plt.subplots(figsize=(14, 5))
gross_exposure = pos_df.abs().sum(axis=1)
ax.fill_between(dates, gross_exposure, 0, alpha=0.7, color=COLORS['strategy'], label='Gross Exposure')
ax.axhline(y=MAX_GROSS, color='red', linestyle='--', alpha=0.7, linewidth=2, label=f'Max Leverage ({MAX_GROSS}x)')
ax.axhline(y=1, color='green', linestyle='--', alpha=0.5, linewidth=1.5, label='1x (Unlevered)')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Gross Exposure (x)', fontsize=12)
ax.set_title('Gross Exposure Through Time', fontsize=14)
ax.legend(loc='upper right')
ax.set_ylim(0, MAX_GROSS + 0.5)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/gross_exposure.png', dpi=150, bbox_inches='tight')
print("   ✅ figures/gross_exposure.png")
plt.close()

# 7. Net Exposure Through Time
fig, ax = plt.subplots(figsize=(14, 5))
net_exposure = pos_df.sum(axis=1)
ax.fill_between(dates, net_exposure, 0, where=(net_exposure >= 0), alpha=0.7, color=COLORS['strategy'], label='Net Long')
ax.fill_between(dates, net_exposure, 0, where=(net_exposure < 0), alpha=0.7, color='#E74C3C', label='Net Short')
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, linewidth=3, label='+50% Net')
ax.axhline(y=-0.5, color='red', linestyle='--', alpha=0.5, linewidth=3, label='-50% Net')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Net Exposure (x)', fontsize=12)
ax.set_title('Net Exposure Through Time (Long/Short Balance)', fontsize=14)
ax.legend(loc='upper right')
ax.set_ylim(-3, 3)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/net_exposure.png', dpi=150, bbox_inches='tight')
print("   ✅ figures/net_exposure.png")
plt.close()

# ============================================
# UPDATE README TABLE
# ============================================
start_str = dates[0].strftime('%Y-%m-%d')
end_str = dates[-1].strftime('%Y-%m-%d')

readme_update = f"""
## Performance

![Equity Curve](figures/equity_curve.png)

| Metric | Strategy | SPY Buy & Hold |
|--------|----------|----------------|
| Total Return | {total_return:+.1%} | {spy_total_return:+.1%} |
| Sharpe Ratio | {sharpe:.2f} | {spy_sharpe:.2f} |
| Max Drawdown | {max_dd:.1%} | {spy_dd:.1%} |

*Backtest: {start_str} to {end_str} | {pos_df.shape[1]} symbols (equal weight)*

![Performance Summary](figures/performance_summary.png)
"""

with open('figures/readme_performance.md', 'w') as f:
    f.write(readme_update)
print("   ✅ figures/readme_performance.md")

print(f"\n{'='*50}")
print("DONE! Charts saved in figures/")
print(f"{'='*50}")
