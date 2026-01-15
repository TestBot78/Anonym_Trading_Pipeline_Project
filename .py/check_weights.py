# ============================================================
# Copyright (c) 2026 Anonym_
# All rights reserved.
#
# This code is provided for educational and personal use only.
# Unauthorized commercial use, redistribution, or modification
# without explicit permission is prohibited.
# ============================================================

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi

with open('trading_config.json', 'r') as f:
    config = json.load(f)

from enhanced_pipeline import (
    EnhancedProductionPipeline,
    VolatilitySignal,
    MomentumSignal, 
    TailRiskSignal,
    BetaSignal,
    CarrySignal
)

print("="*70)
print("🔬 DIAGNOSTIC: POURQUOI COMPOSITE = 0?")
print("="*70)

# Initialize
pipeline = EnhancedProductionPipeline()
pipeline.add_signal(VolatilitySignal)
pipeline.add_signal(MomentumSignal)
pipeline.add_signal(TailRiskSignal)
pipeline.add_signal(BetaSignal)
pipeline.add_signal(CarrySignal)

# Get data
api = tradeapi.REST(
    config['alpaca_api_key'],
    config['alpaca_api_secret'],
    base_url='https://paper-api.alpaca.markets'
)

start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
bars = api.get_bars('SPY', tradeapi.TimeFrame.Day, start=start_date).df

prices = bars['close']
returns = prices.pct_change()
volume = bars['volume']

data = {
    'returns': returns,
    'close': prices,
    'market_returns': returns * 0.6
}

print(f"\n📊 Calcul des signaux...")

# Compute signals and validate
signals_df = pd.DataFrame(index=prices.index)

for name, signal_obj in pipeline.signals.items():
    signals_df[name] = signal_obj.compute(data)
    signal_obj.validate_rolling_vectorized(signals_df[name], returns, 252)

print(f"✅ Signaux calculés")

print("\n" + "="*70)
print("🏥 HEALTH SCORES")
print("="*70)

costs = 0.0005
active_signals = {}

for name, signal_obj in pipeline.signals.items():
    health_score = signal_obj.compute_health_score(63)
    pnl_attr = signal_obj.compute_pnl_attribution_vectorized(signals_df[name], returns, costs)
    
    print(f"\n{name}:")
    print(f"   Health Score: {health_score:.3f}")
    print(f"   Sharpe:       {pnl_attr['sharpe']:.2f}")
    print(f"   Hit Rate:     {pnl_attr['hit_rate']:.2%}")
    
    if health_score > 0.3:
        active_signals[name] = signal_obj
        print(f"   Status:       ✅ ACTIVE")
    else:
        print(f"   Status:       ❌ INACTIVE (health < 0.3)")

print("\n" + "="*70)
print("⚖️  WEIGHTS CALCULATION")
print("="*70)

if len(active_signals) == 0:
    print("\n❌ NO ACTIVE SIGNALS!")
    print("   → Composite = 0 because all signals have health < 0.3")
    print("\n💡 SOLUTION:")
    print("   1. Le marché est peut-être en phase difficile")
    print("   2. Les signaux ont besoin de plus de données")
    print("   3. Attends quelques jours pour que les health scores s'améliorent")
else:
    print(f"\n✅ {len(active_signals)} active signals")
    
    # Calculate weights
    weights = {}
    for name, sig in active_signals.items():
        pnl_sharpe = sig.pnl_attribution.get('sharpe', 0)
        
        # Score calculation (same as pipeline)
        from scipy.special import expit
        pnl_score = expit(pnl_sharpe / 1.0)
        composite_score = 0.5 * sig.health_score + 0.5 * pnl_score
        weights[name] = composite_score
        
        print(f"\n{name}:")
        print(f"   Health:    {sig.health_score:.3f}")
        print(f"   PnL Score: {pnl_score:.3f}")
        print(f"   Composite: {composite_score:.3f}")
        print(f"   Weight:    {composite_score:.3f} (before normalization)")
    
    # Normalize
    total = sum(weights.values())
    normalized_weights = {k: v/total for k, v in weights.items()}
    
    print("\n" + "="*70)
    print("📊 FINAL WEIGHTS (normalized)")
    print("="*70)
    
    for name, weight in normalized_weights.items():
        print(f"   {name:.<20} {weight:.3f}")
    
    # Calculate composite
    print("\n" + "="*70)
    print("🎯 COMPOSITE SIGNAL")
    print("="*70)
    
    signal_values = {}
    for name in active_signals.keys():
        sig = signals_df[name]
        signal_values[name.lower()] = sig.iloc[-1]
    
    composite = sum(
        normalized_weights.get(name, 0) * signal_values.get(name.lower(), 0)
        for name in active_signals.keys()
    )
    
    print(f"\nCalculation:")
    for name in active_signals.keys():
        weight = normalized_weights.get(name, 0)
        value = signal_values.get(name.lower(), 0)
        contribution = weight * value
        print(f"   {weight:.3f} × {value:>10.6f} = {contribution:>10.6f}  ({name})")
    
    print(f"\n   Composite = {composite:.6f}")
    print(f"   Signal Strength = {np.tanh(composite * 2):.6f}")
    
    if abs(composite) < 0.01:
        print("\n⚠️  Composite est proche de 0!")
        print("   → Les signaux se compensent (certains positifs, d'autres négatifs)")
        print("   → C'est normal: le marché est neutre/indécis")

print("\n" + "="*70)
print("✅ DIAGNOSTIC TERMINÉ")
print("="*70)
