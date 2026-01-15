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
import logging
from typing import Dict
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi

class HybridWeightedPipeline:
    def __init__(self, pipeline, api, logger: logging.Logger):
        self.pipeline = pipeline
        self.api = api
        self.logger = logger
        self.daily_weight = 0.70
        self.hourly_weight = 0.30
        self.hourly_config = {
            'momentum_window': 20,
            'vol_window': 20,
            'rsi_window': 14,
            'price_action_window': 20
        }
    
    def get_daily_data(self, symbol: str) -> pd.DataFrame:
        """Get 10 years of daily data from ALPACA"""
        try:
            start_date = (datetime.now() - timedelta(days=365*10)).strftime('%Y-%m-%d')
            
            self.logger.info(f'📊 [ALPACA] Fetching DAILY bars from {start_date}...')
            
            import alpaca_trade_api as tradeapi
            
            bars = self.api.get_bars(
                symbol,
                tradeapi.TimeFrame.Day,  # ← Alpaca format
                start=start_date
            )
            
            self.logger.info(f'   ✅ Retrieved {len(bars)} daily bars (~{len(bars)/252:.1f} years)')
            
            return bars
            
        except Exception as e:
            self.logger.error(f'Failed to get daily data: {e}')
            import traceback
            self.logger.error(traceback.format_exc())
            return pd.DataFrame()

    def get_hourly_data(self, symbol: str) -> pd.DataFrame:
        """Get 2 years of hourly data from ALPACA"""
        try:
            # Calculate start date (2 years ago)
            start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
            
            self.logger.info(f'📊 [ALPACA] Fetching HOURLY bars from {start_date}...')
            
            import alpaca_trade_api as tradeapi
            
            # Use START instead of LIMIT
            bars = self.api.get_bars(
                symbol,
                tradeapi.TimeFrame.Hour,
                start=start_date  # ← CHANGEMENT ICI
            )
            
            self.logger.info(f'   ✅ Retrieved {len(bars)} hourly bars (~{len(bars)/(252*6.5):.1f} years)')
            
            return bars
            
        except Exception as e:
            self.logger.error(f'Failed to get hourly data: {e}')
            import traceback
            self.logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    def calculate_daily_signals_weighted(self, bars: pd.DataFrame, market_returns=None) -> Dict:
        if len(bars) < 252:
            self.logger.warning(f'Not enough daily bars: {len(bars)}')
            return {}
        
        prices = bars['close']
        returns = prices.pct_change()
        volume = bars.get('volume', pd.Series([0]*len(bars)))
        
        self.logger.info('Running FULL PIPELINE on daily data...')
        
        # ✅ NOUVEAU: Use real market returns if provided
        if market_returns is not None and len(market_returns) > 0:
            # Align market returns with stock returns
            try:
                # Take last N returns to match length
                if len(market_returns) >= len(returns):
                    market_returns_aligned = market_returns.iloc[-len(returns):].reset_index(drop=True)
                    market_returns_aligned.index = returns.index
                else:
                    # If SPY has fewer bars, pad with approximation
                    market_returns_aligned = returns * 0.6
                
                self.logger.info(f'   ✅ Using REAL SPY benchmark ({len(market_returns_aligned)} bars)')
            except Exception as e:
                self.logger.warning(f'   Failed to align SPY returns: {e}, using approximation')
                market_returns_aligned = returns * 0.6
        else:
            self.logger.warning('   No SPY data provided, using approximation')
            market_returns_aligned = returns * 0.6
        
        results = self.pipeline.run_full_pipeline_vectorized(prices, returns, volume)
        
        if results is None:
            self.logger.error('Pipeline returned None')
            return {}
        
        # ✅ Use the aligned market returns
        data = {'returns': returns, 'close': prices, 'market_returns': market_returns_aligned}
        
        signal_values = {}
        health_scores = {}
        
        for name, signal_obj in self.pipeline.signals.items():
            sig = signal_obj.compute(data)
            signal_values[name.lower()] = sig.iloc[-1] if len(sig) > 0 else 0
            health_scores[name.lower()] = signal_obj.health_score
        
        if not self.pipeline.weights:
            self.logger.warning('Pipeline weights empty, using equal weights')
            weights = {name: 1.0/len(self.pipeline.signals) for name in self.pipeline.signals.keys()}
        else:
            weights = self.pipeline.weights
        
        composite = 0.0
        for name, weight in weights.items():
            signal_val = signal_values.get(name.lower(), 0)
            composite += weight * signal_val
        
        self.logger.info(f'Daily composite (weighted): {composite:.4f}')
        
        return {
            'composite': composite, 
            'weights': weights, 
            'signal_values': signal_values, 
            'health_scores': health_scores
        }
    
    def calculate_hourly_signals(self, bars: pd.DataFrame) -> Dict:
        if len(bars) < 100:
            self.logger.warning(f'Not enough hourly bars: {len(bars)}')
            return {}
        prices = bars['close']
        returns = prices.pct_change()
        signals = {}
        momentum = returns.rolling(self.hourly_config['momentum_window']).sum()
        signals['momentum'] = momentum.iloc[-1]
        hourly_vol = returns.rolling(self.hourly_config['vol_window']).std()
        vol_mean = hourly_vol.rolling(100).mean()
        vol_std = hourly_vol.rolling(100).std()
        vol_zscore = (hourly_vol - vol_mean) / (vol_std + 1e-8)
        signals['volatility'] = -vol_zscore.iloc[-1]
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(self.hourly_config['rsi_window']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.hourly_config['rsi_window']).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        rsi_signal = (50 - rsi.iloc[-1]) / 50
        signals['rsi'] = rsi_signal
        high_20 = prices.rolling(self.hourly_config['price_action_window']).max()
        low_20 = prices.rolling(self.hourly_config['price_action_window']).min()
        price_position = (prices - low_20) / (high_20 - low_20 + 1e-8)
        signals['price_action'] = 0.5 - price_position.iloc[-1]
        composite_hourly = np.mean([signals['momentum'], signals['volatility'], signals['rsi'], signals['price_action']])
        signals['composite'] = composite_hourly
        self.logger.info(f'Hourly composite: {composite_hourly:.4f}')
        return signals
    
    def calculate_hybrid_weighted_signal(self, symbol: str, market_return=None) -> Dict:
        self.logger.info(f'HYBRID WEIGHTED CALCULATION: {symbol}')
        daily_bars = self.get_daily_data(symbol)
        hourly_bars = self.get_hourly_data(symbol)
        if len(daily_bars) < 252 or len(hourly_bars) < 100:
            self.logger.error('Insufficient data')
            return {}
        daily_result = self.calculate_daily_signals_weighted(daily_bars)
        if not daily_result:
            self.logger.error('Daily calculation failed')
            return {}
        hourly_result = self.calculate_hourly_signals(hourly_bars)
        if not hourly_result:
            self.logger.error('Hourly calculation failed')
            return {}
        daily_composite = daily_result['composite']
        hourly_composite = hourly_result['composite']
        hybrid_composite = self.daily_weight * daily_composite + self.hourly_weight * hourly_composite
        agreement = np.sign(daily_composite) == np.sign(hourly_composite)
        magnitude_daily = abs(daily_composite)
        magnitude_hourly = abs(hourly_composite)
        if agreement:
            confidence = 0.7 + 0.3 * min(magnitude_daily, magnitude_hourly)
        else:
            confidence = 0.3 * (1 - abs(magnitude_daily - magnitude_hourly))
        if daily_composite > 0.1:
            trend = 'BULLISH'
        elif daily_composite < -0.1:
            trend = 'BEARISH'
        else:
            trend = 'NEUTRAL'
        if hourly_composite > 0.05:
            timing = 'BUY'
        elif hourly_composite < -0.05:
            timing = 'SELL'
        else:
            timing = 'HOLD'
        self.logger.info(f'HYBRID SUMMARY: Trend={trend}, Timing={timing}, Signal={hybrid_composite:.4f}, Confidence={confidence:.2%}')
        return {
            'composite': hybrid_composite,
            'daily_composite': daily_composite,
            'hourly_composite': hourly_composite,
            'daily_weights': daily_result['weights'],
            'daily_signals': daily_result['signal_values'],
            'health_scores': daily_result['health_scores'],
            'hourly_signals': {k: v for k, v in hourly_result.items() if k != 'composite'},
            'confidence': confidence,
            'agreement': agreement,
            'trend': trend,
            'timing': timing,
            'daily_bars': len(daily_bars),
            'hourly_bars': len(hourly_bars)
        }
