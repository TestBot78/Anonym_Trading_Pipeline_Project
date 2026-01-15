# ============================================================
# Copyright (c) 2026 Anonym_
# All rights reserved.
#
# This code is provided for educational and personal use only.
# Unauthorized commercial use, redistribution, or modification
# without explicit permission is prohibited.
# ============================================================


import alpaca_trade_api as alpaca_api
from ibkr_adapter import IBKRAdapter
import logging
from datetime import datetime

class DualBrokerAdapter:
    """
    Dual Broker: Alpaca for data, IBKR for execution
    Drop-in replacement for Alpaca API with IBKR execution backend
    """
    
    def __init__(self, alpaca_key, alpaca_secret, alpaca_base_url='https://paper-api.alpaca.markets',
                 ibkr_host='127.0.0.1', ibkr_port=7497, ibkr_client_id=1):
        
        self.logger = logging.getLogger('DualBroker')
        
        # ========================================
        # ALPACA for DATA
        # ========================================
        self.logger.info("Connecting to Alpaca for market data...")
        self.alpaca = alpaca_api.REST(
            alpaca_key,
            alpaca_secret,
            base_url=alpaca_base_url
        )
        self.logger.info("✅ Alpaca connected (data source)")
        
        # ========================================
        # IBKR for EXECUTION
        # ========================================
        self.logger.info("Connecting to IBKR for order execution...")
        self.ibkr = IBKRAdapter(host=ibkr_host, port=ibkr_port, client_id=ibkr_client_id)
        self.logger.info("✅ IBKR connected (execution)")
        
        self.logger.info("")
        self.logger.info("="*70)
        self.logger.info("🎯 DUAL BROKER MODE ACTIVE")
        self.logger.info("   Data Source:  Alpaca (10y daily, 2y hourly)")
        self.logger.info("   Execution:    IBKR (NO PDT RULE!)")
        self.logger.info("="*70)
    
    # ========================================
    # MARKET DATA - Routed to ALPACA
    # ========================================
    
    def get_bars(self, symbol, timeframe, start=None, end=None, limit=None):
        """
        Get historical bars from ALPACA
        Returns: pandas DataFrame
        """
        self.logger.debug(f"[ALPACA] Getting bars: {symbol} {timeframe}")
        
        try:
            bars = self.alpaca.get_bars(
                symbol,
                timeframe,
                start=start,
                end=end,
                limit=limit
            ).df
            
            self.logger.debug(f"[ALPACA] Retrieved {len(bars)} bars")
            return bars
            
        except Exception as e:
            self.logger.error(f"[ALPACA] Failed to get bars: {e}")
            import pandas as pd
            return pd.DataFrame()
    
    def get_latest_bar(self, symbol):
        """Get latest bar from ALPACA"""
        self.logger.debug(f"[ALPACA] Getting latest bar: {symbol}")
        return self.alpaca.get_latest_bar(symbol)
    
    def get_clock(self):
        """Get market clock from ALPACA"""
        return self.alpaca.get_clock()
    
    # ========================================
    # ACCOUNT & POSITIONS - Routed to IBKR
    # ========================================
    
    def get_account(self):
        """Get account info from IBKR"""
        self.logger.debug(f"[IBKR] Getting account info")
        account = self.ibkr.get_account()
        
        # Log important info
        self.logger.debug(
            f"[IBKR] Account: equity=${float(account.equity):,.0f}, "
            f"buying_power=${float(account.buying_power):,.0f}"
        )
        
        return account
    
    def list_positions(self):
        """Get current positions from IBKR"""
        self.logger.debug(f"[IBKR] Getting positions")
        positions = self.ibkr.list_positions()
        
        if positions:
            self.logger.debug(f"[IBKR] Found {len(positions)} positions")
        
        return positions

    def get_position(self, symbol):
        """
        Get specific position by symbol from IBKR
        Returns position object or None if not found
        """
        self.logger.debug(f"[IBKR] Getting position for {symbol}")
        
        try:
            positions = self.ibkr.list_positions()
            
            for pos in positions:
                if pos.symbol == symbol:
                    self.logger.debug(f"[IBKR] Found position: {pos.symbol} {pos.qty} shares")
                    return pos
            
            # Position not found
            self.logger.debug(f"[IBKR] No position found for {symbol}")
            return None
            
        except Exception as e:
            self.logger.error(f"[IBKR] Failed to get position {symbol}: {e}")
            return None
    
    
    
    # ========================================
    # ORDER EXECUTION - Routed to IBKR
    # ========================================
    
    def submit_order(self, symbol, qty, side, type='market', time_in_force='day', limit_price=None):
        """
        Submit order to IBKR (NO PDT RESTRICTIONS!)
        """
        self.logger.info(f"[IBKR] Submitting order: {side.upper()} {qty} {symbol}")
        
        try:
            order = self.ibkr.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type=type,
                time_in_force=time_in_force,
                limit_price=limit_price
            )
            
            self.logger.info(f"[IBKR] ✅ Order submitted: {order.id}")
            return order
            
        except Exception as e:
            self.logger.error(f"[IBKR] ❌ Order failed: {e}")
            raise
    
    def list_orders(self, status='all', limit=50):
        """List orders from IBKR"""
        self.logger.debug(f"[IBKR] Listing orders (status={status})")
        return self.ibkr.list_orders(status=status, limit=limit)
    
    def close_position(self, symbol):
        """Close position on IBKR"""
        self.logger.info(f"[IBKR] Closing position: {symbol}")
        return self.ibkr.close_position(symbol)
    
    def close_all_positions(self):
        """Close all positions on IBKR"""
        self.logger.info(f"[IBKR] Closing ALL positions")
        positions = self.list_positions()
        
        for pos in positions:
            try:
                self.close_position(pos.symbol)
            except Exception as e:
                self.logger.error(f"[IBKR] Failed to close {pos.symbol}: {e}")
    
    # ========================================
    # UTILITY
    # ========================================
    
    def disconnect(self):
        """Disconnect both brokers"""
        self.logger.info("Disconnecting dual broker...")
        try:
            self.ibkr.disconnect()
            self.logger.info("✅ Disconnected")
        except:
            pass
    
    def __del__(self):
        """Cleanup on deletion"""
        try:
            self.disconnect()
        except:
            pass