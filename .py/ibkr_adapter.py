# ============================================================
# Copyright (c) 2026 Anonym_
# All rights reserved.
#
# This code is provided for educational and personal use only.
# Unauthorized commercial use, redistribution, or modification
# without explicit permission is prohibited.
# ============================================================


from ib_insync import IB, Stock, MarketOrder, LimitOrder
import pandas as pd
from datetime import datetime, timedelta
import time
import logging

class IBKRAdapter:
    """
    Adapter pour IBKR qui imite l'API Alpaca
    Compatible avec paper_trading.py sans modifications
    """
    
    def __init__(self, host='127.0.0.1', port=7497, client_id=1):
        self.ib = IB()
        self.host = host
        self.port = port
        self.client_id = client_id
        self.logger = logging.getLogger('IBKR')
        
        # Connect
        self.connect()
    
    def connect(self):
        """Connect to TWS with increased timeout"""
        try:
            # Increase timeout to 30 seconds
            self.ib.connect(
                self.host, 
                self.port, 
                clientId=self.client_id,
                timeout=30,
                readonly=False  # Important for trading
            )
            self.logger.info(f"Connected to IBKR TWS on {self.host}:{self.port}")
        except Exception as e:
            self.logger.error(f"Failed to connect to IBKR: {e}")
            raise
    
    def disconnect(self):
        """Disconnect from TWS"""
        self.ib.disconnect()
    
    def get_account(self):
        """Get account info (Alpaca-compatible format)"""
        summary = self.ib.accountSummary()
        account_dict = {item.tag: item.value for item in summary}
        
        # Create Alpaca-like account object
        class Account:
            def __init__(self, data):
                self.portfolio_value = float(data.get('NetLiquidation', 0))
                self.cash = float(data.get('TotalCashValue', 0))
                self.equity = float(data.get('NetLiquidation', 0))
                self.buying_power = float(data.get('BuyingPower', 0))
                self.daytrading_buying_power = self.buying_power  # No PDT in cash account
                self.multiplier = '1'  # Cash account
                self.pattern_day_trader = False
                self.daytrade_count = 0
        
        return Account(account_dict)
    
    def get_bars(self, symbol, timeframe, start=None, end=None, limit=None):
        """Get historical bars - IBKR compliant"""
        contract = Stock(symbol, 'SMART', 'USD')
        self.ib.qualifyContracts(contract)
        
        # Convert timeframe to string if needed
        timeframe_str = str(timeframe)
        
        # ========================================
        # DAILY BARS
        # ========================================
        if 'Day' in timeframe_str or 'day' in timeframe_str.lower():
            barsize = '1 day'
            
            if limit:
                # Calculate duration from limit
                if limit > 365:
                    # MUST use years for > 365 days
                    years = int(limit / 252) + 1  # 252 trading days/year
                    duration = f'{years} Y'
                else:
                    duration = f'{limit} D'
            
            elif start:
                # Calculate days from start date
                start_dt = datetime.fromisoformat(start.replace('Z', ''))
                days = (datetime.now() - start_dt).days
                
                if days > 365:
                    # MUST use years for > 365 days
                    years = int(days / 365) + 1
                    duration = f'{years} Y'
                else:
                    duration = f'{days} D'
            
            else:
                # Default: 10 years
                duration = '10 Y'  # ← Use YEARS not days!
        
        # ========================================
        # HOURLY BARS
        # ========================================
        elif 'Hour' in timeframe_str or 'hour' in timeframe_str.lower():
            barsize = '1 hour'
            
            # IBKR limits hourly to 86400 seconds (24 hours) per request
            # So we need to make MULTIPLE requests and concatenate
            
            if limit and limit > 24:
                # Need multiple requests
                all_bars = []
                
                # Request in chunks of 1 day (24 hours) at a time
                end_dt = datetime.now()
                remaining = limit
                
                while remaining > 0:
                    chunk_size = min(24, remaining)  # Max 24 hours per request
                    
                    bars_chunk = self.ib.reqHistoricalData(
                        contract,
                        endDateTime=end_dt.strftime('%Y%m%d %H:%M:%S'),
                        durationStr=f'{chunk_size * 3600} S',  # seconds
                        barSizeSetting=barsize,
                        whatToShow='TRADES',
                        useRTH=True,
                        formatDate=1
                    )
                    
                    if bars_chunk:
                        all_bars.extend(bars_chunk)
                        # Move end_dt back
                        if len(bars_chunk) > 0:
                            end_dt = bars_chunk[0].date
                        else:
                            break
                    else:
                        break
                    
                    remaining -= chunk_size
                    
                    # Rate limit: wait between requests
                    import time
                    time.sleep(1)
                
                bars = all_bars
            
            elif start:
                # Use days instead of seconds for longer periods
                start_dt = datetime.fromisoformat(start.replace('Z', ''))
                days = (datetime.now() - start_dt).days
                duration = f'{min(days, 30)} D'  # Max 30 days at once for hourly
                
                bars = self.ib.reqHistoricalData(
                    contract,
                    endDateTime='',
                    durationStr=duration,
                    barSizeSetting=barsize,
                    whatToShow='TRADES',
                    useRTH=True,
                    formatDate=1
                )
            
            else:
                # Default: 1 day of hourly data
                duration = '1 D'
                
                bars = self.ib.reqHistoricalData(
                    contract,
                    endDateTime='',
                    durationStr=duration,
                    barSizeSetting=barsize,
                    whatToShow='TRADES',
                    useRTH=True,
                    formatDate=1
                )
        
        else:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        
        # Convert to DataFrame
        if not bars:
            return pd.DataFrame()
        
        df = pd.DataFrame([{
            'open': b.open,
            'high': b.high,
            'low': b.low,
            'close': b.close,
            'volume': b.volume
        } for b in bars], index=[b.date for b in bars])
        
        return df
    
    def get_latest_bar(self, symbol):
        """Get latest bar for a symbol"""
        contract = Stock(symbol, 'SMART', 'USD')
        self.ib.qualifyContracts(contract)
        
        # Get latest snapshot
        ticker = self.ib.reqMktData(contract, '', False, False)
        self.ib.sleep(2)  # Wait for data
        
        class Bar:
            def __init__(self, price):
                self.c = price  # close
                self.o = price  # open
                self.h = price  # high
                self.l = price  # low
        
        price = ticker.last if ticker.last == ticker.last else ticker.close
        return Bar(price)
    
    def list_positions(self):
        """Get current positions (Alpaca format)"""
        ibkr_positions = self.ib.positions()
        
        class Position:
            def __init__(self, ibkr_pos):
                # Basic info
                self.symbol = ibkr_pos.contract.symbol
                self.qty = str(int(ibkr_pos.position))
                self.side = 'long' if ibkr_pos.position > 0 else 'short'
                
                # Pricing
                self.avg_entry_price = str(abs(ibkr_pos.avgCost))
                
                # Current price (try to get market price, fallback to avg cost)
                try:
                    if hasattr(ibkr_pos, 'marketPrice') and ibkr_pos.marketPrice > 0:
                        self.current_price = str(ibkr_pos.marketPrice)
                    else:
                        self.current_price = str(abs(ibkr_pos.avgCost))
                except:
                    self.current_price = str(abs(ibkr_pos.avgCost))
                
                # Market value
                self.market_value = str(ibkr_pos.position * ibkr_pos.avgCost)
                self.cost_basis = str(abs(ibkr_pos.position * ibkr_pos.avgCost))
                
                # Unrealized P&L
                try:
                    self.unrealized_pl = str(ibkr_pos.unrealizedPnL)
                except AttributeError:
                    self.unrealized_pl = "0"
                
                # Unrealized P&L percentage
                try:
                    cost = abs(ibkr_pos.position * ibkr_pos.avgCost)
                    if cost > 0 and hasattr(ibkr_pos, 'unrealizedPnL'):
                        self.unrealized_plpc = str(ibkr_pos.unrealizedPnL / cost)
                    else:
                        self.unrealized_plpc = "0"
                except (AttributeError, ZeroDivisionError):
                    self.unrealized_plpc = "0"
                
                # Asset classification
                self.asset_class = 'us_equity'
                
                # Exchange
                try:
                    self.exchange = ibkr_pos.contract.primaryExchange
                except AttributeError:
                    self.exchange = 'NASDAQ'
        
        return [Position(p) for p in ibkr_positions]
    
    def get_position(self, symbol):
        """Get specific position by symbol"""
        positions = self.list_positions()
        
        for pos in positions:
            if pos.symbol == symbol:
                return pos
        
        # Position not found, return None
        return None
    
    def submit_order(self, symbol, qty, side, type='market', time_in_force='day', limit_price=None):
        """Submit an order (Alpaca-compatible)"""
        contract = Stock(symbol, 'SMART', 'USD')
        self.ib.qualifyContracts(contract)
        
        # Create order
        action = 'BUY' if side.lower() == 'buy' else 'SELL'
        
        # ✅ FIX: Use GTC if market is closed
        # Check if market is open
        import datetime
        now = datetime.datetime.now()
        hour = now.hour
        
        # Simple check: if outside 9:30-16:00 EST (14:30-21:00 UTC for Paris)
        # Use GTC to allow orders outside market hours
        if hour < 14 or hour >= 21:
            # Market closed, use GTC
            tif = 'GTC'
            self.logger.info(f"Market closed, using GTC order")
        else:
            tif = 'DAY' if time_in_force == 'day' else 'GTC'
        
        if type == 'market':
            order = MarketOrder(action, abs(int(qty)))
            order.tif = tif  # ← Set Time In Force
        elif type == 'limit':
            order = LimitOrder(action, abs(int(qty)), limit_price)
            order.tif = tif
        else:
            raise ValueError(f"Unsupported order type: {type}")
        
        # Submit
        trade = self.ib.placeOrder(contract, order)
        
        # Wait a bit for order to be processed
        self.ib.sleep(1)
        
        # Create Alpaca-like order object
        class Order:
            def __init__(self, ibkr_trade):
                self.id = str(ibkr_trade.order.orderId)
                self.symbol = symbol
                self.side = side
                self.qty = str(qty)
                self.filled_qty = str(ibkr_trade.orderStatus.filled)
                self.filled_avg_price = str(ibkr_trade.orderStatus.avgFillPrice)
                self.status = ibkr_trade.orderStatus.status.lower()
                self.submitted_at = datetime.datetime.now()
                self.filled_at = datetime.datetime.now() if self.status == 'filled' else None
        
        return Order(trade)
    
    def list_orders(self, status='all', limit=50):
        """List orders"""
        trades = self.ib.trades()
        
        class Order:
            def __init__(self, ibkr_trade):
                self.id = str(ibkr_trade.order.orderId)
                self.symbol = ibkr_trade.contract.symbol if ibkr_trade.contract else 'UNKNOWN'
                self.side = ibkr_trade.order.action.lower()
                self.qty = str(ibkr_trade.order.totalQuantity)
                self.filled_qty = str(ibkr_trade.orderStatus.filled)
                self.filled_avg_price = str(ibkr_trade.orderStatus.avgFillPrice)
                self.status = ibkr_trade.orderStatus.status.lower()
                self.submitted_at = datetime.now()
                self.filled_at = datetime.now() if self.status == 'filled' else None
        
        orders = [Order(t) for t in trades[-limit:]]
        
        if status != 'all':
            orders = [o for o in orders if o.status == status]
        
        return orders
    
    def close_position(self, symbol):
        """Close a position"""
        for pos in self.ib.positions():
            if pos.contract.symbol == symbol:
                qty = abs(int(pos.position))
                side = 'sell' if pos.position > 0 else 'buy'
                return self.submit_order(symbol, qty, side)
        
        return None
    
    def get_clock(self):
        """Get market clock with proper US market hours"""
        from datetime import datetime, timezone
        import pytz
        
        # US Eastern Time
        eastern = pytz.timezone('US/Eastern')
        now_et = datetime.now(eastern)
        
        # Check if weekday (0=Monday, 6=Sunday)
        is_weekday = now_et.weekday() < 5
        
        # Market hours: 9:30 AM - 4:00 PM ET
        hour_minute = now_et.hour + now_et.minute / 60
        in_market_hours = 9.5 <= hour_minute < 16.0
        
        is_open = is_weekday and in_market_hours
        
        class Clock:
            def __init__(self, is_open_val, now):
                self.is_open = is_open_val
                self.timestamp = now
                
                # Calculate next open
                if is_open_val:
                    # Market is open, next close is today at 4 PM
                    self.next_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
                    # Next open is tomorrow at 9:30 AM
                    tomorrow = now + timedelta(days=1)
                    self.next_open = tomorrow.replace(hour=9, minute=30, second=0, microsecond=0)
                else:
                    # Market is closed
                    if hour_minute >= 16.0:
                        # After close, next open is tomorrow 9:30 AM
                        tomorrow = now + timedelta(days=1)
                        # Skip weekend
                        while tomorrow.weekday() >= 5:
                            tomorrow += timedelta(days=1)
                        self.next_open = tomorrow.replace(hour=9, minute=30, second=0, microsecond=0)
                    else:
                        # Before open, next open is today 9:30 AM
                        if is_weekday:
                            self.next_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
                        else:
                            # Weekend, next Monday
                            days_ahead = 7 - now.weekday()
                            next_monday = now + timedelta(days=days_ahead)
                            self.next_open = next_monday.replace(hour=9, minute=30, second=0, microsecond=0)
                    
                    self.next_close = self.next_open.replace(hour=16, minute=0)
        
        return Clock(is_open, now_et)
        
        return Clock()