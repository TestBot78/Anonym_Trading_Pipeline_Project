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
import alpaca_trade_api as tradeapi
from dual_broker_adapter import DualBrokerAdapter
from datetime import datetime, timedelta
from hybrid_pipeline_weighted import HybridWeightedPipeline
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import threading
from dataclasses import dataclass, asdict
import sqlite3
import sys
from dual_broker_adapter import DualBrokerAdapter

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class TradingConfig:
    """Configuration complète pour paper trading"""

    # Broker selection
    broker: str = "ALPACA"  # ← AJOUTE CETTE LIGNE
    
    # Alpaca API
    alpaca_api_key: str = "YOUR_ALPACA_KEY"
    alpaca_api_secret: str = "YOUR_ALPACA_SECRET"
    alpaca_base_url: str = "https://paper-api.alpaca.markets"
    
    # Trading parameters
    symbols: List[str] = None
    rebalance_frequency_minutes: int = 30
    min_trade_size: int = 10
    max_position_size: float = 0.30
    
    # Risk limits
    max_daily_loss: float = -0.05
    max_drawdown: float = -0.15
    min_leverage: float = 1.0
    max_leverage: float = 1.5
    
    # Monitoring
    check_interval_seconds: int = 60
    reconciliation_interval_minutes: int = 5
    
    # Alerts
    email_alerts: bool = False
    email_from: str = "pipeline@trading.com"
    email_to: List[str] = None
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    
    slack_webhook: str = ""
    
    # Database
    db_path: str = "paper_trading.db"
    
    # Logging
    log_file: str = "paper_trading.log"
    log_level: str = "INFO"
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ['SPY']
        if self.email_to is None:
            self.email_to = []
    
    def save(self, path: str):
        """Save config to JSON"""
        with open(path, 'w') as f:
            config_dict = asdict(self)
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: str):
        """Load config from JSON"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
            return cls(**config_dict)


def setup_logger(name: str = 'pipeline', log_file: str = None):
    """Setup logging avec UTF-8 pour Windows"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    # Console handler avec UTF-8
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Force UTF-8 sur Windows
    if sys.platform == 'win32':
        try:
            console_handler.stream = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
        except:
            pass
    
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# ============================================================================
# DATABASE MANAGER
# ============================================================================

class TradingDatabase:
    """SQLite database for tracking everything"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.create_tables()
    
    def create_tables(self):
        """Create all necessary tables"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_data (
                timestamp TEXT,
                symbol TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                PRIMARY KEY (timestamp, symbol)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                timestamp TEXT PRIMARY KEY,
                composite_signal REAL,
                volatility_signal REAL,
                momentum_signal REAL,
                tail_risk_signal REAL,
                beta_signal REAL,
                carry_signal REAL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                timestamp TEXT,
                symbol TEXT,
                quantity INTEGER,
                avg_entry_price REAL,
                current_price REAL,
                market_value REAL,
                unrealized_pnl REAL,
                PRIMARY KEY (timestamp, symbol)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS orders (
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
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS portfolio (
                timestamp TEXT PRIMARY KEY,
                cash REAL,
                portfolio_value REAL,
                equity REAL,
                buying_power REAL,
                daily_pnl REAL,
                total_pnl REAL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                level TEXT,
                message TEXT,
                acknowledged INTEGER DEFAULT 0
            )
        """)
        
        self.conn.commit()
    
    def insert_market_data(self, timestamp: str, symbol: str, data: Dict):
        """Insert market data"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO market_data 
            (timestamp, symbol, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (timestamp, symbol, data['open'], data['high'], 
              data['low'], data['close'], data['volume']))
        self.conn.commit()
    
    def insert_signal(self, timestamp: str, signals: Dict):
        """Insert signal values"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO signals
            (timestamp, composite_signal, volatility_signal, momentum_signal,
             tail_risk_signal, beta_signal, carry_signal)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (timestamp, signals.get('composite', 0), 
              signals.get('volatility', 0), signals.get('momentum', 0),
              signals.get('tailrisk', 0), signals.get('beta', 0),
              signals.get('carry', 0)))
        self.conn.commit()
    
    def insert_position(self, timestamp: str, symbol: str, position: Dict):
        """Insert position snapshot"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO positions
            (timestamp, symbol, quantity, avg_entry_price, current_price,
             market_value, unrealized_pnl)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (timestamp, symbol, position['qty'], position['avg_entry_price'],
              position['current_price'], position['market_value'],
              position['unrealized_pnl']))
        self.conn.commit()
    
    def insert_order(self, order: Dict):
        """Insert order"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO orders
            (order_id, timestamp, symbol, side, quantity, order_type,
             status, filled_qty, filled_avg_price, commission)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (order['id'], order['timestamp'], order['symbol'], 
              order['side'], order['qty'], order['type'], order['status'],
              order.get('filled_qty', 0), order.get('filled_avg_price', 0),
              order.get('commission', 0)))
        self.conn.commit()
    
    def insert_portfolio(self, timestamp: str, portfolio: Dict):
        """Insert portfolio snapshot"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO portfolio
            (timestamp, cash, portfolio_value, equity, buying_power,
             daily_pnl, total_pnl)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (timestamp, portfolio['cash'], portfolio['portfolio_value'],
              portfolio['equity'], portfolio['buying_power'],
              portfolio.get('daily_pnl', 0), portfolio.get('total_pnl', 0)))
        self.conn.commit()
    
    def insert_alert(self, timestamp: str, level: str, message: str):
        """Insert alert"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO alerts (timestamp, level, message)
            VALUES (?, ?, ?)
        """, (timestamp, level, message))
        self.conn.commit()
    
    def get_historical_data(self, symbol: str, limit: int = 500) -> pd.DataFrame:
        """Get historical data from DB"""
        query = f"""
            SELECT * FROM market_data 
            WHERE symbol = '{symbol}'
            ORDER BY timestamp DESC
            LIMIT {limit}
        """
        df = pd.read_sql_query(query, self.conn)
        if len(df) == 0:
            return pd.DataFrame()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        return df.set_index('timestamp')
    
    def close(self):
        """Close database connection"""
        self.conn.close()


# ============================================================================
# ALERT MANAGER
# ============================================================================

class AlertManager:
    """Manages all alerts"""
    
    def __init__(self, config: TradingConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def send_email(self, subject: str, body: str, level: str = "INFO"):
        """Send email alert"""
        if not self.config.email_alerts or not self.config.email_to:
            return
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config.email_from
            msg['To'] = ', '.join(self.config.email_to)
            msg['Subject'] = f"[{level}] {subject}"
            
            html_body = f"""
            <html>
                <body>
                    <h2>{subject}</h2>
                    <p>{body.replace(chr(10), '<br>')}</p>
                    <hr>
                    <small>Sent at {datetime.now()}</small>
                </body>
            </html>
            """
            
            msg.attach(MIMEText(html_body, 'html'))
            
            server = smtplib.SMTP(self.config.smtp_server, self.config.smtp_port)
            server.starttls()
            server.login(self.config.smtp_user, self.config.smtp_password)
            server.send_message(msg)
            server.quit()
            
            self.logger.info(f"Email sent: {subject}")
            
        except Exception as e:
            self.logger.error(f"Failed to send email: {e}")
    
    def send_slack(self, message: str, level: str = "INFO"):
        """Send Slack alert"""
        if not self.config.slack_webhook:
            return
        
        try:
            payload = {'text': f"*{level}*\n{message}"}
            
            response = requests.post(
                self.config.slack_webhook,
                json=payload,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                self.logger.info("Slack alert sent")
            else:
                self.logger.error(f"Slack alert failed: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Failed to send Slack alert: {e}")
    
    def alert(self, subject: str, message: str, level: str = "INFO"):
        """Send alert via all configured channels"""
        full_message = f"{subject}\n{message}"
        
        log_func = {
            'INFO': self.logger.info,
            'WARNING': self.logger.warning,
            'ERROR': self.logger.error,
            'CRITICAL': self.logger.critical
        }.get(level, self.logger.info)
        
        log_func(full_message)
        
        if level in ['CRITICAL', 'ERROR']:
            self.send_email(subject, message, level)
            self.send_slack(full_message, level)
        elif level == 'WARNING':
            self.send_slack(full_message, level)


# ============================================================================
# RISK MONITOR
# ============================================================================

class RiskMonitor:
    """Real-time risk monitoring"""
    
    def __init__(self, config: TradingConfig, alert_manager: AlertManager):
        self.config = config
        self.alert_manager = alert_manager
        self.daily_peak = None
        self.overall_peak = None
    
    def check_risk_limits(self, account: Dict, positions: Dict) -> bool:
        """Check all risk limits"""
        portfolio_value = float(account['portfolio_value'])
        
        if self.daily_peak is None:
            self.daily_peak = portfolio_value
        if self.overall_peak is None:
            self.overall_peak = portfolio_value
        
        self.overall_peak = max(self.overall_peak, portfolio_value)
        
        daily_pnl = float(account.get('daily_pnl', 0))
        daily_pnl_pct = daily_pnl / self.daily_peak if self.daily_peak > 0 else 0
        
        if daily_pnl_pct < self.config.max_daily_loss:
            self.alert_manager.alert(
                "DAILY LOSS LIMIT BREACHED",
                f"Daily P&L: {daily_pnl_pct:.2%}",
                level="CRITICAL"
            )
            return False
        
        drawdown = (portfolio_value - self.overall_peak) / self.overall_peak
        
        if drawdown < self.config.max_drawdown:
            self.alert_manager.alert(
                "MAX DRAWDOWN LIMIT BREACHED",
                f"Drawdown: {drawdown:.2%}",
                level="CRITICAL"
            )
            return False
        
        equity = float(account['equity'])
        total_exposure = sum(abs(float(p.get('market_value', 0))) for p in positions.values())
        leverage = total_exposure / equity if equity > 0 else 0

        leverage_emergency_threshold = self.config.max_leverage * 2
        
        if leverage > self.config.max_leverage:
            self.alert_manager.alert(
                "LEVERAGE LIMIT EXCEEDED",
                f"Leverage: {leverage:.2f}x",
                level="WARNING"
            )
            return False
        
        if leverage > self.config.max_leverage:
            self.logger.warning(
                f"Leverage elevated ({leverage:.2f}x > {self.config.max_leverage:.2f}x) "
                f"but below emergency threshold. Deleveraging should handle this."
            )
        
        return True
    
    def reset_daily_peak(self):
        """Reset daily peak"""
        self.daily_peak = self.overall_peak


# ============================================================================
# PAPER TRADING ENGINE
# ============================================================================

class PaperTradingEngine:
    """Main paper trading engine"""
    
    def __init__(self, pipeline, config: TradingConfig):
        self.pipeline = pipeline
        self.config = config
        
        # Setup logging
        self.logger = setup_logger('PaperTrading', config.log_file)
        
        # Initialize components
        self.db = TradingDatabase(config.db_path)
        self.alert_manager = AlertManager(config, self.logger)
        self.risk_monitor = RiskMonitor(config, self.alert_manager)
        
        # Dual Broker: Alpaca data + IBKR execution
        self.api = DualBrokerAdapter(
            alpaca_key=self.config.alpaca_api_key,
            alpaca_secret=self.config.alpaca_api_secret,
            alpaca_base_url=self.config.alpaca_base_url,
            ibkr_host='127.0.0.1',
            ibkr_port=7497,
            ibkr_client_id=1
        )

        # State
        self.is_running = False
        self.last_rebalance = None
        self.last_reconciliation = None
        self.emergency_stop = False
        
        self.logger.info("="*70)
        self.logger.info("PAPER TRADING ENGINE INITIALIZED")
        self.logger.info("="*70)
    
    def get_market_data(self, symbol: str, limit: int = 1000) -> pd.DataFrame:
        """Get market data from Alpaca avec plus de données"""
        try:
            # Essaye d'abord avec limit
            bars = self.api.get_bars(
                symbol,
                tradeapi.TimeFrame.Day,
                limit=limit
            ).df
            
            # Si pas assez, essaye avec date de début
            if len(bars) < 252:
                self.logger.warning(f"Only {len(bars)} bars available, fetching from 10 years ago...")
                
                start_date = (datetime.now() - timedelta(days=365*10)).strftime('%Y-%m-%d')
                
                bars = self.api.get_bars(
                    symbol,
                    tradeapi.TimeFrame.Day,
                    start=start_date
                ).df
                
                self.logger.info(f"Retrieved {len(bars)} bars total")
            
            # Store in database
            for timestamp, row in bars.iterrows():
                self.db.insert_market_data(
                    timestamp.isoformat(),
                    symbol,
                    {
                        'open': row['open'],
                        'high': row['high'],
                        'low': row['low'],
                        'close': row['close'],
                        'volume': row['volume']
                    }
                )
            
            return bars
            
        except Exception as e:
            self.logger.error(f"Failed to get market data for {symbol}: {e}")
            return self.db.get_historical_data(symbol, limit)
    
    def calculate_signals(self) -> Dict:
        """Calculate HYBRID WEIGHTED signals (daily poids optimisés + hourly timing)"""
        try:
            # Initialize hybrid weighted pipeline
            if not hasattr(self, 'hybrid_weighted'):
                self.hybrid_weighted = HybridWeightedPipeline(
                    self.pipeline,  # Pass the full pipeline for daily
                    self.api,
                    self.logger
                )
            
            signals = {}

            try:
                spy_bars = self.api.get_bars('SPY', tradeapi.TimeFrame.Day, limit=500)
                spy_returns = spy_bars['close'].pct_change()
                self.logger.info(f'✅ Using real SPY benchmark ({len(spy_returns)} bars)')
            except Exception as e:
                self.logger.warning(f'Failed to get SPY benchmark: {e}, using approximation')
                spy_returns = None
            
            for symbol in self.config.symbols:
                # Calculate hybrid weighted signal
                hybrid_result = self.hybrid_weighted.calculate_hybrid_weighted_signal(symbol, 
                market_return=spy_returns  # ✅ Pass real benchmark
                )
            
                
                if not hybrid_result:
                    self.logger.error(f"Failed to calculate hybrid signal for {symbol}")
                    continue
                
                # Store signal values
                signal_values = {
                    'composite': hybrid_result['composite'],
                    'daily_trend': hybrid_result['daily_composite'],
                    'hourly_timing': hybrid_result['hourly_composite'],
                    'confidence': hybrid_result['confidence'],
                    
                    # Store individual daily signals for database
                    'volatility': hybrid_result['daily_signals'].get('volatility', 0),
                    'momentum': hybrid_result['daily_signals'].get('momentum', 0),
                    'tailrisk': hybrid_result['daily_signals'].get('tailrisk', 0),
                    'beta': hybrid_result['daily_signals'].get('beta', 0),
                    'carry': hybrid_result['daily_signals'].get('carry', 0),
                }
                
                signals[symbol] = signal_values
                
                # Store in database
                self.db.insert_signal(
                    datetime.now().isoformat(),
                    {
                        'composite_signal': signal_values['composite'],
                        'volatility_signal': signal_values.get('volatility', 0),
                        'momentum_signal': signal_values.get('momentum', 0),
                        'tail_risk_signal': signal_values.get('tailrisk', 0),
                        'beta_signal': signal_values.get('beta', 0),
                        'carry_signal': signal_values.get('carry', 0)
                    }
                )
                
                self.logger.info(
                    f"\n✅ {symbol} FINAL: "
                    f"hybrid={signal_values['composite']:.4f}, "
                    f"confidence={signal_values['confidence']:.2%}"
                )
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Failed to calculate signals: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {}
      
    def calculate_target_positions(self, signals: Dict) -> Dict[str, int]:
        """Calculate target positions with leverage awareness + VOLUME CONSTRAINTS"""
        
        account = self.api.get_account()
        equity = float(account.equity)
        
        # Get current positions
        current_positions = self.api.list_positions()
        current_exposure = {}
        total_current_exposure = 0
        
        for pos in current_positions:
            pos_value = abs(float(pos.market_value))
            current_exposure[pos.symbol] = pos_value
            total_current_exposure += pos_value
        
        current_leverage = total_current_exposure / equity if equity > 0 else 0
        
        self.logger.info(f'📊 CURRENT STATE:')
        self.logger.info(f'   Equity: ${equity:,.2f}')
        self.logger.info(f'   Current Exposure: ${total_current_exposure:,.2f}')
        self.logger.info(f'   Current Leverage: {current_leverage:.2f}x')
        
        # Calculate max available exposure
        max_total_exposure = equity * self.config.max_leverage
        
        self.logger.info(f'   Max Total Exposure: ${max_total_exposure:,.2f} ({self.config.max_leverage:.2f}x)')
        
        # Get leverage limits from config
        min_leverage = self.config.__dict__.get('min_leverage', 1.0)
        max_leverage = self.config.max_leverage
        
        # Calculate target positions
        targets = {}
        target_exposures = {}
        
        for symbol, signal_dict in signals.items():
            composite = signal_dict.get('composite', 0)
            confidence = signal_dict.get('confidence', 0.5)
            
            # Get current price AND volume data
            try:
                # Get recent bars for volume calculation
                bars = self.api.get_bars(symbol, tradeapi.TimeFrame.Day, limit=20)
                
                if len(bars) < 20:
                    self.logger.warning(f'Not enough volume data for {symbol}, skipping')
                    targets[symbol] = 0
                    target_exposures[symbol] = 0
                    continue
                
                price = bars['close'].iloc[-1]
                
                # ✅ NOUVEAU: Calculate ADV (Average Daily Volume)
                adv_20 = bars['volume'].mean()  # 20-day average
                
                # ✅ NOUVEAU: Max shares based on liquidity (10% of ADV)
                max_shares_liquidity = int(adv_20 * 0.10)
                
                self.logger.debug(f'{symbol}: ADV={adv_20:,.0f}, Max shares (10% ADV)={max_shares_liquidity:,}')
                
            except Exception as e:
                self.logger.warning(f'Could not get price/volume for {symbol}: {e}')
                targets[symbol] = 0
                target_exposures[symbol] = 0
                continue
            
            # Signal strength [-1, +1]
            signal_strength = float(np.tanh(composite * 2))
            
            # Adjust by confidence
            adjusted_strength = signal_strength * confidence
            
            # Scale leverage between MIN and MAX based on signal strength
            abs_strength = abs(adjusted_strength)
            
            # Interpolate between min and max leverage
            leverage_to_use = min_leverage + (max_leverage - min_leverage) * abs_strength
            
            # Keep the sign (long/short)
            signed_leverage = leverage_to_use * np.sign(adjusted_strength)
            
            # Target notional
            target_notional = equity * signed_leverage * self.config.max_position_size
            
            # Convert to shares
            shares_from_signal = int(target_notional / price)
            
            # Apply hard safety cap (portfolio constraint)
            max_shares_portfolio = int(
                equity * 
                self.config.max_position_size * 
                max_leverage / 
                price
            )
            
            # ✅ NOUVEAU: Apply BOTH portfolio cap AND liquidity cap
            shares_capped = int(np.clip(
                shares_from_signal, 
                -min(max_shares_portfolio, max_shares_liquidity),
                min(max_shares_portfolio, max_shares_liquidity)
            ))
            
            # Check if liquidity constraint is binding
            if abs(shares_from_signal) > max_shares_liquidity:
                self.logger.warning(
                    f'⚠️  {symbol}: Liquidity constrained! '
                    f'Signal wants {shares_from_signal:,} but ADV allows max {max_shares_liquidity:,}'
                )
            
            targets[symbol] = shares_capped
            target_exposures[symbol] = abs(shares_capped * price)
            
            self.logger.info(
                f"\n   {symbol}:"
                f"\n      Signal:           {composite:+.4f}"
                f"\n      Confidence:       {confidence:.2%}"
                f"\n      Leverage Used:    {leverage_to_use:.2f}x"
                f"\n      ADV (20d):        {adv_20:,.0f} shares"
                f"\n      Max (liquidity):  {max_shares_liquidity:,} shares"
                f"\n      Max (portfolio):  {max_shares_portfolio:,} shares"
                f"\n      Target Shares:    {shares_capped:,} shares"
            )
        
        # Calculate total target exposure
        total_target_exposure = sum(target_exposures.values())
        target_leverage = total_target_exposure / equity if equity > 0 else 0
        
        self.logger.info(f'📊 TARGET STATE (before scaling):')
        self.logger.info(f'   Target Exposure: ${total_target_exposure:,.2f}')
        self.logger.info(f'   Target Leverage: {target_leverage:.2f}x')
        
        # Scale down if target leverage exceeds max
        if target_leverage > self.config.max_leverage:
            scale_factor = self.config.max_leverage / target_leverage
            
            self.logger.warning(f'⚠️  TARGET LEVERAGE TOO HIGH!')
            self.logger.warning(f'   Scaling down all positions by {scale_factor:.2%}')
            
            for symbol in targets:
                targets[symbol] = int(targets[symbol] * scale_factor)
                target_exposures[symbol] = target_exposures[symbol] * scale_factor
            
            # Recalculate
            total_target_exposure = sum(target_exposures.values())
            target_leverage = total_target_exposure / equity if equity > 0 else 0
            
            self.logger.info(f'📊 TARGET STATE (after scaling):')
            self.logger.info(f'   Target Exposure: ${total_target_exposure:,.2f}')
            self.logger.info(f'   Target Leverage: {target_leverage:.2f}x')
        
        self.logger.info(f'   ✅ Target leverage OK: {target_leverage:.2f}x <= {self.config.max_leverage:.2f}x')
        
        return targets
    
    def estimate_slippage(self, symbol: str, shares: int, adv: float, volatility: float) -> float:
        """
        Estimate slippage cost for a trade
        
        Args:
            symbol: Stock symbol
            shares: Number of shares to trade
            adv: Average Daily Volume
            volatility: Recent volatility (annualized)
        
        Returns:
            Estimated slippage in basis points (bps)
        """
        
        if adv == 0:
            return 100.0  # 1% if no volume data
        
        # Participation rate (% of ADV)
        participation = abs(shares) / adv
        
        # Base spread (depends on stock liquidity)
        # Large cap: ~5-10 bps, Mid cap: ~10-20 bps, Small cap: ~20-50 bps
        base_spread_bps = 10.0
        
        # Market impact (square root model)
        # Impact ∝ sqrt(participation) * volatility
        impact_bps = 10.0 * np.sqrt(participation) * (volatility / 0.20)
        
        # Total slippage
        total_slippage_bps = base_spread_bps/2 + impact_bps
        
        # Cap at reasonable level
        total_slippage_bps = min(total_slippage_bps, 100.0)  # Max 1%
        
        return total_slippage_bps

    def log_estimated_costs(self, targets: Dict[str, int]):
        """
        Log estimated trading costs BEFORE executing
        """
        self.logger.info('\n💰 ESTIMATED TRADING COSTS:')
        
        total_cost_usd = 0
        
        for symbol, target_shares in targets.items():
            if target_shares == 0:
                continue
            
            try:
                # Get volume and volatility
                bars = self.api.get_bars(symbol, tradeapi.TimeFrame.Day, limit=20)
                
                if len(bars) < 20:
                    continue
                
                price = bars['close'].iloc[-1]
                adv = bars['volume'].mean()
                returns = bars['close'].pct_change()
                volatility = returns.std() * np.sqrt(252)
                
                # Get current position
                current_qty = 0
                try:
                    pos = self.api.get_position(symbol)
                    if pos:
                        current_qty = int(float(pos.qty))
                except:
                    pass
                
                # Shares to trade
                delta = target_shares - current_qty
                
                if delta == 0:
                    continue
                
                # Estimate slippage
                slippage_bps = self.estimate_slippage(symbol, delta, adv, volatility)
                
                # Cost in USD
                notional = abs(delta) * price
                cost_usd = notional * (slippage_bps / 10000)
                
                total_cost_usd += cost_usd
                
                self.logger.info(
                    f'   {symbol}: {abs(delta):,} shares @ ${price:.2f}'
                    f'\n      Participation: {abs(delta)/adv:.2%} of ADV'
                    f'\n      Volatility: {volatility:.1%}'
                    f'\n      Est. Slippage: {slippage_bps:.1f} bps'
                    f'\n      Est. Cost: ${cost_usd:.2f}'
                )
            
            except Exception as e:
                self.logger.warning(f'Could not estimate cost for {symbol}: {e}')
        
        account = self.api.get_account()
        portfolio_value = float(account.portfolio_value)
        
        cost_pct = (total_cost_usd / portfolio_value) * 100 if portfolio_value > 0 else 0
        
        self.logger.info(f'\n   TOTAL ESTIMATED COST: ${total_cost_usd:.2f} ({cost_pct:.3f}% of portfolio)')
    


    def execute_rebalance(self, target_positions: Dict):
        """Execute rebalancing avec buying power check automatique"""
        from datetime import datetime
        import time
        
        try:
            # ✅ PRE-CHECK: Leverage
            self.logger.info("\n🔍 Pre-rebalance leverage check...")
            leverage_ok = self.check_and_reduce_leverage()
            
            if not leverage_ok:
                self.logger.warning("Leverage too high, rebalance aborted")
                return
            
            self.log_estimated_costs(target_positions)
            
            # ✅ PRE-CHECK: Buying Power
            account = self.api.get_account()
            buying_power = float(account.buying_power)
            portfolio_value = float(account.portfolio_value)
            
            self.logger.info(f"\n💰 BUYING POWER CHECK:")
            self.logger.info(f"   Portfolio Value: ${portfolio_value:,.2f}")
            self.logger.info(f"   Buying Power:    ${buying_power:,.2f}")
            
            # Calculate total notional needed for NEW orders
            current_positions = {p.symbol: int(p.qty) if p.side != 'short' else -int(p.qty) 
                            for p in self.api.list_positions()}
            
            total_notional_needed = 0
            order_plan = {}
            
            for symbol, target_qty in target_positions.items():
                current_qty = current_positions.get(symbol, 0)
                delta = target_qty - current_qty
                
                if abs(delta) < self.config.min_trade_size:
                    continue
                
                try:
                    latest = self.api.get_latest_bar(symbol)
                    price = float(latest.c)
                except:
                    bars = self.api.get_bars(symbol, tradeapi.TimeFrame.Day, limit=1)
                    if len(bars) == 0:
                        continue
                    price = float(bars['close'].iloc[-1])
                
                notional = abs(delta * price)
                total_notional_needed += notional
                order_plan[symbol] = {'delta': delta, 'price': price, 'notional': notional}
            
            self.logger.info(f"   Total Needed:    ${total_notional_needed:,.2f}")
            
            # ✅ SCALE DOWN if needed
            if total_notional_needed > buying_power:
                scale_factor = (buying_power / total_notional_needed) * 0.95  # 95% safety margin
                
                self.logger.warning(f"\n⚠️  INSUFFICIENT BUYING POWER!")
                self.logger.warning(f"   Needed:    ${total_notional_needed:,.2f}")
                self.logger.warning(f"   Available: ${buying_power:,.2f}")
                self.logger.warning(f"   Shortage:  ${total_notional_needed - buying_power:,.2f}")
                self.logger.warning(f"   Scaling down by: {scale_factor:.2%}")
                
                # Scale down target positions
                scaled_targets = {}
                for symbol, target_qty in target_positions.items():
                    current_qty = current_positions.get(symbol, 0)
                    delta = target_qty - current_qty
                    scaled_delta = int(delta * scale_factor)
                    scaled_target = current_qty + scaled_delta
                    scaled_targets[symbol] = scaled_target
                    
                    self.logger.info(
                        f"      {symbol}: {target_qty} → {scaled_target} "
                        f"(delta: {delta} → {scaled_delta})"
                    )
                
                target_positions = scaled_targets
            else:
                self.logger.info(f"   ✅ Buying power sufficient")
            
            # Execute orders
            alpaca_positions = self.api.list_positions()
            current_positions = {}
            
            for p in alpaca_positions:
                qty = int(p.qty)
                if p.side == 'short':
                    qty = -qty
                current_positions[p.symbol] = qty
            
            self.logger.info(f"\n📤 EXECUTING ORDERS:")
            self.logger.info(f"   Current positions: {current_positions}")
            
            orders_executed = []
            
            for symbol, target_qty in target_positions.items():
                current_qty = current_positions.get(symbol, 0)
                delta = int(target_qty - current_qty)
                
                self.logger.info(
                    f"\n   {symbol}: current={current_qty}, target={target_qty}, delta={delta}"
                )
                
                if abs(delta) < self.config.min_trade_size:
                    self.logger.info(f"      Delta too small, skipping")
                    continue
                
                # Determine side
                if delta > 0:
                    side = 'buy'
                    qty = abs(delta)
                else:
                    side = 'sell'
                    qty = abs(delta)
                
                self.logger.info(f"      Submitting: {side.upper()} {qty} {symbol}")
                
                try:
                    # Handle position transitions (SHORT→LONG, LONG→SHORT)
                    if current_qty < 0 and target_qty > 0:
                        # SHORT → LONG
                        self.logger.info(f"      Transition: SHORT → LONG")
                        
                        # Close short
                        close_qty = abs(current_qty)
                        self.logger.info(f"         Step 1: Close SHORT {close_qty}")
                        close_order = self.api.submit_order(
                            symbol=symbol, qty=close_qty, side='buy',
                            type='market', time_in_force='day'
                        )
                        self.logger.info(f"         ✅ Order: {close_order.id}")
                        time.sleep(2)
                        
                        # Open long
                        if target_qty > 0:
                            self.logger.info(f"         Step 2: Open LONG {target_qty}")
                            long_order = self.api.submit_order(
                                symbol=symbol, qty=target_qty, side='buy',
                                type='market', time_in_force='day'
                            )
                            self.logger.info(f"         ✅ Order: {long_order.id}")
                            orders_executed.append(long_order)
                        
                    elif current_qty > 0 and target_qty < 0:
                        # LONG → SHORT
                        self.logger.info(f"      Transition: LONG → SHORT")
                        
                        # Close long
                        close_qty = abs(current_qty)
                        self.logger.info(f"         Step 1: Close LONG {close_qty}")
                        close_order = self.api.submit_order(
                            symbol=symbol, qty=close_qty, side='sell',
                            type='market', time_in_force='day'
                        )
                        self.logger.info(f"         ✅ Order: {close_order.id}")
                        time.sleep(2)
                        
                        # Open short
                        if target_qty < 0:
                            short_qty = abs(target_qty)
                            self.logger.info(f"         Step 2: Open SHORT {short_qty}")
                            short_order = self.api.submit_order(
                                symbol=symbol, qty=short_qty, side='sell',
                                type='market', time_in_force='day'
                            )
                            self.logger.info(f"         ✅ Order: {short_order.id}")
                            orders_executed.append(short_order)
                    
                    else:
                        # Simple add/reduce
                        order = self.api.submit_order(
                            symbol=symbol, qty=qty, side=side,
                            type='market', time_in_force='day'
                        )
                        self.logger.info(f"      ✅ Order: {order.id}")
                        orders_executed.append(order)
                        
                        # Store in DB
                        self.db.insert_order({
                            'id': order.id,
                            'timestamp': datetime.now().isoformat(),
                            'symbol': symbol,
                            'side': side,
                            'qty': qty,
                            'type': 'market',
                            'status': order.status,
                            'filled_qty': 0,
                            'filled_avg_price': 0,
                            'commission': 0
                        })
                    
                except Exception as e:
                    error_msg = str(e)
                    
                    if 'insufficient buying power' in error_msg.lower():
                        self.logger.error(f"      ❌ STILL insufficient buying power for {symbol}")
                        self.logger.error(f"         This order will be skipped")
                        # Continue with other symbols
                        continue
                    else:
                        self.logger.error(f"      ❌ Order failed: {e}")
                        import traceback
                        self.logger.error(traceback.format_exc())
            
            # Wait for fills
            if orders_executed:
                self.wait_for_fills(orders_executed, timeout=30)
            
            # Check fills
            try:
                recent_orders = self.api.list_orders(status='all', limit=30)
                for order in recent_orders:
                    if order.status == 'filled':
                        order_time = order.filled_at or order.submitted_at
                        if order_time:
                            time_diff = (datetime.now(order_time.tzinfo) - order_time).total_seconds()
                            if time_diff < 120:  # Last 2 minutes
                                self.logger.info(
                                    f"   ✅ FILLED: {order.side.upper()} {order.filled_qty} {order.symbol} "
                                    f"@ ${float(order.filled_avg_price):.2f}"
                                )
            except Exception as e:
                self.logger.warning(f"Could not fetch order status: {e}")
            
            self.last_rebalance = datetime.now()
            
            # ✅ POST-CHECK: Leverage
            self.logger.info("\n🔍 Post-rebalance leverage check...")
            self.check_and_reduce_leverage()
            
        except Exception as e:
            self.logger.error(f"Rebalance failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

    def wait_for_fills(self, orders, timeout=30):
        """Wait for orders to fill - ignore ghost orders"""
        import time
        
        self.logger.info('⏳ Waiting for order fills...')
        
        # Get order IDs we're actually waiting for
        submitted_order_ids = [str(o.id) for o in orders]
        
        start_time = time.time()
        filled_orders = []
        
        while time.time() - start_time < timeout:
            time.sleep(2)
            
            try:
                # Get recent orders
                recent_orders = self.api.list_orders(status='all', limit=50)
                
                for order in recent_orders:
                    # Only check orders we submitted
                    if str(order.id) not in submitted_order_ids:
                        continue
                    
                    # Only log if actually filled with qty > 0
                    filled_qty = float(order.filled_qty)
                    
                    if filled_qty > 0 and order.status == 'filled':
                        if order.id not in [o.id for o in filled_orders]:
                            filled_orders.append(order)
                            
                            avg_price = float(order.filled_avg_price)
                            
                            self.logger.info(
                                f'   ✅ FILLED: {order.side.upper()} '
                                f'{filled_qty:.0f} {order.symbol} @ ${avg_price:.2f}'
                            )
            
            except Exception as e:
                self.logger.error(f'Error checking orders: {e}')
        
        return filled_orders
    
    def reconcile_positions(self):
        """Reconcile positions"""
        try:
            broker_positions = {p.symbol: int(p.qty) 
                              for p in self.api.list_positions()}
            
            self.logger.info("POSITION RECONCILIATION")
            
            for symbol, qty in broker_positions.items():
                self.logger.info(f"   {symbol}: {qty} shares")
                
                try:
                    position = self.api.get_position(symbol)
                    
                    self.db.insert_position(
                        datetime.now().isoformat(),
                        symbol,
                        {
                            'qty': int(position.qty),
                            'avg_entry_price': float(position.avg_entry_price),
                            'current_price': float(position.current_price),
                            'market_value': float(position.market_value),
                            'unrealized_pnl': float(position.unrealized_pl)
                        }
                    )
                    
                except Exception as e:
                    self.logger.error(f"Failed to get position {symbol}: {e}")
            
            self.last_reconciliation = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Reconciliation failed: {e}")
    
    def log_portfolio_status(self):
        """Log current portfolio status"""
        try:
            account = self.api.get_account()
            
            portfolio_value = float(account.portfolio_value)
            cash = float(account.cash)
            equity = float(account.equity)
            buying_power = float(account.buying_power)
            
            positions = self.api.list_positions()
            
            self.logger.info("="*70)
            self.logger.info("PORTFOLIO STATUS")
            self.logger.info("="*70)
            self.logger.info(f"   Portfolio Value:  ${portfolio_value:,.2f}")
            self.logger.info(f"   Cash:             ${cash:,.2f}")
            self.logger.info(f"   Equity:           ${equity:,.2f}")
            self.logger.info(f"   Buying Power:     ${buying_power:,.2f}")
            self.logger.info(f"   Positions:        {len(positions)}")
            
            self.db.insert_portfolio(
                datetime.now().isoformat(),
                {
                    'cash': cash,
                    'portfolio_value': portfolio_value,
                    'equity': equity,
                    'buying_power': buying_power,
                    'daily_pnl': 0,
                    'total_pnl': portfolio_value - 100000
                }
            )
            
            if positions:
                self.logger.info("\n   Current Positions:")
                for p in positions:
                    self.logger.info(
                        f"      {p.symbol}: {int(p.qty)} shares  "
                        f"${float(p.market_value):,.2f}  "
                        f"P&L: ${float(p.unrealized_pl):,.2f}"
                    )
            
            self.logger.info("="*70)
            
        except Exception as e:
            self.logger.error(f"Failed to log portfolio: {e}")

    def check_and_reduce_leverage(self, max_attempts: int = 3) -> bool:
        """
        Check leverage et réduit automatiquement les positions si dépassé
        
        Args:
            max_attempts: Nombre maximum de tentatives de réduction
        
        Returns:
            True si leverage OK, False si impossible de réduire assez
        """
        try:
            for attempt in range(max_attempts):
                account = self.api.get_account()
                positions = self.api.list_positions()
                
                # Calculate current leverage
                equity = float(account.equity)
                portfolio_value = float(account.portfolio_value)
                
                if equity <= 0:
                    self.logger.critical("EQUITY IS ZERO OR NEGATIVE!")
                    return False
                
                # Total exposure
                total_exposure = sum(abs(float(p.market_value)) for p in positions)
                current_leverage = total_exposure / equity if equity > 0 else 0
                
                # Max leverage from config
                max_leverage = self.config.max_leverage
                
                self.logger.info(f"\n📊 LEVERAGE CHECK (Attempt {attempt + 1}/{max_attempts}):")
                self.logger.info(f"   Equity:           ${equity:,.2f}")
                self.logger.info(f"   Total Exposure:   ${total_exposure:,.2f}")
                self.logger.info(f"   Current Leverage: {current_leverage:.2f}x")
                self.logger.info(f"   Max Leverage:     {max_leverage:.2f}x")
                
                # Check if leverage OK
                if current_leverage <= max_leverage:
                    self.logger.info(f"   ✅ Leverage OK ({current_leverage:.2f}x <= {max_leverage:.2f}x)")
                    return True
                
                # LEVERAGE EXCEEDED - Need to reduce
                excess_leverage = current_leverage - max_leverage
                reduction_pct = excess_leverage / current_leverage
                
                # Add safety margin (reduce a bit more to avoid being right at the edge)
                reduction_pct *= 1.1  # Reduce 10% more than strictly necessary
                
                self.logger.warning(f"\n   ⚠️  LEVERAGE EXCEEDED!")
                self.logger.warning(f"   Current:  {current_leverage:.2f}x")
                self.logger.warning(f"   Max:      {max_leverage:.2f}x")
                self.logger.warning(f"   Excess:   {excess_leverage:.2f}x")
                self.logger.warning(f"   Reducing: {reduction_pct:.2%}")
                
                # Alert
                self.alert_manager.alert(
                    f"LEVERAGE EXCEEDED - Attempt {attempt + 1}",
                    f"Current: {current_leverage:.2f}x, Max: {max_leverage:.2f}x\n"
                    f"Reducing positions by {reduction_pct:.2%}",
                    level="WARNING"
                )
                
                # Sort positions by size (reduce largest first)
                positions_sorted = sorted(
                    positions,
                    key=lambda p: abs(float(p.market_value)),
                    reverse=True
                )
                
                self.logger.info(f"\n🔧 REDUCING POSITIONS:")
                
                positions_reduced = 0
                
                for position in positions_sorted:
                    symbol = position.symbol
                    current_qty = int(position.qty)
                    current_value = abs(float(position.market_value))
                    
                    # Calculate shares to reduce from THIS position
                    shares_to_reduce = int(abs(current_qty) * reduction_pct)
                    
                    # Minimum 1 share if any reduction needed
                    if shares_to_reduce < 1 and reduction_pct > 0.01:
                        shares_to_reduce = 1
                    
                    if shares_to_reduce == 0:
                        continue
                    
                    # Cap at current position size
                    shares_to_reduce = min(shares_to_reduce, abs(current_qty))
                    
                    # Determine side
                    if current_qty > 0:
                        side = 'sell'
                    elif current_qty < 0:
                        side = 'buy'  # Buy to cover short
                    else:
                        continue
                    
                    self.logger.info(
                        f"   {symbol}: {side.upper()} {shares_to_reduce} shares "
                        f"(current: {current_qty}, {reduction_pct:.1%} reduction)"
                    )
                    
                    try:
                        order = self.api.submit_order(
                            symbol=symbol,
                            qty=shares_to_reduce,
                            side=side,
                            type='market',
                            time_in_force='day'
                        )
                        
                        self.logger.info(f"      ✅ Deleveraging order: {order.id}")
                        positions_reduced += 1
                        
                        # Store in DB
                        self.db.insert_order({
                            'id': order.id,
                            'timestamp': datetime.now().isoformat(),
                            'symbol': symbol,
                            'side': side,
                            'qty': shares_to_reduce,
                            'type': 'market',
                            'status': order.status,
                            'filled_qty': 0,
                            'filled_avg_price': 0,
                            'commission': 0
                        })
                        
                    except Exception as e:
                        self.logger.error(f"      ❌ Failed to reduce {symbol}: {e}")
                
                if positions_reduced == 0:
                    self.logger.error(f"   ❌ Could not reduce any positions")
                    return False
                
                # Wait for orders to fill
                import time
                self.logger.info(f"\n⏳ Waiting for deleveraging orders to fill...")
                time.sleep(5)
                
                # Loop will re-check leverage after this attempt
            
            # If we get here, we tried max_attempts and still over leverage
            account_final = self.api.get_account()
            positions_final = self.api.list_positions()
            equity_final = float(account_final.equity)
            exposure_final = sum(abs(float(p.market_value)) for p in positions_final)
            leverage_final = exposure_final / equity_final if equity_final > 0 else 0
            
            self.logger.error(f"\n❌ FAILED TO REDUCE LEVERAGE AFTER {max_attempts} ATTEMPTS")
            self.logger.error(f"   Final Leverage: {leverage_final:.2f}x")
            self.logger.error(f"   Max Leverage:   {max_leverage:.2f}x")
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to check/reduce leverage: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    
    def should_rebalance(self) -> bool:
        """Check if it's time to rebalance"""
        if self.last_rebalance is None:
            return True
        
        time_since = (datetime.now() - self.last_rebalance).total_seconds() / 60
        return time_since >= self.config.rebalance_frequency_minutes
    
    def should_reconcile(self) -> bool:
        """Check if it's time to reconcile"""
        if self.last_reconciliation is None:
            return True
        
        time_since = (datetime.now() - self.last_reconciliation).total_seconds() / 60
        return time_since >= self.config.reconciliation_interval_minutes
    
    def flatten_all_positions(self):
        """Emergency: flatten all positions"""
        try:
            self.logger.critical("FLATTENING ALL POSITIONS")
            
            positions = self.api.list_positions()
            
            for position in positions:
                try:
                    self.api.close_position(position.symbol)
                    self.logger.info(f"   Closed {position.symbol}")
                except Exception as e:
                    self.logger.error(f"   Failed to close {position.symbol}: {e}")
            
            self.emergency_stop = True
            
        except Exception as e:
            self.logger.critical(f"Failed to flatten: {e}")
    
    def run(self):
        """Main trading loop"""
        self.is_running = True
        
        self.logger.info("STARTING PAPER TRADING ENGINE")
        self.logger.info(f"   Symbols: {', '.join(self.config.symbols)}")
        self.logger.info(f"   Rebalance: Every {self.config.rebalance_frequency_minutes} min")
        self.logger.info(f"   Check: Every {self.config.check_interval_seconds} sec")
        
        self.alert_manager.alert(
            "PAPER TRADING STARTED",
            f"Engine started at {datetime.now()}\nSymbols: {', '.join(self.config.symbols)}",
            level="INFO"
        )
        
        iteration = 0
        
        try:
            while self.is_running and not self.emergency_stop:
                iteration += 1
                
                # Check market hours
                clock = self.api.get_clock()
                
                if not clock.is_open:
                    next_open = clock.next_open
                    self.logger.info(f"Market closed. Next open: {next_open}")
                    time.sleep(300)
                    continue
                
                # Portfolio status
                if iteration % 10 == 1:
                    self.log_portfolio_status()
                
                # Check other risk limits
                account = self.api.get_account()
                positions_list = self.api.list_positions()
                positions = {p.symbol: {
                    'market_value': float(p.market_value),
                    'unrealized_pnl': float(p.unrealized_pl)
                } for p in positions_list}
                
                # Daily loss / drawdown checks (reste pareil)
                if not self.risk_monitor.check_risk_limits(
                    {
                        'portfolio_value': account.portfolio_value,
                        'equity': account.equity,
                        'daily_pnl': 0
                    },
                    positions
                ):
                    self.flatten_all_positions()
                    break
                
                # Rebalance
                if self.should_rebalance():
                    self.logger.info("Starting rebalance...")
                    
                    signals = self.calculate_signals()
                    
                    if signals:
                        target_positions = self.calculate_target_positions(signals)


                        self.logger.warning("No signals calculated, skipping rebalance")
                        leverage_ok = self.check_and_reduce_leverage()

                        if leverage_ok:
                            self.execute_rebalance(target_positions)
                        else:
                            self.logger.warning("Leverage too high, rebalance skipped")
                    else:
                        self.logger.warning("No signals calculated, skipping rebalance")
                
                if self.should_reconcile():
                    self.reconcile_positions()
                
                time.sleep(self.config.check_interval_seconds)
                
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received")
            self.stop()
            
        except Exception as e:
            self.logger.critical(f"CRITICAL ERROR: {e}")
            import traceback
            self.logger.critical(traceback.format_exc())
            self.flatten_all_positions()
    
    def stop(self):
        """Stop the engine gracefully with proper error handling"""
        self.logger.info("="*70)
        self.logger.info("STOPPING PAPER TRADING ENGINE")
        self.logger.info("="*70)
        
        self.is_running = False
        
        try:
            account = self.api.get_account()
            positions = self.api.list_positions()
            
            self.logger.info("\nFINAL PORTFOLIO STATUS:")
            self.logger.info(f"   Portfolio Value: ${float(account.portfolio_value):,.2f}")
            self.logger.info(f"   Cash:            ${float(account.cash):,.2f}")
            self.logger.info(f"   Positions:       {len(positions)}")
            
            for p in positions:
                self.logger.info(
                    f"      {p.symbol}: {int(p.qty)} shares, "
                    f"P&L: ${float(p.unrealized_pl):,.2f}"
                )
        except Exception as e:
            self.logger.error(f"Failed to get final status: {e}")
        
        try:
            if hasattr(self, 'db') and self.db:
                self.db.close()
                self.logger.info("Database closed")
        except Exception as e:
            self.logger.error(f"Failed to close database: {e}")
        
        self.logger.info("="*70)
        self.logger.info("ENGINE STOPPED SUCCESSFULLY")
        self.logger.info("="*70)
    
    def initialize_pipeline_weights(self):
        """Calcule les poids du pipeline une fois"""
        self.logger.info("Initializing pipeline weights...")
        
        # Get historical data
        symbol = self.config.symbols[0]  # Use first symbol
        bars = self.get_market_data(symbol)
        
        if len(bars) < 252:
            self.logger.warning(f"Not enough data to initialize weights, using equal weights")
            # Equal weights fallback
            n = len(self.pipeline.signals)
            self.pipeline.weights = {name: 1.0/n for name in self.pipeline.signals.keys()}
            return
        
        prices = bars['close']
        returns = prices.pct_change()
        volume = bars.get('volume', pd.Series([0]*len(bars)))
        
        # Run pipeline once
        try:
            results = self.pipeline.run_full_pipeline_vectorized(prices, returns, volume)
            
            if self.pipeline.weights:
                self.logger.info("✅ Pipeline weights initialized:")
                for name, weight in self.pipeline.weights.items():
                    self.logger.info(f"   {name}: {weight:.3f}")
            else:
                self.logger.warning("Pipeline weights empty, using equal weights")
                n = len(self.pipeline.signals)
                self.pipeline.weights = {name: 1.0/n for name in self.pipeline.signals.keys()}
        
        except Exception as e:
            self.logger.error(f"Failed to initialize weights: {e}")
            n = len(self.pipeline.signals)
            self.pipeline.weights = {name: 1.0/n for name in self.pipeline.signals.keys()}


# ============================================================================
# CLI INTERFACE
# ============================================================================

def create_default_config():
    """Create default config file"""
    config = TradingConfig()
    config.save('trading_config.json')
    print("Default config created: trading_config.json")
    print("\nIMPORTANT: Edit the config file and add your Alpaca API keys!")
    print("   Get free paper trading keys at: https://alpaca.markets")


def run_paper_trading(config_path: str = 'trading_config.json'):
    """Run paper trading with config"""
    if not Path(config_path).exists():
        print(f"Config file not found: {config_path}")
        print("   Creating default config...")
        create_default_config()
        return
    
    config = TradingConfig.load(config_path)
    
    if config.alpaca_api_key == "YOUR_ALPACA_KEY":
        print("Please edit trading_config.json and add your Alpaca API keys")
        return
    
    print("Loading pipeline...")
    try:
        from enhanced_pipeline import (
            EnhancedProductionPipeline, 
            VolatilitySignal, 
            MomentumSignal, 
            TailRiskSignal, 
            BetaSignal, 
            CarrySignal
        )
    except ImportError as e:
        print(f"Failed to import pipeline: {e}")
        return
    
    pipeline = EnhancedProductionPipeline()
    pipeline.add_signal(VolatilitySignal)
    pipeline.add_signal(MomentumSignal)
    pipeline.add_signal(TailRiskSignal)
    pipeline.add_signal(BetaSignal)
    pipeline.add_signal(CarrySignal)
    
    print("Pipeline loaded")
    
    engine = PaperTradingEngine(pipeline, config)
    
    try:
        engine.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        engine.stop()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys
    
    print("="*70)
    print("PAPER TRADING ENGINE")
    print("="*70)
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "init":
            create_default_config()
        elif command == "run":
            config_path = sys.argv[2] if len(sys.argv) > 2 else 'trading_config.json'
            run_paper_trading(config_path)
        else:
            print(f"Unknown command: {command}")
            print("\nUsage:")
            print("  python paper_trading.py init")
            print("  python paper_trading.py run [config]")
    else:
        print("\nUsage:")
        print("  python paper_trading.py init          # Create default config")
        print("  python paper_trading.py run [config]  # Run paper trading")
        print("\nQuick start:")
        print("  1. python paper_trading.py init")
        print("  2. Edit trading_config.json (add your Alpaca keys)")
        print("  3. python paper_trading.py run")