"""
Data Provider Module
====================

Handles real-time and historical data fetching for the risk management dashboard.
Supports multiple data sources including market data APIs, databases, and file uploads.
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import json
import redis
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import asyncio
import aiohttp
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class MarketData:
    """Market data structure"""
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    timestamp: datetime
    bid: float = 0.0
    ask: float = 0.0
    open_price: float = 0.0
    high: float = 0.0
    low: float = 0.0

@dataclass
class RiskFactorData:
    """Risk factor data structure"""
    factor_name: str
    current_value: float
    change_1d: float
    change_1w: float
    change_1m: float
    volatility: float
    last_updated: datetime

class DataProvider:
    """Real-time and historical data provider"""

    def __init__(self, use_redis: bool = True):
        """Initialize data provider"""
        self.use_redis = use_redis
        self.cache_expiry = 300  # 5 minutes

        # Redis connection for real-time data
        if self.use_redis:
            try:
                self.redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
                self.redis_client.ping()
                self.redis_connected = True
            except:
                self.redis_connected = False
                st.warning("Redis not available - using local caching")
        else:
            self.redis_connected = False

        # Initialize local cache
        self.local_cache = {}

        # Database connection for historical data
        self.db_path = "risk_dashboard.db"
        self.init_database()

    def init_database(self):
        """Initialize SQLite database for historical data storage"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    price REAL NOT NULL,
                    volume INTEGER,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    source TEXT DEFAULT 'API'
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS risk_factors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    factor_name TEXT NOT NULL,
                    value REAL NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    portfolio_name TEXT NOT NULL,
                    total_value REAL NOT NULL,
                    var_1d REAL,
                    expected_shortfall REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            conn.commit()
            conn.close()

        except Exception as e:
            st.error(f"Database initialization failed: {e}")

    def get_real_time_prices(self, symbols: List[str]) -> Dict[str, MarketData]:
        """Get real-time market prices for given symbols"""
        market_data = {}

        try:
            # Check cache first
            cached_data = self._get_cached_prices(symbols)
            if cached_data:
                return cached_data

            # Fetch from Yahoo Finance (free tier)
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    hist = ticker.history(period="2d")

                    if not hist.empty:
                        current_price = hist['Close'].iloc[-1]
                        prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                        change = current_price - prev_price
                        change_percent = (change / prev_price) * 100 if prev_price != 0 else 0

                        market_data[symbol] = MarketData(
                            symbol=symbol,
                            price=float(current_price),
                            change=float(change),
                            change_percent=float(change_percent),
                            volume=int(hist['Volume'].iloc[-1]),
                            timestamp=datetime.now(),
                            open_price=float(hist['Open'].iloc[-1]),
                            high=float(hist['High'].iloc[-1]),
                            low=float(hist['Low'].iloc[-1])
                        )
                    else:
                        # Generate simulated data if no real data available
                        market_data[symbol] = self._generate_simulated_data(symbol)

                except Exception as e:
                    # Generate simulated data on error
                    market_data[symbol] = self._generate_simulated_data(symbol)

            # Cache the results
            self._cache_prices(market_data)

            return market_data

        except Exception as e:
            # Return simulated data as fallback
            return {symbol: self._generate_simulated_data(symbol) for symbol in symbols}

    def _generate_simulated_data(self, symbol: str) -> MarketData:
        """Generate simulated market data for demo purposes"""
        # Set base prices for common symbols
        base_prices = {
            'AAPL': 150.0,
            'MSFT': 300.0,
            'GOOGL': 1500.0,
            'TSLA': 600.0,
            'SPY': 400.0,
            'QQQ': 350.0,
            'BTC-USD': 50000.0,
            'ETH-USD': 3000.0,
            'GLD': 180.0,
            'TLT': 120.0
        }

        base_price = base_prices.get(symbol, 100.0)

        # Add some randomness
        price_variation = np.random.normal(0, 0.02)  # 2% volatility
        current_price = base_price * (1 + price_variation)

        change = base_price * price_variation
        change_percent = price_variation * 100

        # Random volume
        volume = np.random.randint(1000000, 10000000)

        return MarketData(
            symbol=symbol,
            price=current_price,
            change=change,
            change_percent=change_percent,
            volume=volume,
            timestamp=datetime.now(),
            bid=current_price * 0.999,
            ask=current_price * 1.001,
            open_price=current_price * (1 + np.random.normal(0, 0.01)),
            high=current_price * (1 + abs(np.random.normal(0, 0.015))),
            low=current_price * (1 - abs(np.random.normal(0, 0.015)))
        )

    def _get_cached_prices(self, symbols: List[str]) -> Optional[Dict[str, MarketData]]:
        """Get cached price data"""
        try:
            if self.redis_connected:
                cached_data = {}
                for symbol in symbols:
                    cached = self.redis_client.get(f"price:{symbol}")
                    if cached:
                        data = json.loads(cached)
                        cached_data[symbol] = MarketData(**data)

                if len(cached_data) == len(symbols):
                    return cached_data
            else:
                # Check local cache
                current_time = datetime.now()
                cached_data = {}

                for symbol in symbols:
                    if symbol in self.local_cache:
                        cached_item = self.local_cache[symbol]
                        if (current_time - cached_item['timestamp']).seconds < self.cache_expiry:
                            cached_data[symbol] = cached_item['data']

                if len(cached_data) == len(symbols):
                    return cached_data

        except Exception as e:
            pass

        return None

    def _cache_prices(self, market_data: Dict[str, MarketData]):
        """Cache price data"""
        try:
            if self.redis_connected:
                for symbol, data in market_data.items():
                    # Convert to dict for JSON serialization
                    data_dict = {
                        'symbol': data.symbol,
                        'price': data.price,
                        'change': data.change,
                        'change_percent': data.change_percent,
                        'volume': data.volume,
                        'timestamp': data.timestamp.isoformat(),
                        'bid': data.bid,
                        'ask': data.ask,
                        'open_price': data.open_price,
                        'high': data.high,
                        'low': data.low
                    }

                    self.redis_client.setex(
                        f"price:{symbol}",
                        self.cache_expiry,
                        json.dumps(data_dict)
                    )
            else:
                # Use local cache
                current_time = datetime.now()
                for symbol, data in market_data.items():
                    self.local_cache[symbol] = {
                        'data': data,
                        'timestamp': current_time
                    }

        except Exception as e:
            pass

    def get_historical_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Get historical price data"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)

            if hist.empty:
                # Generate simulated historical data
                return self._generate_simulated_historical_data(symbol, period)

            return hist

        except Exception as e:
            # Return simulated data on error
            return self._generate_simulated_historical_data(symbol, period)

    def _generate_simulated_historical_data(self, symbol: str, period: str) -> pd.DataFrame:
        """Generate simulated historical data"""
        # Convert period to days
        period_days = {
            "1d": 1, "5d": 5, "1mo": 30, "3mo": 90,
            "6mo": 180, "1y": 252, "2y": 504, "5y": 1260
        }.get(period, 252)

        # Generate dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_days)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')

        # Base price
        base_price = 100.0

        # Generate price series with realistic patterns
        returns = np.random.normal(0.0008, 0.02, len(dates))  # Daily returns
        prices = [base_price]

        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))

        # Create OHLCV data
        df = pd.DataFrame(index=dates)
        df['Close'] = prices[:len(dates)]
        df['Open'] = df['Close'].shift(1).fillna(df['Close'].iloc[0])
        df['High'] = df[['Open', 'Close']].max(axis=1) * (1 + np.random.uniform(0, 0.02, len(df)))
        df['Low'] = df[['Open', 'Close']].min(axis=1) * (1 - np.random.uniform(0, 0.02, len(df)))
        df['Volume'] = np.random.randint(1000000, 10000000, len(df))

        return df

    def get_risk_factors(self) -> Dict[str, RiskFactorData]:
        """Get current risk factor data"""
        risk_factors = {}

        try:
            # Define key risk factors
            factors = {
                'US_10Y_YIELD': 'TNX',
                'USD_INDEX': 'DX-Y.NYB',
                'VIX': '^VIX',
                'OIL_PRICE': 'CL=F',
                'GOLD_PRICE': 'GC=F',
                'EUR_USD': 'EURUSD=X',
                'CREDIT_SPREADS': 'HYG',  # High yield bond ETF as proxy
                'EQUITY_INDEX': '^GSPC'
            }

            for factor_name, symbol in factors.items():
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="1mo")

                    if not hist.empty:
                        current_value = hist['Close'].iloc[-1]
                        change_1d = ((hist['Close'].iloc[-1] / hist['Close'].iloc[-2]) - 1) * 100
                        change_1w = ((hist['Close'].iloc[-1] / hist['Close'].iloc[-7]) - 1) * 100 if len(hist) >= 7 else change_1d
                        change_1m = ((hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1) * 100
                        volatility = hist['Close'].pct_change().std() * np.sqrt(252) * 100  # Annualized vol

                        risk_factors[factor_name] = RiskFactorData(
                            factor_name=factor_name,
                            current_value=float(current_value),
                            change_1d=float(change_1d),
                            change_1w=float(change_1w),
                            change_1m=float(change_1m),
                            volatility=float(volatility),
                            last_updated=datetime.now()
                        )
                    else:
                        # Generate simulated risk factor data
                        risk_factors[factor_name] = self._generate_simulated_risk_factor(factor_name)

                except Exception:
                    risk_factors[factor_name] = self._generate_simulated_risk_factor(factor_name)

            return risk_factors

        except Exception as e:
            # Return simulated data as fallback
            return {name: self._generate_simulated_risk_factor(name) for name in factors.keys()}

    def _generate_simulated_risk_factor(self, factor_name: str) -> RiskFactorData:
        """Generate simulated risk factor data"""
        base_values = {
            'US_10Y_YIELD': 4.5,
            'USD_INDEX': 103.0,
            'VIX': 18.0,
            'OIL_PRICE': 80.0,
            'GOLD_PRICE': 2000.0,
            'EUR_USD': 1.08,
            'CREDIT_SPREADS': 400.0,  # basis points
            'EQUITY_INDEX': 4500.0
        }

        base_value = base_values.get(factor_name, 100.0)

        return RiskFactorData(
            factor_name=factor_name,
            current_value=base_value * (1 + np.random.normal(0, 0.01)),
            change_1d=np.random.normal(0, 1.5),
            change_1w=np.random.normal(0, 3.0),
            change_1m=np.random.normal(0, 6.0),
            volatility=np.random.uniform(10, 30),
            last_updated=datetime.now()
        )

    def upload_portfolio_data(self, uploaded_file) -> pd.DataFrame:
        """Process uploaded portfolio data file"""
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file)
            else:
                raise ValueError("Unsupported file format")

            # Validate required columns
            required_columns = ['symbol', 'quantity', 'market_value', 'asset_class']
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            # Add default values for optional columns
            optional_columns = {
                'sector': 'Other',
                'region': 'Unknown',
                'currency': 'USD',
                'delta': 0.0,
                'gamma': 0.0,
                'theta': 0.0,
                'vega': 0.0,
                'rho': 0.0
            }

            for col, default_value in optional_columns.items():
                if col not in df.columns:
                    df[col] = default_value

            return df

        except Exception as e:
            st.error(f"Error processing uploaded file: {e}")
            return pd.DataFrame()

    def get_correlation_matrix(self, symbols: List[str], period: str = "1y") -> pd.DataFrame:
        """Get correlation matrix for given symbols"""
        try:
            # Fetch historical data for all symbols
            price_data = {}

            for symbol in symbols:
                hist = self.get_historical_data(symbol, period)
                if not hist.empty:
                    price_data[symbol] = hist['Close']

            if price_data:
                # Create DataFrame with all price series
                price_df = pd.DataFrame(price_data)

                # Calculate returns and correlation
                returns_df = price_df.pct_change().dropna()
                correlation_matrix = returns_df.corr()

                return correlation_matrix
            else:
                # Generate simulated correlation matrix
                return self._generate_simulated_correlation_matrix(symbols)

        except Exception as e:
            return self._generate_simulated_correlation_matrix(symbols)

    def _generate_simulated_correlation_matrix(self, symbols: List[str]) -> pd.DataFrame:
        """Generate simulated correlation matrix"""
        n = len(symbols)

        # Create base correlation matrix
        correlation_matrix = np.eye(n)

        # Add realistic correlations
        for i in range(n):
            for j in range(i+1, n):
                # Similar asset classes have higher correlation
                if any(x in symbols[i] and x in symbols[j] for x in ['AAPL', 'MSFT', 'GOOGL']):
                    corr = np.random.uniform(0.6, 0.8)  # Tech stocks
                elif 'USD' in symbols[i] and 'USD' in symbols[j]:
                    corr = np.random.uniform(0.4, 0.7)  # Currency pairs
                else:
                    corr = np.random.uniform(0.1, 0.4)  # General correlation

                correlation_matrix[i, j] = corr
                correlation_matrix[j, i] = corr

        return pd.DataFrame(correlation_matrix, index=symbols, columns=symbols)

    def get_market_regime_indicators(self) -> Dict[str, float]:
        """Get market regime indicators"""
        try:
            # Fetch VIX for volatility regime
            vix_data = self.get_historical_data('^VIX', '1mo')
            current_vix = vix_data['Close'].iloc[-1] if not vix_data.empty else 20.0

            # Fetch yield curve data
            tnx_data = self.get_historical_data('TNX', '1mo')
            current_10y = tnx_data['Close'].iloc[-1] if not tnx_data.empty else 4.5

            # Calculate regime indicators
            indicators = {
                'volatility_regime': min(current_vix / 30.0, 1.0),  # Normalized VIX
                'interest_rate_regime': min(current_10y / 6.0, 1.0),  # Normalized 10Y yield
                'risk_on_off': max(0, 1 - current_vix / 40.0),  # Risk-on indicator
                'market_stress': min(current_vix / 25.0, 1.0)  # Market stress indicator
            }

            return indicators

        except Exception as e:
            # Return default indicators
            return {
                'volatility_regime': 0.6,
                'interest_rate_regime': 0.7,
                'risk_on_off': 0.6,
                'market_stress': 0.4
            }

    def save_portfolio_snapshot(self, portfolio_name: str, metrics: Dict[str, float]):
        """Save portfolio snapshot to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO portfolio_snapshots
                (portfolio_name, total_value, var_1d, expected_shortfall)
                VALUES (?, ?, ?, ?)
            ''', (
                portfolio_name,
                metrics.get('total_value', 0),
                metrics.get('var_1d', 0),
                metrics.get('expected_shortfall', 0)
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            st.error(f"Error saving portfolio snapshot: {e}")

    def get_portfolio_history(self, portfolio_name: str, days: int = 30) -> pd.DataFrame:
        """Get portfolio historical snapshots"""
        try:
            conn = sqlite3.connect(self.db_path)

            query = '''
                SELECT timestamp, total_value, var_1d, expected_shortfall
                FROM portfolio_snapshots
                WHERE portfolio_name = ?
                AND timestamp >= datetime('now', '-{} days')
                ORDER BY timestamp
            '''.format(days)

            df = pd.read_sql_query(query, conn, params=(portfolio_name,))
            conn.close()

            if df.empty:
                # Generate simulated historical data
                return self._generate_simulated_portfolio_history(days)

            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df

        except Exception as e:
            return self._generate_simulated_portfolio_history(days)

    def _generate_simulated_portfolio_history(self, days: int) -> pd.DataFrame:
        """Generate simulated portfolio history"""
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

        # Generate realistic portfolio metrics over time
        base_value = 2500  # Million
        var_base = 15
        es_base = 22

        # Add some realistic variation
        value_series = [base_value * (1 + np.random.normal(0, 0.02)) for _ in range(days)]
        var_series = [var_base * (1 + np.random.normal(0, 0.1)) for _ in range(days)]
        es_series = [es_base * (1 + np.random.normal(0, 0.08)) for _ in range(days)]

        return pd.DataFrame({
            'timestamp': dates,
            'total_value': value_series,
            'var_1d': var_series,
            'expected_shortfall': es_series
        })

    def get_real_time_stream(self, symbols: List[str]) -> Dict[str, MarketData]:
        """Get real-time streaming data (simulated)"""
        # In a real implementation, this would connect to a real-time data feed
        return self.get_real_time_prices(symbols)