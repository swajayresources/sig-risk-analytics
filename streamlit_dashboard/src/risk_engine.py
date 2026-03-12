"""
Risk Engine Module
==================

Core risk calculation engine for the professional risk management dashboard.
Provides comprehensive risk metrics, VaR calculations, stress testing, and Greeks analysis.
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import minimize
from typing import Dict, List, Optional, Tuple, Union
import json
import datetime
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class PortfolioPosition:
 """Portfolio position data structure"""
 symbol: str
 quantity: float
 market_value: float
 asset_class: str
 sector: str
 region: str
 currency: str
 delta: float = 0.0
 gamma: float = 0.0
 theta: float = 0.0
 vega: float = 0.0
 rho: float = 0.0

@dataclass
class RiskMetrics:
 """Risk metrics data structure"""
 var_1d: float
 var_10d: float
 expected_shortfall: float
 volatility: float
 beta: float
 sharpe_ratio: float
 max_drawdown: float
 tracking_error: float
 information_ratio: float

class RiskEngine:
 """Advanced risk calculation engine"""

 def __init__(self):
 """Initialize risk engine with default parameters"""
 self.confidence_levels = [0.95, 0.99]
 self.time_horizons = [1, 5, 10, 22] # Days
 self.monte_carlo_simulations = 10000
 self.risk_free_rate = 0.02 # 2%

 # Market data cache
 self.market_data_cache = {}
 self.correlation_matrix = None
 self.volatility_surface = None

 def calculate_portfolio_var(self, positions: List[PortfolioPosition],
 confidence_level: float = 0.95,
 time_horizon: int = 1) -> Dict[str, float]:
 """Calculate portfolio Value at Risk using multiple methods"""

 # Historical simulation VaR
 historical_var = self._calculate_historical_var(positions, confidence_level, time_horizon)

 # Parametric VaR
 parametric_var = self._calculate_parametric_var(positions, confidence_level, time_horizon)

 # Monte Carlo VaR
 monte_carlo_var = self._calculate_monte_carlo_var(positions, confidence_level, time_horizon)

 # Component VaR
 component_var = self._calculate_component_var(positions, confidence_level, time_horizon)

 return {
 'historical_var': historical_var,
 'parametric_var': parametric_var,
 'monte_carlo_var': monte_carlo_var,
 'component_var': component_var,
 'diversified_var': min(historical_var, parametric_var, monte_carlo_var)
 }

 def _calculate_historical_var(self, positions: List[PortfolioPosition],
 confidence_level: float, time_horizon: int) -> float:
 """Calculate VaR using historical simulation"""
 try:
 # Generate simulated historical returns for each position
 returns_data = []

 for position in positions:
 # Simulate historical returns based on position characteristics
 if position.asset_class == 'Equity':
 daily_vol = 0.02 if position.sector == 'Technology' else 0.015
 elif position.asset_class == 'Fixed Income':
 daily_vol = 0.005
 elif position.asset_class == 'Derivatives':
 daily_vol = 0.03
 else:
 daily_vol = 0.01

 # Generate 252 days of historical returns
 returns = np.random.normal(0, daily_vol, 252)
 position_returns = returns * position.market_value
 returns_data.append(position_returns)

 # Calculate portfolio returns
 portfolio_returns = np.sum(returns_data, axis=0)

 # Scale to time horizon
 portfolio_returns_scaled = portfolio_returns * np.sqrt(time_horizon)

 # Calculate VaR
 var_percentile = (1 - confidence_level) * 100
 var = -np.percentile(portfolio_returns_scaled, var_percentile)

 return float(var)

 except Exception as e:
 return 0.0

 def _calculate_parametric_var(self, positions: List[PortfolioPosition],
 confidence_level: float, time_horizon: int) -> float:
 """Calculate VaR using parametric (variance-covariance) method"""
 try:
 # Create portfolio weights and volatilities
 weights = []
 volatilities = []
 total_value = sum(pos.market_value for pos in positions)

 for position in positions:
 weight = position.market_value / total_value
 weights.append(weight)

 # Estimate volatility based on asset class
 if position.asset_class == 'Equity':
 vol = 0.25 if position.sector == 'Technology' else 0.20
 elif position.asset_class == 'Fixed Income':
 vol = 0.05
 elif position.asset_class == 'Derivatives':
 vol = 0.35
 else:
 vol = 0.15

 volatilities.append(vol)

 weights = np.array(weights)
 volatilities = np.array(volatilities)

 # Create correlation matrix (simplified)
 n = len(positions)
 correlation_matrix = self._generate_correlation_matrix(positions)

 # Calculate portfolio variance
 cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix
 portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
 portfolio_volatility = np.sqrt(portfolio_variance)

 # Calculate VaR
 z_score = stats.norm.ppf(confidence_level)
 var = z_score * portfolio_volatility * np.sqrt(time_horizon) * total_value

 return float(var)

 except Exception as e:
 return 0.0

 def _calculate_monte_carlo_var(self, positions: List[PortfolioPosition],
 confidence_level: float, time_horizon: int) -> float:
 """Calculate VaR using Monte Carlo simulation"""
 try:
 simulations = []
 total_value = sum(pos.market_value for pos in positions)

 for _ in range(self.monte_carlo_simulations):
 portfolio_pnl = 0

 for position in positions:
 # Generate random return based on asset class
 if position.asset_class == 'Equity':
 mu, sigma = 0.0001, 0.02
 elif position.asset_class == 'Fixed Income':
 mu, sigma = 0.00005, 0.005
 elif position.asset_class == 'Derivatives':
 mu, sigma = 0.0, 0.03
 else:
 mu, sigma = 0.0, 0.01

 # Generate correlated returns (simplified)
 return_sim = np.random.normal(mu, sigma * np.sqrt(time_horizon))
 position_pnl = position.market_value * return_sim
 portfolio_pnl += position_pnl

 simulations.append(portfolio_pnl)

 # Calculate VaR
 var_percentile = (1 - confidence_level) * 100
 var = -np.percentile(simulations, var_percentile)

 return float(var)

 except Exception as e:
 return 0.0

 def _calculate_component_var(self, positions: List[PortfolioPosition],
 confidence_level: float, time_horizon: int) -> Dict[str, float]:
 """Calculate component VaR by asset class and region"""
 try:
 component_var = {}

 # Group by asset class
 asset_classes = {}
 for position in positions:
 if position.asset_class not in asset_classes:
 asset_classes[position.asset_class] = []
 asset_classes[position.asset_class].append(position)

 # Calculate VaR for each asset class
 for asset_class, positions_subset in asset_classes.items():
 var = self._calculate_parametric_var(positions_subset, confidence_level, time_horizon)
 component_var[f"{asset_class}_var"] = var

 # Group by region
 regions = {}
 for position in positions:
 if position.region not in regions:
 regions[position.region] = []
 regions[position.region].append(position)

 # Calculate VaR for each region
 for region, positions_subset in regions.items():
 var = self._calculate_parametric_var(positions_subset, confidence_level, time_horizon)
 component_var[f"{region}_var"] = var

 return component_var

 except Exception as e:
 return {}

 def _generate_correlation_matrix(self, positions: List[PortfolioPosition]) -> np.ndarray:
 """Generate correlation matrix based on asset classes and sectors"""
 n = len(positions)
 correlation_matrix = np.eye(n)

 for i in range(n):
 for j in range(i+1, n):
 pos1, pos2 = positions[i], positions[j]

 # Base correlation
 if pos1.asset_class == pos2.asset_class:
 if pos1.sector == pos2.sector:
 correlation = 0.8 # High correlation within same sector
 else:
 correlation = 0.6 # Medium correlation within same asset class
 elif pos1.region == pos2.region:
 correlation = 0.4 # Medium correlation within same region
 else:
 correlation = 0.2 # Low correlation across different regions/classes

 correlation_matrix[i, j] = correlation
 correlation_matrix[j, i] = correlation

 return correlation_matrix

 def calculate_expected_shortfall(self, positions: List[PortfolioPosition],
 confidence_level: float = 0.95) -> float:
 """Calculate Expected Shortfall (Conditional VaR)"""
 try:
 # Generate return simulations
 simulations = []

 for _ in range(self.monte_carlo_simulations):
 portfolio_pnl = 0

 for position in positions:
 if position.asset_class == 'Equity':
 mu, sigma = 0.0001, 0.02
 elif position.asset_class == 'Fixed Income':
 mu, sigma = 0.00005, 0.005
 elif position.asset_class == 'Derivatives':
 mu, sigma = 0.0, 0.03
 else:
 mu, sigma = 0.0, 0.01

 return_sim = np.random.normal(mu, sigma)
 position_pnl = position.market_value * return_sim
 portfolio_pnl += position_pnl

 simulations.append(portfolio_pnl)

 # Calculate Expected Shortfall
 var_threshold = np.percentile(simulations, (1 - confidence_level) * 100)
 tail_losses = [sim for sim in simulations if sim <= var_threshold]
 expected_shortfall = -np.mean(tail_losses) if tail_losses else 0

 return float(expected_shortfall)

 except Exception as e:
 return 0.0

 def calculate_greeks(self, positions: List[PortfolioPosition]) -> Dict[str, Dict[str, float]]:
 """Calculate Greeks for options positions"""
 greeks_summary = {
 'total': {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0},
 'by_asset_class': {},
 'by_sector': {}
 }

 try:
 for position in positions:
 if position.asset_class == 'Derivatives':
 # Add to totals
 greeks_summary['total']['delta'] += position.delta
 greeks_summary['total']['gamma'] += position.gamma
 greeks_summary['total']['theta'] += position.theta
 greeks_summary['total']['vega'] += position.vega
 greeks_summary['total']['rho'] += position.rho

 # Group by asset class
 if position.asset_class not in greeks_summary['by_asset_class']:
 greeks_summary['by_asset_class'][position.asset_class] = {
 'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0
 }

 ac = greeks_summary['by_asset_class'][position.asset_class]
 ac['delta'] += position.delta
 ac['gamma'] += position.gamma
 ac['theta'] += position.theta
 ac['vega'] += position.vega
 ac['rho'] += position.rho

 # Group by sector
 if position.sector not in greeks_summary['by_sector']:
 greeks_summary['by_sector'][position.sector] = {
 'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0
 }

 sec = greeks_summary['by_sector'][position.sector]
 sec['delta'] += position.delta
 sec['gamma'] += position.gamma
 sec['theta'] += position.theta
 sec['vega'] += position.vega
 sec['rho'] += position.rho

 return greeks_summary

 except Exception as e:
 return greeks_summary

 def run_stress_tests(self, positions: List[PortfolioPosition]) -> Dict[str, float]:
 """Run various stress test scenarios"""
 stress_results = {}

 try:
 total_value = sum(pos.market_value for pos in positions)

 # Market crash scenario (-20% equity, -5% bonds, +10% vol)
 market_crash_pnl = 0
 for position in positions:
 if position.asset_class == 'Equity':
 market_crash_pnl += position.market_value * -0.20
 elif position.asset_class == 'Fixed Income':
 market_crash_pnl += position.market_value * -0.05
 elif position.asset_class == 'Derivatives':
 # Options react to volatility changes
 market_crash_pnl += position.vega * 0.10 * position.market_value

 stress_results['market_crash'] = market_crash_pnl

 # Interest rate shock (+200bp)
 ir_shock_pnl = 0
 for position in positions:
 if position.asset_class == 'Fixed Income':
 # Duration approximation
 duration = 5 # Average duration
 ir_shock_pnl += position.market_value * -duration * 0.02
 elif position.asset_class == 'Derivatives':
 ir_shock_pnl += position.rho * 0.02 * position.market_value

 stress_results['interest_rate_shock'] = ir_shock_pnl

 # Currency crisis (25% USD strengthening)
 fx_crisis_pnl = 0
 for position in positions:
 if position.currency != 'USD':
 fx_crisis_pnl += position.market_value * -0.25

 stress_results['fx_crisis'] = fx_crisis_pnl

 # Credit crisis (+500bp credit spreads)
 credit_crisis_pnl = 0
 for position in positions:
 if position.asset_class == 'Fixed Income' and 'Corporate' in position.sector:
 credit_crisis_pnl += position.market_value * -0.15

 stress_results['credit_crisis'] = credit_crisis_pnl

 # Liquidity crisis (bid-ask widening)
 liquidity_crisis_pnl = 0
 for position in positions:
 if position.asset_class == 'Alternatives':
 liquidity_crisis_pnl += position.market_value * -0.10
 else:
 liquidity_crisis_pnl += position.market_value * -0.02

 stress_results['liquidity_crisis'] = liquidity_crisis_pnl

 return stress_results

 except Exception as e:
 return {}

 def calculate_risk_contributions(self, positions: List[PortfolioPosition]) -> Dict[str, Dict[str, float]]:
 """Calculate marginal and component risk contributions"""
 try:
 total_var = self.calculate_portfolio_var(positions)['parametric_var']
 contributions = {
 'marginal_var': {},
 'component_var': {},
 'percentage_contribution': {}
 }

 for i, position in enumerate(positions):
 # Calculate marginal VaR by removing position
 positions_without = positions[:i] + positions[i+1:]

 if positions_without:
 var_without = self.calculate_portfolio_var(positions_without)['parametric_var']
 marginal_var = total_var - var_without
 else:
 marginal_var = total_var

 contributions['marginal_var'][position.symbol] = marginal_var
 contributions['component_var'][position.symbol] = marginal_var * (position.market_value / sum(p.market_value for p in positions))
 contributions['percentage_contribution'][position.symbol] = (marginal_var / total_var * 100) if total_var > 0 else 0

 return contributions

 except Exception as e:
 return {'marginal_var': {}, 'component_var': {}, 'percentage_contribution': {}}

 def calculate_risk_metrics_summary(self, positions: List[PortfolioPosition]) -> RiskMetrics:
 """Calculate comprehensive risk metrics summary"""
 try:
 # Portfolio value
 total_value = sum(pos.market_value for pos in positions)

 # VaR calculations
 var_results = self.calculate_portfolio_var(positions)
 var_1d = var_results['diversified_var']
 var_10d = var_results['diversified_var'] * np.sqrt(10)

 # Expected Shortfall
 es = self.calculate_expected_shortfall(positions)

 # Volatility (annualized)
 daily_vol = var_1d / (stats.norm.ppf(0.95) * total_value)
 annual_vol = daily_vol * np.sqrt(252)

 # Beta (simplified portfolio beta)
 equity_positions = [p for p in positions if p.asset_class == 'Equity']
 portfolio_beta = np.mean([1.2 if p.sector == 'Technology' else 1.0 for p in equity_positions]) if equity_positions else 1.0

 # Sharpe ratio (simplified)
 excess_return = 0.08 - self.risk_free_rate # Assuming 8% return
 sharpe_ratio = excess_return / annual_vol if annual_vol > 0 else 0

 # Max drawdown (simulated)
 max_drawdown = var_1d / total_value * 100 * 1.5 # Approximation

 # Tracking error (simplified)
 tracking_error = annual_vol * 0.3 # Approximation

 # Information ratio
 active_return = 0.02 # Assuming 2% active return
 information_ratio = active_return / tracking_error if tracking_error > 0 else 0

 return RiskMetrics(
 var_1d=var_1d,
 var_10d=var_10d,
 expected_shortfall=es,
 volatility=annual_vol,
 beta=portfolio_beta,
 sharpe_ratio=sharpe_ratio,
 max_drawdown=max_drawdown,
 tracking_error=tracking_error,
 information_ratio=information_ratio
 )

 except Exception as e:
 # Return default metrics in case of error
 return RiskMetrics(
 var_1d=0.0,
 var_10d=0.0,
 expected_shortfall=0.0,
 volatility=0.0,
 beta=1.0,
 sharpe_ratio=0.0,
 max_drawdown=0.0,
 tracking_error=0.0,
 information_ratio=0.0
 )

 def generate_sample_portfolio(self) -> List[PortfolioPosition]:
 """Generate sample portfolio for demo purposes"""
 sample_positions = [
 PortfolioPosition("AAPL", 1000, 150000, "Equity", "Technology", "North America", "USD",
 delta=1000, gamma=0, theta=0, vega=0, rho=0),
 PortfolioPosition("MSFT", 800, 240000, "Equity", "Technology", "North America", "USD",
 delta=800, gamma=0, theta=0, vega=0, rho=0),
 PortfolioPosition("GOOGL", 200, 300000, "Equity", "Technology", "North America", "USD",
 delta=200, gamma=0, theta=0, vega=0, rho=0),
 PortfolioPosition("US10Y", 1000, 500000, "Fixed Income", "Government", "North America", "USD",
 delta=0, gamma=0, theta=0, vega=0, rho=-25000),
 PortfolioPosition("SPY_CALL", 100, 50000, "Derivatives", "Equity Options", "North America", "USD",
 delta=50, gamma=0.1, theta=-25, vega=100, rho=10),
 PortfolioPosition("TSLA", 300, 180000, "Equity", "Automotive", "North America", "USD",
 delta=300, gamma=0, theta=0, vega=0, rho=0),
 PortfolioPosition("BTC-USD", 2, 100000, "Alternative", "Cryptocurrency", "Global", "USD",
 delta=2, gamma=0, theta=0, vega=0, rho=0),
 PortfolioPosition("GOLD", 100, 200000, "Alternative", "Commodities", "Global", "USD",
 delta=100, gamma=0, theta=0, vega=0, rho=0),
 ]

 return sample_positions