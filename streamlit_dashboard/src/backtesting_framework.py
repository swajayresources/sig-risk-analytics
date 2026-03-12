"""
Automated Backtesting Framework
===============================

Comprehensive backtesting framework with walk-forward analysis, out-of-sample testing,
performance attribution, and regulatory compliance validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
from pathlib import Path
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

from.model_validation import ValidationResult, TestResult, ModelValidationFramework

logger = logging.getLogger(__name__)

@dataclass
class BacktestConfig:
 """Backtesting configuration parameters"""
 start_date: datetime
 end_date: datetime
 lookback_window: int = 252 # Trading days
 rebalance_frequency: str = 'daily' # daily, weekly, monthly
 out_of_sample_ratio: float = 0.2 # 20% out-of-sample
 confidence_levels: List[float] = field(default_factory=lambda: [0.95, 0.99])
 rolling_window: bool = True
 include_transaction_costs: bool = True
 benchmark_return: float = 0.02 # Risk-free rate

@dataclass
class PerformanceMetrics:
 """Performance metrics container"""
 total_return: float
 annualized_return: float
 volatility: float
 sharpe_ratio: float
 max_drawdown: float
 calmar_ratio: float
 sortino_ratio: float
 information_ratio: float
 tracking_error: float
 alpha: float
 beta: float
 hit_rate: float
 average_win: float
 average_loss: float
 profit_factor: float
 var_95: float
 var_99: float
 expected_shortfall: float

@dataclass
class BacktestResult:
 """Comprehensive backtesting result"""
 model_name: str
 config: BacktestConfig
 performance_metrics: PerformanceMetrics
 daily_returns: pd.Series
 daily_pnl: pd.Series
 positions: pd.DataFrame
 drawdown_series: pd.Series
 rolling_metrics: pd.DataFrame
 validation_results: List[ValidationResult]
 out_of_sample_metrics: PerformanceMetrics
 attribution_results: Dict[str, float]
 execution_time: float
 metadata: Dict[str, Any] = field(default_factory=dict)

class BaseStrategy(ABC):
 """Base class for trading strategies"""

 def __init__(self, name: str, description: str):
 self.name = name
 self.description = description
 self.parameters: Dict[str, Any] = {}

 @abstractmethod
 def generate_signals(self, data: pd.DataFrame, lookback_window: int) -> pd.Series:
 """Generate trading signals"""
 pass

 @abstractmethod
 def calculate_positions(self, signals: pd.Series, capital: float) -> pd.Series:
 """Calculate position sizes"""
 pass

class RiskParityStrategy(BaseStrategy):
 """Risk parity strategy implementation"""

 def __init__(self, risk_target: float = 0.15):
 super().__init__("Risk Parity", "Equal risk contribution strategy")
 self.risk_target = risk_target

 def generate_signals(self, data: pd.DataFrame, lookback_window: int) -> pd.Series:
 """Generate risk parity signals"""
 returns = data.pct_change().dropna()

 if len(returns) < lookback_window:
 return pd.Series(0, index=data.index)

 # Calculate rolling volatility
 rolling_vol = returns.rolling(window=lookback_window).std()

 # Risk parity weights (inverse volatility)
 weights = 1 / rolling_vol
 weights = weights / weights.sum()

 return weights.fillna(0)

 def calculate_positions(self, signals: pd.Series, capital: float) -> pd.Series:
 """Calculate position sizes based on risk target"""
 return signals * capital

class MomentumStrategy(BaseStrategy):
 """Momentum strategy implementation"""

 def __init__(self, lookback_period: int = 21, rebalance_threshold: float = 0.05):
 super().__init__("Momentum", "Price momentum strategy")
 self.lookback_period = lookback_period
 self.rebalance_threshold = rebalance_threshold

 def generate_signals(self, data: pd.DataFrame, lookback_window: int) -> pd.Series:
 """Generate momentum signals"""
 returns = data.pct_change(self.lookback_period)

 # Generate buy/sell signals based on momentum
 signals = pd.Series(0, index=data.index)
 signals[returns > self.rebalance_threshold] = 1
 signals[returns < -self.rebalance_threshold] = -1

 return signals

 def calculate_positions(self, signals: pd.Series, capital: float) -> pd.Series:
 """Calculate position sizes"""
 return signals * capital * 0.5 # 50% allocation per signal

class BacktestEngine:
 """Core backtesting engine"""

 def __init__(self, data_provider: Any, transaction_cost: float = 0.001):
 self.data_provider = data_provider
 self.transaction_cost = transaction_cost
 self.strategies: Dict[str, BaseStrategy] = {}
 self.validation_framework = ModelValidationFramework()

 def register_strategy(self, strategy: BaseStrategy):
 """Register a trading strategy"""
 self.strategies[strategy.name] = strategy

 def run_backtest(self, strategy_name: str, config: BacktestConfig) -> BacktestResult:
 """Run comprehensive backtest"""
 start_time = datetime.now()

 try:
 strategy = self.strategies[strategy_name]

 # Get historical data
 data = self._get_historical_data(config.start_date, config.end_date)

 # Split into in-sample and out-of-sample
 split_date = self._calculate_split_date(config)
 in_sample_data = data[data.index <= split_date]
 out_of_sample_data = data[data.index > split_date]

 # Run in-sample backtest
 logger.info(f"Running in-sample backtest for {strategy_name}")
 in_sample_result = self._run_single_backtest(strategy, in_sample_data, config)

 # Run out-of-sample backtest
 logger.info(f"Running out-of-sample backtest for {strategy_name}")
 out_of_sample_result = self._run_single_backtest(strategy, out_of_sample_data, config)

 # Calculate comprehensive performance metrics
 performance_metrics = self._calculate_performance_metrics(
 in_sample_result['returns'], config.benchmark_return
 )

 out_of_sample_metrics = self._calculate_performance_metrics(
 out_of_sample_result['returns'], config.benchmark_return
 )

 # Run model validation
 validation_results = self._run_model_validation(
 in_sample_result, out_of_sample_result, config
 )

 # Performance attribution
 attribution_results = self._calculate_attribution(
 in_sample_result['returns'], in_sample_result['positions']
 )

 # Calculate rolling metrics
 rolling_metrics = self._calculate_rolling_metrics(
 in_sample_result['returns'], window=63 # Quarter
 )

 execution_time = (datetime.now() - start_time).total_seconds()

 return BacktestResult(
 model_name=strategy_name,
 config=config,
 performance_metrics=performance_metrics,
 daily_returns=in_sample_result['returns'],
 daily_pnl=in_sample_result['pnl'],
 positions=in_sample_result['positions'],
 drawdown_series=self._calculate_drawdown_series(in_sample_result['returns']),
 rolling_metrics=rolling_metrics,
 validation_results=validation_results,
 out_of_sample_metrics=out_of_sample_metrics,
 attribution_results=attribution_results,
 execution_time=execution_time,
 metadata={
 'split_date': split_date,
 'in_sample_period': (config.start_date, split_date),
 'out_of_sample_period': (split_date, config.end_date),
 'transaction_cost': self.transaction_cost
 }
 )

 except Exception as e:
 logger.error(f"Error in backtest execution: {e}")
 raise

 def _get_historical_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
 """Get historical price data"""
 # This would typically interface with your data provider
 # For demonstration, we'll generate synthetic data

 dates = pd.date_range(start=start_date, end=end_date, freq='D')
 n_days = len(dates)

 # Generate synthetic price data with realistic properties
 np.random.seed(42)
 returns = np.random.normal(0.0005, 0.015, n_days) # Daily returns

 # Add some momentum and mean reversion patterns
 for i in range(1, n_days):
 momentum = 0.1 * returns[i-1] # Momentum effect
 mean_reversion = -0.05 * np.mean(returns[max(0, i-20):i]) # Mean reversion
 returns[i] += momentum + mean_reversion

 # Convert to prices
 prices = 100 * np.exp(np.cumsum(returns))

 return pd.DataFrame({
 'price': prices,
 'volume': np.random.randint(1000000, 10000000, n_days)
 }, index=dates)

 def _calculate_split_date(self, config: BacktestConfig) -> datetime:
 """Calculate in-sample/out-of-sample split date"""
 total_days = (config.end_date - config.start_date).days
 in_sample_days = int(total_days * (1 - config.out_of_sample_ratio))
 return config.start_date + timedelta(days=in_sample_days)

 def _run_single_backtest(self, strategy: BaseStrategy, data: pd.DataFrame,
 config: BacktestConfig) -> Dict[str, pd.Series]:
 """Run backtest for a single period"""

 initial_capital = 1000000 # $1M
 capital = initial_capital
 positions = []
 returns = []
 pnl_values = []

 prices = data['price']

 for i, (date, price) in enumerate(prices.items()):
 if i < config.lookback_window:
 positions.append(0)
 returns.append(0)
 pnl_values.append(0)
 continue

 # Get historical data for signal generation
 historical_data = data.iloc[max(0, i-config.lookback_window):i]

 # Generate signals
 signals = strategy.generate_signals(historical_data, config.lookback_window)
 current_signal = signals.iloc[-1] if len(signals) > 0 else 0

 # Calculate positions
 position = strategy.calculate_positions(
 pd.Series([current_signal]), capital
 ).iloc[0] if not pd.isna(current_signal) else 0

 # Calculate returns and P&L
 if i > 0:
 price_return = (price - prices.iloc[i-1]) / prices.iloc[i-1]
 position_return = position * price_return / capital if capital != 0 else 0

 # Apply transaction costs
 if len(positions) > 0:
 position_change = abs(position - positions[-1])
 transaction_cost = position_change * self.transaction_cost
 position_return -= transaction_cost / capital

 returns.append(position_return)
 pnl_values.append(position_return * capital)
 capital *= (1 + position_return)
 else:
 returns.append(0)
 pnl_values.append(0)

 positions.append(position)

 return {
 'returns': pd.Series(returns, index=prices.index),
 'positions': pd.Series(positions, index=prices.index),
 'pnl': pd.Series(pnl_values, index=prices.index)
 }

 def _calculate_performance_metrics(self, returns: pd.Series,
 benchmark_return: float) -> PerformanceMetrics:
 """Calculate comprehensive performance metrics"""

 # Basic metrics
 total_return = (1 + returns).prod() - 1
 annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
 volatility = returns.std() * np.sqrt(252)

 # Risk-adjusted metrics
 excess_returns = returns - benchmark_return / 252
 sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252)

 # Drawdown metrics
 cumulative_returns = (1 + returns).cumprod()
 rolling_max = cumulative_returns.expanding().max()
 drawdown = (cumulative_returns - rolling_max) / rolling_max
 max_drawdown = drawdown.min()

 # Additional metrics
 calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

 # Downside deviation for Sortino ratio
 negative_returns = returns[returns < 0]
 downside_deviation = negative_returns.std() * np.sqrt(252)
 sortino_ratio = annualized_return / downside_deviation if downside_deviation != 0 else 0

 # Win/loss metrics
 positive_returns = returns[returns > 0]
 negative_returns = returns[returns < 0]

 hit_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0
 average_win = positive_returns.mean() if len(positive_returns) > 0 else 0
 average_loss = negative_returns.mean() if len(negative_returns) > 0 else 0
 profit_factor = abs(average_win * len(positive_returns) / (average_loss * len(negative_returns))) if len(negative_returns) > 0 and average_loss != 0 else 0

 # VaR and Expected Shortfall
 var_95 = np.percentile(returns, 5)
 var_99 = np.percentile(returns, 1)
 tail_returns = returns[returns <= var_95]
 expected_shortfall = tail_returns.mean() if len(tail_returns) > 0 else var_95

 # Market-related metrics (simplified)
 # In practice, these would be calculated against a market benchmark
 alpha = annualized_return - benchmark_return # Simplified alpha
 beta = 1.0 # Simplified beta
 tracking_error = volatility # Simplified tracking error
 information_ratio = alpha / tracking_error if tracking_error != 0 else 0

 return PerformanceMetrics(
 total_return=total_return,
 annualized_return=annualized_return,
 volatility=volatility,
 sharpe_ratio=sharpe_ratio,
 max_drawdown=max_drawdown,
 calmar_ratio=calmar_ratio,
 sortino_ratio=sortino_ratio,
 information_ratio=information_ratio,
 tracking_error=tracking_error,
 alpha=alpha,
 beta=beta,
 hit_rate=hit_rate,
 average_win=average_win,
 average_loss=average_loss,
 profit_factor=profit_factor,
 var_95=var_95,
 var_99=var_99,
 expected_shortfall=expected_shortfall
 )

 def _run_model_validation(self, in_sample_result: Dict, out_of_sample_result: Dict,
 config: BacktestConfig) -> List[ValidationResult]:
 """Run model validation tests"""

 validation_results = []

 # Out-of-sample performance validation
 in_sample_sharpe = self._calculate_sharpe_ratio(in_sample_result['returns'])
 out_of_sample_sharpe = self._calculate_sharpe_ratio(out_of_sample_result['returns'])

 # Test for performance degradation
 performance_degradation = (in_sample_sharpe - out_of_sample_sharpe) / in_sample_sharpe if in_sample_sharpe != 0 else 0

 degradation_test = ValidationResult(
 test_name="Out-of-Sample Performance Test",
 test_type="Model Validation",
 result=TestResult.PASS if performance_degradation < 0.5 else TestResult.FAIL,
 details={
 'in_sample_sharpe': in_sample_sharpe,
 'out_of_sample_sharpe': out_of_sample_sharpe,
 'performance_degradation': performance_degradation
 },
 recommendations=["Monitor model stability"] if performance_degradation > 0.3 else []
 )
 validation_results.append(degradation_test)

 # Statistical significance test
 in_sample_returns = in_sample_result['returns'].dropna()
 out_of_sample_returns = out_of_sample_result['returns'].dropna()

 if len(in_sample_returns) > 20 and len(out_of_sample_returns) > 20:
 from scipy.stats import ttest_ind
 t_stat, p_value = ttest_ind(in_sample_returns, out_of_sample_returns)

 significance_test = ValidationResult(
 test_name="Return Distribution Consistency Test",
 test_type="Statistical Validation",
 result=TestResult.PASS if p_value > 0.05 else TestResult.WARNING,
 p_value=p_value,
 statistic=t_stat,
 details={
 'in_sample_mean': in_sample_returns.mean(),
 'out_of_sample_mean': out_of_sample_returns.mean(),
 'sample_sizes': (len(in_sample_returns), len(out_of_sample_returns))
 }
 )
 validation_results.append(significance_test)

 # VaR backtesting (if applicable)
 for confidence_level in config.confidence_levels:
 var_threshold = np.percentile(in_sample_returns, (1 - confidence_level) * 100)
 violations = np.sum(out_of_sample_returns < var_threshold)
 expected_violations = len(out_of_sample_returns) * (1 - confidence_level)

 var_test = ValidationResult(
 test_name=f"VaR Backtesting ({confidence_level:.0%})",
 test_type="Risk Model Validation",
 result=TestResult.PASS if abs(violations - expected_violations) <= 2 else TestResult.WARNING,
 details={
 'violations': violations,
 'expected_violations': expected_violations,
 'violation_rate': violations / len(out_of_sample_returns),
 'confidence_level': confidence_level
 }
 )
 validation_results.append(var_test)

 return validation_results

 def _calculate_attribution(self, returns: pd.Series,
 positions: pd.Series) -> Dict[str, float]:
 """Calculate performance attribution"""

 # Simplified attribution analysis
 attribution = {}

 # Market timing attribution
 market_returns = returns.mean() # Simplified market return
 timing_attribution = np.corrcoef(positions[:-1], returns[1:])[0, 1] if len(positions) > 1 else 0

 attribution['market_timing'] = timing_attribution * returns.std()
 attribution['security_selection'] = returns.mean() - market_returns
 attribution['interaction_effect'] = returns.var() * 0.1 # Simplified

 return attribution

 def _calculate_rolling_metrics(self, returns: pd.Series, window: int = 63) -> pd.DataFrame:
 """Calculate rolling performance metrics"""

 rolling_metrics = pd.DataFrame(index=returns.index)

 # Rolling Sharpe ratio
 rolling_metrics['sharpe_ratio'] = (
 returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(252)
 )

 # Rolling volatility
 rolling_metrics['volatility'] = returns.rolling(window).std() * np.sqrt(252)

 # Rolling maximum drawdown
 cumulative_returns = (1 + returns).cumprod()
 rolling_max = cumulative_returns.rolling(window).max()
 rolling_drawdown = (cumulative_returns - rolling_max) / rolling_max
 rolling_metrics['max_drawdown'] = rolling_drawdown.rolling(window).min()

 # Rolling VaR
 rolling_metrics['var_95'] = returns.rolling(window).quantile(0.05)
 rolling_metrics['var_99'] = returns.rolling(window).quantile(0.01)

 return rolling_metrics

 def _calculate_drawdown_series(self, returns: pd.Series) -> pd.Series:
 """Calculate drawdown time series"""
 cumulative_returns = (1 + returns).cumprod()
 rolling_max = cumulative_returns.expanding().max()
 drawdown = (cumulative_returns - rolling_max) / rolling_max
 return drawdown

 def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
 """Calculate Sharpe ratio"""
 if returns.std() == 0:
 return 0
 return returns.mean() / returns.std() * np.sqrt(252)

class WalkForwardAnalyzer:
 """Walk-forward analysis implementation"""

 def __init__(self, backtest_engine: BacktestEngine):
 self.backtest_engine = backtest_engine

 def run_walk_forward_analysis(self, strategy_name: str,
 start_date: datetime, end_date: datetime,
 training_window: int = 252,
 test_window: int = 63,
 step_size: int = 21) -> Dict[str, Any]:
 """
 Run walk-forward analysis with rolling windows

 Args:
 strategy_name: Name of strategy to test
 start_date: Analysis start date
 end_date: Analysis end date
 training_window: Days for model training
 test_window: Days for out-of-sample testing
 step_size: Days to step forward between tests
 """

 results = []
 current_date = start_date + timedelta(days=training_window)

 while current_date + timedelta(days=test_window) <= end_date:
 # Define training and testing periods
 train_start = current_date - timedelta(days=training_window)
 train_end = current_date
 test_start = current_date
 test_end = current_date + timedelta(days=test_window)

 # Create configuration for this window
 config = BacktestConfig(
 start_date=train_start,
 end_date=test_end,
 out_of_sample_ratio=test_window / (training_window + test_window)
 )

 try:
 # Run backtest for this window
 result = self.backtest_engine.run_backtest(strategy_name, config)

 results.append({
 'train_period': (train_start, train_end),
 'test_period': (test_start, test_end),
 'in_sample_sharpe': result.performance_metrics.sharpe_ratio,
 'out_of_sample_sharpe': result.out_of_sample_metrics.sharpe_ratio,
 'out_of_sample_return': result.out_of_sample_metrics.total_return,
 'max_drawdown': result.out_of_sample_metrics.max_drawdown,
 'validation_results': result.validation_results
 })

 except Exception as e:
 logger.error(f"Error in walk-forward window {current_date}: {e}")

 current_date += timedelta(days=step_size)

 # Aggregate results
 if results:
 aggregate_metrics = self._aggregate_walk_forward_results(results)

 return {
 'individual_results': results,
 'aggregate_metrics': aggregate_metrics,
 'summary': {
 'total_windows': len(results),
 'successful_windows': len([r for r in results if r['out_of_sample_sharpe'] > 0]),
 'average_out_of_sample_sharpe': np.mean([r['out_of_sample_sharpe'] for r in results]),
 'stability_ratio': np.std([r['out_of_sample_sharpe'] for r in results]) / np.mean([r['out_of_sample_sharpe'] for r in results])
 }
 }

 return {'error': 'No valid walk-forward results'}

 def _aggregate_walk_forward_results(self, results: List[Dict]) -> Dict[str, float]:
 """Aggregate walk-forward analysis results"""

 out_of_sample_returns = [r['out_of_sample_return'] for r in results]
 out_of_sample_sharpes = [r['out_of_sample_sharpe'] for r in results]
 max_drawdowns = [r['max_drawdown'] for r in results]

 return {
 'mean_return': np.mean(out_of_sample_returns),
 'std_return': np.std(out_of_sample_returns),
 'mean_sharpe': np.mean(out_of_sample_sharpes),
 'std_sharpe': np.std(out_of_sample_sharpes),
 'mean_max_drawdown': np.mean(max_drawdowns),
 'worst_drawdown': min(max_drawdowns),
 'consistency_ratio': len([s for s in out_of_sample_sharpes if s > 0]) / len(out_of_sample_sharpes),
 'stability_score': 1 / (1 + np.std(out_of_sample_sharpes)) if np.std(out_of_sample_sharpes) > 0 else 1
 }

class PerformanceAttribution:
 """Advanced performance attribution analysis"""

 def __init__(self):
 self.attribution_factors = ['market_timing', 'security_selection', 'asset_allocation', 'currency', 'interaction']

 def brinson_attribution(self, portfolio_weights: pd.DataFrame,
 portfolio_returns: pd.DataFrame,
 benchmark_weights: pd.DataFrame,
 benchmark_returns: pd.DataFrame) -> Dict[str, pd.Series]:
 """
 Brinson-Fachler performance attribution

 Returns attribution to:
 - Asset Allocation: (wp - wb) * rb
 - Security Selection: wb * (rp - rb)
 - Interaction: (wp - wb) * (rp - rb)
 """

 # Ensure all dataframes have same index and columns
 common_index = portfolio_weights.index.intersection(benchmark_weights.index)
 common_columns = portfolio_weights.columns.intersection(benchmark_weights.columns)

 pw = portfolio_weights.loc[common_index, common_columns]
 pr = portfolio_returns.loc[common_index, common_columns]
 bw = benchmark_weights.loc[common_index, common_columns]
 br = benchmark_returns.loc[common_index, common_columns]

 # Calculate attribution components
 asset_allocation = (pw - bw) * br
 security_selection = bw * (pr - br)
 interaction = (pw - bw) * (pr - br)

 # Sum across assets for each period
 attribution_results = {
 'asset_allocation': asset_allocation.sum(axis=1),
 'security_selection': security_selection.sum(axis=1),
 'interaction': interaction.sum(axis=1),
 'total_active_return': (asset_allocation + security_selection + interaction).sum(axis=1)
 }

 return attribution_results

 def risk_adjusted_attribution(self, returns: pd.Series,
 benchmark_returns: pd.Series,
 risk_factors: pd.DataFrame) -> Dict[str, float]:
 """Risk-adjusted performance attribution using factor model"""

 try:
 from sklearn.linear_model import LinearRegression

 # Prepare data
 excess_returns = returns - benchmark_returns

 # Fit factor model
 model = LinearRegression()
 model.fit(risk_factors, excess_returns)

 # Calculate attribution
 factor_contributions = {}
 for i, factor_name in enumerate(risk_factors.columns):
 factor_exposure = model.coef_[i]
 factor_return = risk_factors.iloc[:, i].mean()
 factor_contributions[factor_name] = factor_exposure * factor_return

 # Alpha (unexplained return)
 factor_contributions['alpha'] = model.intercept_

 # R-squared
 factor_contributions['r_squared'] = model.score(risk_factors, excess_returns)

 return factor_contributions

 except ImportError:
 logger.warning("sklearn not available for risk-adjusted attribution")
 return {}
 except Exception as e:
 logger.error(f"Error in risk-adjusted attribution: {e}")
 return {}

class ModelComparison:
 """Model comparison and selection framework"""

 def __init__(self):
 self.comparison_metrics = [
 'sharpe_ratio', 'calmar_ratio', 'sortino_ratio', 'max_drawdown',
 'volatility', 'var_95', 'expected_shortfall'
 ]

 def compare_models(self, backtest_results: Dict[str, BacktestResult]) -> pd.DataFrame:
 """Compare multiple model backtest results"""

 comparison_data = {}

 for model_name, result in backtest_results.items():
 metrics = result.performance_metrics

 comparison_data[model_name] = {
 'Total Return': f"{metrics.total_return:.2%}",
 'Annualized Return': f"{metrics.annualized_return:.2%}",
 'Volatility': f"{metrics.volatility:.2%}",
 'Sharpe Ratio': f"{metrics.sharpe_ratio:.3f}",
 'Calmar Ratio': f"{metrics.calmar_ratio:.3f}",
 'Sortino Ratio': f"{metrics.sortino_ratio:.3f}",
 'Max Drawdown': f"{metrics.max_drawdown:.2%}",
 'Hit Rate': f"{metrics.hit_rate:.2%}",
 'VaR (95%)': f"{metrics.var_95:.3f}",
 'Expected Shortfall': f"{metrics.expected_shortfall:.3f}",
 'Out-of-Sample Sharpe': f"{result.out_of_sample_metrics.sharpe_ratio:.3f}"
 }

 return pd.DataFrame(comparison_data).T

 def rank_models(self, backtest_results: Dict[str, BacktestResult],
 ranking_criteria: Dict[str, float] = None) -> pd.DataFrame:
 """Rank models based on multiple criteria"""

 if ranking_criteria is None:
 ranking_criteria = {
 'sharpe_ratio': 0.3,
 'calmar_ratio': 0.2,
 'max_drawdown': 0.2, # Lower is better
 'out_of_sample_consistency': 0.3
 }

 scores = {}

 for model_name, result in backtest_results.items():
 score = 0
 metrics = result.performance_metrics
 oos_metrics = result.out_of_sample_metrics

 # Sharpe ratio (higher is better)
 if 'sharpe_ratio' in ranking_criteria:
 normalized_sharpe = min(metrics.sharpe_ratio / 2.0, 1.0) # Cap at 2.0
 score += ranking_criteria['sharpe_ratio'] * normalized_sharpe

 # Calmar ratio (higher is better)
 if 'calmar_ratio' in ranking_criteria:
 normalized_calmar = min(metrics.calmar_ratio / 3.0, 1.0) # Cap at 3.0
 score += ranking_criteria['calmar_ratio'] * normalized_calmar

 # Max drawdown (lower is better)
 if 'max_drawdown' in ranking_criteria:
 normalized_dd = max(1 + metrics.max_drawdown / 0.3, 0) # Penalty for >30% DD
 score += ranking_criteria['max_drawdown'] * normalized_dd

 # Out-of-sample consistency
 if 'out_of_sample_consistency' in ranking_criteria:
 is_oos_sharpe = metrics.sharpe_ratio
 oos_sharpe = oos_metrics.sharpe_ratio
 consistency = 1 - abs(is_oos_sharpe - oos_sharpe) / max(abs(is_oos_sharpe), 0.1)
 score += ranking_criteria['out_of_sample_consistency'] * max(consistency, 0)

 scores[model_name] = score

 # Create ranking dataframe
 ranking_df = pd.DataFrame([
 {
 'Model': model_name,
 'Score': score,
 'Rank': rank
 }
 for rank, (model_name, score) in enumerate(
 sorted(scores.items(), key=lambda x: x[1], reverse=True), 1
 )
 ])

 return ranking_df.set_index('Rank')