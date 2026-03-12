"""
Sophisticated Stress Testing Framework
Implements historical scenarios, custom shocks, and factor-based stress tests
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class StressTestType(Enum):
 HISTORICAL_SCENARIO = "historical_scenario"
 CUSTOM_SHOCK = "custom_shock"
 FACTOR_SHOCK = "factor_shock"
 MONTE_CARLO_STRESS = "monte_carlo_stress"
 TAIL_RISK = "tail_risk"
 REVERSE_STRESS = "reverse_stress"

@dataclass
class StressScenario:
 """Individual stress test scenario"""
 name: str
 description: str
 stress_type: StressTestType
 shocks: Dict[str, float] # asset/factor -> shock magnitude
 probability: Optional[float] = None
 confidence_level: float = 0.95
 horizon_days: int = 1

@dataclass
class StressTestResult:
 """Results from a stress test scenario"""
 scenario_name: str
 portfolio_pnl: float
 portfolio_pnl_percent: float
 position_pnl: Dict[str, float]
 var_breach: bool
 risk_metrics: Dict[str, float]
 stress_contributions: Dict[str, float]
 computation_time: float
 scenario_probability: Optional[float] = None

class HistoricalStressTester:
 """
 Historical scenario stress testing
 Replays major historical market events
 """

 def __init__(self):
 self.historical_scenarios = self._load_historical_scenarios()

 def _load_historical_scenarios(self) -> Dict[str, StressScenario]:
 """Load predefined historical stress scenarios"""
 scenarios = {
 "black_monday_1987": StressScenario(
 name="Black Monday 1987",
 description="October 19, 1987 stock market crash",
 stress_type=StressTestType.HISTORICAL_SCENARIO,
 shocks={
 "US_EQUITY": -0.2218,
 "INTL_EQUITY": -0.15,
 "US_BOND": 0.02,
 "VOLATILITY": 2.0
 },
 probability=0.001
 ),
 "dot_com_crash_2000": StressScenario(
 name="Dot-com Crash 2000",
 description="Technology bubble burst in March 2000",
 stress_type=StressTestType.HISTORICAL_SCENARIO,
 shocks={
 "US_EQUITY": -0.12,
 "TECH_EQUITY": -0.35,
 "INTL_EQUITY": -0.08,
 "US_BOND": 0.015,
 "VOLATILITY": 1.5
 },
 probability=0.005
 ),
 "lehman_crisis_2008": StressScenario(
 name="Lehman Crisis 2008",
 description="September 2008 financial crisis",
 stress_type=StressTestType.HISTORICAL_SCENARIO,
 shocks={
 "US_EQUITY": -0.45,
 "INTL_EQUITY": -0.42,
 "EMERGING_EQUITY": -0.55,
 "CREDIT_SPREADS": 3.0,
 "US_BOND": 0.08,
 "VOLATILITY": 3.0,
 "USD": 0.15
 },
 probability=0.002
 ),
 "covid_crash_2020": StressScenario(
 name="COVID-19 Crash 2020",
 description="March 2020 pandemic market crash",
 stress_type=StressTestType.HISTORICAL_SCENARIO,
 shocks={
 "US_EQUITY": -0.34,
 "INTL_EQUITY": -0.32,
 "EMERGING_EQUITY": -0.31,
 "OIL": -0.65,
 "US_BOND": 0.12,
 "GOLD": 0.08,
 "VOLATILITY": 4.0
 },
 probability=0.001
 ),
 "flash_crash_2010": StressScenario(
 name="Flash Crash 2010",
 description="May 6, 2010 market flash crash",
 stress_type=StressTestType.HISTORICAL_SCENARIO,
 shocks={
 "US_EQUITY": -0.09,
 "VOLATILITY": 2.5
 },
 probability=0.01,
 horizon_days=0.25 # Intraday event
 )
 }
 return scenarios

 def get_available_scenarios(self) -> List[str]:
 """Get list of available historical scenarios"""
 return list(self.historical_scenarios.keys())

 def run_historical_stress(self,
 scenario_name: str,
 portfolio_positions: Dict[str, float],
 asset_mapping: Dict[str, str]) -> StressTestResult:
 """
 Run stress test for a specific historical scenario

 Args:
 scenario_name: Name of historical scenario
 portfolio_positions: Dict of asset -> position value
 asset_mapping: Maps portfolio assets to stress test factors
 """
 import time
 start_time = time.time()

 if scenario_name not in self.historical_scenarios:
 raise ValueError(f"Unknown historical scenario: {scenario_name}")

 scenario = self.historical_scenarios[scenario_name]

 # Calculate position P&L for each asset
 position_pnl = {}
 total_pnl = 0.0

 for asset, position_value in portfolio_positions.items():
 factor = asset_mapping.get(asset, asset)
 shock = scenario.shocks.get(factor, 0.0)

 if abs(shock) > 1e-6: # Apply shock if non-zero
 asset_pnl = position_value * shock
 position_pnl[asset] = asset_pnl
 total_pnl += asset_pnl
 else:
 position_pnl[asset] = 0.0

 # Calculate portfolio-level metrics
 total_portfolio_value = sum(abs(v) for v in portfolio_positions.values())
 pnl_percent = total_pnl / total_portfolio_value if total_portfolio_value > 0 else 0.0

 # Calculate stress contributions
 stress_contributions = {}
 for factor, shock in scenario.shocks.items():
 factor_pnl = sum(
 position_pnl[asset] for asset, mapped_factor in asset_mapping.items()
 if mapped_factor == factor and asset in position_pnl
 )
 stress_contributions[factor] = factor_pnl

 # Risk metrics (simplified)
 risk_metrics = {
 "stressed_var": abs(total_pnl),
 "max_drawdown": min(0.0, total_pnl),
 "volatility_increase": scenario.shocks.get("VOLATILITY", 1.0)
 }

 return StressTestResult(
 scenario_name=scenario.name,
 portfolio_pnl=total_pnl,
 portfolio_pnl_percent=pnl_percent,
 position_pnl=position_pnl,
 var_breach=abs(total_pnl) > 0.05 * total_portfolio_value, # 5% threshold
 risk_metrics=risk_metrics,
 stress_contributions=stress_contributions,
 computation_time=time.time() - start_time,
 scenario_probability=scenario.probability
 )

class CustomStressTester:
 """
 Custom stress testing with user-defined scenarios
 """

 def create_custom_scenario(self,
 name: str,
 description: str,
 asset_shocks: Dict[str, float],
 probability: Optional[float] = None) -> StressScenario:
 """Create a custom stress scenario"""
 return StressScenario(
 name=name,
 description=description,
 stress_type=StressTestType.CUSTOM_SHOCK,
 shocks=asset_shocks,
 probability=probability
 )

 def parallel_shift_scenario(self,
 shift_magnitude: float,
 assets: List[str]) -> StressScenario:
 """Create parallel shift scenario for all assets"""
 shocks = {asset: shift_magnitude for asset in assets}
 return StressScenario(
 name=f"Parallel Shift {shift_magnitude:.1%}",
 description=f"Parallel {shift_magnitude:.1%} shift across all assets",
 stress_type=StressTestType.CUSTOM_SHOCK,
 shocks=shocks
 )

 def correlation_breakdown_scenario(self,
 normal_correlations: Dict[Tuple[str, str], float],
 stress_correlations: Dict[Tuple[str, str], float]) -> StressScenario:
 """Create scenario with correlation breakdown"""
 # Simplified correlation stress - in practice would involve full covariance matrix transformation
 correlation_impact = {}
 for (asset1, asset2), normal_corr in normal_correlations.items():
 stress_corr = stress_correlations.get((asset1, asset2), normal_corr)
 corr_change = stress_corr - normal_corr

 # Simplified impact calculation
 if abs(corr_change) > 0.1:
 correlation_impact[asset1] = correlation_impact.get(asset1, 0.0) + corr_change * 0.1
 correlation_impact[asset2] = correlation_impact.get(asset2, 0.0) + corr_change * 0.1

 return StressScenario(
 name="Correlation Breakdown",
 description="Breakdown of normal asset correlations",
 stress_type=StressTestType.CUSTOM_SHOCK,
 shocks=correlation_impact
 )

class FactorStressTester:
 """
 Factor-based stress testing
 Tests sensitivity to fundamental risk factors
 """

 def __init__(self):
 self.risk_factors = self._initialize_risk_factors()

 def _initialize_risk_factors(self) -> Dict[str, Dict[str, float]]:
 """Initialize factor loadings for different assets"""
 return {
 "EQUITY_FACTOR": {
 "US_EQUITY": 1.0,
 "INTL_EQUITY": 0.8,
 "EMERGING_EQUITY": 1.2,
 "TECH_EQUITY": 1.4
 },
 "INTEREST_RATE_FACTOR": {
 "US_BOND": -5.0, # Duration-based sensitivity
 "INTL_BOND": -4.0,
 "CREDIT": -3.0,
 "US_EQUITY": -0.5 # Negative sensitivity for equities
 },
 "CREDIT_FACTOR": {
 "CREDIT": 1.0,
 "HIGH_YIELD": 1.5,
 "US_EQUITY": 0.3,
 "EMERGING_EQUITY": 0.8
 },
 "CURRENCY_FACTOR": {
 "INTL_EQUITY": 0.5,
 "EMERGING_EQUITY": 1.0,
 "INTL_BOND": 0.8,
 "COMMODITIES": -0.3
 },
 "VOLATILITY_FACTOR": {
 "US_EQUITY": -0.1,
 "OPTIONS": 1.0,
 "STRUCTURED_PRODUCTS": 0.5
 }
 }

 def factor_shock_scenario(self,
 factor_name: str,
 shock_magnitude: float) -> StressScenario:
 """Create stress scenario for a specific risk factor"""
 if factor_name not in self.risk_factors:
 raise ValueError(f"Unknown risk factor: {factor_name}")

 factor_loadings = self.risk_factors[factor_name]
 asset_shocks = {
 asset: loading * shock_magnitude
 for asset, loading in factor_loadings.items()
 }

 return StressScenario(
 name=f"{factor_name} Shock",
 description=f"{shock_magnitude:.1%} shock to {factor_name}",
 stress_type=StressTestType.FACTOR_SHOCK,
 shocks=asset_shocks
 )

 def multi_factor_scenario(self,
 factor_shocks: Dict[str, float]) -> StressScenario:
 """Create scenario with multiple simultaneous factor shocks"""
 combined_shocks = {}

 for factor_name, shock_magnitude in factor_shocks.items():
 if factor_name in self.risk_factors:
 factor_loadings = self.risk_factors[factor_name]
 for asset, loading in factor_loadings.items():
 asset_shock = loading * shock_magnitude
 combined_shocks[asset] = combined_shocks.get(asset, 0.0) + asset_shock

 return StressScenario(
 name="Multi-Factor Stress",
 description="Combined stress across multiple risk factors",
 stress_type=StressTestType.FACTOR_SHOCK,
 shocks=combined_shocks
 )

class MonteCarloStressTester:
 """
 Monte Carlo stress testing
 Generates random stress scenarios based on statistical models
 """

 def __init__(self,
 num_simulations: int = 10000,
 confidence_levels: List[float] = [0.95, 0.99, 0.995]):
 self.num_simulations = num_simulations
 self.confidence_levels = confidence_levels

 def generate_stress_scenarios(self,
 asset_volatilities: Dict[str, float],
 asset_correlations: pd.DataFrame,
 time_horizon: int = 1) -> List[StressScenario]:
 """Generate Monte Carlo stress scenarios"""
 assets = list(asset_volatilities.keys())
 volatilities = np.array([asset_volatilities[asset] for asset in assets])

 # Generate correlated random shocks
 np.random.seed(42) # For reproducibility
 random_normals = np.random.multivariate_normal(
 mean=np.zeros(len(assets)),
 cov=asset_correlations.values,
 size=self.num_simulations
 )

 # Scale by volatilities and time horizon
 scaling_factor = np.sqrt(time_horizon / 252.0) # Assume daily volatilities
 random_shocks = random_normals * volatilities * scaling_factor

 # Generate scenarios for different confidence levels
 scenarios = []
 for confidence_level in self.confidence_levels:
 percentile = (1 - confidence_level) * 100
 tail_shocks = np.percentile(random_shocks, percentile, axis=0)

 shock_dict = {assets[i]: float(tail_shocks[i]) for i in range(len(assets))}

 scenario = StressScenario(
 name=f"Monte Carlo {confidence_level:.1%} Tail",
 description=f"Monte Carlo stress at {confidence_level:.1%} confidence level",
 stress_type=StressTestType.MONTE_CARLO_STRESS,
 shocks=shock_dict,
 confidence_level=confidence_level
 )
 scenarios.append(scenario)

 return scenarios

 def tail_risk_scenarios(self,
 return_history: pd.DataFrame,
 tail_percentiles: List[float] = [1, 5, 10]) -> List[StressScenario]:
 """Generate tail risk scenarios from historical data"""
 scenarios = []

 for percentile in tail_percentiles:
 tail_returns = return_history.quantile(percentile / 100.0)
 shock_dict = {asset: float(tail_returns[asset]) for asset in return_history.columns}

 scenario = StressScenario(
 name=f"Historical {percentile}% Tail",
 description=f"Historical {percentile}th percentile scenario",
 stress_type=StressTestType.TAIL_RISK,
 shocks=shock_dict,
 confidence_level=1 - percentile / 100.0
 )
 scenarios.append(scenario)

 return scenarios

class ReverseStressTester:
 """
 Reverse stress testing
 Finds scenarios that would cause specific losses
 """

 def find_loss_scenarios(self,
 target_loss: float,
 portfolio_positions: Dict[str, float],
 asset_constraints: Dict[str, Tuple[float, float]],
 max_iterations: int = 1000) -> List[StressScenario]:
 """
 Find stress scenarios that would result in target portfolio loss

 Args:
 target_loss: Target portfolio loss (negative number)
 portfolio_positions: Current portfolio positions
 asset_constraints: Min/max shock bounds for each asset
 max_iterations: Maximum optimization iterations
 """
 from scipy.optimize import minimize

 assets = list(portfolio_positions.keys())
 positions = np.array([portfolio_positions[asset] for asset in assets])

 def objective(shocks):
 """Minimize difference between actual and target loss"""
 portfolio_pnl = np.sum(positions * shocks)
 return (portfolio_pnl - target_loss) ** 2

 # Bounds for asset shocks
 bounds = []
 for asset in assets:
 if asset in asset_constraints:
 bounds.append(asset_constraints[asset])
 else:
 bounds.append((-0.5, 0.5)) # Default ±50% shock bounds

 # Multiple random starting points to find different scenarios
 scenarios = []
 for i in range(min(10, max_iterations // 100)):
 x0 = np.random.uniform(-0.1, 0.1, len(assets))

 try:
 result = minimize(
 objective,
 x0,
 bounds=bounds,
 method='L-BFGS-B',
 options={'maxiter': max_iterations // 10}
 )

 if result.success and abs(result.fun) < 1e-6:
 shock_dict = {assets[j]: float(result.x[j]) for j in range(len(assets))}

 scenario = StressScenario(
 name=f"Reverse Stress {i+1}",
 description=f"Scenario causing {target_loss:.0f} portfolio loss",
 stress_type=StressTestType.REVERSE_STRESS,
 shocks=shock_dict
 )
 scenarios.append(scenario)

 except Exception as e:
 logger.warning(f"Reverse stress optimization failed: {e}")
 continue

 return scenarios

class StressTestingEngine:
 """
 Unified stress testing engine
 Coordinates all stress testing capabilities
 """

 def __init__(self):
 self.historical_tester = HistoricalStressTester()
 self.custom_tester = CustomStressTester()
 self.factor_tester = FactorStressTester()
 self.monte_carlo_tester = MonteCarloStressTester()
 self.reverse_tester = ReverseStressTester()

 def run_comprehensive_stress_test(self,
 portfolio_positions: Dict[str, float],
 asset_mapping: Dict[str, str],
 custom_scenarios: Optional[List[StressScenario]] = None) -> Dict[str, StressTestResult]:
 """
 Run comprehensive stress testing battery
 """
 results = {}

 # Historical scenarios
 for scenario_name in self.historical_tester.get_available_scenarios():
 try:
 result = self.historical_tester.run_historical_stress(
 scenario_name, portfolio_positions, asset_mapping)
 results[scenario_name] = result
 except Exception as e:
 logger.error(f"Historical stress test failed for {scenario_name}: {e}")

 # Factor stress tests
 factor_shocks = {
 "equity_crash": {"EQUITY_FACTOR": -0.3},
 "rates_up": {"INTEREST_RATE_FACTOR": 0.02},
 "credit_widening": {"CREDIT_FACTOR": 0.005},
 "vol_spike": {"VOLATILITY_FACTOR": 2.0}
 }

 for test_name, shocks in factor_shocks.items():
 try:
 scenario = self.factor_tester.multi_factor_scenario(shocks)
 result = self._run_generic_stress_test(scenario, portfolio_positions, asset_mapping)
 results[test_name] = result
 except Exception as e:
 logger.error(f"Factor stress test failed for {test_name}: {e}")

 # Custom scenarios
 if custom_scenarios:
 for scenario in custom_scenarios:
 try:
 result = self._run_generic_stress_test(scenario, portfolio_positions, asset_mapping)
 results[scenario.name] = result
 except Exception as e:
 logger.error(f"Custom stress test failed for {scenario.name}: {e}")

 return results

 def run_parallel_stress_tests(self,
 scenarios: List[StressScenario],
 portfolio_positions: Dict[str, float],
 asset_mapping: Dict[str, str],
 max_workers: int = 4) -> Dict[str, StressTestResult]:
 """Run multiple stress tests in parallel"""
 results = {}

 with ThreadPoolExecutor(max_workers=max_workers) as executor:
 # Submit all stress tests
 future_to_scenario = {
 executor.submit(self._run_generic_stress_test, scenario, portfolio_positions, asset_mapping): scenario
 for scenario in scenarios
 }

 # Collect results as they complete
 for future in as_completed(future_to_scenario):
 scenario = future_to_scenario[future]
 try:
 result = future.result()
 results[scenario.name] = result
 except Exception as e:
 logger.error(f"Parallel stress test failed for {scenario.name}: {e}")

 return results

 def _run_generic_stress_test(self,
 scenario: StressScenario,
 portfolio_positions: Dict[str, float],
 asset_mapping: Dict[str, str]) -> StressTestResult:
 """Generic stress test runner for any scenario type"""
 import time
 start_time = time.time()

 position_pnl = {}
 total_pnl = 0.0

 for asset, position_value in portfolio_positions.items():
 factor = asset_mapping.get(asset, asset)
 shock = scenario.shocks.get(factor, 0.0)

 asset_pnl = position_value * shock
 position_pnl[asset] = asset_pnl
 total_pnl += asset_pnl

 total_portfolio_value = sum(abs(v) for v in portfolio_positions.values())
 pnl_percent = total_pnl / total_portfolio_value if total_portfolio_value > 0 else 0.0

 # Calculate stress contributions by factor
 stress_contributions = {}
 for factor, shock in scenario.shocks.items():
 factor_pnl = sum(
 position_pnl[asset] for asset, mapped_factor in asset_mapping.items()
 if mapped_factor == factor and asset in position_pnl
 )
 stress_contributions[factor] = factor_pnl

 risk_metrics = {
 "stressed_var": abs(total_pnl),
 "return_volatility": np.std(list(position_pnl.values())) if len(position_pnl) > 1 else 0.0,
 "max_position_loss": min(position_pnl.values()) if position_pnl else 0.0
 }

 return StressTestResult(
 scenario_name=scenario.name,
 portfolio_pnl=total_pnl,
 portfolio_pnl_percent=pnl_percent,
 position_pnl=position_pnl,
 var_breach=abs(total_pnl) > 0.05 * total_portfolio_value,
 risk_metrics=risk_metrics,
 stress_contributions=stress_contributions,
 computation_time=time.time() - start_time,
 scenario_probability=scenario.probability
 )

 def generate_stress_report(self,
 stress_results: Dict[str, StressTestResult],
 var_limit: float) -> Dict:
 """Generate comprehensive stress testing report"""
 # Sort scenarios by P&L impact
 sorted_results = sorted(
 stress_results.items(),
 key=lambda x: x[1].portfolio_pnl
 )

 worst_scenarios = sorted_results[:5] # Top 5 worst scenarios
 var_breaches = [name for name, result in stress_results.items() if result.var_breach]

 report = {
 "summary": {
 "total_scenarios_tested": len(stress_results),
 "var_breaches": len(var_breaches),
 "worst_case_loss": min(result.portfolio_pnl for result in stress_results.values()),
 "average_loss": np.mean([result.portfolio_pnl for result in stress_results.values()]),
 "var_limit": var_limit
 },
 "worst_scenarios": [
 {
 "name": name,
 "pnl": result.portfolio_pnl,
 "pnl_percent": result.portfolio_pnl_percent,
 "probability": result.scenario_probability
 }
 for name, result in worst_scenarios
 ],
 "var_breaches": var_breaches,
 "risk_factor_analysis": self._analyze_risk_factors(stress_results),
 "detailed_results": {
 name: {
 "pnl": result.portfolio_pnl,
 "pnl_percent": result.portfolio_pnl_percent,
 "risk_metrics": result.risk_metrics,
 "top_contributors": sorted(
 result.stress_contributions.items(),
 key=lambda x: abs(x[1]),
 reverse=True
 )[:3]
 }
 for name, result in stress_results.items()
 }
 }

 return report

 def _analyze_risk_factors(self, stress_results: Dict[str, StressTestResult]) -> Dict:
 """Analyze which risk factors contribute most to portfolio stress"""
 factor_contributions = {}

 for result in stress_results.values():
 for factor, contribution in result.stress_contributions.items():
 if factor not in factor_contributions:
 factor_contributions[factor] = []
 factor_contributions[factor].append(abs(contribution))

 factor_analysis = {}
 for factor, contributions in factor_contributions.items():
 factor_analysis[factor] = {
 "average_contribution": np.mean(contributions),
 "max_contribution": max(contributions),
 "frequency": len(contributions)
 }

 return factor_analysis