"""
Sophisticated Portfolio Optimization Engine
Implements multiple optimization strategies including Mean-Variance, Risk Parity, and Black-Litterman
"""

import numpy as np
import pandas as pd
import scipy.optimize as opt
import cvxpy as cp
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import warnings
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)

class OptimizationObjective(Enum):
 MEAN_VARIANCE = "mean_variance"
 RISK_PARITY = "risk_parity"
 BLACK_LITTERMAN = "black_litterman"
 MAXIMUM_SHARPE = "maximum_sharpe"
 MINIMUM_VARIANCE = "minimum_variance"
 MAXIMUM_DIVERSIFICATION = "maximum_diversification"
 RISK_BUDGETING = "risk_budgeting"

@dataclass
class OptimizationConstraints:
 """Portfolio optimization constraints"""
 min_weights: Optional[Dict[str, float]] = None
 max_weights: Optional[Dict[str, float]] = None
 group_constraints: Optional[Dict[str, float]] = None # sector/region limits
 turnover_limit: Optional[float] = None
 target_return: Optional[float] = None
 target_risk: Optional[float] = None
 leverage_limit: Optional[float] = 1.0
 no_short_selling: bool = True

@dataclass
class OptimizationResult:
 """Portfolio optimization result"""
 weights: Dict[str, float]
 expected_return: float
 expected_risk: float
 sharpe_ratio: float
 diversification_ratio: float
 risk_contributions: Dict[str, float]
 optimization_status: str
 computation_time: float
 objective_value: float

class MeanVarianceOptimizer:
 """
 Mean-Variance Portfolio Optimization (Markowitz)
 """

 def __init__(self,
 risk_aversion: float = 1.0,
 use_shrinkage: bool = True,
 shrinkage_factor: float = 0.1):
 self.risk_aversion = risk_aversion
 self.use_shrinkage = use_shrinkage
 self.shrinkage_factor = shrinkage_factor

 def optimize(self,
 expected_returns: pd.Series,
 covariance_matrix: pd.DataFrame,
 constraints: OptimizationConstraints) -> OptimizationResult:
 """
 Solve mean-variance optimization problem
 """
 import time
 start_time = time.time()

 assets = expected_returns.index.tolist()
 n_assets = len(assets)

 # Apply covariance shrinkage if requested
 if self.use_shrinkage:
 covariance_matrix = self._apply_shrinkage(covariance_matrix)

 # Define optimization variables
 w = cp.Variable(n_assets)

 # Objective function: maximize utility = return - 0.5 * risk_aversion * risk
 portfolio_return = expected_returns.values @ w
 portfolio_risk = cp.quad_form(w, covariance_matrix.values)
 utility = portfolio_return - 0.5 * self.risk_aversion * portfolio_risk

 # Constraints
 constraints_list = [cp.sum(w) == 1] # Weights sum to 1

 if constraints.no_short_selling:
 constraints_list.append(w >= 0)

 if constraints.leverage_limit:
 constraints_list.append(cp.norm(w, 1) <= constraints.leverage_limit)

 # Individual asset weight constraints
 if constraints.min_weights:
 for i, asset in enumerate(assets):
 if asset in constraints.min_weights:
 constraints_list.append(w[i] >= constraints.min_weights[asset])

 if constraints.max_weights:
 for i, asset in enumerate(assets):
 if asset in constraints.max_weights:
 constraints_list.append(w[i] <= constraints.max_weights[asset])

 # Target return constraint
 if constraints.target_return:
 constraints_list.append(portfolio_return >= constraints.target_return)

 # Target risk constraint
 if constraints.target_risk:
 constraints_list.append(portfolio_risk <= constraints.target_risk ** 2)

 # Solve optimization problem
 problem = cp.Problem(cp.Maximize(utility), constraints_list)
 problem.solve(solver=cp.ECOS)

 if problem.status not in ["infeasible", "unbounded"]:
 weights_dict = {assets[i]: float(w.value[i]) for i in range(n_assets)}

 # Calculate portfolio metrics
 portfolio_weights = np.array(list(weights_dict.values()))
 exp_return = float(expected_returns.values @ portfolio_weights)
 exp_risk = float(np.sqrt(portfolio_weights @ covariance_matrix.values @ portfolio_weights))
 sharpe = exp_return / exp_risk if exp_risk > 0 else 0

 # Calculate risk contributions
 risk_contributions = self._calculate_risk_contributions(
 portfolio_weights, covariance_matrix.values, assets)

 # Calculate diversification ratio
 div_ratio = self._calculate_diversification_ratio(
 portfolio_weights, expected_returns.values,
 np.sqrt(np.diag(covariance_matrix.values)))

 return OptimizationResult(
 weights=weights_dict,
 expected_return=exp_return,
 expected_risk=exp_risk,
 sharpe_ratio=sharpe,
 diversification_ratio=div_ratio,
 risk_contributions=risk_contributions,
 optimization_status=problem.status,
 computation_time=time.time() - start_time,
 objective_value=float(utility.value)
 )
 else:
 raise ValueError(f"Optimization failed with status: {problem.status}")

 def efficient_frontier(self,
 expected_returns: pd.Series,
 covariance_matrix: pd.DataFrame,
 constraints: OptimizationConstraints,
 num_points: int = 50) -> List[OptimizationResult]:
 """Generate efficient frontier"""

 # Find minimum variance portfolio
 min_var_result = self._minimize_variance(expected_returns, covariance_matrix, constraints)
 min_return = min_var_result.expected_return

 # Find maximum return portfolio (ignoring risk)
 max_return = expected_returns.max()

 # Generate return targets
 return_targets = np.linspace(min_return, max_return, num_points)

 results = []
 for target_return in return_targets:
 try:
 target_constraints = constraints
 target_constraints.target_return = target_return
 result = self.optimize(expected_returns, covariance_matrix, target_constraints)
 results.append(result)
 except Exception as e:
 logger.warning(f"Failed to optimize for target return {target_return}: {e}")
 continue

 return results

 def _apply_shrinkage(self, covariance_matrix: pd.DataFrame) -> pd.DataFrame:
 """Apply Ledoit-Wolf shrinkage to covariance matrix"""
 n_assets = len(covariance_matrix)

 # Simple constant correlation model as shrinkage target
 mean_var = np.mean(np.diag(covariance_matrix))
 mean_corr = (np.sum(covariance_matrix.values) - np.trace(covariance_matrix.values)) / (n_assets * (n_assets - 1))

 shrinkage_target = np.full_like(covariance_matrix.values, mean_corr * np.sqrt(mean_var))
 np.fill_diagonal(shrinkage_target, mean_var)

 shrunk_cov = (1 - self.shrinkage_factor) * covariance_matrix.values + \
 self.shrinkage_factor * shrinkage_target

 return pd.DataFrame(shrunk_cov, index=covariance_matrix.index, columns=covariance_matrix.columns)

 def _minimize_variance(self, expected_returns, covariance_matrix, constraints):
 """Find minimum variance portfolio"""
 original_risk_aversion = self.risk_aversion
 self.risk_aversion = 1e6 # Very high risk aversion

 try:
 result = self.optimize(expected_returns, covariance_matrix, constraints)
 return result
 finally:
 self.risk_aversion = original_risk_aversion

 def _calculate_risk_contributions(self, weights, cov_matrix, assets):
 """Calculate marginal risk contributions"""
 portfolio_variance = weights @ cov_matrix @ weights
 marginal_contributions = cov_matrix @ weights
 risk_contributions = weights * marginal_contributions / portfolio_variance

 return {assets[i]: float(risk_contributions[i]) for i in range(len(assets))}

 def _calculate_diversification_ratio(self, weights, expected_returns, volatilities):
 """Calculate diversification ratio"""
 weighted_avg_vol = weights @ volatilities
 portfolio_vol = np.sqrt(weights @ np.outer(volatilities, volatilities) @ weights)
 return weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 0

class RiskParityOptimizer:
 """
 Risk Parity Portfolio Optimization
 Equal risk contribution from all assets
 """

 def __init__(self,
 risk_budget: Optional[Dict[str, float]] = None,
 max_iterations: int = 1000,
 tolerance: float = 1e-8):
 self.risk_budget = risk_budget
 self.max_iterations = max_iterations
 self.tolerance = tolerance

 def optimize(self,
 expected_returns: pd.Series,
 covariance_matrix: pd.DataFrame,
 constraints: OptimizationConstraints) -> OptimizationResult:
 """
 Solve risk parity optimization
 """
 import time
 start_time = time.time()

 assets = expected_returns.index.tolist()
 n_assets = len(assets)

 # Default equal risk budgets
 if self.risk_budget is None:
 risk_budgets = np.ones(n_assets) / n_assets
 else:
 risk_budgets = np.array([self.risk_budget.get(asset, 1/n_assets) for asset in assets])
 risk_budgets = risk_budgets / np.sum(risk_budgets) # Normalize

 # Objective function: minimize sum of squared deviations from target risk contributions
 def objective(weights):
 weights = np.maximum(weights, 1e-8) # Avoid division by zero
 portfolio_variance = weights @ covariance_matrix.values @ weights
 marginal_risk = covariance_matrix.values @ weights
 risk_contributions = weights * marginal_risk / portfolio_variance

 deviation = risk_contributions - risk_budgets
 return np.sum(deviation ** 2)

 # Constraints
 constraints_list = []

 # Weights sum to 1
 constraints_list.append({
 'type': 'eq',
 'fun': lambda w: np.sum(w) - 1
 })

 # Bounds for individual weights
 bounds = []
 for i, asset in enumerate(assets):
 min_weight = constraints.min_weights.get(asset, 0.0) if constraints.min_weights else 0.0
 max_weight = constraints.max_weights.get(asset, 1.0) if constraints.max_weights else 1.0

 if constraints.no_short_selling:
 min_weight = max(min_weight, 0.0)

 bounds.append((min_weight, max_weight))

 # Initial guess: equal weights
 x0 = np.ones(n_assets) / n_assets

 # Optimize
 result = opt.minimize(
 objective,
 x0,
 method='SLSQP',
 bounds=bounds,
 constraints=constraints_list,
 options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
 )

 if result.success:
 weights_dict = {assets[i]: float(result.x[i]) for i in range(n_assets)}

 # Calculate portfolio metrics
 portfolio_weights = result.x
 exp_return = float(expected_returns.values @ portfolio_weights)
 exp_risk = float(np.sqrt(portfolio_weights @ covariance_matrix.values @ portfolio_weights))
 sharpe = exp_return / exp_risk if exp_risk > 0 else 0

 # Calculate actual risk contributions
 portfolio_variance = portfolio_weights @ covariance_matrix.values @ portfolio_weights
 marginal_risk = covariance_matrix.values @ portfolio_weights
 risk_contributions = portfolio_weights * marginal_risk / portfolio_variance
 risk_contrib_dict = {assets[i]: float(risk_contributions[i]) for i in range(n_assets)}

 # Calculate diversification ratio
 volatilities = np.sqrt(np.diag(covariance_matrix.values))
 div_ratio = (portfolio_weights @ volatilities) / exp_risk if exp_risk > 0 else 0

 return OptimizationResult(
 weights=weights_dict,
 expected_return=exp_return,
 expected_risk=exp_risk,
 sharpe_ratio=sharpe,
 diversification_ratio=div_ratio,
 risk_contributions=risk_contrib_dict,
 optimization_status="optimal" if result.success else "failed",
 computation_time=time.time() - start_time,
 objective_value=float(result.fun)
 )
 else:
 raise ValueError(f"Risk parity optimization failed: {result.message}")

class BlackLittermanOptimizer:
 """
 Black-Litterman Portfolio Optimization
 Incorporates investor views with market equilibrium
 """

 def __init__(self,
 tau: float = 0.025,
 risk_aversion: float = 3.0):
 self.tau = tau # Scales uncertainty of prior
 self.risk_aversion = risk_aversion

 def optimize(self,
 expected_returns: pd.Series,
 covariance_matrix: pd.DataFrame,
 constraints: OptimizationConstraints,
 market_caps: Optional[pd.Series] = None,
 views_matrix: Optional[np.ndarray] = None,
 views_returns: Optional[np.ndarray] = None,
 views_uncertainty: Optional[np.ndarray] = None) -> OptimizationResult:
 """
 Black-Litterman optimization with investor views
 """
 import time
 start_time = time.time()

 assets = expected_returns.index.tolist()
 n_assets = len(assets)

 # Step 1: Calculate implied equilibrium returns (reverse optimization)
 if market_caps is None:
 # Equal market caps if not provided
 market_caps = pd.Series(np.ones(n_assets) / n_assets, index=assets)

 market_weights = market_caps / market_caps.sum()
 implied_returns = self.risk_aversion * covariance_matrix.values @ market_weights.values

 # Step 2: Incorporate investor views using Black-Litterman formula
 if views_matrix is not None and views_returns is not None:
 # Bayesian update of expected returns
 tau_cov = self.tau * covariance_matrix.values

 if views_uncertainty is None:
 # Default: views uncertainty is proportional to view variance
 views_uncertainty = np.diag(views_matrix @ tau_cov @ views_matrix.T)

 # Black-Litterman formula
 M1 = np.linalg.inv(tau_cov)
 M2 = views_matrix.T @ np.linalg.inv(np.diag(views_uncertainty)) @ views_matrix
 M3 = np.linalg.inv(tau_cov) @ implied_returns
 M4 = views_matrix.T @ np.linalg.inv(np.diag(views_uncertainty)) @ views_returns

 bl_returns = np.linalg.inv(M1 + M2) @ (M3 + M4)
 bl_cov = np.linalg.inv(M1 + M2)

 expected_returns = pd.Series(bl_returns, index=assets)
 covariance_matrix = pd.DataFrame(bl_cov, index=assets, columns=assets)

 # Step 3: Mean-variance optimization with BL inputs
 mv_optimizer = MeanVarianceOptimizer(risk_aversion=self.risk_aversion)
 result = mv_optimizer.optimize(expected_returns, covariance_matrix, constraints)

 result.computation_time = time.time() - start_time
 return result

class MaximumDiversificationOptimizer:
 """
 Maximum Diversification Portfolio
 Maximizes diversification ratio = weighted average volatility / portfolio volatility
 """

 def optimize(self,
 expected_returns: pd.Series,
 covariance_matrix: pd.DataFrame,
 constraints: OptimizationConstraints) -> OptimizationResult:
 """
 Maximize diversification ratio
 """
 import time
 start_time = time.time()

 assets = expected_returns.index.tolist()
 n_assets = len(assets)

 # Individual asset volatilities
 volatilities = np.sqrt(np.diag(covariance_matrix.values))

 # Define optimization variables
 w = cp.Variable(n_assets)

 # Objective: maximize diversification ratio
 # Equivalent to minimizing portfolio volatility for given weighted average volatility
 portfolio_variance = cp.quad_form(w, covariance_matrix.values)
 weighted_avg_vol = volatilities @ w

 # Constraints
 constraints_list = [
 cp.sum(w) == 1, # Weights sum to 1
 weighted_avg_vol == 1 # Normalize weighted average volatility
 ]

 if constraints.no_short_selling:
 constraints_list.append(w >= 0)

 # Individual weight constraints
 if constraints.min_weights:
 for i, asset in enumerate(assets):
 if asset in constraints.min_weights:
 constraints_list.append(w[i] >= constraints.min_weights[asset])

 if constraints.max_weights:
 for i, asset in enumerate(assets):
 if asset in constraints.max_weights:
 constraints_list.append(w[i] <= constraints.max_weights[asset])

 # Solve
 problem = cp.Problem(cp.Minimize(portfolio_variance), constraints_list)
 problem.solve()

 if problem.status not in ["infeasible", "unbounded"]:
 # Rescale weights to sum to 1
 weights_raw = w.value
 weights_scaled = weights_raw / np.sum(weights_raw)
 weights_dict = {assets[i]: float(weights_scaled[i]) for i in range(n_assets)}

 # Calculate metrics
 exp_return = float(expected_returns.values @ weights_scaled)
 exp_risk = float(np.sqrt(weights_scaled @ covariance_matrix.values @ weights_scaled))
 sharpe = exp_return / exp_risk if exp_risk > 0 else 0

 # Diversification ratio
 div_ratio = float((weights_scaled @ volatilities) / exp_risk) if exp_risk > 0 else 0

 # Risk contributions
 portfolio_variance = weights_scaled @ covariance_matrix.values @ weights_scaled
 marginal_risk = covariance_matrix.values @ weights_scaled
 risk_contributions = weights_scaled * marginal_risk / portfolio_variance
 risk_contrib_dict = {assets[i]: float(risk_contributions[i]) for i in range(n_assets)}

 return OptimizationResult(
 weights=weights_dict,
 expected_return=exp_return,
 expected_risk=exp_risk,
 sharpe_ratio=sharpe,
 diversification_ratio=div_ratio,
 risk_contributions=risk_contrib_dict,
 optimization_status=problem.status,
 computation_time=time.time() - start_time,
 objective_value=float(portfolio_variance.value)
 )
 else:
 raise ValueError(f"Maximum diversification optimization failed: {problem.status}")

class PortfolioOptimizationEngine:
 """
 Unified Portfolio Optimization Engine
 Supports multiple optimization objectives and advanced features
 """

 def __init__(self):
 self.optimizers = {
 OptimizationObjective.MEAN_VARIANCE: MeanVarianceOptimizer(),
 OptimizationObjective.RISK_PARITY: RiskParityOptimizer(),
 OptimizationObjective.BLACK_LITTERMAN: BlackLittermanOptimizer(),
 OptimizationObjective.MAXIMUM_DIVERSIFICATION: MaximumDiversificationOptimizer(),
 }

 def optimize_portfolio(self,
 objective: OptimizationObjective,
 expected_returns: pd.Series,
 covariance_matrix: pd.DataFrame,
 constraints: OptimizationConstraints,
 **kwargs) -> OptimizationResult:
 """
 Main optimization interface
 """
 if objective not in self.optimizers:
 raise ValueError(f"Unsupported optimization objective: {objective}")

 optimizer = self.optimizers[objective]
 return optimizer.optimize(expected_returns, covariance_matrix, constraints, **kwargs)

 def multi_objective_optimization(self,
 objectives: List[OptimizationObjective],
 expected_returns: pd.Series,
 covariance_matrix: pd.DataFrame,
 constraints: OptimizationConstraints,
 weights: Optional[List[float]] = None) -> OptimizationResult:
 """
 Multi-objective optimization combining multiple objectives
 """
 if weights is None:
 weights = [1.0 / len(objectives)] * len(objectives)

 if len(weights) != len(objectives):
 raise ValueError("Number of weights must match number of objectives")

 # Run individual optimizations
 results = []
 with ThreadPoolExecutor() as executor:
 futures = []
 for objective in objectives:
 future = executor.submit(self.optimize_portfolio, objective,
 expected_returns, covariance_matrix, constraints)
 futures.append(future)

 for future in futures:
 results.append(future.result())

 # Combine results using weighted average
 combined_weights = {}
 for asset in expected_returns.index:
 combined_weights[asset] = sum(
 w * result.weights.get(asset, 0.0)
 for w, result in zip(weights, results)
 )

 # Normalize weights
 total_weight = sum(combined_weights.values())
 combined_weights = {k: v / total_weight for k, v in combined_weights.items()}

 # Calculate combined metrics
 portfolio_weights = np.array([combined_weights[asset] for asset in expected_returns.index])
 exp_return = float(expected_returns.values @ portfolio_weights)
 exp_risk = float(np.sqrt(portfolio_weights @ covariance_matrix.values @ portfolio_weights))
 sharpe = exp_return / exp_risk if exp_risk > 0 else 0

 volatilities = np.sqrt(np.diag(covariance_matrix.values))
 div_ratio = float((portfolio_weights @ volatilities) / exp_risk) if exp_risk > 0 else 0

 return OptimizationResult(
 weights=combined_weights,
 expected_return=exp_return,
 expected_risk=exp_risk,
 sharpe_ratio=sharpe,
 diversification_ratio=div_ratio,
 risk_contributions={}, # TODO: Calculate
 optimization_status="multi_objective",
 computation_time=sum(r.computation_time for r in results),
 objective_value=0.0 # Combined objective value
 )

 def robust_optimization(self,
 objective: OptimizationObjective,
 expected_returns_scenarios: List[pd.Series],
 covariance_scenarios: List[pd.DataFrame],
 constraints: OptimizationConstraints,
 robustness_measure: str = "worst_case") -> OptimizationResult:
 """
 Robust optimization under parameter uncertainty
 """
 if robustness_measure == "worst_case":
 # Worst-case robust optimization
 return self._worst_case_optimization(
 objective, expected_returns_scenarios, covariance_scenarios, constraints)
 elif robustness_measure == "cvar":
 # Conditional Value at Risk optimization
 return self._cvar_optimization(
 objective, expected_returns_scenarios, covariance_scenarios, constraints)
 else:
 raise ValueError(f"Unsupported robustness measure: {robustness_measure}")

 def _worst_case_optimization(self, objective, return_scenarios, cov_scenarios, constraints):
 """Worst-case robust optimization implementation"""
 # This is a simplified implementation
 # In practice, this would involve solving a min-max problem

 results = []
 for returns, cov in zip(return_scenarios, cov_scenarios):
 try:
 result = self.optimize_portfolio(objective, returns, cov, constraints)
 results.append(result)
 except Exception as e:
 logger.warning(f"Failed to optimize scenario: {e}")
 continue

 if not results:
 raise ValueError("All scenario optimizations failed")

 # Return the portfolio with worst Sharpe ratio (conservative approach)
 worst_result = min(results, key=lambda r: r.sharpe_ratio)
 worst_result.optimization_status = "worst_case_robust"

 return worst_result

 def _cvar_optimization(self, objective, return_scenarios, cov_scenarios, constraints):
 """CVaR-based robust optimization"""
 # Simplified CVaR implementation
 # In practice, this would involve more sophisticated CVaR optimization

 results = []
 for returns, cov in zip(return_scenarios, cov_scenarios):
 try:
 result = self.optimize_portfolio(objective, returns, cov, constraints)
 results.append(result)
 except Exception:
 continue

 if not results:
 raise ValueError("All scenario optimizations failed")

 # Calculate average result weighted by scenario probability
 avg_weights = {}
 for asset in return_scenarios[0].index:
 avg_weights[asset] = np.mean([r.weights.get(asset, 0.0) for r in results])

 # Use first scenario for metric calculation (simplified)
 portfolio_weights = np.array([avg_weights[asset] for asset in return_scenarios[0].index])
 exp_return = float(return_scenarios[0].values @ portfolio_weights)
 exp_risk = float(np.sqrt(portfolio_weights @ cov_scenarios[0].values @ portfolio_weights))

 return OptimizationResult(
 weights=avg_weights,
 expected_return=exp_return,
 expected_risk=exp_risk,
 sharpe_ratio=exp_return / exp_risk if exp_risk > 0 else 0,
 diversification_ratio=0.0, # TODO: Calculate
 risk_contributions={},
 optimization_status="cvar_robust",
 computation_time=sum(r.computation_time for r in results),
 objective_value=0.0
 )