"""
Advanced Portfolio Optimization Algorithms
Implements sophisticated optimization techniques for institutional portfolio management
"""

import numpy as np
import pandas as pd
import scipy.optimize as optimize
import cvxpy as cp
from scipy.linalg import sqrtm, inv, cholesky
from sklearn.covariance import LedoitWolf, EmpiricalCovariance
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from numba import jit, prange

logger = logging.getLogger(__name__)

@dataclass
class OptimizationConfig:
    """Configuration for portfolio optimization"""
    max_iterations: int = 1000
    tolerance: float = 1e-8
    solver: str = 'ECOS'  # CVXPY solver
    use_parallel: bool = True
    max_workers: int = 4
    risk_aversion: float = 1.0
    transaction_cost_rate: float = 0.001
    leverage_limit: float = 1.0
    turnover_limit: Optional[float] = None

@dataclass
class OptimizationConstraints:
    """Portfolio optimization constraints"""
    min_weights: Optional[np.ndarray] = None
    max_weights: Optional[np.ndarray] = None
    group_constraints: Optional[Dict[str, Tuple[List[int], float, float]]] = None
    factor_constraints: Optional[Dict[str, Tuple[np.ndarray, float, float]]] = None
    target_return: Optional[float] = None
    target_risk: Optional[float] = None
    no_short_selling: bool = True
    integer_constraints: Optional[List[int]] = None

class BaseOptimizer(ABC):
    """Base class for portfolio optimizers"""

    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()

    @abstractmethod
    def optimize(self,
                expected_returns: np.ndarray,
                covariance_matrix: np.ndarray,
                constraints: OptimizationConstraints = None,
                **kwargs) -> Dict[str, Union[np.ndarray, float]]:
        """Optimize portfolio allocation"""
        pass

    def _validate_inputs(self,
                        expected_returns: np.ndarray,
                        covariance_matrix: np.ndarray) -> None:
        """Validate optimization inputs"""
        n_assets = len(expected_returns)

        if covariance_matrix.shape != (n_assets, n_assets):
            raise ValueError("Covariance matrix dimensions don't match expected returns")

        if not np.allclose(covariance_matrix, covariance_matrix.T):
            raise ValueError("Covariance matrix is not symmetric")

        eigenvals = np.linalg.eigvals(covariance_matrix)
        if np.any(eigenvals < -1e-8):
            raise ValueError("Covariance matrix is not positive semi-definite")

class MeanVarianceOptimizer(BaseOptimizer):
    """
    Mean-Variance Optimization with advanced features
    """

    def optimize(self,
                expected_returns: np.ndarray,
                covariance_matrix: np.ndarray,
                constraints: OptimizationConstraints = None,
                **kwargs) -> Dict[str, Union[np.ndarray, float]]:
        """
        Solve mean-variance optimization problem

        Args:
            expected_returns: Expected asset returns
            covariance_matrix: Asset return covariance matrix
            constraints: Portfolio constraints
            **kwargs: Additional optimization parameters

        Returns:
            Optimization results including weights and metrics
        """
        self._validate_inputs(expected_returns, covariance_matrix)
        constraints = constraints or OptimizationConstraints()

        n_assets = len(expected_returns)

        # Define optimization variables
        w = cp.Variable(n_assets)

        # Objective function: maximize utility = return - (risk_aversion/2) * risk
        portfolio_return = expected_returns @ w
        portfolio_risk = cp.quad_form(w, covariance_matrix)
        utility = portfolio_return - (self.config.risk_aversion / 2) * portfolio_risk

        # Constraints
        constraints_list = [cp.sum(w) == 1]  # Weights sum to 1

        # No short selling constraint
        if constraints.no_short_selling:
            constraints_list.append(w >= 0)

        # Individual weight bounds
        if constraints.min_weights is not None:
            constraints_list.append(w >= constraints.min_weights)
        if constraints.max_weights is not None:
            constraints_list.append(w <= constraints.max_weights)

        # Leverage constraint
        if self.config.leverage_limit is not None:
            constraints_list.append(cp.norm(w, 1) <= self.config.leverage_limit)

        # Target return constraint
        if constraints.target_return is not None:
            constraints_list.append(portfolio_return >= constraints.target_return)

        # Target risk constraint
        if constraints.target_risk is not None:
            constraints_list.append(cp.sqrt(portfolio_risk) <= constraints.target_risk)

        # Group constraints (sector/region limits)
        if constraints.group_constraints is not None:
            for group_name, (asset_indices, min_exposure, max_exposure) in constraints.group_constraints.items():
                group_weight = cp.sum([w[i] for i in asset_indices])
                if min_exposure is not None:
                    constraints_list.append(group_weight >= min_exposure)
                if max_exposure is not None:
                    constraints_list.append(group_weight <= max_exposure)

        # Factor constraints
        if constraints.factor_constraints is not None:
            for factor_name, (factor_loadings, min_exposure, max_exposure) in constraints.factor_constraints.items():
                factor_exposure = factor_loadings @ w
                if min_exposure is not None:
                    constraints_list.append(factor_exposure >= min_exposure)
                if max_exposure is not None:
                    constraints_list.append(factor_exposure <= max_exposure)

        # Solve optimization problem
        problem = cp.Problem(cp.Maximize(utility), constraints_list)

        try:
            problem.solve(solver=self.config.solver, verbose=False)

            if problem.status not in ["infeasible", "unbounded"]:
                optimal_weights = w.value

                # Calculate portfolio metrics
                portfolio_return_value = float(expected_returns @ optimal_weights)
                portfolio_risk_value = float(np.sqrt(optimal_weights @ covariance_matrix @ optimal_weights))
                sharpe_ratio = portfolio_return_value / portfolio_risk_value if portfolio_risk_value > 0 else 0

                # Calculate risk decomposition
                risk_contributions = self._calculate_risk_contributions(optimal_weights, covariance_matrix)

                return {
                    'weights': optimal_weights,
                    'expected_return': portfolio_return_value,
                    'expected_risk': portfolio_risk_value,
                    'sharpe_ratio': sharpe_ratio,
                    'utility': float(utility.value),
                    'risk_contributions': risk_contributions,
                    'optimization_status': problem.status,
                    'solver_time': problem.solver_stats.solve_time if problem.solver_stats else None
                }
            else:
                raise optimize.OptimizeWarning(f"Optimization failed with status: {problem.status}")

        except Exception as e:
            logger.error(f"Mean-variance optimization failed: {e}")
            # Return equal-weight portfolio as fallback
            equal_weights = np.ones(n_assets) / n_assets
            return {
                'weights': equal_weights,
                'expected_return': float(expected_returns @ equal_weights),
                'expected_risk': float(np.sqrt(equal_weights @ covariance_matrix @ equal_weights)),
                'optimization_status': 'failed',
                'error': str(e)
            }

    def efficient_frontier(self,
                          expected_returns: np.ndarray,
                          covariance_matrix: np.ndarray,
                          constraints: OptimizationConstraints = None,
                          num_points: int = 50) -> Dict[str, np.ndarray]:
        """
        Generate efficient frontier

        Args:
            expected_returns: Expected returns
            covariance_matrix: Covariance matrix
            constraints: Portfolio constraints
            num_points: Number of frontier points

        Returns:
            Efficient frontier data
        """
        constraints = constraints or OptimizationConstraints()

        # Find minimum variance portfolio
        min_var_weights = self._minimize_variance(expected_returns, covariance_matrix, constraints)
        min_return = expected_returns @ min_var_weights

        # Find maximum return portfolio
        max_return_weights = self._maximize_return(expected_returns, covariance_matrix, constraints)
        max_return = expected_returns @ max_return_weights

        # Generate target returns
        target_returns = np.linspace(min_return, max_return, num_points)

        frontier_returns = []
        frontier_risks = []
        frontier_weights = []

        for target_return in target_returns:
            try:
                target_constraints = constraints
                target_constraints.target_return = target_return

                result = self.optimize(expected_returns, covariance_matrix, target_constraints)

                if result['optimization_status'] != 'failed':
                    frontier_returns.append(result['expected_return'])
                    frontier_risks.append(result['expected_risk'])
                    frontier_weights.append(result['weights'])

            except Exception as e:
                logger.warning(f"Failed to optimize for target return {target_return}: {e}")
                continue

        return {
            'returns': np.array(frontier_returns),
            'risks': np.array(frontier_risks),
            'weights': np.array(frontier_weights),
            'sharpe_ratios': np.array(frontier_returns) / np.array(frontier_risks)
        }

    def _minimize_variance(self,
                          expected_returns: np.ndarray,
                          covariance_matrix: np.ndarray,
                          constraints: OptimizationConstraints) -> np.ndarray:
        """Find minimum variance portfolio"""
        n_assets = len(expected_returns)

        w = cp.Variable(n_assets)
        portfolio_risk = cp.quad_form(w, covariance_matrix)

        constraints_list = [cp.sum(w) == 1]

        if constraints.no_short_selling:
            constraints_list.append(w >= 0)

        if constraints.min_weights is not None:
            constraints_list.append(w >= constraints.min_weights)
        if constraints.max_weights is not None:
            constraints_list.append(w <= constraints.max_weights)

        problem = cp.Problem(cp.Minimize(portfolio_risk), constraints_list)
        problem.solve(solver=self.config.solver, verbose=False)

        return w.value if w.value is not None else np.ones(n_assets) / n_assets

    def _maximize_return(self,
                        expected_returns: np.ndarray,
                        covariance_matrix: np.ndarray,
                        constraints: OptimizationConstraints) -> np.ndarray:
        """Find maximum return portfolio"""
        n_assets = len(expected_returns)

        w = cp.Variable(n_assets)
        portfolio_return = expected_returns @ w

        constraints_list = [cp.sum(w) == 1]

        if constraints.no_short_selling:
            constraints_list.append(w >= 0)

        if constraints.min_weights is not None:
            constraints_list.append(w >= constraints.min_weights)
        if constraints.max_weights is not None:
            constraints_list.append(w <= constraints.max_weights)

        problem = cp.Problem(cp.Maximize(portfolio_return), constraints_list)
        problem.solve(solver=self.config.solver, verbose=False)

        return w.value if w.value is not None else np.ones(n_assets) / n_assets

    def _calculate_risk_contributions(self,
                                    weights: np.ndarray,
                                    covariance_matrix: np.ndarray) -> np.ndarray:
        """Calculate marginal risk contributions"""
        portfolio_variance = weights @ covariance_matrix @ weights
        marginal_contributions = covariance_matrix @ weights
        risk_contributions = weights * marginal_contributions / portfolio_variance
        return risk_contributions

class RiskParityOptimizer(BaseOptimizer):
    """
    Risk Parity Optimization
    """

    def optimize(self,
                expected_returns: np.ndarray,
                covariance_matrix: np.ndarray,
                constraints: OptimizationConstraints = None,
                risk_budgets: Optional[np.ndarray] = None,
                **kwargs) -> Dict[str, Union[np.ndarray, float]]:
        """
        Solve risk parity optimization

        Args:
            expected_returns: Expected returns (not used in pure risk parity)
            covariance_matrix: Covariance matrix
            constraints: Portfolio constraints
            risk_budgets: Target risk budgets (default: equal)
            **kwargs: Additional parameters

        Returns:
            Optimization results
        """
        self._validate_inputs(expected_returns, covariance_matrix)
        constraints = constraints or OptimizationConstraints()

        n_assets = len(expected_returns)

        # Default equal risk budgets
        if risk_budgets is None:
            risk_budgets = np.ones(n_assets) / n_assets
        else:
            risk_budgets = risk_budgets / np.sum(risk_budgets)  # Normalize

        # Use CVXPY for convex formulation
        try:
            result = self._cvxpy_risk_parity(covariance_matrix, risk_budgets, constraints)
        except:
            # Fallback to scipy optimization
            result = self._scipy_risk_parity(covariance_matrix, risk_budgets, constraints)

        if result is not None:
            optimal_weights = result

            # Calculate portfolio metrics
            portfolio_return_value = float(expected_returns @ optimal_weights)
            portfolio_risk_value = float(np.sqrt(optimal_weights @ covariance_matrix @ optimal_weights))
            sharpe_ratio = portfolio_return_value / portfolio_risk_value if portfolio_risk_value > 0 else 0

            # Calculate actual risk contributions
            risk_contributions = self._calculate_risk_contributions(optimal_weights, covariance_matrix)

            return {
                'weights': optimal_weights,
                'expected_return': portfolio_return_value,
                'expected_risk': portfolio_risk_value,
                'sharpe_ratio': sharpe_ratio,
                'risk_contributions': risk_contributions,
                'target_risk_budgets': risk_budgets,
                'risk_budget_deviation': np.sum(np.abs(risk_contributions - risk_budgets)),
                'optimization_status': 'optimal'
            }
        else:
            # Return equal-weight fallback
            equal_weights = np.ones(n_assets) / n_assets
            return {
                'weights': equal_weights,
                'expected_return': float(expected_returns @ equal_weights),
                'expected_risk': float(np.sqrt(equal_weights @ covariance_matrix @ equal_weights)),
                'optimization_status': 'failed'
            }

    def _cvxpy_risk_parity(self,
                          covariance_matrix: np.ndarray,
                          risk_budgets: np.ndarray,
                          constraints: OptimizationConstraints) -> Optional[np.ndarray]:
        """Risk parity optimization using CVXPY"""
        n_assets = len(risk_budgets)

        # Use logarithmic formulation for better numerical stability
        x = cp.Variable(n_assets)  # Log weights
        w = cp.exp(x)

        # Constraint: sum of weights = 1
        constraints_list = [cp.sum(w) == 1]

        # Risk parity objective using approximation
        # Minimize sum of squared deviations from target risk contributions
        portfolio_variance = cp.quad_form(w, covariance_matrix)

        # This is an approximation - true risk parity is non-convex
        objective = 0
        for i in range(n_assets):
            marginal_risk = cp.sum(cp.multiply(covariance_matrix[i, :], w))
            risk_contrib = w[i] * marginal_risk / portfolio_variance
            objective += cp.square(risk_contrib - risk_budgets[i])

        # Weight bounds
        if constraints.no_short_selling:
            constraints_list.append(w >= 1e-6)  # Small positive bound

        if constraints.min_weights is not None:
            constraints_list.append(w >= constraints.min_weights)
        if constraints.max_weights is not None:
            constraints_list.append(w <= constraints.max_weights)

        problem = cp.Problem(cp.Minimize(objective), constraints_list)

        try:
            problem.solve(solver='SCS', verbose=False)
            if problem.status in ['optimal', 'optimal_inaccurate']:
                return w.value
        except:
            pass

        return None

    def _scipy_risk_parity(self,
                          covariance_matrix: np.ndarray,
                          risk_budgets: np.ndarray,
                          constraints: OptimizationConstraints) -> Optional[np.ndarray]:
        """Risk parity optimization using scipy"""
        n_assets = len(risk_budgets)

        def objective(weights):
            """Risk parity objective function"""
            weights = np.maximum(weights, 1e-8)  # Ensure positive weights
            portfolio_variance = weights @ covariance_matrix @ weights
            marginal_risk = covariance_matrix @ weights
            risk_contributions = weights * marginal_risk / portfolio_variance
            return np.sum((risk_contributions - risk_budgets) ** 2)

        def jacobian(weights):
            """Analytical jacobian for faster convergence"""
            weights = np.maximum(weights, 1e-8)
            portfolio_variance = weights @ covariance_matrix @ weights
            marginal_risk = covariance_matrix @ weights
            risk_contributions = weights * marginal_risk / portfolio_variance

            # Compute partial derivatives
            grad = np.zeros(n_assets)
            for i in range(n_assets):
                # Derivative of risk contribution w.r.t. weight i
                dRC_dw = (marginal_risk[i] + weights[i] * covariance_matrix[i, i]) / portfolio_variance - \
                        weights[i] * marginal_risk[i] * 2 * marginal_risk[i] / (portfolio_variance ** 2)
                grad[i] = 2 * (risk_contributions[i] - risk_budgets[i]) * dRC_dw

            return grad

        # Constraints
        constraints_list = []

        # Weights sum to 1
        constraints_list.append({
            'type': 'eq',
            'fun': lambda w: np.sum(w) - 1,
            'jac': lambda w: np.ones(n_assets)
        })

        # Bounds
        bounds = []
        for i in range(n_assets):
            min_weight = 1e-6 if constraints.no_short_selling else -1.0
            max_weight = 1.0

            if constraints.min_weights is not None:
                min_weight = max(min_weight, constraints.min_weights[i])
            if constraints.max_weights is not None:
                max_weight = min(max_weight, constraints.max_weights[i])

            bounds.append((min_weight, max_weight))

        # Initial guess: equal weights
        x0 = np.ones(n_assets) / n_assets

        # Optimize
        try:
            result = optimize.minimize(
                objective,
                x0,
                method='SLSQP',
                jac=jacobian,
                bounds=bounds,
                constraints=constraints_list,
                options={'maxiter': self.config.max_iterations, 'ftol': self.config.tolerance}
            )

            if result.success:
                return result.x
        except Exception as e:
            logger.warning(f"Scipy risk parity optimization failed: {e}")

        return None

    def _calculate_risk_contributions(self,
                                    weights: np.ndarray,
                                    covariance_matrix: np.ndarray) -> np.ndarray:
        """Calculate risk contributions"""
        portfolio_variance = weights @ covariance_matrix @ weights
        marginal_risk = covariance_matrix @ weights
        risk_contributions = weights * marginal_risk / portfolio_variance
        return risk_contributions

class BlackLittermanOptimizer(BaseOptimizer):
    """
    Black-Litterman Portfolio Optimization
    """

    def __init__(self, config: OptimizationConfig = None, tau: float = 0.025):
        super().__init__(config)
        self.tau = tau  # Scales uncertainty of prior

    def optimize(self,
                expected_returns: np.ndarray,
                covariance_matrix: np.ndarray,
                constraints: OptimizationConstraints = None,
                market_caps: Optional[np.ndarray] = None,
                views_matrix: Optional[np.ndarray] = None,
                views_returns: Optional[np.ndarray] = None,
                views_uncertainty: Optional[np.ndarray] = None,
                **kwargs) -> Dict[str, Union[np.ndarray, float]]:
        """
        Black-Litterman optimization with investor views

        Args:
            expected_returns: Prior expected returns
            covariance_matrix: Prior covariance matrix
            constraints: Portfolio constraints
            market_caps: Market capitalization weights
            views_matrix: Views matrix (P)
            views_returns: Views on expected returns (Q)
            views_uncertainty: Uncertainty matrix for views (Ω)
            **kwargs: Additional parameters

        Returns:
            Optimization results
        """
        self._validate_inputs(expected_returns, covariance_matrix)
        constraints = constraints or OptimizationConstraints()

        n_assets = len(expected_returns)

        # Step 1: Calculate implied equilibrium returns (reverse optimization)
        if market_caps is None:
            market_caps = np.ones(n_assets) / n_assets  # Equal weights if not provided

        market_weights = market_caps / np.sum(market_caps)
        implied_returns = self.config.risk_aversion * covariance_matrix @ market_weights

        # Step 2: Incorporate investor views using Black-Litterman formula
        bl_returns = implied_returns.copy()
        bl_covariance = covariance_matrix.copy()

        if views_matrix is not None and views_returns is not None:
            # Prepare matrices
            P = views_matrix  # Views matrix
            Q = views_returns  # Views returns
            tau_Sigma = self.tau * covariance_matrix

            # Views uncertainty matrix
            if views_uncertainty is None:
                # Default: proportional to view variance
                Omega = np.diag(np.diag(P @ tau_Sigma @ P.T))
            else:
                Omega = views_uncertainty

            try:
                # Black-Litterman formula
                M1 = inv(tau_Sigma)
                M2 = P.T @ inv(Omega) @ P
                M3 = inv(tau_Sigma) @ implied_returns
                M4 = P.T @ inv(Omega) @ Q

                bl_returns = inv(M1 + M2) @ (M3 + M4)
                bl_covariance = inv(M1 + M2)

            except np.linalg.LinAlgError as e:
                logger.warning(f"Black-Litterman matrix inversion failed: {e}")
                # Fall back to prior
                bl_returns = implied_returns
                bl_covariance = covariance_matrix

        # Step 3: Mean-variance optimization with BL inputs
        mv_optimizer = MeanVarianceOptimizer(self.config)
        result = mv_optimizer.optimize(bl_returns, bl_covariance, constraints)

        # Add Black-Litterman specific metrics
        result.update({
            'implied_returns': implied_returns,
            'bl_returns': bl_returns,
            'bl_covariance': bl_covariance,
            'tau': self.tau,
            'views_incorporated': views_matrix is not None and views_returns is not None
        })

        return result

    def create_views_matrix(self,
                           asset_names: List[str],
                           views: List[Dict[str, Union[str, float]]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create views matrix and returns vector from view specifications

        Args:
            asset_names: List of asset names
            views: List of view dictionaries with format:
                   {'type': 'absolute'/'relative', 'asset': 'AAPL', 'return': 0.15}
                   or {'type': 'relative', 'asset1': 'AAPL', 'asset2': 'GOOGL', 'return': 0.05}

        Returns:
            Tuple of (views_matrix, views_returns)
        """
        n_assets = len(asset_names)
        n_views = len(views)

        P = np.zeros((n_views, n_assets))  # Views matrix
        Q = np.zeros(n_views)  # Views returns

        asset_to_index = {name: i for i, name in enumerate(asset_names)}

        for i, view in enumerate(views):
            if view['type'] == 'absolute':
                # Absolute view on single asset
                asset_idx = asset_to_index[view['asset']]
                P[i, asset_idx] = 1.0
                Q[i] = view['return']

            elif view['type'] == 'relative':
                # Relative view between two assets
                asset1_idx = asset_to_index[view['asset1']]
                asset2_idx = asset_to_index[view['asset2']]
                P[i, asset1_idx] = 1.0
                P[i, asset2_idx] = -1.0
                Q[i] = view['return']

        return P, Q

class HierarchicalRiskParityOptimizer(BaseOptimizer):
    """
    Hierarchical Risk Parity (HRP) Optimization
    Based on machine learning clustering techniques
    """

    def optimize(self,
                expected_returns: np.ndarray,
                covariance_matrix: np.ndarray,
                constraints: OptimizationConstraints = None,
                **kwargs) -> Dict[str, Union[np.ndarray, float]]:
        """
        Hierarchical Risk Parity optimization

        Args:
            expected_returns: Expected returns
            covariance_matrix: Covariance matrix
            constraints: Portfolio constraints (limited support)
            **kwargs: Additional parameters

        Returns:
            Optimization results
        """
        from scipy.cluster.hierarchy import linkage, dendrogram, cut_tree
        from scipy.spatial.distance import squareform

        self._validate_inputs(expected_returns, covariance_matrix)

        n_assets = len(expected_returns)

        # Step 1: Calculate distance matrix from correlation matrix
        correlation_matrix = self._cov_to_corr(covariance_matrix)
        distance_matrix = np.sqrt(0.5 * (1 - correlation_matrix))

        # Step 2: Hierarchical clustering
        condensed_distances = squareform(distance_matrix, checks=False)
        linkage_matrix = linkage(condensed_distances, method='ward')

        # Step 3: Quasi-diagonalization (sort assets by cluster)
        sorted_indices = self._get_quasi_diagonal_order(linkage_matrix, n_assets)

        # Step 4: Recursive bisection for weight allocation
        weights = np.zeros(n_assets)
        self._recursive_bisection(
            sorted_indices, covariance_matrix, weights, allocation=1.0
        )

        # Reorder weights to original asset order
        hrp_weights = np.zeros(n_assets)
        for i, original_idx in enumerate(sorted_indices):
            hrp_weights[original_idx] = weights[i]

        # Calculate portfolio metrics
        portfolio_return = float(expected_returns @ hrp_weights)
        portfolio_risk = float(np.sqrt(hrp_weights @ covariance_matrix @ hrp_weights))
        sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0

        return {
            'weights': hrp_weights,
            'expected_return': portfolio_return,
            'expected_risk': portfolio_risk,
            'sharpe_ratio': sharpe_ratio,
            'sorted_indices': sorted_indices,
            'linkage_matrix': linkage_matrix,
            'optimization_status': 'optimal'
        }

    def _cov_to_corr(self, covariance_matrix: np.ndarray) -> np.ndarray:
        """Convert covariance matrix to correlation matrix"""
        std_devs = np.sqrt(np.diag(covariance_matrix))
        correlation_matrix = covariance_matrix / np.outer(std_devs, std_devs)
        return correlation_matrix

    def _get_quasi_diagonal_order(self, linkage_matrix: np.ndarray, n_assets: int) -> List[int]:
        """Get quasi-diagonal ordering of assets"""
        from scipy.cluster.hierarchy import dendrogram

        # Create dendrogram to get leaf ordering
        dendro = dendrogram(linkage_matrix, no_plot=True)
        sorted_indices = dendro['leaves']

        return sorted_indices

    def _recursive_bisection(self,
                           indices: List[int],
                           covariance_matrix: np.ndarray,
                           weights: np.ndarray,
                           allocation: float) -> None:
        """Recursively allocate weights using inverse variance weighting"""
        if len(indices) == 1:
            weights[indices[0]] = allocation
            return

        # Split cluster into two subclusters
        n_assets = len(indices)
        mid_point = n_assets // 2

        left_indices = indices[:mid_point]
        right_indices = indices[mid_point:]

        # Calculate cluster variances
        left_cov = covariance_matrix[np.ix_(left_indices, left_indices)]
        right_cov = covariance_matrix[np.ix_(right_indices, right_indices)]

        left_var = self._cluster_variance(left_cov)
        right_var = self._cluster_variance(right_cov)

        # Allocate based on inverse variance
        total_inv_var = 1/left_var + 1/right_var
        left_allocation = (1/left_var) / total_inv_var * allocation
        right_allocation = (1/right_var) / total_inv_var * allocation

        # Recursive calls
        self._recursive_bisection(left_indices, covariance_matrix, weights, left_allocation)
        self._recursive_bisection(right_indices, covariance_matrix, weights, right_allocation)

    def _cluster_variance(self, cluster_covariance: np.ndarray) -> float:
        """Calculate cluster variance using inverse variance weighting"""
        n_assets = cluster_covariance.shape[0]
        if n_assets == 1:
            return cluster_covariance[0, 0]

        # Use equal weights for cluster variance calculation
        equal_weights = np.ones(n_assets) / n_assets
        cluster_var = equal_weights @ cluster_covariance @ equal_weights

        return cluster_var

class CriticalLineAlgorithm:
    """
    Critical Line Algorithm for exact efficient frontier computation
    """

    def __init__(self, expected_returns: np.ndarray, covariance_matrix: np.ndarray):
        self.expected_returns = expected_returns
        self.covariance_matrix = covariance_matrix
        self.n_assets = len(expected_returns)

    def solve_efficient_frontier(self,
                                min_weights: Optional[np.ndarray] = None,
                                max_weights: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Solve for exact efficient frontier using Critical Line Algorithm

        Args:
            min_weights: Minimum weight constraints
            max_weights: Maximum weight constraints

        Returns:
            Efficient frontier points
        """
        # Set default bounds
        if min_weights is None:
            min_weights = np.zeros(self.n_assets)
        if max_weights is None:
            max_weights = np.ones(self.n_assets)

        # Initialize
        critical_points = []
        lambdas = []

        # Start with minimum variance portfolio
        current_lambda = 0.0
        active_set = set()
        inactive_set = set(range(self.n_assets))

        while len(critical_points) < 100:  # Maximum iterations
            # Solve for current critical point
            weights, portfolio_return, portfolio_risk = self._solve_constrained_problem(
                current_lambda, active_set, min_weights, max_weights
            )

            if weights is None:
                break

            critical_points.append({
                'weights': weights,
                'return': portfolio_return,
                'risk': portfolio_risk,
                'lambda': current_lambda
            })

            # Find next lambda (turning point)
            next_lambda, entering_asset, leaving_asset = self._find_next_turning_point(
                weights, active_set, min_weights, max_weights, current_lambda
            )

            if next_lambda is None or next_lambda >= 1e10:
                break

            # Update active set
            if entering_asset is not None:
                active_set.add(entering_asset)
                inactive_set.discard(entering_asset)

            if leaving_asset is not None:
                active_set.discard(leaving_asset)
                inactive_set.add(leaving_asset)

            current_lambda = next_lambda
            lambdas.append(current_lambda)

        # Convert to arrays
        frontier_weights = np.array([cp['weights'] for cp in critical_points])
        frontier_returns = np.array([cp['return'] for cp in critical_points])
        frontier_risks = np.array([cp['risk'] for cp in critical_points])

        return {
            'weights': frontier_weights,
            'returns': frontier_returns,
            'risks': frontier_risks,
            'lambdas': np.array(lambdas),
            'critical_points': critical_points
        }

    def _solve_constrained_problem(self,
                                  lambda_val: float,
                                  active_set: set,
                                  min_weights: np.ndarray,
                                  max_weights: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Solve constrained optimization problem for given lambda"""
        # This is a simplified implementation
        # Full CLA would involve more sophisticated constraint handling

        try:
            # Use CVXPY for constrained solution
            w = cp.Variable(self.n_assets)

            # Objective: minimize 0.5 * w'Σw - λ * μ'w
            objective = 0.5 * cp.quad_form(w, self.covariance_matrix) - lambda_val * (self.expected_returns @ w)

            constraints = [
                cp.sum(w) == 1,
                w >= min_weights,
                w <= max_weights
            ]

            problem = cp.Problem(cp.Minimize(objective), constraints)
            problem.solve(solver='ECOS', verbose=False)

            if problem.status == 'optimal':
                weights = w.value
                portfolio_return = float(self.expected_returns @ weights)
                portfolio_risk = float(np.sqrt(weights @ self.covariance_matrix @ weights))
                return weights, portfolio_return, portfolio_risk

        except Exception as e:
            logger.warning(f"Constrained problem solving failed: {e}")

        return None, None, None

    def _find_next_turning_point(self,
                               current_weights: np.ndarray,
                               active_set: set,
                               min_weights: np.ndarray,
                               max_weights: np.ndarray,
                               current_lambda: float) -> Tuple[float, int, int]:
        """Find next turning point in efficient frontier"""
        # Simplified turning point calculation
        # Full implementation would involve solving for exact constraint changes

        candidates = []

        # Check for assets hitting bounds
        for i in range(self.n_assets):
            if i not in active_set:
                # Asset could enter active set
                if current_weights[i] <= min_weights[i] + 1e-8:
                    candidates.append((current_lambda + 0.1, i, None))
                elif current_weights[i] >= max_weights[i] - 1e-8:
                    candidates.append((current_lambda + 0.1, i, None))

        if candidates:
            next_lambda, entering, leaving = min(candidates)
            return next_lambda, entering, leaving

        return None, None, None

# Example integration and usage
if __name__ == "__main__":
    # Example usage of advanced portfolio optimization
    print("=== Advanced Portfolio Optimization Demo ===\n")

    np.random.seed(42)

    # Generate sample data
    n_assets = 8
    asset_names = [f'Asset_{i+1}' for i in range(n_assets)]

    # Generate expected returns
    expected_returns = np.random.uniform(0.05, 0.20, n_assets)

    # Generate covariance matrix
    volatilities = np.random.uniform(0.10, 0.30, n_assets)
    correlation_matrix = np.random.uniform(0.2, 0.8, (n_assets, n_assets))
    correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
    np.fill_diagonal(correlation_matrix, 1.0)

    # Ensure positive definite
    eigenvals, eigenvecs = np.linalg.eigh(correlation_matrix)
    eigenvals = np.maximum(eigenvals, 0.01)
    correlation_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

    covariance_matrix = np.outer(volatilities, volatilities) * correlation_matrix

    # Configuration
    config = OptimizationConfig(
        risk_aversion=3.0,
        max_iterations=1000,
        tolerance=1e-8
    )

    # Constraints
    constraints = OptimizationConstraints(
        min_weights=np.full(n_assets, 0.02),  # Minimum 2% allocation
        max_weights=np.full(n_assets, 0.40),  # Maximum 40% allocation
        no_short_selling=True
    )

    print("1. Mean-Variance Optimization")
    mv_optimizer = MeanVarianceOptimizer(config)
    mv_result = mv_optimizer.optimize(expected_returns, covariance_matrix, constraints)

    print(f"   Expected Return: {mv_result['expected_return']:.4f}")
    print(f"   Expected Risk: {mv_result['expected_risk']:.4f}")
    print(f"   Sharpe Ratio: {mv_result['sharpe_ratio']:.4f}")
    print(f"   Top 3 holdings: {sorted(enumerate(mv_result['weights']), key=lambda x: x[1], reverse=True)[:3]}\n")

    print("2. Risk Parity Optimization")
    rp_optimizer = RiskParityOptimizer(config)
    rp_result = rp_optimizer.optimize(expected_returns, covariance_matrix, constraints)

    print(f"   Expected Return: {rp_result['expected_return']:.4f}")
    print(f"   Expected Risk: {rp_result['expected_risk']:.4f}")
    print(f"   Sharpe Ratio: {rp_result['sharpe_ratio']:.4f}")
    print(f"   Risk Budget Deviation: {rp_result['risk_budget_deviation']:.6f}")
    print(f"   Risk Contributions (first 4): {rp_result['risk_contributions'][:4]}\n")

    print("3. Black-Litterman Optimization")
    bl_optimizer = BlackLittermanOptimizer(config, tau=0.025)

    # Create sample views
    views = [
        {'type': 'absolute', 'asset': 'Asset_1', 'return': 0.25},  # Bullish on Asset 1
        {'type': 'relative', 'asset1': 'Asset_2', 'asset2': 'Asset_3', 'return': 0.05}  # Asset 2 outperforms Asset 3
    ]

    views_matrix, views_returns = bl_optimizer.create_views_matrix(asset_names, views)
    market_caps = np.random.uniform(0.5, 2.0, n_assets)  # Sample market caps

    bl_result = bl_optimizer.optimize(
        expected_returns, covariance_matrix, constraints,
        market_caps=market_caps, views_matrix=views_matrix, views_returns=views_returns
    )

    print(f"   Expected Return: {bl_result['expected_return']:.4f}")
    print(f"   Expected Risk: {bl_result['expected_risk']:.4f}")
    print(f"   Sharpe Ratio: {bl_result['sharpe_ratio']:.4f}")
    print(f"   Views Incorporated: {bl_result['views_incorporated']}")
    print(f"   Top 3 holdings: {sorted(enumerate(bl_result['weights']), key=lambda x: x[1], reverse=True)[:3]}\n")

    print("4. Hierarchical Risk Parity")
    hrp_optimizer = HierarchicalRiskParityOptimizer(config)
    hrp_result = hrp_optimizer.optimize(expected_returns, covariance_matrix)

    print(f"   Expected Return: {hrp_result['expected_return']:.4f}")
    print(f"   Expected Risk: {hrp_result['expected_risk']:.4f}")
    print(f"   Sharpe Ratio: {hrp_result['sharpe_ratio']:.4f}")
    print(f"   Asset Ordering: {hrp_result['sorted_indices']}\n")

    print("5. Efficient Frontier Generation")
    frontier_data = mv_optimizer.efficient_frontier(
        expected_returns, covariance_matrix, constraints, num_points=20
    )

    print(f"   Frontier Points Generated: {len(frontier_data['returns'])}")
    print(f"   Risk Range: [{frontier_data['risks'].min():.4f}, {frontier_data['risks'].max():.4f}]")
    print(f"   Return Range: [{frontier_data['returns'].min():.4f}, {frontier_data['returns'].max():.4f}]")
    print(f"   Max Sharpe Ratio: {frontier_data['sharpe_ratios'].max():.4f}")

    print("\n=== Portfolio Optimization Complete ===")

    # Performance comparison
    print(f"\nPerformance Comparison:")
    print(f"{'Method':<20} {'Return':<10} {'Risk':<10} {'Sharpe':<10}")
    print("-" * 50)
    print(f"{'Mean-Variance':<20} {mv_result['expected_return']:<10.4f} {mv_result['expected_risk']:<10.4f} {mv_result['sharpe_ratio']:<10.4f}")
    print(f"{'Risk Parity':<20} {rp_result['expected_return']:<10.4f} {rp_result['expected_risk']:<10.4f} {rp_result['sharpe_ratio']:<10.4f}")
    print(f"{'Black-Litterman':<20} {bl_result['expected_return']:<10.4f} {bl_result['expected_risk']:<10.4f} {bl_result['sharpe_ratio']:<10.4f}")
    print(f"{'HRP':<20} {hrp_result['expected_return']:<10.4f} {hrp_result['expected_risk']:<10.4f} {hrp_result['sharpe_ratio']:<10.4f}")

    # Diversification analysis
    print(f"\nDiversification Analysis:")
    def calculate_herfindahl_index(weights):
        return np.sum(weights ** 2)

    print(f"{'Method':<20} {'Concentration (HHI)':<20}")
    print("-" * 40)
    print(f"{'Mean-Variance':<20} {calculate_herfindahl_index(mv_result['weights']):<20.4f}")
    print(f"{'Risk Parity':<20} {calculate_herfindahl_index(rp_result['weights']):<20.4f}")
    print(f"{'Black-Litterman':<20} {calculate_herfindahl_index(bl_result['weights']):<20.4f}")
    print(f"{'HRP':<20} {calculate_herfindahl_index(hrp_result['weights']):<20.4f}")
    print(f"{'Equal Weight':<20} {calculate_herfindahl_index(np.ones(n_assets)/n_assets):<20.4f}")