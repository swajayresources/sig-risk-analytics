"""
Advanced Monte Carlo Simulation Engine
GPU-accelerated simulations with variance reduction techniques
"""

import numpy as np
import pandas as pd
from numba import cuda, jit, prange
import cupy as cp  # GPU arrays
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
import concurrent.futures
import time
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    logger.warning("CuPy not available. GPU acceleration disabled.")

@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulations"""
    n_simulations: int = 100000
    n_time_steps: int = 252
    use_gpu: bool = True
    use_antithetic_variates: bool = True
    use_control_variates: bool = True
    use_importance_sampling: bool = False
    use_stratified_sampling: bool = False
    random_seed: int = 42
    batch_size: int = 10000
    precision: str = 'float32'  # 'float32' or 'float64'

class VarianceReduction:
    """
    Variance reduction techniques for Monte Carlo simulations
    """

    @staticmethod
    def antithetic_variates(random_numbers: np.ndarray) -> np.ndarray:
        """
        Apply antithetic variates variance reduction

        Args:
            random_numbers: Original random numbers

        Returns:
            Extended array with antithetic variates
        """
        antithetic = -random_numbers
        return np.concatenate([random_numbers, antithetic], axis=0)

    @staticmethod
    def control_variates(payoffs: np.ndarray,
                        control_payoffs: np.ndarray,
                        control_expectation: float) -> np.ndarray:
        """
        Apply control variates variance reduction

        Args:
            payoffs: Original payoffs
            control_payoffs: Control variate payoffs
            control_expectation: Known expectation of control variate

        Returns:
            Variance-reduced payoffs
        """
        # Optimal control coefficient
        covariance = np.cov(payoffs, control_payoffs)[0, 1]
        control_variance = np.var(control_payoffs)

        if control_variance > 0:
            optimal_c = covariance / control_variance
            control_mean = np.mean(control_payoffs)
            reduced_payoffs = payoffs - optimal_c * (control_payoffs - control_expectation)
            return reduced_payoffs
        else:
            return payoffs

    @staticmethod
    def importance_sampling(payoffs: np.ndarray,
                          importance_weights: np.ndarray) -> Tuple[float, float]:
        """
        Apply importance sampling

        Args:
            payoffs: Payoff values
            importance_weights: Importance sampling weights

        Returns:
            Weighted mean and variance estimate
        """
        weighted_payoffs = payoffs * importance_weights
        mean_estimate = np.mean(weighted_payoffs)
        variance_estimate = np.var(weighted_payoffs) / len(payoffs)

        return mean_estimate, variance_estimate

class QuasiRandomNumbers:
    """
    Quasi-random number generators for better convergence
    """

    @staticmethod
    def sobol_sequence(n_dimensions: int, n_points: int) -> np.ndarray:
        """
        Generate Sobol quasi-random sequence

        Args:
            n_dimensions: Number of dimensions
            n_points: Number of points to generate

        Returns:
            Sobol sequence array
        """
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=n_dimensions, scramble=True)
            sample = sampler.random(n_points)
            return sample
        except ImportError:
            logger.warning("SciPy QMC not available, using pseudo-random numbers")
            return np.random.random((n_points, n_dimensions))

    @staticmethod
    def halton_sequence(n_dimensions: int, n_points: int) -> np.ndarray:
        """
        Generate Halton quasi-random sequence

        Args:
            n_dimensions: Number of dimensions
            n_points: Number of points to generate

        Returns:
            Halton sequence array
        """
        try:
            from scipy.stats import qmc
            sampler = qmc.Halton(d=n_dimensions, scramble=True)
            sample = sampler.random(n_points)
            return sample
        except ImportError:
            logger.warning("SciPy QMC not available, using pseudo-random numbers")
            return np.random.random((n_points, n_dimensions))

class PathGenerator:
    """
    Generate various types of stochastic paths
    """

    @staticmethod
    @jit(nopython=True, parallel=True)
    def geometric_brownian_motion(S0: float,
                                 mu: float,
                                 sigma: float,
                                 T: float,
                                 n_steps: int,
                                 n_paths: int,
                                 random_normals: np.ndarray) -> np.ndarray:
        """
        Generate Geometric Brownian Motion paths using Numba

        Args:
            S0: Initial price
            mu: Drift parameter
            sigma: Volatility parameter
            T: Time horizon
            n_steps: Number of time steps
            n_paths: Number of paths
            random_normals: Pre-generated random normals

        Returns:
            Price paths array (n_paths x n_steps+1)
        """
        dt = T / n_steps
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt)

        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = S0

        for i in prange(n_paths):
            for j in range(n_steps):
                paths[i, j + 1] = paths[i, j] * np.exp(
                    drift + diffusion * random_normals[i, j]
                )

        return paths

    @staticmethod
    @jit(nopython=True, parallel=True)
    def heston_paths(S0: float,
                    v0: float,
                    kappa: float,
                    theta: float,
                    sigma_v: float,
                    rho: float,
                    r: float,
                    T: float,
                    n_steps: int,
                    n_paths: int,
                    random_normals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate Heston stochastic volatility paths

        Args:
            S0: Initial stock price
            v0: Initial variance
            kappa: Mean reversion speed
            theta: Long-term variance
            sigma_v: Volatility of variance
            rho: Correlation between price and variance
            r: Risk-free rate
            T: Time horizon
            n_steps: Number of time steps
            n_paths: Number of paths
            random_normals: Pre-generated correlated random normals

        Returns:
            Tuple of (price_paths, variance_paths)
        """
        dt = T / n_steps

        S_paths = np.zeros((n_paths, n_steps + 1))
        v_paths = np.zeros((n_paths, n_steps + 1))

        S_paths[:, 0] = S0
        v_paths[:, 0] = v0

        for i in prange(n_paths):
            for j in range(n_steps):
                # Current values
                S = S_paths[i, j]
                v = max(v_paths[i, j], 0.0)  # Ensure non-negative variance

                # Random shocks
                dW1 = random_normals[i, j, 0] * np.sqrt(dt)
                dW2 = random_normals[i, j, 1] * np.sqrt(dt)

                # Variance process (Feller square-root)
                dv = kappa * (theta - v) * dt + sigma_v * np.sqrt(v) * dW2
                v_new = v + dv
                v_new = max(v_new, 0.0)  # Reflect at zero boundary

                # Price process
                dS = r * S * dt + np.sqrt(v) * S * dW1
                S_new = S + dS

                S_paths[i, j + 1] = S_new
                v_paths[i, j + 1] = v_new

        return S_paths, v_paths

    @staticmethod
    def jump_diffusion_paths(S0: float,
                           mu: float,
                           sigma: float,
                           lambda_jump: float,
                           mu_jump: float,
                           sigma_jump: float,
                           T: float,
                           n_steps: int,
                           n_paths: int) -> np.ndarray:
        """
        Generate Merton jump-diffusion paths

        Args:
            S0: Initial price
            mu: Drift parameter
            sigma: Diffusion volatility
            lambda_jump: Jump intensity
            mu_jump: Jump size mean
            sigma_jump: Jump size volatility
            T: Time horizon
            n_steps: Number of time steps
            n_paths: Number of paths

        Returns:
            Price paths with jumps
        """
        dt = T / n_steps

        # Generate diffusion paths
        normal_randoms = np.random.normal(0, 1, (n_paths, n_steps))
        gbm_paths = PathGenerator.geometric_brownian_motion(
            S0, mu, sigma, T, n_steps, n_paths, normal_randoms
        )

        # Add jumps
        for i in range(n_paths):
            for j in range(n_steps):
                # Poisson process for jump times
                jump_prob = lambda_jump * dt
                if np.random.random() < jump_prob:
                    # Jump size
                    jump_size = np.random.normal(mu_jump, sigma_jump)
                    gbm_paths[i, j + 1:] *= np.exp(jump_size)

        return gbm_paths

if GPU_AVAILABLE:
    class GPUMonteCarloEngine:
        """
        GPU-accelerated Monte Carlo engine using CuPy
        """

        def __init__(self, config: MonteCarloConfig):
            self.config = config
            if not GPU_AVAILABLE:
                raise RuntimeError("GPU not available")

        def generate_correlated_normals_gpu(self,
                                          n_paths: int,
                                          n_steps: int,
                                          n_assets: int,
                                          correlation_matrix: np.ndarray) -> cp.ndarray:
            """
            Generate correlated normal random numbers on GPU

            Args:
                n_paths: Number of simulation paths
                n_steps: Number of time steps
                n_assets: Number of assets
                correlation_matrix: Correlation matrix

            Returns:
                Correlated random numbers on GPU
            """
            # Move correlation matrix to GPU
            corr_gpu = cp.asarray(correlation_matrix, dtype=self.config.precision)

            # Cholesky decomposition on GPU
            chol_gpu = cp.linalg.cholesky(corr_gpu)

            # Generate independent normals on GPU
            independent_normals = cp.random.normal(
                0, 1, (n_paths, n_steps, n_assets), dtype=self.config.precision
            )

            # Apply correlation structure
            correlated_normals = cp.zeros_like(independent_normals)

            for i in range(n_paths):
                for j in range(n_steps):
                    correlated_normals[i, j] = chol_gpu @ independent_normals[i, j]

            return correlated_normals

        def geometric_brownian_motion_gpu(self,
                                        S0: cp.ndarray,
                                        mu: cp.ndarray,
                                        sigma: cp.ndarray,
                                        T: float,
                                        n_steps: int,
                                        correlated_normals: cp.ndarray) -> cp.ndarray:
            """
            Generate GBM paths on GPU

            Args:
                S0: Initial prices (per asset)
                mu: Drift parameters (per asset)
                sigma: Volatility parameters (per asset)
                T: Time horizon
                n_steps: Number of time steps
                correlated_normals: Correlated random numbers

            Returns:
                Price paths on GPU
            """
            n_paths, _, n_assets = correlated_normals.shape
            dt = T / n_steps

            # Calculate drift and diffusion terms
            drift = (mu - 0.5 * sigma**2) * dt
            diffusion = sigma * cp.sqrt(dt)

            # Initialize paths
            paths = cp.zeros((n_paths, n_steps + 1, n_assets), dtype=self.config.precision)
            paths[:, 0] = S0

            # Generate paths
            for t in range(n_steps):
                log_returns = drift + diffusion * correlated_normals[:, t]
                paths[:, t + 1] = paths[:, t] * cp.exp(log_returns)

            return paths

        def portfolio_monte_carlo_gpu(self,
                                    weights: np.ndarray,
                                    S0: np.ndarray,
                                    mu: np.ndarray,
                                    sigma: np.ndarray,
                                    correlation_matrix: np.ndarray,
                                    T: float = 1/252) -> Dict[str, Union[float, np.ndarray]]:
            """
            Run portfolio Monte Carlo simulation on GPU

            Args:
                weights: Portfolio weights
                S0: Initial prices
                mu: Expected returns
                sigma: Volatilities
                correlation_matrix: Asset correlation matrix
                T: Time horizon

            Returns:
                Simulation results
            """
            n_assets = len(weights)
            n_paths = self.config.n_simulations
            n_steps = self.config.n_time_steps

            # Move data to GPU
            weights_gpu = cp.asarray(weights, dtype=self.config.precision)
            S0_gpu = cp.asarray(S0, dtype=self.config.precision)
            mu_gpu = cp.asarray(mu, dtype=self.config.precision)
            sigma_gpu = cp.asarray(sigma, dtype=self.config.precision)

            # Generate correlated random numbers
            correlated_normals = self.generate_correlated_normals_gpu(
                n_paths, n_steps, n_assets, correlation_matrix
            )

            # Generate price paths
            price_paths = self.geometric_brownian_motion_gpu(
                S0_gpu, mu_gpu, sigma_gpu, T, n_steps, correlated_normals
            )

            # Calculate portfolio values
            portfolio_values = cp.sum(price_paths * weights_gpu, axis=2)

            # Calculate returns
            initial_portfolio_value = cp.sum(S0_gpu * weights_gpu)
            portfolio_returns = (portfolio_values[:, -1] - initial_portfolio_value) / initial_portfolio_value

            # Move results back to CPU for analysis
            portfolio_returns_cpu = cp.asnumpy(portfolio_returns)

            # Calculate risk metrics
            var_95 = -np.percentile(portfolio_returns_cpu, 5)
            var_99 = -np.percentile(portfolio_returns_cpu, 1)
            expected_shortfall_95 = -np.mean(portfolio_returns_cpu[portfolio_returns_cpu <= -var_95])

            return {
                'portfolio_returns': portfolio_returns_cpu,
                'var_95': var_95,
                'var_99': var_99,
                'expected_shortfall_95': expected_shortfall_95,
                'mean_return': np.mean(portfolio_returns_cpu),
                'volatility': np.std(portfolio_returns_cpu),
                'skewness': float(cp.asnumpy(cp.mean((portfolio_returns - cp.mean(portfolio_returns))**3) / cp.std(portfolio_returns)**3)),
                'kurtosis': float(cp.asnumpy(cp.mean((portfolio_returns - cp.mean(portfolio_returns))**4) / cp.std(portfolio_returns)**4)),
                'paths_generated': n_paths
            }

class AdvancedMonteCarloEngine:
    """
    Advanced Monte Carlo engine with multiple variance reduction techniques
    """

    def __init__(self, config: MonteCarloConfig = None):
        self.config = config or MonteCarloConfig()
        self.gpu_engine = None

        if self.config.use_gpu and GPU_AVAILABLE:
            try:
                self.gpu_engine = GPUMonteCarloEngine(self.config)
                logger.info("GPU acceleration enabled")
            except Exception as e:
                logger.warning(f"GPU initialization failed: {e}")
                self.config.use_gpu = False

        # Set random seed
        np.random.seed(self.config.random_seed)

    def portfolio_var_simulation(self,
                                weights: np.ndarray,
                                expected_returns: np.ndarray,
                                covariance_matrix: np.ndarray,
                                confidence_levels: List[float] = None,
                                horizon_days: int = 1) -> Dict[str, Union[float, np.ndarray]]:
        """
        Comprehensive portfolio VaR simulation with variance reduction

        Args:
            weights: Portfolio weights
            expected_returns: Expected returns
            covariance_matrix: Return covariance matrix
            confidence_levels: VaR confidence levels
            horizon_days: Investment horizon in days

        Returns:
            Comprehensive simulation results
        """
        if confidence_levels is None:
            confidence_levels = [0.90, 0.95, 0.99, 0.995]

        start_time = time.time()

        # Portfolio parameters
        portfolio_return = np.dot(weights, expected_returns) * horizon_days
        portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights)) * horizon_days

        # Generate scenarios
        if self.config.use_gpu and self.gpu_engine:
            scenarios = self._gpu_scenario_generation(
                weights, expected_returns, covariance_matrix, horizon_days
            )
        else:
            scenarios = self._cpu_scenario_generation(
                weights, expected_returns, covariance_matrix, horizon_days
            )

        # Apply variance reduction techniques
        if self.config.use_antithetic_variates:
            scenarios = VarianceReduction.antithetic_variates(scenarios)

        if self.config.use_control_variates:
            # Use analytical portfolio return as control variate
            control_expectation = portfolio_return
            control_scenarios = np.full(len(scenarios), portfolio_return)
            scenarios = VarianceReduction.control_variates(
                scenarios, control_scenarios, control_expectation
            )

        # Calculate VaR for different confidence levels
        var_estimates = {}
        es_estimates = {}

        for cl in confidence_levels:
            var_estimates[f'var_{cl:.1%}'] = -np.percentile(scenarios, (1-cl)*100)
            tail_scenarios = scenarios[scenarios <= -var_estimates[f'var_{cl:.1%}']]
            es_estimates[f'es_{cl:.1%}'] = -np.mean(tail_scenarios) if len(tail_scenarios) > 0 else var_estimates[f'var_{cl:.1%}']

        # Calculate additional statistics
        computation_time = time.time() - start_time

        results = {
            'scenarios': scenarios,
            'portfolio_return': portfolio_return,
            'portfolio_volatility': np.sqrt(portfolio_variance),
            'simulation_mean': np.mean(scenarios),
            'simulation_std': np.std(scenarios),
            'skewness': self._calculate_skewness(scenarios),
            'kurtosis': self._calculate_kurtosis(scenarios),
            'computation_time': computation_time,
            'n_scenarios': len(scenarios),
            'variance_reduction_used': {
                'antithetic_variates': self.config.use_antithetic_variates,
                'control_variates': self.config.use_control_variates,
                'importance_sampling': self.config.use_importance_sampling
            }
        }

        results.update(var_estimates)
        results.update(es_estimates)

        return results

    def _gpu_scenario_generation(self,
                                weights: np.ndarray,
                                expected_returns: np.ndarray,
                                covariance_matrix: np.ndarray,
                                horizon_days: int) -> np.ndarray:
        """Generate scenarios using GPU acceleration"""
        if not self.gpu_engine:
            raise RuntimeError("GPU engine not available")

        # Prepare parameters for GPU
        S0 = np.ones(len(weights)) * 100  # Normalized initial prices
        mu = expected_returns
        sigma = np.sqrt(np.diag(covariance_matrix))
        correlation_matrix = np.corrcoef(covariance_matrix)

        # Run GPU simulation
        gpu_results = self.gpu_engine.portfolio_monte_carlo_gpu(
            weights, S0, mu, sigma, correlation_matrix, horizon_days / 252
        )

        return gpu_results['portfolio_returns']

    def _cpu_scenario_generation(self,
                                weights: np.ndarray,
                                expected_returns: np.ndarray,
                                covariance_matrix: np.ndarray,
                                horizon_days: int) -> np.ndarray:
        """Generate scenarios using CPU (with Numba optimization)"""
        n_assets = len(weights)
        n_scenarios = self.config.n_simulations

        # Adjust for horizon
        horizon_mu = expected_returns * horizon_days
        horizon_cov = covariance_matrix * horizon_days

        # Cholesky decomposition for correlation
        try:
            chol_matrix = np.linalg.cholesky(horizon_cov)
        except np.linalg.LinAlgError:
            # Handle non-positive definite matrices
            eigenvals, eigenvecs = np.linalg.eigh(horizon_cov)
            eigenvals = np.maximum(eigenvals, 1e-8)
            horizon_cov = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
            chol_matrix = np.linalg.cholesky(horizon_cov)

        # Generate scenarios using optimized function
        scenarios = self._generate_scenarios_numba(
            n_scenarios, n_assets, chol_matrix, horizon_mu, weights
        )

        return scenarios

    @staticmethod
    @jit(nopython=True, parallel=True)
    def _generate_scenarios_numba(n_scenarios: int,
                                 n_assets: int,
                                 chol_matrix: np.ndarray,
                                 means: np.ndarray,
                                 weights: np.ndarray) -> np.ndarray:
        """Numba-optimized scenario generation"""
        scenarios = np.zeros(n_scenarios)

        for i in prange(n_scenarios):
            # Generate independent normals
            random_vars = np.random.standard_normal(n_assets)

            # Apply Cholesky decomposition
            correlated_vars = np.zeros(n_assets)
            for j in range(n_assets):
                for k in range(j + 1):
                    correlated_vars[j] += chol_matrix[j, k] * random_vars[k]

            # Add means and calculate portfolio return
            asset_returns = means + correlated_vars
            scenarios[i] = np.sum(weights * asset_returns)

        return scenarios

    def options_monte_carlo(self,
                          option_type: str,
                          spot: float,
                          strike: float,
                          time_to_expiry: float,
                          volatility: float,
                          risk_free_rate: float = 0.05,
                          dividend_yield: float = 0.0,
                          barrier_level: float = None,
                          asian_observations: List[float] = None) -> Dict[str, float]:
        """
        Monte Carlo pricing for various option types

        Args:
            option_type: 'european_call', 'european_put', 'asian_call', 'barrier_up_out_call', etc.
            spot: Current stock price
            strike: Strike price
            time_to_expiry: Time to expiration in years
            volatility: Volatility
            risk_free_rate: Risk-free rate
            dividend_yield: Dividend yield
            barrier_level: Barrier level for barrier options
            asian_observations: Observation times for Asian options

        Returns:
            Option pricing results
        """
        n_paths = self.config.n_simulations
        n_steps = max(self.config.n_time_steps, 100)

        # Generate stock price paths
        random_normals = np.random.normal(0, 1, (n_paths, n_steps))
        price_paths = PathGenerator.geometric_brownian_motion(
            spot, risk_free_rate - dividend_yield, volatility,
            time_to_expiry, n_steps, n_paths, random_normals
        )

        # Calculate payoffs based on option type
        payoffs = self._calculate_option_payoffs(
            option_type, price_paths, strike, barrier_level, asian_observations
        )

        # Discount payoffs to present value
        discount_factor = np.exp(-risk_free_rate * time_to_expiry)
        discounted_payoffs = payoffs * discount_factor

        # Apply variance reduction if enabled
        if self.config.use_antithetic_variates:
            # Generate antithetic paths
            antithetic_normals = -random_normals
            antithetic_paths = PathGenerator.geometric_brownian_motion(
                spot, risk_free_rate - dividend_yield, volatility,
                time_to_expiry, n_steps, n_paths, antithetic_normals
            )
            antithetic_payoffs = self._calculate_option_payoffs(
                option_type, antithetic_paths, strike, barrier_level, asian_observations
            )
            antithetic_discounted = antithetic_payoffs * discount_factor

            # Combine original and antithetic
            combined_payoffs = (discounted_payoffs + antithetic_discounted) / 2
            option_price = np.mean(combined_payoffs)
            standard_error = np.std(combined_payoffs) / np.sqrt(len(combined_payoffs))
        else:
            option_price = np.mean(discounted_payoffs)
            standard_error = np.std(discounted_payoffs) / np.sqrt(len(discounted_payoffs))

        # Calculate confidence interval
        confidence_interval = [
            option_price - 1.96 * standard_error,
            option_price + 1.96 * standard_error
        ]

        return {
            'option_price': option_price,
            'standard_error': standard_error,
            'confidence_interval': confidence_interval,
            'paths_used': n_paths * (2 if self.config.use_antithetic_variates else 1),
            'convergence_ratio': standard_error / option_price if option_price > 0 else float('inf')
        }

    def _calculate_option_payoffs(self,
                                option_type: str,
                                price_paths: np.ndarray,
                                strike: float,
                                barrier_level: float = None,
                                asian_observations: List[float] = None) -> np.ndarray:
        """Calculate option payoffs for different option types"""
        n_paths = price_paths.shape[0]
        payoffs = np.zeros(n_paths)

        if option_type == 'european_call':
            final_prices = price_paths[:, -1]
            payoffs = np.maximum(final_prices - strike, 0)

        elif option_type == 'european_put':
            final_prices = price_paths[:, -1]
            payoffs = np.maximum(strike - final_prices, 0)

        elif option_type == 'asian_call':
            if asian_observations is None:
                # Arithmetic average of all time points
                average_prices = np.mean(price_paths, axis=1)
            else:
                # Average of specific observation points
                observation_indices = [int(obs * price_paths.shape[1]) for obs in asian_observations]
                average_prices = np.mean(price_paths[:, observation_indices], axis=1)
            payoffs = np.maximum(average_prices - strike, 0)

        elif option_type == 'asian_put':
            if asian_observations is None:
                average_prices = np.mean(price_paths, axis=1)
            else:
                observation_indices = [int(obs * price_paths.shape[1]) for obs in asian_observations]
                average_prices = np.mean(price_paths[:, observation_indices], axis=1)
            payoffs = np.maximum(strike - average_prices, 0)

        elif option_type == 'barrier_up_out_call' and barrier_level is not None:
            final_prices = price_paths[:, -1]
            # Check if barrier was breached
            max_prices = np.max(price_paths, axis=1)
            barrier_not_breached = max_prices < barrier_level
            payoffs = np.maximum(final_prices - strike, 0) * barrier_not_breached

        elif option_type == 'barrier_down_out_put' and barrier_level is not None:
            final_prices = price_paths[:, -1]
            # Check if barrier was breached
            min_prices = np.min(price_paths, axis=1)
            barrier_not_breached = min_prices > barrier_level
            payoffs = np.maximum(strike - final_prices, 0) * barrier_not_breached

        elif option_type == 'lookback_call':
            max_prices = np.max(price_paths, axis=1)
            final_prices = price_paths[:, -1]
            payoffs = max_prices - final_prices

        elif option_type == 'lookback_put':
            min_prices = np.min(price_paths, axis=1)
            final_prices = price_paths[:, -1]
            payoffs = final_prices - min_prices

        else:
            raise ValueError(f"Unsupported option type: {option_type}")

        return payoffs

    @staticmethod
    def _calculate_skewness(data: np.ndarray) -> float:
        """Calculate skewness of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)

    @staticmethod
    def _calculate_kurtosis(data: np.ndarray) -> float:
        """Calculate excess kurtosis of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3

    def stress_test_simulation(self,
                             base_scenario: Dict[str, float],
                             stress_factors: Dict[str, float],
                             weights: np.ndarray,
                             expected_returns: np.ndarray,
                             covariance_matrix: np.ndarray) -> Dict[str, float]:
        """
        Run stress test simulations

        Args:
            base_scenario: Base case parameters
            stress_factors: Stress scenario factors
            weights: Portfolio weights
            expected_returns: Expected returns
            covariance_matrix: Covariance matrix

        Returns:
            Stress test results
        """
        results = {}

        # Base case simulation
        base_results = self.portfolio_var_simulation(
            weights, expected_returns, covariance_matrix
        )
        results['base_case'] = {
            'var_95': base_results['var_0.95'],
            'expected_shortfall': base_results['es_0.95'],
            'mean_return': base_results['simulation_mean']
        }

        # Stress scenarios
        for stress_name, stress_multiplier in stress_factors.items():
            # Apply stress to expected returns
            stressed_returns = expected_returns * stress_multiplier

            # Apply stress to volatilities (diagonal of covariance matrix)
            stressed_cov = covariance_matrix.copy()
            volatilities = np.sqrt(np.diag(covariance_matrix))
            stressed_volatilities = volatilities * abs(stress_multiplier)

            # Reconstruct covariance matrix with stressed volatilities
            correlation_matrix = covariance_matrix / np.outer(volatilities, volatilities)
            stressed_cov = np.outer(stressed_volatilities, stressed_volatilities) * correlation_matrix

            # Run simulation
            stress_results = self.portfolio_var_simulation(
                weights, stressed_returns, stressed_cov
            )

            results[stress_name] = {
                'var_95': stress_results['var_0.95'],
                'expected_shortfall': stress_results['es_0.95'],
                'mean_return': stress_results['simulation_mean'],
                'stress_multiplier': stress_multiplier
            }

        return results

# Example usage and testing
if __name__ == "__main__":
    # Example usage of the Monte Carlo engine
    print("=== Advanced Monte Carlo Engine Demo ===\n")

    # Configuration
    config = MonteCarloConfig(
        n_simulations=50000,
        use_gpu=GPU_AVAILABLE,
        use_antithetic_variates=True,
        use_control_variates=True,
        random_seed=42
    )

    # Initialize engine
    mc_engine = AdvancedMonteCarloEngine(config)

    # Example 1: Portfolio VaR simulation
    print("1. Portfolio VaR Simulation")
    n_assets = 4
    weights = np.array([0.3, 0.25, 0.25, 0.2])
    expected_returns = np.array([0.08, 0.12, 0.10, 0.15]) / 252  # Daily returns

    # Create covariance matrix
    volatilities = np.array([0.15, 0.20, 0.18, 0.25]) / np.sqrt(252)  # Daily vol
    correlation = np.array([
        [1.0, 0.6, 0.4, 0.3],
        [0.6, 1.0, 0.5, 0.4],
        [0.4, 0.5, 1.0, 0.3],
        [0.3, 0.4, 0.3, 1.0]
    ])
    covariance_matrix = np.outer(volatilities, volatilities) * correlation

    portfolio_results = mc_engine.portfolio_var_simulation(
        weights, expected_returns, covariance_matrix,
        confidence_levels=[0.95, 0.99], horizon_days=1
    )

    print(f"   Portfolio VaR (95%): {portfolio_results['var_95.0%']:.4f}")
    print(f"   Portfolio VaR (99%): {portfolio_results['var_99.0%']:.4f}")
    print(f"   Expected Shortfall (95%): {portfolio_results['es_95.0%']:.4f}")
    print(f"   Simulation Mean: {portfolio_results['simulation_mean']:.4f}")
    print(f"   Simulation Std: {portfolio_results['simulation_std']:.4f}")
    print(f"   Computation Time: {portfolio_results['computation_time']:.3f}s")
    print(f"   Scenarios Used: {portfolio_results['n_scenarios']:,}\n")

    # Example 2: Options pricing
    print("2. Options Monte Carlo Pricing")
    option_results = mc_engine.options_monte_carlo(
        option_type='european_call',
        spot=100,
        strike=105,
        time_to_expiry=0.25,
        volatility=0.2,
        risk_free_rate=0.05
    )

    print(f"   European Call Price: {option_results['option_price']:.4f}")
    print(f"   Standard Error: {option_results['standard_error']:.4f}")
    print(f"   95% Confidence Interval: [{option_results['confidence_interval'][0]:.4f}, {option_results['confidence_interval'][1]:.4f}]")
    print(f"   Convergence Ratio: {option_results['convergence_ratio']:.6f}\n")

    # Example 3: Asian option pricing
    print("3. Asian Option Pricing")
    asian_results = mc_engine.options_monte_carlo(
        option_type='asian_call',
        spot=100,
        strike=100,
        time_to_expiry=1.0,
        volatility=0.25,
        risk_free_rate=0.05
    )

    print(f"   Asian Call Price: {asian_results['option_price']:.4f}")
    print(f"   Standard Error: {asian_results['standard_error']:.4f}\n")

    # Example 4: Stress testing
    print("4. Stress Test Simulation")
    stress_factors = {
        'market_crash': 0.7,  # 30% decline in returns
        'volatility_spike': 1.5,  # 50% increase in volatility
        'correlation_crisis': 1.2  # 20% increase in correlations
    }

    stress_results = mc_engine.stress_test_simulation(
        base_scenario={},
        stress_factors=stress_factors,
        weights=weights,
        expected_returns=expected_returns,
        covariance_matrix=covariance_matrix
    )

    print("   Stress Test Results:")
    for scenario, results in stress_results.items():
        print(f"     {scenario.replace('_', ' ').title()}:")
        print(f"       VaR (95%): {results['var_95']:.4f}")
        print(f"       Expected Shortfall: {results['expected_shortfall']:.4f}")

    print("\n=== Monte Carlo Simulation Complete ===")

    # Performance comparison
    if GPU_AVAILABLE:
        print(f"\nGPU Acceleration: {'Enabled' if config.use_gpu else 'Disabled'}")
    print(f"Variance Reduction Techniques:")
    print(f"  - Antithetic Variates: {config.use_antithetic_variates}")
    print(f"  - Control Variates: {config.use_control_variates}")
    print(f"  - Importance Sampling: {config.use_importance_sampling}")