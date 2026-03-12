"""
Advanced Quantitative Risk Metrics Library
Implements sophisticated mathematical finance models for institutional trading
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.optimize as optimize
from scipy.linalg import cholesky, solve_triangular
from numba import jit, prange
import warnings
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import concurrent.futures
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

@dataclass
class RiskMetricsConfig:
    """Configuration for risk metrics calculations"""
    confidence_levels: List[float] = None
    monte_carlo_paths: int = 100000
    garch_max_lag: int = 5
    var_horizon_days: int = 1
    bootstrap_samples: int = 1000
    numerical_delta: float = 0.01
    risk_free_rate: float = 0.05
    enable_parallel: bool = True
    cache_size: int = 1000

    def __post_init__(self):
        if self.confidence_levels is None:
            self.confidence_levels = [0.90, 0.95, 0.99, 0.995]

class AdvancedRiskMetrics:
    """
    Comprehensive risk metrics calculation engine with advanced mathematical models
    """

    def __init__(self, config: RiskMetricsConfig = None):
        self.config = config or RiskMetricsConfig()
        self._setup_cache()

    def _setup_cache(self):
        """Initialize caching for computationally expensive operations"""
        self.calculate_portfolio_var = lru_cache(maxsize=self.config.cache_size)(
            self._calculate_portfolio_var
        )
        self.estimate_garch_parameters = lru_cache(maxsize=self.config.cache_size)(
            self._estimate_garch_parameters
        )

class HistoricalSimulationVaR:
    """
    Historical Simulation Value-at-Risk implementation with bootstrap
    """

    def __init__(self, lookback_days: int = 252):
        self.lookback_days = lookback_days

    def calculate_var(self,
                     returns: np.ndarray,
                     confidence_level: float = 0.95,
                     bootstrap: bool = True,
                     bootstrap_samples: int = 1000) -> Dict[str, float]:
        """
        Calculate Historical Simulation VaR with optional bootstrap confidence intervals

        Args:
            returns: Historical returns array
            confidence_level: Confidence level for VaR calculation
            bootstrap: Whether to use bootstrap for confidence intervals
            bootstrap_samples: Number of bootstrap samples

        Returns:
            Dictionary with VaR estimate and confidence intervals
        """
        if len(returns) < self.lookback_days:
            logger.warning(f"Insufficient data: {len(returns)} < {self.lookback_days}")

        # Use most recent data
        recent_returns = returns[-self.lookback_days:] if len(returns) > self.lookback_days else returns

        # Calculate percentile VaR
        var_estimate = -np.percentile(recent_returns, (1 - confidence_level) * 100)

        # Calculate Expected Shortfall
        tail_returns = recent_returns[recent_returns <= -var_estimate]
        expected_shortfall = -np.mean(tail_returns) if len(tail_returns) > 0 else var_estimate

        result = {
            'var': var_estimate,
            'expected_shortfall': expected_shortfall,
            'scenarios_used': len(recent_returns)
        }

        if bootstrap:
            bootstrap_vars = self._bootstrap_var(recent_returns, confidence_level, bootstrap_samples)
            result.update({
                'var_confidence_interval': np.percentile(bootstrap_vars, [2.5, 97.5]),
                'var_std_error': np.std(bootstrap_vars)
            })

        return result

    def _bootstrap_var(self, returns: np.ndarray, confidence_level: float, n_samples: int) -> np.ndarray:
        """Bootstrap resampling for VaR confidence intervals"""
        bootstrap_vars = np.zeros(n_samples)
        n_returns = len(returns)

        for i in range(n_samples):
            # Resample with replacement
            bootstrap_sample = np.random.choice(returns, size=n_returns, replace=True)
            bootstrap_vars[i] = -np.percentile(bootstrap_sample, (1 - confidence_level) * 100)

        return bootstrap_vars

class ParametricVaR:
    """
    Parametric VaR using various distributional assumptions
    """

    def __init__(self, distribution: str = 'normal'):
        """
        Initialize parametric VaR calculator

        Args:
            distribution: Distribution assumption ('normal', 't', 'cornish_fisher')
        """
        self.distribution = distribution

    def calculate_var(self,
                     returns: np.ndarray,
                     confidence_level: float = 0.95,
                     horizon_days: int = 1) -> Dict[str, float]:
        """
        Calculate parametric VaR using specified distribution

        Args:
            returns: Historical returns
            confidence_level: Confidence level
            horizon_days: VaR horizon in days

        Returns:
            VaR estimates and distribution parameters
        """
        mu = np.mean(returns)
        sigma = np.std(returns, ddof=1)

        # Scale for horizon
        horizon_mu = mu * horizon_days
        horizon_sigma = sigma * np.sqrt(horizon_days)

        if self.distribution == 'normal':
            var_estimate = self._normal_var(horizon_mu, horizon_sigma, confidence_level)

        elif self.distribution == 't':
            var_estimate, df = self._t_distribution_var(returns, confidence_level, horizon_days)

        elif self.distribution == 'cornish_fisher':
            var_estimate = self._cornish_fisher_var(returns, confidence_level, horizon_days)

        else:
            raise ValueError(f"Unsupported distribution: {self.distribution}")

        # Expected Shortfall calculation
        if self.distribution == 'normal':
            expected_shortfall = self._normal_expected_shortfall(
                horizon_mu, horizon_sigma, confidence_level
            )
        else:
            # Use numerical integration for non-normal distributions
            expected_shortfall = var_estimate * 1.2  # Approximate

        return {
            'var': var_estimate,
            'expected_shortfall': expected_shortfall,
            'distribution': self.distribution,
            'mu': horizon_mu,
            'sigma': horizon_sigma
        }

    def _normal_var(self, mu: float, sigma: float, confidence_level: float) -> float:
        """Calculate VaR under normal distribution assumption"""
        z_score = stats.norm.ppf(1 - confidence_level)
        return -(mu + z_score * sigma)

    def _normal_expected_shortfall(self, mu: float, sigma: float, confidence_level: float) -> float:
        """Calculate Expected Shortfall under normal distribution"""
        z_score = stats.norm.ppf(1 - confidence_level)
        conditional_expectation = stats.norm.pdf(z_score) / (1 - confidence_level)
        return -(mu - sigma * conditional_expectation)

    def _t_distribution_var(self, returns: np.ndarray, confidence_level: float, horizon_days: int) -> Tuple[float, float]:
        """Calculate VaR using Student's t-distribution"""
        # Estimate degrees of freedom using maximum likelihood
        def neg_log_likelihood(df):
            return -np.sum(stats.t.logpdf(returns, df=df,
                                        loc=np.mean(returns),
                                        scale=np.std(returns, ddof=1)))

        # Optimize degrees of freedom
        result = optimize.minimize_scalar(neg_log_likelihood, bounds=(2.1, 50), method='bounded')
        optimal_df = result.x

        mu = np.mean(returns) * horizon_days
        sigma = np.std(returns, ddof=1) * np.sqrt(horizon_days)

        # Adjust sigma for t-distribution
        t_sigma = sigma * np.sqrt((optimal_df - 2) / optimal_df)
        t_quantile = stats.t.ppf(1 - confidence_level, df=optimal_df)

        var_estimate = -(mu + t_quantile * t_sigma)

        return var_estimate, optimal_df

    def _cornish_fisher_var(self, returns: np.ndarray, confidence_level: float, horizon_days: int) -> float:
        """Calculate VaR using Cornish-Fisher expansion for non-normal returns"""
        mu = np.mean(returns) * horizon_days
        sigma = np.std(returns, ddof=1) * np.sqrt(horizon_days)

        # Calculate moments
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns, fisher=True)  # Excess kurtosis

        # Cornish-Fisher quantile adjustment
        z = stats.norm.ppf(1 - confidence_level)

        # Cornish-Fisher expansion
        cf_quantile = (z +
                      (z**2 - 1) * skewness / 6 +
                      (z**3 - 3*z) * kurtosis / 24 -
                      (2*z**3 - 5*z) * skewness**2 / 36)

        var_estimate = -(mu + cf_quantile * sigma)

        return var_estimate

class MonteCarloVaR:
    """
    Monte Carlo VaR simulation with advanced variance reduction techniques
    """

    def __init__(self, n_simulations: int = 100000, use_antithetic: bool = True):
        self.n_simulations = n_simulations
        self.use_antithetic = use_antithetic

    def calculate_var(self,
                     portfolio_weights: np.ndarray,
                     expected_returns: np.ndarray,
                     covariance_matrix: np.ndarray,
                     confidence_level: float = 0.95,
                     horizon_days: int = 1) -> Dict[str, float]:
        """
        Calculate portfolio VaR using Monte Carlo simulation

        Args:
            portfolio_weights: Asset weights in portfolio
            expected_returns: Expected returns for each asset
            covariance_matrix: Asset return covariance matrix
            confidence_level: Confidence level for VaR
            horizon_days: Investment horizon in days

        Returns:
            VaR metrics and simulation statistics
        """
        # Portfolio parameters
        portfolio_return = np.dot(portfolio_weights, expected_returns) * horizon_days
        portfolio_variance = np.dot(portfolio_weights,
                                  np.dot(covariance_matrix * horizon_days, portfolio_weights))
        portfolio_volatility = np.sqrt(portfolio_variance)

        # Generate correlated random scenarios
        scenarios = self._generate_scenarios(expected_returns, covariance_matrix, horizon_days)

        # Calculate portfolio returns for each scenario
        portfolio_returns = np.dot(scenarios, portfolio_weights)

        # Calculate VaR and Expected Shortfall
        var_estimate = -np.percentile(portfolio_returns, (1 - confidence_level) * 100)

        # Expected Shortfall
        tail_losses = portfolio_returns[portfolio_returns <= -var_estimate]
        expected_shortfall = -np.mean(tail_losses) if len(tail_losses) > 0 else var_estimate

        # Component VaR calculation
        component_vars = self._calculate_component_var(
            portfolio_weights, scenarios, var_estimate
        )

        return {
            'var': var_estimate,
            'expected_shortfall': expected_shortfall,
            'portfolio_return': portfolio_return,
            'portfolio_volatility': portfolio_volatility,
            'component_vars': component_vars,
            'scenarios_used': len(portfolio_returns),
            'simulation_std_error': np.std(portfolio_returns) / np.sqrt(len(portfolio_returns))
        }

    @jit(nopython=True)
    def _generate_scenarios_numba(self,
                                 n_assets: int,
                                 n_scenarios: int,
                                 chol_matrix: np.ndarray,
                                 means: np.ndarray) -> np.ndarray:
        """Numba-optimized scenario generation"""
        scenarios = np.zeros((n_scenarios, n_assets))

        for i in prange(n_scenarios):
            # Generate independent normal random variables
            random_vars = np.random.standard_normal(n_assets)

            # Apply Cholesky decomposition for correlation
            correlated_vars = np.dot(chol_matrix, random_vars)

            # Add mean returns
            scenarios[i] = means + correlated_vars

        return scenarios

    def _generate_scenarios(self,
                           expected_returns: np.ndarray,
                           covariance_matrix: np.ndarray,
                           horizon_days: int) -> np.ndarray:
        """Generate correlated asset return scenarios"""
        n_assets = len(expected_returns)

        # Adjust for horizon
        horizon_means = expected_returns * horizon_days
        horizon_cov = covariance_matrix * horizon_days

        # Cholesky decomposition for correlation
        try:
            chol_matrix = cholesky(horizon_cov, lower=True)
        except np.linalg.LinAlgError:
            # Handle non-positive definite matrices
            eigenvals, eigenvecs = np.linalg.eigh(horizon_cov)
            eigenvals = np.maximum(eigenvals, 1e-8)  # Ensure positive eigenvalues
            horizon_cov = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
            chol_matrix = cholesky(horizon_cov, lower=True)

        # Generate scenarios
        n_scenarios = self.n_simulations // (2 if self.use_antithetic else 1)

        scenarios = self._generate_scenarios_numba(
            n_assets, n_scenarios, chol_matrix, horizon_means
        )

        if self.use_antithetic:
            # Antithetic variates for variance reduction
            antithetic_scenarios = 2 * horizon_means[np.newaxis, :] - scenarios
            scenarios = np.vstack([scenarios, antithetic_scenarios])

        return scenarios

    def _calculate_component_var(self,
                                weights: np.ndarray,
                                scenarios: np.ndarray,
                                portfolio_var: float) -> np.ndarray:
        """Calculate component VaR for each asset"""
        n_assets = len(weights)
        component_vars = np.zeros(n_assets)

        # Small perturbation for numerical derivative
        delta = 0.0001

        for i in range(n_assets):
            # Perturb weight slightly
            perturbed_weights = weights.copy()
            perturbed_weights[i] += delta
            perturbed_weights = perturbed_weights / np.sum(perturbed_weights)  # Renormalize

            # Calculate VaR with perturbed weights
            perturbed_returns = np.dot(scenarios, perturbed_weights)
            perturbed_var = -np.percentile(perturbed_returns, 5)  # 95% VaR

            # Numerical derivative
            component_vars[i] = (perturbed_var - portfolio_var) / delta * weights[i]

        return component_vars

class GARCHVolatilityForecasting:
    """
    GARCH(1,1) and GARCH(p,q) volatility forecasting models
    """

    def __init__(self, p: int = 1, q: int = 1):
        """
        Initialize GARCH model

        Args:
            p: Number of ARCH terms
            q: Number of GARCH terms
        """
        self.p = p
        self.q = q
        self.parameters = None
        self.fitted = False

    def fit(self, returns: np.ndarray, method: str = 'mle') -> Dict[str, float]:
        """
        Fit GARCH model to return series

        Args:
            returns: Return series
            method: Estimation method ('mle' for Maximum Likelihood)

        Returns:
            Fitted parameters and diagnostics
        """
        # Remove mean (assuming zero mean for simplicity)
        residuals = returns - np.mean(returns)

        # Initial parameter guesses
        initial_params = self._get_initial_parameters(residuals)

        # Maximum likelihood estimation
        if method == 'mle':
            result = optimize.minimize(
                self._negative_log_likelihood,
                initial_params,
                args=(residuals,),
                method='L-BFGS-B',
                bounds=self._get_parameter_bounds()
            )

            if result.success:
                self.parameters = result.x
                self.fitted = True

                # Calculate additional diagnostics
                log_likelihood = -result.fun
                aic = 2 * len(self.parameters) - 2 * log_likelihood
                bic = len(self.parameters) * np.log(len(residuals)) - 2 * log_likelihood

                return {
                    'omega': self.parameters[0],
                    'alpha': self.parameters[1:1+self.p],
                    'beta': self.parameters[1+self.p:1+self.p+self.q],
                    'log_likelihood': log_likelihood,
                    'aic': aic,
                    'bic': bic,
                    'convergence': result.success
                }
            else:
                raise RuntimeError("GARCH optimization failed to converge")
        else:
            raise ValueError(f"Unsupported estimation method: {method}")

    def forecast(self,
                steps_ahead: int = 1,
                last_returns: np.ndarray = None) -> np.ndarray:
        """
        Forecast conditional volatility

        Args:
            steps_ahead: Number of steps to forecast
            last_returns: Most recent returns for initialization

        Returns:
            Forecasted conditional variances
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before forecasting")

        omega = self.parameters[0]
        alpha = self.parameters[1:1+self.p]
        beta = self.parameters[1+self.p:1+self.p+self.q]

        # Initialize with last observed values
        if last_returns is not None:
            recent_returns = last_returns[-max(self.p, self.q):]
        else:
            recent_returns = np.zeros(max(self.p, self.q))

        # Calculate unconditional variance for long-term forecast
        unconditional_var = omega / (1 - np.sum(alpha) - np.sum(beta))

        forecasts = np.zeros(steps_ahead)

        # One-step ahead forecast
        if len(recent_returns) >= self.p:
            arch_component = np.sum(alpha * recent_returns[-self.p:]**2)
        else:
            arch_component = 0

        if hasattr(self, '_last_conditional_variance'):
            garch_component = np.sum(beta * np.array([self._last_conditional_variance]))
        else:
            garch_component = 0

        forecasts[0] = omega + arch_component + garch_component

        # Multi-step ahead forecasts
        for t in range(1, steps_ahead):
            # GARCH forecasts converge to unconditional variance
            persistence = np.sum(alpha) + np.sum(beta)
            forecasts[t] = (unconditional_var +
                          (forecasts[0] - unconditional_var) * persistence**t)

        return forecasts

    def _get_initial_parameters(self, residuals: np.ndarray) -> np.ndarray:
        """Get initial parameter estimates"""
        # Simple moment-based initial estimates
        unconditional_var = np.var(residuals)

        omega_init = unconditional_var * 0.1
        alpha_init = np.full(self.p, 0.1 / self.p)
        beta_init = np.full(self.q, 0.8 / self.q)

        return np.concatenate([[omega_init], alpha_init, beta_init])

    def _get_parameter_bounds(self) -> List[Tuple[float, float]]:
        """Get parameter bounds for optimization"""
        bounds = [(1e-6, None)]  # omega > 0
        bounds.extend([(1e-6, 1)] * self.p)  # 0 < alpha < 1
        bounds.extend([(1e-6, 1)] * self.q)  # 0 < beta < 1
        return bounds

    def _negative_log_likelihood(self, params: np.ndarray, residuals: np.ndarray) -> float:
        """Calculate negative log-likelihood for GARCH model"""
        omega = params[0]
        alpha = params[1:1+self.p]
        beta = params[1+self.p:1+self.p+self.q]

        n = len(residuals)
        log_likelihood = 0

        # Initialize conditional variance
        h = np.var(residuals) * np.ones(n)

        # Calculate conditional variances
        for t in range(max(self.p, self.q), n):
            h[t] = omega

            # ARCH terms
            for i in range(self.p):
                if t - i - 1 >= 0:
                    h[t] += alpha[i] * residuals[t - i - 1]**2

            # GARCH terms
            for j in range(self.q):
                if t - j - 1 >= 0:
                    h[t] += beta[j] * h[t - j - 1]

        # Calculate log-likelihood
        for t in range(max(self.p, self.q), n):
            if h[t] > 1e-8:  # Avoid numerical issues
                log_likelihood += -0.5 * (np.log(2 * np.pi) + np.log(h[t]) +
                                         residuals[t]**2 / h[t])

        return -log_likelihood

class OptionsGreeksCalculator:
    """
    Comprehensive options Greeks calculator with advanced features
    """

    def __init__(self):
        self.risk_free_rate = 0.05
        self.dividend_yield = 0.0

    def calculate_all_greeks(self,
                           spot: float,
                           strike: float,
                           time_to_expiry: float,
                           volatility: float,
                           option_type: str = 'call',
                           risk_free_rate: float = None,
                           dividend_yield: float = None) -> Dict[str, float]:
        """
        Calculate all Greeks for European options using Black-Scholes

        Args:
            spot: Current asset price
            strike: Strike price
            time_to_expiry: Time to expiry in years
            volatility: Implied volatility
            option_type: 'call' or 'put'
            risk_free_rate: Risk-free rate
            dividend_yield: Dividend yield

        Returns:
            Dictionary with all Greeks
        """
        r = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
        q = dividend_yield if dividend_yield is not None else self.dividend_yield

        # Calculate d1 and d2
        d1, d2 = self._calculate_d1_d2(spot, strike, time_to_expiry, volatility, r, q)

        # Standard Greeks
        delta = self._calculate_delta(d1, d2, option_type, time_to_expiry, q)
        gamma = self._calculate_gamma(spot, d1, volatility, time_to_expiry, q)
        theta = self._calculate_theta(spot, strike, d1, d2, volatility, time_to_expiry,
                                    option_type, r, q)
        vega = self._calculate_vega(spot, d1, time_to_expiry, q)
        rho = self._calculate_rho(strike, d2, time_to_expiry, option_type, r)
        epsilon = self._calculate_epsilon(spot, d1, time_to_expiry, option_type, q)

        # Higher-order Greeks
        charm = self._calculate_charm(spot, d1, d2, volatility, time_to_expiry,
                                    option_type, r, q)
        vanna = self._calculate_vanna(spot, d1, d2, volatility, time_to_expiry, q)
        volga = self._calculate_volga(spot, d1, d2, volatility, time_to_expiry, q)

        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho,
            'epsilon': epsilon,
            'charm': charm,
            'vanna': vanna,
            'volga': volga,
            'd1': d1,
            'd2': d2
        }

    def _calculate_d1_d2(self, S: float, K: float, T: float, vol: float,
                        r: float, q: float) -> Tuple[float, float]:
        """Calculate d1 and d2 for Black-Scholes formula"""
        if T <= 0:
            return 0.0, 0.0

        d1 = (np.log(S / K) + (r - q + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
        d2 = d1 - vol * np.sqrt(T)

        return d1, d2

    def _calculate_delta(self, d1: float, d2: float, option_type: str,
                        T: float, q: float) -> float:
        """Calculate delta (price sensitivity)"""
        if option_type.lower() == 'call':
            return np.exp(-q * T) * stats.norm.cdf(d1)
        else:
            return np.exp(-q * T) * (stats.norm.cdf(d1) - 1)

    def _calculate_gamma(self, S: float, d1: float, vol: float,
                        T: float, q: float) -> float:
        """Calculate gamma (delta sensitivity)"""
        if T <= 0 or S <= 0:
            return 0.0
        return (np.exp(-q * T) * stats.norm.pdf(d1)) / (S * vol * np.sqrt(T))

    def _calculate_theta(self, S: float, K: float, d1: float, d2: float,
                        vol: float, T: float, option_type: str,
                        r: float, q: float) -> float:
        """Calculate theta (time decay)"""
        if T <= 0:
            return 0.0

        sqrt_T = np.sqrt(T)

        # Common terms
        term1 = -(S * stats.norm.pdf(d1) * vol * np.exp(-q * T)) / (2 * sqrt_T)

        if option_type.lower() == 'call':
            term2 = q * S * stats.norm.cdf(d1) * np.exp(-q * T)
            term3 = -r * K * np.exp(-r * T) * stats.norm.cdf(d2)
        else:
            term2 = -q * S * stats.norm.cdf(-d1) * np.exp(-q * T)
            term3 = r * K * np.exp(-r * T) * stats.norm.cdf(-d2)

        # Convert to per-day theta
        return (term1 + term2 + term3) / 365.25

    def _calculate_vega(self, S: float, d1: float, T: float, q: float) -> float:
        """Calculate vega (volatility sensitivity)"""
        if T <= 0:
            return 0.0
        # Return vega per 1% volatility change
        return S * np.exp(-q * T) * stats.norm.pdf(d1) * np.sqrt(T) / 100

    def _calculate_rho(self, K: float, d2: float, T: float,
                      option_type: str, r: float) -> float:
        """Calculate rho (interest rate sensitivity)"""
        if T <= 0:
            return 0.0

        if option_type.lower() == 'call':
            # Return rho per 1% rate change
            return K * T * np.exp(-r * T) * stats.norm.cdf(d2) / 100
        else:
            return -K * T * np.exp(-r * T) * stats.norm.cdf(-d2) / 100

    def _calculate_epsilon(self, S: float, d1: float, T: float,
                          option_type: str, q: float) -> float:
        """Calculate epsilon (dividend sensitivity)"""
        if T <= 0:
            return 0.0

        if option_type.lower() == 'call':
            return -S * T * np.exp(-q * T) * stats.norm.cdf(d1) / 100
        else:
            return S * T * np.exp(-q * T) * stats.norm.cdf(-d1) / 100

    def _calculate_charm(self, S: float, d1: float, d2: float, vol: float,
                        T: float, option_type: str, r: float, q: float) -> float:
        """Calculate charm (delta decay over time)"""
        if T <= 0:
            return 0.0

        sqrt_T = np.sqrt(T)
        pdf_d1 = stats.norm.pdf(d1)

        if option_type.lower() == 'call':
            charm = q * np.exp(-q * T) * stats.norm.cdf(d1) - np.exp(-q * T) * pdf_d1 * (2 * (r - q) * T - d2 * vol * sqrt_T) / (2 * T * vol * sqrt_T)
        else:
            charm = -q * np.exp(-q * T) * stats.norm.cdf(-d1) - np.exp(-q * T) * pdf_d1 * (2 * (r - q) * T - d2 * vol * sqrt_T) / (2 * T * vol * sqrt_T)

        return charm / 365.25  # Per day

    def _calculate_vanna(self, S: float, d1: float, d2: float, vol: float,
                        T: float, q: float) -> float:
        """Calculate vanna (delta sensitivity to volatility)"""
        if T <= 0 or vol <= 0:
            return 0.0
        return -np.exp(-q * T) * stats.norm.pdf(d1) * d2 / vol / 100

    def _calculate_volga(self, S: float, d1: float, d2: float, vol: float,
                        T: float, q: float) -> float:
        """Calculate volga (vega sensitivity to volatility)"""
        if T <= 0 or vol <= 0:
            return 0.0
        vega_base = S * np.exp(-q * T) * stats.norm.pdf(d1) * np.sqrt(T)
        return vega_base * d1 * d2 / vol / 10000  # Per 1% vol change

class PortfolioRiskMetrics:
    """
    Advanced portfolio risk metrics and attribution
    """

    def __init__(self):
        self.cache = {}

    def calculate_portfolio_var(self,
                              weights: np.ndarray,
                              returns: pd.DataFrame,
                              confidence_level: float = 0.95,
                              method: str = 'historical') -> Dict[str, Union[float, np.ndarray]]:
        """
        Calculate portfolio VaR with component and marginal VaR

        Args:
            weights: Portfolio weights
            returns: Asset return matrix (time x assets)
            confidence_level: Confidence level for VaR
            method: VaR calculation method

        Returns:
            Portfolio VaR metrics including component and marginal VaR
        """
        # Calculate portfolio returns
        portfolio_returns = np.dot(returns.values, weights)

        # Portfolio VaR
        if method == 'historical':
            portfolio_var = -np.percentile(portfolio_returns, (1 - confidence_level) * 100)
        elif method == 'parametric':
            portfolio_var = -stats.norm.ppf(1 - confidence_level) * np.std(portfolio_returns)
        else:
            raise ValueError(f"Unsupported VaR method: {method}")

        # Component VaR calculation
        component_vars = self._calculate_component_var(weights, returns, confidence_level, method)

        # Marginal VaR calculation
        marginal_vars = self._calculate_marginal_var(weights, returns, confidence_level, method)

        # Incremental VaR
        incremental_vars = self._calculate_incremental_var(weights, returns, confidence_level, method)

        return {
            'portfolio_var': portfolio_var,
            'component_vars': component_vars,
            'marginal_vars': marginal_vars,
            'incremental_vars': incremental_vars,
            'diversification_ratio': np.sum(component_vars) / portfolio_var
        }

    def _calculate_component_var(self,
                               weights: np.ndarray,
                               returns: pd.DataFrame,
                               confidence_level: float,
                               method: str) -> np.ndarray:
        """Calculate component VaR for each asset"""
        n_assets = len(weights)
        component_vars = np.zeros(n_assets)

        # Calculate portfolio return series
        portfolio_returns = np.dot(returns.values, weights)

        if method == 'historical':
            portfolio_var = -np.percentile(portfolio_returns, (1 - confidence_level) * 100)

            # Find VaR scenarios
            var_threshold = -portfolio_var
            var_scenarios = portfolio_returns <= var_threshold

            if np.sum(var_scenarios) > 0:
                # Calculate average asset returns during VaR scenarios
                tail_returns = returns.values[var_scenarios]
                avg_tail_returns = np.mean(tail_returns, axis=0)

                # Component VaR = weight * avg_tail_return
                component_vars = weights * avg_tail_returns / np.mean(portfolio_returns[var_scenarios]) * portfolio_var

        elif method == 'parametric':
            # Analytical component VaR for normal distribution
            cov_matrix = np.cov(returns.values, rowvar=False)
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)

            # Marginal contribution to risk
            marginal_contributions = np.dot(cov_matrix, weights) / portfolio_volatility

            # Component VaR
            z_score = -stats.norm.ppf(1 - confidence_level)
            component_vars = weights * marginal_contributions * z_score

        return component_vars

    def _calculate_marginal_var(self,
                              weights: np.ndarray,
                              returns: pd.DataFrame,
                              confidence_level: float,
                              method: str) -> np.ndarray:
        """Calculate marginal VaR for each asset"""
        n_assets = len(weights)
        marginal_vars = np.zeros(n_assets)

        # Base portfolio VaR
        base_portfolio_returns = np.dot(returns.values, weights)
        if method == 'historical':
            base_var = -np.percentile(base_portfolio_returns, (1 - confidence_level) * 100)
        else:
            base_var = -stats.norm.ppf(1 - confidence_level) * np.std(base_portfolio_returns)

        # Small weight perturbation
        epsilon = 0.001

        for i in range(n_assets):
            # Increase weight by epsilon
            perturbed_weights = weights.copy()
            perturbed_weights[i] += epsilon
            perturbed_weights = perturbed_weights / np.sum(perturbed_weights)  # Renormalize

            # Calculate VaR with perturbed weights
            perturbed_returns = np.dot(returns.values, perturbed_weights)
            if method == 'historical':
                perturbed_var = -np.percentile(perturbed_returns, (1 - confidence_level) * 100)
            else:
                perturbed_var = -stats.norm.ppf(1 - confidence_level) * np.std(perturbed_returns)

            # Marginal VaR = dVaR/dw_i
            marginal_vars[i] = (perturbed_var - base_var) / epsilon

        return marginal_vars

    def _calculate_incremental_var(self,
                                 weights: np.ndarray,
                                 returns: pd.DataFrame,
                                 confidence_level: float,
                                 method: str) -> np.ndarray:
        """Calculate incremental VaR for each asset"""
        n_assets = len(weights)
        incremental_vars = np.zeros(n_assets)

        # Base portfolio VaR
        base_portfolio_returns = np.dot(returns.values, weights)
        if method == 'historical':
            base_var = -np.percentile(base_portfolio_returns, (1 - confidence_level) * 100)
        else:
            base_var = -stats.norm.ppf(1 - confidence_level) * np.std(base_portfolio_returns)

        for i in range(n_assets):
            # Remove asset from portfolio
            reduced_weights = weights.copy()
            reduced_weights[i] = 0

            if np.sum(reduced_weights) > 0:
                reduced_weights = reduced_weights / np.sum(reduced_weights)  # Renormalize

                # Calculate VaR without asset i
                reduced_returns = np.dot(returns.values, reduced_weights)
                if method == 'historical':
                    reduced_var = -np.percentile(reduced_returns, (1 - confidence_level) * 100)
                else:
                    reduced_var = -stats.norm.ppf(1 - confidence_level) * np.std(reduced_returns)

                # Incremental VaR = VaR_portfolio - VaR_without_asset_i
                incremental_vars[i] = base_var - reduced_var
            else:
                incremental_vars[i] = base_var

        return incremental_vars

    def calculate_risk_ratios(self,
                            returns: pd.Series,
                            benchmark_returns: pd.Series = None,
                            risk_free_rate: float = 0.02) -> Dict[str, float]:
        """
        Calculate comprehensive risk-adjusted performance ratios

        Args:
            returns: Portfolio return series
            benchmark_returns: Benchmark return series (optional)
            risk_free_rate: Risk-free rate (annualized)

        Returns:
            Dictionary of risk ratios
        """
        # Convert to excess returns
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate

        # Basic statistics
        mean_return = np.mean(returns) * 252  # Annualized
        volatility = np.std(returns) * np.sqrt(252)  # Annualized
        mean_excess = np.mean(excess_returns) * 252

        # Sharpe Ratio
        sharpe_ratio = mean_excess / volatility if volatility > 0 else 0

        # Sortino Ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = mean_excess / downside_deviation if downside_deviation > 0 else 0

        # Calmar Ratio (return/max drawdown)
        max_drawdown = self._calculate_maximum_drawdown(returns)
        calmar_ratio = mean_return / abs(max_drawdown) if abs(max_drawdown) > 0 else 0

        # VaR-based ratios
        var_95 = -np.percentile(returns, 5)
        var_ratio = mean_return / var_95 if var_95 > 0 else 0

        result = {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'var_ratio': var_ratio,
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'skewness': stats.skew(returns),
            'kurtosis': stats.kurtosis(returns, fisher=True)
        }

        # Benchmark-relative metrics
        if benchmark_returns is not None:
            tracking_error = np.std(returns - benchmark_returns) * np.sqrt(252)
            information_ratio = (mean_return - np.mean(benchmark_returns) * 252) / tracking_error if tracking_error > 0 else 0

            # Beta calculation
            covariance = np.cov(returns, benchmark_returns)[0, 1]
            benchmark_variance = np.var(benchmark_returns)
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 0

            # Treynor Ratio
            treynor_ratio = mean_excess / beta if beta > 0 else 0

            result.update({
                'information_ratio': information_ratio,
                'tracking_error': tracking_error,
                'beta': beta,
                'treynor_ratio': treynor_ratio
            })

        return result

    def _calculate_maximum_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown from return series"""
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        return drawdown.min()

# Integration and utilities
class RiskMetricsValidator:
    """
    Model validation and backtesting for risk metrics
    """

    @staticmethod
    def backtest_var(realized_returns: np.ndarray,
                    var_forecasts: np.ndarray,
                    confidence_level: float = 0.95) -> Dict[str, float]:
        """
        Backtest VaR model using Kupiec and Christoffersen tests

        Args:
            realized_returns: Actual returns
            var_forecasts: VaR forecasts (positive values)
            confidence_level: VaR confidence level

        Returns:
            Backtesting statistics and test results
        """
        # Violation indicator
        violations = realized_returns < -var_forecasts
        n_violations = np.sum(violations)
        n_observations = len(realized_returns)

        # Expected number of violations
        expected_violations = n_observations * (1 - confidence_level)

        # Violation rate
        violation_rate = n_violations / n_observations

        # Kupiec test (unconditional coverage)
        if n_violations > 0 and n_violations < n_observations:
            kupiec_stat = 2 * (n_violations * np.log(violation_rate / (1 - confidence_level)) +
                             (n_observations - n_violations) * np.log((1 - violation_rate) / confidence_level))
            kupiec_p_value = 1 - stats.chi2.cdf(kupiec_stat, df=1)
        else:
            kupiec_stat = float('inf')
            kupiec_p_value = 0.0

        # Christoffersen test (independence)
        # Transition matrix for violation sequences
        transitions = np.zeros((2, 2))
        for i in range(1, len(violations)):
            transitions[int(violations[i-1]), int(violations[i])] += 1

        # Independence test statistic
        if np.all(transitions.sum(axis=1) > 0):
            p01 = transitions[0, 1] / transitions[0, :].sum() if transitions[0, :].sum() > 0 else 0
            p11 = transitions[1, 1] / transitions[1, :].sum() if transitions[1, :].sum() > 0 else 0

            if p01 > 0 and p11 > 0 and p01 < 1 and p11 < 1:
                lr_ind = 2 * (transitions[0, 1] * np.log(p01) + transitions[1, 1] * np.log(p11) +
                             transitions[0, 0] * np.log(1 - p01) + transitions[1, 0] * np.log(1 - p11) -
                             n_violations * np.log(violation_rate) - (n_observations - n_violations) * np.log(1 - violation_rate))
                christoffersen_p_value = 1 - stats.chi2.cdf(lr_ind, df=1)
            else:
                christoffersen_p_value = 1.0
        else:
            christoffersen_p_value = 1.0

        return {
            'violation_rate': violation_rate,
            'expected_violation_rate': 1 - confidence_level,
            'n_violations': n_violations,
            'n_observations': n_observations,
            'kupiec_statistic': kupiec_stat,
            'kupiec_p_value': kupiec_p_value,
            'christoffersen_p_value': christoffersen_p_value,
            'model_valid': kupiec_p_value > 0.05 and christoffersen_p_value > 0.05
        }

# Example usage and integration
if __name__ == "__main__":
    # Example usage of the advanced risk metrics
    np.random.seed(42)

    # Generate sample data
    n_assets = 5
    n_observations = 1000

    # Simulate correlated asset returns
    correlation_matrix = np.random.uniform(0.2, 0.8, (n_assets, n_assets))
    correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
    np.fill_diagonal(correlation_matrix, 1.0)

    # Ensure positive definite
    eigenvals, eigenvecs = np.linalg.eigh(correlation_matrix)
    eigenvals = np.maximum(eigenvals, 0.01)
    correlation_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

    volatilities = np.random.uniform(0.15, 0.35, n_assets)
    covariance_matrix = np.outer(volatilities, volatilities) * correlation_matrix

    returns = np.random.multivariate_normal(
        mean=np.random.uniform(-0.001, 0.002, n_assets),
        cov=covariance_matrix / 252,  # Daily returns
        size=n_observations
    )

    returns_df = pd.DataFrame(returns, columns=[f'Asset_{i}' for i in range(n_assets)])

    # Portfolio weights
    weights = np.random.dirichlet(np.ones(n_assets))

    print("=== Advanced Risk Metrics Demo ===\n")

    # 1. Historical Simulation VaR
    print("1. Historical Simulation VaR")
    hist_var = HistoricalSimulationVaR()
    portfolio_returns = np.dot(returns, weights)
    hist_result = hist_var.calculate_var(portfolio_returns, confidence_level=0.95, bootstrap=True)
    print(f"   VaR (95%): {hist_result['var']:.4f}")
    print(f"   Expected Shortfall: {hist_result['expected_shortfall']:.4f}")
    print(f"   VaR Std Error: {hist_result['var_std_error']:.4f}\n")

    # 2. Parametric VaR
    print("2. Parametric VaR")
    param_var = ParametricVaR(distribution='cornish_fisher')
    param_result = param_var.calculate_var(portfolio_returns, confidence_level=0.95)
    print(f"   VaR (95%): {param_result['var']:.4f}")
    print(f"   Expected Shortfall: {param_result['expected_shortfall']:.4f}")
    print(f"   Distribution: {param_result['distribution']}\n")

    # 3. Monte Carlo VaR
    print("3. Monte Carlo VaR")
    mc_var = MonteCarloVaR(n_simulations=10000)
    expected_returns = np.mean(returns, axis=0)
    mc_result = mc_var.calculate_var(weights, expected_returns, covariance_matrix)
    print(f"   VaR (95%): {mc_result['var']:.4f}")
    print(f"   Expected Shortfall: {mc_result['expected_shortfall']:.4f}")
    print(f"   Portfolio Volatility: {mc_result['portfolio_volatility']:.4f}\n")

    # 4. GARCH Volatility Forecasting
    print("4. GARCH Volatility Forecasting")
    garch = GARCHVolatilityForecasting()
    try:
        garch_params = garch.fit(portfolio_returns)
        forecasts = garch.forecast(steps_ahead=5)
        print(f"   GARCH Parameters: ω={garch_params['omega']:.6f}, α={garch_params['alpha'][0]:.4f}, β={garch_params['beta'][0]:.4f}")
        print(f"   5-day vol forecast: {np.sqrt(forecasts)}\n")
    except Exception as e:
        print(f"   GARCH fitting failed: {e}\n")

    # 5. Options Greeks
    print("5. Options Greeks Calculation")
    greeks_calc = OptionsGreeksCalculator()
    greeks = greeks_calc.calculate_all_greeks(
        spot=100, strike=100, time_to_expiry=0.25, volatility=0.2
    )
    print(f"   Delta: {greeks['delta']:.4f}")
    print(f"   Gamma: {greeks['gamma']:.4f}")
    print(f"   Theta: {greeks['theta']:.4f}")
    print(f"   Vega: {greeks['vega']:.4f}")
    print(f"   Charm: {greeks['charm']:.4f}\n")

    # 6. Portfolio Risk Metrics
    print("6. Portfolio Risk Metrics")
    portfolio_risk = PortfolioRiskMetrics()
    portfolio_var_result = portfolio_risk.calculate_portfolio_var(
        weights, returns_df, confidence_level=0.95, method='historical'
    )
    print(f"   Portfolio VaR: {portfolio_var_result['portfolio_var']:.4f}")
    print(f"   Diversification Ratio: {portfolio_var_result['diversification_ratio']:.4f}")

    risk_ratios = portfolio_risk.calculate_risk_ratios(pd.Series(portfolio_returns))
    print(f"   Sharpe Ratio: {risk_ratios['sharpe_ratio']:.4f}")
    print(f"   Sortino Ratio: {risk_ratios['sortino_ratio']:.4f}")
    print(f"   Maximum Drawdown: {risk_ratios['max_drawdown']:.4f}\n")

    # 7. Model Validation
    print("7. VaR Model Backtesting")
    validator = RiskMetricsValidator()

    # Generate VaR forecasts for backtesting
    var_forecasts = np.full(len(portfolio_returns), hist_result['var'])
    backtest_result = validator.backtest_var(
        portfolio_returns, var_forecasts, confidence_level=0.95
    )
    print(f"   Violation Rate: {backtest_result['violation_rate']:.4f}")
    print(f"   Expected Rate: {backtest_result['expected_violation_rate']:.4f}")
    print(f"   Kupiec p-value: {backtest_result['kupiec_p_value']:.4f}")
    print(f"   Model Valid: {backtest_result['model_valid']}")

    print("\n=== Risk Metrics Calculation Complete ===")