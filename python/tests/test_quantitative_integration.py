"""
Comprehensive Integration Tests for Quantitative Risk Analytics Engine
Tests the entire quantitative workflow from data input to risk calculations
"""

import pytest
import numpy as np
import pandas as pd
import warnings
from unittest.mock import Mock, patch
from typing import Dict, List, Tuple
import logging

# Suppress warnings for cleaner test output
warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.ERROR)

# Import our modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from risk_engine.advanced_risk_metrics import (
    HistoricalSimulationVaR, ParametricVaR, MonteCarloVaR,
    GARCHVolatilityForecaster, BlackScholesGreeks, PortfolioRiskMetrics
)
from risk_engine.monte_carlo_engine import (
    MonteCarloEngine, PathGenerationConfig, VarianceReductionConfig
)
from optimization.advanced_portfolio_optimization import (
    MeanVarianceOptimizer, RiskParityOptimizer, BlackLittermanOptimizer,
    HierarchicalRiskParityOptimizer, OptimizationConfig, OptimizationConstraints
)

class TestQuantitativeIntegration:
    """Comprehensive integration tests for the entire quantitative engine"""

    @pytest.fixture
    def sample_market_data(self):
        """Generate realistic market data for testing"""
        np.random.seed(42)

        # 252 trading days, 10 assets
        n_days = 252
        n_assets = 10

        # Generate correlated returns
        base_returns = np.random.multivariate_normal(
            mean=np.full(n_assets, 0.0008),  # ~20% annual return
            cov=np.eye(n_assets) * 0.0004 + np.full((n_assets, n_assets), 0.0001),  # ~20% annual vol
            size=n_days
        )

        # Create price series
        prices = 100 * np.exp(np.cumsum(base_returns, axis=0))

        # Asset names and metadata
        asset_names = [f'ASSET_{i+1:02d}' for i in range(n_assets)]
        dates = pd.date_range('2023-01-01', periods=n_days, freq='D')

        return {
            'returns': base_returns,
            'prices': prices,
            'asset_names': asset_names,
            'dates': dates,
            'current_prices': prices[-1, :],
            'n_assets': n_assets,
            'n_days': n_days
        }

    @pytest.fixture
    def sample_portfolio(self, sample_market_data):
        """Generate sample portfolio positions"""
        n_assets = sample_market_data['n_assets']
        current_prices = sample_market_data['current_prices']

        # Random position sizes
        np.random.seed(123)
        quantities = np.random.randint(100, 10000, n_assets)
        market_values = quantities * current_prices
        weights = market_values / np.sum(market_values)

        return {
            'quantities': quantities,
            'market_values': market_values,
            'weights': weights,
            'total_value': np.sum(market_values)
        }

    def test_risk_metrics_calculation_workflow(self, sample_market_data, sample_portfolio):
        """Test complete risk metrics calculation workflow"""
        returns = sample_market_data['returns']
        weights = sample_portfolio['weights']
        portfolio_value = sample_portfolio['total_value']

        # 1. Historical Simulation VaR
        hist_var = HistoricalSimulationVaR(lookback_days=60)
        portfolio_returns = returns @ weights

        var_results = hist_var.calculate_var(portfolio_returns, confidence_level=0.95)

        # Validate VaR results
        assert 'var' in var_results
        assert 'expected_shortfall' in var_results
        assert var_results['var'] > 0
        assert var_results['expected_shortfall'] >= var_results['var']

        # 2. Parametric VaR
        param_var = ParametricVaR()
        covariance_matrix = np.cov(returns.T)

        parametric_results = param_var.calculate_var(
            weights, covariance_matrix, confidence_level=0.95
        )

        assert 'var' in parametric_results
        assert 'component_var' in parametric_results
        assert len(parametric_results['component_var']) == len(weights)

        # 3. Monte Carlo VaR
        mc_var = MonteCarloVaR(n_simulations=5000)
        mc_results = mc_var.calculate_var(
            weights, np.mean(returns, axis=0), covariance_matrix, confidence_level=0.95
        )

        assert 'var' in mc_results
        assert 'expected_shortfall' in mc_results
        assert mc_results['var'] > 0

        # 4. GARCH Volatility Forecasting
        garch_forecaster = GARCHVolatilityForecaster()
        vol_forecast = garch_forecaster.forecast_volatility(portfolio_returns, horizon=5)

        assert 'forecasted_volatility' in vol_forecast
        assert len(vol_forecast['forecasted_volatility']) == 5
        assert all(vol > 0 for vol in vol_forecast['forecasted_volatility'])

        # 5. Portfolio Risk Metrics
        portfolio_metrics = PortfolioRiskMetrics()

        # Test component VaR
        component_var = portfolio_metrics.calculate_component_var(
            weights, covariance_matrix, confidence_level=0.95
        )

        assert len(component_var) == len(weights)
        assert np.isclose(np.sum(component_var), parametric_results['var'], rtol=0.01)

        # Test marginal VaR
        marginal_var = portfolio_metrics.calculate_marginal_var(
            weights, covariance_matrix, confidence_level=0.95
        )

        assert len(marginal_var) == len(weights)

        print(f"✓ Risk Metrics Workflow Complete")
        print(f"  Historical VaR: ${var_results['var'] * portfolio_value:,.2f}")
        print(f"  Parametric VaR: ${parametric_results['var'] * portfolio_value:,.2f}")
        print(f"  Monte Carlo VaR: ${mc_results['var'] * portfolio_value:,.2f}")
        print(f"  GARCH Vol Forecast (1-day): {vol_forecast['forecasted_volatility'][0]:.4f}")

    def test_options_greeks_calculation(self, sample_market_data):
        """Test options Greeks calculation workflow"""
        current_prices = sample_market_data['current_prices']
        returns = sample_market_data['returns']

        # Calculate implied volatilities from historical data
        volatilities = np.std(returns, axis=0) * np.sqrt(252)

        # Sample options portfolio
        options_data = []
        for i in range(5):  # 5 different options
            options_data.append({
                'underlying_price': current_prices[i],
                'strike_price': current_prices[i] * np.random.uniform(0.9, 1.1),
                'time_to_expiry': np.random.uniform(0.1, 1.0),
                'volatility': volatilities[i],
                'risk_free_rate': 0.05,
                'dividend_yield': 0.02,
                'option_type': 'call' if np.random.rand() > 0.5 else 'put'
            })

        # Calculate Greeks
        greeks_calculator = BlackScholesGreeks()

        portfolio_greeks = {
            'delta': 0,
            'gamma': 0,
            'theta': 0,
            'vega': 0,
            'rho': 0
        }

        for option in options_data:
            greeks = greeks_calculator.calculate_greeks(
                S=option['underlying_price'],
                K=option['strike_price'],
                T=option['time_to_expiry'],
                r=option['risk_free_rate'],
                sigma=option['volatility'],
                q=option['dividend_yield'],
                option_type=option['option_type']
            )

            # Aggregate portfolio Greeks
            for greek in portfolio_greeks:
                portfolio_greeks[greek] += greeks[greek]

            # Validate individual option Greeks
            assert isinstance(greeks['delta'], float)
            assert isinstance(greeks['gamma'], float)
            assert greeks['gamma'] >= 0  # Gamma is always non-negative
            assert isinstance(greeks['theta'], float)
            assert isinstance(greeks['vega'], float)
            assert greeks['vega'] >= 0  # Vega is always non-negative

        print(f"✓ Options Greeks Calculation Complete")
        print(f"  Portfolio Delta: {portfolio_greeks['delta']:.4f}")
        print(f"  Portfolio Gamma: {portfolio_greeks['gamma']:.4f}")
        print(f"  Portfolio Theta: {portfolio_greeks['theta']:.4f}")
        print(f"  Portfolio Vega: {portfolio_greeks['vega']:.4f}")

    def test_monte_carlo_simulation_workflow(self, sample_market_data, sample_portfolio):
        """Test Monte Carlo simulation workflow"""
        returns = sample_market_data['returns']
        current_prices = sample_market_data['current_prices']
        weights = sample_portfolio['weights']

        # Setup Monte Carlo engine
        path_config = PathGenerationConfig(
            n_paths=10000,
            n_steps=21,  # 21 trading days (1 month)
            use_antithetic=True,
            use_moment_matching=True
        )

        variance_config = VarianceReductionConfig(
            use_control_variates=True,
            use_importance_sampling=False,  # Skip for integration test
            control_variate_beta=0.5
        )

        mc_engine = MonteCarloEngine(path_config, variance_config)

        # 1. Test Geometric Brownian Motion simulation
        mu = np.mean(returns, axis=0) * 252  # Annualized returns
        sigma = np.std(returns, axis=0) * np.sqrt(252)  # Annualized volatilities
        correlation_matrix = np.corrcoef(returns.T)

        # Generate price paths
        price_paths = mc_engine.geometric_brownian_motion(
            S0=current_prices,
            mu=mu,
            sigma=sigma,
            T=1/12,  # 1 month
            n_steps=21,
            correlation_matrix=correlation_matrix
        )

        assert price_paths.shape == (path_config.n_paths, path_config.n_steps + 1, len(current_prices))
        assert np.all(price_paths > 0)  # All prices should be positive

        # 2. Test portfolio simulation
        try:
            portfolio_values = mc_engine.portfolio_monte_carlo_gpu(
                weights=weights,
                S0=current_prices,
                mu=mu,
                sigma=sigma,
                correlation_matrix=correlation_matrix,
                T=1/12
            )
        except:
            # Fallback to CPU version if GPU not available
            portfolio_values = mc_engine.portfolio_monte_carlo_cpu(
                weights=weights,
                S0=current_prices,
                mu=mu,
                sigma=sigma,
                correlation_matrix=correlation_matrix,
                T=1/12
            )

        assert len(portfolio_values) == path_config.n_paths
        assert np.all(portfolio_values > 0)

        # 3. Test European option pricing
        option_strike = current_prices[0] * 1.05
        option_prices = mc_engine.price_european_option(
            S0=current_prices[0],
            K=option_strike,
            T=0.25,  # 3 months
            r=0.05,
            sigma=sigma[0],
            option_type='call'
        )

        assert 'option_price' in option_prices
        assert 'confidence_interval' in option_prices
        assert option_prices['option_price'] >= 0

        print(f"✓ Monte Carlo Simulation Workflow Complete")
        print(f"  Price Paths Generated: {price_paths.shape[0]:,}")
        print(f"  Portfolio Simulation Complete: {len(portfolio_values):,} paths")
        print(f"  Option Price (Call): ${option_prices['option_price']:.4f}")
        print(f"  Option Price CI: ${option_prices['confidence_interval'][0]:.4f} - ${option_prices['confidence_interval'][1]:.4f}")

    def test_portfolio_optimization_workflow(self, sample_market_data):
        """Test portfolio optimization workflow"""
        returns = sample_market_data['returns']
        asset_names = sample_market_data['asset_names']
        n_assets = sample_market_data['n_assets']

        # Calculate inputs for optimization
        expected_returns = np.mean(returns, axis=0) * 252  # Annualized
        covariance_matrix = np.cov(returns.T) * 252  # Annualized

        # Optimization configuration
        config = OptimizationConfig(
            risk_aversion=3.0,
            max_iterations=1000,
            tolerance=1e-8
        )

        constraints = OptimizationConstraints(
            min_weights=np.full(n_assets, 0.01),  # Min 1%
            max_weights=np.full(n_assets, 0.30),  # Max 30%
            no_short_selling=True
        )

        optimization_results = {}

        # 1. Mean-Variance Optimization
        mv_optimizer = MeanVarianceOptimizer(config)
        mv_result = mv_optimizer.optimize(expected_returns, covariance_matrix, constraints)

        assert mv_result['optimization_status'] != 'failed'
        assert np.isclose(np.sum(mv_result['weights']), 1.0, atol=1e-6)
        assert np.all(mv_result['weights'] >= constraints.min_weights - 1e-6)
        assert np.all(mv_result['weights'] <= constraints.max_weights + 1e-6)

        optimization_results['Mean-Variance'] = mv_result

        # 2. Risk Parity Optimization
        rp_optimizer = RiskParityOptimizer(config)
        rp_result = rp_optimizer.optimize(expected_returns, covariance_matrix, constraints)

        assert rp_result['optimization_status'] != 'failed'
        assert np.isclose(np.sum(rp_result['weights']), 1.0, atol=1e-6)

        optimization_results['Risk Parity'] = rp_result

        # 3. Black-Litterman Optimization
        bl_optimizer = BlackLittermanOptimizer(config, tau=0.025)

        # Create investor views
        views = [
            {'type': 'absolute', 'asset': asset_names[0], 'return': 0.25},
            {'type': 'relative', 'asset1': asset_names[1], 'asset2': asset_names[2], 'return': 0.05}
        ]

        views_matrix, views_returns = bl_optimizer.create_views_matrix(asset_names, views)
        market_caps = np.random.uniform(0.5, 2.0, n_assets)

        bl_result = bl_optimizer.optimize(
            expected_returns, covariance_matrix, constraints,
            market_caps=market_caps, views_matrix=views_matrix, views_returns=views_returns
        )

        assert bl_result['optimization_status'] != 'failed'
        assert np.isclose(np.sum(bl_result['weights']), 1.0, atol=1e-6)
        assert bl_result['views_incorporated'] == True

        optimization_results['Black-Litterman'] = bl_result

        # 4. Hierarchical Risk Parity
        hrp_optimizer = HierarchicalRiskParityOptimizer(config)
        hrp_result = hrp_optimizer.optimize(expected_returns, covariance_matrix)

        assert hrp_result['optimization_status'] == 'optimal'
        assert np.isclose(np.sum(hrp_result['weights']), 1.0, atol=1e-6)
        assert np.all(hrp_result['weights'] >= 0)  # No short positions

        optimization_results['HRP'] = hrp_result

        # 5. Test efficient frontier generation
        frontier_data = mv_optimizer.efficient_frontier(
            expected_returns, covariance_matrix, constraints, num_points=10
        )

        assert len(frontier_data['returns']) > 0
        assert len(frontier_data['risks']) == len(frontier_data['returns'])
        assert np.all(np.diff(frontier_data['risks']) >= -1e-6)  # Risk should be non-decreasing

        print(f"✓ Portfolio Optimization Workflow Complete")
        for method, result in optimization_results.items():
            print(f"  {method:<20}: Return={result['expected_return']:.4f}, Risk={result['expected_risk']:.4f}, Sharpe={result['sharpe_ratio']:.4f}")

        return optimization_results

    def test_stress_testing_workflow(self, sample_market_data, sample_portfolio):
        """Test stress testing workflow"""
        returns = sample_market_data['returns']
        weights = sample_portfolio['weights']
        current_prices = sample_market_data['current_prices']

        # Define stress scenarios
        stress_scenarios = {
            'Market Crash': {
                'equity_shock': -0.30,
                'volatility_spike': 2.0,
                'correlation_increase': 0.20
            },
            'Interest Rate Shock': {
                'rate_increase': 0.02,
                'duration_effect': -0.10
            },
            'Sector Rotation': {
                'tech_underperform': -0.15,
                'defensive_outperform': 0.10
            }
        }

        stress_results = {}

        for scenario_name, scenario_params in stress_scenarios.items():
            # Simulate stressed returns
            stressed_returns = returns.copy()

            if 'equity_shock' in scenario_params:
                # Apply equity shock to all assets
                shock_magnitude = scenario_params['equity_shock']
                stressed_returns[-1, :] += shock_magnitude

            if 'volatility_spike' in scenario_params:
                # Increase volatility
                vol_multiplier = scenario_params['volatility_spike']
                stressed_returns *= vol_multiplier

            # Calculate portfolio impact
            portfolio_returns = stressed_returns @ weights

            # Calculate stress VaR
            hist_var = HistoricalSimulationVaR(lookback_days=30)
            stress_var = hist_var.calculate_var(portfolio_returns, confidence_level=0.95)

            # Calculate scenario-specific portfolio return
            scenario_return = np.sum(stressed_returns[-1, :] * weights)

            stress_results[scenario_name] = {
                'scenario_return': scenario_return,
                'stress_var': stress_var['var'],
                'stress_es': stress_var['expected_shortfall']
            }

        # Validate stress test results
        for scenario_name, results in stress_results.items():
            assert isinstance(results['scenario_return'], float)
            assert isinstance(results['stress_var'], float)
            assert isinstance(results['stress_es'], float)
            assert results['stress_var'] > 0
            assert results['stress_es'] >= results['stress_var']

        print(f"✓ Stress Testing Workflow Complete")
        for scenario, results in stress_results.items():
            print(f"  {scenario:<20}: Return={results['scenario_return']:.4f}, VaR={results['stress_var']:.4f}")

        return stress_results

    def test_model_validation_workflow(self, sample_market_data, sample_portfolio):
        """Test model validation and backtesting workflow"""
        returns = sample_market_data['returns']
        weights = sample_portfolio['weights']

        # Portfolio returns
        portfolio_returns = returns @ weights

        # Split data for backtesting
        train_size = int(len(portfolio_returns) * 0.8)
        train_returns = portfolio_returns[:train_size]
        test_returns = portfolio_returns[train_size:]

        # Backtest VaR models
        var_models = {
            'Historical': HistoricalSimulationVaR(lookback_days=60),
            'Parametric': ParametricVaR(),
            'Monte Carlo': MonteCarloVaR(n_simulations=5000)
        }

        validation_results = {}

        for model_name, model in var_models.items():
            if model_name == 'Historical':
                var_estimates = []
                for i in range(len(test_returns)):
                    # Rolling window VaR estimation
                    if i + 60 < len(train_returns):
                        train_subset = train_returns[i:i+60]
                        var_result = model.calculate_var(train_subset, confidence_level=0.95)
                        var_estimates.append(var_result['var'])
                    else:
                        var_estimates.append(var_estimates[-1] if var_estimates else 0.02)

            elif model_name == 'Parametric':
                # Use rolling covariance matrix
                covariance_matrix = np.cov(returns[:train_size].T)
                var_result = model.calculate_var(weights, covariance_matrix, confidence_level=0.95)
                var_estimates = [var_result['var']] * len(test_returns)

            else:  # Monte Carlo
                mu = np.mean(train_returns)
                sigma = np.std(train_returns)
                var_result = model.calculate_var(
                    weights=np.array([1.0]),  # Single portfolio
                    expected_returns=np.array([mu]),
                    covariance_matrix=np.array([[sigma**2]]),
                    confidence_level=0.95
                )
                var_estimates = [var_result['var']] * len(test_returns)

            # Validate estimates
            var_estimates = np.array(var_estimates[:len(test_returns)])

            # Kupiec Test (simplified)
            violations = np.sum(test_returns < -var_estimates)
            expected_violations = len(test_returns) * 0.05  # 5% for 95% confidence
            kupiec_stat = 2 * (violations * np.log(violations / expected_violations) if violations > 0 else 0)

            validation_results[model_name] = {
                'violations': violations,
                'expected_violations': expected_violations,
                'violation_rate': violations / len(test_returns),
                'kupiec_statistic': kupiec_stat,
                'mean_var_estimate': np.mean(var_estimates)
            }

        # Validate results
        for model_name, results in validation_results.items():
            assert isinstance(results['violations'], (int, np.integer))
            assert isinstance(results['violation_rate'], float)
            assert 0 <= results['violation_rate'] <= 1
            assert results['mean_var_estimate'] > 0

        print(f"✓ Model Validation Workflow Complete")
        for model, results in validation_results.items():
            print(f"  {model:<15}: Violations={results['violations']}/{len(test_returns)} ({results['violation_rate']:.3f}), Mean VaR={results['mean_var_estimate']:.4f}")

        return validation_results

    def test_end_to_end_workflow(self, sample_market_data, sample_portfolio):
        """Test complete end-to-end quantitative workflow"""
        print("\n" + "="*60)
        print("QUANTITATIVE RISK ANALYTICS ENGINE - INTEGRATION TEST")
        print("="*60)

        # 1. Risk Metrics Calculation
        print("\n1. RISK METRICS CALCULATION")
        print("-" * 30)
        self.test_risk_metrics_calculation_workflow(sample_market_data, sample_portfolio)

        # 2. Options Greeks
        print("\n2. OPTIONS GREEKS CALCULATION")
        print("-" * 30)
        self.test_options_greeks_calculation(sample_market_data)

        # 3. Monte Carlo Simulations
        print("\n3. MONTE CARLO SIMULATIONS")
        print("-" * 30)
        self.test_monte_carlo_simulation_workflow(sample_market_data, sample_portfolio)

        # 4. Portfolio Optimization
        print("\n4. PORTFOLIO OPTIMIZATION")
        print("-" * 30)
        optimization_results = self.test_portfolio_optimization_workflow(sample_market_data)

        # 5. Stress Testing
        print("\n5. STRESS TESTING")
        print("-" * 30)
        stress_results = self.test_stress_testing_workflow(sample_market_data, sample_portfolio)

        # 6. Model Validation
        print("\n6. MODEL VALIDATION")
        print("-" * 30)
        validation_results = self.test_model_validation_workflow(sample_market_data, sample_portfolio)

        # Summary
        print("\n" + "="*60)
        print("INTEGRATION TEST SUMMARY")
        print("="*60)
        print("✓ All quantitative modules integrated successfully")
        print("✓ Risk metrics calculated across multiple methodologies")
        print("✓ Options Greeks computed for complex portfolios")
        print("✓ Monte Carlo simulations with variance reduction")
        print("✓ Multi-algorithm portfolio optimization completed")
        print("✓ Comprehensive stress testing scenarios executed")
        print("✓ Model validation and backtesting performed")

        # Performance Summary
        n_assets = sample_market_data['n_assets']
        portfolio_value = sample_portfolio['total_value']

        print(f"\nPERFORMANCE METRICS:")
        print(f"  Assets Processed: {n_assets}")
        print(f"  Portfolio Value: ${portfolio_value:,.2f}")
        print(f"  Historical Data Points: {sample_market_data['n_days']}")
        print(f"  Optimization Methods: {len(optimization_results)}")
        print(f"  Stress Scenarios: {len(stress_results)}")
        print(f"  Validation Models: {len(validation_results)}")

        return {
            'optimization_results': optimization_results,
            'stress_results': stress_results,
            'validation_results': validation_results,
            'summary': {
                'n_assets': n_assets,
                'portfolio_value': portfolio_value,
                'data_points': sample_market_data['n_days'],
                'status': 'SUCCESS'
            }
        }

# Standalone test execution
if __name__ == "__main__":
    # Initialize test class
    test_suite = TestQuantitativeIntegration()

    # Generate test data
    market_data = test_suite.sample_market_data()
    portfolio_data = test_suite.sample_portfolio(market_data)

    # Run end-to-end test
    results = test_suite.test_end_to_end_workflow(market_data, portfolio_data)

    print(f"\n🎉 Integration Test Completed Successfully!")
    print(f"Status: {results['summary']['status']}")