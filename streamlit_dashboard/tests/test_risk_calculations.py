"""
Unit Tests for Risk Calculations
================================

Comprehensive unit tests for all risk calculation functions including VaR,
Greeks, stress testing, and Monte Carlo simulations.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from risk_engine import RiskEngine, PortfolioPosition, RiskMetrics
from model_validation import VaRBacktester, GreeksValidator, StressTestValidator, MonteCarloValidator
from backtesting_framework import BacktestEngine, RiskParityStrategy, BacktestConfig
from data_quality import DataQualityValidator, MissingValueCheck, OutlierDetectionCheck

class TestRiskEngine(unittest.TestCase):
    """Test cases for RiskEngine class"""

    def setUp(self):
        """Set up test fixtures"""
        self.risk_engine = RiskEngine()
        self.sample_positions = self._create_sample_positions()

    def _create_sample_positions(self):
        """Create sample portfolio positions for testing"""
        return [
            PortfolioPosition("AAPL", 1000, 150000, "Equity", "Technology", "North America", "USD",
                            delta=1000, gamma=0, theta=0, vega=0, rho=0),
            PortfolioPosition("MSFT", 800, 240000, "Equity", "Technology", "North America", "USD",
                            delta=800, gamma=0, theta=0, vega=0, rho=0),
            PortfolioPosition("US10Y", 1000, 500000, "Fixed Income", "Government", "North America", "USD",
                            delta=0, gamma=0, theta=0, vega=0, rho=-25000),
            PortfolioPosition("SPY_CALL", 100, 50000, "Derivatives", "Equity Options", "North America", "USD",
                            delta=50, gamma=0.1, theta=-25, vega=100, rho=10)
        ]

    def test_var_calculation_methods(self):
        """Test VaR calculation using different methods"""
        var_results = self.risk_engine.calculate_portfolio_var(self.sample_positions)

        # Check that all methods return results
        self.assertIn('historical_var', var_results)
        self.assertIn('parametric_var', var_results)
        self.assertIn('monte_carlo_var', var_results)
        self.assertIn('diversified_var', var_results)

        # Check that results are reasonable (positive values)
        for method, var_value in var_results.items():
            self.assertGreater(var_value, 0, f"{method} should return positive VaR")
            self.assertLess(var_value, 100000000, f"{method} VaR seems unreasonably high")

    def test_expected_shortfall_calculation(self):
        """Test Expected Shortfall calculation"""
        es = self.risk_engine.calculate_expected_shortfall(self.sample_positions)

        self.assertIsInstance(es, float)
        self.assertGreater(es, 0)

        # ES should be greater than VaR
        var_95 = self.risk_engine.calculate_portfolio_var(self.sample_positions)['diversified_var']
        self.assertGreater(es, var_95, "Expected Shortfall should be greater than VaR")

    def test_greeks_calculation(self):
        """Test Greeks calculation for options positions"""
        greeks = self.risk_engine.calculate_greeks(self.sample_positions)

        # Check structure
        self.assertIn('total', greeks)
        self.assertIn('by_asset_class', greeks)
        self.assertIn('by_sector', greeks)

        # Check that Greeks are calculated for derivatives
        total_greeks = greeks['total']
        self.assertIn('delta', total_greeks)
        self.assertIn('gamma', total_greeks)
        self.assertIn('theta', total_greeks)
        self.assertIn('vega', total_greeks)
        self.assertIn('rho', total_greeks)

        # Verify Greeks values are reasonable
        self.assertEqual(total_greeks['delta'], 50, "Delta should match options position")
        self.assertEqual(total_greeks['gamma'], 0.1, "Gamma should match options position")

    def test_stress_testing(self):
        """Test stress testing scenarios"""
        stress_results = self.risk_engine.run_stress_tests(self.sample_positions)

        # Check that all standard scenarios are included
        expected_scenarios = ['market_crash', 'interest_rate_shock', 'fx_crisis', 'credit_crisis', 'liquidity_crisis']

        for scenario in expected_scenarios:
            self.assertIn(scenario, stress_results, f"Missing stress scenario: {scenario}")

        # Stress test results should be negative (losses)
        for scenario, result in stress_results.items():
            self.assertLessEqual(result, 0, f"Stress test {scenario} should result in losses")

    def test_risk_contributions(self):
        """Test risk contribution calculations"""
        contributions = self.risk_engine.calculate_risk_contributions(self.sample_positions)

        self.assertIn('marginal_var', contributions)
        self.assertIn('component_var', contributions)
        self.assertIn('percentage_contribution', contributions)

        # Check that contributions sum approximately to total VaR
        total_component_var = sum(contributions['component_var'].values())
        total_var = self.risk_engine.calculate_portfolio_var(self.sample_positions)['diversified_var']

        # Allow some tolerance due to approximation methods
        self.assertAlmostEqual(total_component_var, total_var, delta=total_var * 0.1)

    def test_risk_metrics_summary(self):
        """Test comprehensive risk metrics calculation"""
        risk_metrics = self.risk_engine.calculate_risk_metrics_summary(self.sample_positions)

        self.assertIsInstance(risk_metrics, RiskMetrics)

        # Check that all metrics are calculated
        self.assertGreater(risk_metrics.var_1d, 0)
        self.assertGreater(risk_metrics.var_10d, risk_metrics.var_1d)
        self.assertGreater(risk_metrics.expected_shortfall, 0)
        self.assertGreater(risk_metrics.volatility, 0)
        self.assertNotEqual(risk_metrics.beta, 0)

    def test_correlation_matrix_generation(self):
        """Test correlation matrix generation"""
        correlation_matrix = self.risk_engine._generate_correlation_matrix(self.sample_positions)

        n = len(self.sample_positions)
        self.assertEqual(correlation_matrix.shape, (n, n))

        # Check diagonal elements are 1
        np.testing.assert_array_equal(np.diag(correlation_matrix), np.ones(n))

        # Check matrix is symmetric
        np.testing.assert_array_equal(correlation_matrix, correlation_matrix.T)

        # Check correlation values are between -1 and 1
        self.assertTrue(np.all(correlation_matrix >= -1))
        self.assertTrue(np.all(correlation_matrix <= 1))

class TestVaRBacktesting(unittest.TestCase):
    """Test cases for VaR backtesting"""

    def setUp(self):
        """Set up test fixtures"""
        self.backtester = VaRBacktester()

        # Generate synthetic data for testing
        np.random.seed(42)
        self.returns = np.random.normal(0, 0.02, 250)  # 250 days of returns
        self.var_forecasts = np.abs(np.random.normal(0.03, 0.01, 250))  # VaR forecasts

    def test_kupiec_test(self):
        """Test Kupiec POF test"""
        # Test with expected number of violations
        violations = 12  # Approximately 5% of 250
        observations = 250
        confidence_level = 0.95

        result = self.backtester.kupiec_test(violations, observations, confidence_level)

        self.assertEqual(result.test_name, "Kupiec POF Test")
        self.assertIsNotNone(result.p_value)
        self.assertIsNotNone(result.statistic)
        self.assertIn('violations', result.details)
        self.assertIn('violation_rate', result.details)

    def test_christoffersen_test(self):
        """Test Christoffersen independence test"""
        # Create violations array with some clustering
        violations = np.zeros(100)
        violations[10:15] = 1  # Cluster of violations
        violations[50] = 1
        violations[80:82] = 1

        result = self.backtester.christoffersen_test(violations)

        self.assertEqual(result.test_name, "Christoffersen Independence Test")
        self.assertIsNotNone(result.p_value)
        self.assertIn('transitions', result.details)

    def test_traffic_light_test(self):
        """Test Basel III traffic light test"""
        # Test green zone
        result_green = self.backtester.traffic_light_test(3)
        self.assertEqual(result_green.details['zone'], 'green')

        # Test yellow zone
        result_yellow = self.backtester.traffic_light_test(7)
        self.assertEqual(result_yellow.details['zone'], 'yellow')

        # Test red zone
        result_red = self.backtester.traffic_light_test(12)
        self.assertEqual(result_red.details['zone'], 'red')

    def test_var_model_validation(self):
        """Test comprehensive VaR model validation"""
        backtest_result = self.backtester.validate_var_model(
            self.returns, self.var_forecasts
        )

        self.assertIsNotNone(backtest_result.model_name)
        self.assertGreater(len(backtest_result.test_results), 0)
        self.assertIn('hit_rate', backtest_result.performance_metrics)
        self.assertIn('mean_absolute_error', backtest_result.performance_metrics)

class TestGreeksValidation(unittest.TestCase):
    """Test cases for Greeks validation"""

    def setUp(self):
        """Set up test fixtures"""
        self.validator = GreeksValidator()

        # Generate synthetic Greeks data
        np.random.seed(42)
        self.theoretical_deltas = np.random.uniform(0.3, 0.8, 100)
        self.market_deltas = self.theoretical_deltas + np.random.normal(0, 0.02, 100)

    def test_delta_accuracy_validation(self):
        """Test delta accuracy validation"""
        result = self.validator.validate_delta_accuracy(
            self.theoretical_deltas, self.market_deltas
        )

        self.assertEqual(result.test_name, "Delta Accuracy Test")
        self.assertIsNotNone(result.p_value)
        self.assertIn('mean_absolute_error', result.details)
        self.assertIn('within_tolerance_rate', result.details)

    def test_gamma_convexity_validation(self):
        """Test gamma convexity validation"""
        # Generate synthetic price and option data
        spot_prices = np.array([100, 101, 102, 103, 104])
        theoretical_gammas = np.array([0.05, 0.048, 0.046, 0.044, 0.042])
        market_prices = np.array([5.0, 5.8, 6.7, 7.7, 8.8])

        result = self.validator.validate_gamma_convexity(
            spot_prices, theoretical_gammas, market_prices
        )

        self.assertEqual(result.test_name, "Gamma Convexity Test")

class TestStressTestValidation(unittest.TestCase):
    """Test cases for stress test validation"""

    def setUp(self):
        """Set up test fixtures"""
        self.validator = StressTestValidator()

        # Sample stress test results
        self.scenario_results = {
            'market_crash': -50.0,
            'interest_rate_shock': -15.0,
            'fx_crisis': -8.0,
            'credit_crisis': -12.0,
            'liquidity_crisis': -5.0
        }

        # Sample historical data
        np.random.seed(42)
        self.historical_data = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, 1000)))

    def test_scenario_plausibility(self):
        """Test scenario plausibility validation"""
        result = self.validator.validate_scenario_plausibility(
            self.scenario_results, self.historical_data
        )

        self.assertEqual(result.test_name, "Scenario Plausibility Test")
        self.assertIn('historical_volatility', result.details)
        self.assertIn('extreme_percentiles', result.details)

    def test_stress_test_coverage(self):
        """Test stress test coverage validation"""
        portfolio_exposures = {
            'equity': 0.6,
            'fixed_income': 0.3,
            'alternatives': 0.1
        }

        result = self.validator.validate_stress_test_coverage(
            self.scenario_results, portfolio_exposures
        )

        self.assertEqual(result.test_name, "Stress Test Coverage Analysis")
        self.assertIn('coverage_rate', result.details)
        self.assertIn('covered_categories', result.details)

class TestMonteCarloValidation(unittest.TestCase):
    """Test cases for Monte Carlo validation"""

    def setUp(self):
        """Set up test fixtures"""
        self.validator = MonteCarloValidator()

        # Generate synthetic Monte Carlo results
        np.random.seed(42)
        self.mc_results = np.random.normal(-2, 15, 50000)

    def test_convergence_testing(self):
        """Test Monte Carlo convergence"""
        result = self.validator.test_convergence(self.mc_results)

        self.assertEqual(result.test_name, "Monte Carlo Convergence Test")
        self.assertIn('coefficient_of_variation', result.details)
        self.assertIn('converged', result.details)

    def test_distribution_properties(self):
        """Test Monte Carlo distribution properties"""
        result = self.validator.test_distribution_properties(self.mc_results)

        self.assertEqual(result.test_name, "Distribution Properties Test")
        self.assertIn('mean', result.details)
        self.assertIn('standard_deviation', result.details)
        self.assertIn('skewness', result.details)
        self.assertIn('kurtosis', result.details)

class TestBacktestingFramework(unittest.TestCase):
    """Test cases for backtesting framework"""

    def setUp(self):
        """Set up test fixtures"""
        self.backtest_engine = BacktestEngine(data_provider=None)
        self.strategy = RiskParityStrategy()
        self.backtest_engine.register_strategy(self.strategy)

        self.config = BacktestConfig(
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2023, 1, 1),
            lookback_window=63,
            out_of_sample_ratio=0.2
        )

    def test_strategy_registration(self):
        """Test strategy registration"""
        self.assertIn(self.strategy.name, self.backtest_engine.strategies)

    def test_performance_metrics_calculation(self):
        """Test performance metrics calculation"""
        # Generate synthetic returns
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))

        metrics = self.backtest_engine._calculate_performance_metrics(returns, 0.02)

        self.assertIsNotNone(metrics.total_return)
        self.assertIsNotNone(metrics.sharpe_ratio)
        self.assertIsNotNone(metrics.max_drawdown)
        self.assertIsNotNone(metrics.volatility)

    def test_split_date_calculation(self):
        """Test in-sample/out-of-sample split calculation"""
        split_date = self.backtest_engine._calculate_split_date(self.config)

        self.assertGreater(split_date, self.config.start_date)
        self.assertLess(split_date, self.config.end_date)

class TestDataQuality(unittest.TestCase):
    """Test cases for data quality validation"""

    def setUp(self):
        """Set up test fixtures"""
        self.validator = DataQualityValidator()

        # Create sample dataset with various quality issues
        np.random.seed(42)
        self.sample_data = pd.DataFrame({
            'price': np.random.uniform(50, 150, 1000),
            'volume': np.random.randint(1000, 10000, 1000),
            'return': np.random.normal(0, 0.02, 1000),
            'category': np.random.choice(['A', 'B', 'C'], 1000)
        })

        # Introduce some quality issues
        self.sample_data.loc[50:60, 'price'] = np.nan  # Missing values
        self.sample_data.loc[100, 'price'] = 1000  # Outlier
        self.sample_data.loc[200, :] = self.sample_data.loc[199, :]  # Duplicate

    def test_missing_value_check(self):
        """Test missing value detection"""
        check = MissingValueCheck(threshold=0.05)
        result = check.check(self.sample_data)

        self.assertEqual(result.check_name, "Missing Value Check")
        self.assertIn('missing_rate', result.details)
        self.assertIn('missing_by_column', result.details)

    def test_outlier_detection_check(self):
        """Test outlier detection"""
        check = OutlierDetectionCheck(method='iqr')
        result = check.check(self.sample_data)

        self.assertEqual(result.check_name, "Outlier Detection")
        self.assertIn('outlier_rate', result.details)
        self.assertIn('outliers_by_column', result.details)

    def test_data_quality_validation(self):
        """Test comprehensive data quality validation"""
        self.validator.add_check(MissingValueCheck())
        self.validator.add_check(OutlierDetectionCheck())

        report = self.validator.validate(self.sample_data, "Test Dataset")

        self.assertEqual(report.dataset_name, "Test Dataset")
        self.assertGreater(report.checks_performed, 0)
        self.assertIsNotNone(report.overall_score)
        self.assertIn('basic_info', report.summary_statistics)

class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflows"""

    def setUp(self):
        """Set up integration test fixtures"""
        self.risk_engine = RiskEngine()
        self.sample_positions = self.risk_engine.generate_sample_portfolio()

    def test_end_to_end_risk_calculation(self):
        """Test complete risk calculation workflow"""
        # Calculate VaR
        var_results = self.risk_engine.calculate_portfolio_var(self.sample_positions)

        # Calculate Greeks
        greeks = self.risk_engine.calculate_greeks(self.sample_positions)

        # Run stress tests
        stress_results = self.risk_engine.run_stress_tests(self.sample_positions)

        # Calculate risk metrics
        risk_metrics = self.risk_engine.calculate_risk_metrics_summary(self.sample_positions)

        # Validate all components completed successfully
        self.assertIsNotNone(var_results)
        self.assertIsNotNone(greeks)
        self.assertIsNotNone(stress_results)
        self.assertIsNotNone(risk_metrics)

    def test_model_validation_workflow(self):
        """Test complete model validation workflow"""
        from model_validation import ModelValidationFramework

        framework = ModelValidationFramework()

        # Create validation data
        np.random.seed(42)
        validation_data = {
            'var_data': {
                'returns': np.random.normal(0, 0.02, 250),
                'var_forecasts': np.abs(np.random.normal(0.03, 0.01, 250))
            },
            'monte_carlo_data': {
                'simulation_results': np.random.normal(-2, 15, 10000)
            }
        }

        # Run validation
        results = framework.run_comprehensive_validation(validation_data)

        self.assertIn('validation_summary', results)
        self.assertIn('detailed_results', results)
        self.assertIn('overall_status', results)

def create_test_suite():
    """Create comprehensive test suite"""
    suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestRiskEngine,
        TestVaRBacktesting,
        TestGreeksValidation,
        TestStressTestValidation,
        TestMonteCarloValidation,
        TestBacktestingFramework,
        TestDataQuality,
        TestIntegration
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    return suite

if __name__ == '__main__':
    # Configure logging for tests
    import logging
    logging.basicConfig(level=logging.INFO)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    suite = create_test_suite()
    result = runner.run(suite)

    # Print summary
    print(f"\nTest Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")

    # Exit with appropriate code
    exit_code = 0 if result.wasSuccessful() else 1
    sys.exit(exit_code)