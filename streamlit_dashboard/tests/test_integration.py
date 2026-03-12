"""
Integration Tests for Risk Dashboard
====================================

Comprehensive integration tests for the complete risk management dashboard,
testing workflows, data flow, and system integration.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os
import tempfile
import json

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from risk_engine import RiskEngine
from data_provider import DataProvider
from alert_manager import AlertManager
from export_manager import ExportManager
from visualization_engine import VisualizationEngine
from model_validation import ModelValidationFramework
from backtesting_framework import BacktestEngine, RiskParityStrategy, BacktestConfig
from data_quality import DataQualityValidator, create_financial_data_validator

class TestDashboardIntegration(unittest.TestCase):
    """Integration tests for complete dashboard functionality"""

    def setUp(self):
        """Set up integration test environment"""
        self.risk_engine = RiskEngine()
        self.data_provider = DataProvider(use_redis=False)  # Use local mode for testing
        self.alert_manager = AlertManager()
        self.export_manager = ExportManager()
        self.viz_engine = VisualizationEngine()
        self.validation_framework = ModelValidationFramework()

        # Create temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test environment"""
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass

    def test_portfolio_data_flow(self):
        """Test complete portfolio data flow from input to output"""
        # 1. Generate sample portfolio
        sample_portfolio = self.risk_engine.generate_sample_portfolio()
        self.assertGreater(len(sample_portfolio), 0)

        # 2. Calculate risk metrics
        risk_metrics = self.risk_engine.calculate_risk_metrics_summary(sample_portfolio)
        self.assertIsNotNone(risk_metrics)

        # 3. Check for risk limit breaches
        portfolio_metrics = {
            'var_1d': risk_metrics.var_1d,
            'expected_shortfall': risk_metrics.expected_shortfall,
            'concentration': 8.5,  # Sample concentration
            'leverage': 2.1
        }

        alerts = self.alert_manager.check_risk_limits(portfolio_metrics)
        self.assertIsInstance(alerts, list)

        # 4. Generate visualizations
        var_breakdown = {
            'by_asset_class': {
                'Equity': {'total_var': 12.5},
                'Fixed Income': {'total_var': 3.2}
            }
        }

        fig = self.viz_engine.create_var_breakdown_chart(var_breakdown)
        self.assertIsNotNone(fig)

        # 5. Export results
        portfolio_data = []
        for pos in sample_portfolio:
            portfolio_data.append({
                'symbol': pos.symbol,
                'quantity': pos.quantity,
                'market_value': pos.market_value,
                'asset_class': pos.asset_class
            })

        excel_data = self.export_manager.export_portfolio_to_excel(
            portfolio_data, portfolio_metrics
        )
        self.assertGreater(len(excel_data), 0)

    def test_real_time_data_integration(self):
        """Test real-time data integration and processing"""
        # 1. Get real-time market data
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        market_data = self.data_provider.get_real_time_prices(symbols)

        self.assertEqual(len(market_data), len(symbols))
        for symbol in symbols:
            self.assertIn(symbol, market_data)
            self.assertGreater(market_data[symbol].price, 0)

        # 2. Get risk factors
        risk_factors = self.data_provider.get_risk_factors()
        self.assertGreater(len(risk_factors), 0)

        # 3. Create risk factor heatmap
        fig = self.viz_engine.create_risk_factor_heatmap(risk_factors)
        self.assertIsNotNone(fig)

        # 4. Validate data quality
        validator = create_financial_data_validator()

        # Create sample price data for validation
        price_data = pd.DataFrame({
            'symbol': symbols * 100,
            'price': [market_data[symbol].price for symbol in symbols] * 100,
            'volume': [market_data[symbol].volume for symbol in symbols] * 100,
            'timestamp': [datetime.now()] * 300
        })

        quality_report = validator.validate(price_data, "Real-time Market Data")
        self.assertIsNotNone(quality_report)
        self.assertGreater(quality_report.overall_score, 0)

    def test_model_validation_integration(self):
        """Test complete model validation workflow integration"""
        # 1. Generate sample portfolio and calculate VaR
        sample_portfolio = self.risk_engine.generate_sample_portfolio()
        var_results = self.risk_engine.calculate_portfolio_var(sample_portfolio)

        # 2. Create validation data
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 250)
        var_forecasts = np.full(250, var_results['diversified_var'] / 1000000)  # Scale to daily

        # 3. Run comprehensive validation
        validation_data = {
            'var_data': {
                'returns': returns,
                'var_forecasts': var_forecasts,
                'confidence_level': 0.95
            },
            'stress_test_data': {
                'scenario_results': self.risk_engine.run_stress_tests(sample_portfolio),
                'portfolio_exposures': {'equity': 0.6, 'fixed_income': 0.3, 'alternatives': 0.1}
            },
            'monte_carlo_data': {
                'simulation_results': np.random.normal(-2, 15, 10000)
            }
        }

        results = self.validation_framework.run_comprehensive_validation(validation_data)

        # 4. Validate results structure
        self.assertIn('validation_summary', results)
        self.assertIn('detailed_results', results)
        self.assertIn('overall_status', results)

        # 5. Generate validation report
        report = self.validation_framework.generate_validation_report(results)
        self.assertIsInstance(report, str)
        self.assertIn('MODEL VALIDATION REPORT', report)

    def test_backtesting_integration(self):
        """Test backtesting framework integration"""
        # 1. Set up backtesting engine
        backtest_engine = BacktestEngine(data_provider=self.data_provider)
        strategy = RiskParityStrategy()
        backtest_engine.register_strategy(strategy)

        # 2. Create backtest configuration
        config = BacktestConfig(
            start_date=datetime(2022, 1, 1),
            end_date=datetime(2023, 1, 1),
            lookback_window=63,
            out_of_sample_ratio=0.2
        )

        # 3. Run backtest
        backtest_result = backtest_engine.run_backtest(strategy.name, config)

        # 4. Validate backtest results
        self.assertIsNotNone(backtest_result)
        self.assertEqual(backtest_result.model_name, strategy.name)
        self.assertIsNotNone(backtest_result.performance_metrics)
        self.assertGreater(len(backtest_result.validation_results), 0)

        # 5. Test performance metrics
        metrics = backtest_result.performance_metrics
        self.assertIsInstance(metrics.total_return, float)
        self.assertIsInstance(metrics.sharpe_ratio, float)
        self.assertIsInstance(metrics.max_drawdown, float)

    def test_alert_system_integration(self):
        """Test alert system integration and workflow"""
        # 1. Generate demo alerts
        self.alert_manager.simulate_demo_alerts()

        # 2. Get active alerts
        active_alerts = self.alert_manager.get_active_alerts()
        self.assertGreater(len(active_alerts), 0)

        # 3. Test alert statistics
        stats = self.alert_manager.get_alert_statistics()
        self.assertIn('total', stats)
        self.assertIn('critical', stats)
        self.assertIn('high', stats)

        # 4. Test alert acknowledgment
        if active_alerts:
            first_alert = active_alerts[0]
            success = self.alert_manager.acknowledge_alert(
                first_alert.alert_id, "test_user", "Integration test acknowledgment"
            )
            self.assertTrue(success)

    def test_export_system_integration(self):
        """Test export system integration and report generation"""
        # 1. Generate sample data
        sample_portfolio = self.risk_engine.generate_sample_portfolio()
        risk_metrics = self.risk_engine.calculate_risk_metrics_summary(sample_portfolio)

        portfolio_data = {'name': 'Test Portfolio', 'total_value': 2500}
        metrics_dict = {
            'var_1d': risk_metrics.var_1d,
            'expected_shortfall': risk_metrics.expected_shortfall,
            'sharpe_ratio': risk_metrics.sharpe_ratio,
            'max_drawdown': risk_metrics.max_drawdown
        }

        # 2. Test PDF report generation
        pdf_data = self.export_manager.generate_executive_summary_pdf(portfolio_data, metrics_dict)
        self.assertGreater(len(pdf_data), 0)

        # 3. Test Excel export
        portfolio_list = []
        for pos in sample_portfolio:
            portfolio_list.append({
                'symbol': pos.symbol,
                'quantity': pos.quantity,
                'market_value': pos.market_value,
                'asset_class': pos.asset_class,
                'sector': pos.sector,
                'region': pos.region
            })

        excel_data = self.export_manager.export_portfolio_to_excel(portfolio_list, metrics_dict)
        self.assertGreater(len(excel_data), 0)

        # 4. Test VaR analysis report
        var_data = {
            'historical_var': risk_metrics.var_1d,
            'parametric_var': risk_metrics.var_1d * 0.95,
            'monte_carlo_var': risk_metrics.var_1d * 1.05,
            'diversified_var': risk_metrics.var_1d,
            'expected_shortfall': risk_metrics.expected_shortfall
        }
        var_report = self.export_manager.generate_var_analysis_report(var_data, {})
        self.assertGreater(len(var_report), 0)

    def test_visualization_integration(self):
        """Test visualization engine integration"""
        # 1. Generate sample data
        sample_portfolio = self.risk_engine.generate_sample_portfolio()

        # 2. Test 3D risk surface
        portfolio_data = []
        for pos in sample_portfolio:
            portfolio_data.append({
                'sector': pos.sector,
                'region': pos.region,
                'market_value': pos.market_value
            })

        fig_3d = self.viz_engine.create_3d_risk_surface(portfolio_data)
        self.assertIsNotNone(fig_3d)

        # 3. Test correlation heatmap
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
        correlation_data = self.data_provider.get_correlation_matrix(symbols)
        fig_corr = self.viz_engine.create_correlation_heatmap(correlation_data)
        self.assertIsNotNone(fig_corr)

        # 4. Test Monte Carlo distribution
        np.random.seed(42)
        mc_results = np.random.normal(-2, 15, 10000)
        fig_mc = self.viz_engine.create_monte_carlo_distribution(mc_results)
        self.assertIsNotNone(fig_mc)

        # 5. Test Greeks ladder
        greeks_data = self.risk_engine.calculate_greeks(sample_portfolio)
        greeks_by_category = {
            'Equity Options': {
                'delta': 850, 'gamma': 12.5, 'theta': -125, 'vega': 2500, 'rho': 450
            },
            'Index Options': {
                'delta': 350, 'gamma': 8.2, 'theta': -85, 'vega': 1800, 'rho': 320
            }
        }
        fig_greeks = self.viz_engine.create_greeks_ladder_chart(greeks_by_category)
        self.assertIsNotNone(fig_greeks)

    def test_data_quality_integration(self):
        """Test data quality validation integration"""
        # 1. Create sample financial dataset with quality issues
        np.random.seed(42)

        # Generate base data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        n_days = len(dates)

        sample_data = pd.DataFrame({
            'date': dates,
            'price': np.random.uniform(50, 150, n_days),
            'volume': np.random.randint(1000000, 10000000, n_days),
            'return': np.random.normal(0, 0.02, n_days),
            'volatility': np.random.uniform(0.1, 0.5, n_days),
            'market_cap': np.random.uniform(1e9, 1e12, n_days)
        })

        # Introduce quality issues
        sample_data.loc[50:60, 'price'] = np.nan  # Missing values
        sample_data.loc[100, 'price'] = 1000  # Outlier
        sample_data.loc[200, :] = sample_data.loc[199, :]  # Duplicate

        # 2. Run data quality validation
        validator = create_financial_data_validator()
        quality_report = validator.validate(sample_data, "Financial Market Data")

        # 3. Validate quality report
        self.assertIsNotNone(quality_report)
        self.assertEqual(quality_report.dataset_name, "Financial Market Data")
        self.assertGreater(quality_report.checks_performed, 0)
        self.assertGreater(quality_report.issues_found, 0)  # We introduced issues

        # 4. Generate HTML report
        html_report = validator.generate_report_html(quality_report)
        self.assertIsInstance(html_report, str)
        self.assertIn('Data Quality Report', html_report)

    def test_error_handling_and_resilience(self):
        """Test system resilience and error handling"""
        # 1. Test with empty portfolio
        empty_portfolio = []
        try:
            risk_metrics = self.risk_engine.calculate_risk_metrics_summary(empty_portfolio)
            # Should handle gracefully
            self.assertIsNotNone(risk_metrics)
        except Exception as e:
            self.fail(f"Empty portfolio handling failed: {e}")

        # 2. Test with invalid data
        invalid_data = pd.DataFrame({
            'invalid_column': [None, None, None]
        })

        validator = DataQualityValidator()
        try:
            report = validator.validate(invalid_data, "Invalid Data")
            self.assertIsNotNone(report)
        except Exception as e:
            self.fail(f"Invalid data handling failed: {e}")

        # 3. Test with extreme values
        extreme_portfolio = [
            self.risk_engine.generate_sample_portfolio()[0]  # Take one position
        ]
        extreme_portfolio[0].market_value = 1e15  # Extreme value

        try:
            var_results = self.risk_engine.calculate_portfolio_var(extreme_portfolio)
            self.assertIsNotNone(var_results)
        except Exception as e:
            self.fail(f"Extreme value handling failed: {e}")

    def test_performance_benchmarks(self):
        """Test performance benchmarks and execution times"""
        import time

        # 1. Benchmark VaR calculation
        sample_portfolio = self.risk_engine.generate_sample_portfolio()

        start_time = time.time()
        var_results = self.risk_engine.calculate_portfolio_var(sample_portfolio)
        var_time = time.time() - start_time

        self.assertLess(var_time, 5.0, "VaR calculation took too long")

        # 2. Benchmark Monte Carlo simulation
        start_time = time.time()
        mc_results = np.random.normal(-2, 15, 10000)
        mc_time = time.time() - start_time

        self.assertLess(mc_time, 1.0, "Monte Carlo simulation took too long")

        # 3. Benchmark data quality validation
        sample_data = pd.DataFrame({
            'price': np.random.uniform(50, 150, 10000),
            'volume': np.random.randint(1000, 10000, 10000),
            'return': np.random.normal(0, 0.02, 10000)
        })

        validator = create_financial_data_validator()
        start_time = time.time()
        quality_report = validator.validate(sample_data, "Performance Test")
        validation_time = time.time() - start_time

        self.assertLess(validation_time, 10.0, "Data quality validation took too long")

    def test_memory_usage(self):
        """Test memory usage patterns"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Perform memory-intensive operations
        large_portfolio = []
        for i in range(1000):  # Large portfolio
            pos = self.risk_engine.generate_sample_portfolio()[0]
            pos.symbol = f"STOCK_{i}"
            large_portfolio.append(pos)

        # Calculate risk metrics
        risk_metrics = self.risk_engine.calculate_risk_metrics_summary(large_portfolio)

        # Generate large Monte Carlo simulation
        large_mc_results = np.random.normal(-2, 15, 100000)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 500MB for these operations)
        self.assertLess(memory_increase, 500, f"Memory usage increased by {memory_increase:.1f}MB")

class TestRegularoryCompliance(unittest.TestCase):
    """Test regulatory compliance features"""

    def setUp(self):
        """Set up regulatory compliance tests"""
        self.validation_framework = ModelValidationFramework()
        self.alert_manager = AlertManager()

    def test_basel_iii_compliance(self):
        """Test Basel III compliance features"""
        # 1. Test VaR backtesting with regulatory requirements
        from model_validation import VaRBacktester

        backtester = VaRBacktester()

        # Test with different violation scenarios
        test_cases = [
            (3, 250, 'green'),   # Green zone
            (7, 250, 'yellow'),  # Yellow zone
            (12, 250, 'red')     # Red zone
        ]

        for violations, observations, expected_zone in test_cases:
            result = backtester.traffic_light_test(violations)
            self.assertEqual(result.details['zone'], expected_zone)

    def test_model_documentation_compliance(self):
        """Test model documentation and audit trail requirements"""
        # 1. Generate validation results
        np.random.seed(42)
        validation_data = {
            'var_data': {
                'returns': np.random.normal(0, 0.02, 250),
                'var_forecasts': np.abs(np.random.normal(0.03, 0.01, 250))
            }
        }

        results = self.validation_framework.run_comprehensive_validation(validation_data)

        # 2. Check that results include required documentation
        self.assertIn('timestamp', results)
        self.assertIn('validation_summary', results)
        self.assertIn('detailed_results', results)

        # 3. Verify audit trail information
        for test_results in results['detailed_results'].values():
            if hasattr(test_results, 'test_results'):
                for test in test_results.test_results:
                    self.assertIsNotNone(test.timestamp)
                    self.assertIsNotNone(test.test_name)
                    self.assertIsNotNone(test.test_type)

    def test_risk_limit_compliance(self):
        """Test risk limit monitoring compliance"""
        # 1. Test risk limit validation
        portfolio_metrics = {
            'var_1d': 28.0,  # Above limit
            'expected_shortfall': 45.0,  # Above limit
            'concentration': 12.0,  # Above limit
            'leverage': 1.8  # Below limit
        }

        alerts = self.alert_manager.check_risk_limits(portfolio_metrics)

        # 2. Verify alerts were generated for breaches
        breach_alerts = [alert for alert in alerts if 'breach' in alert.title.lower()]
        self.assertGreater(len(breach_alerts), 0)

        # 3. Check alert documentation
        for alert in alerts:
            self.assertIsNotNone(alert.timestamp)
            self.assertIsNotNone(alert.details)
            self.assertIsNotNone(alert.recommendations)

def run_integration_tests():
    """Run all integration tests"""
    # Create test suite
    suite = unittest.TestSuite()

    # Add integration test classes
    test_classes = [
        TestDashboardIntegration,
        TestRegularoryCompliance
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result

if __name__ == '__main__':
    # Configure logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)  # Reduce log noise

    print("Running Integration Tests...")
    print("=" * 50)

    result = run_integration_tests()

    # Print summary
    print(f"\nIntegration Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split(chr(10))[-2]}")

    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split(chr(10))[-2]}")

    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100) if result.testsRun > 0 else 0
    print(f"Success rate: {success_rate:.1f}%")

    # Exit with appropriate code
    exit_code = 0 if result.wasSuccessful() else 1
    sys.exit(exit_code)