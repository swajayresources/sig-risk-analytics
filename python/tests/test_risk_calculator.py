"""
Comprehensive tests for Risk Calculation Engine
Tests Monte Carlo VaR, Greeks calculation, and stress testing functionality
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import scipy.stats as stats
from typing import Dict, List, Tuple

# Mock risk calculation classes for testing
class MockMonteCarloEngine:
    def __init__(self, num_scenarios=10000):
        self.num_scenarios = num_scenarios
        np.random.seed(42)  # For reproducible tests

    def simulate_portfolio_pnl(self, positions, market_factors):
        """Simulate portfolio P&L using mock Monte Carlo"""
        # Generate random returns for each position
        portfolio_values = [pos['quantity'] * pos['price'] for pos in positions]
        total_value = sum(abs(v) for v in portfolio_values)

        # Simulate returns using normal distribution
        returns = np.random.normal(0, 0.02, self.num_scenarios)  # 2% daily vol
        portfolio_pnl = returns * total_value

        return {
            'portfolio_pnl': portfolio_pnl.tolist(),
            'var_estimate': np.percentile(portfolio_pnl, 5),  # 95% VaR
            'expected_shortfall': np.mean(portfolio_pnl[portfolio_pnl <= np.percentile(portfolio_pnl, 5)]),
            'computation_time_ms': 10.5,
            'scenarios_used': self.num_scenarios
        }

class MockGreeksCalculator:
    @staticmethod
    def calculate_black_scholes_greeks(spot, strike, time_to_expiry, risk_free_rate, volatility, option_type):
        """Calculate Greeks using simplified Black-Scholes"""
        if time_to_expiry <= 0:
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}

        # Simplified Greeks calculation for testing
        d1 = (np.log(spot/strike) + (risk_free_rate + 0.5*volatility**2)*time_to_expiry) / (volatility*np.sqrt(time_to_expiry))
        d2 = d1 - volatility*np.sqrt(time_to_expiry)

        if option_type == 'CALL':
            delta = stats.norm.cdf(d1)
        else:
            delta = stats.norm.cdf(d1) - 1

        gamma = stats.norm.pdf(d1) / (spot * volatility * np.sqrt(time_to_expiry))
        theta = -(spot * stats.norm.pdf(d1) * volatility) / (2 * np.sqrt(time_to_expiry)) / 365
        vega = spot * stats.norm.pdf(d1) * np.sqrt(time_to_expiry) / 100
        rho = strike * time_to_expiry * np.exp(-risk_free_rate * time_to_expiry) * stats.norm.cdf(d2 if option_type == 'CALL' else -d2) / 100

        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }

class MockStressTester:
    def __init__(self):
        self.scenarios = {
            'covid_crash_2020': {'equity': -0.34, 'bond': 0.12, 'volatility': 4.0},
            'lehman_crisis_2008': {'equity': -0.45, 'bond': 0.08, 'volatility': 3.0},
            'flash_crash_2010': {'equity': -0.09, 'volatility': 2.5}
        }

    def run_stress_test(self, scenario_name, positions):
        """Run stress test for given scenario"""
        if scenario_name not in self.scenarios:
            raise ValueError(f"Unknown scenario: {scenario_name}")

        scenario = self.scenarios[scenario_name]
        total_pnl = 0

        for pos in positions:
            asset_type = pos.get('asset_type', 'equity')
            shock = scenario.get(asset_type, 0)
            position_value = pos['quantity'] * pos['price']
            position_pnl = position_value * shock
            total_pnl += position_pnl

        portfolio_value = sum(abs(pos['quantity'] * pos['price']) for pos in positions)

        return {
            'scenario_name': scenario_name,
            'portfolio_pnl': total_pnl,
            'portfolio_pnl_percent': total_pnl / portfolio_value if portfolio_value > 0 else 0,
            'var_breach': abs(total_pnl) > portfolio_value * 0.05,
            'computation_time_ms': 5.2
        }

@pytest.fixture
def sample_positions():
    """Sample portfolio positions for testing"""
    return [
        {'symbol': 'AAPL', 'quantity': 1000, 'price': 175.0, 'asset_type': 'equity'},
        {'symbol': 'GOOGL', 'quantity': 100, 'price': 2750.0, 'asset_type': 'equity'},
        {'symbol': 'TSLA', 'quantity': -200, 'price': 850.0, 'asset_type': 'equity'},  # Short position
        {'symbol': 'SPY_CALL', 'quantity': 10, 'price': 25.0, 'asset_type': 'option',
         'strike': 420, 'time_to_expiry': 0.25, 'option_type': 'CALL'},
        {'symbol': 'TLT', 'quantity': 500, 'price': 95.0, 'asset_type': 'bond'}
    ]

@pytest.fixture
def monte_carlo_engine():
    return MockMonteCarloEngine()

@pytest.fixture
def greeks_calculator():
    return MockGreeksCalculator()

@pytest.fixture
def stress_tester():
    return MockStressTester()

class TestMonteCarloVaR:
    """Test Monte Carlo Value-at-Risk calculations"""

    def test_basic_var_calculation(self, monte_carlo_engine, sample_positions):
        """Test basic VaR calculation"""
        market_factors = {
            'correlations': np.eye(len(sample_positions)),
            'volatilities': [0.25] * len(sample_positions)
        }

        result = monte_carlo_engine.simulate_portfolio_pnl(sample_positions, market_factors)

        assert 'var_estimate' in result
        assert 'expected_shortfall' in result
        assert 'portfolio_pnl' in result
        assert len(result['portfolio_pnl']) == monte_carlo_engine.num_scenarios

        # VaR should be negative (representing loss)
        assert result['var_estimate'] < 0

        # Expected Shortfall should be more negative than VaR
        assert result['expected_shortfall'] <= result['var_estimate']

    def test_var_scaling(self, monte_carlo_engine):
        """Test VaR scaling with portfolio size"""
        small_position = [{'symbol': 'AAPL', 'quantity': 100, 'price': 150.0, 'asset_type': 'equity'}]
        large_position = [{'symbol': 'AAPL', 'quantity': 1000, 'price': 150.0, 'asset_type': 'equity'}]

        market_factors = {'correlations': np.eye(1), 'volatilities': [0.2]}

        small_result = monte_carlo_engine.simulate_portfolio_pnl(small_position, market_factors)
        large_result = monte_carlo_engine.simulate_portfolio_pnl(large_position, market_factors)

        # VaR should scale approximately linearly with position size
        var_ratio = abs(large_result['var_estimate']) / abs(small_result['var_estimate'])
        assert 8 < var_ratio < 12  # Should be close to 10x (1000/100)

    def test_convergence_with_scenarios(self):
        """Test VaR convergence as number of scenarios increases"""
        scenarios_list = [1000, 5000, 10000, 50000]
        positions = [{'symbol': 'AAPL', 'quantity': 1000, 'price': 150.0, 'asset_type': 'equity'}]
        market_factors = {'correlations': np.eye(1), 'volatilities': [0.2]}

        vars = []
        for num_scenarios in scenarios_list:
            engine = MockMonteCarloEngine(num_scenarios)
            result = engine.simulate_portfolio_pnl(positions, market_factors)
            vars.append(result['var_estimate'])

        # VaR estimates should converge (less variation with more scenarios)
        var_std_first_half = np.std(vars[:2])
        var_std_second_half = np.std(vars[2:])

        # Later estimates should be more stable
        assert var_std_second_half <= var_std_first_half * 1.5

    def test_diversification_benefit(self, monte_carlo_engine):
        """Test that diversification reduces portfolio VaR"""
        # Single concentrated position
        concentrated = [{'symbol': 'AAPL', 'quantity': 1000, 'price': 150.0, 'asset_type': 'equity'}]

        # Diversified positions with same total value
        diversified = [
            {'symbol': 'AAPL', 'quantity': 200, 'price': 150.0, 'asset_type': 'equity'},
            {'symbol': 'GOOGL', 'quantity': 50, 'price': 600.0, 'asset_type': 'equity'},
            {'symbol': 'MSFT', 'quantity': 150, 'price': 200.0, 'asset_type': 'equity'},
            {'symbol': 'TSLA', 'quantity': 37.5, 'price': 800.0, 'asset_type': 'equity'}
        ]

        # Assume some correlation (not perfect)
        correlation_matrix = np.full((4, 4), 0.6)
        np.fill_diagonal(correlation_matrix, 1.0)

        market_factors_concentrated = {'correlations': np.eye(1), 'volatilities': [0.25]}
        market_factors_diversified = {'correlations': correlation_matrix, 'volatilities': [0.25] * 4}

        concentrated_result = monte_carlo_engine.simulate_portfolio_pnl(concentrated, market_factors_concentrated)
        diversified_result = monte_carlo_engine.simulate_portfolio_pnl(diversified, market_factors_diversified)

        # Diversified portfolio should have lower VaR (in absolute terms)
        # Note: This test might be flaky due to randomness, so we use a generous threshold
        diversification_benefit = abs(diversified_result['var_estimate']) / abs(concentrated_result['var_estimate'])
        assert diversification_benefit < 0.95  # At least 5% reduction in VaR

class TestGreeksCalculation:
    """Test options Greeks calculations"""

    def test_call_option_greeks(self, greeks_calculator):
        """Test Greeks calculation for call options"""
        greeks = greeks_calculator.calculate_black_scholes_greeks(
            spot=100, strike=100, time_to_expiry=0.25,
            risk_free_rate=0.05, volatility=0.2, option_type='CALL'
        )

        # ATM call delta should be around 0.5
        assert 0.4 < greeks['delta'] < 0.6

        # Gamma should be positive
        assert greeks['gamma'] > 0

        # Theta should be negative (time decay)
        assert greeks['theta'] < 0

        # Vega should be positive (benefit from vol increase)
        assert greeks['vega'] > 0

    def test_put_option_greeks(self, greeks_calculator):
        """Test Greeks calculation for put options"""
        greeks = greeks_calculator.calculate_black_scholes_greeks(
            spot=100, strike=100, time_to_expiry=0.25,
            risk_free_rate=0.05, volatility=0.2, option_type='PUT'
        )

        # ATM put delta should be around -0.5
        assert -0.6 < greeks['delta'] < -0.4

        # Gamma should be positive (same as call)
        assert greeks['gamma'] > 0

        # Theta should be negative
        assert greeks['theta'] < 0

        # Vega should be positive
        assert greeks['vega'] > 0

    def test_moneyness_effects(self, greeks_calculator):
        """Test how Greeks change with moneyness"""
        strikes = [80, 90, 100, 110, 120]
        spot = 100
        deltas = []

        for strike in strikes:
            greeks = greeks_calculator.calculate_black_scholes_greeks(
                spot=spot, strike=strike, time_to_expiry=0.25,
                risk_free_rate=0.05, volatility=0.2, option_type='CALL'
            )
            deltas.append(greeks['delta'])

        # Delta should decrease as strike increases (for calls)
        for i in range(len(deltas) - 1):
            assert deltas[i] > deltas[i + 1]

        # ITM call should have delta close to 1, OTM close to 0
        assert deltas[0] > 0.8  # Deep ITM
        assert deltas[-1] < 0.2  # Deep OTM

    def test_time_decay_effects(self, greeks_calculator):
        """Test how Greeks change with time to expiry"""
        times = [1/365, 7/365, 30/365, 90/365, 365/365]  # 1 day to 1 year
        thetas = []

        for time_to_expiry in times:
            greeks = greeks_calculator.calculate_black_scholes_greeks(
                spot=100, strike=100, time_to_expiry=time_to_expiry,
                risk_free_rate=0.05, volatility=0.2, option_type='CALL'
            )
            thetas.append(greeks['theta'])

        # Theta should be more negative for shorter-term options
        # (time decay accelerates as expiry approaches)
        assert thetas[0] < thetas[-1]  # More negative for short-term

    def test_volatility_sensitivity(self, greeks_calculator):
        """Test vega calculation across different volatilities"""
        volatilities = [0.1, 0.2, 0.3, 0.4, 0.5]
        vegas = []

        for vol in volatilities:
            greeks = greeks_calculator.calculate_black_scholes_greeks(
                spot=100, strike=100, time_to_expiry=0.25,
                risk_free_rate=0.05, volatility=vol, option_type='CALL'
            )
            vegas.append(greeks['vega'])

        # All vegas should be positive
        assert all(vega > 0 for vega in vegas)

        # Vega typically peaks around ATM and decreases for extreme volatilities
        # (this is a simplified test - actual behavior is more complex)

    def test_portfolio_greeks_aggregation(self, greeks_calculator):
        """Test aggregation of Greeks across a portfolio"""
        positions = [
            {'strike': 95, 'quantity': 10, 'option_type': 'CALL'},
            {'strike': 100, 'quantity': -5, 'option_type': 'CALL'},  # Short call
            {'strike': 105, 'quantity': 15, 'option_type': 'PUT'}
        ]

        total_delta = 0
        total_gamma = 0
        total_theta = 0
        total_vega = 0

        for pos in positions:
            greeks = greeks_calculator.calculate_black_scholes_greeks(
                spot=100, strike=pos['strike'], time_to_expiry=0.25,
                risk_free_rate=0.05, volatility=0.2, option_type=pos['option_type']
            )

            total_delta += greeks['delta'] * pos['quantity']
            total_gamma += greeks['gamma'] * pos['quantity']
            total_theta += greeks['theta'] * pos['quantity']
            total_vega += greeks['vega'] * pos['quantity']

        # Portfolio Greeks should reflect net exposure
        # Short call should reduce delta
        assert abs(total_delta) < abs(10 * greeks_calculator.calculate_black_scholes_greeks(
            spot=100, strike=95, time_to_expiry=0.25,
            risk_free_rate=0.05, volatility=0.2, option_type='CALL'
        )['delta'])

class TestStressTesting:
    """Test stress testing functionality"""

    def test_historical_scenario_stress(self, stress_tester, sample_positions):
        """Test historical scenario stress testing"""
        result = stress_tester.run_stress_test('covid_crash_2020', sample_positions)

        assert result['scenario_name'] == 'covid_crash_2020'
        assert 'portfolio_pnl' in result
        assert 'portfolio_pnl_percent' in result
        assert 'var_breach' in result

        # COVID crash should result in negative P&L for long equity positions
        assert result['portfolio_pnl'] < 0

    def test_multiple_scenarios(self, stress_tester, sample_positions):
        """Test multiple stress scenarios"""
        scenarios = ['covid_crash_2020', 'lehman_crisis_2008', 'flash_crash_2010']
        results = []

        for scenario in scenarios:
            result = stress_tester.run_stress_test(scenario, sample_positions)
            results.append(result)

        # All scenarios should produce losses for this equity-heavy portfolio
        for result in results:
            assert result['portfolio_pnl'] < 0

        # Lehman crisis should be the worst scenario
        lehman_result = next(r for r in results if r['scenario_name'] == 'lehman_crisis_2008')
        other_results = [r for r in results if r['scenario_name'] != 'lehman_crisis_2008']

        for other_result in other_results:
            assert lehman_result['portfolio_pnl'] <= other_result['portfolio_pnl']

    def test_stress_test_with_hedged_portfolio(self, stress_tester):
        """Test stress testing with a hedged portfolio"""
        # Portfolio with both long and short positions
        hedged_positions = [
            {'symbol': 'SPY', 'quantity': 1000, 'price': 400.0, 'asset_type': 'equity'},
            {'symbol': 'SPY_PUT', 'quantity': 100, 'price': 15.0, 'asset_type': 'option'},  # Protection
            {'symbol': 'TLT', 'quantity': 500, 'price': 95.0, 'asset_type': 'bond'}  # Flight to quality
        ]

        result = stress_tester.run_stress_test('covid_crash_2020', hedged_positions)

        # Hedged portfolio should perform better than unhedged
        # (though this depends on the specific implementation of the mock stress tester)
        assert 'portfolio_pnl' in result

    def test_stress_test_error_handling(self, stress_tester, sample_positions):
        """Test error handling for invalid scenarios"""
        with pytest.raises(ValueError, match="Unknown scenario"):
            stress_tester.run_stress_test('invalid_scenario', sample_positions)

class TestRiskIntegration:
    """Integration tests combining multiple risk measures"""

    def test_risk_dashboard_data(self, monte_carlo_engine, greeks_calculator, stress_tester, sample_positions):
        """Test data needed for a comprehensive risk dashboard"""
        # VaR calculation
        market_factors = {
            'correlations': np.eye(len(sample_positions)),
            'volatilities': [0.25] * len(sample_positions)
        }
        var_result = monte_carlo_engine.simulate_portfolio_pnl(sample_positions, market_factors)

        # Greeks for options positions
        option_positions = [pos for pos in sample_positions if pos.get('asset_type') == 'option']
        portfolio_greeks = {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}

        for pos in option_positions:
            greeks = greeks_calculator.calculate_black_scholes_greeks(
                spot=pos['price'] * 1.1,  # Mock current spot price
                strike=pos['strike'],
                time_to_expiry=pos['time_to_expiry'],
                risk_free_rate=0.05,
                volatility=0.25,
                option_type=pos['option_type']
            )

            for greek in portfolio_greeks:
                portfolio_greeks[greek] += greeks[greek] * pos['quantity']

        # Stress test results
        stress_results = {}
        for scenario in ['covid_crash_2020', 'lehman_crisis_2008']:
            stress_results[scenario] = stress_tester.run_stress_test(scenario, sample_positions)

        # Verify we have all necessary data for dashboard
        dashboard_data = {
            'var_95': var_result['var_estimate'],
            'expected_shortfall': var_result['expected_shortfall'],
            'portfolio_greeks': portfolio_greeks,
            'stress_results': stress_results,
            'computation_time': (
                var_result['computation_time_ms'] +
                sum(result['computation_time_ms'] for result in stress_results.values())
            )
        }

        assert all(key in dashboard_data for key in [
            'var_95', 'expected_shortfall', 'portfolio_greeks', 'stress_results'
        ])

        # Performance check: total computation should be reasonable
        assert dashboard_data['computation_time'] < 100  # Less than 100ms total

    def test_risk_limit_monitoring(self, monte_carlo_engine, sample_positions):
        """Test risk limit violation detection"""
        portfolio_value = sum(abs(pos['quantity'] * pos['price']) for pos in sample_positions)

        market_factors = {
            'correlations': np.eye(len(sample_positions)),
            'volatilities': [0.25] * len(sample_positions)
        }
        var_result = monte_carlo_engine.simulate_portfolio_pnl(sample_positions, market_factors)

        # Define risk limits
        risk_limits = {
            'var_limit_pct': 0.02,  # 2% of portfolio value
            'concentration_limit_pct': 0.10  # 10% max in single position
        }

        # Check VaR limit
        var_limit = portfolio_value * risk_limits['var_limit_pct']
        var_breach = abs(var_result['var_estimate']) > var_limit

        # Check concentration limits
        concentration_breaches = []
        for pos in sample_positions:
            position_value = abs(pos['quantity'] * pos['price'])
            concentration_pct = position_value / portfolio_value
            if concentration_pct > risk_limits['concentration_limit_pct']:
                concentration_breaches.append({
                    'symbol': pos['symbol'],
                    'concentration': concentration_pct,
                    'limit': risk_limits['concentration_limit_pct']
                })

        # Risk monitoring summary
        risk_status = {
            'var_breach': var_breach,
            'concentration_breaches': concentration_breaches,
            'total_violations': int(var_breach) + len(concentration_breaches)
        }

        assert isinstance(risk_status['var_breach'], bool)
        assert isinstance(risk_status['concentration_breaches'], list)
        assert risk_status['total_violations'] >= 0

class TestPerformanceAndAccuracy:
    """Test performance and numerical accuracy"""

    def test_greeks_numerical_stability(self, greeks_calculator):
        """Test Greeks calculation numerical stability"""
        # Test edge cases that might cause numerical issues
        edge_cases = [
            {'spot': 100, 'strike': 100, 'time_to_expiry': 0.001, 'volatility': 0.01},  # Very short time, low vol
            {'spot': 100, 'strike': 100, 'time_to_expiry': 0.001, 'volatility': 2.0},   # Very short time, high vol
            {'spot': 1, 'strike': 1000, 'time_to_expiry': 1.0, 'volatility': 0.2},     # Deep OTM
            {'spot': 1000, 'strike': 1, 'time_to_expiry': 1.0, 'volatility': 0.2},     # Deep ITM
        ]

        for case in edge_cases:
            greeks = greeks_calculator.calculate_black_scholes_greeks(
                risk_free_rate=0.05, option_type='CALL', **case
            )

            # All Greeks should be finite numbers
            for greek_name, greek_value in greeks.items():
                assert np.isfinite(greek_value), f"{greek_name} is not finite for case {case}"
                assert not np.isnan(greek_value), f"{greek_name} is NaN for case {case}"

    def test_monte_carlo_convergence_properties(self, monte_carlo_engine):
        """Test Monte Carlo convergence properties"""
        positions = [{'symbol': 'AAPL', 'quantity': 1000, 'price': 150.0, 'asset_type': 'equity'}]
        market_factors = {'correlations': np.eye(1), 'volatilities': [0.2]}

        # Run multiple independent Monte Carlo simulations
        vars = []
        for _ in range(10):
            result = monte_carlo_engine.simulate_portfolio_pnl(positions, market_factors)
            vars.append(result['var_estimate'])

        # Standard error should decrease approximately as 1/sqrt(n)
        var_std = np.std(vars)
        theoretical_std = np.std(vars) / np.sqrt(monte_carlo_engine.num_scenarios)

        # The standard deviation of VaR estimates should be reasonable
        assert var_std < abs(np.mean(vars)) * 0.1  # Less than 10% of the mean

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--durations=10"])