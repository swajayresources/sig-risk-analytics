"""
Model Validation Framework
==========================

Comprehensive model validation and backtesting framework that meets regulatory standards
including Basel III compliance, statistical testing, and quality assurance.
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import minimize
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
from abc import ABC, abstractmethod
import json
import pickle
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestResult(Enum):
    """Test result enumeration"""
    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"
    ERROR = "ERROR"

@dataclass
class ValidationResult:
    """Validation result data structure"""
    test_name: str
    test_type: str
    result: TestResult
    p_value: Optional[float] = None
    statistic: Optional[float] = None
    critical_value: Optional[float] = None
    confidence_level: float = 0.95
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    recommendations: List[str] = field(default_factory=list)

@dataclass
class BacktestResult:
    """Backtesting result data structure"""
    model_name: str
    period: Tuple[datetime, datetime]
    total_observations: int
    violations: int
    expected_violations: float
    violation_rate: float
    coverage_rate: float
    test_results: List[ValidationResult] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)

class BaseValidator(ABC):
    """Base class for all validators"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.results: List[ValidationResult] = []

    @abstractmethod
    def validate(self, *args, **kwargs) -> ValidationResult:
        """Abstract validation method"""
        pass

    def clear_results(self):
        """Clear previous validation results"""
        self.results.clear()

class VaRBacktester(BaseValidator):
    """VaR model backtesting with regulatory compliance"""

    def __init__(self):
        super().__init__("VaR Backtester", "VaR model validation and backtesting")
        self.confidence_levels = [0.95, 0.99]
        self.traffic_light_zones = {
            'green': (0, 4),
            'yellow': (5, 9),
            'red': (10, float('inf'))
        }

    def kupiec_test(self, violations: int, observations: int,
                   confidence_level: float = 0.95) -> ValidationResult:
        """
        Kupiec Proportion of Failures (POF) test for VaR model validation

        H0: The proportion of violations equals the expected rate
        H1: The proportion of violations does not equal the expected rate
        """
        try:
            expected_rate = 1 - confidence_level
            expected_violations = observations * expected_rate

            if violations == 0 or violations == observations:
                # Handle boundary cases
                lr_statistic = float('inf') if violations == 0 else 0
                p_value = 0.0
            else:
                # Likelihood ratio test statistic
                observed_rate = violations / observations

                lr_statistic = 2 * (
                    violations * np.log(observed_rate / expected_rate) +
                    (observations - violations) * np.log((1 - observed_rate) / (1 - expected_rate))
                )

                # Chi-squared distribution with 1 degree of freedom
                p_value = 1 - stats.chi2.cdf(lr_statistic, df=1)

            critical_value = stats.chi2.ppf(0.95, df=1)  # 95% confidence

            result = TestResult.PASS if p_value > 0.05 else TestResult.FAIL

            recommendations = []
            if result == TestResult.FAIL:
                if violations > expected_violations:
                    recommendations.append("Model underestimates risk - consider recalibration")
                else:
                    recommendations.append("Model overestimates risk - may be too conservative")

            return ValidationResult(
                test_name="Kupiec POF Test",
                test_type="VaR Backtesting",
                result=result,
                p_value=p_value,
                statistic=lr_statistic,
                critical_value=critical_value,
                details={
                    'violations': violations,
                    'observations': observations,
                    'violation_rate': violations / observations,
                    'expected_rate': expected_rate,
                    'confidence_level': confidence_level
                },
                recommendations=recommendations
            )

        except Exception as e:
            logger.error(f"Error in Kupiec test: {e}")
            return ValidationResult(
                test_name="Kupiec POF Test",
                test_type="VaR Backtesting",
                result=TestResult.ERROR,
                details={'error': str(e)}
            )

    def christoffersen_test(self, violations: np.ndarray,
                          confidence_level: float = 0.95) -> ValidationResult:
        """
        Christoffersen test for independence of VaR violations

        Tests both unconditional coverage and independence of violations
        """
        try:
            violations = np.array(violations, dtype=int)
            n = len(violations)
            n1 = np.sum(violations)

            if n1 == 0 or n1 == n:
                return ValidationResult(
                    test_name="Christoffersen Independence Test",
                    test_type="VaR Backtesting",
                    result=TestResult.WARNING,
                    details={'warning': 'No violations or all violations - test not applicable'}
                )

            # Count transitions
            n00 = n01 = n10 = n11 = 0

            for i in range(n - 1):
                if violations[i] == 0 and violations[i + 1] == 0:
                    n00 += 1
                elif violations[i] == 0 and violations[i + 1] == 1:
                    n01 += 1
                elif violations[i] == 1 and violations[i + 1] == 0:
                    n10 += 1
                elif violations[i] == 1 and violations[i + 1] == 1:
                    n11 += 1

            # Calculate likelihood ratio
            if n01 + n00 > 0 and n10 + n11 > 0:
                pi_01 = n01 / (n01 + n00) if (n01 + n00) > 0 else 0
                pi_11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
                pi = n1 / n

                if pi_01 > 0 and pi_11 > 0 and pi > 0 and (1 - pi) > 0:
                    lr_ind = 2 * (
                        n00 * np.log(1 - pi_01) + n01 * np.log(pi_01) +
                        n10 * np.log(1 - pi_11) + n11 * np.log(pi_11) -
                        (n00 + n10) * np.log(1 - pi) - (n01 + n11) * np.log(pi)
                    )
                else:
                    lr_ind = 0
            else:
                lr_ind = 0

            # P-value from chi-squared distribution with 1 degree of freedom
            p_value = 1 - stats.chi2.cdf(lr_ind, df=1) if lr_ind > 0 else 1.0
            critical_value = stats.chi2.ppf(0.95, df=1)

            result = TestResult.PASS if p_value > 0.05 else TestResult.FAIL

            recommendations = []
            if result == TestResult.FAIL:
                recommendations.append("Violations show clustering - model may not capture time-varying risk")
                recommendations.append("Consider GARCH or regime-switching models")

            return ValidationResult(
                test_name="Christoffersen Independence Test",
                test_type="VaR Backtesting",
                result=result,
                p_value=p_value,
                statistic=lr_ind,
                critical_value=critical_value,
                details={
                    'transitions': {'n00': n00, 'n01': n01, 'n10': n10, 'n11': n11},
                    'violation_rate': n1 / n,
                    'clustering_detected': result == TestResult.FAIL
                },
                recommendations=recommendations
            )

        except Exception as e:
            logger.error(f"Error in Christoffersen test: {e}")
            return ValidationResult(
                test_name="Christoffersen Independence Test",
                test_type="VaR Backtesting",
                result=TestResult.ERROR,
                details={'error': str(e)}
            )

    def traffic_light_test(self, violations: int, confidence_level: float = 0.95) -> ValidationResult:
        """
        Basel III traffic light test for VaR model validation
        """
        try:
            # Determine traffic light zone based on violations in last 250 days
            if violations <= self.traffic_light_zones['green'][1]:
                zone = 'green'
                multiplier = 3.0
                result = TestResult.PASS
            elif violations <= self.traffic_light_zones['yellow'][1]:
                zone = 'yellow'
                multiplier = 3.0 + 0.2 * (violations - 4)  # Penalty factor
                result = TestResult.WARNING
            else:
                zone = 'red'
                multiplier = 4.0  # Maximum penalty
                result = TestResult.FAIL

            recommendations = []
            if zone == 'yellow':
                recommendations.append("Model requires attention - consider recalibration")
                recommendations.append("Monitor closely for additional violations")
            elif zone == 'red':
                recommendations.append("Model requires immediate recalibration")
                recommendations.append("Consider alternative modeling approaches")
                recommendations.append("Increase capital multiplier to 4.0")

            return ValidationResult(
                test_name="Basel III Traffic Light Test",
                test_type="Regulatory Compliance",
                result=result,
                details={
                    'violations': violations,
                    'zone': zone,
                    'capital_multiplier': multiplier,
                    'confidence_level': confidence_level
                },
                recommendations=recommendations
            )

        except Exception as e:
            logger.error(f"Error in traffic light test: {e}")
            return ValidationResult(
                test_name="Basel III Traffic Light Test",
                test_type="Regulatory Compliance",
                result=TestResult.ERROR,
                details={'error': str(e)}
            )

    def validate_var_model(self, returns: np.ndarray, var_forecasts: np.ndarray,
                          confidence_level: float = 0.95) -> BacktestResult:
        """
        Comprehensive VaR model validation
        """
        try:
            returns = np.array(returns)
            var_forecasts = np.array(var_forecasts)

            # Calculate violations (returns worse than VaR forecast)
            violations_array = (returns < -var_forecasts).astype(int)
            violations = np.sum(violations_array)
            observations = len(returns)

            expected_violations = observations * (1 - confidence_level)
            violation_rate = violations / observations
            coverage_rate = 1 - violation_rate

            # Run validation tests
            test_results = []

            # Kupiec test
            kupiec_result = self.kupiec_test(violations, observations, confidence_level)
            test_results.append(kupiec_result)

            # Christoffersen test
            if len(violations_array) > 1:
                christoffersen_result = self.christoffersen_test(violations_array, confidence_level)
                test_results.append(christoffersen_result)

            # Traffic light test (for last 250 observations)
            recent_violations = np.sum(violations_array[-250:]) if len(violations_array) >= 250 else violations
            traffic_light_result = self.traffic_light_test(recent_violations, confidence_level)
            test_results.append(traffic_light_result)

            # Performance metrics
            performance_metrics = {
                'mean_absolute_error': np.mean(np.abs(returns + var_forecasts)),
                'root_mean_squared_error': np.sqrt(np.mean((returns + var_forecasts) ** 2)),
                'hit_rate': violation_rate,
                'average_loss_given_violation': np.mean(returns[violations_array == 1]) if violations > 0 else 0,
                'maximum_loss': np.min(returns),
                'confidence_level': confidence_level
            }

            # Overall model assessment
            period = (datetime.now() - timedelta(days=observations), datetime.now())

            return BacktestResult(
                model_name=f"VaR Model ({confidence_level:.0%})",
                period=period,
                total_observations=observations,
                violations=violations,
                expected_violations=expected_violations,
                violation_rate=violation_rate,
                coverage_rate=coverage_rate,
                test_results=test_results,
                performance_metrics=performance_metrics
            )

        except Exception as e:
            logger.error(f"Error in VaR model validation: {e}")
            raise

class GreeksValidator(BaseValidator):
    """Greeks accuracy validation against market prices"""

    def __init__(self):
        super().__init__("Greeks Validator", "Options Greeks accuracy validation")
        self.tolerance_levels = {
            'delta': 0.05,    # 5% tolerance
            'gamma': 0.10,    # 10% tolerance
            'theta': 0.15,    # 15% tolerance
            'vega': 0.10,     # 10% tolerance
            'rho': 0.20       # 20% tolerance
        }

    def validate_delta_accuracy(self, theoretical_deltas: np.ndarray,
                               market_deltas: np.ndarray) -> ValidationResult:
        """Validate delta accuracy against market prices"""
        try:
            theoretical_deltas = np.array(theoretical_deltas)
            market_deltas = np.array(market_deltas)

            # Calculate accuracy metrics
            differences = theoretical_deltas - market_deltas
            mae = np.mean(np.abs(differences))
            rmse = np.sqrt(np.mean(differences ** 2))
            max_error = np.max(np.abs(differences))

            # Statistical significance test
            t_stat, p_value = stats.ttest_1samp(differences, 0)

            # Tolerance test
            tolerance = self.tolerance_levels['delta']
            within_tolerance = np.mean(np.abs(differences) <= tolerance)

            result = TestResult.PASS if within_tolerance >= 0.95 and p_value > 0.05 else TestResult.FAIL

            recommendations = []
            if result == TestResult.FAIL:
                if mae > tolerance:
                    recommendations.append("Delta calculations show systematic bias")
                    recommendations.append("Review option pricing model parameters")
                if within_tolerance < 0.95:
                    recommendations.append(f"Only {within_tolerance:.1%} of deltas within tolerance")
                    recommendations.append("Consider improving volatility surface modeling")

            return ValidationResult(
                test_name="Delta Accuracy Test",
                test_type="Greeks Validation",
                result=result,
                p_value=p_value,
                statistic=t_stat,
                details={
                    'mean_absolute_error': mae,
                    'root_mean_squared_error': rmse,
                    'maximum_error': max_error,
                    'within_tolerance_rate': within_tolerance,
                    'tolerance_level': tolerance,
                    'sample_size': len(theoretical_deltas)
                },
                recommendations=recommendations
            )

        except Exception as e:
            logger.error(f"Error in delta validation: {e}")
            return ValidationResult(
                test_name="Delta Accuracy Test",
                test_type="Greeks Validation",
                result=TestResult.ERROR,
                details={'error': str(e)}
            )

    def validate_gamma_convexity(self, spot_prices: np.ndarray,
                                theoretical_gammas: np.ndarray,
                                market_prices: np.ndarray) -> ValidationResult:
        """Validate gamma through convexity analysis"""
        try:
            # Calculate second-order price sensitivity
            if len(spot_prices) < 3:
                return ValidationResult(
                    test_name="Gamma Convexity Test",
                    test_type="Greeks Validation",
                    result=TestResult.WARNING,
                    details={'warning': 'Insufficient data for convexity analysis'}
                )

            # Numerical second derivative
            price_changes = np.diff(spot_prices)
            price_change_2nd = np.diff(price_changes)
            option_price_changes = np.diff(market_prices)
            option_price_change_2nd = np.diff(option_price_changes)

            if len(price_change_2nd) > 0 and np.any(price_change_2nd != 0):
                empirical_gamma = option_price_change_2nd / (price_change_2nd ** 2)
                theoretical_gamma_matched = theoretical_gammas[1:-1]  # Match dimensions

                # Compare theoretical vs empirical gamma
                if len(empirical_gamma) > 0 and len(theoretical_gamma_matched) > 0:
                    correlation = np.corrcoef(empirical_gamma, theoretical_gamma_matched)[0, 1]
                    mae = np.mean(np.abs(empirical_gamma - theoretical_gamma_matched))

                    result = TestResult.PASS if correlation > 0.7 and not np.isnan(correlation) else TestResult.FAIL

                    return ValidationResult(
                        test_name="Gamma Convexity Test",
                        test_type="Greeks Validation",
                        result=result,
                        details={
                            'correlation': correlation if not np.isnan(correlation) else 0,
                            'mean_absolute_error': mae,
                            'sample_size': len(empirical_gamma)
                        }
                    )

            return ValidationResult(
                test_name="Gamma Convexity Test",
                test_type="Greeks Validation",
                result=TestResult.WARNING,
                details={'warning': 'Unable to calculate empirical gamma'}
            )

        except Exception as e:
            logger.error(f"Error in gamma validation: {e}")
            return ValidationResult(
                test_name="Gamma Convexity Test",
                test_type="Greeks Validation",
                result=TestResult.ERROR,
                details={'error': str(e)}
            )

class StressTestValidator(BaseValidator):
    """Stress test model validation and calibration"""

    def __init__(self):
        super().__init__("Stress Test Validator", "Stress test model validation")
        self.scenario_types = ['historical', 'hypothetical', 'monte_carlo']

    def validate_scenario_plausibility(self, scenario_results: Dict[str, float],
                                     historical_data: np.ndarray) -> ValidationResult:
        """Validate stress test scenario plausibility"""
        try:
            # Calculate historical volatility and extreme percentiles
            historical_returns = np.diff(np.log(historical_data))
            historical_vol = np.std(historical_returns) * np.sqrt(252)  # Annualized

            # Extreme percentiles
            percentile_1 = np.percentile(historical_returns, 1)
            percentile_99 = np.percentile(historical_returns, 99)

            # Validate scenario magnitudes
            implausible_scenarios = []
            for scenario_name, impact in scenario_results.items():
                # Convert impact to return equivalent
                if abs(impact) > 0:
                    implied_return = impact / 100  # Assuming impact is in percentage

                    # Check if scenario is within historical bounds (with some tolerance)
                    if implied_return < percentile_1 * 2:  # 2x worst historical
                        implausible_scenarios.append(f"{scenario_name}: too extreme")
                    elif abs(implied_return) < historical_vol * 0.1:  # Too small
                        implausible_scenarios.append(f"{scenario_name}: too mild")

            result = TestResult.PASS if len(implausible_scenarios) == 0 else TestResult.WARNING

            recommendations = []
            if implausible_scenarios:
                recommendations.extend([
                    "Review scenario calibration",
                    "Consider historical precedents",
                    "Validate stress test parameters"
                ])

            return ValidationResult(
                test_name="Scenario Plausibility Test",
                test_type="Stress Test Validation",
                result=result,
                details={
                    'historical_volatility': historical_vol,
                    'extreme_percentiles': {'1%': percentile_1, '99%': percentile_99},
                    'implausible_scenarios': implausible_scenarios,
                    'total_scenarios': len(scenario_results)
                },
                recommendations=recommendations
            )

        except Exception as e:
            logger.error(f"Error in scenario plausibility validation: {e}")
            return ValidationResult(
                test_name="Scenario Plausibility Test",
                test_type="Stress Test Validation",
                result=TestResult.ERROR,
                details={'error': str(e)}
            )

    def validate_stress_test_coverage(self, stress_results: Dict[str, float],
                                    portfolio_exposures: Dict[str, float]) -> ValidationResult:
        """Validate stress test coverage of portfolio risks"""
        try:
            # Define required stress categories
            required_categories = {
                'equity_risk', 'interest_rate_risk', 'credit_risk',
                'fx_risk', 'liquidity_risk', 'operational_risk'
            }

            # Map stress test names to categories
            covered_categories = set()
            for scenario_name in stress_results.keys():
                scenario_lower = scenario_name.lower()
                if any(keyword in scenario_lower for keyword in ['equity', 'stock', 'market']):
                    covered_categories.add('equity_risk')
                if any(keyword in scenario_lower for keyword in ['rate', 'yield', 'bond']):
                    covered_categories.add('interest_rate_risk')
                if any(keyword in scenario_lower for keyword in ['credit', 'spread']):
                    covered_categories.add('credit_risk')
                if any(keyword in scenario_lower for keyword in ['fx', 'currency', 'exchange']):
                    covered_categories.add('fx_risk')
                if any(keyword in scenario_lower for keyword in ['liquidity']):
                    covered_categories.add('liquidity_risk')
                if any(keyword in scenario_lower for keyword in ['operational', 'cyber']):
                    covered_categories.add('operational_risk')

            missing_categories = required_categories - covered_categories
            coverage_rate = len(covered_categories) / len(required_categories)

            result = TestResult.PASS if coverage_rate >= 0.8 else TestResult.FAIL

            recommendations = []
            if missing_categories:
                recommendations.append(f"Add stress tests for: {', '.join(missing_categories)}")
                recommendations.append("Ensure comprehensive risk factor coverage")

            return ValidationResult(
                test_name="Stress Test Coverage Analysis",
                test_type="Stress Test Validation",
                result=result,
                details={
                    'coverage_rate': coverage_rate,
                    'covered_categories': list(covered_categories),
                    'missing_categories': list(missing_categories),
                    'total_scenarios': len(stress_results)
                },
                recommendations=recommendations
            )

        except Exception as e:
            logger.error(f"Error in stress test coverage validation: {e}")
            return ValidationResult(
                test_name="Stress Test Coverage Analysis",
                test_type="Stress Test Validation",
                result=TestResult.ERROR,
                details={'error': str(e)}
            )

class MonteCarloValidator(BaseValidator):
    """Monte Carlo model convergence and accuracy testing"""

    def __init__(self):
        super().__init__("Monte Carlo Validator", "Monte Carlo model validation")
        self.min_simulations = 10000
        self.convergence_tolerance = 0.01

    def test_convergence(self, simulation_results: np.ndarray,
                        target_metric: str = 'var') -> ValidationResult:
        """Test Monte Carlo convergence"""
        try:
            simulations = np.array(simulation_results)
            n_sims = len(simulations)

            if n_sims < self.min_simulations:
                return ValidationResult(
                    test_name="Monte Carlo Convergence Test",
                    test_type="Monte Carlo Validation",
                    result=TestResult.WARNING,
                    details={'warning': f'Insufficient simulations: {n_sims} < {self.min_simulations}'}
                )

            # Test convergence by examining running statistics
            convergence_points = []
            batch_size = max(1000, n_sims // 100)

            for i in range(batch_size, n_sims, batch_size):
                if target_metric == 'var':
                    current_metric = np.percentile(simulations[:i], 5)  # 95% VaR
                elif target_metric == 'mean':
                    current_metric = np.mean(simulations[:i])
                elif target_metric == 'std':
                    current_metric = np.std(simulations[:i])
                else:
                    current_metric = np.percentile(simulations[:i], 5)

                convergence_points.append(current_metric)

            # Check for convergence (stable values in last 20% of batches)
            if len(convergence_points) >= 5:
                recent_points = convergence_points[-max(5, len(convergence_points)//5):]
                coefficient_of_variation = np.std(recent_points) / abs(np.mean(recent_points))

                converged = coefficient_of_variation < self.convergence_tolerance

                result = TestResult.PASS if converged else TestResult.FAIL

                recommendations = []
                if not converged:
                    recommendations.append(f"Increase number of simulations (current: {n_sims})")
                    recommendations.append("Results may be unstable")

                return ValidationResult(
                    test_name="Monte Carlo Convergence Test",
                    test_type="Monte Carlo Validation",
                    result=result,
                    details={
                        'coefficient_of_variation': coefficient_of_variation,
                        'tolerance': self.convergence_tolerance,
                        'converged': converged,
                        'simulations': n_sims,
                        'final_metric_value': convergence_points[-1] if convergence_points else None
                    },
                    recommendations=recommendations
                )

            return ValidationResult(
                test_name="Monte Carlo Convergence Test",
                test_type="Monte Carlo Validation",
                result=TestResult.WARNING,
                details={'warning': 'Insufficient convergence test points'}
            )

        except Exception as e:
            logger.error(f"Error in Monte Carlo convergence test: {e}")
            return ValidationResult(
                test_name="Monte Carlo Convergence Test",
                test_type="Monte Carlo Validation",
                result=TestResult.ERROR,
                details={'error': str(e)}
            )

    def test_distribution_properties(self, simulation_results: np.ndarray) -> ValidationResult:
        """Test statistical properties of Monte Carlo distribution"""
        try:
            simulations = np.array(simulation_results)

            # Basic statistical tests
            mean = np.mean(simulations)
            std = np.std(simulations)
            skewness = stats.skew(simulations)
            kurtosis = stats.kurtosis(simulations)

            # Normality test
            if len(simulations) > 20:
                shapiro_stat, shapiro_p = stats.shapiro(simulations[:5000])  # Limit for performance
                jarque_bera_stat, jarque_bera_p = stats.jarque_bera(simulations)
            else:
                shapiro_stat = shapiro_p = jarque_bera_stat = jarque_bera_p = None

            # Test for excessive skewness or kurtosis
            excessive_skewness = abs(skewness) > 2
            excessive_kurtosis = abs(kurtosis) > 7

            distribution_ok = not (excessive_skewness or excessive_kurtosis)
            result = TestResult.PASS if distribution_ok else TestResult.WARNING

            recommendations = []
            if excessive_skewness:
                recommendations.append("Distribution shows excessive skewness - review model assumptions")
            if excessive_kurtosis:
                recommendations.append("Distribution shows fat tails - consider tail risk modeling")

            return ValidationResult(
                test_name="Distribution Properties Test",
                test_type="Monte Carlo Validation",
                result=result,
                details={
                    'mean': mean,
                    'standard_deviation': std,
                    'skewness': skewness,
                    'kurtosis': kurtosis,
                    'shapiro_p_value': shapiro_p,
                    'jarque_bera_p_value': jarque_bera_p,
                    'excessive_skewness': excessive_skewness,
                    'excessive_kurtosis': excessive_kurtosis,
                    'sample_size': len(simulations)
                },
                recommendations=recommendations
            )

        except Exception as e:
            logger.error(f"Error in distribution properties test: {e}")
            return ValidationResult(
                test_name="Distribution Properties Test",
                test_type="Monte Carlo Validation",
                result=TestResult.ERROR,
                details={'error': str(e)}
            )

class ModelValidationFramework:
    """Main model validation framework coordinator"""

    def __init__(self, output_dir: str = "validation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Initialize validators
        self.var_backtester = VaRBacktester()
        self.greeks_validator = GreeksValidator()
        self.stress_test_validator = StressTestValidator()
        self.monte_carlo_validator = MonteCarloValidator()

        self.validation_history: List[Dict] = []

    def run_comprehensive_validation(self, validation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run comprehensive model validation suite"""
        try:
            results = {
                'timestamp': datetime.now().isoformat(),
                'validation_summary': {},
                'detailed_results': {},
                'overall_status': TestResult.PASS
            }

            # VaR Model Validation
            if 'var_data' in validation_data:
                var_data = validation_data['var_data']
                var_results = self.var_backtester.validate_var_model(
                    returns=var_data['returns'],
                    var_forecasts=var_data['var_forecasts'],
                    confidence_level=var_data.get('confidence_level', 0.95)
                )
                results['detailed_results']['var_validation'] = var_results

                # Check if any VaR tests failed
                failed_tests = [test for test in var_results.test_results if test.result == TestResult.FAIL]
                if failed_tests:
                    results['overall_status'] = TestResult.FAIL

            # Greeks Validation
            if 'greeks_data' in validation_data:
                greeks_data = validation_data['greeks_data']
                greeks_results = []

                if 'delta' in greeks_data:
                    delta_result = self.greeks_validator.validate_delta_accuracy(
                        greeks_data['delta']['theoretical'],
                        greeks_data['delta']['market']
                    )
                    greeks_results.append(delta_result)

                results['detailed_results']['greeks_validation'] = greeks_results

                # Check Greeks validation results
                failed_greeks = [test for test in greeks_results if test.result == TestResult.FAIL]
                if failed_greeks:
                    results['overall_status'] = TestResult.FAIL

            # Stress Test Validation
            if 'stress_test_data' in validation_data:
                stress_data = validation_data['stress_test_data']
                stress_results = []

                if 'scenario_results' in stress_data:
                    plausibility_result = self.stress_test_validator.validate_scenario_plausibility(
                        stress_data['scenario_results'],
                        stress_data.get('historical_data', np.random.normal(0, 0.01, 1000))
                    )
                    stress_results.append(plausibility_result)

                if 'portfolio_exposures' in stress_data:
                    coverage_result = self.stress_test_validator.validate_stress_test_coverage(
                        stress_data['scenario_results'],
                        stress_data['portfolio_exposures']
                    )
                    stress_results.append(coverage_result)

                results['detailed_results']['stress_test_validation'] = stress_results

            # Monte Carlo Validation
            if 'monte_carlo_data' in validation_data:
                mc_data = validation_data['monte_carlo_data']
                mc_results = []

                convergence_result = self.monte_carlo_validator.test_convergence(
                    mc_data['simulation_results']
                )
                mc_results.append(convergence_result)

                distribution_result = self.monte_carlo_validator.test_distribution_properties(
                    mc_data['simulation_results']
                )
                mc_results.append(distribution_result)

                results['detailed_results']['monte_carlo_validation'] = mc_results

                # Check Monte Carlo results
                failed_mc = [test for test in mc_results if test.result == TestResult.FAIL]
                if failed_mc:
                    results['overall_status'] = TestResult.FAIL

            # Generate summary
            results['validation_summary'] = self._generate_validation_summary(results)

            # Save results
            self._save_validation_results(results)

            return results

        except Exception as e:
            logger.error(f"Error in comprehensive validation: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'overall_status': TestResult.ERROR
            }

    def _generate_validation_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate validation summary"""
        summary = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'warning_tests': 0,
            'error_tests': 0,
            'pass_rate': 0.0,
            'critical_issues': [],
            'recommendations': []
        }

        # Count test results
        for validation_type, validation_results in results['detailed_results'].items():
            if isinstance(validation_results, BacktestResult):
                test_results = validation_results.test_results
            elif isinstance(validation_results, list):
                test_results = validation_results
            else:
                continue

            for test in test_results:
                summary['total_tests'] += 1
                if test.result == TestResult.PASS:
                    summary['passed_tests'] += 1
                elif test.result == TestResult.FAIL:
                    summary['failed_tests'] += 1
                    summary['critical_issues'].append(f"{test.test_name}: {test.test_type}")
                elif test.result == TestResult.WARNING:
                    summary['warning_tests'] += 1
                elif test.result == TestResult.ERROR:
                    summary['error_tests'] += 1

                # Collect recommendations
                if test.recommendations:
                    summary['recommendations'].extend(test.recommendations)

        # Calculate pass rate
        if summary['total_tests'] > 0:
            summary['pass_rate'] = summary['passed_tests'] / summary['total_tests']

        return summary

    def _save_validation_results(self, results: Dict[str, Any]):
        """Save validation results to file"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = self.output_dir / f"validation_results_{timestamp}.json"

            # Convert complex objects to serializable format
            serializable_results = self._make_serializable(results)

            with open(filename, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)

            logger.info(f"Validation results saved to {filename}")

        except Exception as e:
            logger.error(f"Error saving validation results: {e}")

    def _make_serializable(self, obj):
        """Convert objects to JSON serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (ValidationResult, BacktestResult)):
            return obj.__dict__
        elif isinstance(obj, TestResult):
            return obj.value
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            return obj

    def generate_validation_report(self, results: Dict[str, Any]) -> str:
        """Generate validation report"""
        try:
            report = []
            report.append("=" * 80)
            report.append("MODEL VALIDATION REPORT")
            report.append("=" * 80)
            report.append(f"Generated: {results['timestamp']}")
            report.append(f"Overall Status: {results['overall_status']}")
            report.append("")

            # Summary section
            summary = results['validation_summary']
            report.append("VALIDATION SUMMARY")
            report.append("-" * 40)
            report.append(f"Total Tests: {summary['total_tests']}")
            report.append(f"Passed: {summary['passed_tests']}")
            report.append(f"Failed: {summary['failed_tests']}")
            report.append(f"Warnings: {summary['warning_tests']}")
            report.append(f"Errors: {summary['error_tests']}")
            report.append(f"Pass Rate: {summary['pass_rate']:.1%}")
            report.append("")

            # Critical issues
            if summary['critical_issues']:
                report.append("CRITICAL ISSUES")
                report.append("-" * 40)
                for issue in summary['critical_issues']:
                    report.append(f"• {issue}")
                report.append("")

            # Recommendations
            if summary['recommendations']:
                report.append("RECOMMENDATIONS")
                report.append("-" * 40)
                for rec in set(summary['recommendations']):  # Remove duplicates
                    report.append(f"• {rec}")
                report.append("")

            # Detailed results
            report.append("DETAILED RESULTS")
            report.append("-" * 40)

            for validation_type, validation_results in results['detailed_results'].items():
                report.append(f"\n{validation_type.upper().replace('_', ' ')}")
                report.append("~" * 30)

                if isinstance(validation_results, BacktestResult):
                    report.append(f"Model: {validation_results.model_name}")
                    report.append(f"Period: {validation_results.period[0].date()} to {validation_results.period[1].date()}")
                    report.append(f"Observations: {validation_results.total_observations}")
                    report.append(f"Violations: {validation_results.violations}")
                    report.append(f"Expected Violations: {validation_results.expected_violations:.1f}")
                    report.append(f"Violation Rate: {validation_results.violation_rate:.2%}")
                    report.append("")

                    for test in validation_results.test_results:
                        report.append(f"  {test.test_name}: {test.result.value}")
                        if test.p_value is not None:
                            report.append(f"    P-value: {test.p_value:.4f}")
                        report.append("")

                elif isinstance(validation_results, list):
                    for test in validation_results:
                        report.append(f"  {test.test_name}: {test.result.value}")
                        if test.p_value is not None:
                            report.append(f"    P-value: {test.p_value:.4f}")
                        if test.recommendations:
                            report.append("    Recommendations:")
                            for rec in test.recommendations:
                                report.append(f"      • {rec}")
                        report.append("")

            return "\n".join(report)

        except Exception as e:
            logger.error(f"Error generating validation report: {e}")
            return f"Error generating report: {e}"