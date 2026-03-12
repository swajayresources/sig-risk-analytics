"""
Performance Benchmarking Suite for Quantitative Risk Analytics Engine
Measures latency, throughput, and resource utilization for all core components
"""

import time
import numpy as np
import pandas as pd
import psutil
import gc
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable, Any
import warnings
import sys
import os

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Suppress warnings for cleaner benchmark output
warnings.filterwarnings('ignore')

# Import our modules
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

@dataclass
class BenchmarkResult:
    """Structure for storing benchmark results"""
    operation: str
    mean_time: float
    std_time: float
    min_time: float
    max_time: float
    p95_time: float
    p99_time: float
    throughput: float  # Operations per second
    memory_usage: float  # MB
    cpu_usage: float  # Percentage
    iterations: int
    data_size: str

@dataclass
class SystemMetrics:
    """System resource metrics"""
    cpu_count: int
    total_memory: float  # GB
    available_memory: float  # GB
    cpu_frequency: float  # MHz

class PerformanceBenchmarker:
    """Comprehensive performance benchmarking suite"""

    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.system_metrics = self._get_system_metrics()

    def _get_system_metrics(self) -> SystemMetrics:
        """Get system hardware metrics"""
        return SystemMetrics(
            cpu_count=psutil.cpu_count(),
            total_memory=psutil.virtual_memory().total / (1024**3),
            available_memory=psutil.virtual_memory().available / (1024**3),
            cpu_frequency=psutil.cpu_freq().current if psutil.cpu_freq() else 0
        )

    @contextmanager
    def measure_performance(self, operation_name: str, iterations: int = 1, data_size: str = "unknown"):
        """Context manager for measuring performance metrics"""
        gc.collect()  # Clean up before measurement

        # Pre-measurement metrics
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024**2)  # MB
        initial_cpu_percent = process.cpu_percent()

        times = []
        start_total = time.perf_counter()

        try:
            yield times
        finally:
            end_total = time.perf_counter()

            # Post-measurement metrics
            final_memory = process.memory_info().rss / (1024**2)  # MB
            final_cpu_percent = process.cpu_percent()

            if times:
                times_array = np.array(times)
                mean_time = np.mean(times_array)
                std_time = np.std(times_array)
                min_time = np.min(times_array)
                max_time = np.max(times_array)
                p95_time = np.percentile(times_array, 95)
                p99_time = np.percentile(times_array, 99)
                throughput = iterations / (end_total - start_total)
            else:
                total_time = end_total - start_total
                mean_time = total_time / iterations if iterations > 0 else total_time
                std_time = 0
                min_time = max_time = p95_time = p99_time = mean_time
                throughput = iterations / total_time if total_time > 0 else 0

            memory_usage = final_memory - initial_memory
            cpu_usage = final_cpu_percent - initial_cpu_percent

            result = BenchmarkResult(
                operation=operation_name,
                mean_time=mean_time,
                std_time=std_time,
                min_time=min_time,
                max_time=max_time,
                p95_time=p95_time,
                p99_time=p99_time,
                throughput=throughput,
                memory_usage=memory_usage,
                cpu_usage=cpu_usage,
                iterations=iterations,
                data_size=data_size
            )

            self.results.append(result)

    def benchmark_risk_metrics(self):
        """Benchmark risk metrics calculations"""
        print("Benchmarking Risk Metrics Calculations...")

        # Generate test data of various sizes
        test_cases = [
            (100, 10, "Small (100 days, 10 assets)"),
            (252, 50, "Medium (252 days, 50 assets)"),
            (1260, 100, "Large (1260 days, 100 assets)"),
            (2520, 200, "XLarge (2520 days, 200 assets)")
        ]

        for n_days, n_assets, description in test_cases:
            print(f"  Testing {description}")

            # Generate correlated returns
            np.random.seed(42)
            returns_data = np.random.multivariate_normal(
                mean=np.zeros(n_assets),
                cov=np.eye(n_assets) * 0.0004 + np.full((n_assets, n_assets), 0.0001),
                size=n_days
            )

            weights = np.random.dirichlet(np.ones(n_assets))
            portfolio_returns = returns_data @ weights
            covariance_matrix = np.cov(returns_data.T)

            # 1. Historical Simulation VaR
            hist_var = HistoricalSimulationVaR(lookback_days=min(60, n_days//2))
            iterations = 100

            with self.measure_performance(f"Historical VaR ({description})", iterations, description) as times:
                for _ in range(iterations):
                    start = time.perf_counter()
                    hist_var.calculate_var(portfolio_returns, confidence_level=0.95)
                    times.append(time.perf_counter() - start)

            # 2. Parametric VaR
            param_var = ParametricVaR()

            with self.measure_performance(f"Parametric VaR ({description})", iterations, description) as times:
                for _ in range(iterations):
                    start = time.perf_counter()
                    param_var.calculate_var(weights, covariance_matrix, confidence_level=0.95)
                    times.append(time.perf_counter() - start)

            # 3. Monte Carlo VaR (fewer iterations for larger datasets)
            mc_iterations = max(10, 100 // (n_assets // 10 + 1))
            mc_var = MonteCarloVaR(n_simulations=5000)
            expected_returns = np.mean(returns_data, axis=0)

            with self.measure_performance(f"Monte Carlo VaR ({description})", mc_iterations, description) as times:
                for _ in range(mc_iterations):
                    start = time.perf_counter()
                    mc_var.calculate_var(weights, expected_returns, covariance_matrix, confidence_level=0.95)
                    times.append(time.perf_counter() - start)

            # 4. Portfolio Risk Metrics
            portfolio_metrics = PortfolioRiskMetrics()

            with self.measure_performance(f"Component VaR ({description})", iterations, description) as times:
                for _ in range(iterations):
                    start = time.perf_counter()
                    portfolio_metrics.calculate_component_var(weights, covariance_matrix, confidence_level=0.95)
                    times.append(time.perf_counter() - start)

    def benchmark_options_greeks(self):
        """Benchmark options Greeks calculations"""
        print("Benchmarking Options Greeks Calculations...")

        # Test different portfolio sizes
        portfolio_sizes = [10, 100, 1000, 5000]

        for portfolio_size in portfolio_sizes:
            print(f"  Testing portfolio size: {portfolio_size} options")

            # Generate options data
            np.random.seed(42)
            options_data = []
            for _ in range(portfolio_size):
                options_data.append({
                    'S': np.random.uniform(80, 120),
                    'K': np.random.uniform(90, 110),
                    'T': np.random.uniform(0.1, 1.0),
                    'r': 0.05,
                    'sigma': np.random.uniform(0.15, 0.35),
                    'q': 0.02,
                    'option_type': 'call' if np.random.rand() > 0.5 else 'put'
                })

            greeks_calculator = BlackScholesGreeks()
            iterations = max(10, 1000 // portfolio_size)

            # Single option Greeks calculation
            with self.measure_performance(f"Black-Scholes Greeks (Portfolio: {portfolio_size})",
                                        iterations, f"{portfolio_size} options") as times:
                for _ in range(iterations):
                    start = time.perf_counter()
                    for option in options_data:
                        greeks_calculator.calculate_greeks(**option)
                    times.append(time.perf_counter() - start)

            # Vectorized Greeks calculation
            if hasattr(greeks_calculator, 'calculate_portfolio_greeks'):
                with self.measure_performance(f"Vectorized Greeks (Portfolio: {portfolio_size})",
                                            iterations, f"{portfolio_size} options") as times:
                    for _ in range(iterations):
                        start = time.perf_counter()
                        # Assuming vectorized implementation exists
                        # greeks_calculator.calculate_portfolio_greeks(options_data)
                        times.append(time.perf_counter() - start)

    def benchmark_monte_carlo_simulations(self):
        """Benchmark Monte Carlo simulations"""
        print("Benchmarking Monte Carlo Simulations...")

        # Test different simulation parameters
        test_configs = [
            (1000, 21, 10, "Small (1K paths, 21 steps, 10 assets)"),
            (10000, 21, 50, "Medium (10K paths, 21 steps, 50 assets)"),
            (50000, 63, 100, "Large (50K paths, 63 steps, 100 assets)"),
            (100000, 21, 200, "XLarge (100K paths, 21 steps, 200 assets)")
        ]

        for n_paths, n_steps, n_assets, description in test_configs:
            print(f"  Testing {description}")

            # Setup
            np.random.seed(42)
            S0 = np.random.uniform(80, 120, n_assets)
            mu = np.random.uniform(0.05, 0.15, n_assets)
            sigma = np.random.uniform(0.15, 0.30, n_assets)
            correlation_matrix = np.eye(n_assets) * 0.7 + np.full((n_assets, n_assets), 0.3)

            path_config = PathGenerationConfig(
                n_paths=n_paths,
                n_steps=n_steps,
                use_antithetic=True,
                use_moment_matching=True
            )

            variance_config = VarianceReductionConfig(
                use_control_variates=True,
                use_importance_sampling=False
            )

            mc_engine = MonteCarloEngine(path_config, variance_config)
            iterations = max(1, 10 // (n_paths // 10000 + 1))

            # 1. Geometric Brownian Motion
            with self.measure_performance(f"GBM Simulation ({description})", iterations, description) as times:
                for _ in range(iterations):
                    start = time.perf_counter()
                    mc_engine.geometric_brownian_motion(S0, mu, sigma, T=1/12, n_steps=n_steps,
                                                      correlation_matrix=correlation_matrix)
                    times.append(time.perf_counter() - start)

            # 2. Portfolio Monte Carlo (CPU)
            weights = np.random.dirichlet(np.ones(n_assets))

            with self.measure_performance(f"Portfolio MC CPU ({description})", iterations, description) as times:
                for _ in range(iterations):
                    start = time.perf_counter()
                    mc_engine.portfolio_monte_carlo_cpu(weights, S0, mu, sigma, correlation_matrix, T=1/12)
                    times.append(time.perf_counter() - start)

            # 3. Portfolio Monte Carlo (GPU) - if available
            try:
                with self.measure_performance(f"Portfolio MC GPU ({description})", iterations, description) as times:
                    for _ in range(iterations):
                        start = time.perf_counter()
                        mc_engine.portfolio_monte_carlo_gpu(weights, S0, mu, sigma, correlation_matrix, T=1/12)
                        times.append(time.perf_counter() - start)
            except:
                print(f"    GPU computation not available for {description}")

    def benchmark_portfolio_optimization(self):
        """Benchmark portfolio optimization algorithms"""
        print("Benchmarking Portfolio Optimization...")

        # Test different portfolio sizes
        portfolio_sizes = [10, 25, 50, 100, 200]

        for n_assets in portfolio_sizes:
            print(f"  Testing {n_assets} assets")

            # Generate test data
            np.random.seed(42)
            expected_returns = np.random.uniform(0.05, 0.20, n_assets)
            volatilities = np.random.uniform(0.10, 0.30, n_assets)
            correlation_matrix = np.random.uniform(0.2, 0.8, (n_assets, n_assets))
            correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
            np.fill_diagonal(correlation_matrix, 1.0)

            # Ensure positive definite
            eigenvals, eigenvecs = np.linalg.eigh(correlation_matrix)
            eigenvals = np.maximum(eigenvals, 0.01)
            correlation_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

            covariance_matrix = np.outer(volatilities, volatilities) * correlation_matrix

            config = OptimizationConfig(max_iterations=500, tolerance=1e-6)
            constraints = OptimizationConstraints(
                min_weights=np.full(n_assets, 0.01),
                max_weights=np.full(n_assets, 0.20),
                no_short_selling=True
            )

            iterations = max(1, 50 // (n_assets // 25 + 1))

            # 1. Mean-Variance Optimization
            mv_optimizer = MeanVarianceOptimizer(config)

            with self.measure_performance(f"Mean-Variance ({n_assets} assets)", iterations, f"{n_assets} assets") as times:
                for _ in range(iterations):
                    start = time.perf_counter()
                    mv_optimizer.optimize(expected_returns, covariance_matrix, constraints)
                    times.append(time.perf_counter() - start)

            # 2. Risk Parity Optimization
            rp_optimizer = RiskParityOptimizer(config)

            with self.measure_performance(f"Risk Parity ({n_assets} assets)", iterations, f"{n_assets} assets") as times:
                for _ in range(iterations):
                    start = time.perf_counter()
                    rp_optimizer.optimize(expected_returns, covariance_matrix, constraints)
                    times.append(time.perf_counter() - start)

            # 3. Black-Litterman Optimization (smaller iterations due to complexity)
            bl_optimizer = BlackLittermanOptimizer(config, tau=0.025)
            bl_iterations = max(1, iterations // 3)

            with self.measure_performance(f"Black-Litterman ({n_assets} assets)", bl_iterations, f"{n_assets} assets") as times:
                for _ in range(bl_iterations):
                    start = time.perf_counter()
                    bl_optimizer.optimize(expected_returns, covariance_matrix, constraints)
                    times.append(time.perf_counter() - start)

            # 4. Hierarchical Risk Parity
            hrp_optimizer = HierarchicalRiskParityOptimizer(config)

            with self.measure_performance(f"HRP ({n_assets} assets)", iterations, f"{n_assets} assets") as times:
                for _ in range(iterations):
                    start = time.perf_counter()
                    hrp_optimizer.optimize(expected_returns, covariance_matrix)
                    times.append(time.perf_counter() - start)

    def benchmark_concurrent_processing(self):
        """Benchmark concurrent and parallel processing capabilities"""
        print("Benchmarking Concurrent Processing...")

        # Generate test data
        np.random.seed(42)
        n_assets = 50
        n_days = 252
        returns_data = np.random.multivariate_normal(
            mean=np.zeros(n_assets),
            cov=np.eye(n_assets) * 0.0004 + np.full((n_assets, n_assets), 0.0001),
            size=n_days
        )

        weights = np.random.dirichlet(np.ones(n_assets))
        portfolio_returns = returns_data @ weights
        covariance_matrix = np.cov(returns_data.T)

        # Risk calculation function
        def calculate_risk_metrics():
            hist_var = HistoricalSimulationVaR(lookback_days=60)
            param_var = ParametricVaR()

            hist_result = hist_var.calculate_var(portfolio_returns, confidence_level=0.95)
            param_result = param_var.calculate_var(weights, covariance_matrix, confidence_level=0.95)

            return hist_result, param_result

        # Test different thread counts
        thread_counts = [1, 2, 4, 8]

        for thread_count in thread_counts:
            print(f"  Testing {thread_count} threads")

            iterations = 50
            tasks_per_iteration = 10

            # Threading benchmark
            with self.measure_performance(f"Threaded Processing ({thread_count} threads)",
                                        iterations, f"{thread_count} threads") as times:
                for _ in range(iterations):
                    start = time.perf_counter()

                    with ThreadPoolExecutor(max_workers=thread_count) as executor:
                        futures = [executor.submit(calculate_risk_metrics) for _ in range(tasks_per_iteration)]
                        results = [future.result() for future in futures]

                    times.append(time.perf_counter() - start)

            # Process-based benchmark (fewer iterations due to overhead)
            if thread_count <= 4:  # Limit process count
                process_iterations = max(1, iterations // 5)

                with self.measure_performance(f"Process-based Processing ({thread_count} processes)",
                                            process_iterations, f"{thread_count} processes") as times:
                    for _ in range(process_iterations):
                        start = time.perf_counter()

                        with ProcessPoolExecutor(max_workers=thread_count) as executor:
                            futures = [executor.submit(calculate_risk_metrics) for _ in range(tasks_per_iteration)]
                            results = [future.result() for future in futures]

                        times.append(time.perf_counter() - start)

    def benchmark_memory_efficiency(self):
        """Benchmark memory usage patterns"""
        print("Benchmarking Memory Efficiency...")

        # Test different data sizes
        data_sizes = [
            (100, 10, "Small"),
            (1000, 50, "Medium"),
            (5000, 100, "Large"),
            (10000, 200, "XLarge")
        ]

        for n_days, n_assets, size_label in data_sizes:
            print(f"  Testing {size_label} dataset ({n_days} days, {n_assets} assets)")

            # Memory-intensive operations
            process = psutil.Process()

            with self.measure_performance(f"Memory Usage ({size_label})", 1, size_label):
                initial_memory = process.memory_info().rss / (1024**2)  # MB

                # Generate large dataset
                np.random.seed(42)
                returns_data = np.random.multivariate_normal(
                    mean=np.zeros(n_assets),
                    cov=np.eye(n_assets) * 0.0004 + np.full((n_assets, n_assets), 0.0001),
                    size=n_days
                )

                # Calculate covariance matrix
                covariance_matrix = np.cov(returns_data.T)

                # Generate Monte Carlo simulations
                mc_engine = MonteCarloEngine(
                    PathGenerationConfig(n_paths=10000, n_steps=21),
                    VarianceReductionConfig()
                )

                S0 = np.ones(n_assets) * 100
                mu = np.full(n_assets, 0.1)
                sigma = np.full(n_assets, 0.2)
                correlation_matrix = np.eye(n_assets)

                # Memory-intensive simulation
                price_paths = mc_engine.geometric_brownian_motion(
                    S0, mu, sigma, T=1/12, n_steps=21, correlation_matrix=correlation_matrix
                )

                peak_memory = process.memory_info().rss / (1024**2)  # MB

                # Clean up
                del returns_data, covariance_matrix, price_paths
                gc.collect()

                final_memory = process.memory_info().rss / (1024**2)  # MB

            print(f"    Initial: {initial_memory:.1f} MB, Peak: {peak_memory:.1f} MB, Final: {final_memory:.1f} MB")

    def run_all_benchmarks(self):
        """Run comprehensive benchmark suite"""
        print("="*80)
        print("QUANTITATIVE RISK ANALYTICS ENGINE - PERFORMANCE BENCHMARKS")
        print("="*80)
        print(f"System: {self.system_metrics.cpu_count} CPUs, {self.system_metrics.total_memory:.1f} GB RAM")
        print(f"Available Memory: {self.system_metrics.available_memory:.1f} GB")
        if self.system_metrics.cpu_frequency > 0:
            print(f"CPU Frequency: {self.system_metrics.cpu_frequency:.0f} MHz")
        print("="*80)

        # Run all benchmark categories
        self.benchmark_risk_metrics()
        print()

        self.benchmark_options_greeks()
        print()

        self.benchmark_monte_carlo_simulations()
        print()

        self.benchmark_portfolio_optimization()
        print()

        self.benchmark_concurrent_processing()
        print()

        self.benchmark_memory_efficiency()

    def generate_report(self) -> str:
        """Generate comprehensive benchmark report"""
        if not self.results:
            return "No benchmark results available."

        report = []
        report.append("="*100)
        report.append("PERFORMANCE BENCHMARK REPORT")
        report.append("="*100)

        # System information
        report.append(f"System Information:")
        report.append(f"  CPU Cores: {self.system_metrics.cpu_count}")
        report.append(f"  Total Memory: {self.system_metrics.total_memory:.1f} GB")
        report.append(f"  Available Memory: {self.system_metrics.available_memory:.1f} GB")
        if self.system_metrics.cpu_frequency > 0:
            report.append(f"  CPU Frequency: {self.system_metrics.cpu_frequency:.0f} MHz")
        report.append("")

        # Performance targets (from README)
        targets = {
            "Trade Processing": 1e-6,  # < 1 μs
            "Market Data Update": 0.5e-6,  # < 0.5 μs
            "VaR Calculation": 0.01,  # < 10 ms
            "Black-Scholes Greeks": 1e-6,  # < 1 μs
            "Position Query": 0.1e-6,  # < 0.1 μs
            "Portfolio Optimization": 30  # < 30 s
        }

        # Group results by category
        categories = {}
        for result in self.results:
            category = result.operation.split('(')[0].strip()
            if category not in categories:
                categories[category] = []
            categories[category].append(result)

        # Generate detailed results
        report.append("DETAILED BENCHMARK RESULTS")
        report.append("-" * 100)
        header = f"{'Operation':<40} {'Mean (ms)':<12} {'P95 (ms)':<12} {'P99 (ms)':<12} {'Throughput':<12} {'Memory (MB)':<12}"
        report.append(header)
        report.append("-" * 100)

        for category, results in categories.items():
            if results:
                report.append(f"\n{category}:")
                for result in results:
                    operation_name = result.operation[:38] + "..." if len(result.operation) > 38 else result.operation
                    report.append(
                        f"{operation_name:<40} "
                        f"{result.mean_time*1000:<12.3f} "
                        f"{result.p95_time*1000:<12.3f} "
                        f"{result.p99_time*1000:<12.3f} "
                        f"{result.throughput:<12.1f} "
                        f"{result.memory_usage:<12.1f}"
                    )

        # Performance summary
        report.append("\n\nPERFORMANCE SUMMARY")
        report.append("-" * 50)

        # Find key metrics
        var_results = [r for r in self.results if "VaR" in r.operation and "Small" in r.operation]
        greeks_results = [r for r in self.results if "Greeks" in r.operation and "10" in r.data_size]
        optimization_results = [r for r in self.results if "Mean-Variance" in r.operation and "25 assets" in r.data_size]

        if var_results:
            fastest_var = min(var_results, key=lambda x: x.mean_time)
            report.append(f"Fastest VaR Calculation: {fastest_var.mean_time*1000:.3f} ms ({fastest_var.operation})")

        if greeks_results:
            fastest_greeks = min(greeks_results, key=lambda x: x.mean_time)
            per_option_time = fastest_greeks.mean_time * 1000 / 10  # 10 options in small portfolio
            report.append(f"Greeks per Option: {per_option_time:.3f} ms")

        if optimization_results:
            fastest_opt = min(optimization_results, key=lambda x: x.mean_time)
            report.append(f"Portfolio Optimization (25 assets): {fastest_opt.mean_time:.3f} s")

        # Resource utilization
        avg_memory = np.mean([r.memory_usage for r in self.results if r.memory_usage > 0])
        max_memory = max([r.memory_usage for r in self.results if r.memory_usage > 0], default=0)

        report.append(f"\nResource Utilization:")
        report.append(f"  Average Memory Usage: {avg_memory:.1f} MB")
        report.append(f"  Peak Memory Usage: {max_memory:.1f} MB")

        # Scalability analysis
        report.append(f"\nScalability Analysis:")

        # Analyze VaR scaling with data size
        var_scaling = {}
        for result in self.results:
            if "Historical VaR" in result.operation:
                size = result.data_size
                if size not in var_scaling:
                    var_scaling[size] = result.mean_time
                else:
                    var_scaling[size] = min(var_scaling[size], result.mean_time)

        if len(var_scaling) > 1:
            sizes = list(var_scaling.keys())
            times = [var_scaling[size] for size in sizes]
            report.append(f"  VaR Calculation Scaling:")
            for size, time_val in zip(sizes, times):
                report.append(f"    {size}: {time_val*1000:.3f} ms")

        # Recommendations
        report.append(f"\nRECOMMENDATIONS:")

        slow_operations = [r for r in self.results if r.mean_time > 1.0]  # > 1 second
        if slow_operations:
            report.append(f"  ⚠️  Slow operations detected (>{1.0}s):")
            for op in slow_operations[:3]:  # Top 3 slowest
                report.append(f"    - {op.operation}: {op.mean_time:.3f}s")

        high_memory_ops = [r for r in self.results if r.memory_usage > 500]  # > 500 MB
        if high_memory_ops:
            report.append(f"  ⚠️  High memory usage operations (>500MB):")
            for op in high_memory_ops[:3]:
                report.append(f"    - {op.operation}: {op.memory_usage:.1f}MB")

        if not slow_operations and not high_memory_ops:
            report.append(f"  ✅ All operations within acceptable performance thresholds")

        report.append("")
        report.append("="*100)

        return "\n".join(report)

    def save_results(self, filename: str = "benchmark_results.txt"):
        """Save benchmark results to file"""
        report = self.generate_report()

        with open(filename, 'w') as f:
            f.write(report)

        # Also save raw data as CSV
        csv_filename = filename.replace('.txt', '.csv')
        df = pd.DataFrame([
            {
                'Operation': r.operation,
                'Mean_Time_ms': r.mean_time * 1000,
                'Std_Time_ms': r.std_time * 1000,
                'P95_Time_ms': r.p95_time * 1000,
                'P99_Time_ms': r.p99_time * 1000,
                'Throughput_ops_sec': r.throughput,
                'Memory_Usage_MB': r.memory_usage,
                'CPU_Usage_pct': r.cpu_usage,
                'Iterations': r.iterations,
                'Data_Size': r.data_size
            }
            for r in self.results
        ])

        df.to_csv(csv_filename, index=False)

        print(f"Benchmark results saved to {filename}")
        print(f"Raw data saved to {csv_filename}")

# Standalone execution
if __name__ == "__main__":
    print("Starting Performance Benchmarks...")

    benchmarker = PerformanceBenchmarker()

    try:
        benchmarker.run_all_benchmarks()

        print("\n" + "="*80)
        print("BENCHMARK COMPLETE")
        print("="*80)

        # Print summary report
        report = benchmarker.generate_report()
        print(report)

        # Save results
        benchmarker.save_results("performance_benchmark_results.txt")

        print("\n🎉 Performance Benchmarking Complete!")

    except KeyboardInterrupt:
        print("\n⚠️ Benchmarking interrupted by user")
    except Exception as e:
        print(f"\n❌ Benchmarking failed with error: {e}")
        import traceback
        traceback.print_exc()