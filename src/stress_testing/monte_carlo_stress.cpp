#include "stress_testing/stress_framework.hpp"
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <execution>
#include <fstream>

namespace stress_testing {

class MonteCarloStressTesterImpl {
private:
    std::mt19937_64 rng_;
    std::normal_distribution<double> normal_dist_;
    std::uniform_real_distribution<double> uniform_dist_;
    std::gamma_distribution<double> gamma_dist_;

    // Variance reduction techniques
    bool use_antithetic_variates_;
    bool use_control_variates_;
    bool use_importance_sampling_;

    // Cached correlation matrices and decompositions
    mutable std::mutex correlation_cache_mutex_;
    std::unordered_map<std::string, Eigen::MatrixXd> cholesky_cache_;
    std::unordered_map<std::string, Eigen::MatrixXd> correlation_cache_;

public:
    MonteCarloStressTesterImpl() :
        rng_(std::random_device{}()),
        normal_dist_(0.0, 1.0),
        uniform_dist_(0.0, 1.0),
        gamma_dist_(2.0, 1.0),
        use_antithetic_variates_(true),
        use_control_variates_(false),
        use_importance_sampling_(false) {}

    void configure_variance_reduction(bool antithetic, bool control, bool importance) {
        use_antithetic_variates_ = antithetic;
        use_control_variates_ = control;
        use_importance_sampling_ = importance;
    }

    MonteCarloResults run_monte_carlo_stress_test(
        const std::vector<RiskFactor>& factors,
        const MonteCarloStressParameters& params) {

        MonteCarloResults results;
        results.num_scenarios = params.num_scenarios;
        results.confidence_levels = params.confidence_levels;
        results.time_horizon_days = params.time_horizon_days;

        // Prepare correlation matrix
        Eigen::MatrixXd correlation_matrix = prepare_correlation_matrix(factors, params);
        Eigen::MatrixXd cholesky = get_cholesky_decomposition(correlation_matrix);

        // Pre-allocate results storage
        std::vector<std::vector<double>> all_pnl_paths(params.num_scenarios);
        std::vector<double> final_pnls(params.num_scenarios);
        std::vector<double> max_drawdowns(params.num_scenarios);

        // Parallel Monte Carlo simulation
        std::mutex results_mutex;
        std::atomic<int> completed_scenarios{0};

        auto run_scenario = [&](int scenario_idx) {
            // Create thread-local random generator
            std::mt19937_64 local_rng(params.random_seed + scenario_idx);
            std::normal_distribution<double> local_normal(0.0, 1.0);

            // Generate correlated shocks for all time steps
            std::vector<std::vector<double>> scenario_shocks;
            generate_correlated_time_series(scenario_shocks, factors, params, cholesky, local_rng, local_normal);

            // Calculate P&L path for this scenario
            std::vector<double> pnl_path;
            double max_drawdown = calculate_scenario_pnl(
                scenario_shocks, factors, params, pnl_path);

            // Store results (thread-safe)
            {
                std::lock_guard<std::mutex> lock(results_mutex);
                all_pnl_paths[scenario_idx] = std::move(pnl_path);
                final_pnls[scenario_idx] = all_pnl_paths[scenario_idx].back();
                max_drawdowns[scenario_idx] = max_drawdown;
            }

            completed_scenarios.fetch_add(1);

            // Progress reporting
            if (completed_scenarios % 1000 == 0) {
                double progress = static_cast<double>(completed_scenarios) / params.num_scenarios * 100.0;
                std::cout << "Monte Carlo progress: " << progress << "%\r" << std::flush;
            }
        };

        // Execute scenarios in parallel
        std::vector<int> scenario_indices(params.num_scenarios);
        std::iota(scenario_indices.begin(), scenario_indices.end(), 0);

        std::for_each(std::execution::par_unseq, scenario_indices.begin(), scenario_indices.end(), run_scenario);

        std::cout << std::endl; // New line after progress reporting

        // Calculate statistics
        calculate_monte_carlo_statistics(results, final_pnls, max_drawdowns, all_pnl_paths, params);

        // Generate percentile paths
        generate_percentile_paths(results, all_pnl_paths, params);

        return results;
    }

    CorrelatedSamples generate_correlated_samples(
        const std::vector<RiskFactor>& factors,
        const Eigen::MatrixXd& correlation_matrix,
        int num_samples,
        uint64_t seed = 0) {

        CorrelatedSamples samples;
        samples.factor_names.reserve(factors.size());
        for (const auto& factor : factors) {
            samples.factor_names.push_back(factor.name);
        }

        size_t num_factors = factors.size();
        samples.samples.resize(num_samples, std::vector<double>(num_factors));

        // Get Cholesky decomposition
        Eigen::MatrixXd cholesky = get_cholesky_decomposition(correlation_matrix);

        // Generate samples
        std::mt19937_64 local_rng(seed != 0 ? seed : rng_());
        std::normal_distribution<double> local_normal(0.0, 1.0);

        for (int i = 0; i < num_samples; ++i) {
            // Generate independent samples
            Eigen::VectorXd independent_samples(num_factors);
            for (size_t j = 0; j < num_factors; ++j) {
                independent_samples(j) = local_normal(local_rng);
            }

            // Apply correlation structure
            Eigen::VectorXd correlated_samples = cholesky * independent_samples;

            // Store in results
            for (size_t j = 0; j < num_factors; ++j) {
                samples.samples[i][j] = correlated_samples(j);
            }
        }

        // Calculate sample statistics
        calculate_sample_statistics(samples);

        return samples;
    }

    TailRiskMetrics calculate_tail_risk_metrics(
        const std::vector<double>& pnl_distribution,
        const std::vector<double>& confidence_levels) {

        TailRiskMetrics metrics;

        // Sort P&L distribution
        std::vector<double> sorted_pnls = pnl_distribution;
        std::sort(sorted_pnls.begin(), sorted_pnls.end());

        size_t n = sorted_pnls.size();

        // Calculate VaR for each confidence level
        for (double confidence : confidence_levels) {
            size_t index = static_cast<size_t>((1.0 - confidence) * n);
            index = std::min(index, n - 1);

            double var = -sorted_pnls[index]; // Negative because we want loss
            metrics.var_estimates[confidence] = var;

            // Calculate Expected Shortfall (Conditional VaR)
            double es = 0.0;
            for (size_t i = 0; i <= index; ++i) {
                es += sorted_pnls[i];
            }
            es /= (index + 1);
            metrics.expected_shortfall[confidence] = -es;
        }

        // Calculate additional tail risk metrics
        metrics.maximum_loss = -sorted_pnls[0];
        metrics.skewness = calculate_skewness(sorted_pnls);
        metrics.excess_kurtosis = calculate_excess_kurtosis(sorted_pnls);

        // Calculate tail index (Hill estimator for extreme value theory)
        metrics.tail_index = calculate_hill_estimator(sorted_pnls, 0.1); // Use top 10% for tail

        return metrics;
    }

    StressTestResults run_multi_scenario_stress_test(
        const std::vector<RiskFactor>& factors,
        const std::vector<CustomScenario>& scenarios,
        const MultiScenarioParameters& params) {

        StressTestResults results;
        results.scenario_results.reserve(scenarios.size());

        // Run each scenario
        for (const auto& scenario : scenarios) {
            ScenarioResult scenario_result;
            scenario_result.scenario_name = scenario.name;
            scenario_result.scenario_type = scenario.scenario_type;

            // Apply shocks and calculate P&L
            scenario_result.total_pnl = calculate_scenario_impact(scenario.factor_shocks, factors, params);

            // Calculate factor contributions
            calculate_factor_contributions(scenario_result, scenario.factor_shocks, factors, params);

            // Calculate risk metrics for this scenario
            calculate_scenario_risk_metrics(scenario_result, scenario.factor_shocks, factors, params);

            results.scenario_results.push_back(std::move(scenario_result));
        }

        // Calculate aggregate statistics across all scenarios
        calculate_aggregate_statistics(results);

        return results;
    }

private:
    Eigen::MatrixXd prepare_correlation_matrix(
        const std::vector<RiskFactor>& factors,
        const MonteCarloStressParameters& params) {

        if (params.correlation_matrix.has_value()) {
            return params.correlation_matrix.value();
        }

        // Generate default correlation matrix
        return generate_default_correlation_matrix(factors);
    }

    Eigen::MatrixXd get_cholesky_decomposition(const Eigen::MatrixXd& correlation_matrix) {
        std::lock_guard<std::mutex> lock(correlation_cache_mutex_);

        std::string cache_key = matrix_to_cache_key(correlation_matrix);
        auto it = cholesky_cache_.find(cache_key);
        if (it != cholesky_cache_.end()) {
            return it->second;
        }

        // Compute Cholesky decomposition with regularization if needed
        Eigen::MatrixXd regularized_matrix = regularize_correlation_matrix(correlation_matrix);
        Eigen::LLT<Eigen::MatrixXd> llt(regularized_matrix);

        if (llt.info() != Eigen::Success) {
            throw std::runtime_error("Failed to compute Cholesky decomposition");
        }

        Eigen::MatrixXd cholesky = llt.matrixL();
        cholesky_cache_[cache_key] = cholesky;

        return cholesky;
    }

    void generate_correlated_time_series(
        std::vector<std::vector<double>>& scenario_shocks,
        const std::vector<RiskFactor>& factors,
        const MonteCarloStressParameters& params,
        const Eigen::MatrixXd& cholesky,
        std::mt19937_64& rng,
        std::normal_distribution<double>& normal_dist) {

        size_t num_factors = factors.size();
        size_t num_time_steps = params.time_horizon_days;

        scenario_shocks.resize(num_time_steps, std::vector<double>(num_factors));

        // Generate time series with autocorrelation if specified
        std::vector<double> previous_shocks(num_factors, 0.0);

        for (size_t t = 0; t < num_time_steps; ++t) {
            // Generate independent shocks
            Eigen::VectorXd independent_shocks(num_factors);
            for (size_t i = 0; i < num_factors; ++i) {
                independent_shocks(i) = normal_dist(rng);
            }

            // Apply cross-sectional correlation
            Eigen::VectorXd correlated_shocks = cholesky * independent_shocks;

            // Apply autocorrelation if specified
            for (size_t i = 0; i < num_factors; ++i) {
                double autocorr = get_factor_autocorrelation(factors[i], params);
                scenario_shocks[t][i] = autocorr * previous_shocks[i] +
                                      std::sqrt(1.0 - autocorr * autocorr) * correlated_shocks(i);
            }

            // Update previous shocks for next iteration
            previous_shocks = scenario_shocks[t];
        }

        // Apply variance reduction techniques
        if (use_antithetic_variates_) {
            apply_antithetic_variates(scenario_shocks);
        }
    }

    double calculate_scenario_pnl(
        const std::vector<std::vector<double>>& scenario_shocks,
        const std::vector<RiskFactor>& factors,
        const MonteCarloStressParameters& params,
        std::vector<double>& pnl_path) {

        size_t num_time_steps = scenario_shocks.size();
        pnl_path.resize(num_time_steps);

        double cumulative_pnl = 0.0;
        double max_drawdown = 0.0;
        double peak_pnl = 0.0;

        for (size_t t = 0; t < num_time_steps; ++t) {
            double daily_pnl = 0.0;

            // Calculate P&L for each factor
            for (size_t i = 0; i < factors.size() && i < scenario_shocks[t].size(); ++i) {
                double factor_pnl = calculate_factor_pnl(
                    factors[i], scenario_shocks[t][i], params);
                daily_pnl += factor_pnl;
            }

            cumulative_pnl += daily_pnl;
            pnl_path[t] = cumulative_pnl;

            // Update maximum drawdown
            if (cumulative_pnl > peak_pnl) {
                peak_pnl = cumulative_pnl;
            } else {
                double current_drawdown = peak_pnl - cumulative_pnl;
                max_drawdown = std::max(max_drawdown, current_drawdown);
            }
        }

        return max_drawdown;
    }

    void calculate_monte_carlo_statistics(
        MonteCarloResults& results,
        const std::vector<double>& final_pnls,
        const std::vector<double>& max_drawdowns,
        const std::vector<std::vector<double>>& all_pnl_paths,
        const MonteCarloStressParameters& params) {

        // Basic statistics
        results.mean_pnl = std::accumulate(final_pnls.begin(), final_pnls.end(), 0.0) / final_pnls.size();

        double variance = 0.0;
        for (double pnl : final_pnls) {
            variance += (pnl - results.mean_pnl) * (pnl - results.mean_pnl);
        }
        results.pnl_std_deviation = std::sqrt(variance / (final_pnls.size() - 1));

        results.mean_max_drawdown = std::accumulate(max_drawdowns.begin(), max_drawdowns.end(), 0.0) / max_drawdowns.size();

        // Calculate tail risk metrics
        results.tail_metrics = calculate_tail_risk_metrics(final_pnls, params.confidence_levels);

        // Calculate percentiles
        std::vector<double> sorted_pnls = final_pnls;
        std::sort(sorted_pnls.begin(), sorted_pnls.end());

        results.percentiles[0.01] = sorted_pnls[static_cast<size_t>(0.01 * sorted_pnls.size())];
        results.percentiles[0.05] = sorted_pnls[static_cast<size_t>(0.05 * sorted_pnls.size())];
        results.percentiles[0.25] = sorted_pnls[static_cast<size_t>(0.25 * sorted_pnls.size())];
        results.percentiles[0.50] = sorted_pnls[static_cast<size_t>(0.50 * sorted_pnls.size())];
        results.percentiles[0.75] = sorted_pnls[static_cast<size_t>(0.75 * sorted_pnls.size())];
        results.percentiles[0.95] = sorted_pnls[static_cast<size_t>(0.95 * sorted_pnls.size())];
        results.percentiles[0.99] = sorted_pnls[static_cast<size_t>(0.99 * sorted_pnls.size())];
    }

    void generate_percentile_paths(
        MonteCarloResults& results,
        const std::vector<std::vector<double>>& all_pnl_paths,
        const MonteCarloStressParameters& params) {

        if (all_pnl_paths.empty()) return;

        size_t num_time_steps = all_pnl_paths[0].size();
        std::vector<double> percentiles = {0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99};

        for (double percentile : percentiles) {
            std::vector<double> percentile_path(num_time_steps);

            for (size_t t = 0; t < num_time_steps; ++t) {
                std::vector<double> time_step_pnls;
                time_step_pnls.reserve(all_pnl_paths.size());

                for (const auto& path : all_pnl_paths) {
                    time_step_pnls.push_back(path[t]);
                }

                std::sort(time_step_pnls.begin(), time_step_pnls.end());
                size_t index = static_cast<size_t>(percentile * time_step_pnls.size());
                index = std::min(index, time_step_pnls.size() - 1);

                percentile_path[t] = time_step_pnls[index];
            }

            results.percentile_paths[percentile] = std::move(percentile_path);
        }
    }

    // Helper functions
    Eigen::MatrixXd generate_default_correlation_matrix(const std::vector<RiskFactor>& factors) {
        size_t n = factors.size();
        Eigen::MatrixXd correlation_matrix = Eigen::MatrixXd::Identity(n, n);

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                double correlation = calculate_default_correlation(factors[i], factors[j]);
                correlation_matrix(i, j) = correlation;
                correlation_matrix(j, i) = correlation;
            }
        }

        return correlation_matrix;
    }

    double calculate_default_correlation(const RiskFactor& factor1, const RiskFactor& factor2) {
        if (factor1.category == factor2.category) {
            switch (factor1.category) {
                case RiskFactorCategory::EQUITY: return 0.7;
                case RiskFactorCategory::INTEREST_RATE: return 0.9;
                case RiskFactorCategory::FX: return 0.3;
                case RiskFactorCategory::COMMODITY: return 0.4;
                case RiskFactorCategory::CREDIT: return 0.6;
                case RiskFactorCategory::VOLATILITY: return 0.5;
            }
        }

        // Cross-category correlations
        if ((factor1.category == RiskFactorCategory::EQUITY && factor2.category == RiskFactorCategory::VOLATILITY) ||
            (factor1.category == RiskFactorCategory::VOLATILITY && factor2.category == RiskFactorCategory::EQUITY)) {
            return -0.7;
        }

        return 0.1; // Default low correlation
    }

    std::string matrix_to_cache_key(const Eigen::MatrixXd& matrix) {
        std::stringstream ss;
        ss << matrix.rows() << "x" << matrix.cols() << "_";
        for (int i = 0; i < std::min(matrix.rows(), 5L); ++i) {
            for (int j = 0; j < std::min(matrix.cols(), 5L); ++j) {
                ss << matrix(i, j) << "_";
            }
        }
        return ss.str();
    }

    Eigen::MatrixXd regularize_correlation_matrix(const Eigen::MatrixXd& matrix) {
        // Add small regularization to diagonal to ensure positive definiteness
        Eigen::MatrixXd regularized = matrix;
        double regularization = 1e-6;

        for (int i = 0; i < regularized.rows(); ++i) {
            regularized(i, i) += regularization;
        }

        return regularized;
    }

    double get_factor_autocorrelation(const RiskFactor& factor, const MonteCarloStressParameters& params) {
        // Default autocorrelations by factor category
        switch (factor.category) {
            case RiskFactorCategory::EQUITY: return 0.05;
            case RiskFactorCategory::INTEREST_RATE: return 0.3;
            case RiskFactorCategory::FX: return 0.02;
            case RiskFactorCategory::COMMODITY: return 0.1;
            case RiskFactorCategory::CREDIT: return 0.2;
            case RiskFactorCategory::VOLATILITY: return 0.4;
        }
        return 0.1;
    }

    void apply_antithetic_variates(std::vector<std::vector<double>>& scenario_shocks) {
        // For every scenario, create an antithetic scenario with opposite shocks
        size_t original_size = scenario_shocks.size();
        scenario_shocks.resize(original_size * 2);

        for (size_t t = 0; t < original_size; ++t) {
            scenario_shocks[original_size + t] = scenario_shocks[t];
            for (double& shock : scenario_shocks[original_size + t]) {
                shock = -shock; // Antithetic variate
            }
        }
    }

    double calculate_factor_pnl(const RiskFactor& factor, double shock, const MonteCarloStressParameters& params) {
        // Simplified P&L calculation - in practice this would use complex pricing models
        double notional = get_factor_notional(factor, params);
        double sensitivity = get_factor_sensitivity(factor, params);

        return notional * sensitivity * shock;
    }

    double get_factor_notional(const RiskFactor& factor, const MonteCarloStressParameters& params) {
        // Default notionals by factor category
        switch (factor.category) {
            case RiskFactorCategory::EQUITY: return 1000000.0; // $1M
            case RiskFactorCategory::INTEREST_RATE: return 10000000.0; // $10M
            case RiskFactorCategory::FX: return 5000000.0; // $5M
            case RiskFactorCategory::COMMODITY: return 2000000.0; // $2M
            case RiskFactorCategory::CREDIT: return 3000000.0; // $3M
            case RiskFactorCategory::VOLATILITY: return 500000.0; // $500K
        }
        return 1000000.0;
    }

    double get_factor_sensitivity(const RiskFactor& factor, const MonteCarloStressParameters& params) {
        // Default sensitivities (dollar change per unit shock)
        switch (factor.category) {
            case RiskFactorCategory::EQUITY: return 1.0; // 1:1 sensitivity
            case RiskFactorCategory::INTEREST_RATE: return 0.05; // Duration-like sensitivity
            case RiskFactorCategory::FX: return 1.0;
            case RiskFactorCategory::COMMODITY: return 0.8;
            case RiskFactorCategory::CREDIT: return 0.1;
            case RiskFactorCategory::VOLATILITY: return 0.02;
        }
        return 1.0;
    }

    // Additional statistical functions
    double calculate_skewness(const std::vector<double>& data) {
        double mean = std::accumulate(data.begin(), data.end(), 0.0) / data.size();
        double variance = 0.0, skewness = 0.0;

        for (double value : data) {
            double diff = value - mean;
            variance += diff * diff;
            skewness += diff * diff * diff;
        }

        variance /= (data.size() - 1);
        skewness /= data.size();

        return skewness / std::pow(variance, 1.5);
    }

    double calculate_excess_kurtosis(const std::vector<double>& data) {
        double mean = std::accumulate(data.begin(), data.end(), 0.0) / data.size();
        double variance = 0.0, kurtosis = 0.0;

        for (double value : data) {
            double diff = value - mean;
            variance += diff * diff;
            kurtosis += diff * diff * diff * diff;
        }

        variance /= (data.size() - 1);
        kurtosis /= data.size();

        return (kurtosis / (variance * variance)) - 3.0; // Excess kurtosis
    }

    double calculate_hill_estimator(const std::vector<double>& sorted_data, double tail_fraction) {
        size_t k = static_cast<size_t>(tail_fraction * sorted_data.size());
        k = std::max(k, size_t(1));

        double sum_log_ratios = 0.0;
        double threshold = sorted_data[sorted_data.size() - k - 1];

        for (size_t i = sorted_data.size() - k; i < sorted_data.size(); ++i) {
            sum_log_ratios += std::log(sorted_data[i] / threshold);
        }

        return sum_log_ratios / k;
    }

    void calculate_sample_statistics(CorrelatedSamples& samples) {
        // Calculate correlation matrix of generated samples for validation
        // Implementation would compute empirical correlations
    }

    // Placeholder implementations for multi-scenario methods
    double calculate_scenario_impact(const std::vector<RiskFactorShock>& shocks,
                                   const std::vector<RiskFactor>& factors,
                                   const MultiScenarioParameters& params) {
        return 0.0; // Placeholder
    }

    void calculate_factor_contributions(ScenarioResult& result,
                                      const std::vector<RiskFactorShock>& shocks,
                                      const std::vector<RiskFactor>& factors,
                                      const MultiScenarioParameters& params) {
        // Placeholder
    }

    void calculate_scenario_risk_metrics(ScenarioResult& result,
                                        const std::vector<RiskFactorShock>& shocks,
                                        const std::vector<RiskFactor>& factors,
                                        const MultiScenarioParameters& params) {
        // Placeholder
    }

    void calculate_aggregate_statistics(StressTestResults& results) {
        // Placeholder
    }
};

// MonteCarloStressTester implementation
MonteCarloStressTester::MonteCarloStressTester() :
    impl_(std::make_unique<MonteCarloStressTesterImpl>()) {}

MonteCarloStressTester::~MonteCarloStressTester() = default;

void MonteCarloStressTester::configure_variance_reduction(bool antithetic, bool control, bool importance) {
    impl_->configure_variance_reduction(antithetic, control, importance);
}

MonteCarloResults MonteCarloStressTester::run_monte_carlo_stress_test(
    const std::vector<RiskFactor>& factors,
    const MonteCarloStressParameters& params) {
    return impl_->run_monte_carlo_stress_test(factors, params);
}

CorrelatedSamples MonteCarloStressTester::generate_correlated_samples(
    const std::vector<RiskFactor>& factors,
    const Eigen::MatrixXd& correlation_matrix,
    int num_samples,
    uint64_t seed) {
    return impl_->generate_correlated_samples(factors, correlation_matrix, num_samples, seed);
}

TailRiskMetrics MonteCarloStressTester::calculate_tail_risk_metrics(
    const std::vector<double>& pnl_distribution,
    const std::vector<double>& confidence_levels) {
    return impl_->calculate_tail_risk_metrics(pnl_distribution, confidence_levels);
}

StressTestResults MonteCarloStressTester::run_multi_scenario_stress_test(
    const std::vector<RiskFactor>& factors,
    const std::vector<CustomScenario>& scenarios,
    const MultiScenarioParameters& params) {
    return impl_->run_multi_scenario_stress_test(factors, scenarios, params);
}

} // namespace stress_testing