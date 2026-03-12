#include "risk/risk_calculator.hpp"
#include <algorithm>
#include <execution>
#include <chrono>
#include <numeric>
#include <cmath>

namespace risk_engine {

MonteCarloEngine::MonteCarloEngine(const SimulationParameters& params)
    : params_(params), rng_(std::random_device{}()) {
}

MonteCarloEngine::SimulationResult MonteCarloEngine::simulate_portfolio_pnl(
    const std::vector<std::shared_ptr<Position>>& positions,
    const MarketFactors& factors) {

    auto start_time = std::chrono::high_resolution_clock::now();

    // Generate correlated price scenarios
    auto scenarios = generate_correlated_scenarios(factors);

    if (params_.use_antithetic_variates) {
        scenarios = apply_antithetic_variates(scenarios);
    }

    // Compute portfolio P&L for each scenario
    auto portfolio_pnl = compute_portfolio_pnl_for_scenarios(positions, scenarios);

    // Calculate risk metrics
    double var_estimate = calculate_var(portfolio_pnl, params_.confidence_level);
    double expected_shortfall = calculate_expected_shortfall(portfolio_pnl, params_.confidence_level);

    auto [ci_lower, ci_upper] = calculate_confidence_interval(portfolio_pnl, params_.confidence_level);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    return SimulationResult{
        .portfolio_pnl = std::move(portfolio_pnl),
        .var_estimate = var_estimate,
        .expected_shortfall = expected_shortfall,
        .confidence_interval_lower = ci_lower,
        .confidence_interval_upper = ci_upper,
        .computation_time_ms = static_cast<double>(duration.count()),
        .scenarios_used = static_cast<int>(scenarios.size())
    };
}

MonteCarloEngine::SimulationResult MonteCarloEngine::calculate_var_and_es(
    const std::vector<std::shared_ptr<Position>>& positions,
    const MarketFactors& factors) {

    return simulate_portfolio_pnl(positions, factors);
}

std::vector<std::vector<double>> MonteCarloEngine::generate_correlated_scenarios(
    const MarketFactors& factors) {

    const int n_assets = factors.symbols.size();
    const int n_scenarios = params_.num_scenarios;

    // Cholesky decomposition of correlation matrix
    auto chol_matrix = cholesky_decomposition(factors.correlation_matrix);

    std::vector<std::vector<double>> scenarios(n_scenarios, std::vector<double>(n_assets));

    std::normal_distribution<double> normal_dist(0.0, 1.0);

    // Generate independent random variables
    std::vector<std::vector<double>> independent_normals(n_scenarios, std::vector<double>(n_assets));

    for (int i = 0; i < n_scenarios; ++i) {
        for (int j = 0; j < n_assets; ++j) {
            independent_normals[i][j] = normal_dist(rng_);
        }
    }

    // Apply Cholesky decomposition to create correlated normals
    for (int i = 0; i < n_scenarios; ++i) {
        for (int j = 0; j < n_assets; ++j) {
            double correlated_normal = 0.0;
            for (int k = 0; k <= j; ++k) {
                correlated_normal += chol_matrix[j][k] * independent_normals[i][k];
            }

            // Convert to price scenario using GBM
            double dt = params_.horizon_days / 252.0; // Assuming 252 trading days per year
            double drift = (factors.returns_mean[j] - 0.5 * factors.returns_volatility[j] * factors.returns_volatility[j]) * dt;
            double diffusion = factors.returns_volatility[j] * std::sqrt(dt) * correlated_normal;

            scenarios[i][j] = std::exp(drift + diffusion);
        }
    }

    return scenarios;
}

std::vector<double> MonteCarloEngine::compute_portfolio_pnl_for_scenarios(
    const std::vector<std::shared_ptr<Position>>& positions,
    const std::vector<std::vector<double>>& price_scenarios) {

    const int n_scenarios = price_scenarios.size();
    std::vector<double> portfolio_pnl(n_scenarios, 0.0);

    // Create symbol to index mapping
    std::unordered_map<Symbol, int> symbol_to_index;
    for (size_t i = 0; i < positions.size(); ++i) {
        symbol_to_index[positions[i]->instrument.symbol] = i;
    }

    // Parallel computation of P&L across scenarios
    const int n_threads = params_.num_threads;
    const int scenarios_per_thread = (n_scenarios + n_threads - 1) / n_threads;

    std::vector<std::future<std::vector<double>>> futures;

    for (int t = 0; t < n_threads; ++t) {
        int start_idx = t * scenarios_per_thread;
        int end_idx = std::min(start_idx + scenarios_per_thread, n_scenarios);

        if (start_idx < end_idx) {
            futures.push_back(std::async(std::launch::async,
                [this, &positions, &price_scenarios, &symbol_to_index, start_idx, end_idx]() {
                    return parallel_scenario_processing(positions, price_scenarios, start_idx, end_idx);
                }));
        }
    }

    // Collect results
    int idx = 0;
    for (auto& future : futures) {
        auto thread_results = future.get();
        std::copy(thread_results.begin(), thread_results.end(), portfolio_pnl.begin() + idx);
        idx += thread_results.size();
    }

    return portfolio_pnl;
}

std::vector<double> MonteCarloEngine::parallel_scenario_processing(
    const std::vector<std::shared_ptr<Position>>& positions,
    const std::vector<std::vector<double>>& scenarios,
    int start_idx, int end_idx) {

    std::vector<double> pnl_results;
    pnl_results.reserve(end_idx - start_idx);

    for (int i = start_idx; i < end_idx; ++i) {
        double scenario_pnl = 0.0;

        for (const auto& position : positions) {
            Quantity qty = position->quantity.load();
            Price current_price = position->market_price.load();

            if (qty != 0 && current_price > EPSILON) {
                // Find the price scenario for this symbol
                // This is simplified - in practice, you'd need a proper symbol mapping
                double price_multiplier = scenarios[i][0]; // Simplified: using first factor
                Price new_price = current_price * price_multiplier;

                // Calculate P&L for this position
                double position_pnl = qty * (new_price - current_price);
                scenario_pnl += position_pnl;
            }
        }

        pnl_results.push_back(scenario_pnl);
    }

    return pnl_results;
}

std::vector<std::vector<double>> MonteCarloEngine::cholesky_decomposition(
    const std::vector<std::vector<double>>& correlation_matrix) {

    const int n = correlation_matrix.size();
    std::vector<std::vector<double>> L(n, std::vector<double>(n, 0.0));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            if (i == j) {
                double sum = 0.0;
                for (int k = 0; k < j; ++k) {
                    sum += L[i][k] * L[i][k];
                }
                L[i][j] = std::sqrt(correlation_matrix[i][i] - sum);
            } else {
                double sum = 0.0;
                for (int k = 0; k < j; ++k) {
                    sum += L[i][k] * L[j][k];
                }
                if (L[j][j] > EPSILON) {
                    L[i][j] = (correlation_matrix[i][j] - sum) / L[j][j];
                }
            }
        }
    }

    return L;
}

std::vector<std::vector<double>> MonteCarloEngine::apply_antithetic_variates(
    const std::vector<std::vector<double>>& scenarios) {

    std::vector<std::vector<double>> enhanced_scenarios;
    enhanced_scenarios.reserve(scenarios.size() * 2);

    for (const auto& scenario : scenarios) {
        enhanced_scenarios.push_back(scenario);

        // Create antithetic scenario
        std::vector<double> antithetic_scenario;
        antithetic_scenario.reserve(scenario.size());

        for (double value : scenario) {
            // For price multipliers, antithetic is 2 - original
            antithetic_scenario.push_back(2.0 - value);
        }

        enhanced_scenarios.push_back(antithetic_scenario);
    }

    return enhanced_scenarios;
}

double MonteCarloEngine::calculate_var(const std::vector<double>& returns, double confidence_level) {
    if (returns.empty()) return 0.0;

    std::vector<double> sorted_returns = returns;
    std::sort(sorted_returns.begin(), sorted_returns.end());

    int var_index = static_cast<int>((1.0 - confidence_level) * sorted_returns.size());
    var_index = std::max(0, std::min(var_index, static_cast<int>(sorted_returns.size()) - 1));

    return -sorted_returns[var_index]; // Negative because VaR is typically reported as positive loss
}

double MonteCarloEngine::calculate_expected_shortfall(const std::vector<double>& returns, double confidence_level) {
    if (returns.empty()) return 0.0;

    std::vector<double> sorted_returns = returns;
    std::sort(sorted_returns.begin(), sorted_returns.end());

    int var_index = static_cast<int>((1.0 - confidence_level) * sorted_returns.size());
    var_index = std::max(0, std::min(var_index, static_cast<int>(sorted_returns.size()) - 1));

    double sum = 0.0;
    for (int i = 0; i <= var_index; ++i) {
        sum += sorted_returns[i];
    }

    return -(sum / (var_index + 1)); // Negative for positive loss reporting
}

std::pair<double, double> MonteCarloEngine::calculate_confidence_interval(
    const std::vector<double>& returns, double confidence_level) {

    if (returns.empty()) return {0.0, 0.0};

    std::vector<double> sorted_returns = returns;
    std::sort(sorted_returns.begin(), sorted_returns.end());

    double alpha = 1.0 - confidence_level;
    int lower_index = static_cast<int>(alpha / 2.0 * sorted_returns.size());
    int upper_index = static_cast<int>((1.0 - alpha / 2.0) * sorted_returns.size());

    lower_index = std::max(0, std::min(lower_index, static_cast<int>(sorted_returns.size()) - 1));
    upper_index = std::max(0, std::min(upper_index, static_cast<int>(sorted_returns.size()) - 1));

    return {sorted_returns[lower_index], sorted_returns[upper_index]};
}

} // namespace risk_engine