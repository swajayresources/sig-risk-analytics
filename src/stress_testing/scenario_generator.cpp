#include "stress_testing/stress_framework.hpp"
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <execution>

namespace stress_testing {

class ScenarioGeneratorImpl {
private:
    std::mt19937_64 rng_;
    std::normal_distribution<double> normal_dist_;
    std::uniform_real_distribution<double> uniform_dist_;

    // Correlation matrix and Cholesky decomposition cache
    mutable std::mutex correlation_mutex_;
    std::unordered_map<std::string, Eigen::MatrixXd> cholesky_cache_;

    // Advanced random number generators
    std::gamma_distribution<double> gamma_dist_;
    std::chi_squared_distribution<double> chi_squared_dist_;
    std::student_t_distribution<double> t_dist_;

public:
    ScenarioGeneratorImpl() :
        rng_(std::random_device{}()),
        normal_dist_(0.0, 1.0),
        uniform_dist_(0.0, 1.0),
        gamma_dist_(2.0, 1.0),
        chi_squared_dist_(1.0),
        t_dist_(3.0) {}

    void set_seed(uint64_t seed) {
        rng_.seed(seed);
    }

    // Generate correlated shocks using Cholesky decomposition
    std::vector<double> generate_correlated_shocks(
        const std::vector<RiskFactor>& factors,
        const Eigen::MatrixXd& correlation_matrix) {

        size_t n = factors.size();
        std::vector<double> independent_shocks(n);

        // Generate independent standard normal variables
        std::generate(independent_shocks.begin(), independent_shocks.end(),
            [this]() { return normal_dist_(rng_); });

        // Apply Cholesky decomposition for correlation
        Eigen::VectorXd shocks = Eigen::Map<Eigen::VectorXd>(independent_shocks.data(), n);
        Eigen::MatrixXd cholesky = get_cholesky_decomposition(correlation_matrix);
        Eigen::VectorXd correlated_shocks = cholesky * shocks;

        return std::vector<double>(correlated_shocks.data(),
                                 correlated_shocks.data() + correlated_shocks.size());
    }

    // Advanced shock generation with fat tails
    std::vector<double> generate_fat_tail_shocks(
        const std::vector<RiskFactor>& factors,
        double tail_probability = 0.05) {

        std::vector<double> shocks;
        shocks.reserve(factors.size());

        for (const auto& factor : factors) {
            double shock = 0.0;

            if (uniform_dist_(rng_) < tail_probability) {
                // Generate extreme tail event
                double tail_multiplier = 3.0 + std::abs(normal_dist_(rng_));
                shock = (uniform_dist_(rng_) < 0.5 ? -1.0 : 1.0) * tail_multiplier;
            } else {
                // Generate normal shock with Student-t distribution
                shock = t_dist_(rng_);
            }

            shocks.push_back(shock);
        }

        return shocks;
    }

    // Economic regime-dependent shock generation
    std::vector<double> generate_regime_dependent_shocks(
        const std::vector<RiskFactor>& factors,
        EconomicRegime regime) {

        std::vector<double> shocks;
        shocks.reserve(factors.size());

        // Regime-specific parameters
        double volatility_multiplier = 1.0;
        double mean_reversion_strength = 0.1;

        switch (regime) {
            case EconomicRegime::RECESSION:
                volatility_multiplier = 2.5;
                mean_reversion_strength = 0.05;
                break;
            case EconomicRegime::EXPANSION:
                volatility_multiplier = 0.8;
                mean_reversion_strength = 0.15;
                break;
            case EconomicRegime::STAGFLATION:
                volatility_multiplier = 1.8;
                mean_reversion_strength = 0.03;
                break;
            case EconomicRegime::DEFLATION:
                volatility_multiplier = 2.0;
                mean_reversion_strength = 0.02;
                break;
        }

        for (const auto& factor : factors) {
            double base_shock = normal_dist_(rng_) * volatility_multiplier;

            // Apply factor-specific adjustments based on regime
            if (factor.category == RiskFactorCategory::EQUITY && regime == EconomicRegime::RECESSION) {
                base_shock -= 0.5; // Negative bias for equities in recession
            } else if (factor.category == RiskFactorCategory::INTEREST_RATE && regime == EconomicRegime::STAGFLATION) {
                base_shock += 0.3; // Positive bias for rates in stagflation
            }

            shocks.push_back(base_shock);
        }

        return shocks;
    }

private:
    Eigen::MatrixXd get_cholesky_decomposition(const Eigen::MatrixXd& correlation_matrix) {
        std::lock_guard<std::mutex> lock(correlation_mutex_);

        // Create cache key
        std::string cache_key = matrix_to_string(correlation_matrix);

        auto it = cholesky_cache_.find(cache_key);
        if (it != cholesky_cache_.end()) {
            return it->second;
        }

        // Compute Cholesky decomposition
        Eigen::LLT<Eigen::MatrixXd> llt(correlation_matrix);
        if (llt.info() != Eigen::Success) {
            // Fallback to modified Cholesky for non-positive definite matrices
            Eigen::MatrixXd modified_matrix = make_positive_definite(correlation_matrix);
            Eigen::LLT<Eigen::MatrixXd> modified_llt(modified_matrix);
            cholesky_cache_[cache_key] = modified_llt.matrixL();
        } else {
            cholesky_cache_[cache_key] = llt.matrixL();
        }

        return cholesky_cache_[cache_key];
    }

    std::string matrix_to_string(const Eigen::MatrixXd& matrix) {
        std::stringstream ss;
        ss << matrix;
        return ss.str();
    }

    Eigen::MatrixXd make_positive_definite(const Eigen::MatrixXd& matrix) {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(matrix);
        Eigen::VectorXd eigenvalues = solver.eigenvalues();
        Eigen::MatrixXd eigenvectors = solver.eigenvectors();

        // Clamp negative eigenvalues to small positive value
        const double min_eigenvalue = 1e-8;
        for (int i = 0; i < eigenvalues.size(); ++i) {
            if (eigenvalues(i) < min_eigenvalue) {
                eigenvalues(i) = min_eigenvalue;
            }
        }

        return eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose();
    }
};

// ScenarioGenerator implementation
ScenarioGenerator::ScenarioGenerator() :
    impl_(std::make_unique<ScenarioGeneratorImpl>()) {}

ScenarioGenerator::~ScenarioGenerator() = default;

void ScenarioGenerator::configure(const ScenarioConfig& config) {
    config_ = config;
    impl_->set_seed(config.random_seed);
}

CustomScenario ScenarioGenerator::generate_custom_scenario(
    const std::vector<RiskFactor>& factors,
    const ScenarioParameters& params) {

    CustomScenario scenario;
    scenario.name = params.name;
    scenario.description = params.description;
    scenario.scenario_type = params.scenario_type;
    scenario.time_horizon_days = params.time_horizon_days;

    // Generate base correlation matrix if not provided
    Eigen::MatrixXd correlation_matrix;
    if (params.correlation_matrix.has_value()) {
        correlation_matrix = params.correlation_matrix.value();
    } else {
        correlation_matrix = generate_default_correlation_matrix(factors);
    }

    // Generate shocks based on scenario type
    std::vector<double> shock_magnitudes;

    switch (params.scenario_type) {
        case ScenarioType::MONTE_CARLO:
            shock_magnitudes = impl_->generate_correlated_shocks(factors, correlation_matrix);
            break;

        case ScenarioType::TAIL_RISK:
            shock_magnitudes = impl_->generate_fat_tail_shocks(factors, params.tail_probability);
            break;

        case ScenarioType::REGIME_DEPENDENT:
            shock_magnitudes = impl_->generate_regime_dependent_shocks(factors, params.economic_regime);
            break;

        default:
            shock_magnitudes = impl_->generate_correlated_shocks(factors, correlation_matrix);
            break;
    }

    // Apply factor-specific scaling and constraints
    for (size_t i = 0; i < factors.size() && i < shock_magnitudes.size(); ++i) {
        RiskFactorShock shock;
        shock.factor_id = factors[i].id;
        shock.factor_name = factors[i].name;
        shock.shock_magnitude = shock_magnitudes[i] * params.volatility_scaling;
        shock.shock_type = ShockType::RELATIVE;

        // Apply constraints
        if (params.max_shock_magnitude.has_value()) {
            shock.shock_magnitude = std::clamp(shock.shock_magnitude,
                -params.max_shock_magnitude.value(),
                params.max_shock_magnitude.value());
        }

        // Factor-specific adjustments
        if (factors[i].category == RiskFactorCategory::VOLATILITY) {
            shock.shock_magnitude = std::abs(shock.shock_magnitude); // Volatility shocks are positive
        }

        scenario.factor_shocks.push_back(shock);
    }

    // Generate time series if multi-period scenario
    if (params.time_horizon_days > 1) {
        generate_time_series_shocks(scenario, factors, params);
    }

    return scenario;
}

std::vector<CustomScenario> ScenarioGenerator::generate_monte_carlo_scenarios(
    const std::vector<RiskFactor>& factors,
    const MonteCarloParameters& params) {

    std::vector<CustomScenario> scenarios;
    scenarios.reserve(params.num_scenarios);

    // Generate correlation matrix
    Eigen::MatrixXd correlation_matrix = params.correlation_matrix.value_or(
        generate_default_correlation_matrix(factors));

    // Parallel scenario generation
    std::mutex scenarios_mutex;

    auto generate_scenario = [&](int scenario_idx) {
        ScenarioParameters scenario_params;
        scenario_params.name = "MC_Scenario_" + std::to_string(scenario_idx);
        scenario_params.description = "Monte Carlo generated scenario";
        scenario_params.scenario_type = ScenarioType::MONTE_CARLO;
        scenario_params.time_horizon_days = params.time_horizon_days;
        scenario_params.correlation_matrix = correlation_matrix;
        scenario_params.volatility_scaling = params.volatility_scaling;
        scenario_params.random_seed = params.base_seed + scenario_idx;

        // Create temporary generator for this thread
        ScenarioGeneratorImpl temp_impl;
        temp_impl.set_seed(scenario_params.random_seed);

        CustomScenario scenario = generate_custom_scenario(factors, scenario_params);

        std::lock_guard<std::mutex> lock(scenarios_mutex);
        scenarios.push_back(std::move(scenario));
    };

    // Execute scenario generation in parallel
    std::vector<int> indices(params.num_scenarios);
    std::iota(indices.begin(), indices.end(), 0);

    std::for_each(std::execution::par_unseq, indices.begin(), indices.end(), generate_scenario);

    // Sort scenarios by name for deterministic ordering
    std::sort(scenarios.begin(), scenarios.end(),
        [](const CustomScenario& a, const CustomScenario& b) {
            return a.name < b.name;
        });

    return scenarios;
}

TailRiskScenario ScenarioGenerator::generate_tail_risk_scenario(
    const std::vector<RiskFactor>& factors,
    const TailRiskParameters& params) {

    TailRiskScenario scenario;
    scenario.name = params.name;
    scenario.description = params.description;
    scenario.confidence_level = params.confidence_level;
    scenario.tail_probability = params.tail_probability;
    scenario.time_horizon_days = params.time_horizon_days;

    // Generate extreme shocks using fat-tail distribution
    std::vector<double> shock_magnitudes = impl_->generate_fat_tail_shocks(
        factors, params.tail_probability);

    // Apply copula-based dependence if specified
    if (params.use_copula_dependence) {
        shock_magnitudes = apply_copula_dependence(shock_magnitudes, factors, params);
    }

    // Create factor shocks
    for (size_t i = 0; i < factors.size() && i < shock_magnitudes.size(); ++i) {
        RiskFactorShock shock;
        shock.factor_id = factors[i].id;
        shock.factor_name = factors[i].name;
        shock.shock_magnitude = shock_magnitudes[i] * params.extreme_multiplier;
        shock.shock_type = ShockType::RELATIVE;

        scenario.factor_shocks.push_back(shock);
    }

    // Calculate tail risk metrics
    scenario.expected_shortfall = calculate_expected_shortfall(scenario.factor_shocks, params);
    scenario.maximum_drawdown = calculate_maximum_drawdown(scenario.factor_shocks, params);

    return scenario;
}

RegimeScenario ScenarioGenerator::generate_regime_scenario(
    const std::vector<RiskFactor>& factors,
    const RegimeParameters& params) {

    RegimeScenario scenario;
    scenario.name = params.name;
    scenario.description = params.description;
    scenario.regime = params.regime;
    scenario.transition_probability = params.transition_probability;
    scenario.persistence_parameter = params.persistence_parameter;

    // Generate regime-dependent shocks
    std::vector<double> shock_magnitudes = impl_->generate_regime_dependent_shocks(
        factors, params.regime);

    // Apply regime-specific correlation structure
    if (params.regime_correlation_matrix.has_value()) {
        Eigen::VectorXd shocks = Eigen::Map<Eigen::VectorXd>(shock_magnitudes.data(), shock_magnitudes.size());
        Eigen::MatrixXd cholesky = impl_->get_cholesky_decomposition(params.regime_correlation_matrix.value());
        Eigen::VectorXd correlated_shocks = cholesky * shocks;
        shock_magnitudes.assign(correlated_shocks.data(), correlated_shocks.data() + correlated_shocks.size());
    }

    // Create factor shocks with regime adjustments
    for (size_t i = 0; i < factors.size() && i < shock_magnitudes.size(); ++i) {
        RiskFactorShock shock;
        shock.factor_id = factors[i].id;
        shock.factor_name = factors[i].name;
        shock.shock_magnitude = shock_magnitudes[i];
        shock.shock_type = ShockType::RELATIVE;

        // Apply regime-specific factor adjustments
        apply_regime_factor_adjustments(shock, factors[i], params.regime);

        scenario.factor_shocks.push_back(shock);
    }

    return scenario;
}

private:
Eigen::MatrixXd ScenarioGenerator::generate_default_correlation_matrix(
    const std::vector<RiskFactor>& factors) {

    size_t n = factors.size();
    Eigen::MatrixXd correlation_matrix = Eigen::MatrixXd::Identity(n, n);

    // Apply default correlations based on factor categories
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            double correlation = calculate_default_correlation(factors[i], factors[j]);
            correlation_matrix(i, j) = correlation;
            correlation_matrix(j, i) = correlation;
        }
    }

    return correlation_matrix;
}

double ScenarioGenerator::calculate_default_correlation(
    const RiskFactor& factor1,
    const RiskFactor& factor2) {

    // Same category correlations
    if (factor1.category == factor2.category) {
        switch (factor1.category) {
            case RiskFactorCategory::EQUITY:
                return 0.7; // High correlation between equity indices
            case RiskFactorCategory::INTEREST_RATE:
                return 0.8; // Very high correlation between rates
            case RiskFactorCategory::FX:
                return 0.3; // Moderate correlation between currencies
            case RiskFactorCategory::COMMODITY:
                return 0.4; // Moderate correlation between commodities
            case RiskFactorCategory::CREDIT:
                return 0.6; // High correlation between credit spreads
            case RiskFactorCategory::VOLATILITY:
                return 0.5; // Moderate correlation between volatilities
        }
    }

    // Cross-category correlations
    if ((factor1.category == RiskFactorCategory::EQUITY && factor2.category == RiskFactorCategory::VOLATILITY) ||
        (factor1.category == RiskFactorCategory::VOLATILITY && factor2.category == RiskFactorCategory::EQUITY)) {
        return -0.7; // Negative correlation between equity and volatility
    }

    if ((factor1.category == RiskFactorCategory::EQUITY && factor2.category == RiskFactorCategory::CREDIT) ||
        (factor1.category == RiskFactorCategory::CREDIT && factor2.category == RiskFactorCategory::EQUITY)) {
        return -0.5; // Negative correlation between equity and credit spreads
    }

    if ((factor1.category == RiskFactorCategory::INTEREST_RATE && factor2.category == RiskFactorCategory::FX) ||
        (factor1.category == RiskFactorCategory::FX && factor2.category == RiskFactorCategory::INTEREST_RATE)) {
        return 0.4; // Moderate positive correlation
    }

    return 0.1; // Default low correlation
}

void ScenarioGenerator::generate_time_series_shocks(
    CustomScenario& scenario,
    const std::vector<RiskFactor>& factors,
    const ScenarioParameters& params) {

    // Implementation for multi-period scenarios with autocorrelation
    // This would generate correlated shocks across time periods
    // using GARCH models or other time series techniques
}

std::vector<double> ScenarioGenerator::apply_copula_dependence(
    const std::vector<double>& shocks,
    const std::vector<RiskFactor>& factors,
    const TailRiskParameters& params) {

    // Implementation of copula-based dependence structure
    // for modeling tail dependence between risk factors
    return shocks; // Placeholder
}

double ScenarioGenerator::calculate_expected_shortfall(
    const std::vector<RiskFactorShock>& shocks,
    const TailRiskParameters& params) {

    // Calculate expected shortfall (conditional VaR)
    // Implementation would depend on portfolio composition
    return 0.0; // Placeholder
}

double ScenarioGenerator::calculate_maximum_drawdown(
    const std::vector<RiskFactorShock>& shocks,
    const TailRiskParameters& params) {

    // Calculate maximum drawdown over scenario horizon
    return 0.0; // Placeholder
}

void ScenarioGenerator::apply_regime_factor_adjustments(
    RiskFactorShock& shock,
    const RiskFactor& factor,
    EconomicRegime regime) {

    // Apply regime-specific adjustments to individual factors
    switch (regime) {
        case EconomicRegime::RECESSION:
            if (factor.category == RiskFactorCategory::EQUITY) {
                shock.shock_magnitude *= 1.5; // Amplify equity shocks in recession
            }
            break;
        case EconomicRegime::STAGFLATION:
            if (factor.category == RiskFactorCategory::INTEREST_RATE) {
                shock.shock_magnitude += 0.5; // Add positive bias to rates
            }
            break;
        // Add other regime adjustments...
    }
}

} // namespace stress_testing