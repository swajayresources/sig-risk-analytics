#pragma once

#include "core/types.hpp"
#include "core/position_engine.hpp"
#include <vector>
#include <random>
#include <memory>
#include <future>
#include <functional>

namespace risk_engine {

class MonteCarloEngine {
public:
    struct SimulationParameters {
        int num_scenarios = 100000;
        int horizon_days = 1;
        double confidence_level = 0.95;
        int num_threads = std::thread::hardware_concurrency();
        bool use_antithetic_variates = true;
        bool use_control_variates = true;
    };

    struct MarketFactors {
        std::vector<Symbol> symbols;
        std::vector<double> returns_mean;
        std::vector<double> returns_volatility;
        std::vector<std::vector<double>> correlation_matrix;
        std::vector<double> dividend_yields;
        std::vector<double> risk_free_rates;
    };

    struct SimulationResult {
        std::vector<double> portfolio_pnl;
        double var_estimate;
        double expected_shortfall;
        double confidence_interval_lower;
        double confidence_interval_upper;
        double computation_time_ms;
        int scenarios_used;
    };

    MonteCarloEngine(const SimulationParameters& params = SimulationParameters{});

    // Main simulation methods
    SimulationResult simulate_portfolio_pnl(
        const std::vector<std::shared_ptr<Position>>& positions,
        const MarketFactors& factors);

    SimulationResult calculate_var_and_es(
        const std::vector<std::shared_ptr<Position>>& positions,
        const MarketFactors& factors);

    // Scenario generation
    std::vector<std::vector<double>> generate_correlated_scenarios(
        const MarketFactors& factors);

    // Statistical methods
    static double calculate_var(const std::vector<double>& returns, double confidence_level);
    static double calculate_expected_shortfall(const std::vector<double>& returns, double confidence_level);
    static std::pair<double, double> calculate_confidence_interval(
        const std::vector<double>& returns, double confidence_level);

    // Variance reduction techniques
    std::vector<std::vector<double>> apply_antithetic_variates(
        const std::vector<std::vector<double>>& scenarios);

    void set_parameters(const SimulationParameters& params) { params_ = params; }
    const SimulationParameters& get_parameters() const { return params_; }

private:
    SimulationParameters params_;
    mutable std::mt19937_64 rng_;

    // Internal computation methods
    std::vector<double> compute_portfolio_pnl_for_scenarios(
        const std::vector<std::shared_ptr<Position>>& positions,
        const std::vector<std::vector<double>>& price_scenarios);

    std::vector<std::vector<double>> cholesky_decomposition(
        const std::vector<std::vector<double>>& correlation_matrix);

    // Parallel processing helpers
    std::vector<double> parallel_scenario_processing(
        const std::vector<std::shared_ptr<Position>>& positions,
        const std::vector<std::vector<double>>& scenarios,
        int start_idx, int end_idx);
};

class HistoricalSimulation {
public:
    struct HistoricalData {
        std::vector<Symbol> symbols;
        std::vector<std::vector<double>> price_history; // [symbol][time]
        std::vector<Timestamp> timestamps;
    };

    struct HistoricalResult {
        std::vector<double> historical_pnl;
        double var_estimate;
        double expected_shortfall;
        int scenarios_used;
        double computation_time_ms;
    };

    HistoricalSimulation(int lookback_days = 252, double confidence_level = 0.95);

    // Main calculation methods
    HistoricalResult calculate_historical_var(
        const std::vector<std::shared_ptr<Position>>& positions,
        const HistoricalData& historical_data);

    HistoricalResult bootstrap_var(
        const std::vector<std::shared_ptr<Position>>& positions,
        const HistoricalData& historical_data,
        int bootstrap_samples = 1000);

    // Data preprocessing
    static std::vector<std::vector<double>> calculate_returns(
        const std::vector<std::vector<double>>& price_history);

    static HistoricalData load_historical_data(
        const std::vector<Symbol>& symbols,
        const Timestamp& start_date,
        const Timestamp& end_date);

    void set_lookback_days(int days) { lookback_days_ = days; }
    void set_confidence_level(double level) { confidence_level_ = level; }

private:
    int lookback_days_;
    double confidence_level_;

    std::vector<double> compute_historical_portfolio_pnl(
        const std::vector<std::shared_ptr<Position>>& positions,
        const std::vector<std::vector<double>>& returns);
};

class ParametricVar {
public:
    struct CovarianceMatrix {
        std::vector<Symbol> symbols;
        std::vector<std::vector<double>> covariance;
        Timestamp last_updated;
    };

    struct ParametricResult {
        double var_estimate;
        double portfolio_volatility;
        double diversification_ratio;
        std::vector<double> component_contributions;
        double computation_time_ms;
    };

    ParametricVar(double confidence_level = 0.95, int horizon_days = 1);

    // Main calculation methods
    ParametricResult calculate_parametric_var(
        const std::vector<std::shared_ptr<Position>>& positions,
        const CovarianceMatrix& covariance_matrix);

    ParametricResult calculate_component_var(
        const std::vector<std::shared_ptr<Position>>& positions,
        const CovarianceMatrix& covariance_matrix);

    // Risk decomposition
    std::vector<double> calculate_marginal_var(
        const std::vector<std::shared_ptr<Position>>& positions,
        const CovarianceMatrix& covariance_matrix);

    std::vector<double> calculate_component_contributions(
        const std::vector<std::shared_ptr<Position>>& positions,
        const CovarianceMatrix& covariance_matrix);

    // Covariance matrix estimation
    static CovarianceMatrix estimate_covariance_matrix(
        const std::vector<Symbol>& symbols,
        const std::vector<std::vector<double>>& returns,
        double decay_factor = 0.94);

    static CovarianceMatrix shrink_covariance_matrix(
        const CovarianceMatrix& sample_cov,
        double shrinkage_intensity = 0.1);

    void set_confidence_level(double level) { confidence_level_ = level; }
    void set_horizon_days(int days) { horizon_days_ = days; }

private:
    double confidence_level_;
    int horizon_days_;

    double calculate_portfolio_variance(
        const std::vector<double>& weights,
        const std::vector<std::vector<double>>& covariance_matrix);

    std::vector<double> extract_position_weights(
        const std::vector<std::shared_ptr<Position>>& positions);
};

class RiskFactorModel {
public:
    struct RiskFactor {
        std::string name;
        std::vector<double> returns;
        double volatility;
        std::vector<Symbol> constituents;
    };

    struct FactorExposure {
        Symbol symbol;
        std::vector<double> factor_loadings;
        double idiosyncratic_risk;
    };

    struct FactorResult {
        double systematic_var;
        double idiosyncratic_var;
        double total_var;
        std::vector<double> factor_contributions;
        double computation_time_ms;
    };

    RiskFactorModel(const std::vector<RiskFactor>& factors);

    // Main calculation methods
    FactorResult calculate_factor_var(
        const std::vector<std::shared_ptr<Position>>& positions,
        double confidence_level = 0.95);

    // Factor exposure estimation
    std::vector<FactorExposure> estimate_factor_exposures(
        const std::vector<Symbol>& symbols,
        const std::vector<std::vector<double>>& returns);

    // Factor model utilities
    static std::vector<RiskFactor> create_equity_factors();
    static std::vector<RiskFactor> create_fixed_income_factors();
    static std::vector<RiskFactor> create_commodity_factors();

    void update_factors(const std::vector<RiskFactor>& factors) { factors_ = factors; }

private:
    std::vector<RiskFactor> factors_;
    std::vector<FactorExposure> factor_exposures_;

    std::vector<std::vector<double>> create_factor_covariance_matrix();
    double calculate_factor_portfolio_variance(
        const std::vector<double>& factor_exposures,
        const std::vector<std::vector<double>>& factor_covariance);
};

// Unified Risk Calculator
class RiskCalculator {
public:
    struct RiskMetricsResult {
        // VaR estimates
        double monte_carlo_var;
        double historical_var;
        double parametric_var;
        double factor_var;

        // Expected Shortfall
        double expected_shortfall;

        // Portfolio metrics
        double portfolio_value;
        double portfolio_volatility;
        double beta;
        double tracking_error;

        // Risk decomposition
        std::vector<double> position_contributions;
        std::vector<double> factor_contributions;

        // Confidence intervals
        double var_confidence_lower;
        double var_confidence_upper;

        // Computation metadata
        double total_computation_time_ms;
        Timestamp calculation_timestamp;
    };

    RiskCalculator(const PositionEngine& position_engine);

    // Main calculation interface
    std::future<RiskMetricsResult> calculate_portfolio_risk_async(
        double confidence_level = 0.95,
        int horizon_days = 1);

    RiskMetricsResult calculate_portfolio_risk(
        double confidence_level = 0.95,
        int horizon_days = 1);

    // Individual risk measure calculations
    double calculate_var(const std::string& method = "monte_carlo",
                        double confidence_level = 0.95);

    double calculate_expected_shortfall(double confidence_level = 0.95);

    // Risk limit monitoring
    struct RiskLimits {
        double max_var;
        double max_portfolio_value;
        double max_position_concentration;
        double max_sector_concentration;
        std::unordered_map<Currency, double> max_currency_exposure;
    };

    bool check_risk_limits(const RiskLimits& limits);
    std::vector<std::string> get_limit_violations(const RiskLimits& limits);

    // Configuration
    void configure_monte_carlo(const MonteCarloEngine::SimulationParameters& params);
    void configure_historical_simulation(int lookback_days, double confidence_level);

private:
    const PositionEngine& position_engine_;
    std::unique_ptr<MonteCarloEngine> monte_carlo_engine_;
    std::unique_ptr<HistoricalSimulation> historical_simulation_;
    std::unique_ptr<ParametricVar> parametric_var_;
    std::unique_ptr<RiskFactorModel> factor_model_;

    // Market data management
    MonteCarloEngine::MarketFactors get_market_factors();
    HistoricalSimulation::HistoricalData get_historical_data(int lookback_days);
    ParametricVar::CovarianceMatrix get_covariance_matrix();
};

} // namespace risk_engine