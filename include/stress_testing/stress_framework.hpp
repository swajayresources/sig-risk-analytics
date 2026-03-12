/**
 * Comprehensive Stress Testing and Scenario Analysis Framework
 *
 * Advanced stress testing capabilities for financial risk management including:
 * - Historical scenario replay and analysis
 * - Custom shock construction and Monte Carlo stress testing
 * - Multi-factor risk modeling and correlation breakdown
 * - Regulatory compliance scenarios (CCAR, EBA, Basel III)
 * - Climate risk and ESG stress testing
 * - Real-time stress execution and monitoring
 */

#pragma once

#include "hpc/hpc_framework.hpp"
#include "core/types.hpp"
#include <vector>
#include <unordered_map>
#include <string>
#include <chrono>
#include <memory>
#include <functional>
#include <optional>

namespace risk_analytics {
namespace stress_testing {

/**
 * Risk factor types for stress testing
 */
enum class RiskFactorType : uint8_t {
    EQUITY_INDEX = 1,
    INTEREST_RATE = 2,
    FX_RATE = 3,
    COMMODITY = 4,
    CREDIT_SPREAD = 5,
    VOLATILITY = 6,
    CORRELATION = 7,
    LIQUIDITY = 8,
    INFLATION = 9,
    GDP_GROWTH = 10,
    UNEMPLOYMENT = 11,
    CLIMATE_RISK = 12
};

/**
 * Stress test scenario types
 */
enum class ScenarioType : uint8_t {
    HISTORICAL = 1,        // Historical crisis replay
    HYPOTHETICAL = 2,      // Custom constructed scenarios
    MONTE_CARLO = 3,       // Stochastic scenario generation
    REGULATORY = 4,        // Regulatory required scenarios
    TAIL_RISK = 5,         // Extreme tail event scenarios
    CLIMATE = 6,           // Climate and ESG scenarios
    GEOPOLITICAL = 7,      // Geopolitical event scenarios
    MARKET_STRUCTURE = 8   // Market microstructure changes
};

/**
 * Regulatory stress test types
 */
enum class RegulatoryScenarioType : uint8_t {
    CCAR_SEVERELY_ADVERSE = 1,    // Fed CCAR severely adverse scenario
    CCAR_ADVERSE = 2,             // Fed CCAR adverse scenario
    EBA_ADVERSE = 3,              // European Banking Authority adverse
    BASEL_III = 4,                // Basel III stress scenarios
    BOE_STRESS = 5,               // Bank of England stress tests
    CUSTOM_REGULATORY = 6         // Custom regulatory scenarios
};

/**
 * Risk factor shock definition
 */
struct RiskFactorShock {
    RiskFactorType factor_type;
    std::string factor_name;
    std::string currency;
    std::string sector;
    double shock_magnitude;        // Percentage or absolute change
    bool is_relative;             // true for %, false for absolute
    uint32_t time_horizon_days;   // Time horizon for the shock
    double recovery_half_life;    // Half-life for shock recovery
    std::vector<std::string> affected_instruments;

    RiskFactorShock(RiskFactorType type, const std::string& name, double magnitude)
        : factor_type(type), factor_name(name), shock_magnitude(magnitude),
          is_relative(true), time_horizon_days(1), recovery_half_life(0.0) {}
};

/**
 * Historical crisis scenario definitions
 */
struct HistoricalScenario {
    std::string scenario_name;
    std::string description;
    std::chrono::system_clock::time_point start_date;
    std::chrono::system_clock::time_point end_date;
    uint32_t duration_days;
    std::vector<RiskFactorShock> factor_shocks;
    std::unordered_map<std::string, std::vector<double>> historical_data;
    double severity_score;        // 1-10 scale
    std::vector<std::string> affected_regions;
    std::vector<std::string> affected_sectors;

    HistoricalScenario(const std::string& name, const std::string& desc)
        : scenario_name(name), description(desc), duration_days(0), severity_score(5.0) {}
};

/**
 * Stress test configuration
 */
struct StressTestConfig {
    ScenarioType scenario_type;
    std::string scenario_name;
    uint32_t time_horizon_days;
    uint32_t monte_carlo_paths;
    double confidence_level;
    bool include_second_order_effects;
    bool use_historical_correlations;
    bool apply_liquidity_adjustments;
    std::vector<std::string> risk_factors_to_shock;
    std::unordered_map<std::string, double> custom_correlations;

    StressTestConfig()
        : scenario_type(ScenarioType::HYPOTHETICAL), time_horizon_days(1),
          monte_carlo_paths(10000), confidence_level(0.95),
          include_second_order_effects(true), use_historical_correlations(true),
          apply_liquidity_adjustments(false) {}
};

/**
 * Stress test results
 */
struct StressTestResult {
    std::string scenario_name;
    ScenarioType scenario_type;
    std::chrono::system_clock::time_point execution_time;
    uint32_t computation_time_ms;

    // Portfolio-level results
    double portfolio_value_base;
    double portfolio_value_stressed;
    double portfolio_pnl;
    double portfolio_pnl_percent;

    // Risk metric results
    double var_base;
    double var_stressed;
    double expected_shortfall_base;
    double expected_shortfall_stressed;
    double max_drawdown;

    // Position-level results
    std::unordered_map<std::string, double> position_pnl;
    std::unordered_map<std::string, double> position_pnl_percent;

    // Risk factor contributions
    std::unordered_map<std::string, double> factor_contributions;

    // Concentration and correlation analysis
    double herfindahl_index;
    std::unordered_map<std::string, double> sector_concentrations;
    std::unordered_map<std::string, double> correlation_breakdown;

    // Liquidity impact
    std::optional<double> liquidity_adjusted_pnl;
    std::optional<double> funding_cost_impact;

    // Statistical measures
    double probability_of_loss;
    double tail_expectation;
    std::vector<double> percentile_losses;

    StressTestResult() : portfolio_value_base(0), portfolio_value_stressed(0),
                        portfolio_pnl(0), portfolio_pnl_percent(0),
                        var_base(0), var_stressed(0), expected_shortfall_base(0),
                        expected_shortfall_stressed(0), max_drawdown(0),
                        herfindahl_index(0), probability_of_loss(0), tail_expectation(0) {}
};

/**
 * Climate risk scenario parameters
 */
struct ClimateRiskScenario {
    enum class TransitionType {
        ORDERLY,           // Gradual transition to low-carbon economy
        DISORDERLY,        // Sudden policy changes and market disruption
        HOT_HOUSE          // Failed transition, physical risks dominate
    };

    TransitionType transition_type;
    double temperature_increase_celsius;
    uint32_t time_horizon_years;
    double carbon_price_usd_per_ton;
    double renewable_energy_share;
    std::unordered_map<std::string, double> sector_transition_costs;
    std::unordered_map<std::string, double> physical_risk_multipliers;
    double stranded_asset_haircut;
    double green_premium_discount;

    ClimateRiskScenario() : transition_type(TransitionType::ORDERLY),
                          temperature_increase_celsius(1.5), time_horizon_years(30),
                          carbon_price_usd_per_ton(100.0), renewable_energy_share(0.8),
                          stranded_asset_haircut(0.3), green_premium_discount(0.1) {}
};

/**
 * Forward-looking stress scenario generator
 */
class ScenarioGenerator {
private:
    std::mt19937 rng_;
    std::unordered_map<RiskFactorType, std::normal_distribution<double>> factor_distributions_;
    std::unordered_map<std::string, std::vector<double>> historical_correlations_;

public:
    ScenarioGenerator(uint32_t seed = std::random_device{}())
        : rng_(seed) {
        initialize_factor_distributions();
    }

    /**
     * Generate Monte Carlo stress scenarios
     */
    std::vector<std::vector<RiskFactorShock>> generate_monte_carlo_scenarios(
        const std::vector<RiskFactorType>& factors,
        uint32_t num_scenarios,
        uint32_t time_horizon_days,
        const std::unordered_map<std::string, double>& correlation_matrix = {}
    );

    /**
     * Generate correlated factor shocks
     */
    std::vector<RiskFactorShock> generate_correlated_shocks(
        const std::vector<RiskFactorType>& factors,
        const std::vector<std::vector<double>>& correlation_matrix
    );

    /**
     * Generate tail risk scenarios
     */
    std::vector<RiskFactorShock> generate_tail_risk_scenario(
        double tail_probability = 0.01,
        uint32_t time_horizon_days = 1
    );

    /**
     * Generate climate transition scenarios
     */
    std::vector<RiskFactorShock> generate_climate_scenario(
        const ClimateRiskScenario& climate_params
    );

    /**
     * Generate regulatory stress scenarios
     */
    std::vector<RiskFactorShock> generate_regulatory_scenario(
        RegulatoryScenarioType reg_type,
        const std::string& jurisdiction = "US"
    );

private:
    void initialize_factor_distributions();
    std::vector<std::vector<double>> generate_correlation_matrix(
        const std::vector<RiskFactorType>& factors
    );
};

/**
 * Historical scenario database and replay system
 */
class HistoricalScenarioDatabase {
private:
    std::unordered_map<std::string, HistoricalScenario> scenarios_;
    std::unordered_map<std::string, std::unordered_map<std::string, std::vector<double>>> historical_data_;

public:
    HistoricalScenarioDatabase() {
        initialize_historical_scenarios();
        load_historical_data();
    }

    /**
     * Get predefined historical scenario
     */
    const HistoricalScenario* get_scenario(const std::string& scenario_name) const;

    /**
     * List all available scenarios
     */
    std::vector<std::string> list_scenarios() const;

    /**
     * Get historical market data for scenario replay
     */
    std::vector<double> get_historical_data(
        const std::string& scenario_name,
        const std::string& risk_factor,
        const std::chrono::system_clock::time_point& start_date,
        const std::chrono::system_clock::time_point& end_date
    ) const;

    /**
     * Add custom historical scenario
     */
    void add_scenario(const HistoricalScenario& scenario);

    /**
     * Calculate factor shocks from historical data
     */
    std::vector<RiskFactorShock> calculate_historical_shocks(
        const std::string& scenario_name,
        const std::vector<std::string>& risk_factors
    ) const;

private:
    void initialize_historical_scenarios();
    void load_historical_data();
    void add_lehman_crisis_scenario();
    void add_covid19_scenario();
    void add_dot_com_crash_scenario();
    void add_european_debt_crisis_scenario();
    void add_flash_crash_scenario();
    void add_brexit_scenario();
    void add_china_market_crash_scenario();
};

/**
 * Portfolio sensitivity analyzer
 */
class PortfolioSensitivityAnalyzer {
private:
    const Portfolio& portfolio_;

public:
    explicit PortfolioSensitivityAnalyzer(const Portfolio& portfolio)
        : portfolio_(portfolio) {}

    /**
     * Calculate portfolio sensitivity to risk factors
     */
    std::unordered_map<std::string, double> calculate_factor_sensitivities(
        const std::vector<RiskFactorShock>& shocks
    ) const;

    /**
     * Calculate concentration risk metrics
     */
    struct ConcentrationRisk {
        double herfindahl_index;
        std::unordered_map<std::string, double> sector_concentrations;
        std::unordered_map<std::string, double> geographic_concentrations;
        std::unordered_map<std::string, double> issuer_concentrations;
        double largest_position_weight;
        double top_10_positions_weight;
    };

    ConcentrationRisk analyze_concentration_risk() const;

    /**
     * Analyze correlation breakdown scenarios
     */
    std::unordered_map<std::string, double> analyze_correlation_breakdown(
        const std::vector<std::string>& asset_pairs,
        double stress_correlation = 0.9
    ) const;

    /**
     * Calculate portfolio Greeks aggregation
     */
    struct PortfolioGreeks {
        double net_delta;
        double net_gamma;
        double net_theta;
        double net_vega;
        double net_rho;
        std::unordered_map<std::string, double> delta_by_underlying;
        std::unordered_map<std::string, double> gamma_by_underlying;
    };

    PortfolioGreeks calculate_portfolio_greeks() const;
};

/**
 * Liquidity stress testing framework
 */
class LiquidityStressTester {
public:
    enum class LiquidityScenario {
        NORMAL_MARKET,
        STRESSED_MARKET,
        SEVERE_STRESS,
        MARKET_CLOSURE
    };

    struct LiquidityMetrics {
        double bid_ask_spread_normal;
        double bid_ask_spread_stressed;
        double market_impact_normal;
        double market_impact_stressed;
        uint32_t time_to_liquidate_days;
        double liquidity_cost_basis_points;
        double funding_gap;
        double cash_available;
        double committed_facilities;
    };

private:
    std::unordered_map<std::string, LiquidityMetrics> asset_liquidity_;

public:
    /**
     * Set liquidity parameters for assets
     */
    void set_asset_liquidity(const std::string& asset, const LiquidityMetrics& metrics);

    /**
     * Calculate liquidity-adjusted portfolio value
     */
    double calculate_liquidity_adjusted_value(
        const Portfolio& portfolio,
        LiquidityScenario scenario,
        uint32_t liquidation_horizon_days = 30
    ) const;

    /**
     * Calculate funding stress impact
     */
    double calculate_funding_stress_impact(
        const Portfolio& portfolio,
        double funding_cost_increase_bps,
        uint32_t stress_duration_days
    ) const;

    /**
     * Assess portfolio liquidity risk
     */
    struct LiquidityRiskAssessment {
        double liquidity_coverage_ratio;
        double net_stable_funding_ratio;
        double cash_flow_gap_30_days;
        double cash_flow_gap_90_days;
        std::unordered_map<std::string, double> asset_liquidity_scores;
        double portfolio_liquidity_score;
    };

    LiquidityRiskAssessment assess_liquidity_risk(const Portfolio& portfolio) const;
};

/**
 * Multi-factor stress testing engine
 */
class MultiFactorStressTester {
private:
    std::unique_ptr<ScenarioGenerator> scenario_generator_;
    std::unique_ptr<HistoricalScenarioDatabase> historical_db_;
    std::unique_ptr<PortfolioSensitivityAnalyzer> sensitivity_analyzer_;
    std::unique_ptr<LiquidityStressTester> liquidity_tester_;

    // Risk factor models
    std::unordered_map<RiskFactorType, std::function<double(const RiskFactorShock&, const Position&)>>
        factor_impact_models_;

public:
    MultiFactorStressTester();
    ~MultiFactorStressTester() = default;

    /**
     * Execute comprehensive stress test
     */
    StressTestResult execute_stress_test(
        const Portfolio& portfolio,
        const StressTestConfig& config
    );

    /**
     * Execute historical scenario replay
     */
    StressTestResult execute_historical_scenario(
        const Portfolio& portfolio,
        const std::string& scenario_name,
        const StressTestConfig& config = StressTestConfig{}
    );

    /**
     * Execute Monte Carlo stress testing
     */
    std::vector<StressTestResult> execute_monte_carlo_stress_test(
        const Portfolio& portfolio,
        const std::vector<RiskFactorType>& factors,
        uint32_t num_scenarios,
        const StressTestConfig& config = StressTestConfig{}
    );

    /**
     * Execute regulatory stress test
     */
    StressTestResult execute_regulatory_stress_test(
        const Portfolio& portfolio,
        RegulatoryScenarioType reg_scenario,
        const std::string& jurisdiction = "US"
    );

    /**
     * Execute climate risk stress test
     */
    StressTestResult execute_climate_stress_test(
        const Portfolio& portfolio,
        const ClimateRiskScenario& climate_scenario
    );

    /**
     * Execute tail risk analysis
     */
    std::vector<StressTestResult> execute_tail_risk_analysis(
        const Portfolio& portfolio,
        const std::vector<double>& tail_probabilities = {0.01, 0.005, 0.001}
    );

    /**
     * Execute reverse stress testing
     */
    struct ReverseStressResult {
        double target_loss_amount;
        std::vector<RiskFactorShock> required_shocks;
        double scenario_probability;
        std::string scenario_description;
    };

    ReverseStressResult execute_reverse_stress_test(
        const Portfolio& portfolio,
        double target_loss_amount
    );

private:
    void initialize_factor_models();
    double calculate_position_impact(
        const Position& position,
        const std::vector<RiskFactorShock>& shocks,
        const StressTestConfig& config
    );

    std::vector<RiskFactorShock> apply_correlation_structure(
        const std::vector<RiskFactorShock>& base_shocks,
        const std::unordered_map<std::string, double>& correlations
    );
};

/**
 * Stress test reporting and visualization
 */
class StressTestReporter {
public:
    /**
     * Generate comprehensive stress test report
     */
    std::string generate_stress_test_report(
        const StressTestResult& result,
        const Portfolio& portfolio
    ) const;

    /**
     * Generate regulatory stress test report
     */
    std::string generate_regulatory_report(
        const std::vector<StressTestResult>& results,
        RegulatoryScenarioType reg_type
    ) const;

    /**
     * Generate risk dashboard JSON for web visualization
     */
    std::string generate_dashboard_json(
        const std::vector<StressTestResult>& results
    ) const;

    /**
     * Export results to CSV format
     */
    void export_to_csv(
        const std::vector<StressTestResult>& results,
        const std::string& filename
    ) const;

    /**
     * Generate executive summary
     */
    std::string generate_executive_summary(
        const std::vector<StressTestResult>& results
    ) const;

    /**
     * Generate scenario comparison analysis
     */
    std::string generate_scenario_comparison(
        const std::vector<StressTestResult>& results
    ) const;

private:
    std::string format_currency(double amount) const;
    std::string format_percentage(double percentage) const;
    std::string get_risk_rating(double pnl_percent) const;
};

/**
 * Real-time stress testing monitor
 */
class RealTimeStressMonitor {
private:
    MultiFactorStressTester& stress_tester_;
    std::vector<StressTestConfig> continuous_scenarios_;
    std::thread monitoring_thread_;
    std::atomic<bool> running_{false};
    std::chrono::seconds update_frequency_{300}; // 5 minutes default

    // Alert thresholds
    double var_alert_threshold_;
    double pnl_alert_threshold_;
    std::function<void(const StressTestResult&)> alert_callback_;

public:
    explicit RealTimeStressMonitor(MultiFactorStressTester& tester)
        : stress_tester_(tester), var_alert_threshold_(0.02), pnl_alert_threshold_(0.05) {}

    ~RealTimeStressMonitor() {
        stop();
    }

    /**
     * Start continuous stress monitoring
     */
    void start(const Portfolio& portfolio);

    /**
     * Stop monitoring
     */
    void stop();

    /**
     * Add scenario to continuous monitoring
     */
    void add_continuous_scenario(const StressTestConfig& config);

    /**
     * Set alert thresholds and callback
     */
    void set_alert_parameters(
        double var_threshold,
        double pnl_threshold,
        std::function<void(const StressTestResult&)> callback
    );

    /**
     * Get latest stress test results
     */
    std::vector<StressTestResult> get_latest_results() const;

private:
    void monitoring_worker(const Portfolio& portfolio);
    void check_alert_conditions(const StressTestResult& result);
};

} // namespace stress_testing
} // namespace risk_analytics