#pragma once

#include "core/types.hpp"
#include <cmath>
#include <memory>
#include <vector>
#include <functional>

namespace risk_engine {

class BlackScholesCalculator {
public:
    struct OptionParameters {
        Price spot_price;
        Price strike_price;
        double time_to_expiry; // in years
        double risk_free_rate;
        double volatility;
        double dividend_yield = 0.0;
        OptionType option_type;
    };

    struct OptionResult {
        Price option_price;
        Greeks greeks;
        double implied_volatility = 0.0;
    };

    static OptionResult calculate_option_value(const OptionParameters& params);
    static Greeks calculate_greeks(const OptionParameters& params);

    // Individual Greeks calculations
    static double calculate_delta(const OptionParameters& params);
    static double calculate_gamma(const OptionParameters& params);
    static double calculate_theta(const OptionParameters& params);
    static double calculate_vega(const OptionParameters& params);
    static double calculate_rho(const OptionParameters& params);
    static double calculate_epsilon(const OptionParameters& params); // Dividend sensitivity

    // Implied volatility calculation
    static double calculate_implied_volatility(
        Price market_price,
        const OptionParameters& params,
        double tolerance = 1e-6,
        int max_iterations = 100);

    // American option approximations
    static OptionResult calculate_american_option(const OptionParameters& params);
    static double barone_adesi_whaley_approximation(const OptionParameters& params);

private:
    // Helper functions
    static double normal_cdf(double x);
    static double normal_pdf(double x);
    static double d1(const OptionParameters& params);
    static double d2(const OptionParameters& params);

    // Black-Scholes formula components
    static double call_price(const OptionParameters& params);
    static double put_price(const OptionParameters& params);
};

class BinomialTreeCalculator {
public:
    struct TreeParameters {
        Price spot_price;
        Price strike_price;
        double time_to_expiry;
        double risk_free_rate;
        double volatility;
        double dividend_yield = 0.0;
        OptionType option_type;
        int num_steps = 100;
        bool is_american = false;
        bool use_richardson_extrapolation = true;
    };

    struct TreeResult {
        Price option_price;
        Greeks greeks;
        std::vector<std::vector<double>> price_tree;
        std::vector<std::vector<double>> option_tree;
    };

    static TreeResult calculate_option_value(const TreeParameters& params);
    static Greeks calculate_greeks_finite_difference(const TreeParameters& params);

    // Specific tree models
    static TreeResult cox_ross_rubinstein_tree(const TreeParameters& params);
    static TreeResult jarrow_rudd_tree(const TreeParameters& params);
    static TreeResult leisen_reimer_tree(const TreeParameters& params);

    // Trinomial tree for better convergence
    static TreeResult trinomial_tree(const TreeParameters& params);

private:
    static double calculate_up_factor(double volatility, double dt);
    static double calculate_down_factor(double volatility, double dt);
    static double calculate_risk_neutral_probability(
        double up_factor, double down_factor, double risk_free_rate, double dt);

    // Early exercise check for American options
    static bool should_exercise_early(
        double option_value, double intrinsic_value, bool is_american);
};

class FiniteDifferenceCalculator {
public:
    struct FDParameters {
        Price spot_min;
        Price spot_max;
        double time_to_expiry;
        double risk_free_rate;
        double volatility;
        double dividend_yield = 0.0;
        Price strike_price;
        OptionType option_type;

        // Grid parameters
        int num_spot_steps = 100;
        int num_time_steps = 100;
        bool is_american = false;

        // Scheme type
        enum class Scheme { EXPLICIT, IMPLICIT, CRANK_NICOLSON } scheme = Scheme::CRANK_NICOLSON;
    };

    struct FDResult {
        Price option_price;
        Greeks greeks;
        std::vector<std::vector<double>> price_grid;
        std::vector<std::vector<double>> option_grid;
    };

    static FDResult calculate_option_value(const FDParameters& params);
    static Greeks calculate_greeks_from_grid(
        const std::vector<std::vector<double>>& option_grid,
        const FDParameters& params);

private:
    // Grid setup
    static std::vector<double> create_spot_grid(const FDParameters& params);
    static std::vector<double> create_time_grid(const FDParameters& params);

    // Boundary conditions
    static void apply_boundary_conditions(
        std::vector<std::vector<double>>& grid,
        const FDParameters& params);

    // Finite difference schemes
    static void explicit_scheme_step(
        std::vector<std::vector<double>>& grid,
        int time_idx, const FDParameters& params);

    static void implicit_scheme_step(
        std::vector<std::vector<double>>& grid,
        int time_idx, const FDParameters& params);

    static void crank_nicolson_step(
        std::vector<std::vector<double>>& grid,
        int time_idx, const FDParameters& params);
};

class ExoticOptionsCalculator {
public:
    // Asian options
    struct AsianOptionParams {
        Price spot_price;
        Price strike_price;
        double time_to_expiry;
        double risk_free_rate;
        double volatility;
        std::vector<double> observation_times;
        std::vector<Price> observed_prices;
        enum class AverageType { ARITHMETIC, GEOMETRIC } average_type = AverageType::ARITHMETIC;
        OptionType option_type;
    };

    static OptionResult calculate_asian_option(const AsianOptionParams& params);
    static OptionResult geometric_asian_analytical(const AsianOptionParams& params);
    static OptionResult arithmetic_asian_monte_carlo(
        const AsianOptionParams& params, int num_simulations = 100000);

    // Barrier options
    struct BarrierOptionParams {
        Price spot_price;
        Price strike_price;
        Price barrier_level;
        double time_to_expiry;
        double risk_free_rate;
        double volatility;
        double dividend_yield = 0.0;
        OptionType option_type;
        enum class BarrierType { UP_AND_IN, UP_AND_OUT, DOWN_AND_IN, DOWN_AND_OUT } barrier_type;
        double rebate = 0.0;
    };

    static OptionResult calculate_barrier_option(const BarrierOptionParams& params);

    // Lookback options
    struct LookbackOptionParams {
        Price spot_price;
        double time_to_expiry;
        double risk_free_rate;
        double volatility;
        double dividend_yield = 0.0;
        Price current_min = 0.0; // For partial lookback
        Price current_max = 0.0; // For partial lookback
        enum class LookbackType { FLOATING_STRIKE, FIXED_STRIKE } lookback_type;
        Price fixed_strike = 0.0; // Only for fixed strike lookback
        OptionType option_type;
    };

    static OptionResult calculate_lookback_option(const LookbackOptionParams& params);

    // Rainbow (multi-asset) options
    struct RainbowOptionParams {
        std::vector<Price> spot_prices;
        Price strike_price;
        double time_to_expiry;
        double risk_free_rate;
        std::vector<double> volatilities;
        std::vector<double> dividend_yields;
        std::vector<std::vector<double>> correlation_matrix;
        enum class RainbowType { BEST_OF, WORST_OF, SPREAD } rainbow_type;
        OptionType option_type;
    };

    static OptionResult calculate_rainbow_option(
        const RainbowOptionParams& params, int num_simulations = 100000);

private:
    // Helper functions for exotic options
    static double calculate_barrier_adjustment(const BarrierOptionParams& params);
    static std::vector<double> generate_correlated_brownian_paths(
        const std::vector<std::vector<double>>& correlation_matrix,
        int num_steps, double dt);
};

// Portfolio Greeks Calculator
class PortfolioGreeksCalculator {
public:
    struct PortfolioGreeks {
        double total_delta = 0.0;
        double total_gamma = 0.0;
        double total_theta = 0.0;
        double total_vega = 0.0;
        double total_rho = 0.0;
        double total_epsilon = 0.0;

        // Risk metrics
        double gamma_adjusted_delta = 0.0;
        double vega_weighted_implied_vol = 0.0;
        double time_decay_per_day = 0.0;

        // Concentration metrics
        std::unordered_map<Symbol, Greeks> position_greeks;
        std::unordered_map<std::string, double> sector_deltas;
        std::unordered_map<double, double> strike_gammas; // strike -> gamma

        Timestamp last_calculated;
    };

    PortfolioGreeksCalculator(const PositionEngine& position_engine);

    // Main calculation methods
    PortfolioGreeks calculate_portfolio_greeks();
    PortfolioGreeks calculate_portfolio_greeks_parallel();

    // Greeks aggregation
    Greeks aggregate_position_greeks(const std::vector<std::shared_ptr<Position>>& positions);

    // Risk scenario analysis
    double calculate_portfolio_pnl_for_spot_move(double spot_change_percent);
    double calculate_portfolio_pnl_for_vol_move(double vol_change_percent);
    double calculate_portfolio_pnl_for_time_decay(double days);

    // Greeks hedging
    struct HedgingRecommendation {
        Symbol hedge_instrument;
        Quantity recommended_quantity;
        double hedge_ratio;
        double residual_risk;
        std::string rationale;
    };

    std::vector<HedgingRecommendation> suggest_delta_hedges();
    std::vector<HedgingRecommendation> suggest_gamma_hedges();
    std::vector<HedgingRecommendation> suggest_vega_hedges();

    // Real-time Greeks monitoring
    void start_real_time_monitoring();
    void stop_real_time_monitoring();

    using GreeksCallback = std::function<void(const PortfolioGreeks&)>;
    void register_greeks_callback(GreeksCallback callback);

private:
    const PositionEngine& position_engine_;
    std::vector<GreeksCallback> greeks_callbacks_;

    // Market data for Greeks calculations
    struct MarketDataForGreeks {
        std::unordered_map<Symbol, Price> spot_prices;
        std::unordered_map<Symbol, double> implied_volatilities;
        std::unordered_map<Symbol, double> risk_free_rates;
        std::unordered_map<Symbol, double> dividend_yields;
    };

    MarketDataForGreeks get_market_data();
    Greeks calculate_position_greeks(const Position& position, const MarketDataForGreeks& market_data);

    // Real-time monitoring
    std::atomic<bool> monitoring_active_{false};
    std::thread monitoring_thread_;
    void run_greeks_monitoring_loop();
};

// Volatility Surface Manager
class VolatilitySurface {
public:
    struct VolatilityPoint {
        double strike;
        double time_to_expiry;
        double implied_volatility;
        double bid_vol;
        double ask_vol;
        Timestamp timestamp;
    };

    struct SurfaceParameters {
        Symbol underlying;
        std::vector<VolatilityPoint> vol_points;
        double spot_price;
        double forward_rate;
        Timestamp last_updated;
    };

    VolatilitySurface(const Symbol& underlying);

    // Surface management
    void update_volatility_point(const VolatilityPoint& point);
    void update_full_surface(const std::vector<VolatilityPoint>& points);

    // Interpolation methods
    double interpolate_volatility(double strike, double time_to_expiry) const;
    double extrapolate_volatility(double strike, double time_to_expiry) const;

    // Surface analytics
    double calculate_skew(double time_to_expiry) const;
    double calculate_term_structure(double strike) const;
    double calculate_surface_smoothness() const;

    // Arbitrage detection
    std::vector<std::string> check_arbitrage_conditions() const;
    bool validate_surface_consistency() const;

    // Greeks with surface
    Greeks calculate_greeks_with_surface(
        const BlackScholesCalculator::OptionParameters& params) const;

private:
    Symbol underlying_;
    SurfaceParameters surface_params_;

    // Interpolation helpers
    double bilinear_interpolation(double strike, double time_to_expiry) const;
    double cubic_spline_interpolation(double strike, double time_to_expiry) const;

    // Surface fitting
    void fit_surface_model();
    std::vector<double> sabr_model_calibration() const;
};

} // namespace risk_engine