#include "risk/greeks_calculator.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace risk_engine {

constexpr double PI = 3.14159265358979323846;
constexpr double SQRT_2PI = std::sqrt(2.0 * PI);

BlackScholesCalculator::OptionResult BlackScholesCalculator::calculate_option_value(const OptionParameters& params) {
    OptionResult result;

    if (params.option_type == OptionType::CALL) {
        result.option_price = call_price(params);
    } else {
        result.option_price = put_price(params);
    }

    result.greeks = calculate_greeks(params);
    result.greeks.last_calculated = std::chrono::high_resolution_clock::now();

    return result;
}

Greeks BlackScholesCalculator::calculate_greeks(const OptionParameters& params) {
    Greeks greeks;

    greeks.delta = calculate_delta(params);
    greeks.gamma = calculate_gamma(params);
    greeks.theta = calculate_theta(params);
    greeks.vega = calculate_vega(params);
    greeks.rho = calculate_rho(params);
    greeks.epsilon = calculate_epsilon(params);

    return greeks;
}

double BlackScholesCalculator::calculate_delta(const OptionParameters& params) {
    if (params.time_to_expiry <= 0.0) {
        // At expiration
        if (params.option_type == OptionType::CALL) {
            return (params.spot_price > params.strike_price) ? 1.0 : 0.0;
        } else {
            return (params.spot_price < params.strike_price) ? -1.0 : 0.0;
        }
    }

    double d1_val = d1(params);
    double dividend_discount = std::exp(-params.dividend_yield * params.time_to_expiry);

    if (params.option_type == OptionType::CALL) {
        return dividend_discount * normal_cdf(d1_val);
    } else {
        return dividend_discount * (normal_cdf(d1_val) - 1.0);
    }
}

double BlackScholesCalculator::calculate_gamma(const OptionParameters& params) {
    if (params.time_to_expiry <= 0.0) {
        return 0.0; // Gamma is zero at expiration
    }

    double d1_val = d1(params);
    double dividend_discount = std::exp(-params.dividend_yield * params.time_to_expiry);

    return (dividend_discount * normal_pdf(d1_val)) /
           (params.spot_price * params.volatility * std::sqrt(params.time_to_expiry));
}

double BlackScholesCalculator::calculate_theta(const OptionParameters& params) {
    if (params.time_to_expiry <= 0.0) {
        return 0.0;
    }

    double d1_val = d1(params);
    double d2_val = d2(params);
    double sqrt_t = std::sqrt(params.time_to_expiry);
    double dividend_discount = std::exp(-params.dividend_yield * params.time_to_expiry);
    double risk_free_discount = std::exp(-params.risk_free_rate * params.time_to_expiry);

    double term1 = -(params.spot_price * normal_pdf(d1_val) * params.volatility * dividend_discount) / (2.0 * sqrt_t);

    if (params.option_type == OptionType::CALL) {
        double term2 = params.dividend_yield * params.spot_price * normal_cdf(d1_val) * dividend_discount;
        double term3 = params.risk_free_rate * params.strike_price * risk_free_discount * normal_cdf(d2_val);
        return (term1 + term2 - term3) / 365.0; // Convert to per-day
    } else {
        double term2 = -params.dividend_yield * params.spot_price * normal_cdf(-d1_val) * dividend_discount;
        double term3 = -params.risk_free_rate * params.strike_price * risk_free_discount * normal_cdf(-d2_val);
        return (term1 + term2 - term3) / 365.0; // Convert to per-day
    }
}

double BlackScholesCalculator::calculate_vega(const OptionParameters& params) {
    if (params.time_to_expiry <= 0.0) {
        return 0.0;
    }

    double d1_val = d1(params);
    double dividend_discount = std::exp(-params.dividend_yield * params.time_to_expiry);

    return (params.spot_price * dividend_discount * normal_pdf(d1_val) * std::sqrt(params.time_to_expiry)) / 100.0; // Per 1% vol change
}

double BlackScholesCalculator::calculate_rho(const OptionParameters& params) {
    if (params.time_to_expiry <= 0.0) {
        return 0.0;
    }

    double d2_val = d2(params);
    double risk_free_discount = std::exp(-params.risk_free_rate * params.time_to_expiry);

    if (params.option_type == OptionType::CALL) {
        return (params.strike_price * params.time_to_expiry * risk_free_discount * normal_cdf(d2_val)) / 100.0; // Per 1% rate change
    } else {
        return (-params.strike_price * params.time_to_expiry * risk_free_discount * normal_cdf(-d2_val)) / 100.0;
    }
}

double BlackScholesCalculator::calculate_epsilon(const OptionParameters& params) {
    if (params.time_to_expiry <= 0.0) {
        return 0.0;
    }

    double d1_val = d1(params);
    double dividend_discount = std::exp(-params.dividend_yield * params.time_to_expiry);

    if (params.option_type == OptionType::CALL) {
        return (-params.spot_price * params.time_to_expiry * dividend_discount * normal_cdf(d1_val)) / 100.0;
    } else {
        return (params.spot_price * params.time_to_expiry * dividend_discount * normal_cdf(-d1_val)) / 100.0;
    }
}

double BlackScholesCalculator::calculate_implied_volatility(
    Price market_price, const OptionParameters& params, double tolerance, int max_iterations) {

    // Newton-Raphson method for implied volatility
    double vol_estimate = 0.2; // Initial guess: 20% volatility

    for (int i = 0; i < max_iterations; ++i) {
        OptionParameters temp_params = params;
        temp_params.volatility = vol_estimate;

        double calculated_price = (params.option_type == OptionType::CALL) ?
                                 call_price(temp_params) : put_price(temp_params);

        double price_diff = calculated_price - market_price;

        if (std::abs(price_diff) < tolerance) {
            return vol_estimate;
        }

        double vega = calculate_vega(temp_params) * 100.0; // Convert back from per 1%

        if (std::abs(vega) < 1e-10) {
            break; // Avoid division by zero
        }

        vol_estimate = vol_estimate - price_diff / vega;

        // Keep volatility in reasonable bounds
        vol_estimate = std::max(0.001, std::min(vol_estimate, 10.0));
    }

    return vol_estimate;
}

BlackScholesCalculator::OptionResult BlackScholesCalculator::calculate_american_option(const OptionParameters& params) {
    // Use Barone-Adesi-Whaley approximation for American options
    OptionResult result;
    result.option_price = barone_adesi_whaley_approximation(params);
    result.greeks = calculate_greeks(params); // Approximate greeks using European formulas
    return result;
}

double BlackScholesCalculator::barone_adesi_whaley_approximation(const OptionParameters& params) {
    // Simplified Barone-Adesi-Whaley approximation
    // For full implementation, this would be much more complex

    double european_price = (params.option_type == OptionType::CALL) ?
                           call_price(params) : put_price(params);

    if (params.dividend_yield <= 0.0) {
        // American call with no dividends has same value as European call
        if (params.option_type == OptionType::CALL) {
            return european_price;
        }
    }

    // Simplified early exercise premium calculation
    double intrinsic_value = (params.option_type == OptionType::CALL) ?
                            std::max(0.0, params.spot_price - params.strike_price) :
                            std::max(0.0, params.strike_price - params.spot_price);

    double early_exercise_premium = std::max(0.0, intrinsic_value - european_price) * 0.1; // Simplified

    return european_price + early_exercise_premium;
}

double BlackScholesCalculator::normal_cdf(double x) {
    // Abramowitz and Stegun approximation
    if (x < 0) {
        return 1.0 - normal_cdf(-x);
    }

    double k = 1.0 / (1.0 + 0.2316419 * x);
    double k_sq = k * k;
    double k_cu = k_sq * k;
    double k_qu = k_cu * k;
    double k_qi = k_qu * k;

    double A1 = 0.319381530;
    double A2 = -0.356563782;
    double A3 = 1.781477937;
    double A4 = -1.821255978;
    double A5 = 1.330274429;

    double result = 1.0 - (1.0 / SQRT_2PI) * std::exp(-0.5 * x * x) *
                    (A1 * k + A2 * k_sq + A3 * k_cu + A4 * k_qu + A5 * k_qi);

    return result;
}

double BlackScholesCalculator::normal_pdf(double x) {
    return (1.0 / SQRT_2PI) * std::exp(-0.5 * x * x);
}

double BlackScholesCalculator::d1(const OptionParameters& params) {
    if (params.time_to_expiry <= 0.0 || params.volatility <= 0.0) {
        return 0.0;
    }

    double log_ratio = std::log(params.spot_price / params.strike_price);
    double vol_term = (params.risk_free_rate - params.dividend_yield + 0.5 * params.volatility * params.volatility) * params.time_to_expiry;
    double denominator = params.volatility * std::sqrt(params.time_to_expiry);

    return (log_ratio + vol_term) / denominator;
}

double BlackScholesCalculator::d2(const OptionParameters& params) {
    return d1(params) - params.volatility * std::sqrt(params.time_to_expiry);
}

double BlackScholesCalculator::call_price(const OptionParameters& params) {
    if (params.time_to_expiry <= 0.0) {
        return std::max(0.0, params.spot_price - params.strike_price);
    }

    double d1_val = d1(params);
    double d2_val = d2(params);

    double spot_term = params.spot_price * std::exp(-params.dividend_yield * params.time_to_expiry) * normal_cdf(d1_val);
    double strike_term = params.strike_price * std::exp(-params.risk_free_rate * params.time_to_expiry) * normal_cdf(d2_val);

    return spot_term - strike_term;
}

double BlackScholesCalculator::put_price(const OptionParameters& params) {
    if (params.time_to_expiry <= 0.0) {
        return std::max(0.0, params.strike_price - params.spot_price);
    }

    double d1_val = d1(params);
    double d2_val = d2(params);

    double strike_term = params.strike_price * std::exp(-params.risk_free_rate * params.time_to_expiry) * normal_cdf(-d2_val);
    double spot_term = params.spot_price * std::exp(-params.dividend_yield * params.time_to_expiry) * normal_cdf(-d1_val);

    return strike_term - spot_term;
}

// Portfolio Greeks Calculator Implementation
PortfolioGreeksCalculator::PortfolioGreeksCalculator(const PositionEngine& position_engine)
    : position_engine_(position_engine) {
}

PortfolioGreeksCalculator::PortfolioGreeks PortfolioGreeksCalculator::calculate_portfolio_greeks() {
    auto positions = position_engine_.get_all_positions();
    auto market_data = get_market_data();

    PortfolioGreeks portfolio_greeks;
    portfolio_greeks.last_calculated = std::chrono::high_resolution_clock::now();

    for (const auto& position : positions) {
        if (position->instrument.asset_type == AssetType::OPTION) {
            Greeks pos_greeks = calculate_position_greeks(*position, market_data);
            Quantity qty = position->quantity.load();

            // Aggregate portfolio Greeks
            portfolio_greeks.total_delta += pos_greeks.delta * qty;
            portfolio_greeks.total_gamma += pos_greeks.gamma * qty;
            portfolio_greeks.total_theta += pos_greeks.theta * qty;
            portfolio_greeks.total_vega += pos_greeks.vega * qty;
            portfolio_greeks.total_rho += pos_greeks.rho * qty;
            portfolio_greeks.total_epsilon += pos_greeks.epsilon * qty;

            // Store position-level Greeks
            portfolio_greeks.position_greeks[position->instrument.symbol] = pos_greeks;
        } else if (position->instrument.asset_type == AssetType::EQUITY ||
                  position->instrument.asset_type == AssetType::FUTURE) {
            // Linear instruments have delta = quantity
            Quantity qty = position->quantity.load();
            portfolio_greeks.total_delta += qty;
        }
    }

    // Calculate risk metrics
    portfolio_greeks.gamma_adjusted_delta = portfolio_greeks.total_delta +
                                          0.5 * portfolio_greeks.total_gamma * 0.01; // Assume 1% move

    portfolio_greeks.time_decay_per_day = portfolio_greeks.total_theta;

    return portfolio_greeks;
}

Greeks PortfolioGreeksCalculator::calculate_position_greeks(
    const Position& position, const MarketDataForGreeks& market_data) {

    if (position.instrument.asset_type != AssetType::OPTION) {
        return Greeks{}; // Only calculate for options
    }

    auto spot_it = market_data.spot_prices.find(position.instrument.symbol);
    auto vol_it = market_data.implied_volatilities.find(position.instrument.symbol);
    auto rate_it = market_data.risk_free_rates.find(position.instrument.symbol);
    auto div_it = market_data.dividend_yields.find(position.instrument.symbol);

    if (spot_it == market_data.spot_prices.end() ||
        vol_it == market_data.implied_volatilities.end() ||
        rate_it == market_data.risk_free_rates.end()) {
        return Greeks{}; // Missing market data
    }

    // Calculate time to expiry
    auto now = std::chrono::high_resolution_clock::now();
    auto time_diff = position.instrument.expiry - now;
    double time_to_expiry = std::chrono::duration<double>(time_diff).count() / (365.25 * 24 * 3600); // Convert to years

    if (time_to_expiry <= 0.0) {
        return Greeks{}; // Expired option
    }

    BlackScholesCalculator::OptionParameters params;
    params.spot_price = spot_it->second;
    params.strike_price = position.instrument.strike;
    params.time_to_expiry = time_to_expiry;
    params.risk_free_rate = rate_it->second;
    params.volatility = vol_it->second;
    params.dividend_yield = (div_it != market_data.dividend_yields.end()) ? div_it->second : 0.0;
    params.option_type = position.instrument.option_type;

    return BlackScholesCalculator::calculate_greeks(params);
}

PortfolioGreeksCalculator::MarketDataForGreeks PortfolioGreeksCalculator::get_market_data() {
    MarketDataForGreeks market_data;

    // In a real implementation, this would fetch from market data feeds
    // For now, using placeholder values
    auto positions = position_engine_.get_all_positions();

    for (const auto& position : positions) {
        const auto& symbol = position->instrument.symbol;

        if (market_data.spot_prices.find(symbol) == market_data.spot_prices.end()) {
            // Use current market price from position
            market_data.spot_prices[symbol] = position->market_price.load();

            // Default values (in practice, these would come from market data)
            market_data.implied_volatilities[symbol] = 0.2; // 20% vol
            market_data.risk_free_rates[symbol] = 0.05; // 5% risk-free rate
            market_data.dividend_yields[symbol] = 0.02; // 2% dividend yield
        }
    }

    return market_data;
}

double PortfolioGreeksCalculator::calculate_portfolio_pnl_for_spot_move(double spot_change_percent) {
    auto portfolio_greeks = calculate_portfolio_greeks();

    // P&L approximation using Greeks
    double spot_change = spot_change_percent / 100.0;
    double delta_pnl = portfolio_greeks.total_delta * spot_change;
    double gamma_pnl = 0.5 * portfolio_greeks.total_gamma * spot_change * spot_change;

    return delta_pnl + gamma_pnl;
}

double PortfolioGreeksCalculator::calculate_portfolio_pnl_for_vol_move(double vol_change_percent) {
    auto portfolio_greeks = calculate_portfolio_greeks();
    return portfolio_greeks.total_vega * vol_change_percent;
}

double PortfolioGreeksCalculator::calculate_portfolio_pnl_for_time_decay(double days) {
    auto portfolio_greeks = calculate_portfolio_greeks();
    return portfolio_greeks.total_theta * days;
}

std::vector<PortfolioGreeksCalculator::HedgingRecommendation> PortfolioGreeksCalculator::suggest_delta_hedges() {
    std::vector<HedgingRecommendation> recommendations;
    auto portfolio_greeks = calculate_portfolio_greeks();

    if (std::abs(portfolio_greeks.total_delta) > 100.0) { // Threshold for hedging
        HedgingRecommendation hedge;
        hedge.hedge_instrument = "SPY"; // Use SPY as hedge instrument
        hedge.recommended_quantity = -static_cast<Quantity>(portfolio_greeks.total_delta);
        hedge.hedge_ratio = 1.0;
        hedge.residual_risk = 0.0; // Perfect delta hedge
        hedge.rationale = "Delta hedge to neutralize directional exposure";

        recommendations.push_back(hedge);
    }

    return recommendations;
}

} // namespace risk_engine