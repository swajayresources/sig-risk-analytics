#include "stress_testing/stress_framework.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <execution>

namespace stress_testing {

class ShockConstructorImpl {
private:
    std::mt19937_64 rng_;
    std::uniform_real_distribution<double> uniform_dist_;
    std::normal_distribution<double> normal_dist_;

    // Factor model parameters
    struct FactorModel {
        Eigen::MatrixXd factor_loadings;
        Eigen::VectorXd factor_means;
        Eigen::MatrixXd factor_covariance;
        std::vector<std::string> factor_names;
    };

    std::unordered_map<std::string, FactorModel> factor_models_;

public:
    ShockConstructorImpl() :
        rng_(std::random_device{}()),
        uniform_dist_(0.0, 1.0),
        normal_dist_(0.0, 1.0) {}

    void calibrate_factor_model(const std::string& model_name,
                               const std::vector<RiskFactor>& factors,
                               const Eigen::MatrixXd& historical_returns) {

        FactorModel model;

        // Perform Principal Component Analysis for factor extraction
        Eigen::MatrixXd centered_returns = center_matrix(historical_returns);
        Eigen::MatrixXd covariance = (centered_returns.transpose() * centered_returns) /
                                   (historical_returns.rows() - 1);

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(covariance);
        Eigen::VectorXd eigenvalues = solver.eigenvalues().reverse();
        Eigen::MatrixXd eigenvectors = solver.eigenvectors().rowwise().reverse();

        // Select number of factors explaining 95% of variance
        double total_variance = eigenvalues.sum();
        double cumulative_variance = 0.0;
        int num_factors = 0;

        for (int i = 0; i < eigenvalues.size(); ++i) {
            cumulative_variance += eigenvalues(i);
            num_factors++;
            if (cumulative_variance / total_variance >= 0.95) {
                break;
            }
        }

        // Extract factor loadings (first num_factors principal components)
        model.factor_loadings = eigenvectors.leftCols(num_factors);
        model.factor_covariance = eigenvalues.head(num_factors).asDiagonal();
        model.factor_means = Eigen::VectorXd::Zero(num_factors);

        // Generate factor names
        model.factor_names.clear();
        for (int i = 0; i < num_factors; ++i) {
            model.factor_names.push_back("Factor_" + std::to_string(i + 1));
        }

        factor_models_[model_name] = std::move(model);
    }

    std::vector<RiskFactorShock> construct_factor_based_shocks(
        const std::string& model_name,
        const std::vector<RiskFactor>& factors,
        const Eigen::VectorXd& factor_shocks) {

        auto it = factor_models_.find(model_name);
        if (it == factor_models_.end()) {
            throw std::runtime_error("Factor model not found: " + model_name);
        }

        const FactorModel& model = it->second;

        // Transform factor shocks to asset-specific shocks
        Eigen::VectorXd asset_shocks = model.factor_loadings * factor_shocks;

        std::vector<RiskFactorShock> result;
        result.reserve(factors.size());

        for (size_t i = 0; i < factors.size() && i < asset_shocks.size(); ++i) {
            RiskFactorShock shock;
            shock.factor_id = factors[i].id;
            shock.factor_name = factors[i].name;
            shock.shock_magnitude = asset_shocks(i);
            shock.shock_type = ShockType::RELATIVE;
            result.push_back(shock);
        }

        return result;
    }

    std::vector<RiskFactorShock> construct_level_shift_shocks(
        const std::vector<RiskFactor>& factors,
        const LevelShiftParameters& params) {

        std::vector<RiskFactorShock> shocks;
        shocks.reserve(factors.size());

        for (const auto& factor : factors) {
            RiskFactorShock shock;
            shock.factor_id = factor.id;
            shock.factor_name = factor.name;
            shock.shock_type = ShockType::ABSOLUTE;

            // Apply category-specific level shifts
            switch (factor.category) {
                case RiskFactorCategory::INTEREST_RATE:
                    shock.shock_magnitude = params.rate_shift_bp / 10000.0; // Convert basis points
                    break;
                case RiskFactorCategory::EQUITY:
                    shock.shock_magnitude = params.equity_shift_percent / 100.0;
                    break;
                case RiskFactorCategory::FX:
                    shock.shock_magnitude = params.fx_shift_percent / 100.0;
                    break;
                case RiskFactorCategory::COMMODITY:
                    shock.shock_magnitude = params.commodity_shift_percent / 100.0;
                    break;
                case RiskFactorCategory::CREDIT:
                    shock.shock_magnitude = params.credit_spread_shift_bp / 10000.0;
                    break;
                case RiskFactorCategory::VOLATILITY:
                    shock.shock_magnitude = params.volatility_shift_percent / 100.0;
                    break;
            }

            // Apply directional bias if specified
            if (params.apply_directional_bias) {
                shock.shock_magnitude *= get_directional_bias(factor.category, params.market_direction);
            }

            shocks.push_back(shock);
        }

        return shocks;
    }

    std::vector<RiskFactorShock> construct_volatility_scaled_shocks(
        const std::vector<RiskFactor>& factors,
        const VolatilityScalingParameters& params) {

        std::vector<RiskFactorShock> shocks;
        shocks.reserve(factors.size());

        for (const auto& factor : factors) {
            RiskFactorShock shock;
            shock.factor_id = factor.id;
            shock.factor_name = factor.name;
            shock.shock_type = ShockType::RELATIVE;

            // Get factor-specific volatility
            double factor_volatility = get_factor_volatility(factor, params);

            // Generate random shock scaled by volatility
            double random_shock = normal_dist_(rng_);
            shock.shock_magnitude = random_shock * factor_volatility * params.volatility_multiplier;

            // Apply volatility regime adjustments
            if (params.volatility_regime == VolatilityRegime::HIGH) {
                shock.shock_magnitude *= 2.0;
            } else if (params.volatility_regime == VolatilityRegime::LOW) {
                shock.shock_magnitude *= 0.5;
            }

            shocks.push_back(shock);
        }

        return shocks;
    }

    std::vector<RiskFactorShock> construct_curve_steepening_shocks(
        const std::vector<RiskFactor>& factors,
        const CurveSteepnessParameters& params) {

        std::vector<RiskFactorShock> shocks;

        // Filter for interest rate factors
        std::vector<RiskFactor> rate_factors;
        std::copy_if(factors.begin(), factors.end(), std::back_inserter(rate_factors),
            [](const RiskFactor& factor) {
                return factor.category == RiskFactorCategory::INTEREST_RATE;
            });

        if (rate_factors.empty()) {
            return shocks;
        }

        // Sort by tenor (assuming factor names contain tenor information)
        std::sort(rate_factors.begin(), rate_factors.end(),
            [](const RiskFactor& a, const RiskFactor& b) {
                return extract_tenor_years(a.name) < extract_tenor_years(b.name);
            });

        for (size_t i = 0; i < rate_factors.size(); ++i) {
            RiskFactorShock shock;
            shock.factor_id = rate_factors[i].id;
            shock.factor_name = rate_factors[i].name;
            shock.shock_type = ShockType::ABSOLUTE;

            double tenor_years = extract_tenor_years(rate_factors[i].name);

            // Apply steepening/flattening based on tenor
            if (params.steepening_type == CurveSteepnessType::BEAR_STEEPENING) {
                // Short rates up less, long rates up more
                shock.shock_magnitude = params.base_shift_bp * (1.0 + tenor_years * params.steepness_factor);
            } else if (params.steepening_type == CurveSteepnessType::BULL_STEEPENING) {
                // Short rates down more, long rates down less
                shock.shock_magnitude = -params.base_shift_bp * (2.0 - tenor_years * params.steepness_factor);
            } else if (params.steepening_type == CurveSteepnessType::BEAR_FLATTENING) {
                // Short rates up more, long rates up less
                shock.shock_magnitude = params.base_shift_bp * (2.0 - tenor_years * params.steepness_factor);
            } else { // BULL_FLATTENING
                // Short rates down less, long rates down more
                shock.shock_magnitude = -params.base_shift_bp * (1.0 + tenor_years * params.steepness_factor);
            }

            shock.shock_magnitude /= 10000.0; // Convert basis points
            shocks.push_back(shock);
        }

        return shocks;
    }

    std::vector<RiskFactorShock> construct_cross_asset_correlation_shocks(
        const std::vector<RiskFactor>& factors,
        const CrossAssetCorrelationParameters& params) {

        // Group factors by asset class
        std::unordered_map<RiskFactorCategory, std::vector<size_t>> factor_groups;
        for (size_t i = 0; i < factors.size(); ++i) {
            factor_groups[factors[i].category].push_back(i);
        }

        std::vector<RiskFactorShock> shocks(factors.size());

        // Generate base shocks for each asset class
        for (const auto& [category, indices] : factor_groups) {
            std::vector<double> base_shocks(indices.size());
            std::generate(base_shocks.begin(), base_shocks.end(),
                [this]() { return normal_dist_(rng_); });

            // Apply intra-asset correlation
            double intra_correlation = get_intra_asset_correlation(category, params);
            apply_correlation_structure(base_shocks, intra_correlation);

            // Assign shocks to factors
            for (size_t i = 0; i < indices.size(); ++i) {
                size_t factor_idx = indices[i];
                shocks[factor_idx].factor_id = factors[factor_idx].id;
                shocks[factor_idx].factor_name = factors[factor_idx].name;
                shocks[factor_idx].shock_magnitude = base_shocks[i] * params.volatility_scaling;
                shocks[factor_idx].shock_type = ShockType::RELATIVE;
            }
        }

        // Apply cross-asset correlations
        apply_cross_asset_correlations(shocks, factor_groups, params);

        return shocks;
    }

private:
    Eigen::MatrixXd center_matrix(const Eigen::MatrixXd& matrix) {
        Eigen::VectorXd means = matrix.colwise().mean();
        return matrix.rowwise() - means.transpose();
    }

    double get_directional_bias(RiskFactorCategory category, MarketDirection direction) {
        if (direction == MarketDirection::NEUTRAL) {
            return (uniform_dist_(rng_) < 0.5) ? 1.0 : -1.0;
        }

        bool positive_bias = (direction == MarketDirection::UP);

        // Invert for categories that typically move inverse to market
        if (category == RiskFactorCategory::VOLATILITY ||
            category == RiskFactorCategory::CREDIT) {
            positive_bias = !positive_bias;
        }

        return positive_bias ? 1.0 : -1.0;
    }

    double get_factor_volatility(const RiskFactor& factor,
                                const VolatilityScalingParameters& params) {
        // Default volatilities by category (annualized)
        static const std::unordered_map<RiskFactorCategory, double> default_volatilities = {
            {RiskFactorCategory::EQUITY, 0.20},
            {RiskFactorCategory::INTEREST_RATE, 0.015},
            {RiskFactorCategory::FX, 0.12},
            {RiskFactorCategory::COMMODITY, 0.25},
            {RiskFactorCategory::CREDIT, 0.08},
            {RiskFactorCategory::VOLATILITY, 0.50}
        };

        auto it = default_volatilities.find(factor.category);
        if (it != default_volatilities.end()) {
            return it->second;
        }

        return 0.15; // Default volatility
    }

    double extract_tenor_years(const std::string& factor_name) {
        // Simple tenor extraction (assumes naming like "USD_1Y", "EUR_10Y", etc.)
        std::regex tenor_regex(R"((\d+)([YMD]))");
        std::smatch match;

        if (std::regex_search(factor_name, match, tenor_regex)) {
            int value = std::stoi(match[1].str());
            char unit = match[2].str()[0];

            switch (unit) {
                case 'Y': return static_cast<double>(value);
                case 'M': return static_cast<double>(value) / 12.0;
                case 'D': return static_cast<double>(value) / 365.0;
            }
        }

        return 1.0; // Default to 1 year
    }

    double get_intra_asset_correlation(RiskFactorCategory category,
                                      const CrossAssetCorrelationParameters& params) {
        // Default intra-asset correlations
        switch (category) {
            case RiskFactorCategory::EQUITY: return 0.7;
            case RiskFactorCategory::INTEREST_RATE: return 0.9;
            case RiskFactorCategory::FX: return 0.3;
            case RiskFactorCategory::COMMODITY: return 0.4;
            case RiskFactorCategory::CREDIT: return 0.6;
            case RiskFactorCategory::VOLATILITY: return 0.5;
        }
        return 0.5;
    }

    void apply_correlation_structure(std::vector<double>& shocks, double correlation) {
        if (shocks.size() <= 1 || correlation <= 0.0) {
            return;
        }

        // Simple correlation structure using first shock as common factor
        double common_factor = shocks[0];

        for (size_t i = 1; i < shocks.size(); ++i) {
            shocks[i] = correlation * common_factor +
                       std::sqrt(1.0 - correlation * correlation) * shocks[i];
        }
    }

    void apply_cross_asset_correlations(
        std::vector<RiskFactorShock>& shocks,
        const std::unordered_map<RiskFactorCategory, std::vector<size_t>>& factor_groups,
        const CrossAssetCorrelationParameters& params) {

        // Apply cross-asset correlations (simplified implementation)
        // In practice, this would use a full correlation matrix

        if (factor_groups.find(RiskFactorCategory::EQUITY) != factor_groups.end() &&
            factor_groups.find(RiskFactorCategory::VOLATILITY) != factor_groups.end()) {

            // Apply negative correlation between equity and volatility
            auto& equity_indices = factor_groups.at(RiskFactorCategory::EQUITY);
            auto& vol_indices = factor_groups.at(RiskFactorCategory::VOLATILITY);

            for (size_t eq_idx : equity_indices) {
                for (size_t vol_idx : vol_indices) {
                    // Apply negative correlation
                    shocks[vol_idx].shock_magnitude = -0.7 * shocks[eq_idx].shock_magnitude +
                                                     0.3 * shocks[vol_idx].shock_magnitude;
                }
            }
        }
    }
};

// ShockConstructor implementation
ShockConstructor::ShockConstructor() :
    impl_(std::make_unique<ShockConstructorImpl>()) {}

ShockConstructor::~ShockConstructor() = default;

void ShockConstructor::calibrate_factor_model(const std::string& model_name,
                                             const std::vector<RiskFactor>& factors,
                                             const Eigen::MatrixXd& historical_returns) {
    impl_->calibrate_factor_model(model_name, factors, historical_returns);
}

std::vector<RiskFactorShock> ShockConstructor::construct_factor_based_shocks(
    const std::string& model_name,
    const std::vector<RiskFactor>& factors,
    const Eigen::VectorXd& factor_shocks) {
    return impl_->construct_factor_based_shocks(model_name, factors, factor_shocks);
}

std::vector<RiskFactorShock> ShockConstructor::construct_level_shift_shocks(
    const std::vector<RiskFactor>& factors,
    const LevelShiftParameters& params) {
    return impl_->construct_level_shift_shocks(factors, params);
}

std::vector<RiskFactorShock> ShockConstructor::construct_volatility_scaled_shocks(
    const std::vector<RiskFactor>& factors,
    const VolatilityScalingParameters& params) {
    return impl_->construct_volatility_scaled_shocks(factors, params);
}

std::vector<RiskFactorShock> ShockConstructor::construct_curve_steepening_shocks(
    const std::vector<RiskFactor>& factors,
    const CurveSteepnessParameters& params) {
    return impl_->construct_curve_steepening_shocks(factors, params);
}

std::vector<RiskFactorShock> ShockConstructor::construct_cross_asset_correlation_shocks(
    const std::vector<RiskFactor>& factors,
    const CrossAssetCorrelationParameters& params) {
    return impl_->construct_cross_asset_correlation_shocks(factors, params);
}

} // namespace stress_testing