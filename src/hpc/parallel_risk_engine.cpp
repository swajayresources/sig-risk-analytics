/**
 * Multi-threaded Risk Calculation Engine
 *
 * High-performance parallel implementation of risk metrics calculations
 * using OpenMP, SIMD vectorization, and optimized algorithms
 */

#include "hpc/hpc_framework.hpp"
#include "core/types.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>
#include <immintrin.h>

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace risk_analytics {
namespace hpc {

/**
 * Parallel Risk Calculation Engine
 */
class ParallelRiskEngine {
private:
    HPCFramework& framework_;
    PerformanceProfiler& profiler_;

    // Pre-allocated working memory for calculations
    mutable std::vector<AlignedVector<Real, 64>> thread_local_storage_;

    // NUMA-aware memory allocation
    void* allocate_numa_memory(size_t size, int node = -1);
    void free_numa_memory(void* ptr, size_t size);

    // Parallel algorithms
    void parallel_portfolio_returns(const std::vector<Position>& positions,
                                   const RealMatrix& historical_returns,
                                   RealVector& portfolio_returns) const;

    Real parallel_covariance_element(const RealVector& returns1,
                                   const RealVector& returns2) const;

public:
    ParallelRiskEngine(HPCFramework& framework)
        : framework_(framework), profiler_(framework.get_profiler()) {

        // Initialize thread-local storage
        const size_t num_threads = framework_.get_config().num_calculation_threads;
        thread_local_storage_.resize(num_threads);

        for (size_t i = 0; i < num_threads; ++i) {
            thread_local_storage_[i] = AlignedVector<Real, 64>(65536); // 64KB per thread
        }
    }

    /**
     * Parallel Historical Simulation VaR
     */
    struct VaRResult {
        Real var;
        Real expected_shortfall;
        Real max_loss;
        Real mean_return;
        Real volatility;
        std::vector<Real> percentiles;
    };

    VaRResult calculate_historical_var_parallel(
        const Portfolio& portfolio,
        const RealMatrix& historical_returns,
        Real confidence_level = 0.95,
        int lookback_days = 252) const {

        PROFILE_SCOPE(profiler_, "HistoricalVaR_Parallel");

        const auto positions = portfolio.get_all_positions();
        const size_t num_assets = positions.size();
        const size_t num_observations = std::min(static_cast<size_t>(lookback_days),
                                                historical_returns.size());

        // Calculate portfolio weights
        AlignedVector<Real, 64> weights(num_assets);
        const Real total_value = portfolio.get_total_value();

        #pragma omp parallel for
        for (size_t i = 0; i < num_assets; ++i) {
            weights[i] = positions[i].market_value / total_value;
        }

        // Calculate portfolio returns in parallel
        AlignedVector<Real, 64> portfolio_returns(num_observations);

        #pragma omp parallel
        {
            const int thread_id = omp_get_thread_num();
            const int num_threads = omp_get_num_threads();
            const size_t chunk_size = num_observations / num_threads;
            const size_t start_idx = thread_id * chunk_size;
            const size_t end_idx = (thread_id == num_threads - 1) ?
                                  num_observations : start_idx + chunk_size;

            for (size_t t = start_idx; t < end_idx; ++t) {
                Real portfolio_return = 0.0;

                // SIMD-optimized dot product
                if (num_assets >= 4 && framework_.get_config().enable_simd) {
                    portfolio_return = simd::dot_product_avx2(
                        weights.get(),
                        historical_returns[t].data(),
                        num_assets
                    );
                } else {
                    // Fallback scalar implementation
                    for (size_t i = 0; i < num_assets; ++i) {
                        portfolio_return += weights[i] * historical_returns[t][i];
                    }
                }

                portfolio_returns[t] = portfolio_return;
            }
        }

        // Sort returns for percentile calculations (parallel quicksort)
        AlignedVector<Real, 64> sorted_returns = portfolio_returns;

        #ifdef USE_OPENMP
        // Use parallel sort if available
        __gnu_parallel::sort(sorted_returns.data(),
                           sorted_returns.data() + sorted_returns.size());
        #else
        std::sort(sorted_returns.data(),
                 sorted_returns.data() + sorted_returns.size());
        #endif

        // Calculate VaR and Expected Shortfall
        const size_t var_index = static_cast<size_t>((1.0 - confidence_level) * num_observations);
        const Real var = -sorted_returns[var_index];

        // Expected Shortfall calculation (parallel reduction)
        Real expected_shortfall = 0.0;
        size_t tail_count = 0;

        #pragma omp parallel for reduction(+:expected_shortfall, tail_count)
        for (size_t i = 0; i <= var_index; ++i) {
            if (sorted_returns[i] <= -var) {
                expected_shortfall += sorted_returns[i];
                tail_count++;
            }
        }

        expected_shortfall = tail_count > 0 ? -expected_shortfall / tail_count : var;

        // Calculate additional statistics in parallel
        Real mean_return = 0.0;
        Real variance = 0.0;

        #pragma omp parallel for reduction(+:mean_return)
        for (size_t i = 0; i < num_observations; ++i) {
            mean_return += portfolio_returns[i];
        }
        mean_return /= num_observations;

        #pragma omp parallel for reduction(+:variance)
        for (size_t i = 0; i < num_observations; ++i) {
            const Real diff = portfolio_returns[i] - mean_return;
            variance += diff * diff;
        }
        variance /= (num_observations - 1);

        // Calculate percentiles
        std::vector<Real> percentile_levels = {0.01, 0.05, 0.10, 0.90, 0.95, 0.99};
        std::vector<Real> percentiles(percentile_levels.size());

        #pragma omp parallel for
        for (size_t i = 0; i < percentile_levels.size(); ++i) {
            const size_t idx = static_cast<size_t>(percentile_levels[i] * (num_observations - 1));
            percentiles[i] = sorted_returns[idx];
        }

        return VaRResult{
            .var = var,
            .expected_shortfall = expected_shortfall,
            .max_loss = -sorted_returns[0],
            .mean_return = mean_return,
            .volatility = std::sqrt(variance),
            .percentiles = std::move(percentiles)
        };
    }

    /**
     * Parallel Parametric VaR with covariance matrix estimation
     */
    VaRResult calculate_parametric_var_parallel(
        const Portfolio& portfolio,
        const RealMatrix& historical_returns,
        Real confidence_level = 0.95,
        bool use_ledoit_wolf = true) const {

        PROFILE_SCOPE(profiler_, "ParametricVaR_Parallel");

        const auto positions = portfolio.get_all_positions();
        const size_t num_assets = positions.size();
        const size_t num_observations = historical_returns.size();

        // Calculate portfolio weights
        AlignedVector<Real, 64> weights(num_assets);
        const Real total_value = portfolio.get_total_value();

        #pragma omp parallel for
        for (size_t i = 0; i < num_assets; ++i) {
            weights[i] = positions[i].market_value / total_value;
        }

        // Calculate mean returns in parallel
        AlignedVector<Real, 64> mean_returns(num_assets);

        #pragma omp parallel for
        for (size_t i = 0; i < num_assets; ++i) {
            Real sum = 0.0;

            #pragma omp simd reduction(+:sum)
            for (size_t t = 0; t < num_observations; ++t) {
                sum += historical_returns[t][i];
            }

            mean_returns[i] = sum / num_observations;
        }

        // Calculate covariance matrix in parallel
        RealMatrix covariance_matrix(num_assets, RealVector(num_assets));

        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < num_assets; ++i) {
            for (size_t j = i; j < num_assets; ++j) {
                Real covariance = 0.0;

                for (size_t t = 0; t < num_observations; ++t) {
                    const Real dev_i = historical_returns[t][i] - mean_returns[i];
                    const Real dev_j = historical_returns[t][j] - mean_returns[j];
                    covariance += dev_i * dev_j;
                }

                covariance /= (num_observations - 1);
                covariance_matrix[i][j] = covariance;
                covariance_matrix[j][i] = covariance; // Symmetric matrix
            }
        }

        // Apply Ledoit-Wolf shrinkage if requested
        if (use_ledoit_wolf) {
            apply_ledoit_wolf_shrinkage(covariance_matrix, historical_returns);
        }

        // Calculate portfolio variance using SIMD
        Real portfolio_variance = 0.0;

        #pragma omp parallel for reduction(+:portfolio_variance)
        for (size_t i = 0; i < num_assets; ++i) {
            Real row_contribution = 0.0;

            // SIMD dot product for covariance matrix row
            if (num_assets >= 4 && framework_.get_config().enable_simd) {
                row_contribution = simd::dot_product_avx2(
                    covariance_matrix[i].data(),
                    weights.get(),
                    num_assets
                );
            } else {
                for (size_t j = 0; j < num_assets; ++j) {
                    row_contribution += covariance_matrix[i][j] * weights[j];
                }
            }

            portfolio_variance += weights[i] * row_contribution;
        }

        const Real portfolio_volatility = std::sqrt(portfolio_variance);

        // Calculate portfolio mean return
        Real portfolio_mean = 0.0;
        if (framework_.get_config().enable_simd && num_assets >= 4) {
            portfolio_mean = simd::dot_product_avx2(
                weights.get(), mean_returns.get(), num_assets
            );
        } else {
            #pragma omp parallel for reduction(+:portfolio_mean)
            for (size_t i = 0; i < num_assets; ++i) {
                portfolio_mean += weights[i] * mean_returns[i];
            }
        }

        // Calculate VaR using normal distribution assumption
        const Real normal_quantile = calculate_normal_quantile(1.0 - confidence_level);
        const Real var = -(portfolio_mean + normal_quantile * portfolio_volatility);

        // Expected Shortfall for normal distribution
        const Real expected_shortfall = var + portfolio_volatility *
            std::exp(-0.5 * normal_quantile * normal_quantile) /
            (std::sqrt(2.0 * M_PI) * (1.0 - confidence_level));

        return VaRResult{
            .var = var,
            .expected_shortfall = expected_shortfall,
            .max_loss = var * 3.0, // 3-sigma approximation
            .mean_return = portfolio_mean,
            .volatility = portfolio_volatility,
            .percentiles = {} // Not calculated for parametric method
        };
    }

    /**
     * Parallel Component VaR calculation
     */
    std::vector<Real> calculate_component_var_parallel(
        const Portfolio& portfolio,
        const RealMatrix& covariance_matrix,
        Real confidence_level = 0.95) const {

        PROFILE_SCOPE(profiler_, "ComponentVaR_Parallel");

        const auto positions = portfolio.get_all_positions();
        const size_t num_assets = positions.size();

        // Calculate portfolio weights
        AlignedVector<Real, 64> weights(num_assets);
        const Real total_value = portfolio.get_total_value();

        #pragma omp parallel for
        for (size_t i = 0; i < num_assets; ++i) {
            weights[i] = positions[i].market_value / total_value;
        }

        // Calculate marginal contributions (Cov * w)
        AlignedVector<Real, 64> marginal_contributions(num_assets);

        #pragma omp parallel for
        for (size_t i = 0; i < num_assets; ++i) {
            Real contribution = 0.0;

            if (framework_.get_config().enable_simd && num_assets >= 4) {
                contribution = simd::dot_product_avx2(
                    covariance_matrix[i].data(),
                    weights.get(),
                    num_assets
                );
            } else {
                for (size_t j = 0; j < num_assets; ++j) {
                    contribution += covariance_matrix[i][j] * weights[j];
                }
            }

            marginal_contributions[i] = contribution;
        }

        // Calculate portfolio variance
        Real portfolio_variance = 0.0;
        if (framework_.get_config().enable_simd && num_assets >= 4) {
            portfolio_variance = simd::dot_product_avx2(
                weights.get(), marginal_contributions.get(), num_assets
            );
        } else {
            #pragma omp parallel for reduction(+:portfolio_variance)
            for (size_t i = 0; i < num_assets; ++i) {
                portfolio_variance += weights[i] * marginal_contributions[i];
            }
        }

        const Real portfolio_volatility = std::sqrt(portfolio_variance);
        const Real normal_quantile = calculate_normal_quantile(1.0 - confidence_level);
        const Real portfolio_var = normal_quantile * portfolio_volatility;

        // Calculate component VaR for each asset
        std::vector<Real> component_var(num_assets);

        #pragma omp parallel for
        for (size_t i = 0; i < num_assets; ++i) {
            component_var[i] = (weights[i] * marginal_contributions[i] / portfolio_variance) * portfolio_var;
        }

        return component_var;
    }

    /**
     * Parallel Greeks calculation for options portfolio
     */
    struct GreeksResult {
        Real delta;
        Real gamma;
        Real theta;
        Real vega;
        Real rho;
        Real portfolio_delta;
        Real portfolio_gamma;
        Real portfolio_theta;
        Real portfolio_vega;
        Real portfolio_rho;
    };

    std::vector<GreeksResult> calculate_portfolio_greeks_parallel(
        const std::vector<OptionContract>& options) const {

        PROFILE_SCOPE(profiler_, "PortfolioGreeks_Parallel");

        const size_t num_options = options.size();
        std::vector<GreeksResult> results(num_options);

        // Parallel Greeks calculation using Black-Scholes
        #pragma omp parallel for
        for (size_t i = 0; i < num_options; ++i) {
            const auto& option = options[i];

            const Real S = option.underlying_price;
            const Real K = option.strike_price;
            const Real T = option.time_to_expiry;
            const Real r = option.risk_free_rate;
            const Real sigma = option.volatility;
            const Real q = option.dividend_yield;

            // Calculate d1 and d2
            const Real sqrt_T = std::sqrt(T);
            const Real d1 = (std::log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T);
            const Real d2 = d1 - sigma * sqrt_T;

            // Calculate Greeks using vectorized normal distribution functions
            alignas(32) Real input_array[4] = {d1, d2, -d1, -d2};
            alignas(32) Real cdf_results[4];
            alignas(32) Real pdf_results[4];

            if (framework_.get_config().enable_simd) {
                simd::normal_cdf_array_avx2(input_array, cdf_results, 4);

                // Calculate PDF values
                for (int j = 0; j < 4; ++j) {
                    pdf_results[j] = std::exp(-0.5 * input_array[j] * input_array[j]) / std::sqrt(2.0 * M_PI);
                }
            } else {
                for (int j = 0; j < 4; ++j) {
                    cdf_results[j] = std::erfc(-input_array[j] / std::sqrt(2.0)) / 2.0;
                    pdf_results[j] = std::exp(-0.5 * input_array[j] * input_array[j]) / std::sqrt(2.0 * M_PI);
                }
            }

            const Real N_d1 = cdf_results[0];
            const Real N_d2 = cdf_results[1];
            const Real n_d1 = pdf_results[0];

            const Real discount_factor = std::exp(-r * T);
            const Real dividend_factor = std::exp(-q * T);

            // Calculate Greeks
            Real delta, gamma, theta, vega, rho;

            if (option.is_call) {
                delta = dividend_factor * N_d1;
                gamma = dividend_factor * n_d1 / (S * sigma * sqrt_T);
                theta = (-dividend_factor * S * n_d1 * sigma / (2.0 * sqrt_T)
                        - r * K * discount_factor * N_d2
                        + q * S * dividend_factor * N_d1) / 365.0;
                vega = S * dividend_factor * n_d1 * sqrt_T / 100.0;
                rho = K * T * discount_factor * N_d2 / 100.0;
            } else {
                delta = -dividend_factor * cdf_results[2]; // N(-d1)
                gamma = dividend_factor * n_d1 / (S * sigma * sqrt_T);
                theta = (-dividend_factor * S * n_d1 * sigma / (2.0 * sqrt_T)
                        + r * K * discount_factor * cdf_results[3]
                        - q * S * dividend_factor * cdf_results[2]) / 365.0;
                vega = S * dividend_factor * n_d1 * sqrt_T / 100.0;
                rho = -K * T * discount_factor * cdf_results[3] / 100.0;
            }

            results[i] = GreeksResult{
                .delta = delta * option.quantity,
                .gamma = gamma * option.quantity,
                .theta = theta * option.quantity,
                .vega = vega * option.quantity,
                .rho = rho * option.quantity
            };
        }

        // Calculate portfolio-level Greeks using parallel reduction
        GreeksResult portfolio_greeks{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

        #pragma omp parallel for reduction(+:portfolio_greeks.portfolio_delta, \
                                           portfolio_greeks.portfolio_gamma, \
                                           portfolio_greeks.portfolio_theta, \
                                           portfolio_greeks.portfolio_vega, \
                                           portfolio_greeks.portfolio_rho)
        for (size_t i = 0; i < num_options; ++i) {
            portfolio_greeks.portfolio_delta += results[i].delta;
            portfolio_greeks.portfolio_gamma += results[i].gamma;
            portfolio_greeks.portfolio_theta += results[i].theta;
            portfolio_greeks.portfolio_vega += results[i].vega;
            portfolio_greeks.portfolio_rho += results[i].rho;
        }

        // Add portfolio totals to first result
        if (!results.empty()) {
            results[0].portfolio_delta = portfolio_greeks.portfolio_delta;
            results[0].portfolio_gamma = portfolio_greeks.portfolio_gamma;
            results[0].portfolio_theta = portfolio_greeks.portfolio_theta;
            results[0].portfolio_vega = portfolio_greeks.portfolio_vega;
            results[0].portfolio_rho = portfolio_greeks.portfolio_rho;
        }

        return results;
    }

private:
    /**
     * Apply Ledoit-Wolf shrinkage to covariance matrix
     */
    void apply_ledoit_wolf_shrinkage(RealMatrix& covariance_matrix,
                                   const RealMatrix& historical_returns) const {
        const size_t num_assets = covariance_matrix.size();
        const size_t num_observations = historical_returns.size();

        // Calculate sample covariance matrix trace
        Real trace = 0.0;
        #pragma omp parallel for reduction(+:trace)
        for (size_t i = 0; i < num_assets; ++i) {
            trace += covariance_matrix[i][i];
        }

        const Real avg_variance = trace / num_assets;

        // Calculate shrinkage intensity (simplified version)
        Real shrinkage_intensity = 0.0;

        #pragma omp parallel for reduction(+:shrinkage_intensity)
        for (size_t i = 0; i < num_assets; ++i) {
            for (size_t j = 0; j < num_assets; ++j) {
                const Real target_value = (i == j) ? avg_variance : 0.0;
                const Real diff = covariance_matrix[i][j] - target_value;
                shrinkage_intensity += diff * diff;
            }
        }

        const Real lambda = std::min(1.0, shrinkage_intensity / (num_observations * trace * trace));

        // Apply shrinkage
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < num_assets; ++i) {
            for (size_t j = 0; j < num_assets; ++j) {
                const Real target_value = (i == j) ? avg_variance : 0.0;
                covariance_matrix[i][j] = lambda * target_value +
                                        (1.0 - lambda) * covariance_matrix[i][j];
            }
        }
    }

    /**
     * Fast normal quantile approximation
     */
    Real calculate_normal_quantile(Real p) const {
        // Beasley-Springer-Moro algorithm for normal quantile
        static constexpr Real a0 = 2.515517;
        static constexpr Real a1 = 0.802853;
        static constexpr Real a2 = 0.010328;
        static constexpr Real b1 = 1.432788;
        static constexpr Real b2 = 0.189269;
        static constexpr Real b3 = 0.001308;

        if (p > 0.5) {
            p = 1.0 - p;
        }

        const Real t = std::sqrt(-2.0 * std::log(p));
        const Real numerator = a0 + a1 * t + a2 * t * t;
        const Real denominator = 1.0 + b1 * t + b2 * t * t + b3 * t * t * t;

        Real result = t - numerator / denominator;

        return (p > 0.5) ? result : -result;
    }
};

} // namespace hpc
} // namespace risk_analytics