/**
 * GPU-Accelerated Monte Carlo Simulation Framework
 *
 * High-performance CUDA implementation for Monte Carlo risk calculations
 * with variance reduction techniques and optimized memory access patterns
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cublas_v2.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>
#include <cub/cub.cuh>

namespace cg = cooperative_groups;

namespace risk_analytics {
namespace hpc {
namespace gpu {

// CUDA error checking macros
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error at %s:%d - %d\n", __FILE__, __LINE__, status); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// GPU configuration constants
constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;
constexpr int MAX_ASSETS = 1024;
constexpr int MAX_PATHS_PER_BLOCK = 1024;

/**
 * Device function for fast normal distribution generation using Box-Muller
 */
__device__ __forceinline__ void box_muller_transform(
    curandState* state, double& z1, double& z2) {

    double u1, u2;
    do {
        u1 = curand_uniform_double(state);
    } while (u1 <= 1e-7); // Avoid log(0)

    u2 = curand_uniform_double(state);

    const double log_u1 = log(u1);
    const double two_pi_u2 = 2.0 * M_PI * u2;

    const double magnitude = sqrt(-2.0 * log_u1);
    z1 = magnitude * cos(two_pi_u2);
    z2 = magnitude * sin(two_pi_u2);
}

/**
 * Device function for correlated normal generation using Cholesky decomposition
 */
__device__ void generate_correlated_normals(
    curandState* state,
    const double* __restrict__ cholesky_matrix,
    double* __restrict__ correlated_normals,
    int num_assets) {

    // Generate independent normals
    double* independent_normals = (double*)extern_shared_memory();

    for (int i = 0; i < num_assets; i += 2) {
        double z1, z2;
        box_muller_transform(state, z1, z2);
        independent_normals[i] = z1;
        if (i + 1 < num_assets) {
            independent_normals[i + 1] = z2;
        }
    }

    // Apply Cholesky transformation: y = L * x
    for (int i = 0; i < num_assets; ++i) {
        double sum = 0.0;
        for (int j = 0; j <= i; ++j) {
            sum += cholesky_matrix[i * num_assets + j] * independent_normals[j];
        }
        correlated_normals[i] = sum;
    }
}

/**
 * CUDA kernel for Geometric Brownian Motion path generation
 */
__global__ void geometric_brownian_motion_kernel(
    double* __restrict__ price_paths,
    const double* __restrict__ initial_prices,
    const double* __restrict__ drift,
    const double* __restrict__ volatility,
    const double* __restrict__ cholesky_matrix,
    const double dt,
    const int num_paths,
    const int num_steps,
    const int num_assets,
    const unsigned long long seed) {

    const int path_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    // Initialize random state
    curandState local_state;
    curand_init(seed, path_idx, 0, &local_state);

    // Shared memory for correlated normals
    extern __shared__ double shared_memory[];

    for (int path = path_idx; path < num_paths; path += stride) {
        // Initialize prices for this path
        for (int asset = 0; asset < num_assets; ++asset) {
            const int price_idx = path * (num_steps + 1) * num_assets + asset;
            price_paths[price_idx] = initial_prices[asset];
        }

        // Generate path for each time step
        for (int step = 0; step < num_steps; ++step) {
            // Generate correlated random numbers
            double* correlated_normals = &shared_memory[threadIdx.x * num_assets];
            generate_correlated_normals(&local_state, cholesky_matrix,
                                      correlated_normals, num_assets);

            // Update prices using GBM formula
            for (int asset = 0; asset < num_assets; ++asset) {
                const int prev_idx = path * (num_steps + 1) * num_assets + step * num_assets + asset;
                const int curr_idx = path * (num_steps + 1) * num_assets + (step + 1) * num_assets + asset;

                const double prev_price = price_paths[prev_idx];
                const double mu = drift[asset];
                const double sigma = volatility[asset];
                const double dW = correlated_normals[asset] * sqrt(dt);

                // S(t+dt) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*dW)
                const double exponent = (mu - 0.5 * sigma * sigma) * dt + sigma * dW;
                price_paths[curr_idx] = prev_price * exp(exponent);
            }
        }
    }
}

/**
 * CUDA kernel for portfolio Monte Carlo simulation with variance reduction
 */
__global__ void portfolio_monte_carlo_kernel(
    double* __restrict__ portfolio_values,
    double* __restrict__ control_variates,
    const double* __restrict__ price_paths,
    const double* __restrict__ weights,
    const double* __restrict__ initial_prices,
    const int num_paths,
    const int num_steps,
    const int num_assets,
    const bool use_antithetic,
    const bool use_control_variates) {

    const int path_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    for (int path = path_idx; path < num_paths; path += stride) {
        double portfolio_value = 0.0;
        double control_value = 0.0;

        // Calculate final portfolio value
        for (int asset = 0; asset < num_assets; ++asset) {
            const int final_price_idx = path * (num_steps + 1) * num_assets + num_steps * num_assets + asset;
            const double final_price = price_paths[final_price_idx];
            portfolio_value += weights[asset] * final_price;

            // Control variate: geometric average price
            if (use_control_variates) {
                control_value += weights[asset] * initial_prices[asset] *
                    exp(0.5 * log(final_price / initial_prices[asset]));
            }
        }

        portfolio_values[path] = portfolio_value;

        if (use_control_variates) {
            control_variates[path] = control_value;
        }

        // Antithetic variates: use negative of random numbers for next path
        if (use_antithetic && path + num_paths < 2 * num_paths) {
            // This would be implemented with additional logic for antithetic paths
        }
    }
}

/**
 * CUDA kernel for parallel VaR calculation using reduction
 */
__global__ void calculate_var_kernel(
    const double* __restrict__ portfolio_returns,
    double* __restrict__ sorted_returns,
    double* __restrict__ var_result,
    const int num_simulations,
    const double confidence_level) {

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Use CUB for efficient parallel sorting
    typedef cub::BlockRadixSort<double, BLOCK_SIZE> BlockRadixSort;
    __shared__ typename BlockRadixSort::TempStorage temp_storage;

    // Load data into shared memory
    __shared__ double shared_data[BLOCK_SIZE];

    if (tid < num_simulations) {
        shared_data[threadIdx.x] = portfolio_returns[tid];
    } else {
        shared_data[threadIdx.x] = INFINITY; // Padding for sorting
    }

    __syncthreads();

    // Sort block data
    BlockRadixSort(temp_storage).Sort(shared_data[threadIdx.x]);

    __syncthreads();

    // Store sorted data back to global memory
    if (tid < num_simulations) {
        sorted_returns[tid] = shared_data[threadIdx.x];
    }

    __syncthreads();

    // Calculate VaR (only one thread per block)
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        const int var_index = static_cast<int>((1.0 - confidence_level) * num_simulations);
        *var_result = -sorted_returns[var_index];
    }
}

/**
 * CUDA kernel for Heston stochastic volatility model
 */
__global__ void heston_model_kernel(
    double* __restrict__ price_paths,
    double* __restrict__ volatility_paths,
    const double* __restrict__ initial_prices,
    const double* __restrict__ initial_volatility,
    const double* __restrict__ mean_reversion_rate,
    const double* __restrict__ long_term_variance,
    const double* __restrict__ vol_of_vol,
    const double* __restrict__ correlation,
    const double risk_free_rate,
    const double dt,
    const int num_paths,
    const int num_steps,
    const int num_assets,
    const unsigned long long seed) {

    const int path_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    // Initialize random state
    curandState local_state;
    curand_init(seed, path_idx, 0, &local_state);

    for (int path = path_idx; path < num_paths; path += stride) {
        // Initialize for each asset
        for (int asset = 0; asset < num_assets; ++asset) {
            const int base_idx = path * (num_steps + 1) * num_assets;
            price_paths[base_idx + asset] = initial_prices[asset];
            volatility_paths[base_idx + asset] = initial_volatility[asset];
        }

        // Generate Heston paths
        for (int step = 0; step < num_steps; ++step) {
            for (int asset = 0; asset < num_assets; ++asset) {
                const int prev_idx = path * (num_steps + 1) * num_assets + step * num_assets + asset;
                const int curr_idx = path * (num_steps + 1) * num_assets + (step + 1) * num_assets + asset;

                // Generate correlated Brownian motions
                double dW1, dW2_indep;
                box_muller_transform(&local_state, dW1, dW2_indep);

                const double rho = correlation[asset];
                const double dW2 = rho * dW1 + sqrt(1.0 - rho * rho) * dW2_indep;

                // Current values
                const double S = price_paths[prev_idx];
                const double v = fmax(volatility_paths[prev_idx], 0.0); // Ensure non-negative variance

                // Heston parameters
                const double kappa = mean_reversion_rate[asset];
                const double theta = long_term_variance[asset];
                const double sigma_v = vol_of_vol[asset];

                // Update variance using Euler scheme
                const double dv = kappa * (theta - v) * dt + sigma_v * sqrt(v * dt) * dW1;
                const double new_v = fmax(v + dv, 0.0);

                // Update price
                const double dS = risk_free_rate * S * dt + sqrt(v) * S * sqrt(dt) * dW2;
                const double new_S = S + dS;

                price_paths[curr_idx] = fmax(new_S, 0.001); // Ensure positive prices
                volatility_paths[curr_idx] = new_v;
            }
        }
    }
}

/**
 * CUDA kernel for European option pricing using Monte Carlo
 */
__global__ void european_option_pricing_kernel(
    double* __restrict__ option_values,
    const double* __restrict__ final_prices,
    const double* __restrict__ strikes,
    const bool* __restrict__ is_call,
    const double risk_free_rate,
    const double time_to_expiry,
    const int num_paths,
    const int num_options) {

    const int path_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    const double discount_factor = exp(-risk_free_rate * time_to_expiry);

    for (int path = path_idx; path < num_paths; path += stride) {
        for (int option = 0; option < num_options; ++option) {
            const double S = final_prices[path * num_options + option];
            const double K = strikes[option];

            double payoff;
            if (is_call[option]) {
                payoff = fmax(S - K, 0.0);
            } else {
                payoff = fmax(K - S, 0.0);
            }

            option_values[path * num_options + option] = payoff * discount_factor;
        }
    }
}

/**
 * CUDA kernel for Asian option pricing (arithmetic average)
 */
__global__ void asian_option_pricing_kernel(
    double* __restrict__ option_values,
    const double* __restrict__ price_paths,
    const double* __restrict__ strikes,
    const bool* __restrict__ is_call,
    const double risk_free_rate,
    const double time_to_expiry,
    const int num_paths,
    const int num_steps,
    const int num_options) {

    const int path_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    const double discount_factor = exp(-risk_free_rate * time_to_expiry);

    for (int path = path_idx; path < num_paths; path += stride) {
        for (int option = 0; option < num_options; ++option) {
            // Calculate arithmetic average
            double sum = 0.0;
            for (int step = 0; step <= num_steps; ++step) {
                const int price_idx = path * (num_steps + 1) * num_options + step * num_options + option;
                sum += price_paths[price_idx];
            }
            const double average_price = sum / (num_steps + 1);

            const double K = strikes[option];

            double payoff;
            if (is_call[option]) {
                payoff = fmax(average_price - K, 0.0);
            } else {
                payoff = fmax(K - average_price, 0.0);
            }

            option_values[path * num_options + option] = payoff * discount_factor;
        }
    }
}

/**
 * CUDA kernel for barrier option pricing
 */
__global__ void barrier_option_pricing_kernel(
    double* __restrict__ option_values,
    const double* __restrict__ price_paths,
    const double* __restrict__ strikes,
    const double* __restrict__ barriers,
    const bool* __restrict__ is_call,
    const bool* __restrict__ is_knock_in,
    const double risk_free_rate,
    const double time_to_expiry,
    const int num_paths,
    const int num_steps,
    const int num_options) {

    const int path_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    const double discount_factor = exp(-risk_free_rate * time_to_expiry);

    for (int path = path_idx; path < num_paths; path += stride) {
        for (int option = 0; option < num_options; ++option) {
            // Check barrier condition
            bool barrier_hit = false;
            const double barrier_level = barriers[option];

            for (int step = 0; step <= num_steps; ++step) {
                const int price_idx = path * (num_steps + 1) * num_options + step * num_options + option;
                const double price = price_paths[price_idx];

                if (price >= barrier_level) {
                    barrier_hit = true;
                    break;
                }
            }

            // Final price for payoff calculation
            const int final_idx = path * (num_steps + 1) * num_options + num_steps * num_options + option;
            const double final_price = price_paths[final_idx];
            const double K = strikes[option];

            double payoff = 0.0;

            // Determine if option is active based on barrier type
            bool option_active;
            if (is_knock_in[option]) {
                option_active = barrier_hit;  // Knock-in: activated if barrier hit
            } else {
                option_active = !barrier_hit; // Knock-out: active if barrier not hit
            }

            if (option_active) {
                if (is_call[option]) {
                    payoff = fmax(final_price - K, 0.0);
                } else {
                    payoff = fmax(K - final_price, 0.0);
                }
            }

            option_values[path * num_options + option] = payoff * discount_factor;
        }
    }
}

/**
 * Host class for GPU Monte Carlo operations
 */
class GPUMonteCarloEngine {
private:
    cudaDeviceProp device_props_;
    cublasHandle_t cublas_handle_;
    curandGenerator_t curand_gen_;

    // Device memory pools
    double* d_price_paths_;
    double* d_portfolio_values_;
    double* d_random_numbers_;
    size_t max_paths_;
    size_t max_assets_;
    size_t max_steps_;

    // Performance counters
    cudaEvent_t start_event_, stop_event_;

public:
    GPUMonteCarloEngine(size_t max_paths = 1000000,
                       size_t max_assets = 1024,
                       size_t max_steps = 252)
        : max_paths_(max_paths), max_assets_(max_assets), max_steps_(max_steps) {

        // Initialize CUDA
        CUDA_CHECK(cudaGetDeviceProperties(&device_props_, 0));
        CUBLAS_CHECK(cublasCreate(&cublas_handle_));

        // Create cuRAND generator
        curandCreateGenerator(&curand_gen_, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(curand_gen_, time(nullptr));

        // Allocate device memory
        allocate_device_memory();

        // Create CUDA events for timing
        CUDA_CHECK(cudaEventCreate(&start_event_));
        CUDA_CHECK(cudaEventCreate(&stop_event_));

        printf("GPU Monte Carlo Engine initialized:\n");
        printf("  Device: %s\n", device_props_.name);
        printf("  Compute Capability: %d.%d\n", device_props_.major, device_props_.minor);
        printf("  Global Memory: %.2f GB\n", device_props_.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  Max Threads per Block: %d\n", device_props_.maxThreadsPerBlock);
        printf("  Multiprocessor Count: %d\n", device_props_.multiProcessorCount);
    }

    ~GPUMonteCarloEngine() {
        // Free device memory
        cudaFree(d_price_paths_);
        cudaFree(d_portfolio_values_);
        cudaFree(d_random_numbers_);

        // Cleanup handles
        cublasDestroy(cublas_handle_);
        curandDestroyGenerator(curand_gen_);

        // Destroy events
        cudaEventDestroy(start_event_);
        cudaEventDestroy(stop_event_);
    }

    /**
     * Generate Monte Carlo price paths using GPU
     */
    std::vector<std::vector<std::vector<double>>> generate_price_paths(
        const std::vector<double>& initial_prices,
        const std::vector<double>& drift,
        const std::vector<double>& volatility,
        const std::vector<std::vector<double>>& correlation_matrix,
        double dt,
        int num_paths,
        int num_steps) {

        const int num_assets = initial_prices.size();

        // Calculate Cholesky decomposition on CPU
        std::vector<std::vector<double>> cholesky = cholesky_decomposition(correlation_matrix);

        // Copy data to GPU
        double* d_initial_prices;
        double* d_drift;
        double* d_volatility;
        double* d_cholesky;

        const size_t asset_size = num_assets * sizeof(double);
        const size_t cholesky_size = num_assets * num_assets * sizeof(double);

        CUDA_CHECK(cudaMalloc(&d_initial_prices, asset_size));
        CUDA_CHECK(cudaMalloc(&d_drift, asset_size));
        CUDA_CHECK(cudaMalloc(&d_volatility, asset_size));
        CUDA_CHECK(cudaMalloc(&d_cholesky, cholesky_size));

        CUDA_CHECK(cudaMemcpy(d_initial_prices, initial_prices.data(), asset_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_drift, drift.data(), asset_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_volatility, volatility.data(), asset_size, cudaMemcpyHostToDevice));

        // Flatten Cholesky matrix
        std::vector<double> flat_cholesky(num_assets * num_assets);
        for (int i = 0; i < num_assets; ++i) {
            for (int j = 0; j < num_assets; ++j) {
                flat_cholesky[i * num_assets + j] = cholesky[i][j];
            }
        }
        CUDA_CHECK(cudaMemcpy(d_cholesky, flat_cholesky.data(), cholesky_size, cudaMemcpyHostToDevice));

        // Launch kernel
        const int block_size = BLOCK_SIZE;
        const int grid_size = (num_paths + block_size - 1) / block_size;
        const size_t shared_mem_size = block_size * num_assets * sizeof(double);

        CUDA_CHECK(cudaEventRecord(start_event_));

        geometric_brownian_motion_kernel<<<grid_size, block_size, shared_mem_size>>>(
            d_price_paths_,
            d_initial_prices,
            d_drift,
            d_volatility,
            d_cholesky,
            dt,
            num_paths,
            num_steps,
            num_assets,
            static_cast<unsigned long long>(time(nullptr))
        );

        CUDA_CHECK(cudaEventRecord(stop_event_));
        CUDA_CHECK(cudaEventSynchronize(stop_event_));

        // Calculate execution time
        float elapsed_time;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start_event_, stop_event_));
        printf("GPU GBM simulation completed in %.2f ms\n", elapsed_time);

        // Copy results back to host
        const size_t path_size = num_paths * (num_steps + 1) * num_assets * sizeof(double);
        std::vector<double> host_paths(num_paths * (num_steps + 1) * num_assets);
        CUDA_CHECK(cudaMemcpy(host_paths.data(), d_price_paths_, path_size, cudaMemcpyDeviceToHost));

        // Reshape to 3D vector
        std::vector<std::vector<std::vector<double>>> result(
            num_paths, std::vector<std::vector<double>>(
                num_steps + 1, std::vector<double>(num_assets)
            )
        );

        for (int path = 0; path < num_paths; ++path) {
            for (int step = 0; step <= num_steps; ++step) {
                for (int asset = 0; asset < num_assets; ++asset) {
                    const int idx = path * (num_steps + 1) * num_assets + step * num_assets + asset;
                    result[path][step][asset] = host_paths[idx];
                }
            }
        }

        // Cleanup
        cudaFree(d_initial_prices);
        cudaFree(d_drift);
        cudaFree(d_volatility);
        cudaFree(d_cholesky);

        return result;
    }

    /**
     * Calculate portfolio VaR using GPU Monte Carlo
     */
    double calculate_portfolio_var_gpu(
        const std::vector<double>& weights,
        const std::vector<double>& initial_prices,
        const std::vector<double>& expected_returns,
        const std::vector<std::vector<double>>& covariance_matrix,
        double confidence_level = 0.95,
        int num_simulations = 100000) {

        const int num_assets = weights.size();

        // Generate portfolio returns
        std::vector<double> portfolio_returns = generate_portfolio_returns_gpu(
            weights, initial_prices, expected_returns, covariance_matrix, num_simulations
        );

        // Calculate VaR on GPU
        double* d_returns;
        double* d_sorted_returns;
        double* d_var_result;

        const size_t returns_size = num_simulations * sizeof(double);

        CUDA_CHECK(cudaMalloc(&d_returns, returns_size));
        CUDA_CHECK(cudaMalloc(&d_sorted_returns, returns_size));
        CUDA_CHECK(cudaMalloc(&d_var_result, sizeof(double)));

        CUDA_CHECK(cudaMemcpy(d_returns, portfolio_returns.data(), returns_size, cudaMemcpyHostToDevice));

        // Launch VaR calculation kernel
        const int block_size = BLOCK_SIZE;
        const int grid_size = (num_simulations + block_size - 1) / block_size;

        calculate_var_kernel<<<grid_size, block_size>>>(
            d_returns,
            d_sorted_returns,
            d_var_result,
            num_simulations,
            confidence_level
        );

        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy result back
        double var_result;
        CUDA_CHECK(cudaMemcpy(&var_result, d_var_result, sizeof(double), cudaMemcpyDeviceToHost));

        // Cleanup
        cudaFree(d_returns);
        cudaFree(d_sorted_returns);
        cudaFree(d_var_result);

        return var_result;
    }

private:
    void allocate_device_memory() {
        const size_t path_memory = max_paths_ * (max_steps_ + 1) * max_assets_ * sizeof(double);
        const size_t portfolio_memory = max_paths_ * sizeof(double);
        const size_t random_memory = max_paths_ * max_assets_ * sizeof(double);

        CUDA_CHECK(cudaMalloc(&d_price_paths_, path_memory));
        CUDA_CHECK(cudaMalloc(&d_portfolio_values_, portfolio_memory));
        CUDA_CHECK(cudaMalloc(&d_random_numbers_, random_memory));

        printf("Allocated %.2f GB of GPU memory\n",
               (path_memory + portfolio_memory + random_memory) / (1024.0 * 1024.0 * 1024.0));
    }

    std::vector<std::vector<double>> cholesky_decomposition(
        const std::vector<std::vector<double>>& matrix) {

        const int n = matrix.size();
        std::vector<std::vector<double>> L(n, std::vector<double>(n, 0.0));

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j <= i; ++j) {
                if (i == j) {
                    double sum = 0.0;
                    for (int k = 0; k < j; ++k) {
                        sum += L[j][k] * L[j][k];
                    }
                    L[j][j] = std::sqrt(matrix[j][j] - sum);
                } else {
                    double sum = 0.0;
                    for (int k = 0; k < j; ++k) {
                        sum += L[i][k] * L[j][k];
                    }
                    L[i][j] = (matrix[i][j] - sum) / L[j][j];
                }
            }
        }

        return L;
    }

    std::vector<double> generate_portfolio_returns_gpu(
        const std::vector<double>& weights,
        const std::vector<double>& initial_prices,
        const std::vector<double>& expected_returns,
        const std::vector<std::vector<double>>& covariance_matrix,
        int num_simulations) {

        // Implementation for GPU-based portfolio return generation
        // This would involve similar CUDA kernel launches

        std::vector<double> returns(num_simulations);
        // ... GPU implementation ...

        return returns;
    }
};

} // namespace gpu
} // namespace hpc
} // namespace risk_analytics