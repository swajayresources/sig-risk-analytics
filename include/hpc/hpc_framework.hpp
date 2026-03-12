/**
 * High-Performance Computing Framework for Real-Time Risk Analytics
 *
 * This framework provides:
 * - Multi-threaded risk calculations with OpenMP
 * - SIMD vectorization for mathematical operations
 * - GPU acceleration with CUDA/OpenCL
 * - Lock-free data structures for concurrent access
 * - Memory-optimized algorithms and cache-friendly designs
 * - Real-time event-driven processing
 */

#pragma once

#include <memory>
#include <vector>
#include <array>
#include <atomic>
#include <thread>
#include <future>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <functional>
#include <unordered_map>
#include <immintrin.h>  // SIMD intrinsics

#ifdef USE_OPENMP
#include <omp.h>
#endif

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#endif

#ifdef USE_OPENCL
#include <CL/cl.hpp>
#endif

namespace risk_analytics {
namespace hpc {

// Forward declarations
class ThreadPool;
class MemoryPool;
class EventProcessor;
class GPUContext;

/**
 * High-precision floating point type for financial calculations
 */
using Real = double;
using RealVector = std::vector<Real>;
using RealMatrix = std::vector<std::vector<Real>>;

/**
 * SIMD-aligned data structures for vectorized operations
 */
template<typename T, size_t Alignment = 32>
struct AlignedVector {
    std::vector<T> data;

    AlignedVector(size_t size) : data(size) {
        // Ensure proper alignment for SIMD operations
        static_assert(Alignment % sizeof(T) == 0, "Alignment must be multiple of element size");
    }

    T* get() { return data.data(); }
    const T* get() const { return data.data(); }
    size_t size() const { return data.size(); }

    T& operator[](size_t idx) { return data[idx]; }
    const T& operator[](size_t idx) const { return data[idx]; }
};

/**
 * Lock-free ring buffer for high-throughput data exchange
 */
template<typename T, size_t Size>
class LockFreeRingBuffer {
private:
    alignas(64) std::array<T, Size> buffer_;
    alignas(64) std::atomic<size_t> head_{0};
    alignas(64) std::atomic<size_t> tail_{0};

public:
    static constexpr size_t capacity = Size;

    bool push(const T& item) {
        const size_t current_tail = tail_.load(std::memory_order_relaxed);
        const size_t next_tail = (current_tail + 1) % Size;

        if (next_tail == head_.load(std::memory_order_acquire)) {
            return false; // Buffer full
        }

        buffer_[current_tail] = item;
        tail_.store(next_tail, std::memory_order_release);
        return true;
    }

    bool pop(T& item) {
        const size_t current_head = head_.load(std::memory_order_relaxed);

        if (current_head == tail_.load(std::memory_order_acquire)) {
            return false; // Buffer empty
        }

        item = buffer_[current_head];
        head_.store((current_head + 1) % Size, std::memory_order_release);
        return true;
    }

    size_t size() const {
        const size_t tail = tail_.load(std::memory_order_acquire);
        const size_t head = head_.load(std::memory_order_acquire);
        return (tail >= head) ? (tail - head) : (Size + tail - head);
    }

    bool empty() const {
        return head_.load(std::memory_order_acquire) == tail_.load(std::memory_order_acquire);
    }
};

/**
 * High-performance memory pool for frequent allocations
 */
class MemoryPool {
private:
    struct Block {
        char* data;
        size_t size;
        std::atomic<bool> in_use{false};
        Block* next;
    };

    std::unique_ptr<char[]> memory_;
    Block* free_blocks_;
    std::mutex mutex_;
    size_t block_size_;
    size_t num_blocks_;

public:
    MemoryPool(size_t block_size, size_t num_blocks);
    ~MemoryPool();

    void* allocate(size_t size);
    void deallocate(void* ptr);

    size_t get_available_blocks() const;
    size_t get_total_memory() const { return block_size_ * num_blocks_; }
};

/**
 * SIMD-optimized mathematical operations
 */
namespace simd {

    /**
     * Vectorized dot product using AVX2
     */
    inline Real dot_product_avx2(const Real* a, const Real* b, size_t size) {
        Real result = 0.0;

        #ifdef __AVX2__
        const size_t simd_size = size & ~3; // Process 4 doubles at a time
        __m256d sum = _mm256_setzero_pd();

        for (size_t i = 0; i < simd_size; i += 4) {
            __m256d va = _mm256_loadu_pd(&a[i]);
            __m256d vb = _mm256_loadu_pd(&b[i]);
            sum = _mm256_fmadd_pd(va, vb, sum);
        }

        // Horizontal sum
        __m128d sum_high = _mm256_extractf128_pd(sum, 1);
        __m128d sum_low = _mm256_castpd256_pd128(sum);
        __m128d sum_total = _mm_add_pd(sum_low, sum_high);
        __m128d sum_final = _mm_hadd_pd(sum_total, sum_total);
        result = _mm_cvtsd_f64(sum_final);

        // Handle remaining elements
        for (size_t i = simd_size; i < size; ++i) {
            result += a[i] * b[i];
        }
        #else
        // Fallback to scalar implementation
        for (size_t i = 0; i < size; ++i) {
            result += a[i] * b[i];
        }
        #endif

        return result;
    }

    /**
     * Vectorized matrix-vector multiplication
     */
    void matrix_vector_multiply_avx2(const Real* matrix, const Real* vector,
                                   Real* result, size_t rows, size_t cols);

    /**
     * Vectorized exponential function for arrays
     */
    void exp_array_avx2(const Real* input, Real* output, size_t size);

    /**
     * Vectorized normal CDF approximation
     */
    void normal_cdf_array_avx2(const Real* input, Real* output, size_t size);

} // namespace simd

/**
 * GPU computation context for CUDA/OpenCL operations
 */
class GPUContext {
private:
    bool cuda_available_;
    bool opencl_available_;

    #ifdef USE_CUDA
    cudaDeviceProp cuda_props_;
    cublasHandle_t cublas_handle_;
    curandGenerator_t curand_gen_;
    #endif

    #ifdef USE_OPENCL
    cl::Context opencl_context_;
    cl::CommandQueue opencl_queue_;
    cl::Device opencl_device_;
    #endif

public:
    GPUContext();
    ~GPUContext();

    bool is_cuda_available() const { return cuda_available_; }
    bool is_opencl_available() const { return opencl_available_; }

    // CUDA operations
    #ifdef USE_CUDA
    cudaError_t cuda_matrix_multiply(const Real* A, const Real* B, Real* C,
                                   int m, int n, int k);
    cudaError_t cuda_monte_carlo_paths(Real* paths, const Real* initial_prices,
                                     const Real* drift, const Real* volatility,
                                     int num_paths, int num_steps, int num_assets);
    cudaError_t cuda_portfolio_var(const Real* returns, const Real* weights,
                                 Real* var_result, int num_scenarios, int num_assets);
    #endif

    // OpenCL operations
    #ifdef USE_OPENCL
    cl_int opencl_matrix_multiply(const Real* A, const Real* B, Real* C,
                                size_t m, size_t n, size_t k);
    cl_int opencl_monte_carlo_paths(Real* paths, const Real* initial_prices,
                                  const Real* drift, const Real* volatility,
                                  size_t num_paths, size_t num_steps, size_t num_assets);
    #endif
};

/**
 * High-performance thread pool for parallel processing
 */
class ThreadPool {
private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    std::atomic<bool> stop_{false};

public:
    ThreadPool(size_t num_threads = std::thread::hardware_concurrency());
    ~ThreadPool();

    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args)
        -> std::future<typename std::result_of<F(Args...)>::type> {

        using return_type = typename std::result_of<F(Args...)>::type;

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );

        std::future<return_type> res = task->get_future();

        {
            std::unique_lock<std::mutex> lock(queue_mutex_);

            if (stop_) {
                throw std::runtime_error("enqueue on stopped ThreadPool");
            }

            tasks_.emplace([task](){ (*task)(); });
        }

        condition_.notify_one();
        return res;
    }

    size_t get_num_threads() const { return workers_.size(); }
    size_t get_queue_size() const;
};

/**
 * Performance profiler for monitoring system performance
 */
class PerformanceProfiler {
private:
    struct ProfileData {
        std::chrono::high_resolution_clock::time_point start_time;
        std::chrono::nanoseconds total_time{0};
        size_t call_count{0};
        std::string name;
    };

    std::unordered_map<std::string, ProfileData> profiles_;
    std::mutex mutex_;

public:
    class ScopedTimer {
    private:
        PerformanceProfiler& profiler_;
        std::string name_;
        std::chrono::high_resolution_clock::time_point start_time_;

    public:
        ScopedTimer(PerformanceProfiler& profiler, const std::string& name)
            : profiler_(profiler), name_(name),
              start_time_(std::chrono::high_resolution_clock::now()) {}

        ~ScopedTimer() {
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = end_time - start_time_;
            profiler_.record_timing(name_, duration);
        }
    };

    void record_timing(const std::string& name, std::chrono::nanoseconds duration);
    void print_profile_report() const;
    void reset_profiles();

    // Convenience macro for profiling
    #define PROFILE_SCOPE(profiler, name) \
        PerformanceProfiler::ScopedTimer timer(profiler, name)
};

/**
 * Event-driven data structure for real-time updates
 */
template<typename EventType>
class EventQueue {
private:
    LockFreeRingBuffer<EventType, 65536> buffer_;
    std::vector<std::function<void(const EventType&)>> handlers_;
    std::atomic<bool> processing_{false};
    std::thread processing_thread_;

public:
    EventQueue() {
        processing_.store(true);
        processing_thread_ = std::thread([this]() { process_events(); });
    }

    ~EventQueue() {
        processing_.store(false);
        if (processing_thread_.joinable()) {
            processing_thread_.join();
        }
    }

    bool publish(const EventType& event) {
        return buffer_.push(event);
    }

    void subscribe(std::function<void(const EventType&)> handler) {
        handlers_.push_back(std::move(handler));
    }

private:
    void process_events() {
        EventType event;
        while (processing_.load(std::memory_order_acquire)) {
            if (buffer_.pop(event)) {
                for (auto& handler : handlers_) {
                    handler(event);
                }
            } else {
                std::this_thread::yield();
            }
        }
    }
};

/**
 * Cache-friendly data structure for position storage
 */
struct Position {
    alignas(64) struct {
        char symbol[16];        // 16 bytes - Symbol identifier
        double quantity;        // 8 bytes - Position size
        double price;          // 8 bytes - Current price
        double market_value;   // 8 bytes - Market value
        double delta;          // 8 bytes - Price sensitivity
        double gamma;          // 8 bytes - Delta sensitivity
        double theta;          // 8 bytes - Time decay
        double vega;           // 8 bytes - Volatility sensitivity
    };

    Position() = default;
    Position(const char* sym, double qty, double prc)
        : quantity(qty), price(prc), market_value(qty * prc),
          delta(0), gamma(0), theta(0), vega(0) {
        std::strncpy(symbol, sym, sizeof(symbol) - 1);
        symbol[sizeof(symbol) - 1] = '\0';
    }
};

/**
 * High-performance portfolio container with fast lookups
 */
class Portfolio {
private:
    std::vector<Position> positions_;
    std::unordered_map<std::string, size_t> symbol_index_;
    std::atomic<double> total_value_{0.0};
    std::atomic<size_t> version_{0};  // For optimistic locking
    mutable std::shared_mutex mutex_;

public:
    Portfolio() = default;

    void add_position(const Position& position);
    void update_position(const std::string& symbol, double new_price);
    void remove_position(const std::string& symbol);

    const Position* get_position(const std::string& symbol) const;
    std::vector<Position> get_all_positions() const;

    double get_total_value() const { return total_value_.load(); }
    size_t get_version() const { return version_.load(); }
    size_t size() const;

    // Lock-free read operations for hot paths
    bool try_get_position_fast(const std::string& symbol, Position& out_position) const;
};

/**
 * Configuration for high-performance computing
 */
struct HPCConfig {
    // Threading configuration
    size_t num_worker_threads = std::thread::hardware_concurrency();
    size_t num_calculation_threads = std::thread::hardware_concurrency() / 2;

    // Memory configuration
    size_t memory_pool_block_size = 1024 * 1024;  // 1 MB blocks
    size_t memory_pool_num_blocks = 1024;         // 1 GB total

    // SIMD configuration
    bool enable_simd = true;
    bool enable_avx2 = true;
    bool enable_avx512 = false;

    // GPU configuration
    bool enable_cuda = true;
    bool enable_opencl = false;
    size_t gpu_memory_limit = 2ULL * 1024 * 1024 * 1024;  // 2 GB

    // Performance tuning
    bool enable_numa_binding = true;
    bool enable_cpu_affinity = true;
    bool enable_huge_pages = false;

    // Cache configuration
    size_t l1_cache_size = 32 * 1024;      // 32 KB
    size_t l2_cache_size = 256 * 1024;     // 256 KB
    size_t l3_cache_size = 8 * 1024 * 1024; // 8 MB
};

/**
 * Main HPC framework coordinator
 */
class HPCFramework {
private:
    HPCConfig config_;
    std::unique_ptr<ThreadPool> thread_pool_;
    std::unique_ptr<MemoryPool> memory_pool_;
    std::unique_ptr<GPUContext> gpu_context_;
    std::unique_ptr<PerformanceProfiler> profiler_;

    // System initialization
    void initialize_numa();
    void initialize_cpu_affinity();
    void initialize_huge_pages();
    void detect_hardware_capabilities();

public:
    HPCFramework(const HPCConfig& config = HPCConfig{});
    ~HPCFramework();

    // Accessors
    ThreadPool& get_thread_pool() { return *thread_pool_; }
    MemoryPool& get_memory_pool() { return *memory_pool_; }
    GPUContext& get_gpu_context() { return *gpu_context_; }
    PerformanceProfiler& get_profiler() { return *profiler_; }

    const HPCConfig& get_config() const { return config_; }

    // System information
    void print_system_info() const;
    void print_performance_report() const;

    // Benchmark utilities
    void run_benchmark_suite();
};

} // namespace hpc
} // namespace risk_analytics