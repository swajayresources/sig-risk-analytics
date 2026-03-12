/**
 * Performance Monitoring and Profiling Tools
 *
 * Comprehensive real-time performance monitoring system for
 * high-frequency risk analytics with detailed profiling capabilities
 */

#pragma once

#include "hpc_framework.hpp"
#include "lockfree_structures.hpp"
#include <chrono>
#include <atomic>
#include <thread>
#include <array>
#include <fstream>
#include <iomanip>
#include <sstream>

#ifdef __linux__
#include <sys/resource.h>
#include <sys/times.h>
#include <unistd.h>
#include <numa.h>
#include <proc/readproc.h>
#include <perf_event.h>
#include <sys/syscall.h>
#include <sys/ioctl.h>
#endif

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#include <pdh.h>
#endif

namespace risk_analytics {
namespace hpc {
namespace monitoring {

/**
 * High-resolution timer for precise latency measurements
 */
class HighResolutionTimer {
private:
    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point end_time_;
    bool is_running_;

public:
    HighResolutionTimer() : is_running_(false) {}

    void start() {
        start_time_ = std::chrono::high_resolution_clock::now();
        is_running_ = true;
    }

    void stop() {
        end_time_ = std::chrono::high_resolution_clock::now();
        is_running_ = false;
    }

    double elapsed_nanoseconds() const {
        auto end = is_running_ ? std::chrono::high_resolution_clock::now() : end_time_;
        return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_time_).count();
    }

    double elapsed_microseconds() const {
        return elapsed_nanoseconds() / 1000.0;
    }

    double elapsed_milliseconds() const {
        return elapsed_nanoseconds() / 1000000.0;
    }

    double elapsed_seconds() const {
        return elapsed_nanoseconds() / 1000000000.0;
    }
};

/**
 * Latency histogram for detailed distribution analysis
 */
template<size_t NumBuckets = 64>
class LatencyHistogram {
private:
    static constexpr double MAX_LATENCY_US = 100000.0; // 100ms max
    static constexpr double BUCKET_SIZE = MAX_LATENCY_US / NumBuckets;

    std::array<std::atomic<uint64_t>, NumBuckets> buckets_;
    std::atomic<uint64_t> total_samples_{0};
    std::atomic<uint64_t> overflow_samples_{0};
    std::atomic<double> sum_latency_{0.0};
    std::atomic<double> sum_squared_latency_{0.0};
    std::atomic<double> min_latency_{std::numeric_limits<double>::max()};
    std::atomic<double> max_latency_{0.0};

public:
    LatencyHistogram() {
        for (auto& bucket : buckets_) {
            bucket.store(0);
        }
    }

    void record_latency(double latency_us) {
        total_samples_.fetch_add(1, std::memory_order_relaxed);
        sum_latency_.fetch_add(latency_us, std::memory_order_relaxed);
        sum_squared_latency_.fetch_add(latency_us * latency_us, std::memory_order_relaxed);

        // Update min/max
        update_min_max(latency_us);

        // Determine bucket
        if (latency_us >= MAX_LATENCY_US) {
            overflow_samples_.fetch_add(1, std::memory_order_relaxed);
        } else {
            size_t bucket_index = static_cast<size_t>(latency_us / BUCKET_SIZE);
            if (bucket_index < NumBuckets) {
                buckets_[bucket_index].fetch_add(1, std::memory_order_relaxed);
            }
        }
    }

    struct Statistics {
        uint64_t total_samples;
        double mean_latency_us;
        double std_dev_latency_us;
        double min_latency_us;
        double max_latency_us;
        double percentile_50;
        double percentile_90;
        double percentile_95;
        double percentile_99;
        double percentile_999;
        uint64_t overflow_samples;
    };

    Statistics get_statistics() const {
        const uint64_t samples = total_samples_.load(std::memory_order_acquire);
        if (samples == 0) {
            return Statistics{};
        }

        const double sum = sum_latency_.load(std::memory_order_acquire);
        const double sum_sq = sum_squared_latency_.load(std::memory_order_acquire);
        const double mean = sum / samples;
        const double variance = (sum_sq / samples) - (mean * mean);
        const double std_dev = std::sqrt(std::max(0.0, variance));

        // Calculate percentiles
        std::array<uint64_t, NumBuckets> bucket_counts;
        for (size_t i = 0; i < NumBuckets; ++i) {
            bucket_counts[i] = buckets_[i].load(std::memory_order_acquire);
        }

        return Statistics{
            .total_samples = samples,
            .mean_latency_us = mean,
            .std_dev_latency_us = std_dev,
            .min_latency_us = min_latency_.load(std::memory_order_acquire),
            .max_latency_us = max_latency_.load(std::memory_order_acquire),
            .percentile_50 = calculate_percentile(bucket_counts, samples, 0.50),
            .percentile_90 = calculate_percentile(bucket_counts, samples, 0.90),
            .percentile_95 = calculate_percentile(bucket_counts, samples, 0.95),
            .percentile_99 = calculate_percentile(bucket_counts, samples, 0.99),
            .percentile_999 = calculate_percentile(bucket_counts, samples, 0.999),
            .overflow_samples = overflow_samples_.load(std::memory_order_acquire)
        };
    }

    void reset() {
        for (auto& bucket : buckets_) {
            bucket.store(0, std::memory_order_relaxed);
        }
        total_samples_.store(0, std::memory_order_relaxed);
        overflow_samples_.store(0, std::memory_order_relaxed);
        sum_latency_.store(0.0, std::memory_order_relaxed);
        sum_squared_latency_.store(0.0, std::memory_order_relaxed);
        min_latency_.store(std::numeric_limits<double>::max(), std::memory_order_relaxed);
        max_latency_.store(0.0, std::memory_order_relaxed);
    }

    void print_histogram() const {
        auto stats = get_statistics();
        printf("\nLatency Histogram (μs):\n");
        printf("Samples: %lu, Mean: %.2f, StdDev: %.2f, Min: %.2f, Max: %.2f\n",
               stats.total_samples, stats.mean_latency_us, stats.std_dev_latency_us,
               stats.min_latency_us, stats.max_latency_us);
        printf("P50: %.2f, P90: %.2f, P95: %.2f, P99: %.2f, P99.9: %.2f\n",
               stats.percentile_50, stats.percentile_90, stats.percentile_95,
               stats.percentile_99, stats.percentile_999);

        printf("\nDistribution:\n");
        for (size_t i = 0; i < NumBuckets; ++i) {
            uint64_t count = buckets_[i].load(std::memory_order_acquire);
            if (count > 0) {
                double bucket_start = i * BUCKET_SIZE;
                double bucket_end = (i + 1) * BUCKET_SIZE;
                double percentage = (double(count) / stats.total_samples) * 100.0;
                printf("[%6.1f - %6.1f): %8lu (%5.2f%%) ",
                       bucket_start, bucket_end, count, percentage);

                // Simple ASCII bar chart
                int bar_length = static_cast<int>(percentage * 0.5); // Scale down
                for (int j = 0; j < bar_length; ++j) {
                    printf("█");
                }
                printf("\n");
            }
        }

        if (stats.overflow_samples > 0) {
            double percentage = (double(stats.overflow_samples) / stats.total_samples) * 100.0;
            printf("[%6.1f+     ): %8lu (%5.2f%%)\n",
                   MAX_LATENCY_US, stats.overflow_samples, percentage);
        }
    }

private:
    void update_min_max(double latency) {
        // Update minimum
        double current_min = min_latency_.load(std::memory_order_relaxed);
        while (latency < current_min &&
               !min_latency_.compare_exchange_weak(current_min, latency,
                                                  std::memory_order_relaxed)) {
            // Loop until successful update
        }

        // Update maximum
        double current_max = max_latency_.load(std::memory_order_relaxed);
        while (latency > current_max &&
               !max_latency_.compare_exchange_weak(current_max, latency,
                                                  std::memory_order_relaxed)) {
            // Loop until successful update
        }
    }

    double calculate_percentile(const std::array<uint64_t, NumBuckets>& buckets,
                               uint64_t total_samples, double percentile) const {
        uint64_t target_sample = static_cast<uint64_t>(total_samples * percentile);
        uint64_t accumulated = 0;

        for (size_t i = 0; i < NumBuckets; ++i) {
            accumulated += buckets[i];
            if (accumulated >= target_sample) {
                return (i + 0.5) * BUCKET_SIZE; // Mid-bucket value
            }
        }

        return MAX_LATENCY_US; // Fallback
    }
};

/**
 * System resource monitor for CPU, memory, and I/O usage
 */
class SystemResourceMonitor {
private:
    struct ResourceSnapshot {
        double cpu_usage_percent;
        uint64_t memory_used_bytes;
        uint64_t memory_total_bytes;
        double memory_usage_percent;
        uint64_t disk_read_bytes_per_sec;
        uint64_t disk_write_bytes_per_sec;
        uint64_t network_rx_bytes_per_sec;
        uint64_t network_tx_bytes_per_sec;
        uint32_t open_file_descriptors;
        uint32_t thread_count;
        double load_average_1min;
        std::chrono::time_point<std::chrono::high_resolution_clock> timestamp;
    };

    lockfree::LockFreeRingBuffer<ResourceSnapshot, 3600> snapshots_; // 1 hour at 1Hz
    std::thread monitoring_thread_;
    std::atomic<bool> running_{false};

    // Performance counters (Linux)
    #ifdef __linux__
    int perf_fd_cycles_;
    int perf_fd_instructions_;
    int perf_fd_cache_misses_;
    #endif

    // Performance counters (Windows)
    #ifdef _WIN32
    PDH_HQUERY pdh_query_;
    PDH_HCOUNTER cpu_counter_;
    PDH_HCOUNTER memory_counter_;
    #endif

public:
    SystemResourceMonitor() {
        initialize_performance_counters();
    }

    ~SystemResourceMonitor() {
        stop();
        cleanup_performance_counters();
    }

    void start() {
        running_.store(true);
        monitoring_thread_ = std::thread([this]() { monitoring_worker(); });
    }

    void stop() {
        running_.store(false);
        if (monitoring_thread_.joinable()) {
            monitoring_thread_.join();
        }
    }

    ResourceSnapshot get_current_snapshot() const {
        ResourceSnapshot snapshot;
        if (!snapshots_.empty()) {
            // Get the most recent snapshot
            snapshots_.pop(const_cast<ResourceSnapshot&>(snapshot));
        }
        return snapshot;
    }

    struct ResourceStatistics {
        double avg_cpu_usage;
        double max_cpu_usage;
        double avg_memory_usage;
        double max_memory_usage;
        uint64_t peak_memory_bytes;
        double avg_load;
        uint32_t max_threads;
        uint32_t max_file_descriptors;
    };

    ResourceStatistics get_statistics() const {
        // Calculate statistics from stored snapshots
        ResourceStatistics stats{};

        // Implementation would aggregate data from the ring buffer
        // This is a simplified version
        auto current = get_current_snapshot();
        stats.avg_cpu_usage = current.cpu_usage_percent;
        stats.max_cpu_usage = current.cpu_usage_percent;
        stats.avg_memory_usage = current.memory_usage_percent;
        stats.max_memory_usage = current.memory_usage_percent;
        stats.peak_memory_bytes = current.memory_used_bytes;
        stats.avg_load = current.load_average_1min;
        stats.max_threads = current.thread_count;
        stats.max_file_descriptors = current.open_file_descriptors;

        return stats;
    }

    void print_current_status() const {
        auto snapshot = get_current_snapshot();
        printf("\n=== System Resource Status ===\n");
        printf("CPU Usage:      %6.2f%%\n", snapshot.cpu_usage_percent);
        printf("Memory Usage:   %6.2f%% (%lu MB / %lu MB)\n",
               snapshot.memory_usage_percent,
               snapshot.memory_used_bytes / (1024 * 1024),
               snapshot.memory_total_bytes / (1024 * 1024));
        printf("Load Average:   %6.2f\n", snapshot.load_average_1min);
        printf("Threads:        %6u\n", snapshot.thread_count);
        printf("File Desc:      %6u\n", snapshot.open_file_descriptors);
        printf("Disk I/O:       %6lu KB/s read, %6lu KB/s write\n",
               snapshot.disk_read_bytes_per_sec / 1024,
               snapshot.disk_write_bytes_per_sec / 1024);
        printf("Network I/O:    %6lu KB/s rx, %6lu KB/s tx\n",
               snapshot.network_rx_bytes_per_sec / 1024,
               snapshot.network_tx_bytes_per_sec / 1024);
    }

private:
    void initialize_performance_counters() {
        #ifdef __linux__
        // Initialize Linux perf events
        struct perf_event_attr pe;
        memset(&pe, 0, sizeof(pe));
        pe.type = PERF_TYPE_HARDWARE;
        pe.size = sizeof(pe);
        pe.config = PERF_COUNT_HW_CPU_CYCLES;
        pe.disabled = 1;
        pe.exclude_kernel = 1;
        pe.exclude_hv = 1;

        perf_fd_cycles_ = syscall(SYS_perf_event_open, &pe, 0, -1, -1, 0);

        pe.config = PERF_COUNT_HW_INSTRUCTIONS;
        perf_fd_instructions_ = syscall(SYS_perf_event_open, &pe, 0, -1, -1, 0);

        pe.config = PERF_COUNT_HW_CACHE_MISSES;
        perf_fd_cache_misses_ = syscall(SYS_perf_event_open, &pe, 0, -1, -1, 0);
        #endif

        #ifdef _WIN32
        // Initialize Windows performance counters
        PdhOpenQuery(NULL, 0, &pdh_query_);
        PdhAddCounter(pdh_query_, L"\\Processor(_Total)\\% Processor Time", 0, &cpu_counter_);
        PdhAddCounter(pdh_query_, L"\\Memory\\Available Bytes", 0, &memory_counter_);
        PdhCollectQueryData(pdh_query_);
        #endif
    }

    void cleanup_performance_counters() {
        #ifdef __linux__
        if (perf_fd_cycles_ >= 0) close(perf_fd_cycles_);
        if (perf_fd_instructions_ >= 0) close(perf_fd_instructions_);
        if (perf_fd_cache_misses_ >= 0) close(perf_fd_cache_misses_);
        #endif

        #ifdef _WIN32
        PdhCloseQuery(pdh_query_);
        #endif
    }

    void monitoring_worker() {
        while (running_.load()) {
            ResourceSnapshot snapshot = collect_resource_snapshot();
            snapshots_.push(snapshot);
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }

    ResourceSnapshot collect_resource_snapshot() {
        ResourceSnapshot snapshot{};
        snapshot.timestamp = std::chrono::high_resolution_clock::now();

        #ifdef __linux__
        collect_linux_metrics(snapshot);
        #endif

        #ifdef _WIN32
        collect_windows_metrics(snapshot);
        #endif

        return snapshot;
    }

    #ifdef __linux__
    void collect_linux_metrics(ResourceSnapshot& snapshot) {
        // CPU usage
        static long long prev_idle = 0, prev_total = 0;
        std::ifstream stat_file("/proc/stat");
        std::string line;
        if (std::getline(stat_file, line)) {
            std::istringstream iss(line);
            std::string cpu;
            long long user, nice, system, idle, iowait, irq, softirq, steal;
            iss >> cpu >> user >> nice >> system >> idle >> iowait >> irq >> softirq >> steal;

            long long total = user + nice + system + idle + iowait + irq + softirq + steal;
            long long diff_idle = idle - prev_idle;
            long long diff_total = total - prev_total;

            if (diff_total > 0) {
                snapshot.cpu_usage_percent = 100.0 * (diff_total - diff_idle) / diff_total;
            }

            prev_idle = idle;
            prev_total = total;
        }

        // Memory usage
        std::ifstream meminfo("/proc/meminfo");
        std::string key;
        uint64_t value;
        uint64_t mem_total = 0, mem_available = 0;

        while (meminfo >> key >> value) {
            if (key == "MemTotal:") {
                mem_total = value * 1024; // Convert from KB to bytes
            } else if (key == "MemAvailable:") {
                mem_available = value * 1024;
                break;
            }
        }

        snapshot.memory_total_bytes = mem_total;
        snapshot.memory_used_bytes = mem_total - mem_available;
        snapshot.memory_usage_percent = mem_total > 0 ?
            100.0 * snapshot.memory_used_bytes / mem_total : 0.0;

        // Load average
        std::ifstream loadavg("/proc/loadavg");
        loadavg >> snapshot.load_average_1min;

        // Process info
        std::ifstream status("/proc/self/status");
        while (std::getline(status, line)) {
            if (line.find("Threads:") == 0) {
                std::istringstream iss(line);
                std::string label;
                iss >> label >> snapshot.thread_count;
                break;
            }
        }

        // File descriptors
        std::ifstream fd_count("/proc/self/fd");
        snapshot.open_file_descriptors = std::distance(
            std::istreambuf_iterator<char>(fd_count),
            std::istreambuf_iterator<char>()
        );

        // Network I/O (simplified)
        static uint64_t prev_rx_bytes = 0, prev_tx_bytes = 0;
        std::ifstream net_dev("/proc/net/dev");
        std::string dev_line;
        uint64_t total_rx = 0, total_tx = 0;

        while (std::getline(net_dev, dev_line)) {
            if (dev_line.find(':') != std::string::npos) {
                std::istringstream iss(dev_line);
                std::string iface;
                uint64_t rx_bytes, rx_packets, tx_bytes, tx_packets;
                // Skip other fields
                uint64_t dummy;

                iss >> iface >> rx_bytes >> rx_packets;
                for (int i = 0; i < 6; ++i) iss >> dummy; // Skip fields
                iss >> tx_bytes >> tx_packets;

                total_rx += rx_bytes;
                total_tx += tx_bytes;
            }
        }

        snapshot.network_rx_bytes_per_sec = total_rx - prev_rx_bytes;
        snapshot.network_tx_bytes_per_sec = total_tx - prev_tx_bytes;
        prev_rx_bytes = total_rx;
        prev_tx_bytes = total_tx;
    }
    #endif

    #ifdef _WIN32
    void collect_windows_metrics(ResourceSnapshot& snapshot) {
        // Collect Windows performance data
        PdhCollectQueryData(pdh_query_);

        PDH_FMT_COUNTERVALUE cpu_value, memory_value;
        PdhGetFormattedCounterValue(cpu_counter_, PDH_FMT_DOUBLE, NULL, &cpu_value);
        PdhGetFormattedCounterValue(memory_counter_, PDH_FMT_LARGE, NULL, &memory_value);

        snapshot.cpu_usage_percent = cpu_value.doubleValue;

        MEMORYSTATUSEX mem_status;
        mem_status.dwLength = sizeof(mem_status);
        GlobalMemoryStatusEx(&mem_status);

        snapshot.memory_total_bytes = mem_status.ullTotalPhys;
        snapshot.memory_used_bytes = mem_status.ullTotalPhys - mem_status.ullAvailPhys;
        snapshot.memory_usage_percent = mem_status.dwMemoryLoad;

        // Thread count
        SYSTEM_INFO sys_info;
        GetSystemInfo(&sys_info);
        snapshot.thread_count = sys_info.dwNumberOfProcessors; // Approximation
    }
    #endif
};

/**
 * Performance profiler with sampling and call stack analysis
 */
class PerformanceProfiler {
public:
    struct ProfileEntry {
        std::string function_name;
        std::string file_name;
        uint32_t line_number;
        uint64_t call_count;
        uint64_t total_time_ns;
        uint64_t min_time_ns;
        uint64_t max_time_ns;
        double avg_time_ns;
        std::chrono::time_point<std::chrono::high_resolution_clock> first_call;
        std::chrono::time_point<std::chrono::high_resolution_clock> last_call;
    };

private:
    lockfree::LockFreeHashMap<std::string, ProfileEntry> profile_data_;
    std::atomic<bool> profiling_enabled_{true};

public:
    class ScopedProfiler {
    private:
        PerformanceProfiler& profiler_;
        std::string key_;
        HighResolutionTimer timer_;

    public:
        ScopedProfiler(PerformanceProfiler& profiler, const std::string& function_name,
                      const std::string& file_name = "", uint32_t line_number = 0)
            : profiler_(profiler), key_(function_name + ":" + file_name + ":" + std::to_string(line_number)) {

            if (profiler_.profiling_enabled_.load(std::memory_order_acquire)) {
                timer_.start();
            }
        }

        ~ScopedProfiler() {
            if (profiler_.profiling_enabled_.load(std::memory_order_acquire)) {
                timer_.stop();
                profiler_.record_timing(key_, timer_.elapsed_nanoseconds());
            }
        }
    };

    void enable_profiling() { profiling_enabled_.store(true); }
    void disable_profiling() { profiling_enabled_.store(false); }
    bool is_profiling_enabled() const { return profiling_enabled_.load(); }

    void record_timing(const std::string& key, double elapsed_ns) {
        if (!profiling_enabled_.load(std::memory_order_acquire)) {
            return;
        }

        ProfileEntry entry;
        bool found = profile_data_.find(key, entry);

        if (found) {
            // Update existing entry
            entry.call_count++;
            entry.total_time_ns += static_cast<uint64_t>(elapsed_ns);
            entry.min_time_ns = std::min(entry.min_time_ns, static_cast<uint64_t>(elapsed_ns));
            entry.max_time_ns = std::max(entry.max_time_ns, static_cast<uint64_t>(elapsed_ns));
            entry.avg_time_ns = static_cast<double>(entry.total_time_ns) / entry.call_count;
            entry.last_call = std::chrono::high_resolution_clock::now();
        } else {
            // Create new entry
            auto now = std::chrono::high_resolution_clock::now();
            entry.call_count = 1;
            entry.total_time_ns = static_cast<uint64_t>(elapsed_ns);
            entry.min_time_ns = static_cast<uint64_t>(elapsed_ns);
            entry.max_time_ns = static_cast<uint64_t>(elapsed_ns);
            entry.avg_time_ns = elapsed_ns;
            entry.first_call = now;
            entry.last_call = now;

            // Parse key to extract function info
            size_t first_colon = key.find(':');
            size_t second_colon = key.find(':', first_colon + 1);

            entry.function_name = key.substr(0, first_colon);
            if (first_colon != std::string::npos && second_colon != std::string::npos) {
                entry.file_name = key.substr(first_colon + 1, second_colon - first_colon - 1);
                entry.line_number = std::stoul(key.substr(second_colon + 1));
            }
        }

        profile_data_.insert(key, entry);
    }

    void print_profile_report() const {
        printf("\n=== Performance Profile Report ===\n");
        printf("%-50s %10s %12s %12s %12s %12s\n",
               "Function", "Calls", "Total(ms)", "Avg(μs)", "Min(μs)", "Max(μs)");
        printf("%s\n", std::string(120, '-').c_str());

        // Collect all entries (simplified - full implementation would sort by total time)
        std::vector<std::pair<std::string, ProfileEntry>> entries;

        // Note: This is a simplified extraction since LockFreeHashMap doesn't have iterators
        // Full implementation would need a different approach

        for (const auto& [key, entry] : entries) {
            printf("%-50s %10lu %12.3f %12.3f %12.3f %12.3f\n",
                   entry.function_name.c_str(),
                   entry.call_count,
                   entry.total_time_ns / 1000000.0, // Convert to ms
                   entry.avg_time_ns / 1000.0,      // Convert to μs
                   entry.min_time_ns / 1000.0,      // Convert to μs
                   entry.max_time_ns / 1000.0);     // Convert to μs
        }
    }

    void reset_profile_data() {
        // Clear all profile data
        // Note: LockFreeHashMap doesn't have a clear method in this implementation
        // Full implementation would need this functionality
    }

    void save_profile_to_file(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            printf("Failed to open profile output file: %s\n", filename.c_str());
            return;
        }

        file << "Function,Calls,Total_ns,Avg_ns,Min_ns,Max_ns,File,Line\n";

        // Export profile data to CSV format
        // Implementation would iterate through all entries

        file.close();
        printf("Profile data saved to: %s\n", filename.c_str());
    }
};

/**
 * Real-time performance dashboard
 */
class PerformanceDashboard {
private:
    SystemResourceMonitor& resource_monitor_;
    PerformanceProfiler& profiler_;

    // Latency histograms for different operations
    LatencyHistogram<64> risk_calculation_latency_;
    LatencyHistogram<64> market_data_latency_;
    LatencyHistogram<64> trade_processing_latency_;
    LatencyHistogram<64> portfolio_update_latency_;

    // Throughput counters
    std::atomic<uint64_t> trades_processed_{0};
    std::atomic<uint64_t> risk_calculations_{0};
    std::atomic<uint64_t> market_data_updates_{0};
    std::atomic<uint64_t> portfolio_updates_{0};

    // Real-time display thread
    std::thread dashboard_thread_;
    std::atomic<bool> running_{false};

public:
    PerformanceDashboard(SystemResourceMonitor& resource_monitor,
                        PerformanceProfiler& profiler)
        : resource_monitor_(resource_monitor), profiler_(profiler) {}

    ~PerformanceDashboard() {
        stop();
    }

    void start() {
        running_.store(true);
        dashboard_thread_ = std::thread([this]() { dashboard_worker(); });
    }

    void stop() {
        running_.store(false);
        if (dashboard_thread_.joinable()) {
            dashboard_thread_.join();
        }
    }

    // Record latencies for different operations
    void record_risk_calculation_latency(double latency_us) {
        risk_calculation_latency_.record_latency(latency_us);
        risk_calculations_.fetch_add(1);
    }

    void record_market_data_latency(double latency_us) {
        market_data_latency_.record_latency(latency_us);
        market_data_updates_.fetch_add(1);
    }

    void record_trade_processing_latency(double latency_us) {
        trade_processing_latency_.record_latency(latency_us);
        trades_processed_.fetch_add(1);
    }

    void record_portfolio_update_latency(double latency_us) {
        portfolio_update_latency_.record_latency(latency_us);
        portfolio_updates_.fetch_add(1);
    }

    void print_dashboard() const {
        // Clear screen (Unix/Linux)
        printf("\033[2J\033[H");

        printf("═══════════════════════════════════════════════════════════════════════════════\n");
        printf("              HIGH-PERFORMANCE RISK ANALYTICS - REAL-TIME DASHBOARD            \n");
        printf("═══════════════════════════════════════════════════════════════════════════════\n");

        // System resources
        resource_monitor_.print_current_status();

        // Throughput metrics
        printf("\n=== Throughput Metrics ===\n");
        printf("Trades Processed:       %12lu\n", trades_processed_.load());
        printf("Risk Calculations:      %12lu\n", risk_calculations_.load());
        printf("Market Data Updates:    %12lu\n", market_data_updates_.load());
        printf("Portfolio Updates:      %12lu\n", portfolio_updates_.load());

        // Latency statistics
        printf("\n=== Latency Statistics (μs) ===\n");

        auto risk_stats = risk_calculation_latency_.get_statistics();
        printf("Risk Calculations:  Mean=%6.2f, P95=%6.2f, P99=%6.2f, Max=%6.2f\n",
               risk_stats.mean_latency_us, risk_stats.percentile_95,
               risk_stats.percentile_99, risk_stats.max_latency_us);

        auto market_stats = market_data_latency_.get_statistics();
        printf("Market Data:        Mean=%6.2f, P95=%6.2f, P99=%6.2f, Max=%6.2f\n",
               market_stats.mean_latency_us, market_stats.percentile_95,
               market_stats.percentile_99, market_stats.max_latency_us);

        auto trade_stats = trade_processing_latency_.get_statistics();
        printf("Trade Processing:   Mean=%6.2f, P95=%6.2f, P99=%6.2f, Max=%6.2f\n",
               trade_stats.mean_latency_us, trade_stats.percentile_95,
               trade_stats.percentile_99, trade_stats.max_latency_us);

        auto portfolio_stats = portfolio_update_latency_.get_statistics();
        printf("Portfolio Updates:  Mean=%6.2f, P95=%6.2f, P99=%6.2f, Max=%6.2f\n",
               portfolio_stats.mean_latency_us, portfolio_stats.percentile_95,
               portfolio_stats.percentile_99, portfolio_stats.max_latency_us);

        printf("\n=== Performance Targets vs Actual ===\n");
        printf("Operation             Target     Actual     Status\n");
        printf("─────────────────────────────────────────────────────\n");

        auto check_target = [](double actual, double target, const char* name) {
            const char* status = actual <= target ? "✓ PASS" : "✗ FAIL";
            printf("%-20s %8.2f %10.2f   %s\n", name, target, actual, status);
        };

        check_target(risk_stats.percentile_95, 10000.0, "Risk Calc P95 (μs)");
        check_target(market_stats.percentile_99, 500.0, "Market Data P99 (μs)");
        check_target(trade_stats.percentile_95, 1000.0, "Trade Proc P95 (μs)");
        check_target(portfolio_stats.mean_latency_us, 100.0, "Portfolio Mean (μs)");

        printf("\n═══════════════════════════════════════════════════════════════════════════════\n");
        printf("Last Updated: %s", std::ctime(&(std::time_t){std::time(nullptr)}));
    }

    void generate_performance_report() const {
        printf("\n=== COMPREHENSIVE PERFORMANCE REPORT ===\n");

        // System summary
        auto resource_stats = resource_monitor_.get_statistics();
        printf("\nSystem Resource Summary:\n");
        printf("  Average CPU Usage:     %6.2f%%\n", resource_stats.avg_cpu_usage);
        printf("  Peak CPU Usage:        %6.2f%%\n", resource_stats.max_cpu_usage);
        printf("  Average Memory Usage:  %6.2f%%\n", resource_stats.avg_memory_usage);
        printf("  Peak Memory Usage:     %6.2f%% (%lu MB)\n",
               resource_stats.max_memory_usage,
               resource_stats.peak_memory_bytes / (1024 * 1024));

        // Detailed latency histograms
        printf("\nRisk Calculation Latency Distribution:\n");
        risk_calculation_latency_.print_histogram();

        printf("\nMarket Data Processing Latency Distribution:\n");
        market_data_latency_.print_histogram();

        printf("\nTrade Processing Latency Distribution:\n");
        trade_processing_latency_.print_histogram();

        printf("\nPortfolio Update Latency Distribution:\n");
        portfolio_update_latency_.print_histogram();

        // Profiler report
        profiler_.print_profile_report();
    }

private:
    void dashboard_worker() {
        while (running_.load()) {
            print_dashboard();
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
};

// Convenience macros for profiling
#define PROFILE_FUNCTION(profiler) \
    PerformanceProfiler::ScopedProfiler _prof(profiler, __FUNCTION__, __FILE__, __LINE__)

#define PROFILE_SCOPE(profiler, name) \
    PerformanceProfiler::ScopedProfiler _prof(profiler, name, __FILE__, __LINE__)

#define RECORD_LATENCY(histogram, timer) \
    do { \
        timer.stop(); \
        histogram.record_latency(timer.elapsed_microseconds()); \
    } while(0)

} // namespace monitoring
} // namespace hpc
} // namespace risk_analytics