#pragma once

#include "types.hpp"
#include <unordered_map>
#include <shared_mutex>
#include <memory>
#include <vector>
#include <functional>
#include <atomic>
#include <thread>
#include <queue>

namespace risk_engine {

struct InstrumentKeyHash {
    std::size_t operator()(const InstrumentKey& key) const noexcept {
        std::size_t h1 = std::hash<std::string>{}(key.symbol);
        std::size_t h2 = std::hash<uint8_t>{}(static_cast<uint8_t>(key.asset_type));
        std::size_t h3 = std::hash<std::string>{}(key.currency);
        std::size_t h4 = std::hash<double>{}(key.strike);
        std::size_t h5 = std::hash<std::chrono::nanoseconds>{}(key.expiry.time_since_epoch());
        std::size_t h6 = std::hash<uint8_t>{}(static_cast<uint8_t>(key.option_type));

        return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3) ^ (h5 << 4) ^ (h6 << 5);
    }
};

class PositionEngine {
public:
    using PositionMap = std::unordered_map<InstrumentKey, std::shared_ptr<Position>, InstrumentKeyHash>;
    using PositionCallback = std::function<void(const Position&)>;
    using TradeCallback = std::function<void(const Trade&)>;

    PositionEngine();
    ~PositionEngine();

    // Core position management
    void add_trade(const Trade& trade);
    void update_market_price(const Symbol& symbol, Price price, Timestamp timestamp = std::chrono::high_resolution_clock::now());
    void update_market_data(const MarketData& market_data);

    // Position queries
    std::shared_ptr<Position> get_position(const InstrumentKey& key) const;
    std::vector<std::shared_ptr<Position>> get_all_positions() const;
    std::vector<std::shared_ptr<Position>> get_positions_by_symbol(const Symbol& symbol) const;
    std::vector<std::shared_ptr<Position>> get_positions_by_asset_type(AssetType type) const;

    // Portfolio aggregation
    Price calculate_total_pnl() const;
    Price calculate_total_portfolio_value() const;
    Price calculate_portfolio_delta() const;
    std::unordered_map<Currency, Price> calculate_currency_exposure() const;

    // Risk calculations
    Price calculate_var(double confidence_level = 0.95, int horizon_days = 1) const;
    Price calculate_expected_shortfall(double confidence_level = 0.95) const;

    // Real-time monitoring
    void register_position_callback(PositionCallback callback);
    void register_trade_callback(TradeCallback callback);
    void start_monitoring();
    void stop_monitoring();

    // Performance monitoring
    struct Statistics {
        std::atomic<uint64_t> trades_processed{0};
        std::atomic<uint64_t> position_updates{0};
        std::atomic<uint64_t> price_updates{0};
        std::atomic<double> avg_processing_time_us{0.0};
        std::atomic<double> max_processing_time_us{0.0};
    };

    const Statistics& get_statistics() const { return stats_; }
    void reset_statistics();

    // Memory management
    void compact_positions();
    size_t get_memory_usage() const;

private:
    mutable std::shared_mutex positions_mutex_;
    PositionMap positions_;

    // Price data
    mutable std::shared_mutex prices_mutex_;
    std::unordered_map<Symbol, MarketData> market_data_;

    // Callbacks
    std::vector<PositionCallback> position_callbacks_;
    std::vector<TradeCallback> trade_callbacks_;
    mutable std::shared_mutex callbacks_mutex_;

    // Performance monitoring
    Statistics stats_;

    // Background processing
    std::atomic<bool> monitoring_active_{false};
    std::thread monitoring_thread_;

    // Internal methods
    void update_position_pnl(Position& position) const;
    void notify_position_callbacks(const Position& position);
    void notify_trade_callbacks(const Trade& trade);
    void run_monitoring_loop();

    // High-frequency operations helpers
    inline void record_processing_time(double time_us);
    inline Price get_current_price(const Symbol& symbol) const;
};

// Lock-free position update queue for high-frequency updates
template<typename T, size_t Size = 1024 * 1024>
class LockFreeQueue {
public:
    LockFreeQueue() : head_(0), tail_(0) {
        static_assert((Size & (Size - 1)) == 0, "Size must be power of 2");
    }

    bool push(const T& item) {
        const auto current_tail = tail_.load(std::memory_order_relaxed);
        const auto next_tail = increment(current_tail);

        if (next_tail == head_.load(std::memory_order_acquire)) {
            return false; // Queue is full
        }

        buffer_[current_tail] = item;
        tail_.store(next_tail, std::memory_order_release);
        return true;
    }

    bool pop(T& item) {
        const auto current_head = head_.load(std::memory_order_relaxed);

        if (current_head == tail_.load(std::memory_order_acquire)) {
            return false; // Queue is empty
        }

        item = buffer_[current_head];
        head_.store(increment(current_head), std::memory_order_release);
        return true;
    }

    bool empty() const {
        return head_.load(std::memory_order_acquire) == tail_.load(std::memory_order_acquire);
    }

    size_t size() const {
        const auto head = head_.load(std::memory_order_acquire);
        const auto tail = tail_.load(std::memory_order_acquire);
        return (tail - head) & (Size - 1);
    }

private:
    alignas(64) std::atomic<size_t> head_;
    alignas(64) std::atomic<size_t> tail_;
    std::array<T, Size> buffer_;

    size_t increment(size_t idx) const {
        return (idx + 1) & (Size - 1);
    }
};

// High-performance position aggregator
class PositionAggregator {
public:
    struct AggregatedPosition {
        Currency currency;
        AssetType asset_type;
        Quantity total_quantity = 0;
        Price total_notional = 0.0;
        Price total_pnl = 0.0;
        Price net_delta = 0.0;
        Price net_gamma = 0.0;
        Price net_vega = 0.0;
        int position_count = 0;
        Timestamp last_update{};
    };

    using AggregationKey = std::pair<Currency, AssetType>;

    explicit PositionAggregator(const PositionEngine& engine);

    void update_aggregations();
    std::vector<AggregatedPosition> get_aggregated_positions() const;
    AggregatedPosition get_aggregation(const Currency& currency, AssetType asset_type) const;

    // Real-time aggregation updates
    void on_position_update(const Position& position);

private:
    const PositionEngine& position_engine_;
    mutable std::shared_mutex aggregations_mutex_;
    std::unordered_map<AggregationKey, AggregatedPosition,
                      boost::hash<AggregationKey>> aggregations_;
};

} // namespace risk_engine