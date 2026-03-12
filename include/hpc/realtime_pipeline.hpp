/**
 * Real-Time Data Processing Pipeline
 *
 * Event-driven architecture for high-throughput market data processing
 * and real-time risk analytics with sub-millisecond latency
 */

#pragma once

#include "hpc_framework.hpp"
#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <variant>

namespace risk_analytics {
namespace hpc {
namespace realtime {

// High-resolution timestamp for latency measurement
using Timestamp = std::chrono::time_point<std::chrono::high_resolution_clock>;
using Duration = std::chrono::nanoseconds;

/**
 * Event types for the processing pipeline
 */
enum class EventType : uint8_t {
    MARKET_DATA_UPDATE = 1,
    POSITION_UPDATE = 2,
    PORTFOLIO_REBALANCE = 3,
    RISK_LIMIT_BREACH = 4,
    TRADE_EXECUTION = 5,
    HEARTBEAT = 6,
    SYSTEM_ALERT = 7
};

/**
 * Base event structure with timestamp and metadata
 */
struct alignas(64) BaseEvent {
    EventType type;
    uint64_t sequence_number;
    Timestamp timestamp;
    uint32_t source_id;
    uint16_t priority;  // 0 = highest priority
    uint16_t flags;     // Bit flags for event properties

    BaseEvent(EventType t, uint32_t src_id = 0, uint16_t prio = 5)
        : type(t), sequence_number(0), timestamp(std::chrono::high_resolution_clock::now()),
          source_id(src_id), priority(prio), flags(0) {}
};

/**
 * Market data update event
 */
struct alignas(64) MarketDataEvent : public BaseEvent {
    char symbol[16];        // Instrument symbol
    double bid_price;       // Best bid price
    double ask_price;       // Best ask price
    double last_price;      // Last traded price
    uint64_t bid_size;      // Bid quantity
    uint64_t ask_size;      // Ask quantity
    uint64_t last_size;     // Last trade size
    double volume;          // Total volume
    double vwap;           // Volume weighted average price
    double volatility;      // Implied volatility (for options)

    MarketDataEvent(const char* sym, double bid, double ask, double last)
        : BaseEvent(EventType::MARKET_DATA_UPDATE), bid_price(bid), ask_price(ask), last_price(last),
          bid_size(0), ask_size(0), last_size(0), volume(0), vwap(0), volatility(0) {
        std::strncpy(symbol, sym, sizeof(symbol) - 1);
        symbol[sizeof(symbol) - 1] = '\0';
    }
};

/**
 * Position update event
 */
struct alignas(64) PositionUpdateEvent : public BaseEvent {
    char symbol[16];        // Instrument symbol
    double quantity;        // Position size (signed)
    double price;          // Entry/update price
    double market_value;   // Current market value
    double unrealized_pnl; // Unrealized P&L
    double delta;          // Risk delta
    double gamma;          // Risk gamma
    uint32_t account_id;   // Account identifier

    PositionUpdateEvent(const char* sym, double qty, double prc)
        : BaseEvent(EventType::POSITION_UPDATE), quantity(qty), price(prc),
          market_value(qty * prc), unrealized_pnl(0), delta(0), gamma(0), account_id(0) {
        std::strncpy(symbol, sym, sizeof(symbol) - 1);
        symbol[sizeof(symbol) - 1] = '\0';
    }
};

/**
 * Risk limit breach event
 */
struct alignas(64) RiskLimitEvent : public BaseEvent {
    enum class LimitType : uint8_t {
        VAR_LIMIT = 1,
        POSITION_LIMIT = 2,
        CONCENTRATION_LIMIT = 3,
        DELTA_LIMIT = 4,
        GAMMA_LIMIT = 5,
        LIQUIDITY_LIMIT = 6
    };

    LimitType limit_type;
    double current_value;
    double limit_value;
    double breach_percentage;
    char description[64];
    uint32_t portfolio_id;

    RiskLimitEvent(LimitType type, double current, double limit)
        : BaseEvent(EventType::RISK_LIMIT_BREACH), limit_type(type),
          current_value(current), limit_value(limit),
          breach_percentage((current - limit) / limit * 100.0), portfolio_id(0) {
        description[0] = '\0';
    }
};

/**
 * Event variant for type-safe event handling
 */
using Event = std::variant<MarketDataEvent, PositionUpdateEvent, RiskLimitEvent>;

/**
 * Event handler interface
 */
class EventHandler {
public:
    virtual ~EventHandler() = default;
    virtual void handle_event(const Event& event, Timestamp processing_time) = 0;
    virtual const char* get_handler_name() const = 0;
    virtual uint16_t get_priority() const { return 5; } // Default priority
};

/**
 * High-performance event bus with lock-free queues
 */
template<size_t QueueSize = 1048576> // 1M events
class EventBus {
private:
    struct EventWrapper {
        Event event;
        Timestamp enqueue_time;
        uint64_t sequence_number;
    };

    // Lock-free ring buffers for different priority levels
    LockFreeRingBuffer<EventWrapper, QueueSize> high_priority_queue_;
    LockFreeRingBuffer<EventWrapper, QueueSize> normal_priority_queue_;
    LockFreeRingBuffer<EventWrapper, QueueSize> low_priority_queue_;

    // Event handlers organized by event type
    std::unordered_map<EventType, std::vector<std::shared_ptr<EventHandler>>> handlers_;
    std::shared_mutex handlers_mutex_;

    // Processing threads
    std::vector<std::thread> processing_threads_;
    std::atomic<bool> running_{false};

    // Performance metrics
    std::atomic<uint64_t> events_processed_{0};
    std::atomic<uint64_t> events_dropped_{0};
    std::atomic<uint64_t> total_latency_ns_{0};
    std::atomic<uint64_t> max_latency_ns_{0};

    // Sequence number generator
    std::atomic<uint64_t> sequence_counter_{0};

public:
    EventBus(size_t num_processing_threads = std::thread::hardware_concurrency())
        : running_(true) {

        // Start processing threads
        for (size_t i = 0; i < num_processing_threads; ++i) {
            processing_threads_.emplace_back([this, i]() {
                process_events_worker(i);
            });

            // Set thread affinity for better cache locality
            #ifdef __linux__
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(i % std::thread::hardware_concurrency(), &cpuset);
            pthread_setaffinity_np(processing_threads_[i].native_handle(),
                                  sizeof(cpu_set_t), &cpuset);
            #endif
        }
    }

    ~EventBus() {
        running_.store(false);
        for (auto& thread : processing_threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }

    /**
     * Publish event to the bus (lock-free)
     */
    bool publish(const Event& event) {
        const uint64_t sequence = sequence_counter_.fetch_add(1, std::memory_order_relaxed);
        const Timestamp now = std::chrono::high_resolution_clock::now();

        EventWrapper wrapper{event, now, sequence};

        // Determine priority based on event type
        uint16_t priority = get_event_priority(event);

        if (priority <= 2) {
            return high_priority_queue_.push(wrapper);
        } else if (priority <= 5) {
            return normal_priority_queue_.push(wrapper);
        } else {
            return low_priority_queue_.push(wrapper);
        }
    }

    /**
     * Subscribe handler to specific event types
     */
    void subscribe(EventType event_type, std::shared_ptr<EventHandler> handler) {
        std::unique_lock<std::shared_mutex> lock(handlers_mutex_);
        handlers_[event_type].push_back(std::move(handler));
    }

    /**
     * Unsubscribe handler
     */
    void unsubscribe(EventType event_type, std::shared_ptr<EventHandler> handler) {
        std::unique_lock<std::shared_mutex> lock(handlers_mutex_);
        auto& handlers = handlers_[event_type];
        handlers.erase(std::remove(handlers.begin(), handlers.end(), handler), handlers.end());
    }

    /**
     * Get performance statistics
     */
    struct Statistics {
        uint64_t events_processed;
        uint64_t events_dropped;
        double average_latency_us;
        double max_latency_us;
        double throughput_events_per_sec;
        size_t high_priority_queue_size;
        size_t normal_priority_queue_size;
        size_t low_priority_queue_size;
    };

    Statistics get_statistics() const {
        const uint64_t processed = events_processed_.load();
        const uint64_t dropped = events_dropped_.load();
        const uint64_t total_latency = total_latency_ns_.load();
        const uint64_t max_latency = max_latency_ns_.load();

        return Statistics{
            .events_processed = processed,
            .events_dropped = dropped,
            .average_latency_us = processed > 0 ? (total_latency / processed) / 1000.0 : 0.0,
            .max_latency_us = max_latency / 1000.0,
            .throughput_events_per_sec = 0.0, // Would need time-based calculation
            .high_priority_queue_size = high_priority_queue_.size(),
            .normal_priority_queue_size = normal_priority_queue_.size(),
            .low_priority_queue_size = low_priority_queue_.size()
        };
    }

    /**
     * Reset performance counters
     */
    void reset_statistics() {
        events_processed_.store(0);
        events_dropped_.store(0);
        total_latency_ns_.store(0);
        max_latency_ns_.store(0);
    }

private:
    void process_events_worker(size_t worker_id) {
        EventWrapper wrapper;

        // Set thread name for debugging
        #ifdef __linux__
        char thread_name[16];
        snprintf(thread_name, sizeof(thread_name), "EventBus-%zu", worker_id);
        pthread_setname_np(pthread_self(), thread_name);
        #endif

        while (running_.load(std::memory_order_acquire)) {
            bool found_event = false;

            // Process high priority events first
            if (high_priority_queue_.pop(wrapper)) {
                found_event = true;
            }
            // Then normal priority
            else if (normal_priority_queue_.pop(wrapper)) {
                found_event = true;
            }
            // Finally low priority
            else if (low_priority_queue_.pop(wrapper)) {
                found_event = true;
            }

            if (found_event) {
                process_event(wrapper);
            } else {
                // No events available, yield CPU
                std::this_thread::yield();
            }
        }
    }

    void process_event(const EventWrapper& wrapper) {
        const Timestamp processing_start = std::chrono::high_resolution_clock::now();

        // Get event type
        EventType event_type = std::visit([](const auto& event) {
            return event.type;
        }, wrapper.event);

        // Find handlers for this event type
        std::shared_lock<std::shared_mutex> lock(handlers_mutex_);
        auto it = handlers_.find(event_type);
        if (it != handlers_.end()) {
            // Call all registered handlers
            for (auto& handler : it->second) {
                try {
                    handler->handle_event(wrapper.event, processing_start);
                } catch (const std::exception& e) {
                    // Log error but continue processing
                    fprintf(stderr, "Event handler error: %s\n", e.what());
                }
            }
        }

        // Update performance metrics
        const Timestamp processing_end = std::chrono::high_resolution_clock::now();
        const auto latency = std::chrono::duration_cast<Duration>(
            processing_end - wrapper.enqueue_time).count();

        events_processed_.fetch_add(1, std::memory_order_relaxed);
        total_latency_ns_.fetch_add(latency, std::memory_order_relaxed);

        // Update max latency (compare-and-swap loop)
        uint64_t current_max = max_latency_ns_.load(std::memory_order_relaxed);
        while (latency > current_max &&
               !max_latency_ns_.compare_exchange_weak(current_max, latency,
                                                     std::memory_order_relaxed)) {
            // Loop until successful update
        }
    }

    uint16_t get_event_priority(const Event& event) const {
        return std::visit([](const auto& e) -> uint16_t {
            return e.priority;
        }, event);
    }
};

/**
 * Market data feed processor with ultra-low latency
 */
class MarketDataProcessor : public EventHandler {
private:
    // Lock-free hash map for fast symbol lookups
    struct SymbolData {
        alignas(64) std::atomic<double> last_price{0.0};
        alignas(64) std::atomic<double> bid_price{0.0};
        alignas(64) std::atomic<double> ask_price{0.0};
        alignas(64) std::atomic<uint64_t> last_update_time{0};
        alignas(64) std::atomic<double> volatility{0.0};
    };

    static constexpr size_t MAX_SYMBOLS = 65536;
    std::array<SymbolData, MAX_SYMBOLS> symbol_data_;
    std::unordered_map<std::string, uint32_t> symbol_index_;
    std::shared_mutex symbol_mutex_;

    // Price change callbacks
    std::vector<std::function<void(const std::string&, double, double)>> price_change_callbacks_;

    // Performance tracking
    std::atomic<uint64_t> updates_processed_{0};
    std::atomic<uint64_t> price_changes_detected_{0};

public:
    MarketDataProcessor() {
        // Initialize symbol data
        for (auto& data : symbol_data_) {
            data.last_price.store(0.0, std::memory_order_relaxed);
            data.bid_price.store(0.0, std::memory_order_relaxed);
            data.ask_price.store(0.0, std::memory_order_relaxed);
            data.last_update_time.store(0, std::memory_order_relaxed);
            data.volatility.store(0.0, std::memory_order_relaxed);
        }
    }

    void handle_event(const Event& event, Timestamp processing_time) override {
        if (const auto* market_event = std::get_if<MarketDataEvent>(&event)) {
            process_market_data(*market_event, processing_time);
        }
    }

    const char* get_handler_name() const override {
        return "MarketDataProcessor";
    }

    uint16_t get_priority() const override {
        return 1; // Highest priority for market data
    }

    /**
     * Get current price for symbol (lock-free read)
     */
    double get_current_price(const std::string& symbol) const {
        uint32_t index = get_symbol_index(symbol);
        if (index < MAX_SYMBOLS) {
            return symbol_data_[index].last_price.load(std::memory_order_acquire);
        }
        return 0.0;
    }

    /**
     * Get bid-ask spread
     */
    std::pair<double, double> get_bid_ask(const std::string& symbol) const {
        uint32_t index = get_symbol_index(symbol);
        if (index < MAX_SYMBOLS) {
            const auto& data = symbol_data_[index];
            return {
                data.bid_price.load(std::memory_order_acquire),
                data.ask_price.load(std::memory_order_acquire)
            };
        }
        return {0.0, 0.0};
    }

    /**
     * Register callback for price changes
     */
    void register_price_change_callback(
        std::function<void(const std::string&, double, double)> callback) {
        price_change_callbacks_.push_back(std::move(callback));
    }

    /**
     * Get processing statistics
     */
    struct ProcessorStats {
        uint64_t updates_processed;
        uint64_t price_changes_detected;
        size_t symbols_tracked;
        double change_detection_rate;
    };

    ProcessorStats get_statistics() const {
        const uint64_t updates = updates_processed_.load();
        const uint64_t changes = price_changes_detected_.load();

        return ProcessorStats{
            .updates_processed = updates,
            .price_changes_detected = changes,
            .symbols_tracked = symbol_index_.size(),
            .change_detection_rate = updates > 0 ? double(changes) / updates : 0.0
        };
    }

private:
    void process_market_data(const MarketDataEvent& event, Timestamp processing_time) {
        const std::string symbol(event.symbol);
        const uint32_t index = get_or_create_symbol_index(symbol);

        if (index >= MAX_SYMBOLS) {
            return; // Symbol table full
        }

        auto& data = symbol_data_[index];

        // Get previous price for change detection
        const double prev_price = data.last_price.load(std::memory_order_acquire);

        // Update prices atomically
        data.bid_price.store(event.bid_price, std::memory_order_release);
        data.ask_price.store(event.ask_price, std::memory_order_release);
        data.last_price.store(event.last_price, std::memory_order_release);
        data.volatility.store(event.volatility, std::memory_order_release);

        // Update timestamp
        const uint64_t timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
            processing_time.time_since_epoch()).count();
        data.last_update_time.store(timestamp, std::memory_order_release);

        // Detect significant price changes
        if (prev_price > 0.0 && std::abs(event.last_price - prev_price) / prev_price > 0.001) {
            price_changes_detected_.fetch_add(1, std::memory_order_relaxed);

            // Notify callbacks
            for (const auto& callback : price_change_callbacks_) {
                callback(symbol, prev_price, event.last_price);
            }
        }

        updates_processed_.fetch_add(1, std::memory_order_relaxed);
    }

    uint32_t get_symbol_index(const std::string& symbol) const {
        std::shared_lock<std::shared_mutex> lock(symbol_mutex_);
        auto it = symbol_index_.find(symbol);
        return (it != symbol_index_.end()) ? it->second : MAX_SYMBOLS;
    }

    uint32_t get_or_create_symbol_index(const std::string& symbol) {
        {
            std::shared_lock<std::shared_mutex> lock(symbol_mutex_);
            auto it = symbol_index_.find(symbol);
            if (it != symbol_index_.end()) {
                return it->second;
            }
        }

        // Need to create new index
        std::unique_lock<std::shared_mutex> lock(symbol_mutex_);

        // Double-check after acquiring write lock
        auto it = symbol_index_.find(symbol);
        if (it != symbol_index_.end()) {
            return it->second;
        }

        // Create new index
        const uint32_t new_index = symbol_index_.size();
        if (new_index < MAX_SYMBOLS) {
            symbol_index_[symbol] = new_index;
            return new_index;
        }

        return MAX_SYMBOLS; // Table full
    }
};

/**
 * Real-time risk monitor with incremental calculations
 */
class RealTimeRiskMonitor : public EventHandler {
private:
    struct PortfolioRiskState {
        alignas(64) std::atomic<double> portfolio_value{0.0};
        alignas(64) std::atomic<double> var_95{0.0};
        alignas(64) std::atomic<double> portfolio_delta{0.0};
        alignas(64) std::atomic<double> portfolio_gamma{0.0};
        alignas(64) std::atomic<uint64_t> last_update_time{0};
    };

    PortfolioRiskState risk_state_;
    MarketDataProcessor& market_data_processor_;

    // Risk limits
    double var_limit_;
    double delta_limit_;
    double concentration_limit_;

    // Incremental calculation state
    std::unordered_map<std::string, double> position_deltas_;
    std::unordered_map<std::string, double> position_gammas_;
    std::shared_mutex positions_mutex_;

    // Event bus for risk alerts
    EventBus<>& event_bus_;

public:
    RealTimeRiskMonitor(MarketDataProcessor& market_processor,
                       EventBus<>& bus,
                       double var_limit = 0.02,
                       double delta_limit = 1000000.0)
        : market_data_processor_(market_processor), event_bus_(bus),
          var_limit_(var_limit), delta_limit_(delta_limit), concentration_limit_(0.1) {

        // Register for price change notifications
        market_data_processor_.register_price_change_callback(
            [this](const std::string& symbol, double old_price, double new_price) {
                update_incremental_risk(symbol, old_price, new_price);
            }
        );
    }

    void handle_event(const Event& event, Timestamp processing_time) override {
        if (const auto* position_event = std::get_if<PositionUpdateEvent>(&event)) {
            process_position_update(*position_event, processing_time);
        }
    }

    const char* get_handler_name() const override {
        return "RealTimeRiskMonitor";
    }

    uint16_t get_priority() const override {
        return 2; // High priority for risk monitoring
    }

    /**
     * Get current portfolio risk metrics
     */
    struct RiskMetrics {
        double portfolio_value;
        double var_95;
        double portfolio_delta;
        double portfolio_gamma;
        uint64_t last_update_time;
    };

    RiskMetrics get_risk_metrics() const {
        return RiskMetrics{
            .portfolio_value = risk_state_.portfolio_value.load(std::memory_order_acquire),
            .var_95 = risk_state_.var_95.load(std::memory_order_acquire),
            .portfolio_delta = risk_state_.portfolio_delta.load(std::memory_order_acquire),
            .portfolio_gamma = risk_state_.portfolio_gamma.load(std::memory_order_acquire),
            .last_update_time = risk_state_.last_update_time.load(std::memory_order_acquire)
        };
    }

private:
    void process_position_update(const PositionUpdateEvent& event, Timestamp processing_time) {
        const std::string symbol(event.symbol);

        // Update position Greeks
        {
            std::unique_lock<std::shared_mutex> lock(positions_mutex_);
            position_deltas_[symbol] = event.delta;
            position_gammas_[symbol] = event.gamma;
        }

        // Recalculate portfolio-level risk metrics
        recalculate_portfolio_risk();

        // Check risk limits
        check_risk_limits();

        // Update timestamp
        const uint64_t timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
            processing_time.time_since_epoch()).count();
        risk_state_.last_update_time.store(timestamp, std::memory_order_release);
    }

    void update_incremental_risk(const std::string& symbol, double old_price, double new_price) {
        // Incremental delta P&L calculation
        std::shared_lock<std::shared_mutex> lock(positions_mutex_);
        auto delta_it = position_deltas_.find(symbol);
        auto gamma_it = position_gammas_.find(symbol);

        if (delta_it != position_deltas_.end()) {
            const double price_change = new_price - old_price;
            const double delta = delta_it->second;

            // First-order delta P&L
            const double delta_pnl = delta * price_change;

            // Second-order gamma P&L
            double gamma_pnl = 0.0;
            if (gamma_it != position_gammas_.end()) {
                const double gamma = gamma_it->second;
                gamma_pnl = 0.5 * gamma * price_change * price_change;
            }

            // Update portfolio value incrementally
            const double total_pnl = delta_pnl + gamma_pnl;
            double current_value = risk_state_.portfolio_value.load(std::memory_order_acquire);
            risk_state_.portfolio_value.store(current_value + total_pnl, std::memory_order_release);
        }
    }

    void recalculate_portfolio_risk() {
        double total_delta = 0.0;
        double total_gamma = 0.0;

        {
            std::shared_lock<std::shared_mutex> lock(positions_mutex_);
            for (const auto& [symbol, delta] : position_deltas_) {
                total_delta += delta;
            }
            for (const auto& [symbol, gamma] : position_gammas_) {
                total_gamma += gamma;
            }
        }

        risk_state_.portfolio_delta.store(total_delta, std::memory_order_release);
        risk_state_.portfolio_gamma.store(total_gamma, std::memory_order_release);

        // Simplified VaR calculation (would use more sophisticated model in production)
        const double portfolio_volatility = 0.02; // 2% daily volatility assumption
        const double normal_quantile = 1.645; // 95% confidence level
        const double portfolio_value = risk_state_.portfolio_value.load(std::memory_order_acquire);
        const double var_95 = portfolio_value * portfolio_volatility * normal_quantile;

        risk_state_.var_95.store(var_95, std::memory_order_release);
    }

    void check_risk_limits() {
        const auto metrics = get_risk_metrics();

        // Check VaR limit
        if (metrics.var_95 > var_limit_ * metrics.portfolio_value) {
            RiskLimitEvent limit_event(RiskLimitEvent::LimitType::VAR_LIMIT,
                                     metrics.var_95,
                                     var_limit_ * metrics.portfolio_value);
            event_bus_.publish(limit_event);
        }

        // Check delta limit
        if (std::abs(metrics.portfolio_delta) > delta_limit_) {
            RiskLimitEvent limit_event(RiskLimitEvent::LimitType::DELTA_LIMIT,
                                     std::abs(metrics.portfolio_delta),
                                     delta_limit_);
            event_bus_.publish(limit_event);
        }
    }
};

/**
 * Ultra-low latency data feed interface
 */
class UltraLowLatencyFeed {
private:
    EventBus<>& event_bus_;
    std::atomic<bool> running_{false};
    std::thread feed_thread_;

    // UDP socket for market data
    int udp_socket_;
    struct sockaddr_in server_addr_;

    // Performance metrics
    std::atomic<uint64_t> packets_received_{0};
    std::atomic<uint64_t> packets_processed_{0};
    std::atomic<uint64_t> total_latency_ns_{0};

public:
    UltraLowLatencyFeed(EventBus<>& bus, const std::string& multicast_addr, uint16_t port)
        : event_bus_(bus) {

        // Initialize UDP socket for multicast
        initialize_udp_socket(multicast_addr, port);
    }

    ~UltraLowLatencyFeed() {
        stop();
    }

    void start() {
        running_.store(true);
        feed_thread_ = std::thread([this]() { feed_worker(); });

        // Set high priority for feed thread
        #ifdef __linux__
        struct sched_param param;
        param.sched_priority = 50;
        pthread_setschedparam(feed_thread_.native_handle(), SCHED_FIFO, &param);
        #endif
    }

    void stop() {
        running_.store(false);
        if (feed_thread_.joinable()) {
            feed_thread_.join();
        }
        close(udp_socket_);
    }

    struct FeedStats {
        uint64_t packets_received;
        uint64_t packets_processed;
        double packet_loss_rate;
        double average_latency_us;
    };

    FeedStats get_statistics() const {
        const uint64_t received = packets_received_.load();
        const uint64_t processed = packets_processed_.load();
        const uint64_t total_latency = total_latency_ns_.load();

        return FeedStats{
            .packets_received = received,
            .packets_processed = processed,
            .packet_loss_rate = received > 0 ? 1.0 - double(processed) / received : 0.0,
            .average_latency_us = processed > 0 ? (total_latency / processed) / 1000.0 : 0.0
        };
    }

private:
    void initialize_udp_socket(const std::string& multicast_addr, uint16_t port) {
        // Socket initialization for multicast UDP
        // Implementation would include proper multicast setup
    }

    void feed_worker() {
        char buffer[4096];

        while (running_.load(std::memory_order_acquire)) {
            // Receive UDP packet
            ssize_t bytes_received = recv(udp_socket_, buffer, sizeof(buffer), MSG_DONTWAIT);

            if (bytes_received > 0) {
                const Timestamp receive_time = std::chrono::high_resolution_clock::now();
                packets_received_.fetch_add(1, std::memory_order_relaxed);

                // Parse market data message
                if (auto market_event = parse_market_data_message(buffer, bytes_received)) {
                    // Publish to event bus
                    if (event_bus_.publish(*market_event)) {
                        packets_processed_.fetch_add(1, std::memory_order_relaxed);

                        // Calculate latency (assuming timestamp in message)
                        const auto latency = std::chrono::duration_cast<Duration>(
                            receive_time - market_event->timestamp).count();
                        total_latency_ns_.fetch_add(latency, std::memory_order_relaxed);
                    }
                }
            } else if (errno != EAGAIN && errno != EWOULDBLOCK) {
                // Handle error
                break;
            }
        }
    }

    std::optional<MarketDataEvent> parse_market_data_message(const char* buffer, size_t size) {
        // Message parsing implementation
        // This would parse binary market data format
        return std::nullopt; // Placeholder
    }
};

} // namespace realtime
} // namespace hpc
} // namespace risk_analytics