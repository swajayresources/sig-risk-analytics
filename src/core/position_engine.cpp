#include "core/position_engine.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <chrono>

namespace risk_engine {

PositionEngine::PositionEngine() {
    positions_.reserve(MAX_POSITIONS);
}

PositionEngine::~PositionEngine() {
    stop_monitoring();
}

void PositionEngine::add_trade(const Trade& trade) {
    auto start_time = std::chrono::high_resolution_clock::now();

    {
        std::unique_lock<std::shared_mutex> lock(positions_mutex_);

        auto it = positions_.find(trade.instrument);
        if (it == positions_.end()) {
            auto position = std::make_shared<Position>(trade.instrument);
            positions_[trade.instrument] = position;
            it = positions_.find(trade.instrument);
        }

        auto& position = *it->second;

        // Calculate new average price and quantity
        Quantity old_qty = position.quantity.load();
        Price old_avg_price = position.avg_price.load();
        Price old_realized_pnl = position.realized_pnl.load();

        Quantity trade_qty = (trade.side == Side::BUY) ? trade.quantity : -trade.quantity;
        Quantity new_qty = old_qty + trade_qty;

        Price new_avg_price = old_avg_price;
        Price realized_pnl_delta = 0.0;

        if (new_qty == 0) {
            // Position closed - realize P&L
            realized_pnl_delta = (trade.price - old_avg_price) * std::abs(trade_qty);
            new_avg_price = 0.0;
        } else if ((old_qty > 0 && new_qty > 0) || (old_qty < 0 && new_qty < 0)) {
            // Same direction - update average price
            new_avg_price = (old_avg_price * std::abs(old_qty) + trade.price * std::abs(trade_qty)) /
                          std::abs(new_qty);
        } else if ((old_qty > 0 && new_qty < 0) || (old_qty < 0 && new_qty > 0)) {
            // Direction change - realize P&L on closed portion
            Quantity closed_qty = std::min(std::abs(old_qty), std::abs(trade_qty));
            realized_pnl_delta = (trade.price - old_avg_price) * closed_qty *
                               ((old_qty > 0) ? 1.0 : -1.0);
            new_avg_price = trade.price; // Reset average price for remaining position
        }

        // Update position atomically
        position.quantity.store(new_qty);
        position.avg_price.store(new_avg_price);
        position.realized_pnl.store(old_realized_pnl + realized_pnl_delta);
        position.last_update.store(trade.timestamp);

        // Update P&L with current market price
        update_position_pnl(position);

        stats_.trades_processed.fetch_add(1);
        stats_.position_updates.fetch_add(1);

        notify_trade_callbacks(trade);
        notify_position_callbacks(position);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    record_processing_time(static_cast<double>(duration.count()));
}

void PositionEngine::update_market_price(const Symbol& symbol, Price price, Timestamp timestamp) {
    {
        std::unique_lock<std::shared_mutex> lock(prices_mutex_);
        auto& data = market_data_[symbol];
        data.symbol = symbol;
        data.last = price;
        data.timestamp = timestamp;
    }

    // Update P&L for all positions with this symbol
    std::shared_lock<std::shared_mutex> positions_lock(positions_mutex_);
    for (const auto& [key, position] : positions_) {
        if (key.symbol == symbol) {
            position->market_price.store(price);
            update_position_pnl(*position);
            notify_position_callbacks(*position);
        }
    }

    stats_.price_updates.fetch_add(1);
}

void PositionEngine::update_market_data(const MarketData& market_data) {
    {
        std::unique_lock<std::shared_mutex> lock(prices_mutex_);
        market_data_[market_data.symbol] = market_data;
    }

    Price price = (market_data.bid + market_data.ask) / 2.0;
    if (price > EPSILON) {
        update_market_price(market_data.symbol, price, market_data.timestamp);
    }
}

std::shared_ptr<Position> PositionEngine::get_position(const InstrumentKey& key) const {
    std::shared_lock<std::shared_mutex> lock(positions_mutex_);
    auto it = positions_.find(key);
    return (it != positions_.end()) ? it->second : nullptr;
}

std::vector<std::shared_ptr<Position>> PositionEngine::get_all_positions() const {
    std::shared_lock<std::shared_mutex> lock(positions_mutex_);
    std::vector<std::shared_ptr<Position>> result;
    result.reserve(positions_.size());

    for (const auto& [key, position] : positions_) {
        if (position->quantity.load() != 0) {
            result.push_back(position);
        }
    }

    return result;
}

std::vector<std::shared_ptr<Position>> PositionEngine::get_positions_by_symbol(const Symbol& symbol) const {
    std::shared_lock<std::shared_mutex> lock(positions_mutex_);
    std::vector<std::shared_ptr<Position>> result;

    for (const auto& [key, position] : positions_) {
        if (key.symbol == symbol && position->quantity.load() != 0) {
            result.push_back(position);
        }
    }

    return result;
}

std::vector<std::shared_ptr<Position>> PositionEngine::get_positions_by_asset_type(AssetType type) const {
    std::shared_lock<std::shared_mutex> lock(positions_mutex_);
    std::vector<std::shared_ptr<Position>> result;

    for (const auto& [key, position] : positions_) {
        if (key.asset_type == type && position->quantity.load() != 0) {
            result.push_back(position);
        }
    }

    return result;
}

Price PositionEngine::calculate_total_pnl() const {
    std::shared_lock<std::shared_mutex> lock(positions_mutex_);
    Price total_pnl = 0.0;

    for (const auto& [key, position] : positions_) {
        total_pnl += position->unrealized_pnl.load() + position->realized_pnl.load();
    }

    return total_pnl;
}

Price PositionEngine::calculate_total_portfolio_value() const {
    std::shared_lock<std::shared_mutex> lock(positions_mutex_);
    Price total_value = 0.0;

    for (const auto& [key, position] : positions_) {
        Quantity qty = position->quantity.load();
        Price market_price = position->market_price.load();

        if (qty != 0 && market_price > EPSILON) {
            total_value += std::abs(qty) * market_price;
        }
    }

    return total_value;
}

Price PositionEngine::calculate_portfolio_delta() const {
    std::shared_lock<std::shared_mutex> lock(positions_mutex_);
    Price total_delta = 0.0;

    for (const auto& [key, position] : positions_) {
        Quantity qty = position->quantity.load();

        if (key.asset_type == AssetType::EQUITY || key.asset_type == AssetType::FUTURE) {
            total_delta += qty; // Delta = 1 for linear instruments
        }
        // For options, delta would need to be calculated separately and stored
    }

    return total_delta;
}

std::unordered_map<Currency, Price> PositionEngine::calculate_currency_exposure() const {
    std::shared_lock<std::shared_mutex> lock(positions_mutex_);
    std::unordered_map<Currency, Price> exposure;

    for (const auto& [key, position] : positions_) {
        Quantity qty = position->quantity.load();
        Price market_price = position->market_price.load();

        if (qty != 0 && market_price > EPSILON) {
            Price notional = qty * market_price;
            exposure[key.currency] += notional;
        }
    }

    return exposure;
}

void PositionEngine::register_position_callback(PositionCallback callback) {
    std::unique_lock<std::shared_mutex> lock(callbacks_mutex_);
    position_callbacks_.push_back(std::move(callback));
}

void PositionEngine::register_trade_callback(TradeCallback callback) {
    std::unique_lock<std::shared_mutex> lock(callbacks_mutex_);
    trade_callbacks_.push_back(std::move(callback));
}

void PositionEngine::start_monitoring() {
    if (!monitoring_active_.exchange(true)) {
        monitoring_thread_ = std::thread(&PositionEngine::run_monitoring_loop, this);
    }
}

void PositionEngine::stop_monitoring() {
    if (monitoring_active_.exchange(false)) {
        if (monitoring_thread_.joinable()) {
            monitoring_thread_.join();
        }
    }
}

void PositionEngine::reset_statistics() {
    stats_.trades_processed.store(0);
    stats_.position_updates.store(0);
    stats_.price_updates.store(0);
    stats_.avg_processing_time_us.store(0.0);
    stats_.max_processing_time_us.store(0.0);
}

void PositionEngine::compact_positions() {
    std::unique_lock<std::shared_mutex> lock(positions_mutex_);

    auto it = positions_.begin();
    while (it != positions_.end()) {
        if (it->second->quantity.load() == 0) {
            it = positions_.erase(it);
        } else {
            ++it;
        }
    }
}

size_t PositionEngine::get_memory_usage() const {
    std::shared_lock<std::shared_mutex> lock(positions_mutex_);
    return positions_.size() * (sizeof(InstrumentKey) + sizeof(Position));
}

void PositionEngine::update_position_pnl(Position& position) const {
    Quantity qty = position.quantity.load();
    Price avg_price = position.avg_price.load();
    Price market_price = position.market_price.load();

    if (qty != 0 && market_price > EPSILON) {
        Price unrealized_pnl = qty * (market_price - avg_price);
        position.unrealized_pnl.store(unrealized_pnl);
    }
}

void PositionEngine::notify_position_callbacks(const Position& position) {
    std::shared_lock<std::shared_mutex> lock(callbacks_mutex_);
    for (const auto& callback : position_callbacks_) {
        try {
            callback(position);
        } catch (...) {
            // Ignore callback exceptions to prevent disruption
        }
    }
}

void PositionEngine::notify_trade_callbacks(const Trade& trade) {
    std::shared_lock<std::shared_mutex> lock(callbacks_mutex_);
    for (const auto& callback : trade_callbacks_) {
        try {
            callback(trade);
        } catch (...) {
            // Ignore callback exceptions to prevent disruption
        }
    }
}

void PositionEngine::run_monitoring_loop() {
    while (monitoring_active_.load()) {
        // Periodic cleanup and maintenance
        std::this_thread::sleep_for(std::chrono::seconds(10));

        // Update P&L for all positions
        auto positions = get_all_positions();
        for (const auto& position : positions) {
            update_position_pnl(*position);
        }
    }
}

inline void PositionEngine::record_processing_time(double time_us) {
    double current_avg = stats_.avg_processing_time_us.load();
    uint64_t count = stats_.trades_processed.load();

    if (count > 0) {
        double new_avg = (current_avg * (count - 1) + time_us) / count;
        stats_.avg_processing_time_us.store(new_avg);
    }

    double current_max = stats_.max_processing_time_us.load();
    if (time_us > current_max) {
        stats_.max_processing_time_us.store(time_us);
    }
}

inline Price PositionEngine::get_current_price(const Symbol& symbol) const {
    std::shared_lock<std::shared_mutex> lock(prices_mutex_);
    auto it = market_data_.find(symbol);
    return (it != market_data_.end()) ? it->second.last : 0.0;
}

// PositionAggregator implementation
PositionAggregator::PositionAggregator(const PositionEngine& engine)
    : position_engine_(engine) {
}

void PositionAggregator::update_aggregations() {
    std::unique_lock<std::shared_mutex> lock(aggregations_mutex_);
    aggregations_.clear();

    auto positions = position_engine_.get_all_positions();
    for (const auto& position : positions) {
        AggregationKey key{position->instrument.currency, position->instrument.asset_type};
        auto& agg = aggregations_[key];

        agg.currency = position->instrument.currency;
        agg.asset_type = position->instrument.asset_type;
        agg.total_quantity += position->quantity.load();
        agg.total_notional += position->quantity.load() * position->market_price.load();
        agg.total_pnl += position->unrealized_pnl.load() + position->realized_pnl.load();
        agg.position_count++;
        agg.last_update = std::max(agg.last_update, position->last_update.load());
    }
}

std::vector<PositionAggregator::AggregatedPosition> PositionAggregator::get_aggregated_positions() const {
    std::shared_lock<std::shared_mutex> lock(aggregations_mutex_);
    std::vector<AggregatedPosition> result;
    result.reserve(aggregations_.size());

    for (const auto& [key, agg] : aggregations_) {
        result.push_back(agg);
    }

    return result;
}

PositionAggregator::AggregatedPosition PositionAggregator::get_aggregation(
    const Currency& currency, AssetType asset_type) const {
    std::shared_lock<std::shared_mutex> lock(aggregations_mutex_);
    AggregationKey key{currency, asset_type};
    auto it = aggregations_.find(key);
    return (it != aggregations_.end()) ? it->second : AggregatedPosition{};
}

void PositionAggregator::on_position_update(const Position& position) {
    std::unique_lock<std::shared_mutex> lock(aggregations_mutex_);
    AggregationKey key{position.instrument.currency, position.instrument.asset_type};

    // For real-time updates, we'd need to maintain deltas and update incrementally
    // This is a simplified version that triggers full recalculation
    update_aggregations();
}

} // namespace risk_engine