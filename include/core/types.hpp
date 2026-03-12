#pragma once

#include <string>
#include <chrono>
#include <cstdint>
#include <atomic>

namespace risk_engine {

using Timestamp = std::chrono::time_point<std::chrono::high_resolution_clock>;
using Price = double;
using Quantity = int64_t;
using Currency = std::string;
using Symbol = std::string;

enum class AssetType : uint8_t {
    EQUITY = 0,
    OPTION = 1,
    FUTURE = 2,
    BOND = 3,
    FX = 4,
    COMMODITY = 5,
    CRYPTO = 6
};

enum class OptionType : uint8_t {
    CALL = 0,
    PUT = 1
};

enum class Side : uint8_t {
    BUY = 0,
    SELL = 1
};

struct InstrumentKey {
    Symbol symbol;
    AssetType asset_type;
    Currency currency;

    // Option-specific fields
    Price strike = 0.0;
    Timestamp expiry{};
    OptionType option_type = OptionType::CALL;

    bool operator<(const InstrumentKey& other) const noexcept {
        if (symbol != other.symbol) return symbol < other.symbol;
        if (asset_type != other.asset_type) return asset_type < other.asset_type;
        if (currency != other.currency) return currency < other.currency;
        if (strike != other.strike) return strike < other.strike;
        if (expiry != other.expiry) return expiry < other.expiry;
        return option_type < other.option_type;
    }

    bool operator==(const InstrumentKey& other) const noexcept {
        return symbol == other.symbol &&
               asset_type == other.asset_type &&
               currency == other.currency &&
               strike == other.strike &&
               expiry == other.expiry &&
               option_type == other.option_type;
    }
};

struct Position {
    InstrumentKey instrument;
    std::atomic<Quantity> quantity{0};
    std::atomic<Price> avg_price{0.0};
    std::atomic<Price> market_price{0.0};
    std::atomic<Timestamp> last_update{};
    std::atomic<Price> unrealized_pnl{0.0};
    std::atomic<Price> realized_pnl{0.0};

    Position() = default;
    Position(const InstrumentKey& key) : instrument(key) {}

    Position(const Position& other)
        : instrument(other.instrument)
        , quantity(other.quantity.load())
        , avg_price(other.avg_price.load())
        , market_price(other.market_price.load())
        , last_update(other.last_update.load())
        , unrealized_pnl(other.unrealized_pnl.load())
        , realized_pnl(other.realized_pnl.load()) {}
};

struct Trade {
    InstrumentKey instrument;
    Quantity quantity;
    Price price;
    Side side;
    Timestamp timestamp;
    std::string trade_id;
    std::string account_id;
    Price commission = 0.0;
};

struct MarketData {
    Symbol symbol;
    Price bid = 0.0;
    Price ask = 0.0;
    Price last = 0.0;
    Quantity bid_size = 0;
    Quantity ask_size = 0;
    Quantity volume = 0;
    Timestamp timestamp{};

    Price mid() const noexcept {
        return (bid + ask) / 2.0;
    }

    Price spread() const noexcept {
        return ask - bid;
    }
};

struct RiskMetrics {
    Price portfolio_value = 0.0;
    Price var_1d = 0.0;
    Price var_10d = 0.0;
    Price expected_shortfall = 0.0;
    Price beta = 0.0;
    Price sharpe_ratio = 0.0;
    Price max_drawdown = 0.0;
    Timestamp last_calculated{};
};

struct Greeks {
    Price delta = 0.0;
    Price gamma = 0.0;
    Price theta = 0.0;
    Price vega = 0.0;
    Price rho = 0.0;
    Price epsilon = 0.0;  // dividend sensitivity
    Timestamp last_calculated{};
};

constexpr double EPSILON = 1e-10;
constexpr int MAX_SYMBOLS = 100000;
constexpr int MAX_POSITIONS = 1000000;

} // namespace risk_engine