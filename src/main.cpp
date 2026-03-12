#include <iostream>
#include <memory>
#include <thread>
#include <chrono>
#include <signal.h>

#include "core/position_engine.hpp"
#include "risk/risk_calculator.hpp"
#include "risk/greeks_calculator.hpp"

using namespace risk_engine;

// Global flag for graceful shutdown
std::atomic<bool> g_running{true};

void signal_handler(int signal) {
    std::cout << "\nReceived signal " << signal << ". Shutting down gracefully..." << std::endl;
    g_running = false;
}

void demo_trading_simulation(PositionEngine& position_engine) {
    std::cout << "Starting trading simulation..." << std::endl;

    // Demo instruments
    std::vector<InstrumentKey> instruments = {
        {.symbol = "AAPL", .asset_type = AssetType::EQUITY, .currency = "USD"},
        {.symbol = "GOOGL", .asset_type = AssetType::EQUITY, .currency = "USD"},
        {.symbol = "MSFT", .asset_type = AssetType::EQUITY, .currency = "USD"},
        {.symbol = "TSLA", .asset_type = AssetType::EQUITY, .currency = "USD"},
        {.symbol = "SPY", .asset_type = AssetType::EQUITY, .currency = "USD"},
        {.symbol = "AAPL_CALL_150", .asset_type = AssetType::OPTION, .currency = "USD",
         .strike = 150.0, .option_type = OptionType::CALL,
         .expiry = std::chrono::high_resolution_clock::now() + std::chrono::hours(24 * 30)}
    };

    // Initial market prices
    std::unordered_map<Symbol, Price> initial_prices = {
        {"AAPL", 175.50},
        {"GOOGL", 2750.25},
        {"MSFT", 338.75},
        {"TSLA", 850.00},
        {"SPY", 425.50},
        {"AAPL_CALL_150", 28.75}
    };

    // Set initial market prices
    for (const auto& [symbol, price] : initial_prices) {
        position_engine.update_market_price(symbol, price);
    }

    // Generate demo trades
    std::vector<Trade> demo_trades = {
        {instruments[0], 1000, 174.25, Side::BUY, std::chrono::high_resolution_clock::now(), "T001", "DEMO_ACCOUNT"},
        {instruments[1], 100, 2740.50, Side::BUY, std::chrono::high_resolution_clock::now(), "T002", "DEMO_ACCOUNT"},
        {instruments[2], 500, 340.00, Side::BUY, std::chrono::high_resolution_clock::now(), "T003", "DEMO_ACCOUNT"},
        {instruments[3], -200, 855.75, Side::SELL, std::chrono::high_resolution_clock::now(), "T004", "DEMO_ACCOUNT"},
        {instruments[4], 2000, 424.75, Side::BUY, std::chrono::high_resolution_clock::now(), "T005", "DEMO_ACCOUNT"},
        {instruments[5], 10, 29.50, Side::BUY, std::chrono::high_resolution_clock::now(), "T006", "DEMO_ACCOUNT"}
    };

    // Execute demo trades
    for (const auto& trade : demo_trades) {
        position_engine.add_trade(trade);
        std::cout << "Executed trade: " << trade.trade_id << " - "
                  << trade.instrument.symbol << " " << trade.quantity
                  << " @ " << trade.price << std::endl;
    }

    std::cout << "Demo trades executed. Portfolio value: "
              << position_engine.calculate_total_portfolio_value() << std::endl;
}

void market_data_simulation(PositionEngine& position_engine) {
    std::cout << "Starting market data simulation..." << std::endl;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> price_change(0.0, 0.02); // 2% volatility

    std::unordered_map<Symbol, Price> current_prices = {
        {"AAPL", 175.50},
        {"GOOGL", 2750.25},
        {"MSFT", 338.75},
        {"TSLA", 850.00},
        {"SPY", 425.50},
        {"AAPL_CALL_150", 28.75}
    };

    while (g_running) {
        // Update market prices with random walks
        for (auto& [symbol, price] : current_prices) {
            double change = price_change(gen);
            price *= (1.0 + change);
            price = std::max(price, 1.0); // Minimum price of $1

            position_engine.update_market_price(symbol, price);
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // 10 FPS
    }
}

void risk_monitoring(const PositionEngine& position_engine) {
    std::cout << "Starting risk monitoring..." << std::endl;

    RiskCalculator risk_calculator(position_engine);
    PortfolioGreeksCalculator greeks_calculator(position_engine);

    auto last_report_time = std::chrono::steady_clock::now();
    const auto report_interval = std::chrono::seconds(10);

    while (g_running) {
        auto now = std::chrono::steady_clock::now();

        if (now - last_report_time >= report_interval) {
            try {
                // Calculate risk metrics
                auto risk_result = risk_calculator.calculate_portfolio_risk();
                auto portfolio_greeks = greeks_calculator.calculate_portfolio_greeks();

                // Print risk report
                std::cout << "\n=== Risk Report ===" << std::endl;
                std::cout << "Portfolio Value: $" << risk_result.portfolio_value << std::endl;
                std::cout << "Monte Carlo VaR: $" << risk_result.monte_carlo_var << std::endl;
                std::cout << "Expected Shortfall: $" << risk_result.expected_shortfall << std::endl;
                std::cout << "Portfolio Beta: " << risk_result.beta << std::endl;
                std::cout << "Sharpe Ratio: " << risk_result.tracking_error << std::endl;

                std::cout << "\n=== Greeks ===" << std::endl;
                std::cout << "Total Delta: " << portfolio_greeks.total_delta << std::endl;
                std::cout << "Total Gamma: " << portfolio_greeks.total_gamma << std::endl;
                std::cout << "Total Theta: " << portfolio_greeks.total_theta << std::endl;
                std::cout << "Total Vega: " << portfolio_greeks.total_vega << std::endl;

                // Check for risk limit violations
                const double VAR_LIMIT = risk_result.portfolio_value * 0.05; // 5% of portfolio
                if (risk_result.monte_carlo_var > VAR_LIMIT) {
                    std::cout << "*** ALERT: VaR limit breach! ***" << std::endl;
                }

                std::cout << "===================" << std::endl;

                last_report_time = now;
            } catch (const std::exception& e) {
                std::cerr << "Risk calculation error: " << e.what() << std::endl;
            }
        }

        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

void performance_monitoring(const PositionEngine& position_engine) {
    std::cout << "Starting performance monitoring..." << std::endl;

    auto last_stats_time = std::chrono::steady_clock::now();
    const auto stats_interval = std::chrono::seconds(30);

    while (g_running) {
        auto now = std::chrono::steady_clock::now();

        if (now - last_stats_time >= stats_interval) {
            const auto& stats = position_engine.get_statistics();

            std::cout << "\n=== Performance Stats ===" << std::endl;
            std::cout << "Trades processed: " << stats.trades_processed.load() << std::endl;
            std::cout << "Position updates: " << stats.position_updates.load() << std::endl;
            std::cout << "Price updates: " << stats.price_updates.load() << std::endl;
            std::cout << "Avg processing time: " << stats.avg_processing_time_us.load() << " μs" << std::endl;
            std::cout << "Max processing time: " << stats.max_processing_time_us.load() << " μs" << std::endl;
            std::cout << "Memory usage: " << position_engine.get_memory_usage() << " bytes" << std::endl;
            std::cout << "=========================" << std::endl;

            last_stats_time = now;
        }

        std::this_thread::sleep_for(std::chrono::seconds(5));
    }
}

int main(int argc, char* argv[]) {
    std::cout << "Quantitative Risk Analytics Engine v1.0" << std::endl;
    std::cout << "=========================================" << std::endl;

    // Set up signal handlers for graceful shutdown
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    try {
        // Initialize core engine
        PositionEngine position_engine;
        position_engine.start_monitoring();

        // Set up position update callback
        position_engine.register_position_callback([](const Position& position) {
            // This could integrate with external systems
            // For demo, we'll just count the updates
        });

        position_engine.register_trade_callback([](const Trade& trade) {
            // This could publish to message queues, audit logs, etc.
            // For demo, we'll just count the trades
        });

        std::cout << "Position engine initialized." << std::endl;

        // Run demo trading simulation
        demo_trading_simulation(position_engine);

        // Start background threads
        std::vector<std::thread> background_threads;

        // Market data simulation thread
        background_threads.emplace_back(market_data_simulation, std::ref(position_engine));

        // Risk monitoring thread
        background_threads.emplace_back(risk_monitoring, std::cref(position_engine));

        // Performance monitoring thread
        background_threads.emplace_back(performance_monitoring, std::cref(position_engine));

        std::cout << "All systems started. Press Ctrl+C to shutdown." << std::endl;

        // Main loop
        while (g_running) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

        std::cout << "Shutting down systems..." << std::endl;

        // Stop position engine monitoring
        position_engine.stop_monitoring();

        // Join all background threads
        for (auto& thread : background_threads) {
            if (thread.joinable()) {
                thread.join();
            }
        }

        // Final statistics
        const auto& final_stats = position_engine.get_statistics();
        std::cout << "\nFinal Statistics:" << std::endl;
        std::cout << "Total trades processed: " << final_stats.trades_processed.load() << std::endl;
        std::cout << "Total position updates: " << final_stats.position_updates.load() << std::endl;
        std::cout << "Total price updates: " << final_stats.price_updates.load() << std::endl;

        std::cout << "Shutdown complete." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown fatal error occurred" << std::endl;
        return 1;
    }

    return 0;
}