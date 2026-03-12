#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <thread>
#include <memory>
#include <iomanip>

#include "core/position_engine.hpp"
#include "risk/risk_calculator.hpp"
#include "risk/greeks_calculator.hpp"

using namespace risk_engine;
using namespace std::chrono;

class BenchmarkTimer {
public:
    BenchmarkTimer(const std::string& name) : name_(name), start_(high_resolution_clock::now()) {}

    ~BenchmarkTimer() {
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start_);
        std::cout << std::setw(40) << std::left << name_
                  << std::setw(15) << std::right << duration.count() << " μs" << std::endl;
    }

private:
    std::string name_;
    high_resolution_clock::time_point start_;
};

class PerformanceBenchmarks {
public:
    PerformanceBenchmarks() : rng_(std::random_device{}()) {}

    void run_all_benchmarks() {
        std::cout << "=== Quantitative Risk Engine Performance Benchmarks ===" << std::endl;
        std::cout << std::setw(40) << std::left << "Benchmark"
                  << std::setw(15) << std::right << "Time" << std::endl;
        std::cout << std::string(55, '-') << std::endl;

        benchmark_position_engine();
        benchmark_risk_calculations();
        benchmark_greeks_calculations();
        benchmark_concurrent_operations();
        benchmark_memory_performance();

        std::cout << std::string(55, '=') << std::endl;
    }

private:
    std::mt19937 rng_;

    void benchmark_position_engine() {
        std::cout << "\n--- Position Engine Benchmarks ---" << std::endl;

        PositionEngine engine;

        // Benchmark 1: Single trade processing
        {
            BenchmarkTimer timer("Single trade processing");
            InstrumentKey key{"AAPL", AssetType::EQUITY, "USD"};
            Trade trade{key, 1000, 150.50, Side::BUY, high_resolution_clock::now(), "T001", "ACC001"};
            engine.add_trade(trade);
        }

        // Benchmark 2: Batch trade processing
        {
            BenchmarkTimer timer("10,000 trade batch");
            std::vector<Trade> trades;
            trades.reserve(10000);

            for (int i = 0; i < 10000; ++i) {
                InstrumentKey key{
                    "STOCK_" + std::to_string(i % 100),
                    AssetType::EQUITY,
                    "USD"
                };

                Trade trade{
                    key,
                    static_cast<Quantity>(std::uniform_int_distribution<int>(-1000, 1000)(rng_)),
                    std::uniform_real_distribution<Price>(50.0, 500.0)(rng_),
                    std::bernoulli_distribution(0.5)(rng_) ? Side::BUY : Side::SELL,
                    high_resolution_clock::now(),
                    "T" + std::to_string(i),
                    "ACC001"
                };

                auto start = high_resolution_clock::now();
                engine.add_trade(trade);
                auto end = high_resolution_clock::now();

                if (i == 0) {
                    // First trade timing (cold cache)
                    auto duration = duration_cast<nanoseconds>(end - start);
                    std::cout << std::setw(40) << std::left << "  First trade (cold cache)"
                              << std::setw(15) << std::right << duration.count() << " ns" << std::endl;
                }
            }
        }

        // Benchmark 3: Market data updates
        {
            BenchmarkTimer timer("100,000 price updates");
            for (int i = 0; i < 100000; ++i) {
                std::string symbol = "STOCK_" + std::to_string(i % 100);
                Price price = std::uniform_real_distribution<Price>(50.0, 500.0)(rng_);
                engine.update_market_price(symbol, price);
            }
        }

        // Benchmark 4: Position queries
        {
            BenchmarkTimer timer("10,000 position queries");
            for (int i = 0; i < 10000; ++i) {
                std::string symbol = "STOCK_" + std::to_string(i % 100);
                InstrumentKey key{symbol, AssetType::EQUITY, "USD"};
                auto position = engine.get_position(key);
            }
        }

        // Benchmark 5: Portfolio aggregation
        {
            BenchmarkTimer timer("Portfolio value calculation");
            auto total_value = engine.calculate_total_portfolio_value();
            auto total_pnl = engine.calculate_total_pnl();
        }

        // Print final statistics
        const auto& stats = engine.get_statistics();
        std::cout << std::setw(40) << std::left << "  Total trades processed"
                  << std::setw(15) << std::right << stats.trades_processed.load() << std::endl;
        std::cout << std::setw(40) << std::left << "  Avg processing time"
                  << std::setw(15) << std::right << stats.avg_processing_time_us.load() << " μs" << std::endl;
    }

    void benchmark_risk_calculations() {
        std::cout << "\n--- Risk Calculation Benchmarks ---" << std::endl;

        PositionEngine engine;

        // Create sample portfolio
        create_sample_portfolio(engine);

        RiskCalculator risk_calculator(engine);

        // Benchmark Monte Carlo VaR
        {
            BenchmarkTimer timer("Monte Carlo VaR (10K scenarios)");
            try {
                auto result = risk_calculator.calculate_var("monte_carlo", 0.95);
            } catch (const std::exception& e) {
                std::cout << "Monte Carlo VaR failed: " << e.what() << std::endl;
            }
        }

        // Benchmark Historical VaR
        {
            BenchmarkTimer timer("Historical VaR");
            try {
                auto result = risk_calculator.calculate_var("historical", 0.95);
            } catch (const std::exception& e) {
                std::cout << "Historical VaR calculation not implemented" << std::endl;
            }
        }

        // Benchmark Expected Shortfall
        {
            BenchmarkTimer timer("Expected Shortfall");
            try {
                auto result = risk_calculator.calculate_expected_shortfall(0.95);
            } catch (const std::exception& e) {
                std::cout << "Expected Shortfall calculation not implemented" << std::endl;
            }
        }
    }

    void benchmark_greeks_calculations() {
        std::cout << "\n--- Greeks Calculation Benchmarks ---" << std::endl;

        // Benchmark Black-Scholes calculations
        {
            BenchmarkTimer timer("1,000 Black-Scholes calculations");

            for (int i = 0; i < 1000; ++i) {
                BlackScholesCalculator::OptionParameters params{
                    .spot_price = 100.0 + std::uniform_real_distribution<double>(-10.0, 10.0)(rng_),
                    .strike_price = 100.0,
                    .time_to_expiry = std::uniform_real_distribution<double>(0.01, 1.0)(rng_),
                    .risk_free_rate = 0.05,
                    .volatility = std::uniform_real_distribution<double>(0.1, 0.5)(rng_),
                    .dividend_yield = 0.02,
                    .option_type = std::bernoulli_distribution(0.5)(rng_) ? OptionType::CALL : OptionType::PUT
                };

                auto result = BlackScholesCalculator::calculate_option_value(params);
            }
        }

        // Benchmark single Greeks calculation
        {
            BenchmarkTimer timer("Single Greeks calculation");

            BlackScholesCalculator::OptionParameters params{
                .spot_price = 100.0,
                .strike_price = 100.0,
                .time_to_expiry = 0.25,
                .risk_free_rate = 0.05,
                .volatility = 0.2,
                .dividend_yield = 0.02,
                .option_type = OptionType::CALL
            };

            auto greeks = BlackScholesCalculator::calculate_greeks(params);
        }

        // Benchmark binomial tree
        {
            BenchmarkTimer timer("Binomial tree (100 steps)");

            BinomialTreeCalculator::TreeParameters params{
                .spot_price = 100.0,
                .strike_price = 100.0,
                .time_to_expiry = 0.25,
                .risk_free_rate = 0.05,
                .volatility = 0.2,
                .dividend_yield = 0.02,
                .option_type = OptionType::CALL,
                .num_steps = 100,
                .is_american = false
            };

            auto result = BinomialTreeCalculator::calculate_option_value(params);
        }

        // Benchmark implied volatility calculation
        {
            BenchmarkTimer timer("100 implied vol calculations");

            for (int i = 0; i < 100; ++i) {
                BlackScholesCalculator::OptionParameters params{
                    .spot_price = 100.0,
                    .strike_price = 100.0,
                    .time_to_expiry = 0.25,
                    .risk_free_rate = 0.05,
                    .volatility = 0.2,  // Will be overridden
                    .dividend_yield = 0.02,
                    .option_type = OptionType::CALL
                };

                Price market_price = 10.0 + std::uniform_real_distribution<double>(-2.0, 2.0)(rng_);
                auto implied_vol = BlackScholesCalculator::calculate_implied_volatility(market_price, params);
            }
        }
    }

    void benchmark_concurrent_operations() {
        std::cout << "\n--- Concurrent Operations Benchmarks ---" << std::endl;

        PositionEngine engine;
        engine.start_monitoring();

        const int num_threads = std::thread::hardware_concurrency();
        const int operations_per_thread = 1000;

        // Benchmark concurrent trade processing
        {
            BenchmarkTimer timer("Concurrent trade processing");

            std::vector<std::thread> threads;

            for (int t = 0; t < num_threads; ++t) {
                threads.emplace_back([&engine, t, operations_per_thread, this]() {
                    std::mt19937 local_rng(std::random_device{}());

                    for (int i = 0; i < operations_per_thread; ++i) {
                        InstrumentKey key{
                            "STOCK_" + std::to_string((t * operations_per_thread + i) % 50),
                            AssetType::EQUITY,
                            "USD"
                        };

                        Trade trade{
                            key,
                            static_cast<Quantity>(std::uniform_int_distribution<int>(-500, 500)(local_rng)),
                            std::uniform_real_distribution<Price>(50.0, 500.0)(local_rng),
                            std::bernoulli_distribution(0.5)(local_rng) ? Side::BUY : Side::SELL,
                            high_resolution_clock::now(),
                            "T" + std::to_string(t) + "_" + std::to_string(i),
                            "ACC" + std::to_string(t)
                        };

                        engine.add_trade(trade);
                    }
                });
            }

            for (auto& thread : threads) {
                thread.join();
            }
        }

        // Benchmark concurrent price updates
        {
            BenchmarkTimer timer("Concurrent price updates");

            std::vector<std::thread> threads;

            for (int t = 0; t < num_threads; ++t) {
                threads.emplace_back([&engine, t, operations_per_thread, this]() {
                    std::mt19937 local_rng(std::random_device{}());

                    for (int i = 0; i < operations_per_thread; ++i) {
                        std::string symbol = "STOCK_" + std::to_string((t * operations_per_thread + i) % 50);
                        Price price = std::uniform_real_distribution<Price>(50.0, 500.0)(local_rng);
                        engine.update_market_price(symbol, price);
                    }
                });
            }

            for (auto& thread : threads) {
                thread.join();
            }
        }

        engine.stop_monitoring();

        std::cout << std::setw(40) << std::left << "  Threads used"
                  << std::setw(15) << std::right << num_threads << std::endl;
        std::cout << std::setw(40) << std::left << "  Operations per thread"
                  << std::setw(15) << std::right << operations_per_thread << std::endl;
    }

    void benchmark_memory_performance() {
        std::cout << "\n--- Memory Performance Benchmarks ---" << std::endl;

        // Benchmark memory allocation/deallocation
        {
            BenchmarkTimer timer("100,000 position allocations");

            std::vector<std::unique_ptr<Position>> positions;
            positions.reserve(100000);

            for (int i = 0; i < 100000; ++i) {
                InstrumentKey key{
                    "STOCK_" + std::to_string(i),
                    AssetType::EQUITY,
                    "USD"
                };

                auto position = std::make_unique<Position>(key);
                position->quantity.store(std::uniform_int_distribution<int>(-1000, 1000)(rng_));
                position->avg_price.store(std::uniform_real_distribution<double>(50.0, 500.0)(rng_));

                positions.push_back(std::move(position));
            }
        }

        // Benchmark cache performance
        {
            BenchmarkTimer timer("Cache-friendly data access");

            PositionEngine engine;
            create_sample_portfolio(engine);

            // Sequential access pattern (cache-friendly)
            auto positions = engine.get_all_positions();
            double total_value = 0.0;

            for (const auto& position : positions) {
                total_value += position->quantity.load() * position->market_price.load();
            }
        }

        // Benchmark with memory fragmentation
        {
            BenchmarkTimer timer("Random access pattern");

            PositionEngine engine;
            create_sample_portfolio(engine);

            // Random access pattern (cache-unfriendly)
            std::vector<std::string> symbols;
            for (int i = 0; i < 100; ++i) {
                symbols.push_back("STOCK_" + std::to_string(i));
            }

            std::shuffle(symbols.begin(), symbols.end(), rng_);

            double total_value = 0.0;
            for (const auto& symbol : symbols) {
                InstrumentKey key{symbol, AssetType::EQUITY, "USD"};
                auto position = engine.get_position(key);
                if (position) {
                    total_value += position->quantity.load() * position->market_price.load();
                }
            }
        }
    }

    void create_sample_portfolio(PositionEngine& engine) {
        // Create a diverse sample portfolio for testing
        std::vector<std::string> symbols = {
            "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "AMD",
            "JPM", "BAC", "WFC", "C", "GS", "MS", "V", "MA",
            "JNJ", "PFE", "MRK", "ABBV", "BMY", "LLY", "UNH", "CVS"
        };

        for (size_t i = 0; i < symbols.size(); ++i) {
            InstrumentKey key{symbols[i], AssetType::EQUITY, "USD"};

            Trade trade{
                key,
                static_cast<Quantity>(std::uniform_int_distribution<int>(-1000, 1000)(rng_)),
                std::uniform_real_distribution<Price>(50.0, 500.0)(rng_),
                std::bernoulli_distribution(0.5)(rng_) ? Side::BUY : Side::SELL,
                high_resolution_clock::now(),
                "INIT_" + std::to_string(i),
                "SAMPLE_ACCOUNT"
            };

            engine.add_trade(trade);

            // Update market price
            Price market_price = trade.price * std::uniform_real_distribution<double>(0.95, 1.05)(rng_);
            engine.update_market_price(symbols[i], market_price);
        }
    }
};

int main() {
    std::cout << "Starting Performance Benchmarks..." << std::endl;
    std::cout << "Hardware Concurrency: " << std::thread::hardware_concurrency() << " threads" << std::endl;
    std::cout << "Compiler: " <<
#ifdef __GNUC__
        "GCC " << __GNUC__ << "." << __GNUC_MINOR__
#elif defined(_MSC_VER)
        "MSVC " << _MSC_VER
#elif defined(__clang__)
        "Clang " << __clang_major__ << "." << __clang_minor__
#else
        "Unknown"
#endif
        << std::endl;

#ifdef NDEBUG
    std::cout << "Build Type: Release (Optimized)" << std::endl;
#else
    std::cout << "Build Type: Debug (Non-optimized)" << std::endl;
#endif

    std::cout << std::endl;

    PerformanceBenchmarks benchmarks;
    benchmarks.run_all_benchmarks();

    std::cout << "\nBenchmarks completed!" << std::endl;
    std::cout << "\nPerformance Targets:" << std::endl;
    std::cout << "- Single trade processing: < 1 μs" << std::endl;
    std::cout << "- Market data update: < 0.5 μs" << std::endl;
    std::cout << "- Monte Carlo VaR (10K): < 10 ms" << std::endl;
    std::cout << "- Black-Scholes Greeks: < 1 μs" << std::endl;
    std::cout << "- Position query: < 0.1 μs" << std::endl;

    return 0;
}