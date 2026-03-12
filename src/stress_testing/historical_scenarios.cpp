/**
 * Historical Scenario Database and Replay System
 *
 * Comprehensive database of major financial crises and market stress events
 * with detailed factor shock definitions and historical data replay capabilities
 */

#include "stress_testing/stress_framework.hpp"
#include <cmath>
#include <algorithm>
#include <sstream>
#include <iomanip>

namespace risk_analytics {
namespace stress_testing {

void HistoricalScenarioDatabase::initialize_historical_scenarios() {
    add_lehman_crisis_scenario();
    add_covid19_scenario();
    add_dot_com_crash_scenario();
    add_european_debt_crisis_scenario();
    add_flash_crash_scenario();
    add_brexit_scenario();
    add_china_market_crash_scenario();
    add_russia_ukraine_scenario();
    add_swiss_franc_shock_scenario();
    add_oil_price_crash_scenario();
}

void HistoricalScenarioDatabase::add_lehman_crisis_scenario() {
    HistoricalScenario scenario("2008_Financial_Crisis",
        "Global Financial Crisis triggered by Lehman Brothers collapse");

    // Set dates (September 15, 2008 - March 9, 2009)
    std::tm start_tm = {};
    start_tm.tm_year = 108; // 2008
    start_tm.tm_mon = 8;    // September (0-based)
    start_tm.tm_mday = 15;
    scenario.start_date = std::chrono::system_clock::from_time_t(std::mktime(&start_tm));

    std::tm end_tm = {};
    end_tm.tm_year = 109; // 2009
    end_tm.tm_mon = 2;    // March
    end_tm.tm_mday = 9;
    scenario.end_date = std::chrono::system_clock::from_time_t(std::mktime(&end_tm));

    scenario.duration_days = 175;
    scenario.severity_score = 10.0;
    scenario.affected_regions = {"North America", "Europe", "Asia", "Global"};
    scenario.affected_sectors = {"Financial", "Real Estate", "Consumer Discretionary", "Industrials"};

    // Define major risk factor shocks
    std::vector<RiskFactorShock> shocks;

    // Equity market shocks
    shocks.emplace_back(RiskFactorType::EQUITY_INDEX, "S&P 500", -56.8);
    shocks.back().affected_instruments = {"SPY", "ES", "US_LARGE_CAP"};
    shocks.back().time_horizon_days = 175;

    shocks.emplace_back(RiskFactorType::EQUITY_INDEX, "FTSE 100", -31.3);
    shocks.back().affected_instruments = {"UKX", "UK_LARGE_CAP"};

    shocks.emplace_back(RiskFactorType::EQUITY_INDEX, "Nikkei 225", -42.1);
    shocks.back().affected_instruments = {"NKY", "JP_LARGE_CAP"};

    shocks.emplace_back(RiskFactorType::EQUITY_INDEX, "DAX", -40.4);
    shocks.back().affected_instruments = {"DAX", "DE_LARGE_CAP"};

    // Interest rate shocks
    shocks.emplace_back(RiskFactorType::INTEREST_RATE, "US_10Y_TREASURY", -189.0);
    shocks.back().is_relative = false; // Absolute change in basis points
    shocks.back().affected_instruments = {"US10Y", "TREASURY_BONDS"};

    shocks.emplace_back(RiskFactorType::INTEREST_RATE, "US_3M_LIBOR", -324.0);
    shocks.back().is_relative = false;
    shocks.back().affected_instruments = {"USD3M", "MONEY_MARKET"};

    // Credit spread shocks
    shocks.emplace_back(RiskFactorType::CREDIT_SPREAD, "IG_CORPORATE_SPREAD", +590.0);
    shocks.back().is_relative = false; // Basis points
    shocks.back().affected_instruments = {"LQD", "CORPORATE_BONDS"};

    shocks.emplace_back(RiskFactorType::CREDIT_SPREAD, "HY_CORPORATE_SPREAD", +1970.0);
    shocks.back().is_relative = false;
    shocks.back().affected_instruments = {"HYG", "JNK", "HIGH_YIELD"};

    // FX rate shocks (USD strengthening)
    shocks.emplace_back(RiskFactorType::FX_RATE, "EUR/USD", -21.7);
    shocks.back().affected_instruments = {"EURUSD", "EUR_CROSSES"};

    shocks.emplace_back(RiskFactorType::FX_RATE, "GBP/USD", -26.8);
    shocks.back().affected_instruments = {"GBPUSD", "GBP_CROSSES"};

    shocks.emplace_back(RiskFactorType::FX_RATE, "USD/JPY", -23.1);
    shocks.back().affected_instruments = {"USDJPY", "JPY_CROSSES"};

    // Volatility shocks
    shocks.emplace_back(RiskFactorType::VOLATILITY, "VIX", +572.0);
    shocks.back().affected_instruments = {"VIX", "VOLATILITY_INDICES"};

    // Commodity shocks
    shocks.emplace_back(RiskFactorType::COMMODITY, "WTI_CRUDE", -78.1);
    shocks.back().affected_instruments = {"WTI", "CRUDE_OIL"};

    shocks.emplace_back(RiskFactorType::COMMODITY, "GOLD", +25.0);
    shocks.back().affected_instruments = {"GOLD", "PRECIOUS_METALS"};

    scenario.factor_shocks = std::move(shocks);
    scenarios_["2008_Financial_Crisis"] = std::move(scenario);
}

void HistoricalScenarioDatabase::add_covid19_scenario() {
    HistoricalScenario scenario("2020_COVID19_Crisis",
        "Global pandemic-induced market crash and subsequent recovery");

    // March 2020 crash (February 19 - March 23, 2020)
    std::tm start_tm = {};
    start_tm.tm_year = 120; // 2020
    start_tm.tm_mon = 1;    // February
    start_tm.tm_mday = 19;
    scenario.start_date = std::chrono::system_clock::from_time_t(std::mktime(&start_tm));

    std::tm end_tm = {};
    end_tm.tm_year = 120; // 2020
    end_tm.tm_mon = 2;    // March
    end_tm.tm_mday = 23;
    scenario.end_date = std::chrono::system_clock::from_time_t(std::mktime(&end_tm));

    scenario.duration_days = 33;
    scenario.severity_score = 9.5;
    scenario.affected_regions = {"Global"};
    scenario.affected_sectors = {"Travel", "Hospitality", "Energy", "Financial", "Retail"};

    std::vector<RiskFactorShock> shocks;

    // Equity market shocks (rapid crash)
    shocks.emplace_back(RiskFactorType::EQUITY_INDEX, "S&P 500", -33.9);
    shocks.back().time_horizon_days = 33;
    shocks.back().affected_instruments = {"SPY", "ES", "US_LARGE_CAP"};

    shocks.emplace_back(RiskFactorType::EQUITY_INDEX, "STOXX 600", -39.3);
    shocks.back().affected_instruments = {"STOXX", "EU_LARGE_CAP"};

    shocks.emplace_back(RiskFactorType::EQUITY_INDEX, "Nikkei 225", -29.1);
    shocks.back().affected_instruments = {"NKY", "JP_LARGE_CAP"};

    // Interest rate shocks (flight to quality)
    shocks.emplace_back(RiskFactorType::INTEREST_RATE, "US_10Y_TREASURY", -125.0);
    shocks.back().is_relative = false;
    shocks.back().affected_instruments = {"US10Y", "TREASURY_BONDS"};

    shocks.emplace_back(RiskFactorType::INTEREST_RATE, "US_2Y_TREASURY", -150.0);
    shocks.back().is_relative = false;
    shocks.back().affected_instruments = {"US2Y"};

    // Credit spread widening
    shocks.emplace_back(RiskFactorType::CREDIT_SPREAD, "IG_CORPORATE_SPREAD", +370.0);
    shocks.back().is_relative = false;
    shocks.back().affected_instruments = {"LQD", "CORPORATE_BONDS"};

    shocks.emplace_back(RiskFactorType::CREDIT_SPREAD, "HY_CORPORATE_SPREAD", +1080.0);
    shocks.back().is_relative = false;
    shocks.back().affected_instruments = {"HYG", "JNK"};

    // FX volatility and USD strength
    shocks.emplace_back(RiskFactorType::FX_RATE, "EUR/USD", -6.7);
    shocks.back().affected_instruments = {"EURUSD"};

    shocks.emplace_back(RiskFactorType::FX_RATE, "GBP/USD", -12.2);
    shocks.back().affected_instruments = {"GBPUSD"};

    // Extreme volatility spike
    shocks.emplace_back(RiskFactorType::VOLATILITY, "VIX", +395.0);
    shocks.back().affected_instruments = {"VIX", "VOLATILITY_INDICES"};

    // Oil price collapse
    shocks.emplace_back(RiskFactorType::COMMODITY, "WTI_CRUDE", -65.4);
    shocks.back().affected_instruments = {"WTI", "CRUDE_OIL"};

    // Gold surge
    shocks.emplace_back(RiskFactorType::COMMODITY, "GOLD", +12.8);
    shocks.back().affected_instruments = {"GOLD"};

    scenario.factor_shocks = std::move(shocks);
    scenarios_["2020_COVID19_Crisis"] = std::move(scenario);
}

void HistoricalScenarioDatabase::add_dot_com_crash_scenario() {
    HistoricalScenario scenario("2000_DotCom_Crash",
        "Technology bubble burst and NASDAQ crash");

    // March 2000 - October 2002
    std::tm start_tm = {};
    start_tm.tm_year = 100; // 2000
    start_tm.tm_mon = 2;    // March
    start_tm.tm_mday = 10;
    scenario.start_date = std::chrono::system_clock::from_time_t(std::mktime(&start_tm));

    std::tm end_tm = {};
    end_tm.tm_year = 102; // 2002
    end_tm.tm_mon = 9;    // October
    end_tm.tm_mday = 9;
    scenario.end_date = std::chrono::system_clock::from_time_t(std::mktime(&end_tm));

    scenario.duration_days = 944;
    scenario.severity_score = 8.5;
    scenario.affected_regions = {"North America", "Europe"};
    scenario.affected_sectors = {"Technology", "Telecommunications", "Media"};

    std::vector<RiskFactorShock> shocks;

    // Tech-heavy equity crashes
    shocks.emplace_back(RiskFactorType::EQUITY_INDEX, "NASDAQ", -78.0);
    shocks.back().time_horizon_days = 944;
    shocks.back().affected_instruments = {"QQQ", "TECH_STOCKS"};

    shocks.emplace_back(RiskFactorType::EQUITY_INDEX, "S&P 500", -49.1);
    shocks.back().affected_instruments = {"SPY", "US_LARGE_CAP"};

    // Sector-specific shocks
    shocks.emplace_back(RiskFactorType::EQUITY_INDEX, "TECHNOLOGY_SECTOR", -83.0);
    shocks.back().affected_instruments = {"XLK", "TECH_SECTOR"};

    shocks.emplace_back(RiskFactorType::EQUITY_INDEX, "TELECOM_SECTOR", -67.0);
    shocks.back().affected_instruments = {"XTL", "TELECOM_SECTOR"};

    // Interest rate environment (Fed cutting rates)
    shocks.emplace_back(RiskFactorType::INTEREST_RATE, "US_FED_FUNDS", -500.0);
    shocks.back().is_relative = false;
    shocks.back().affected_instruments = {"FED_FUNDS"};

    scenario.factor_shocks = std::move(shocks);
    scenarios_["2000_DotCom_Crash"] = std::move(scenario);
}

void HistoricalScenarioDatabase::add_european_debt_crisis_scenario() {
    HistoricalScenario scenario("2010_European_Debt_Crisis",
        "European sovereign debt crisis and banking sector stress");

    // May 2010 - July 2012
    std::tm start_tm = {};
    start_tm.tm_year = 110; // 2010
    start_tm.tm_mon = 4;    // May
    start_tm.tm_mday = 6;
    scenario.start_date = std::chrono::system_clock::from_time_t(std::mktime(&start_tm));

    std::tm end_tm = {};
    end_tm.tm_year = 112; // 2012
    end_tm.tm_mon = 6;    // July
    end_tm.tm_mday = 26;
    scenario.end_date = std::chrono::system_clock::from_time_t(std::mktime(&end_tm));

    scenario.duration_days = 811;
    scenario.severity_score = 8.0;
    scenario.affected_regions = {"Europe", "Global"};
    scenario.affected_sectors = {"Financial", "Government", "Banking"};

    std::vector<RiskFactorShock> shocks;

    // European equity markets
    shocks.emplace_back(RiskFactorType::EQUITY_INDEX, "STOXX 600", -44.7);
    shocks.back().time_horizon_days = 811;
    shocks.back().affected_instruments = {"STOXX", "EU_LARGE_CAP"};

    shocks.emplace_back(RiskFactorType::EQUITY_INDEX, "DAX", -45.1);
    shocks.back().affected_instruments = {"DAX", "DE_LARGE_CAP"};

    // Sovereign bond spreads widening
    shocks.emplace_back(RiskFactorType::CREDIT_SPREAD, "ITALY_10Y_SPREAD", +490.0);
    shocks.back().is_relative = false;
    shocks.back().affected_instruments = {"IT10Y"};

    shocks.emplace_back(RiskFactorType::CREDIT_SPREAD, "SPAIN_10Y_SPREAD", +570.0);
    shocks.back().is_relative = false;
    shocks.back().affected_instruments = {"ES10Y"};

    shocks.emplace_back(RiskFactorType::CREDIT_SPREAD, "PORTUGAL_10Y_SPREAD", +1150.0);
    shocks.back().is_relative = false;
    shocks.back().affected_instruments = {"PT10Y"};

    // EUR weakening
    shocks.emplace_back(RiskFactorType::FX_RATE, "EUR/USD", -21.4);
    shocks.back().affected_instruments = {"EURUSD"};

    // Financial sector stress
    shocks.emplace_back(RiskFactorType::EQUITY_INDEX, "EU_BANKS", -58.3);
    shocks.back().affected_instruments = {"EU_BANK_SECTOR"};

    scenario.factor_shocks = std::move(shocks);
    scenarios_["2010_European_Debt_Crisis"] = std::move(scenario);
}

void HistoricalScenarioDatabase::add_flash_crash_scenario() {
    HistoricalScenario scenario("2010_Flash_Crash",
        "Sudden market crash due to algorithmic trading malfunction");

    // May 6, 2010 (single day event)
    std::tm crash_tm = {};
    crash_tm.tm_year = 110; // 2010
    crash_tm.tm_mon = 4;    // May
    crash_tm.tm_mday = 6;
    scenario.start_date = std::chrono::system_clock::from_time_t(std::mktime(&crash_tm));
    scenario.end_date = scenario.start_date;

    scenario.duration_days = 1;
    scenario.severity_score = 7.0;
    scenario.affected_regions = {"North America"};
    scenario.affected_sectors = {"All Sectors"};

    std::vector<RiskFactorShock> shocks;

    // Rapid intraday crash and recovery
    shocks.emplace_back(RiskFactorType::EQUITY_INDEX, "S&P 500", -5.7);
    shocks.back().time_horizon_days = 1;
    shocks.back().recovery_half_life = 0.1; // Very fast recovery
    shocks.back().affected_instruments = {"SPY", "ES"};

    shocks.emplace_back(RiskFactorType::EQUITY_INDEX, "DOW_JONES", -6.1);
    shocks.back().recovery_half_life = 0.1;
    shocks.back().affected_instruments = {"DIA", "YM"};

    // Extreme volatility spike
    shocks.emplace_back(RiskFactorType::VOLATILITY, "VIX", +64.0);
    shocks.back().recovery_half_life = 0.1;
    shocks.back().affected_instruments = {"VIX"};

    // Liquidity crisis
    shocks.emplace_back(RiskFactorType::LIQUIDITY, "MARKET_LIQUIDITY", -90.0);
    shocks.back().recovery_half_life = 0.1;
    shocks.back().affected_instruments = {"ALL_EQUITY"};

    scenario.factor_shocks = std::move(shocks);
    scenarios_["2010_Flash_Crash"] = std::move(scenario);
}

void HistoricalScenarioDatabase::add_brexit_scenario() {
    HistoricalScenario scenario("2016_Brexit_Vote",
        "UK referendum vote to leave European Union");

    // June 23-24, 2016
    std::tm start_tm = {};
    start_tm.tm_year = 116; // 2016
    start_tm.tm_mon = 5;    // June
    start_tm.tm_mday = 23;
    scenario.start_date = std::chrono::system_clock::from_time_t(std::mktime(&start_tm));

    std::tm end_tm = {};
    end_tm.tm_year = 116; // 2016
    end_tm.tm_mon = 5;    // June
    end_tm.tm_mday = 27;
    scenario.end_date = std::chrono::system_clock::from_time_t(std::mktime(&end_tm));

    scenario.duration_days = 4;
    scenario.severity_score = 6.5;
    scenario.affected_regions = {"Europe", "UK"};
    scenario.affected_sectors = {"Financial", "Banking", "Real Estate"};

    std::vector<RiskFactorShock> shocks;

    // UK equity market crash
    shocks.emplace_back(RiskFactorType::EQUITY_INDEX, "FTSE 100", -5.6);
    shocks.back().time_horizon_days = 4;
    shocks.back().affected_instruments = {"UKX", "UK_LARGE_CAP"};

    // GBP collapse
    shocks.emplace_back(RiskFactorType::FX_RATE, "GBP/USD", -11.1);
    shocks.back().affected_instruments = {"GBPUSD", "GBP_CROSSES"};

    shocks.emplace_back(RiskFactorType::FX_RATE, "EUR/GBP", +8.2);
    shocks.back().affected_instruments = {"EURGBP"};

    // European banking sector
    shocks.emplace_back(RiskFactorType::EQUITY_INDEX, "EU_BANKS", -14.2);
    shocks.back().affected_instruments = {"EU_BANK_SECTOR"};

    // UK banking sector
    shocks.emplace_back(RiskFactorType::EQUITY_INDEX, "UK_BANKS", -17.9);
    shocks.back().affected_instruments = {"UK_BANK_SECTOR"};

    // Flight to quality
    shocks.emplace_back(RiskFactorType::INTEREST_RATE, "UK_10Y_GILT", -31.0);
    shocks.back().is_relative = false;
    shocks.back().affected_instruments = {"UK10Y"};

    scenario.factor_shocks = std::move(shocks);
    scenarios_["2016_Brexit_Vote"] = std::move(scenario);
}

void HistoricalScenarioDatabase::add_china_market_crash_scenario() {
    HistoricalScenario scenario("2015_China_Market_Crash",
        "Chinese stock market bubble burst and capital flight");

    // June 12 - August 26, 2015
    std::tm start_tm = {};
    start_tm.tm_year = 115; // 2015
    start_tm.tm_mon = 5;    // June
    start_tm.tm_mday = 12;
    scenario.start_date = std::chrono::system_clock::from_time_t(std::mktime(&start_tm));

    std::tm end_tm = {};
    end_tm.tm_year = 115; // 2015
    end_tm.tm_mon = 7;    // August
    end_tm.tm_mday = 26;
    scenario.end_date = std::chrono::system_clock::from_time_t(std::mktime(&end_tm));

    scenario.duration_days = 75;
    scenario.severity_score = 7.5;
    scenario.affected_regions = {"Asia", "China", "Global"};
    scenario.affected_sectors = {"Technology", "Consumer", "Industrial"};

    std::vector<RiskFactorShock> shocks;

    // Chinese equity markets
    shocks.emplace_back(RiskFactorType::EQUITY_INDEX, "SHANGHAI_COMPOSITE", -43.1);
    shocks.back().time_horizon_days = 75;
    shocks.back().affected_instruments = {"SHCOMP", "CN_LARGE_CAP"};

    shocks.emplace_back(RiskFactorType::EQUITY_INDEX, "SHENZHEN_COMPONENT", -45.3);
    shocks.back().affected_instruments = {"SZCOMP", "CN_TECH"};

    // Global contagion
    shocks.emplace_back(RiskFactorType::EQUITY_INDEX, "S&P 500", -12.4);
    shocks.back().affected_instruments = {"SPY", "US_LARGE_CAP"};

    shocks.emplace_back(RiskFactorType::EQUITY_INDEX, "FTSE 100", -15.2);
    shocks.back().affected_instruments = {"UKX", "UK_LARGE_CAP"};

    // Commodity impact
    shocks.emplace_back(RiskFactorType::COMMODITY, "COPPER", -23.8);
    shocks.back().affected_instruments = {"COPPER", "INDUSTRIAL_METALS"};

    shocks.emplace_back(RiskFactorType::COMMODITY, "IRON_ORE", -32.1);
    shocks.back().affected_instruments = {"IRON_ORE"};

    // CNY devaluation
    shocks.emplace_back(RiskFactorType::FX_RATE, "USD/CNY", +4.6);
    shocks.back().affected_instruments = {"USDCNY", "CNY_CROSSES"};

    scenario.factor_shocks = std::move(shocks);
    scenarios_["2015_China_Market_Crash"] = std::move(scenario);
}

void HistoricalScenarioDatabase::add_russia_ukraine_scenario() {
    HistoricalScenario scenario("2022_Russia_Ukraine_War",
        "Russian invasion of Ukraine and subsequent sanctions");

    // February 24, 2022 - ongoing
    std::tm start_tm = {};
    start_tm.tm_year = 122; // 2022
    start_tm.tm_mon = 1;    // February
    start_tm.tm_mday = 24;
    scenario.start_date = std::chrono::system_clock::from_time_t(std::mktime(&start_tm));

    std::tm end_tm = {};
    end_tm.tm_year = 122; // 2022
    end_tm.tm_mon = 5;    // June
    end_tm.tm_mday = 1;
    scenario.end_date = std::chrono::system_clock::from_time_t(std::mktime(&end_tm));

    scenario.duration_days = 97;
    scenario.severity_score = 8.5;
    scenario.affected_regions = {"Europe", "Russia", "Global"};
    scenario.affected_sectors = {"Energy", "Agriculture", "Defense", "Financial"};

    std::vector<RiskFactorShock> shocks;

    // Energy price surge
    shocks.emplace_back(RiskFactorType::COMMODITY, "NATURAL_GAS_EU", +265.0);
    shocks.back().time_horizon_days = 97;
    shocks.back().affected_instruments = {"NATURAL_GAS", "EU_GAS"};

    shocks.emplace_back(RiskFactorType::COMMODITY, "WTI_CRUDE", +45.2);
    shocks.back().affected_instruments = {"WTI", "CRUDE_OIL"};

    shocks.emplace_back(RiskFactorType::COMMODITY, "BRENT_CRUDE", +48.7);
    shocks.back().affected_instruments = {"BRENT", "CRUDE_OIL"};

    // Agricultural commodity surge
    shocks.emplace_back(RiskFactorType::COMMODITY, "WHEAT", +89.3);
    shocks.back().affected_instruments = {"WHEAT", "AGRICULTURE"};

    shocks.emplace_back(RiskFactorType::COMMODITY, "CORN", +45.1);
    shocks.back().affected_instruments = {"CORN", "AGRICULTURE"};

    // Russian market collapse
    shocks.emplace_back(RiskFactorType::EQUITY_INDEX, "MOEX_RUSSIA", -95.0);
    shocks.back().affected_instruments = {"MOEX", "RUSSIAN_EQUITY"};

    // European markets stress
    shocks.emplace_back(RiskFactorType::EQUITY_INDEX, "STOXX 600", -19.2);
    shocks.back().affected_instruments = {"STOXX", "EU_LARGE_CAP"};

    // Currency impacts
    shocks.emplace_back(RiskFactorType::FX_RATE, "USD/RUB", +89.4);
    shocks.back().affected_instruments = {"USDRUB"};

    shocks.emplace_back(RiskFactorType::FX_RATE, "EUR/USD", -8.1);
    shocks.back().affected_instruments = {"EURUSD"};

    // Safe haven flows
    shocks.emplace_back(RiskFactorType::COMMODITY, "GOLD", +12.8);
    shocks.back().affected_instruments = {"GOLD", "PRECIOUS_METALS"};

    scenario.factor_shocks = std::move(shocks);
    scenarios_["2022_Russia_Ukraine_War"] = std::move(scenario);
}

void HistoricalScenarioDatabase::add_swiss_franc_shock_scenario() {
    HistoricalScenario scenario("2015_Swiss_Franc_Shock",
        "Swiss National Bank abandons EUR/CHF floor");

    // January 15, 2015 (single day event)
    std::tm shock_tm = {};
    shock_tm.tm_year = 115; // 2015
    shock_tm.tm_mon = 0;    // January
    shock_tm.tm_mday = 15;
    scenario.start_date = std::chrono::system_clock::from_time_t(std::mktime(&shock_tm));
    scenario.end_date = scenario.start_date;

    scenario.duration_days = 1;
    scenario.severity_score = 8.0;
    scenario.affected_regions = {"Europe", "Switzerland"};
    scenario.affected_sectors = {"Financial", "Export-dependent"};

    std::vector<RiskFactorShock> shocks;

    // Massive CHF appreciation
    shocks.emplace_back(RiskFactorType::FX_RATE, "EUR/CHF", -15.8);
    shocks.back().time_horizon_days = 1;
    shocks.back().affected_instruments = {"EURCHF", "CHF_CROSSES"};

    shocks.emplace_back(RiskFactorType::FX_RATE, "USD/CHF", -13.2);
    shocks.back().affected_instruments = {"USDCHF"};

    // Swiss equity market impact
    shocks.emplace_back(RiskFactorType::EQUITY_INDEX, "SMI", -8.7);
    shocks.back().affected_instruments = {"SMI", "CH_LARGE_CAP"};

    scenario.factor_shocks = std::move(shocks);
    scenarios_["2015_Swiss_Franc_Shock"] = std::move(scenario);
}

void HistoricalScenarioDatabase::add_oil_price_crash_scenario() {
    HistoricalScenario scenario("2014_Oil_Price_Crash",
        "Oil price collapse due to oversupply and demand concerns");

    // June 2014 - February 2016
    std::tm start_tm = {};
    start_tm.tm_year = 114; // 2014
    start_tm.tm_mon = 5;    // June
    start_tm.tm_mday = 19;
    scenario.start_date = std::chrono::system_clock::from_time_t(std::mktime(&start_tm));

    std::tm end_tm = {};
    end_tm.tm_year = 116; // 2016
    end_tm.tm_mon = 1;    // February
    end_tm.tm_mday = 11;
    scenario.end_date = std::chrono::system_clock::from_time_t(std::mktime(&end_tm));

    scenario.duration_days = 603;
    scenario.severity_score = 7.0;
    scenario.affected_regions = {"Global", "Oil-producing countries"};
    scenario.affected_sectors = {"Energy", "Materials", "Transportation"};

    std::vector<RiskFactorShock> shocks;

    // Oil price collapse
    shocks.emplace_back(RiskFactorType::COMMODITY, "WTI_CRUDE", -76.3);
    shocks.back().time_horizon_days = 603;
    shocks.back().affected_instruments = {"WTI", "CRUDE_OIL"};

    shocks.emplace_back(RiskFactorType::COMMODITY, "BRENT_CRUDE", -75.1);
    shocks.back().affected_instruments = {"BRENT", "CRUDE_OIL"};

    // Energy sector equity crash
    shocks.emplace_back(RiskFactorType::EQUITY_INDEX, "ENERGY_SECTOR", -45.2);
    shocks.back().affected_instruments = {"XLE", "ENERGY_STOCKS"};

    // Currency impacts on oil exporters
    shocks.emplace_back(RiskFactorType::FX_RATE, "USD/CAD", +38.1);
    shocks.back().affected_instruments = {"USDCAD"};

    shocks.emplace_back(RiskFactorType::FX_RATE, "USD/NOK", +47.8);
    shocks.back().affected_instruments = {"USDNOK"};

    shocks.emplace_back(RiskFactorType::FX_RATE, "USD/RUB", +152.3);
    shocks.back().affected_instruments = {"USDRUB"};

    scenario.factor_shocks = std::move(shocks);
    scenarios_["2014_Oil_Price_Crash"] = std::move(scenario);
}

void HistoricalScenarioDatabase::load_historical_data() {
    // Load historical market data for scenario replay
    // This would typically load from external data sources
    // For demonstration, we'll create synthetic data based on known historical moves

    for (const auto& [scenario_name, scenario] : scenarios_) {
        std::unordered_map<std::string, std::vector<double>> scenario_data;

        // Generate time series data for each risk factor
        for (const auto& shock : scenario.factor_shocks) {
            std::vector<double> time_series;

            // Generate synthetic time series based on shock magnitude
            uint32_t num_points = std::max(1u, scenario.duration_days);

            for (uint32_t i = 0; i <= num_points; ++i) {
                double progress = static_cast<double>(i) / num_points;

                // Apply shock progression (linear for simplicity)
                double shock_value;
                if (shock.recovery_half_life > 0) {
                    // Shock with recovery
                    double peak_at = 0.1; // Peak at 10% of timeline
                    if (progress <= peak_at) {
                        shock_value = shock.shock_magnitude * (progress / peak_at);
                    } else {
                        double recovery_progress = (progress - peak_at) / (1.0 - peak_at);
                        shock_value = shock.shock_magnitude * std::exp(-recovery_progress / shock.recovery_half_life);
                    }
                } else {
                    // Monotonic shock
                    shock_value = shock.shock_magnitude * progress;
                }

                time_series.push_back(shock_value);
            }

            scenario_data[shock.factor_name] = std::move(time_series);
        }

        historical_data_[scenario_name] = std::move(scenario_data);
    }
}

const HistoricalScenario* HistoricalScenarioDatabase::get_scenario(const std::string& scenario_name) const {
    auto it = scenarios_.find(scenario_name);
    return (it != scenarios_.end()) ? &it->second : nullptr;
}

std::vector<std::string> HistoricalScenarioDatabase::list_scenarios() const {
    std::vector<std::string> scenario_names;
    scenario_names.reserve(scenarios_.size());

    for (const auto& [name, scenario] : scenarios_) {
        scenario_names.push_back(name);
    }

    std::sort(scenario_names.begin(), scenario_names.end());
    return scenario_names;
}

std::vector<double> HistoricalScenarioDatabase::get_historical_data(
    const std::string& scenario_name,
    const std::string& risk_factor,
    const std::chrono::system_clock::time_point& start_date,
    const std::chrono::system_clock::time_point& end_date) const {

    auto scenario_it = historical_data_.find(scenario_name);
    if (scenario_it == historical_data_.end()) {
        return {};
    }

    auto factor_it = scenario_it->second.find(risk_factor);
    if (factor_it == scenario_it->second.end()) {
        return {};
    }

    // For simplicity, return the full time series
    // In practice, we would filter based on dates
    return factor_it->second;
}

void HistoricalScenarioDatabase::add_scenario(const HistoricalScenario& scenario) {
    scenarios_[scenario.scenario_name] = scenario;
}

std::vector<RiskFactorShock> HistoricalScenarioDatabase::calculate_historical_shocks(
    const std::string& scenario_name,
    const std::vector<std::string>& risk_factors) const {

    const auto* scenario = get_scenario(scenario_name);
    if (!scenario) {
        return {};
    }

    std::vector<RiskFactorShock> shocks;

    for (const auto& factor_name : risk_factors) {
        // Find matching shock in scenario
        auto shock_it = std::find_if(scenario->factor_shocks.begin(), scenario->factor_shocks.end(),
            [&factor_name](const RiskFactorShock& shock) {
                return shock.factor_name == factor_name;
            });

        if (shock_it != scenario->factor_shocks.end()) {
            shocks.push_back(*shock_it);
        }
    }

    return shocks;
}

} // namespace stress_testing
} // namespace risk_analytics