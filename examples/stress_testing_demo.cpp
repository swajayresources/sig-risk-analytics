/**
 * Comprehensive Stress Testing Framework Demo
 *
 * This demo showcases the complete stress testing and scenario analysis framework
 * including historical scenarios, Monte Carlo simulation, regulatory compliance,
 * multi-factor analysis, real-time execution, and visualization capabilities.
 */

#include "stress_testing/stress_framework.hpp"
#include "hpc/hpc_framework.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>

using namespace stress_testing;

class StressTestingDemo {
private:
    // Core framework components
    std::unique_ptr<HistoricalScenarioEngine> historical_engine_;
    std::unique_ptr<ScenarioGenerator> scenario_generator_;
    std::unique_ptr<ShockConstructor> shock_constructor_;
    std::unique_ptr<MonteCarloStressTester> monte_carlo_tester_;
    std::unique_ptr<MultiFactorStressTester> multi_factor_tester_;
    std::unique_ptr<RegulatoryComplianceEngine> regulatory_engine_;
    std::unique_ptr<RealTimeStressExecutor> realtime_executor_;
    std::unique_ptr<VisualizationReportingEngine> visualization_engine_;

    // Sample risk factors
    std::vector<RiskFactor> risk_factors_;

public:
    StressTestingDemo() {
        initialize_components();
        setup_risk_factors();
    }

    void run_comprehensive_demo() {
        std::cout << "=== Comprehensive Stress Testing Framework Demo ===" << std::endl;
        std::cout << "Starting comprehensive stress testing demonstration..." << std::endl << std::endl;

        // 1. Historical Scenario Analysis
        std::cout << "1. Running Historical Scenario Analysis..." << std::endl;
        run_historical_scenario_demo();
        std::cout << std::endl;

        // 2. Custom Scenario Generation
        std::cout << "2. Running Custom Scenario Generation..." << std::endl;
        run_custom_scenario_demo();
        std::cout << std::endl;

        // 3. Monte Carlo Stress Testing
        std::cout << "3. Running Monte Carlo Stress Testing..." << std::endl;
        run_monte_carlo_demo();
        std::cout << std::endl;

        // 4. Multi-Factor Stress Testing
        std::cout << "4. Running Multi-Factor Stress Testing..." << std::endl;
        run_multi_factor_demo();
        std::cout << std::endl;

        // 5. Regulatory Compliance Testing
        std::cout << "5. Running Regulatory Compliance Testing..." << std::endl;
        run_regulatory_compliance_demo();
        std::cout << std::endl;

        // 6. Real-Time Execution
        std::cout << "6. Running Real-Time Execution Demo..." << std::endl;
        run_realtime_execution_demo();
        std::cout << std::endl;

        // 7. Visualization and Reporting
        std::cout << "7. Running Visualization and Reporting Demo..." << std::endl;
        run_visualization_demo();
        std::cout << std::endl;

        std::cout << "=== Demo Complete ===" << std::endl;
        std::cout << "All stress testing framework components successfully demonstrated!" << std::endl;
    }

private:
    void initialize_components() {
        historical_engine_ = std::make_unique<HistoricalScenarioEngine>();
        scenario_generator_ = std::make_unique<ScenarioGenerator>();
        shock_constructor_ = std::make_unique<ShockConstructor>();
        monte_carlo_tester_ = std::make_unique<MonteCarloStressTester>();
        multi_factor_tester_ = std::make_unique<MultiFactorStressTester>();
        regulatory_engine_ = std::make_unique<RegulatoryComplianceEngine>();
        realtime_executor_ = std::make_unique<RealTimeStressExecutor>();
        visualization_engine_ = std::make_unique<VisualizationReportingEngine>();
    }

    void setup_risk_factors() {
        // Equity factors
        risk_factors_.push_back({"EQ_SPX", "S&P 500 Index", RiskFactorCategory::EQUITY});
        risk_factors_.push_back({"EQ_NDX", "NASDAQ 100 Index", RiskFactorCategory::EQUITY});
        risk_factors_.push_back({"EQ_RTY", "Russell 2000 Index", RiskFactorCategory::EQUITY});

        // Interest rate factors
        risk_factors_.push_back({"IR_USD_2Y", "USD 2Y Treasury Rate", RiskFactorCategory::INTEREST_RATE});
        risk_factors_.push_back({"IR_USD_10Y", "USD 10Y Treasury Rate", RiskFactorCategory::INTEREST_RATE});
        risk_factors_.push_back({"IR_USD_30Y", "USD 30Y Treasury Rate", RiskFactorCategory::INTEREST_RATE});

        // FX factors
        risk_factors_.push_back({"FX_EURUSD", "EUR/USD Exchange Rate", RiskFactorCategory::FX});
        risk_factors_.push_back({"FX_USDJPY", "USD/JPY Exchange Rate", RiskFactorCategory::FX});
        risk_factors_.push_back({"FX_GBPUSD", "GBP/USD Exchange Rate", RiskFactorCategory::FX});

        // Credit factors
        risk_factors_.push_back({"CR_IG_SPREAD", "Investment Grade Credit Spread", RiskFactorCategory::CREDIT});
        risk_factors_.push_back({"CR_HY_SPREAD", "High Yield Credit Spread", RiskFactorCategory::CREDIT});

        // Volatility factors
        risk_factors_.push_back({"VOL_VIX", "VIX Volatility Index", RiskFactorCategory::VOLATILITY});
        risk_factors_.push_back({"VOL_MOVE", "MOVE Bond Volatility Index", RiskFactorCategory::VOLATILITY});

        // Commodity factors
        risk_factors_.push_back({"COM_WTI", "WTI Crude Oil", RiskFactorCategory::COMMODITY});
        risk_factors_.push_back({"COM_GOLD", "Gold Spot Price", RiskFactorCategory::COMMODITY});
    }

    void run_historical_scenario_demo() {
        std::cout << "  Loading historical crisis scenarios..." << std::endl;

        // Configure historical engine
        HistoricalScenarioConfig config;
        config.include_2008_crisis = true;
        config.include_covid_crisis = true;
        config.include_brexit = true;
        config.data_frequency = DataFrequency::DAILY;

        historical_engine_->configure(config);

        // Get available scenarios
        auto scenarios = historical_engine_->get_available_scenarios();
        std::cout << "  Found " << scenarios.size() << " historical scenarios" << std::endl;

        // Replay 2008 Financial Crisis
        HistoricalReplayParameters replay_params;
        replay_params.scenario_name = "2008_Financial_Crisis";
        replay_params.start_date = "2008-09-15"; // Lehman collapse
        replay_params.end_date = "2009-03-09";   // Market bottom
        replay_params.stress_multiplier = 1.0;

        auto replay_results = historical_engine_->replay_historical_scenario(
            risk_factors_, replay_params);

        std::cout << "  2008 Crisis Replay Results:" << std::endl;
        std::cout << "    Total P&L Impact: $" << (replay_results.total_pnl_impact / 1000000.0) << "M" << std::endl;
        std::cout << "    Maximum Drawdown: $" << (replay_results.maximum_drawdown / 1000000.0) << "M" << std::endl;
        std::cout << "    Recovery Time: " << replay_results.recovery_time_days << " days" << std::endl;
    }

    void run_custom_scenario_demo() {
        std::cout << "  Generating custom stress scenarios..." << std::endl;

        // Configure scenario generator
        ScenarioConfig config;
        config.random_seed = 12345;
        config.use_importance_sampling = true;

        scenario_generator_->configure(config);

        // Generate Monte Carlo scenarios
        MonteCarloParameters mc_params;
        mc_params.num_scenarios = 1000;
        mc_params.time_horizon_days = 252; // 1 year
        mc_params.volatility_scaling = 1.5;
        mc_params.base_seed = 67890;

        auto mc_scenarios = scenario_generator_->generate_monte_carlo_scenarios(
            risk_factors_, mc_params);

        std::cout << "  Generated " << mc_scenarios.size() << " Monte Carlo scenarios" << std::endl;

        // Generate tail risk scenario
        TailRiskParameters tail_params;
        tail_params.name = "Extreme_Tail_Event";
        tail_params.confidence_level = 0.99;
        tail_params.tail_probability = 0.01;
        tail_params.extreme_multiplier = 3.0;

        auto tail_scenario = scenario_generator_->generate_tail_risk_scenario(
            risk_factors_, tail_params);

        std::cout << "  Tail Risk Scenario Results:" << std::endl;
        std::cout << "    Expected Shortfall: $" << (tail_scenario.expected_shortfall / 1000000.0) << "M" << std::endl;
        std::cout << "    Maximum Drawdown: $" << (tail_scenario.maximum_drawdown / 1000000.0) << "M" << std::endl;
    }

    void run_monte_carlo_demo() {
        std::cout << "  Running Monte Carlo stress testing..." << std::endl;

        // Configure variance reduction techniques
        monte_carlo_tester_->configure_variance_reduction(true, false, false);

        // Set up Monte Carlo parameters
        MonteCarloStressParameters mc_params;
        mc_params.num_scenarios = 10000;
        mc_params.time_horizon_days = 252;
        mc_params.confidence_levels = {0.95, 0.99, 0.999};
        mc_params.random_seed = 42;

        // Run Monte Carlo stress test
        auto mc_results = monte_carlo_tester_->run_monte_carlo_stress_test(
            risk_factors_, mc_params);

        std::cout << "  Monte Carlo Results (" << mc_results.num_scenarios << " scenarios):" << std::endl;
        std::cout << "    Mean P&L: $" << (mc_results.mean_pnl / 1000000.0) << "M" << std::endl;
        std::cout << "    P&L Std Dev: $" << (mc_results.pnl_std_deviation / 1000000.0) << "M" << std::endl;

        for (auto& [confidence, var] : mc_results.tail_metrics.var_estimates) {
            std::cout << "    VaR " << (confidence * 100) << "%: $" << (var / 1000000.0) << "M" << std::endl;
        }

        std::cout << "    Skewness: " << mc_results.tail_metrics.skewness << std::endl;
        std::cout << "    Excess Kurtosis: " << mc_results.tail_metrics.excess_kurtosis << std::endl;
    }

    void run_multi_factor_demo() {
        std::cout << "  Running multi-factor stress testing..." << std::endl;

        // Create sample historical returns matrix (simplified)
        Eigen::MatrixXd historical_returns = Eigen::MatrixXd::Random(252, risk_factors_.size()) * 0.02;

        // Calibrate multi-factor model
        MultiFactorModelParams model_params;
        model_params.decomposition_method = FactorDecompositionMethod::PRINCIPAL_COMPONENT;
        model_params.max_factors = 5;
        model_params.explained_variance_threshold = 0.95;

        multi_factor_tester_->calibrate_multi_factor_model(
            "PCA_Model", risk_factors_, historical_returns, model_params);

        // Run multi-factor stress test
        MultiFactorStressParameters stress_params;
        stress_params.model_name = "PCA_Model";
        stress_params.test_name = "Multi_Factor_Stress";
        stress_params.stress_type = MultiFactorStressType::ALL_FACTORS;
        stress_params.shock_magnitudes = {1.0, 2.0, 3.0};

        auto mf_results = multi_factor_tester_->run_multi_factor_stress_test(
            risk_factors_, stress_params);

        std::cout << "  Multi-Factor Results:" << std::endl;
        std::cout << "    Test Name: " << mf_results.test_name << std::endl;
        std::cout << "    Number of Scenarios: " << mf_results.scenario_results.size() << std::endl;

        if (!mf_results.scenario_results.empty()) {
            const auto& first_result = mf_results.scenario_results[0];
            std::cout << "    Sample Scenario P&L: $" << (first_result.total_pnl / 1000000.0) << "M" << std::endl;
        }

        // Perform factor decomposition
        FactorDecompositionParams decomp_params;
        decomp_params.factor_selection_method = FactorSelectionMethod::EXPLAINED_VARIANCE;
        decomp_params.explained_variance_threshold = 0.95;
        decomp_params.max_factors = 10;

        auto decomp_results = multi_factor_tester_->decompose_risk_factors(
            risk_factors_, historical_returns, decomp_params);

        std::cout << "  Factor Decomposition:" << std::endl;
        std::cout << "    Number of Factors: " << decomp_results.factor_names.size() << std::endl;
        std::cout << "    Total Explained Variance: " << (decomp_results.explained_variance[0] * 100) << "%" << std::endl;
    }

    void run_regulatory_compliance_demo() {
        std::cout << "  Running regulatory compliance testing..." << std::endl;

        // Configure CCAR testing
        RegulatoryStressParameters ccar_params;
        ccar_params.regime = RegulatoryRegime::CCAR;
        ccar_params.test_date = "2024-03-31";
        ccar_params.institution_id = "DEMO_BANK_001";

        auto ccar_results = regulatory_engine_->run_regulatory_stress_test(
            risk_factors_, ccar_params);

        std::cout << "  CCAR Results:" << std::endl;
        std::cout << "    Overall Compliance: " << (ccar_results.overall_compliance_status ? "PASS" : "FAIL") << std::endl;
        std::cout << "    Post-Stress Capital Ratio: " << (ccar_results.post_stress_capital_ratio * 100) << "%" << std::endl;
        std::cout << "    Number of Scenarios Tested: " << ccar_results.scenario_results.size() << std::endl;

        // Configure EBA testing
        EBAParameters eba_params;
        eba_params.test_year = 2024;
        eba_params.institution_name = "Demo Bank";
        eba_params.country_code = "US";

        auto eba_results = regulatory_engine_->run_eba_stress_test(risk_factors_, eba_params);

        std::cout << "  EBA Results:" << std::endl;
        std::cout << "    Institution: " << eba_results.institution_name << std::endl;
        std::cout << "    Test Year: " << eba_results.test_year << std::endl;

        // Generate regulatory report
        ReportingParameters report_params;
        report_params.include_charts = true;
        report_params.include_detailed_analysis = true;

        auto regulatory_report = regulatory_engine_->generate_regulatory_report(ccar_results, report_params);
        std::cout << "  Generated regulatory compliance report (" << regulatory_report.length() << " characters)" << std::endl;
    }

    void run_realtime_execution_demo() {
        std::cout << "  Starting real-time execution engine..." << std::endl;

        // Configure real-time execution
        RealTimeExecutionConfig rt_config;
        rt_config.max_concurrent_stress_tests = 4;
        rt_config.execution_thread_pool_size = 8;
        rt_config.enable_market_data_integration = true;
        rt_config.auto_recalculate_on_market_data_update = true;

        realtime_executor_->configure_execution(rt_config);
        realtime_executor_->start_execution();

        // Submit multiple stress test jobs
        std::vector<std::string> job_ids;

        for (int i = 0; i < 3; ++i) {
            RealTimeStressRequest request;
            request.request_id = "RT_Test_" + std::to_string(i + 1);
            request.stress_test_type = static_cast<StressTestType>(i % 5);
            request.priority = (i == 0) ? TaskPriority::HIGH : TaskPriority::NORMAL;

            std::string job_id = realtime_executor_->submit_stress_test(risk_factors_, request);
            job_ids.push_back(job_id);
            std::cout << "    Submitted job: " << job_id << std::endl;
        }

        // Wait for jobs to complete
        std::cout << "  Waiting for jobs to complete..." << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(2));

        // Get execution statistics
        auto exec_stats = realtime_executor_->get_execution_statistics();
        std::cout << "  Execution Statistics:" << std::endl;
        std::cout << "    Total Jobs: " << exec_stats.total_jobs_submitted << std::endl;
        std::cout << "    Completed: " << exec_stats.jobs_completed << std::endl;
        std::cout << "    Running: " << exec_stats.jobs_running << std::endl;
        std::cout << "    Average Execution Time: " << exec_stats.average_execution_time_ms << "ms" << std::endl;

        // Get market data status
        auto market_status = realtime_executor_->get_market_data_status();
        std::cout << "  Market Data Status:" << std::endl;
        std::cout << "    Connected: " << (market_status.is_connected ? "Yes" : "No") << std::endl;
        std::cout << "    Active Feeds: " << market_status.num_active_feeds << std::endl;

        realtime_executor_->stop_execution();
        std::cout << "  Real-time execution engine stopped" << std::endl;
    }

    void run_visualization_demo() {
        std::cout << "  Generating visualizations and reports..." << std::endl;

        // Create sample stress test results
        std::vector<RealTimeStressResult> sample_results = create_sample_results();

        // Generate interactive dashboard
        DashboardParameters dashboard_params;
        dashboard_params.title = "Risk Management Dashboard";
        dashboard_params.theme = "dark";
        dashboard_params.auto_refresh_interval_seconds = 30;
        dashboard_params.include_real_time_monitoring = true;

        auto dashboard_html = visualization_engine_->generate_stress_test_dashboard(
            sample_results, dashboard_params);

        std::cout << "  Generated interactive dashboard (" << dashboard_html.length() << " characters)" << std::endl;

        // Create sample compliance results
        ComplianceResults compliance_results;
        compliance_results.overall_compliance_status = true;
        compliance_results.post_stress_capital_ratio = 0.085;

        // Generate comprehensive report
        ComprehensiveReportParameters report_params;
        report_params.title = "Comprehensive Stress Testing Report";
        report_params.include_executive_summary = true;
        report_params.include_detailed_analysis = true;
        report_params.include_charts = true;

        auto comprehensive_report = visualization_engine_->generate_comprehensive_report(
            compliance_results, sample_results, report_params);

        std::cout << "  Generated comprehensive report (" << comprehensive_report.length() << " characters)" << std::endl;

        // Export to Excel
        ExcelExportParameters excel_params;
        excel_params.include_charts = true;
        excel_params.include_pivot_tables = true;

        auto excel_content = visualization_engine_->export_results_to_excel(sample_results, excel_params);
        std::cout << "  Generated Excel export (" << excel_content.length() << " characters)" << std::endl;

        // Generate risk metrics visualization
        RiskMetricsVisualizationParameters viz_params;
        viz_params.include_distribution_analysis = true;
        viz_params.include_factor_analysis = true;

        auto risk_viz = visualization_engine_->generate_risk_metrics_visualization(sample_results, viz_params);
        std::cout << "  Generated risk metrics visualization (" << risk_viz.length() << " characters)" << std::endl;
    }

    std::vector<RealTimeStressResult> create_sample_results() {
        std::vector<RealTimeStressResult> results;

        for (int i = 0; i < 5; ++i) {
            RealTimeStressResult result;
            result.job_id = "sample_job_" + std::to_string(i);
            result.status = ExecutionStatus::COMPLETED;
            result.total_pnl = -1000000.0 * (i + 1);
            result.var_95 = 2500000.0 * (i + 1);
            result.expected_shortfall = 3500000.0 * (i + 1);
            result.compliance_status = (i < 3);
            result.execution_time_ms = 1500 + (i * 200);

            result.request.request_id = "Sample_Test_" + std::to_string(i);
            result.request.stress_test_type = static_cast<StressTestType>(i % 5);

            results.push_back(result);
        }

        return results;
    }
};

int main() {
    try {
        std::cout << "Initializing Comprehensive Stress Testing Framework..." << std::endl;
        std::cout << "=================================================" << std::endl << std::endl;

        StressTestingDemo demo;
        demo.run_comprehensive_demo();

        std::cout << std::endl << "Demo completed successfully!" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Demo failed with error: " << e.what() << std::endl;
        return 1;
    }
}