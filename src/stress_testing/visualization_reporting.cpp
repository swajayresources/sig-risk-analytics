#include "stress_testing/stress_framework.hpp"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace stress_testing {

class VisualizationReportingEngineImpl {
private:
    // Chart and visualization configurations
    std::unordered_map<std::string, ChartConfig> chart_templates_;
    std::unordered_map<std::string, DashboardConfig> dashboard_templates_;

    // Report templates
    std::unordered_map<std::string, ReportTemplate> report_templates_;

    // Styling and themes
    std::unordered_map<std::string, VisualizationTheme> themes_;

public:
    VisualizationReportingEngineImpl() {
        initialize_chart_templates();
        initialize_dashboard_templates();
        initialize_report_templates();
        initialize_themes();
    }

    std::string generate_stress_test_dashboard(
        const std::vector<RealTimeStressResult>& results,
        const DashboardParameters& params) {

        std::stringstream dashboard_html;

        // Generate HTML dashboard header
        generate_dashboard_header(dashboard_html, params);

        // Generate executive summary section
        generate_executive_summary_section(dashboard_html, results, params);

        // Generate real-time monitoring section
        generate_real_time_monitoring_section(dashboard_html, results, params);

        // Generate stress test results grid
        generate_results_grid_section(dashboard_html, results, params);

        // Generate risk factor analysis section
        generate_risk_factor_analysis_section(dashboard_html, results, params);

        // Generate performance metrics section
        generate_performance_metrics_section(dashboard_html, results, params);

        // Generate historical comparison section
        if (params.include_historical_comparison) {
            generate_historical_comparison_section(dashboard_html, results, params);
        }

        // Generate charts and visualizations
        generate_charts_section(dashboard_html, results, params);

        // Generate dashboard footer with JavaScript
        generate_dashboard_footer(dashboard_html, params);

        return dashboard_html.str();
    }

    std::string generate_comprehensive_report(
        const ComplianceResults& compliance_results,
        const std::vector<RealTimeStressResult>& stress_results,
        const ComprehensiveReportParameters& params) {

        std::stringstream report;

        // Report header and title page
        generate_report_title_page(report, params);

        // Table of contents
        if (params.include_table_of_contents) {
            generate_table_of_contents(report, params);
        }

        // Executive summary
        generate_executive_summary_report(report, compliance_results, stress_results, params);

        // Methodology section
        generate_methodology_section_report(report, params);

        // Regulatory compliance analysis
        generate_regulatory_compliance_section(report, compliance_results, params);

        // Stress testing results analysis
        generate_stress_testing_results_section(report, stress_results, params);

        // Risk factor analysis
        generate_detailed_risk_factor_analysis(report, stress_results, params);

        // Scenario analysis
        generate_scenario_analysis_section(report, stress_results, params);

        // Portfolio impact analysis
        generate_portfolio_impact_section(report, stress_results, params);

        // Model validation and backtesting
        if (params.include_model_validation) {
            generate_model_validation_section_report(report, stress_results, params);
        }

        // Recommendations and action items
        generate_recommendations_section(report, compliance_results, stress_results, params);

        // Appendices
        generate_appendices_section_report(report, stress_results, params);

        return report.str();
    }

    std::string create_interactive_chart(
        const ChartData& data,
        const ChartConfig& config) {

        std::stringstream chart_html;

        // Generate chart container
        chart_html << "<div id=\"" << config.chart_id << "\" class=\"chart-container\" "
                   << "style=\"width: " << config.width << "px; height: " << config.height << "px;\">\n";

        // Generate chart based on type
        switch (config.chart_type) {
            case ChartType::LINE_CHART:
                generate_line_chart(chart_html, data, config);
                break;
            case ChartType::BAR_CHART:
                generate_bar_chart(chart_html, data, config);
                break;
            case ChartType::SCATTER_PLOT:
                generate_scatter_plot(chart_html, data, config);
                break;
            case ChartType::HISTOGRAM:
                generate_histogram(chart_html, data, config);
                break;
            case ChartType::HEATMAP:
                generate_heatmap(chart_html, data, config);
                break;
            case ChartType::BOX_PLOT:
                generate_box_plot(chart_html, data, config);
                break;
            case ChartType::WATERFALL:
                generate_waterfall_chart(chart_html, data, config);
                break;
            case ChartType::TREEMAP:
                generate_treemap(chart_html, data, config);
                break;
        }

        chart_html << "</div>\n";

        // Generate chart JavaScript
        generate_chart_javascript(chart_html, data, config);

        return chart_html.str();
    }

    std::string export_results_to_excel(
        const std::vector<RealTimeStressResult>& results,
        const ExcelExportParameters& params) {

        std::stringstream excel_xml;

        // Generate Excel XML header
        excel_xml << "<?xml version=\"1.0\"?>\n";
        excel_xml << "<Workbook xmlns=\"urn:schemas-microsoft-com:office:spreadsheet\"\n";
        excel_xml << " xmlns:o=\"urn:schemas-microsoft-com:office:office\"\n";
        excel_xml << " xmlns:x=\"urn:schemas-microsoft-com:office:excel\"\n";
        excel_xml << " xmlns:ss=\"urn:schemas-microsoft-com:office:spreadsheet\"\n";
        excel_xml << " xmlns:html=\"http://www.w3.org/TR/REC-html40\">\n";

        // Generate styles
        generate_excel_styles(excel_xml);

        // Generate summary worksheet
        generate_excel_summary_worksheet(excel_xml, results, params);

        // Generate detailed results worksheet
        generate_excel_detailed_worksheet(excel_xml, results, params);

        // Generate charts worksheet
        if (params.include_charts) {
            generate_excel_charts_worksheet(excel_xml, results, params);
        }

        // Generate pivot table worksheet
        if (params.include_pivot_tables) {
            generate_excel_pivot_worksheet(excel_xml, results, params);
        }

        excel_xml << "</Workbook>\n";

        return excel_xml.str();
    }

    void save_visualization_to_file(
        const std::string& content,
        const std::string& file_path,
        VisualizationFormat format) {

        std::ofstream file(file_path);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file for writing: " + file_path);
        }

        switch (format) {
            case VisualizationFormat::HTML:
                file << content;
                break;
            case VisualizationFormat::PDF:
                // Convert HTML to PDF (would require external library)
                save_as_pdf(content, file_path);
                break;
            case VisualizationFormat::PNG:
                // Convert chart to PNG (would require rendering engine)
                save_as_png(content, file_path);
                break;
            case VisualizationFormat::SVG:
                // Convert chart to SVG
                save_as_svg(content, file_path);
                break;
        }

        file.close();
    }

    std::string generate_risk_metrics_visualization(
        const std::vector<RealTimeStressResult>& results,
        const RiskMetricsVisualizationParameters& params) {

        std::stringstream viz_html;

        // Generate container
        viz_html << "<div class=\"risk-metrics-dashboard\">\n";

        // VaR and ES comparison chart
        generate_var_es_comparison_chart(viz_html, results, params);

        // P&L distribution chart
        generate_pnl_distribution_chart(viz_html, results, params);

        // Factor contribution waterfall chart
        generate_factor_contribution_chart(viz_html, results, params);

        // Risk factor heatmap
        generate_risk_factor_heatmap(viz_html, results, params);

        // Time series performance chart
        generate_time_series_chart(viz_html, results, params);

        viz_html << "</div>\n";

        return viz_html.str();
    }

    std::string create_regulatory_compliance_report(
        const ComplianceResults& results,
        const RegulatoryReportParameters& params) {

        std::stringstream report;

        // Report header
        generate_regulatory_report_header(report, results, params);

        // Compliance summary table
        generate_compliance_summary_table(report, results, params);

        // Capital adequacy charts
        generate_capital_adequacy_charts(report, results, params);

        // Scenario impact analysis
        generate_scenario_impact_charts(report, results, params);

        // Regulatory metrics dashboard
        generate_regulatory_metrics_dashboard(report, results, params);

        // Compliance trend analysis
        if (params.include_trend_analysis) {
            generate_compliance_trend_analysis(report, results, params);
        }

        return report.str();
    }

private:
    void initialize_chart_templates() {
        // Initialize standard chart configurations
        ChartConfig pnl_distribution_config;
        pnl_distribution_config.chart_id = "pnl_distribution";
        pnl_distribution_config.chart_type = ChartType::HISTOGRAM;
        pnl_distribution_config.title = "P&L Distribution";
        pnl_distribution_config.width = 800;
        pnl_distribution_config.height = 400;
        pnl_distribution_config.x_axis_label = "P&L ($)";
        pnl_distribution_config.y_axis_label = "Frequency";
        chart_templates_["pnl_distribution"] = pnl_distribution_config;

        ChartConfig var_timeline_config;
        var_timeline_config.chart_id = "var_timeline";
        var_timeline_config.chart_type = ChartType::LINE_CHART;
        var_timeline_config.title = "VaR Over Time";
        var_timeline_config.width = 1000;
        var_timeline_config.height = 300;
        var_timeline_config.x_axis_label = "Date";
        var_timeline_config.y_axis_label = "VaR ($)";
        chart_templates_["var_timeline"] = var_timeline_config;

        ChartConfig factor_heatmap_config;
        factor_heatmap_config.chart_id = "factor_heatmap";
        factor_heatmap_config.chart_type = ChartType::HEATMAP;
        factor_heatmap_config.title = "Risk Factor Correlation Matrix";
        factor_heatmap_config.width = 600;
        factor_heatmap_config.height = 600;
        chart_templates_["factor_heatmap"] = factor_heatmap_config;

        // Add more chart templates...
    }

    void initialize_dashboard_templates() {
        // Initialize dashboard configurations
        DashboardConfig executive_dashboard;
        executive_dashboard.dashboard_id = "executive_dashboard";
        executive_dashboard.title = "Executive Risk Dashboard";
        executive_dashboard.layout = DashboardLayout::GRID;
        executive_dashboard.refresh_interval_seconds = 30;
        executive_dashboard.theme = "dark";

        dashboard_templates_["executive"] = executive_dashboard;

        DashboardConfig trader_dashboard;
        trader_dashboard.dashboard_id = "trader_dashboard";
        trader_dashboard.title = "Trading Desk Risk Monitor";
        trader_dashboard.layout = DashboardLayout::TABBED;
        trader_dashboard.refresh_interval_seconds = 5;
        trader_dashboard.theme = "light";

        dashboard_templates_["trader"] = trader_dashboard;
    }

    void initialize_report_templates() {
        // Initialize report templates
        ReportTemplate regulatory_template;
        regulatory_template.template_name = "regulatory_stress_report";
        regulatory_template.sections = {
            "executive_summary", "methodology", "scenarios", "results",
            "capital_impact", "recommendations", "appendices"
        };
        regulatory_template.page_orientation = PageOrientation::PORTRAIT;
        regulatory_template.include_charts = true;
        regulatory_template.include_detailed_tables = true;

        report_templates_["regulatory"] = regulatory_template;
    }

    void initialize_themes() {
        // Dark theme
        VisualizationTheme dark_theme;
        dark_theme.name = "dark";
        dark_theme.background_color = "#1e1e1e";
        dark_theme.text_color = "#ffffff";
        dark_theme.primary_color = "#007acc";
        dark_theme.secondary_color = "#ff6b6b";
        dark_theme.grid_color = "#333333";
        dark_theme.chart_colors = {"#007acc", "#ff6b6b", "#4ecdc4", "#45b7d1", "#f9ca24"};

        themes_["dark"] = dark_theme;

        // Light theme
        VisualizationTheme light_theme;
        light_theme.name = "light";
        light_theme.background_color = "#ffffff";
        light_theme.text_color = "#333333";
        light_theme.primary_color = "#2196f3";
        light_theme.secondary_color = "#f44336";
        light_theme.grid_color = "#e0e0e0";
        light_theme.chart_colors = {"#2196f3", "#f44336", "#4caf50", "#ff9800", "#9c27b0"};

        themes_["light"] = light_theme;
    }

    void generate_dashboard_header(std::stringstream& html, const DashboardParameters& params) {
        html << "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n";
        html << "<meta charset=\"UTF-8\">\n";
        html << "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n";
        html << "<title>" << params.title << "</title>\n";

        // Include CSS libraries
        html << "<link rel=\"stylesheet\" href=\"https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.css\">\n";
        html << "<link rel=\"stylesheet\" href=\"https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.2.0/css/bootstrap.min.css\">\n";

        // Custom CSS
        generate_dashboard_css(html, params);

        html << "</head>\n<body>\n";

        // Dashboard navigation
        html << "<nav class=\"navbar navbar-expand-lg navbar-dark bg-dark\">\n";
        html << "<div class=\"container-fluid\">\n";
        html << "<a class=\"navbar-brand\" href=\"#\">" << params.title << "</a>\n";
        html << "<div class=\"navbar-nav ms-auto\">\n";
        html << "<span class=\"navbar-text\">Last Updated: <span id=\"last-update-time\"></span></span>\n";
        html << "</div>\n</div>\n</nav>\n";
    }

    void generate_dashboard_css(std::stringstream& html, const DashboardParameters& params) {
        auto theme = themes_.find(params.theme);
        if (theme == themes_.end()) {
            theme = themes_.find("light");
        }

        html << "<style>\n";
        html << "body { background-color: " << theme->second.background_color << "; ";
        html << "color: " << theme->second.text_color << "; }\n";
        html << ".chart-container { margin: 20px 0; padding: 15px; border: 1px solid " << theme->second.grid_color << "; border-radius: 5px; }\n";
        html << ".metric-card { background: " << theme->second.background_color << "; border: 1px solid " << theme->second.grid_color << "; }\n";
        html << ".metric-value { font-size: 2em; font-weight: bold; color: " << theme->second.primary_color << "; }\n";
        html << ".metric-label { font-size: 0.9em; color: " << theme->second.text_color << "; }\n";
        html << ".alert-high { background-color: rgba(244, 67, 54, 0.1); border-left: 4px solid #f44336; }\n";
        html << ".alert-medium { background-color: rgba(255, 152, 0, 0.1); border-left: 4px solid #ff9800; }\n";
        html << ".alert-low { background-color: rgba(76, 175, 80, 0.1); border-left: 4px solid #4caf50; }\n";
        html << "</style>\n";
    }

    void generate_executive_summary_section(std::stringstream& html,
                                           const std::vector<RealTimeStressResult>& results,
                                           const DashboardParameters& params) {

        html << "<div class=\"container-fluid mt-4\">\n";
        html << "<div class=\"row\">\n";
        html << "<div class=\"col-12\">\n";
        html << "<h2>Executive Summary</h2>\n";
        html << "</div>\n</div>\n";

        // Key metrics cards
        html << "<div class=\"row\">\n";

        // Total P&L metric
        double total_pnl = calculate_aggregate_pnl(results);
        generate_metric_card(html, "Total P&L", format_currency(total_pnl), get_alert_level(total_pnl, 0));

        // VaR metric
        double max_var = calculate_max_var(results);
        generate_metric_card(html, "Maximum VaR", format_currency(max_var), AlertLevel::MEDIUM);

        // Number of stress tests
        generate_metric_card(html, "Active Tests", std::to_string(results.size()), AlertLevel::LOW);

        // Compliance status
        bool compliance_ok = check_overall_compliance(results);
        generate_metric_card(html, "Compliance", compliance_ok ? "PASS" : "FAIL",
                           compliance_ok ? AlertLevel::LOW : AlertLevel::HIGH);

        html << "</div>\n</div>\n";
    }

    void generate_metric_card(std::stringstream& html, const std::string& label,
                             const std::string& value, AlertLevel alert_level) {
        std::string alert_class;
        switch (alert_level) {
            case AlertLevel::HIGH: alert_class = "alert-high"; break;
            case AlertLevel::MEDIUM: alert_class = "alert-medium"; break;
            case AlertLevel::LOW: alert_class = "alert-low"; break;
        }

        html << "<div class=\"col-md-3\">\n";
        html << "<div class=\"card metric-card " << alert_class << "\">\n";
        html << "<div class=\"card-body text-center\">\n";
        html << "<div class=\"metric-value\">" << value << "</div>\n";
        html << "<div class=\"metric-label\">" << label << "</div>\n";
        html << "</div>\n</div>\n</div>\n";
    }

    void generate_real_time_monitoring_section(std::stringstream& html,
                                              const std::vector<RealTimeStressResult>& results,
                                              const DashboardParameters& params) {

        html << "<div class=\"container-fluid mt-4\">\n";
        html << "<div class=\"row\">\n";
        html << "<div class=\"col-12\">\n";
        html << "<h3>Real-Time Monitoring</h3>\n";
        html << "</div>\n</div>\n";

        // Running tests table
        html << "<div class=\"row\">\n";
        html << "<div class=\"col-12\">\n";
        html << "<table class=\"table table-striped\">\n";
        html << "<thead><tr><th>Job ID</th><th>Type</th><th>Status</th><th>Runtime</th><th>Progress</th></tr></thead>\n";
        html << "<tbody>\n";

        for (const auto& result : results) {
            if (result.status == ExecutionStatus::RUNNING) {
                html << "<tr>\n";
                html << "<td>" << result.job_id << "</td>\n";
                html << "<td>" << stress_test_type_to_string(result.request.stress_test_type) << "</td>\n";
                html << "<td><span class=\"badge bg-primary\">Running</span></td>\n";
                html << "<td>" << calculate_runtime_minutes(result) << " min</td>\n";
                html << "<td><div class=\"progress\"><div class=\"progress-bar\" style=\"width: 75%\"></div></div></td>\n";
                html << "</tr>\n";
            }
        }

        html << "</tbody>\n</table>\n";
        html << "</div>\n</div>\n</div>\n";
    }

    void generate_results_grid_section(std::stringstream& html,
                                      const std::vector<RealTimeStressResult>& results,
                                      const DashboardParameters& params) {

        html << "<div class=\"container-fluid mt-4\">\n";
        html << "<div class=\"row\">\n";
        html << "<div class=\"col-12\">\n";
        html << "<h3>Stress Test Results</h3>\n";
        html << "</div>\n</div>\n";

        // Results table
        html << "<div class=\"row\">\n";
        html << "<div class=\"col-12\">\n";
        html << "<table class=\"table table-hover\">\n";
        html << "<thead><tr><th>Test Name</th><th>Type</th><th>P&L</th><th>VaR 95%</th><th>Expected Shortfall</th><th>Status</th></tr></thead>\n";
        html << "<tbody>\n";

        for (const auto& result : results) {
            html << "<tr>\n";
            html << "<td>" << result.request.request_id << "</td>\n";
            html << "<td>" << stress_test_type_to_string(result.request.stress_test_type) << "</td>\n";
            html << "<td>" << format_currency(result.total_pnl) << "</td>\n";
            html << "<td>" << format_currency(result.var_95) << "</td>\n";
            html << "<td>" << format_currency(result.expected_shortfall) << "</td>\n";
            html << "<td>" << execution_status_to_string(result.status) << "</td>\n";
            html << "</tr>\n";
        }

        html << "</tbody>\n</table>\n";
        html << "</div>\n</div>\n</div>\n";
    }

    void generate_line_chart(std::stringstream& html, const ChartData& data, const ChartConfig& config) {
        html << "<canvas id=\"" << config.chart_id << "_canvas\"></canvas>\n";
    }

    void generate_bar_chart(std::stringstream& html, const ChartData& data, const ChartConfig& config) {
        html << "<canvas id=\"" << config.chart_id << "_canvas\"></canvas>\n";
    }

    void generate_scatter_plot(std::stringstream& html, const ChartData& data, const ChartConfig& config) {
        html << "<canvas id=\"" << config.chart_id << "_canvas\"></canvas>\n";
    }

    void generate_histogram(std::stringstream& html, const ChartData& data, const ChartConfig& config) {
        html << "<canvas id=\"" << config.chart_id << "_canvas\"></canvas>\n";
    }

    void generate_heatmap(std::stringstream& html, const ChartData& data, const ChartConfig& config) {
        html << "<div id=\"" << config.chart_id << "_heatmap\" class=\"heatmap\"></div>\n";
    }

    void generate_box_plot(std::stringstream& html, const ChartData& data, const ChartConfig& config) {
        html << "<canvas id=\"" << config.chart_id << "_canvas\"></canvas>\n";
    }

    void generate_waterfall_chart(std::stringstream& html, const ChartData& data, const ChartConfig& config) {
        html << "<canvas id=\"" << config.chart_id << "_canvas\"></canvas>\n";
    }

    void generate_treemap(std::stringstream& html, const ChartData& data, const ChartConfig& config) {
        html << "<div id=\"" << config.chart_id << "_treemap\" class=\"treemap\"></div>\n";
    }

    void generate_chart_javascript(std::stringstream& html, const ChartData& data, const ChartConfig& config) {
        html << "<script>\n";
        html << "// Chart JavaScript for " << config.chart_id << "\n";
        html << "document.addEventListener('DOMContentLoaded', function() {\n";
        html << "  const ctx = document.getElementById('" << config.chart_id << "_canvas').getContext('2d');\n";
        html << "  new Chart(ctx, {\n";
        html << "    type: '" << chart_type_to_js_string(config.chart_type) << "',\n";
        html << "    data: " << convert_chart_data_to_json(data) << ",\n";
        html << "    options: " << generate_chart_options_json(config) << "\n";
        html << "  });\n";
        html << "});\n";
        html << "</script>\n";
    }

    void generate_dashboard_footer(std::stringstream& html, const DashboardParameters& params) {
        // Include JavaScript libraries
        html << "<script src=\"https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js\"></script>\n";
        html << "<script src=\"https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.2.0/js/bootstrap.bundle.min.js\"></script>\n";

        // Auto-refresh functionality
        if (params.auto_refresh_interval_seconds > 0) {
            html << "<script>\n";
            html << "setInterval(function() {\n";
            html << "  document.getElementById('last-update-time').textContent = new Date().toLocaleString();\n";
            html << "  // Add refresh logic here\n";
            html << "}, " << (params.auto_refresh_interval_seconds * 1000) << ");\n";
            html << "</script>\n";
        }

        html << "</body>\n</html>\n";
    }

    // Helper functions
    double calculate_aggregate_pnl(const std::vector<RealTimeStressResult>& results) {
        double total = 0.0;
        for (const auto& result : results) {
            if (result.status == ExecutionStatus::COMPLETED) {
                total += result.total_pnl;
            }
        }
        return total;
    }

    double calculate_max_var(const std::vector<RealTimeStressResult>& results) {
        double max_var = 0.0;
        for (const auto& result : results) {
            if (result.status == ExecutionStatus::COMPLETED) {
                max_var = std::max(max_var, result.var_95);
            }
        }
        return max_var;
    }

    bool check_overall_compliance(const std::vector<RealTimeStressResult>& results) {
        for (const auto& result : results) {
            if (result.status == ExecutionStatus::COMPLETED && !result.compliance_status) {
                return false;
            }
        }
        return true;
    }

    AlertLevel get_alert_level(double value, double threshold) {
        if (value < threshold * -2.0) return AlertLevel::HIGH;
        if (value < threshold * -1.0) return AlertLevel::MEDIUM;
        return AlertLevel::LOW;
    }

    std::string format_currency(double value) {
        std::stringstream ss;
        ss << std::fixed << std::setprecision(0);
        if (std::abs(value) >= 1000000) {
            ss << "$" << (value / 1000000.0) << "M";
        } else if (std::abs(value) >= 1000) {
            ss << "$" << (value / 1000.0) << "K";
        } else {
            ss << "$" << value;
        }
        return ss.str();
    }

    std::string stress_test_type_to_string(StressTestType type) {
        switch (type) {
            case StressTestType::HISTORICAL_SCENARIO: return "Historical";
            case StressTestType::MONTE_CARLO: return "Monte Carlo";
            case StressTestType::REGULATORY: return "Regulatory";
            case StressTestType::CUSTOM_SCENARIO: return "Custom";
            case StressTestType::MULTI_FACTOR: return "Multi-Factor";
            default: return "Unknown";
        }
    }

    std::string execution_status_to_string(ExecutionStatus status) {
        switch (status) {
            case ExecutionStatus::QUEUED: return "Queued";
            case ExecutionStatus::RUNNING: return "Running";
            case ExecutionStatus::COMPLETED: return "Completed";
            case ExecutionStatus::FAILED: return "Failed";
            case ExecutionStatus::CANCELLED: return "Cancelled";
            default: return "Unknown";
        }
    }

    double calculate_runtime_minutes(const RealTimeStressResult& result) {
        auto now = std::chrono::system_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::minutes>(now - result.start_time);
        return duration.count();
    }

    std::string chart_type_to_js_string(ChartType type) {
        switch (type) {
            case ChartType::LINE_CHART: return "line";
            case ChartType::BAR_CHART: return "bar";
            case ChartType::SCATTER_PLOT: return "scatter";
            case ChartType::HISTOGRAM: return "bar";
            default: return "line";
        }
    }

    std::string convert_chart_data_to_json(const ChartData& data) {
        // Simplified JSON conversion - would be more sophisticated in practice
        return "{ labels: [], datasets: [] }";
    }

    std::string generate_chart_options_json(const ChartConfig& config) {
        std::stringstream options;
        options << "{\n";
        options << "  responsive: true,\n";
        options << "  plugins: {\n";
        options << "    title: { display: true, text: '" << config.title << "' }\n";
        options << "  },\n";
        options << "  scales: {\n";
        options << "    x: { title: { display: true, text: '" << config.x_axis_label << "' } },\n";
        options << "    y: { title: { display: true, text: '" << config.y_axis_label << "' } }\n";
        options << "  }\n";
        options << "}";
        return options.str();
    }

    // Placeholder implementations for additional methods
    void generate_risk_factor_analysis_section(std::stringstream& html, const std::vector<RealTimeStressResult>& results, const DashboardParameters& params) {}
    void generate_performance_metrics_section(std::stringstream& html, const std::vector<RealTimeStressResult>& results, const DashboardParameters& params) {}
    void generate_historical_comparison_section(std::stringstream& html, const std::vector<RealTimeStressResult>& results, const DashboardParameters& params) {}
    void generate_charts_section(std::stringstream& html, const std::vector<RealTimeStressResult>& results, const DashboardParameters& params) {}

    // Report generation placeholders
    void generate_report_title_page(std::stringstream& report, const ComprehensiveReportParameters& params) {}
    void generate_table_of_contents(std::stringstream& report, const ComprehensiveReportParameters& params) {}
    void generate_executive_summary_report(std::stringstream& report, const ComplianceResults& compliance_results, const std::vector<RealTimeStressResult>& stress_results, const ComprehensiveReportParameters& params) {}
    void generate_methodology_section_report(std::stringstream& report, const ComprehensiveReportParameters& params) {}
    void generate_regulatory_compliance_section(std::stringstream& report, const ComplianceResults& compliance_results, const ComprehensiveReportParameters& params) {}
    void generate_stress_testing_results_section(std::stringstream& report, const std::vector<RealTimeStressResult>& stress_results, const ComprehensiveReportParameters& params) {}
    void generate_detailed_risk_factor_analysis(std::stringstream& report, const std::vector<RealTimeStressResult>& stress_results, const ComprehensiveReportParameters& params) {}
    void generate_scenario_analysis_section(std::stringstream& report, const std::vector<RealTimeStressResult>& stress_results, const ComprehensiveReportParameters& params) {}
    void generate_portfolio_impact_section(std::stringstream& report, const std::vector<RealTimeStressResult>& stress_results, const ComprehensiveReportParameters& params) {}
    void generate_model_validation_section_report(std::stringstream& report, const std::vector<RealTimeStressResult>& stress_results, const ComprehensiveReportParameters& params) {}
    void generate_recommendations_section(std::stringstream& report, const ComplianceResults& compliance_results, const std::vector<RealTimeStressResult>& stress_results, const ComprehensiveReportParameters& params) {}
    void generate_appendices_section_report(std::stringstream& report, const std::vector<RealTimeStressResult>& stress_results, const ComprehensiveReportParameters& params) {}

    // Excel export placeholders
    void generate_excel_styles(std::stringstream& excel_xml) {}
    void generate_excel_summary_worksheet(std::stringstream& excel_xml, const std::vector<RealTimeStressResult>& results, const ExcelExportParameters& params) {}
    void generate_excel_detailed_worksheet(std::stringstream& excel_xml, const std::vector<RealTimeStressResult>& results, const ExcelExportParameters& params) {}
    void generate_excel_charts_worksheet(std::stringstream& excel_xml, const std::vector<RealTimeStressResult>& results, const ExcelExportParameters& params) {}
    void generate_excel_pivot_worksheet(std::stringstream& excel_xml, const std::vector<RealTimeStressResult>& results, const ExcelExportParameters& params) {}

    // Visualization method placeholders
    void generate_var_es_comparison_chart(std::stringstream& viz_html, const std::vector<RealTimeStressResult>& results, const RiskMetricsVisualizationParameters& params) {}
    void generate_pnl_distribution_chart(std::stringstream& viz_html, const std::vector<RealTimeStressResult>& results, const RiskMetricsVisualizationParameters& params) {}
    void generate_factor_contribution_chart(std::stringstream& viz_html, const std::vector<RealTimeStressResult>& results, const RiskMetricsVisualizationParameters& params) {}
    void generate_risk_factor_heatmap(std::stringstream& viz_html, const std::vector<RealTimeStressResult>& results, const RiskMetricsVisualizationParameters& params) {}
    void generate_time_series_chart(std::stringstream& viz_html, const std::vector<RealTimeStressResult>& results, const RiskMetricsVisualizationParameters& params) {}

    // Regulatory report placeholders
    void generate_regulatory_report_header(std::stringstream& report, const ComplianceResults& results, const RegulatoryReportParameters& params) {}
    void generate_compliance_summary_table(std::stringstream& report, const ComplianceResults& results, const RegulatoryReportParameters& params) {}
    void generate_capital_adequacy_charts(std::stringstream& report, const ComplianceResults& results, const RegulatoryReportParameters& params) {}
    void generate_scenario_impact_charts(std::stringstream& report, const ComplianceResults& results, const RegulatoryReportParameters& params) {}
    void generate_regulatory_metrics_dashboard(std::stringstream& report, const ComplianceResults& results, const RegulatoryReportParameters& params) {}
    void generate_compliance_trend_analysis(std::stringstream& report, const ComplianceResults& results, const RegulatoryReportParameters& params) {}

    // File format conversion placeholders
    void save_as_pdf(const std::string& content, const std::string& file_path) {
        // Would use a library like wkhtmltopdf or similar
    }

    void save_as_png(const std::string& content, const std::string& file_path) {
        // Would use a headless browser or image rendering library
    }

    void save_as_svg(const std::string& content, const std::string& file_path) {
        // Convert HTML/CSS to SVG
    }
};

// VisualizationReportingEngine implementation
VisualizationReportingEngine::VisualizationReportingEngine() :
    impl_(std::make_unique<VisualizationReportingEngineImpl>()) {}

VisualizationReportingEngine::~VisualizationReportingEngine() = default;

std::string VisualizationReportingEngine::generate_stress_test_dashboard(
    const std::vector<RealTimeStressResult>& results,
    const DashboardParameters& params) {
    return impl_->generate_stress_test_dashboard(results, params);
}

std::string VisualizationReportingEngine::generate_comprehensive_report(
    const ComplianceResults& compliance_results,
    const std::vector<RealTimeStressResult>& stress_results,
    const ComprehensiveReportParameters& params) {
    return impl_->generate_comprehensive_report(compliance_results, stress_results, params);
}

std::string VisualizationReportingEngine::create_interactive_chart(
    const ChartData& data,
    const ChartConfig& config) {
    return impl_->create_interactive_chart(data, config);
}

std::string VisualizationReportingEngine::export_results_to_excel(
    const std::vector<RealTimeStressResult>& results,
    const ExcelExportParameters& params) {
    return impl_->export_results_to_excel(results, params);
}

void VisualizationReportingEngine::save_visualization_to_file(
    const std::string& content,
    const std::string& file_path,
    VisualizationFormat format) {
    impl_->save_visualization_to_file(content, file_path, format);
}

std::string VisualizationReportingEngine::generate_risk_metrics_visualization(
    const std::vector<RealTimeStressResult>& results,
    const RiskMetricsVisualizationParameters& params) {
    return impl_->generate_risk_metrics_visualization(results, params);
}

std::string VisualizationReportingEngine::create_regulatory_compliance_report(
    const ComplianceResults& results,
    const RegulatoryReportParameters& params) {
    return impl_->create_regulatory_compliance_report(results, params);
}

} // namespace stress_testing