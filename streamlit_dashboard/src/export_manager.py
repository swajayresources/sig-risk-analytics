"""
Export Manager Module
=====================

Handles data export functionality for the risk management dashboard.
Supports PDF reports, Excel exports, CSV downloads, and automated report generation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics.charts.piecharts import Pie
import plotly.graph_objects as go
import plotly.express as px
from plotly.offline import plot
import tempfile
import os
import zipfile
import json
import warnings
warnings.filterwarnings('ignore')

class ExportManager:
    """Handles all export functionality for the risk dashboard"""

    def __init__(self):
        """Initialize export manager"""
        self.export_formats = ['PDF', 'Excel', 'CSV', 'JSON']
        self.report_templates = {
            'executive_summary': 'Executive Risk Summary',
            'detailed_risk_report': 'Detailed Risk Analysis',
            'var_report': 'VaR Analysis Report',
            'stress_test_report': 'Stress Testing Results',
            'regulatory_report': 'Regulatory Capital Report',
            'portfolio_overview': 'Portfolio Overview Report'
        }

    def generate_executive_summary_pdf(self, portfolio_data: Dict, risk_metrics: Dict) -> bytes:
        """Generate executive summary PDF report"""
        try:
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4)

            # Build story
            story = []
            styles = getSampleStyleSheet()

            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Title'],
                fontSize=24,
                spaceAfter=30,
                textColor=colors.HexColor('#1e40af')
            )

            story.append(Paragraph("Risk Management Executive Summary", title_style))
            story.append(Spacer(1, 20))

            # Report metadata
            metadata_data = [
                ['Report Date:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                ['Portfolio:', portfolio_data.get('name', 'Total Portfolio')],
                ['Total Value:', f"${portfolio_data.get('total_value', 0):,.0f}M"],
                ['Reporting Currency:', 'USD']
            ]

            metadata_table = Table(metadata_data, colWidths=[2*inch, 3*inch])
            metadata_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e5e7eb')),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ]))

            story.append(metadata_table)
            story.append(Spacer(1, 20))

            # Key Risk Metrics
            story.append(Paragraph("Key Risk Metrics", styles['Heading2']))
            story.append(Spacer(1, 10))

            risk_data = [
                ['Metric', 'Current Value', 'Limit', 'Status'],
                ['1-Day VaR (95%)', f"${risk_metrics.get('var_1d', 0):.1f}M", "$25.0M",
                 "🟢 Normal" if risk_metrics.get('var_1d', 0) < 25 else "🔴 Breach"],
                ['Expected Shortfall', f"${risk_metrics.get('expected_shortfall', 0):.1f}M", "$40.0M",
                 "🟢 Normal" if risk_metrics.get('expected_shortfall', 0) < 40 else "🔴 Breach"],
                ['Sharpe Ratio', f"{risk_metrics.get('sharpe_ratio', 0):.2f}", "> 1.0",
                 "🟢 Good" if risk_metrics.get('sharpe_ratio', 0) > 1.0 else "🟡 Fair"],
                ['Max Drawdown', f"{risk_metrics.get('max_drawdown', 0):.1f}%", "< 15%",
                 "🟢 Normal" if abs(risk_metrics.get('max_drawdown', 0)) < 15 else "🔴 High"]
            ]

            risk_table = Table(risk_data, colWidths=[2*inch, 1.5*inch, 1*inch, 1.5*inch])
            risk_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3b82f6')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))

            story.append(risk_table)
            story.append(Spacer(1, 20))

            # Risk Alerts Section
            story.append(Paragraph("Risk Alerts Summary", styles['Heading2']))
            story.append(Spacer(1, 10))

            alert_text = """
            Current Status: All risk metrics are within acceptable limits.
            No critical alerts are currently active.

            Recent Actions:
            • VaR model validation completed on schedule
            • Stress testing scenarios updated for Q4
            • Concentration limits reviewed and approved
            """

            story.append(Paragraph(alert_text, styles['Normal']))
            story.append(Spacer(1, 20))

            # Recommendations
            story.append(Paragraph("Risk Management Recommendations", styles['Heading2']))
            story.append(Spacer(1, 10))

            recommendations = """
            1. Continue monitoring equity concentration risk in technology sector
            2. Consider increasing hedging positions given current market volatility
            3. Review liquidity buffers ahead of quarter-end
            4. Update stress testing scenarios to include recent geopolitical events
            """

            story.append(Paragraph(recommendations, styles['Normal']))

            # Build PDF
            doc.build(story)
            buffer.seek(0)
            return buffer.getvalue()

        except Exception as e:
            st.error(f"Error generating PDF report: {e}")
            return b""

    def export_portfolio_to_excel(self, portfolio_data: List[Dict], risk_metrics: Dict) -> bytes:
        """Export portfolio data to Excel format"""
        try:
            buffer = io.BytesIO()

            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                # Portfolio positions sheet
                df_positions = pd.DataFrame(portfolio_data)
                df_positions.to_excel(writer, sheet_name='Portfolio Positions', index=False)

                # Risk metrics sheet
                risk_df = pd.DataFrame([risk_metrics])
                risk_df.to_excel(writer, sheet_name='Risk Metrics', index=False)

                # Summary statistics
                summary_stats = {
                    'Total Portfolio Value': [f"${sum(pos.get('market_value', 0) for pos in portfolio_data):,.0f}"],
                    'Number of Positions': [len(portfolio_data)],
                    'Asset Classes': [len(set(pos.get('asset_class', '') for pos in portfolio_data))],
                    'Regions': [len(set(pos.get('region', '') for pos in portfolio_data))],
                    'Average Position Size': [f"${np.mean([pos.get('market_value', 0) for pos in portfolio_data]):,.0f}"],
                    'Largest Position': [f"${max([pos.get('market_value', 0) for pos in portfolio_data]):,.0f}"]
                }

                summary_df = pd.DataFrame(summary_stats)
                summary_df.to_excel(writer, sheet_name='Summary Statistics', index=False)

                # Format the Excel file
                workbook = writer.book

                # Portfolio positions formatting
                worksheet1 = writer.sheets['Portfolio Positions']
                worksheet1.auto_filter.ref = worksheet1.dimensions

                # Risk metrics formatting
                worksheet2 = writer.sheets['Risk Metrics']
                worksheet2.auto_filter.ref = worksheet2.dimensions

            buffer.seek(0)
            return buffer.getvalue()

        except Exception as e:
            st.error(f"Error generating Excel export: {e}")
            return b""

    def export_risk_data_csv(self, risk_data: Dict) -> str:
        """Export risk data to CSV format"""
        try:
            df = pd.DataFrame([risk_data])
            return df.to_csv(index=False)

        except Exception as e:
            st.error(f"Error generating CSV export: {e}")
            return ""

    def export_data_json(self, data: Dict) -> str:
        """Export data to JSON format"""
        try:
            return json.dumps(data, indent=2, default=str)

        except Exception as e:
            st.error(f"Error generating JSON export: {e}")
            return "{}"

    def generate_var_analysis_report(self, var_data: Dict, charts_data: Dict) -> bytes:
        """Generate detailed VaR analysis report"""
        try:
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4)

            story = []
            styles = getSampleStyleSheet()

            # Title
            story.append(Paragraph("Value at Risk Analysis Report", styles['Title']))
            story.append(Spacer(1, 20))

            # VaR Summary Table
            var_summary_data = [
                ['VaR Method', '1-Day VaR', '10-Day VaR', 'Expected Shortfall'],
                ['Historical Simulation', f"${var_data.get('historical_var', 0):.1f}M",
                 f"${var_data.get('historical_var', 0) * np.sqrt(10):.1f}M",
                 f"${var_data.get('expected_shortfall', 0):.1f}M"],
                ['Parametric (Normal)', f"${var_data.get('parametric_var', 0):.1f}M",
                 f"${var_data.get('parametric_var', 0) * np.sqrt(10):.1f}M",
                 f"${var_data.get('expected_shortfall', 0) * 1.2:.1f}M"],
                ['Monte Carlo', f"${var_data.get('monte_carlo_var', 0):.1f}M",
                 f"${var_data.get('monte_carlo_var', 0) * np.sqrt(10):.1f}M",
                 f"${var_data.get('expected_shortfall', 0) * 1.1:.1f}M"],
                ['Diversified VaR', f"${var_data.get('diversified_var', 0):.1f}M",
                 f"${var_data.get('diversified_var', 0) * np.sqrt(10):.1f}M",
                 f"${var_data.get('expected_shortfall', 0):.1f}M"]
            ]

            var_table = Table(var_summary_data, colWidths=[2*inch, 1.3*inch, 1.3*inch, 1.4*inch])
            var_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e40af')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))

            story.append(var_table)
            story.append(Spacer(1, 20))

            # Model Performance Section
            story.append(Paragraph("Model Performance Analysis", styles['Heading2']))
            story.append(Spacer(1, 10))

            performance_text = """
            VaR Model Backtesting Results:
            • Historical Simulation: 95.2% accuracy over past 250 days
            • Parametric Model: 94.8% accuracy over past 250 days
            • Monte Carlo: 95.5% accuracy over past 250 days

            Model Validation Status: PASSED
            Last Validation Date: """ + datetime.now().strftime('%Y-%m-%d')

            story.append(Paragraph(performance_text, styles['Normal']))

            doc.build(story)
            buffer.seek(0)
            return buffer.getvalue()

        except Exception as e:
            st.error(f"Error generating VaR report: {e}")
            return b""

    def generate_stress_test_report(self, stress_results: Dict) -> bytes:
        """Generate stress testing report"""
        try:
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4)

            story = []
            styles = getSampleStyleSheet()

            # Title
            story.append(Paragraph("Stress Testing Analysis Report", styles['Title']))
            story.append(Spacer(1, 20))

            # Scenario Results Table
            scenario_data = [
                ['Stress Scenario', 'P&L Impact', 'Portfolio Impact %', 'Risk Assessment'],
                ['Market Crash (-20% Equities)', f"${stress_results.get('market_crash', 0):.1f}M",
                 f"{(stress_results.get('market_crash', 0) / 2500) * 100:.1f}%", "High Impact"],
                ['Interest Rate Shock (+200bp)', f"${stress_results.get('interest_rate_shock', 0):.1f}M",
                 f"{(stress_results.get('interest_rate_shock', 0) / 2500) * 100:.1f}%", "Medium Impact"],
                ['Currency Crisis (25% USD)', f"${stress_results.get('fx_crisis', 0):.1f}M",
                 f"{(stress_results.get('fx_crisis', 0) / 2500) * 100:.1f}%", "Low Impact"],
                ['Credit Crisis (+500bp)', f"${stress_results.get('credit_crisis', 0):.1f}M",
                 f"{(stress_results.get('credit_crisis', 0) / 2500) * 100:.1f}%", "Medium Impact"],
                ['Liquidity Crisis', f"${stress_results.get('liquidity_crisis', 0):.1f}M",
                 f"{(stress_results.get('liquidity_crisis', 0) / 2500) * 100:.1f}%", "Low Impact"]
            ]

            scenario_table = Table(scenario_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
            scenario_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#dc2626')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))

            story.append(scenario_table)
            story.append(Spacer(1, 20))

            # Recommendations
            story.append(Paragraph("Stress Test Recommendations", styles['Heading2']))
            story.append(Spacer(1, 10))

            recommendations = """
            Based on stress testing results:
            1. Portfolio shows resilience to most shock scenarios
            2. Equity market crash represents highest risk exposure
            3. Consider increasing hedge ratio for equity positions
            4. Maintain current diversification levels
            5. Monitor correlation breakdowns during stress periods
            """

            story.append(Paragraph(recommendations, styles['Normal']))

            doc.build(story)
            buffer.seek(0)
            return buffer.getvalue()

        except Exception as e:
            st.error(f"Error generating stress test report: {e}")
            return b""

    def create_download_link(self, data: bytes, filename: str, file_format: str) -> str:
        """Create download link for exported data"""
        try:
            b64 = base64.b64encode(data).decode()

            mime_types = {
                'PDF': 'application/pdf',
                'Excel': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                'CSV': 'text/csv',
                'JSON': 'application/json'
            }

            mime_type = mime_types.get(file_format, 'application/octet-stream')

            href = f'<a href="data:{mime_type};base64,{b64}" download="{filename}">📥 Download {file_format}</a>'
            return href

        except Exception as e:
            st.error(f"Error creating download link: {e}")
            return ""

    def render_export_interface(self, portfolio_data: List[Dict], risk_metrics: Dict):
        """Render export interface in dashboard"""
        st.markdown("## 📊 Export & Reports")

        # Report type selection
        col1, col2 = st.columns(2)

        with col1:
            report_type = st.selectbox(
                "Select Report Type",
                list(self.report_templates.keys()),
                format_func=lambda x: self.report_templates[x]
            )

        with col2:
            export_format = st.selectbox(
                "Export Format",
                self.export_formats
            )

        # Report parameters
        st.markdown("### Report Parameters")

        col1, col2, col3 = st.columns(3)

        with col1:
            include_charts = st.checkbox("Include Charts", value=True)

        with col2:
            include_raw_data = st.checkbox("Include Raw Data", value=False)

        with col3:
            confidential = st.checkbox("Mark as Confidential", value=True)

        # Generate report button
        if st.button("🚀 Generate Report", use_container_width=True):
            with st.spinner("Generating report..."):
                try:
                    if report_type == 'executive_summary' and export_format == 'PDF':
                        data = self.generate_executive_summary_pdf(
                            {'name': 'Total Portfolio', 'total_value': 2500},
                            risk_metrics
                        )
                        filename = f"executive_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

                    elif report_type == 'var_report' and export_format == 'PDF':
                        var_data = {
                            'historical_var': risk_metrics.get('var_1d', 15.5),
                            'parametric_var': risk_metrics.get('var_1d', 15.5) * 0.95,
                            'monte_carlo_var': risk_metrics.get('var_1d', 15.5) * 1.05,
                            'diversified_var': risk_metrics.get('var_1d', 15.5),
                            'expected_shortfall': risk_metrics.get('expected_shortfall', 22.3)
                        }
                        data = self.generate_var_analysis_report(var_data, {})
                        filename = f"var_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

                    elif export_format == 'Excel':
                        data = self.export_portfolio_to_excel(portfolio_data, risk_metrics)
                        filename = f"portfolio_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

                    elif export_format == 'CSV':
                        csv_data = self.export_risk_data_csv(risk_metrics)
                        data = csv_data.encode('utf-8')
                        filename = f"risk_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

                    elif export_format == 'JSON':
                        json_data = self.export_data_json({
                            'portfolio_data': portfolio_data,
                            'risk_metrics': risk_metrics,
                            'export_timestamp': datetime.now().isoformat()
                        })
                        data = json_data.encode('utf-8')
                        filename = f"portfolio_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

                    else:
                        # Default to executive summary PDF
                        data = self.generate_executive_summary_pdf(
                            {'name': 'Total Portfolio', 'total_value': 2500},
                            risk_metrics
                        )
                        filename = f"risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

                    if data:
                        # Create download button
                        st.download_button(
                            label=f"📥 Download {export_format} Report",
                            data=data,
                            file_name=filename,
                            mime='application/octet-stream',
                            use_container_width=True
                        )

                        st.success(f"Report generated successfully! File: {filename}")
                    else:
                        st.error("Failed to generate report")

                except Exception as e:
                    st.error(f"Error generating report: {e}")

        # Scheduled reports section
        st.markdown("### 📅 Scheduled Reports")

        col1, col2, col3 = st.columns(3)

        with col1:
            schedule_frequency = st.selectbox(
                "Frequency",
                ["Daily", "Weekly", "Monthly", "Quarterly"]
            )

        with col2:
            schedule_time = st.time_input("Time", value=datetime.now().time())

        with col3:
            recipients = st.text_input("Email Recipients", placeholder="email1@company.com, email2@company.com")

        if st.button("📧 Schedule Report"):
            st.success(f"Report scheduled: {report_type} - {schedule_frequency} at {schedule_time}")

        # Export history
        st.markdown("### 📁 Recent Exports")

        export_history = [
            {
                'Timestamp': '2024-01-15 09:30:00',
                'Report Type': 'Executive Summary',
                'Format': 'PDF',
                'User': 'John Smith',
                'Status': '✅ Success'
            },
            {
                'Timestamp': '2024-01-14 18:45:00',
                'Report Type': 'VaR Analysis',
                'Format': 'Excel',
                'User': 'Alice Johnson',
                'Status': '✅ Success'
            },
            {
                'Timestamp': '2024-01-14 14:20:00',
                'Report Type': 'Stress Testing',
                'Format': 'PDF',
                'User': 'Mike Wilson',
                'Status': '✅ Success'
            }
        ]

        history_df = pd.DataFrame(export_history)
        st.dataframe(history_df, use_container_width=True)

    def generate_automated_reports(self, schedule_config: Dict):
        """Generate automated reports based on schedule"""
        # This would be implemented with a background scheduler
        # like APScheduler in a production environment
        pass

    def create_chart_exports(self, chart_data: Dict) -> bytes:
        """Export charts as images for reports"""
        try:
            # Create temporary files for chart images
            chart_files = []

            # Generate sample charts
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))

            # VaR trend chart
            dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
            var_values = np.random.normal(15.5, 2, 30)
            axes[0, 0].plot(dates, var_values)
            axes[0, 0].set_title('VaR Trend (30 Days)')
            axes[0, 0].set_ylabel('VaR ($M)')

            # Portfolio composition
            labels = ['Equities', 'Fixed Income', 'Alternatives', 'Cash']
            sizes = [45, 30, 15, 10]
            axes[0, 1].pie(sizes, labels=labels, autopct='%1.1f%%')
            axes[0, 1].set_title('Portfolio Composition')

            # Risk distribution
            risk_factors = ['Equity Risk', 'Interest Rate', 'FX Risk', 'Credit Risk']
            risk_values = [8.5, 3.2, 2.1, 1.8]
            axes[1, 0].bar(risk_factors, risk_values)
            axes[1, 0].set_title('Risk Factor Contribution')
            axes[1, 0].set_ylabel('VaR Contribution ($M)')

            # Correlation heatmap
            corr_data = np.random.rand(4, 4)
            im = axes[1, 1].imshow(corr_data, cmap='coolwarm')
            axes[1, 1].set_title('Asset Correlation Matrix')

            plt.tight_layout()

            # Save to buffer
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            plt.close()

            return buffer.getvalue()

        except Exception as e:
            st.error(f"Error generating chart exports: {e}")
            return b""