"""
Professional Risk Management Dashboard
=======================================

A comprehensive risk management dashboard using Streamlit that rivals Bloomberg/Refinitiv
with real-time risk metrics, advanced visualizations, and professional features.

Author: Risk Management Team
Version: 1.0.0
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime, timedelta
import redis
import asyncio
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add current directory to Python path
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import custom modules
from src.risk_engine import RiskEngine
from src.data_provider import DataProvider
from src.alert_manager import AlertManager
from src.auth_manager import AuthManager
from src.export_manager import ExportManager
from src.visualization_engine import VisualizationEngine
from src.regulatory_compliance import RegulatoryComplianceEngine

# Page configuration
st.set_page_config(
    page_title="Professional Risk Management Dashboard",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://docs.riskdashboard.com',
        'Report a bug': "https://github.com/riskdashboard/issues",
        'About': "# Professional Risk Management Dashboard\nVersion 1.0.0"
    }
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Main dashboard styling */
    .main-header {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #3b82f6;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin: 0.5rem 0;
    }

    .risk-alert-high {
        border-left-color: #ef4444 !important;
        background: linear-gradient(90deg, #fef2f2 0%, #fff 100%);
    }

    .risk-alert-medium {
        border-left-color: #f59e0b !important;
        background: linear-gradient(90deg, #fffbeb 0%, #fff 100%);
    }

    .risk-alert-low {
        border-left-color: #10b981 !important;
        background: linear-gradient(90deg, #f0fdf4 0%, #fff 100%);
    }

    .sidebar-nav {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }

    .chart-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }

    .data-table {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    /* Custom button styling */
    .stButton > button {
        background: linear-gradient(90deg, #3b82f6 0%, #1e40af 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }

    /* Alert styling */
    .alert-critical {
        background: #fef2f2;
        border: 1px solid #fecaca;
        border-radius: 8px;
        padding: 1rem;
        color: #991b1b;
    }

    .alert-warning {
        background: #fffbeb;
        border: 1px solid #fed7aa;
        border-radius: 8px;
        padding: 1rem;
        color: #92400e;
    }

    .alert-info {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 8px;
        padding: 1rem;
        color: #1e40af;
    }

    /* Loading animation */
    .loading-spinner {
        border: 4px solid #f3f3f3;
        border-top: 4px solid #3b82f6;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
</style>
""", unsafe_allow_html=True)

class RiskDashboard:
    """Main Risk Management Dashboard Class"""

    def __init__(self):
        """Initialize the dashboard with necessary components"""
        self.initialize_session_state()
        self.auth_manager = AuthManager()
        self.risk_engine = RiskEngine()
        self.data_provider = DataProvider()
        self.alert_manager = AlertManager()
        self.export_manager = ExportManager()
        self.viz_engine = VisualizationEngine()
        self.compliance_engine = RegulatoryComplianceEngine()

        # Redis connection for real-time data
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
            self.redis_client.ping()
            self.redis_connected = True
        except:
            self.redis_connected = False
            st.warning("Redis not connected - using simulated data")

    def initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
        if 'user_role' not in st.session_state:
            st.session_state.user_role = None
        if 'user_name' not in st.session_state:
            st.session_state.user_name = None
        if 'dashboard_config' not in st.session_state:
            st.session_state.dashboard_config = self.get_default_config()
        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = True
        if 'refresh_interval' not in st.session_state:
            st.session_state.refresh_interval = 30
        if 'selected_portfolio' not in st.session_state:
            st.session_state.selected_portfolio = 'Total Portfolio'
        if 'risk_alerts' not in st.session_state:
            st.session_state.risk_alerts = []

    def get_default_config(self) -> Dict:
        """Get default dashboard configuration"""
        return {
            'theme': 'professional',
            'auto_refresh': True,
            'refresh_interval': 30,
            'charts_per_row': 2,
            'show_advanced_metrics': True,
            'currency': 'USD',
            'time_zone': 'UTC',
            'default_var_confidence': 0.95,
            'default_time_horizon': 1
        }

    def render_header(self):
        """Render the main dashboard header"""
        st.markdown("""
        <div class="main-header">
            <h1>🏦 Professional Risk Management Dashboard</h1>
            <p>Real-time portfolio risk monitoring and analytics</p>
        </div>
        """, unsafe_allow_html=True)

        # Status bar
        col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 2])

        with col1:
            st.info(f"🔄 Last Update: {datetime.now().strftime('%H:%M:%S')}")

        with col2:
            connection_status = "🟢 Connected" if self.redis_connected else "🔴 Offline"
            st.info(f"📡 Data Feed: {connection_status}")

        with col3:
            st.info(f"👤 User: {st.session_state.get('user_name', 'Guest')}")

        with col4:
            st.info(f"🏛️ Role: {st.session_state.get('user_role', 'Viewer')}")

        with col5:
            st.info(f"💰 Base Currency: {st.session_state.dashboard_config['currency']}")

    def render_sidebar(self):
        """Render the navigation sidebar"""
        with st.sidebar:
            st.markdown('<div class="sidebar-nav">', unsafe_allow_html=True)
            st.title("🎯 Navigation")

            # Main navigation
            page = st.selectbox(
                "Select View",
                [
                    "📊 Portfolio Overview",
                    "📈 VaR Analysis",
                    "🧪 Stress Testing",
                    "📉 Greeks Monitoring",
                    "🌡️ Risk Limits",
                    "📋 Regulatory Capital",
                    "⚖️ Regulatory Compliance",
                    "🔍 Factor Analysis",
                    "📱 Mobile View",
                    "⚙️ Settings"
                ],
                key="main_navigation"
            )

            st.markdown('</div>', unsafe_allow_html=True)

            # Portfolio selection
            st.markdown("### 📁 Portfolio Selection")
            portfolios = self.get_available_portfolios()
            selected_portfolio = st.selectbox(
                "Choose Portfolio",
                portfolios,
                index=portfolios.index(st.session_state.selected_portfolio) if st.session_state.selected_portfolio in portfolios else 0,
                key="portfolio_selector"
            )
            st.session_state.selected_portfolio = selected_portfolio

            # Time range selection
            st.markdown("### 📅 Time Range")
            time_range = st.selectbox(
                "Analysis Period",
                ["1D", "1W", "1M", "3M", "6M", "1Y", "YTD", "Custom"],
                index=2,
                key="time_range_selector"
            )

            if time_range == "Custom":
                start_date = st.date_input("Start Date", datetime.now() - timedelta(days=30))
                end_date = st.date_input("End Date", datetime.now())

            # Refresh settings
            st.markdown("### 🔄 Auto Refresh")
            auto_refresh = st.checkbox(
                "Enable Auto Refresh",
                value=st.session_state.auto_refresh,
                key="auto_refresh_toggle"
            )

            if auto_refresh:
                refresh_interval = st.slider(
                    "Refresh Interval (seconds)",
                    min_value=5,
                    max_value=300,
                    value=st.session_state.refresh_interval,
                    step=5,
                    key="refresh_interval_slider"
                )
                st.session_state.refresh_interval = refresh_interval

            st.session_state.auto_refresh = auto_refresh

            # Quick actions
            st.markdown("### ⚡ Quick Actions")

            if st.button("🔄 Refresh Data", use_container_width=True):
                self.refresh_data()
                st.rerun()

            if st.button("📊 Generate Report", use_container_width=True):
                self.generate_risk_report()

            if st.button("📧 Send Alerts", use_container_width=True):
                self.send_risk_alerts()

            if st.button("💾 Export Data", use_container_width=True):
                self.export_portfolio_data()

            # Risk alerts summary
            st.markdown("### 🚨 Active Alerts")
            alerts = self.get_active_alerts()
            if alerts:
                for alert in alerts[:5]:  # Show top 5 alerts
                    severity_color = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
                    st.write(f"{severity_color.get(alert['severity'], '⚪')} {alert['message']}")

                if len(alerts) > 5:
                    st.write(f"... and {len(alerts) - 5} more alerts")
            else:
                st.success("No active alerts")

            return page

    def get_available_portfolios(self) -> List[str]:
        """Get list of available portfolios"""
        return [
            "Total Portfolio",
            "Equity Portfolio",
            "Fixed Income Portfolio",
            "Derivatives Portfolio",
            "Alternative Investments",
            "Cash & Equivalents"
        ]

    def refresh_data(self):
        """Refresh all dashboard data"""
        with st.spinner("Refreshing data..."):
            # Simulate data refresh
            time.sleep(2)
            st.success("Data refreshed successfully!")

    def get_active_alerts(self) -> List[Dict]:
        """Get active risk alerts"""
        # Simulate active alerts
        return [
            {"severity": "HIGH", "message": "VaR limit breach - Equity desk"},
            {"severity": "MEDIUM", "message": "Concentration risk - Tech sector"},
            {"severity": "LOW", "message": "Model validation due"}
        ]

    def render_portfolio_overview(self):
        """Render the portfolio overview page"""
        st.markdown("## 📊 Portfolio Overview")

        # Key risk metrics - top row
        self.render_key_metrics()

        # Charts row
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            self.render_portfolio_composition_chart()
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            self.render_pnl_attribution_chart()
            st.markdown('</div>', unsafe_allow_html=True)

        # Risk metrics table
        st.markdown("### 📋 Detailed Risk Metrics")
        self.render_risk_metrics_table()

        # Real-time P&L chart
        st.markdown("### 📈 Real-time P&L Evolution")
        self.render_realtime_pnl_chart()

    def render_key_metrics(self):
        """Render key portfolio risk metrics"""
        # Get current risk metrics
        metrics = self.get_current_risk_metrics()

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            self.render_metric_card(
                "Portfolio Value",
                f"${metrics['portfolio_value']:,.0f}M",
                f"{metrics['portfolio_value_change']:+.1f}%",
                "💼"
            )

        with col2:
            self.render_metric_card(
                "1-Day VaR (95%)",
                f"${metrics['var_1d']:,.1f}M",
                f"{metrics['var_1d_change']:+.1f}%",
                "⚠️"
            )

        with col3:
            self.render_metric_card(
                "Expected Shortfall",
                f"${metrics['expected_shortfall']:,.1f}M",
                f"{metrics['es_change']:+.1f}%",
                "📉"
            )

        with col4:
            self.render_metric_card(
                "Sharpe Ratio",
                f"{metrics['sharpe_ratio']:.2f}",
                f"{metrics['sharpe_change']:+.2f}",
                "📊"
            )

        with col5:
            self.render_metric_card(
                "Max Drawdown",
                f"{metrics['max_drawdown']:.1f}%",
                f"{metrics['drawdown_change']:+.1f}%",
                "📉"
            )

    def render_metric_card(self, title: str, value: str, change: str, icon: str):
        """Render a metric card with styling"""
        # Determine alert level based on change
        change_value = float(change.replace('%', '').replace('+', ''))
        if abs(change_value) > 10:
            alert_class = "risk-alert-high"
        elif abs(change_value) > 5:
            alert_class = "risk-alert-medium"
        else:
            alert_class = "risk-alert-low"

        st.markdown(f"""
        <div class="metric-card {alert_class}">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h4 style="margin: 0; color: #374151;">{icon} {title}</h4>
                    <h2 style="margin: 0.5rem 0; color: #111827;">{value}</h2>
                    <p style="margin: 0; color: {'#dc2626' if change_value < 0 else '#059669'}; font-weight: 600;">
                        {change}
                    </p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    def get_current_risk_metrics(self) -> Dict:
        """Get current portfolio risk metrics"""
        # Simulate real-time risk metrics
        np.random.seed(int(time.time()) % 1000)

        return {
            'portfolio_value': 2500 + np.random.normal(0, 50),
            'portfolio_value_change': np.random.normal(0, 2),
            'var_1d': 15.5 + np.random.normal(0, 2),
            'var_1d_change': np.random.normal(0, 5),
            'expected_shortfall': 22.3 + np.random.normal(0, 3),
            'es_change': np.random.normal(0, 6),
            'sharpe_ratio': 1.45 + np.random.normal(0, 0.1),
            'sharpe_change': np.random.normal(0, 0.05),
            'max_drawdown': -8.5 + np.random.normal(0, 1),
            'drawdown_change': np.random.normal(0, 2)
        }

    def render_portfolio_composition_chart(self):
        """Render portfolio composition pie chart"""
        st.subheader("🥧 Portfolio Composition")

        # Sample portfolio data
        composition_data = {
            'Asset Class': ['Equities', 'Fixed Income', 'Alternatives', 'Cash', 'Derivatives'],
            'Value (%)': [45, 30, 15, 5, 5],
            'VaR Contribution (%)': [60, 20, 12, 1, 7]
        }

        df = pd.DataFrame(composition_data)

        # Create pie chart
        fig = px.pie(
            df,
            values='Value (%)',
            names='Asset Class',
            title="Portfolio Allocation",
            color_discrete_sequence=px.colors.qualitative.Set3
        )

        fig.update_layout(
            showlegend=True,
            height=400,
            font=dict(size=12)
        )

        st.plotly_chart(fig, use_container_width=True)

        # VaR contribution chart
        fig2 = px.bar(
            df,
            x='Asset Class',
            y='VaR Contribution (%)',
            title="VaR Contribution by Asset Class",
            color='VaR Contribution (%)',
            color_continuous_scale='Reds'
        )

        fig2.update_layout(height=300)
        st.plotly_chart(fig2, use_container_width=True)

    def render_pnl_attribution_chart(self):
        """Render P&L attribution waterfall chart"""
        st.subheader("💧 P&L Attribution")

        # Sample P&L attribution data
        categories = ['Starting P&L', 'Equities', 'Fixed Income', 'FX', 'Interest Rates', 'Credit', 'Other', 'Ending P&L']
        values = [0, 2.5, -0.8, 1.2, -0.3, 0.6, 0.2, 0]

        # Calculate cumulative values
        cumulative = np.cumsum([0] + values[1:-1])

        fig = go.Figure(go.Waterfall(
            name="P&L Attribution",
            orientation="v",
            measure=["absolute"] + ["relative"] * (len(categories) - 2) + ["total"],
            x=categories,
            textposition="outside",
            text=[f"${v:+.1f}M" for v in values],
            y=values,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": "#10b981"}},
            decreasing={"marker": {"color": "#ef4444"}},
            totals={"marker": {"color": "#3b82f6"}}
        ))

        fig.update_layout(
            title="Daily P&L Attribution Breakdown",
            showlegend=False,
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    def render_risk_metrics_table(self):
        """Render detailed risk metrics table"""
        # Generate sample risk metrics data
        risk_data = {
            'Risk Factor': ['Equity Risk', 'Interest Rate Risk', 'FX Risk', 'Credit Risk', 'Commodity Risk'],
            'VaR (1D 95%)': [8.5, 3.2, 2.1, 1.8, 0.9],
            'VaR (10D 95%)': [26.9, 10.1, 6.6, 5.7, 2.8],
            'Expected Shortfall': [12.1, 4.6, 3.0, 2.6, 1.3],
            'Limit Utilization (%)': [68, 42, 33, 28, 15],
            'Status': ['🟡 Warning', '🟢 Normal', '🟢 Normal', '🟢 Normal', '🟢 Normal']
        }

        df = pd.DataFrame(risk_data)

        # Style the dataframe
        def highlight_utilization(val):
            if isinstance(val, (int, float)):
                if val > 80:
                    return 'background-color: #fef2f2; color: #991b1b'
                elif val > 60:
                    return 'background-color: #fffbeb; color: #92400e'
                else:
                    return 'background-color: #f0fdf4; color: #166534'
            return ''

        styled_df = df.style.applymap(highlight_utilization, subset=['Limit Utilization (%)'])

        st.dataframe(styled_df, use_container_width=True, height=250)

    def render_realtime_pnl_chart(self):
        """Render real-time P&L evolution chart"""
        # Generate sample time series data
        times = pd.date_range(start='09:00', end='16:00', freq='5min')
        np.random.seed(42)

        # Simulate intraday P&L
        returns = np.random.normal(0, 0.1, len(times))
        pnl = np.cumsum(returns)

        df = pd.DataFrame({
            'Time': times,
            'P&L (M$)': pnl,
            'Cumulative Return (%)': pnl * 0.1
        })

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Intraday P&L Evolution', 'Cumulative Return'),
            vertical_spacing=0.1
        )

        # P&L line chart
        fig.add_trace(
            go.Scatter(
                x=df['Time'],
                y=df['P&L (M$)'],
                mode='lines',
                name='P&L',
                line=dict(color='#3b82f6', width=2),
                fill='tonexty' if any(df['P&L (M$)'] > 0) else 'tozeroy',
                fillcolor='rgba(59, 130, 246, 0.1)'
            ),
            row=1, col=1
        )

        # Cumulative return
        fig.add_trace(
            go.Scatter(
                x=df['Time'],
                y=df['Cumulative Return (%)'],
                mode='lines',
                name='Return %',
                line=dict(color='#10b981', width=2)
            ),
            row=2, col=1
        )

        fig.update_layout(
            height=500,
            showlegend=False,
            xaxis_tickformat='%H:%M'
        )

        st.plotly_chart(fig, use_container_width=True)

    def run(self):
        """Main dashboard execution"""
        # Check authentication
        if not st.session_state.authenticated:
            self.auth_manager.render_login()
            return

        # Render header
        self.render_header()

        # Render sidebar and get selected page
        selected_page = self.render_sidebar()

        # Auto-refresh mechanism
        if st.session_state.auto_refresh:
            placeholder = st.empty()
            with placeholder.container():
                # Route to appropriate page
                if "Portfolio Overview" in selected_page:
                    self.render_portfolio_overview()
                elif "VaR Analysis" in selected_page:
                    self.render_var_analysis()
                elif "Stress Testing" in selected_page:
                    self.render_stress_testing()
                elif "Greeks Monitoring" in selected_page:
                    self.render_greeks_monitoring()
                elif "Risk Limits" in selected_page:
                    self.render_risk_limits()
                elif "Regulatory Capital" in selected_page:
                    self.render_regulatory_capital()
                elif "Regulatory Compliance" in selected_page:
                    self.render_regulatory_compliance()
                elif "Factor Analysis" in selected_page:
                    self.render_factor_analysis()
                elif "Mobile View" in selected_page:
                    self.render_mobile_view()
                elif "Settings" in selected_page:
                    self.render_settings()

            # Auto-refresh timer
            time.sleep(st.session_state.refresh_interval)
            st.rerun()
        else:
            # Route to appropriate page without auto-refresh
            if "Portfolio Overview" in selected_page:
                self.render_portfolio_overview()
            elif "VaR Analysis" in selected_page:
                self.render_var_analysis()
            elif "Stress Testing" in selected_page:
                self.render_stress_testing()
            elif "Greeks Monitoring" in selected_page:
                self.render_greeks_monitoring()
            elif "Risk Limits" in selected_page:
                self.render_risk_limits()
            elif "Regulatory Capital" in selected_page:
                self.render_regulatory_capital()
            elif "Regulatory Compliance" in selected_page:
                self.render_regulatory_compliance()
            elif "Factor Analysis" in selected_page:
                self.render_factor_analysis()
            elif "Mobile View" in selected_page:
                self.render_mobile_view()
            elif "Settings" in selected_page:
                self.render_settings()

    def render_var_analysis(self):
        """Render comprehensive VaR analysis page"""
        st.markdown("## 📈 Value at Risk Analysis")

        # Get sample portfolio for analysis
        sample_portfolio = self.risk_engine.generate_sample_portfolio()

        # Calculate VaR using different methods
        var_results = self.risk_engine.calculate_portfolio_var(sample_portfolio)

        # VaR method comparison
        st.markdown("### 🔍 VaR Methodology Comparison")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Historical Simulation",
                f"${var_results['historical_var']:.1f}M",
                delta=f"{np.random.normal(0, 5):.1f}%"
            )

        with col2:
            st.metric(
                "Parametric (Normal)",
                f"${var_results['parametric_var']:.1f}M",
                delta=f"{np.random.normal(0, 4):.1f}%"
            )

        with col3:
            st.metric(
                "Monte Carlo",
                f"${var_results['monte_carlo_var']:.1f}M",
                delta=f"{np.random.normal(0, 3):.1f}%"
            )

        with col4:
            st.metric(
                "Diversified VaR",
                f"${var_results['diversified_var']:.1f}M",
                delta=f"{np.random.normal(0, 2):.1f}%"
            )

        # Monte Carlo simulation distribution
        st.markdown("### 📊 Monte Carlo Simulation Results")

        # Generate simulation data
        np.random.seed(42)
        mc_results = np.random.normal(-2, 15, 10000)  # Simulated P&L distribution

        fig_mc = self.viz_engine.create_monte_carlo_distribution(mc_results)
        st.plotly_chart(fig_mc, use_container_width=True)

        # VaR breakdown by asset class and region
        st.markdown("### 🍰 VaR Breakdown Analysis")

        # Generate breakdown data
        var_breakdown = {
            'by_asset_class': {
                'Equity': {'total_var': 12.5},
                'Fixed Income': {'total_var': 3.2},
                'Derivatives': {'total_var': 2.8},
                'Alternatives': {'total_var': 1.0}
            },
            'by_region': {
                'North America': {'total_var': 8.7},
                'Europe': {'total_var': 4.2},
                'Asia Pacific': {'total_var': 3.1},
                'Emerging Markets': {'total_var': 3.5}
            }
        }

        fig_breakdown = self.viz_engine.create_var_breakdown_chart(var_breakdown)
        st.plotly_chart(fig_breakdown, use_container_width=True)

        # Component VaR analysis
        st.markdown("### 🔧 Component VaR Analysis")

        component_data = {
            'Asset Class': ['Equities', 'Fixed Income', 'Derivatives', 'Alternatives', 'Cash'],
            'Portfolio Weight (%)': [45, 30, 15, 8, 2],
            'Standalone VaR': [18.5, 4.2, 8.1, 3.2, 0.1],
            'Component VaR': [12.5, 3.2, 2.8, 1.0, 0.05],
            'Marginal VaR': [0.28, 0.11, 0.19, 0.13, 0.03],
            'Diversification Benefit': [6.0, 1.0, 5.3, 2.2, 0.05]
        }

        component_df = pd.DataFrame(component_data)

        # Style the dataframe
        def highlight_high_var(val):
            if isinstance(val, (int, float)) and val > 10:
                return 'background-color: #fef2f2; color: #991b1b'
            elif isinstance(val, (int, float)) and val > 5:
                return 'background-color: #fffbeb; color: #92400e'
            return ''

        styled_df = component_df.style.applymap(highlight_high_var, subset=['Standalone VaR', 'Component VaR'])
        st.dataframe(styled_df, use_container_width=True)

        # Model validation and backtesting
        st.markdown("### ✅ Model Validation & Backtesting")

        col1, col2 = st.columns(2)

        with col1:
            # Backtesting results
            st.markdown("#### Backtesting Results (Last 250 Days)")

            backtest_data = {
                'Model': ['Historical Simulation', 'Parametric', 'Monte Carlo'],
                'Violations': [12, 15, 11],
                'Expected Violations': [12.5, 12.5, 12.5],
                'Success Rate (%)': [95.2, 94.0, 95.6],
                'Traffic Light': ['🟢 Green', '🟡 Yellow', '🟢 Green']
            }

            backtest_df = pd.DataFrame(backtest_data)
            st.dataframe(backtest_df, use_container_width=True)

        with col2:
            # Model performance metrics
            st.markdown("#### Model Performance Metrics")

            perf_metrics = {
                'Metric': ['Kupiec Test p-value', 'Christoffersen Test', 'Average Loss Given Breach', 'Maximum Loss'],
                'Historical Sim': [0.85, 'PASS', '$28.5M', '$45.2M'],
                'Parametric': [0.42, 'PASS', '$31.2M', '$52.1M'],
                'Monte Carlo': [0.91, 'PASS', '$27.8M', '$44.6M']
            }

            perf_df = pd.DataFrame(perf_metrics)
            st.dataframe(perf_df, use_container_width=True)

    def render_stress_testing(self):
        """Render comprehensive stress testing page"""
        st.markdown("## 🧪 Stress Testing & Scenario Analysis")

        # Get sample portfolio for stress testing
        sample_portfolio = self.risk_engine.generate_sample_portfolio()

        # Run stress tests
        stress_results = self.risk_engine.run_stress_tests(sample_portfolio)

        # Stress test results overview
        st.markdown("### 📊 Stress Test Results Overview")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            worst_scenario = min(stress_results.values())
            st.metric(
                "Worst Case Scenario",
                f"${worst_scenario:.1f}M",
                delta=f"{(worst_scenario / 2500) * 100:.1f}%",
                delta_color="inverse"
            )

        with col2:
            avg_loss = np.mean(list(stress_results.values()))
            st.metric(
                "Average Stress Loss",
                f"${avg_loss:.1f}M",
                delta=f"{(avg_loss / 2500) * 100:.1f}%"
            )

        with col3:
            scenarios_breached = sum(1 for loss in stress_results.values() if loss < -50)
            st.metric(
                "Severe Scenarios",
                f"{scenarios_breached}/5",
                delta="High Impact" if scenarios_breached > 2 else "Manageable"
            )

        with col4:
            max_impact_pct = abs(worst_scenario / 2500) * 100
            st.metric(
                "Max Portfolio Impact",
                f"{max_impact_pct:.1f}%",
                delta="Critical" if max_impact_pct > 15 else "Acceptable"
            )

        # Scenario analysis chart
        st.markdown("### 📈 Scenario Impact Analysis")

        fig_scenarios = self.viz_engine.create_scenario_analysis_chart(stress_results)
        st.plotly_chart(fig_scenarios, use_container_width=True)

        # Detailed scenario breakdown
        st.markdown("### 🔍 Detailed Scenario Analysis")

        col1, col2 = st.columns(2)

        with col1:
            # Market scenarios
            st.markdown("#### Market Risk Scenarios")

            market_scenarios = {
                'Scenario': ['Equity Crash (-20%)', 'Bond Crash (+200bp)', 'Volatility Spike (+50%)', 'Credit Crisis (+500bp)'],
                'Portfolio Impact': [f"${stress_results.get('market_crash', -45):.1f}M",
                                   f"${stress_results.get('interest_rate_shock', -12):.1f}M",
                                   "$-8.5M",
                                   f"${stress_results.get('credit_crisis', -6):.1f}M"],
                'Probability': ['5%', '10%', '15%', '8%'],
                'Time Horizon': ['1 Week', '1 Month', '1 Day', '2 Weeks']
            }

            market_df = pd.DataFrame(market_scenarios)
            st.dataframe(market_df, use_container_width=True)

        with col2:
            # Operational scenarios
            st.markdown("#### Operational Risk Scenarios")

            operational_scenarios = {
                'Scenario': ['Liquidity Crisis', 'Cyber Attack', 'Key Person Risk', 'Regulatory Change'],
                'Portfolio Impact': [f"${stress_results.get('liquidity_crisis', -5):.1f}M",
                                   "$-15.2M", "$-3.8M", "$-8.1M"],
                'Probability': ['3%', '2%', '5%', '12%'],
                'Recovery Time': ['2-4 Weeks', '1-2 Days', '3-6 Months', '6-12 Months']
            }

            operational_df = pd.DataFrame(operational_scenarios)
            st.dataframe(operational_df, use_container_width=True)

        # Custom scenario builder
        st.markdown("### 🛠️ Custom Scenario Builder")

        col1, col2, col3 = st.columns(3)

        with col1:
            equity_shock = st.slider("Equity Market Shock (%)", -50, 50, -20)
            bond_shock = st.slider("Bond Yield Change (bp)", -300, 500, 200)

        with col2:
            fx_shock = st.slider("USD Strengthening (%)", -30, 30, 10)
            vol_shock = st.slider("Volatility Change (%)", -50, 100, 25)

        with col3:
            credit_shock = st.slider("Credit Spread Widening (bp)", 0, 1000, 250)
            commodity_shock = st.slider("Commodity Shock (%)", -40, 40, -15)

        if st.button("🔥 Run Custom Scenario", use_container_width=True):
            # Calculate custom scenario impact
            custom_impact = (equity_shock * 0.45 * 25 +  # 45% equity exposure
                           bond_shock * 0.30 * 0.05 +     # 30% bond exposure
                           fx_shock * 0.20 * 25 +         # 20% foreign exposure
                           vol_shock * 0.15 * 0.1)        # Options exposure

            st.success(f"Custom Scenario Impact: ${custom_impact:.1f}M ({(custom_impact/2500)*100:.1f}% of portfolio)")

        # Stress testing configuration
        st.markdown("### ⚙️ Stress Testing Configuration")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Model Parameters")

            confidence_level = st.selectbox("Confidence Level", ["95%", "99%", "99.9%"], index=1)
            time_horizon = st.selectbox("Time Horizon", ["1 Day", "1 Week", "1 Month"], index=1)
            correlation_breakdown = st.checkbox("Include Correlation Breakdown", value=True)

        with col2:
            st.markdown("#### Historical Scenarios")

            historical_scenario = st.selectbox(
                "Historical Event",
                ["2008 Financial Crisis", "2020 COVID-19", "1998 LTCM", "2010 Flash Crash", "Brexit"]
            )

            if st.button("📅 Apply Historical Scenario"):
                scenario_impacts = {
                    "2008 Financial Crisis": -185.5,
                    "2020 COVID-19": -142.3,
                    "1998 LTCM": -67.8,
                    "2010 Flash Crash": -23.4,
                    "Brexit": -45.6
                }

                impact = scenario_impacts.get(historical_scenario, -50)
                st.error(f"{historical_scenario} Impact: ${impact:.1f}M ({(impact/2500)*100:.1f}% of portfolio)")

    def render_greeks_monitoring(self):
        """Render Greeks monitoring page for options portfolios"""
        st.markdown("## 📉 Greeks Monitoring")

        # Get sample portfolio for Greeks analysis
        sample_portfolio = self.risk_engine.generate_sample_portfolio()

        # Calculate Greeks
        greeks_data = self.risk_engine.calculate_greeks(sample_portfolio)

        # Total portfolio Greeks
        st.markdown("### 📈 Portfolio Greeks Summary")

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            delta_total = greeks_data['total']['delta']
            st.metric(
                "Total Delta",
                f"{delta_total:,.0f}",
                delta=f"{np.random.normal(0, 100):.0f}"
            )

        with col2:
            gamma_total = greeks_data['total']['gamma']
            st.metric(
                "Total Gamma",
                f"{gamma_total:.1f}",
                delta=f"{np.random.normal(0, 0.5):.1f}"
            )

        with col3:
            theta_total = greeks_data['total']['theta']
            st.metric(
                "Total Theta",
                f"{theta_total:,.0f}",
                delta=f"{np.random.normal(0, 50):.0f}",
                delta_color="inverse"
            )

        with col4:
            vega_total = greeks_data['total']['vega']
            st.metric(
                "Total Vega",
                f"{vega_total:,.0f}",
                delta=f"{np.random.normal(0, 200):.0f}"
            )

        with col5:
            rho_total = greeks_data['total']['rho']
            st.metric(
                "Total Rho",
                f"{rho_total:,.0f}",
                delta=f"{np.random.normal(0, 500):.0f}"
            )

        # Greeks ladder chart
        st.markdown("### 🔲 Greeks Ladder by Asset Class")

        # Generate detailed Greeks data
        greeks_by_category = {
            'Equity Options': {
                'delta': 850, 'gamma': 12.5, 'theta': -125, 'vega': 2500, 'rho': 450
            },
            'Index Options': {
                'delta': 350, 'gamma': 8.2, 'theta': -85, 'vega': 1800, 'rho': 320
            },
            'FX Options': {
                'delta': 200, 'gamma': 5.1, 'theta': -45, 'vega': 900, 'rho': 180
            },
            'Bond Options': {
                'delta': 150, 'gamma': 3.8, 'theta': -35, 'vega': 650, 'rho': 1200
            }
        }

        fig_greeks = self.viz_engine.create_greeks_ladder_chart(greeks_by_category)
        st.plotly_chart(fig_greeks, use_container_width=True)

        # Greeks risk analysis
        st.markdown("### ⚠️ Greeks Risk Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Delta Analysis")

            delta_data = {
                'Underlying': ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM'],
                'Net Delta': [450, 285, 125, 95, 65],
                'Delta %': [45.2, 28.7, 12.6, 9.5, 6.5],
                '1% Move P&L': ['$4.5M', '$2.9M', '$1.3M', '$0.95M', '$0.65M'],
                'Hedge Ratio': ['Neutral', 'Long Bias', 'Neutral', 'Short Bias', 'Neutral']
            }

            delta_df = pd.DataFrame(delta_data)
            st.dataframe(delta_df, use_container_width=True)

        with col2:
            st.markdown("#### Gamma & Theta Analysis")

            gamma_theta_data = {
                'Strategy': ['Long Straddles', 'Short Iron Condors', 'Covered Calls', 'Protective Puts'],
                'Gamma Exposure': [15.2, -8.5, -2.1, 3.8],
                'Theta Decay': [-85, 45, 25, -15],
                'Days to Expiry': [25, 15, 35, 45],
                'Risk Level': ['High', 'Medium', 'Low', 'Medium']
            }

            gt_df = pd.DataFrame(gamma_theta_data)

            # Color code by risk level
            def highlight_risk(row):
                if row['Risk Level'] == 'High':
                    return ['background-color: #fef2f2'] * len(row)
                elif row['Risk Level'] == 'Medium':
                    return ['background-color: #fffbeb'] * len(row)
                else:
                    return ['background-color: #f0fdf4'] * len(row)

            styled_gt_df = gt_df.style.apply(highlight_risk, axis=1)
            st.dataframe(styled_gt_df, use_container_width=True)

        # Scenario analysis for Greeks
        st.markdown("### 🔄 Greeks Scenario Analysis")

        scenario_tabs = st.tabs(["Volatility Scenarios", "Time Decay", "Interest Rate"])

        with scenario_tabs[0]:
            st.markdown("#### Impact of Volatility Changes")

            vol_scenarios = [-50, -25, -10, 0, 10, 25, 50]
            vega_impact = [vega_total * (vol/100) for vol in vol_scenarios]

            vol_df = pd.DataFrame({
                'Vol Change (%)': vol_scenarios,
                'Portfolio Impact': [f"${impact/1000:.1f}M" for impact in vega_impact],
                'New Portfolio Value': [f"${(2500 + impact/1000):.1f}M" for impact in vega_impact]
            })

            st.dataframe(vol_df, use_container_width=True)

        with scenario_tabs[1]:
            st.markdown("#### Time Decay Analysis")

            days = [1, 7, 14, 30]
            theta_impact = [theta_total * day for day in days]

            theta_df = pd.DataFrame({
                'Days Forward': days,
                'Theta Decay': [f"${impact/1000:.1f}M" for impact in theta_impact],
                'Cumulative Impact': [f"${sum(theta_impact[:i+1])/1000:.1f}M" for i in range(len(days))]
            })

            st.dataframe(theta_df, use_container_width=True)

        with scenario_tabs[2]:
            st.markdown("#### Interest Rate Sensitivity")

            ir_changes = [-100, -50, -25, 0, 25, 50, 100]
            rho_impact = [rho_total * (ir/10000) for ir in ir_changes]  # Convert bp to decimal

            rho_df = pd.DataFrame({
                'Rate Change (bp)': ir_changes,
                'Portfolio Impact': [f"${impact/1000:.2f}M" for impact in rho_impact],
                'Impact %': [f"{(impact/25000):.2f}%" for impact in rho_impact]
            })

            st.dataframe(rho_df, use_container_width=True)

        # Greeks limits and alerts
        st.markdown("### 🚨 Greeks Limits & Monitoring")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Current Limits Utilization")

            limits_data = {
                'Greek': ['Delta', 'Gamma', 'Theta', 'Vega', 'Rho'],
                'Current': [1550, 29.6, -290, 5850, 2150],
                'Limit': [2000, 50, -500, 8000, 3000],
                'Utilization (%)': [77.5, 59.2, 58.0, 73.1, 71.7],
                'Status': ['🟡 Warning', '🟢 Normal', '🟢 Normal', '🟡 Warning', '🟡 Warning']
            }

            limits_df = pd.DataFrame(limits_data)
            st.dataframe(limits_df, use_container_width=True)

        with col2:
            st.markdown("#### Real-time Alerts")

            st.info("🟢 Delta exposure within acceptable range")
            st.warning("🟡 Vega limit utilization above 70%")
            st.warning("🟡 Rho exposure increasing with rate volatility")
            st.success("✅ All Greeks positions properly hedged")

        # Greeks hedging recommendations
        st.markdown("### 🛡️ Hedging Recommendations")

        recommendations = [
            {
                'Priority': 'High',
                'Greek': 'Vega',
                'Action': 'Sell volatility',
                'Recommendation': 'Reduce vega exposure by 1,000 through short vol strategies',
                'Expected Impact': 'Reduce portfolio vol sensitivity by 15%'
            },
            {
                'Priority': 'Medium',
                'Greek': 'Delta',
                'Action': 'Rebalance hedge',
                'Recommendation': 'Adjust index hedge ratio to maintain delta neutrality',
                'Expected Impact': 'Improve directional risk control'
            },
            {
                'Priority': 'Low',
                'Greek': 'Theta',
                'Action': 'Monitor time decay',
                'Recommendation': 'Close short-dated positions approaching expiry',
                'Expected Impact': 'Optimize time decay profile'
            }
        ]

        rec_df = pd.DataFrame(recommendations)

        # Color code by priority
        def highlight_priority(row):
            if row['Priority'] == 'High':
                return ['background-color: #fef2f2; color: #991b1b'] * len(row)
            elif row['Priority'] == 'Medium':
                return ['background-color: #fffbeb; color: #92400e'] * len(row)
            else:
                return ['background-color: #f0fdf4; color: #166534'] * len(row)

        styled_rec_df = rec_df.style.apply(highlight_priority, axis=1)
        st.dataframe(styled_rec_df, use_container_width=True)

    def render_risk_limits(self):
        """Render risk limits monitoring page"""
        st.markdown("## 🌡️ Risk Limits Monitoring")

        # Load risk limits from alert manager
        risk_limits = self.alert_manager.risk_limits

        # Risk limits overview
        st.markdown("### 📊 Risk Limits Dashboard")

        # Simulate current portfolio metrics for limit checking
        current_metrics = {
            'var_1d': 22.5,
            'expected_shortfall': 34.2,
            'concentration': 8.5,
            'sector_concentration': 32.0,
            'leverage': 2.1,
            'liquidity_coverage': 105.0
        }

        # Check limits and generate alerts
        active_alerts = self.alert_manager.check_risk_limits(current_metrics)

        # Display limit utilization
        st.markdown("#### Limit Utilization Overview")

        cols = st.columns(3)
        limit_count = 0

        for limit in risk_limits:
            if limit_count >= 6:  # Display first 6 limits
                break

            col_idx = limit_count % 3
            current_value = current_metrics.get(limit.metric_type, 0)

            with cols[col_idx]:
                # Calculate utilization percentage
                if limit.breach_threshold != 0:
                    utilization = (current_value / limit.breach_threshold) * 100
                else:
                    utilization = 0

                # Determine status color
                if current_value >= limit.critical_threshold:
                    status_color = "red"
                    status_text = "CRITICAL"
                elif current_value >= limit.breach_threshold:
                    status_color = "orange"
                    status_text = "BREACH"
                elif current_value >= limit.warning_threshold:
                    status_color = "yellow"
                    status_text = "WARNING"
                else:
                    status_color = "green"
                    status_text = "NORMAL"

                st.metric(
                    limit.limit_name,
                    f"{current_value:.1f} {limit.currency}",
                    delta=f"{utilization:.1f}% utilized"
                )

                # Status indicator
                st.markdown(f"**Status:** :{status_color}[{status_text}]")

            limit_count += 1

        # Detailed limits table
        st.markdown("### 📋 Detailed Risk Limits")

        limits_data = []
        for limit in risk_limits:
            current_value = current_metrics.get(limit.metric_type, 0)

            # Determine status
            if current_value >= limit.critical_threshold:
                status = "🔴 CRITICAL"
                utilization_pct = (current_value / limit.critical_threshold) * 100
            elif current_value >= limit.breach_threshold:
                status = "🟠 BREACH"
                utilization_pct = (current_value / limit.breach_threshold) * 100
            elif current_value >= limit.warning_threshold:
                status = "🟡 WARNING"
                utilization_pct = (current_value / limit.warning_threshold) * 100
            else:
                status = "🟢 NORMAL"
                utilization_pct = (current_value / limit.warning_threshold) * 100

            limits_data.append({
                'Limit Name': limit.limit_name,
                'Current Value': f"{current_value:.1f} {limit.currency}",
                'Warning': f"{limit.warning_threshold:.1f}",
                'Breach': f"{limit.breach_threshold:.1f}",
                'Critical': f"{limit.critical_threshold:.1f}",
                'Utilization (%)': f"{utilization_pct:.1f}%",
                'Status': status,
                'Portfolio': limit.portfolio_name
            })

        limits_df = pd.DataFrame(limits_data)

        # Style the dataframe
        def highlight_status(row):
            if 'CRITICAL' in row['Status']:
                return ['background-color: #fef2f2; color: #991b1b'] * len(row)
            elif 'BREACH' in row['Status']:
                return ['background-color: #fef3c7; color: #92400e'] * len(row)
            elif 'WARNING' in row['Status']:
                return ['background-color: #fffbeb; color: #d97706'] * len(row)
            else:
                return ['background-color: #f0fdf4; color: #166534'] * len(row)

        styled_limits_df = limits_df.style.apply(highlight_status, axis=1)
        st.dataframe(styled_limits_df, use_container_width=True)

        # Alert management
        if active_alerts:
            st.markdown("### 🚨 Active Risk Alerts")
            self.alert_manager.render_alert_dashboard()
        else:
            st.success("🎉 All risk metrics are within acceptable limits")

        # Limit configuration
        st.markdown("### ⚙️ Limit Configuration")

        # Only show for authorized users
        if st.session_state.get('user_role') in ['Administrator', 'Risk Manager']:
            st.markdown("#### Modify Risk Limits")

            selected_limit = st.selectbox(
                "Select Limit to Modify",
                [limit.limit_name for limit in risk_limits]
            )

            # Find the selected limit
            limit_to_modify = next((limit for limit in risk_limits if limit.limit_name == selected_limit), None)

            if limit_to_modify:
                col1, col2, col3 = st.columns(3)

                with col1:
                    new_warning = st.number_input(
                        "Warning Threshold",
                        value=float(limit_to_modify.warning_threshold),
                        step=0.1
                    )

                with col2:
                    new_breach = st.number_input(
                        "Breach Threshold",
                        value=float(limit_to_modify.breach_threshold),
                        step=0.1
                    )

                with col3:
                    new_critical = st.number_input(
                        "Critical Threshold",
                        value=float(limit_to_modify.critical_threshold),
                        step=0.1
                    )

                if st.button("Update Limit"):
                    st.success(f"Limit '{selected_limit}' updated successfully")
                    st.info("Note: In production, this would update the database and trigger notifications")

        else:
            st.info("🔒 Limit modification requires Risk Manager or Administrator privileges")

        # Limit breach history
        st.markdown("### 📈 Historical Limit Breaches")

        # Simulate historical breach data
        breach_history = {
            'Date': ['2024-01-10', '2024-01-05', '2023-12-28', '2023-12-15'],
            'Limit Breached': ['VaR 1-Day Total', 'Sector Concentration', 'VaR 1-Day Equity', 'Expected Shortfall'],
            'Breach Value': ['$26.5M', '36.2%', '$21.8M', '$42.1M'],
            'Limit Value': ['$25.0M', '35.0%', '$20.0M', '$40.0M'],
            'Duration (Hours)': [4.5, 12.0, 2.5, 8.0],
            'Resolution': ['Market normalization', 'Position reduction', 'Hedge adjustment', 'Risk reduction']
        }

        breach_df = pd.DataFrame(breach_history)
        st.dataframe(breach_df, use_container_width=True)

        # Risk limit analytics
        st.markdown("### 📊 Limit Performance Analytics")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Breach Frequency (Last 12 Months)")

            breach_freq = {
                'Limit Type': ['VaR Limits', 'Concentration Limits', 'Leverage Limits', 'Liquidity Limits'],
                'Breaches': [8, 12, 3, 5],
                'Avg Duration (hrs)': [6.2, 18.4, 4.1, 12.8],
                'Max Duration (hrs)': [24.5, 48.2, 8.5, 36.0]
            }

            freq_df = pd.DataFrame(breach_freq)
            st.dataframe(freq_df, use_container_width=True)

        with col2:
            st.markdown("#### Limit Effectiveness Metrics")

            effectiveness = {
                'Metric': ['True Positive Rate', 'False Positive Rate', 'Average Response Time', 'Limit Calibration'],
                'Current Value': ['94.2%', '8.5%', '15 minutes', 'Good'],
                'Target': ['>90%', '<10%', '<20 minutes', 'Excellent'],
                'Status': ['✅ Good', '✅ Good', '✅ Good', '🟡 Fair']
            }

            eff_df = pd.DataFrame(effectiveness)
            st.dataframe(eff_df, use_container_width=True)

    def render_regulatory_capital(self):
        """Render regulatory capital tracking page"""
        st.markdown("## 📋 Regulatory Capital Requirements")

        # Capital adequacy overview
        st.markdown("### 🏦 Capital Adequacy Overview")

        # Simulate regulatory capital data
        regulatory_data = {
            'total_capital': 450.0,
            'tier1_capital': 380.0,
            'common_equity_tier1': 320.0,
            'risk_weighted_assets': 2100.0,
            'leverage_exposure': 2800.0,
            'operational_risk_capital': 85.0,
            'market_risk_capital': 120.0,
            'credit_risk_capital': 180.0
        }

        # Calculate ratios
        cet1_ratio = (regulatory_data['common_equity_tier1'] / regulatory_data['risk_weighted_assets']) * 100
        tier1_ratio = (regulatory_data['tier1_capital'] / regulatory_data['risk_weighted_assets']) * 100
        total_capital_ratio = (regulatory_data['total_capital'] / regulatory_data['risk_weighted_assets']) * 100
        leverage_ratio = (regulatory_data['tier1_capital'] / regulatory_data['leverage_exposure']) * 100

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "CET1 Ratio",
                f"{cet1_ratio:.2f}%",
                delta=f"+0.15%",
                help="Common Equity Tier 1 capital ratio (Min: 4.5%)"
            )
            if cet1_ratio >= 4.5:
                st.success("✅ Above minimum")
            else:
                st.error("❌ Below minimum")

        with col2:
            st.metric(
                "Tier 1 Ratio",
                f"{tier1_ratio:.2f}%",
                delta=f"+0.08%",
                help="Tier 1 capital ratio (Min: 6.0%)"
            )
            if tier1_ratio >= 6.0:
                st.success("✅ Above minimum")
            else:
                st.error("❌ Below minimum")

        with col3:
            st.metric(
                "Total Capital Ratio",
                f"{total_capital_ratio:.2f}%",
                delta=f"+0.12%",
                help="Total capital ratio (Min: 8.0%)"
            )
            if total_capital_ratio >= 8.0:
                st.success("✅ Above minimum")
            else:
                st.error("❌ Below minimum")

        with col4:
            st.metric(
                "Leverage Ratio",
                f"{leverage_ratio:.2f}%",
                delta=f"-0.05%",
                help="Leverage ratio (Min: 3.0%)"
            )
            if leverage_ratio >= 3.0:
                st.success("✅ Above minimum")
            else:
                st.error("❌ Below minimum")

        # Capital components breakdown
        st.markdown("### 🔍 Capital Components Breakdown")

        col1, col2 = st.columns(2)

        with col1:
            # Capital composition pie chart
            capital_components = {
                'Common Equity Tier 1': regulatory_data['common_equity_tier1'],
                'Additional Tier 1': regulatory_data['tier1_capital'] - regulatory_data['common_equity_tier1'],
                'Tier 2 Capital': regulatory_data['total_capital'] - regulatory_data['tier1_capital']
            }

            fig_capital = px.pie(
                values=list(capital_components.values()),
                names=list(capital_components.keys()),
                title="Capital Composition ($M)"
            )
            st.plotly_chart(fig_capital, use_container_width=True)

        with col2:
            # Risk-weighted assets breakdown
            rwa_components = {
                'Credit Risk RWA': 1400,
                'Market Risk RWA': 480,
                'Operational Risk RWA': 220
            }

            fig_rwa = px.pie(
                values=list(rwa_components.values()),
                names=list(rwa_components.keys()),
                title="Risk-Weighted Assets ($M)"
            )
            st.plotly_chart(fig_rwa, use_container_width=True)

        # Detailed capital requirements
        st.markdown("### 📋 Detailed Capital Requirements")

        capital_details = {
            'Capital Component': [
                'Common Equity Tier 1',
                'Additional Tier 1',
                'Tier 2 Capital',
                'Total Regulatory Capital'
            ],
            'Current Amount ($M)': [320, 60, 70, 450],
            'Required Amount ($M)': [94.5, 126.0, 168.0, 168.0],
            'Excess/(Deficit) ($M)': [225.5, -66.0, -98.0, 282.0],
            'Buffer Available ($M)': [225.5, 126.0, 168.0, 282.0],
            'Ratio (%)': [f"{cet1_ratio:.2f}%", f"{tier1_ratio:.2f}%", f"{total_capital_ratio:.2f}%", f"{total_capital_ratio:.2f}%"]
        }

        capital_df = pd.DataFrame(capital_details)

        # Style the dataframe
        def highlight_deficit(val):
            if isinstance(val, str) and '(' in val and ')' in val:
                return 'background-color: #fef2f2; color: #991b1b'
            elif isinstance(val, (int, float)) and val < 0:
                return 'background-color: #fef2f2; color: #991b1b'
            return ''

        styled_capital_df = capital_df.style.applymap(highlight_deficit, subset=['Excess/(Deficit) ($M)'])
        st.dataframe(styled_capital_df, use_container_width=True)

        # Capital planning and stress testing
        st.markdown("### 📈 Capital Planning & Stress Testing")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Regulatory Stress Test Results")

            stress_scenarios = {
                'Scenario': ['Baseline', 'Adverse', 'Severely Adverse'],
                'CET1 Ratio (%)': [15.2, 12.8, 9.5],
                'Min CET1 Ratio (%)': [15.2, 11.2, 8.1],
                'Capital Shortfall ($M)': [0, 0, 15.5],
                'Pass/Fail': ['✅ Pass', '✅ Pass', '🟡 Conditional']
            }

            stress_df = pd.DataFrame(stress_scenarios)
            st.dataframe(stress_df, use_container_width=True)

        with col2:
            st.markdown("#### Capital Forecast (Next 4 Quarters)")

            forecast = {
                'Quarter': ['Q1 2024', 'Q2 2024', 'Q3 2024', 'Q4 2024'],
                'Projected CET1 (%)': [15.8, 16.2, 16.5, 16.8],
                'Projected RWA ($B)': [2.15, 2.25, 2.35, 2.45],
                'Capital Actions': ['None', 'None', 'Dividend', 'Share Buyback'],
                'Buffer Above Min (%)': [11.3, 11.7, 12.0, 12.3]
            }

            forecast_df = pd.DataFrame(forecast)
            st.dataframe(forecast_df, use_container_width=True)

        # Regulatory reporting status
        st.markdown("### 📄 Regulatory Reporting Status")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Upcoming Submissions")

            submissions = {
                'Report': ['Basel III Return', 'CCAR Submission', 'Liquidity Coverage Ratio', 'Net Stable Funding Ratio'],
                'Due Date': ['2024-01-31', '2024-04-05', '2024-01-15', '2024-01-31'],
                'Status': ['🟡 In Progress', '🟢 Complete', '⚠️ Overdue', '🟡 In Progress'],
                'Responsible': ['Risk Team', 'Capital Planning', 'Treasury', 'Treasury']
            }

            sub_df = pd.DataFrame(submissions)
            st.dataframe(sub_df, use_container_width=True)

        with col2:
            st.markdown("#### Regulatory Communications")

            communications = {
                'Date': ['2024-01-10', '2023-12-15', '2023-11-28', '2023-10-20'],
                'From': ['Federal Reserve', 'OCC', 'Federal Reserve', 'FDIC'],
                'Subject': ['Capital Planning Guidance', 'Stress Test Results', 'Basel III Updates', 'Risk Management'],
                'Action Required': ['Review', 'Implement', 'Comply', 'Report']
            }

            comm_df = pd.DataFrame(communications)
            st.dataframe(comm_df, use_container_width=True)

        # Capital optimization recommendations
        st.markdown("### 🚀 Capital Optimization Recommendations")

        recommendations = [
            {
                'Priority': 'High',
                'Recommendation': 'Optimize credit risk models to reduce RWA',
                'Impact': 'Potential 50-75bp CET1 improvement',
                'Timeline': '6-9 months',
                'Owner': 'Risk Modeling Team'
            },
            {
                'Priority': 'Medium',
                'Recommendation': 'Implement netting agreements for derivatives',
                'Impact': 'Potential 20-30bp leverage ratio improvement',
                'Timeline': '3-6 months',
                'Owner': 'Legal & Operations'
            },
            {
                'Priority': 'Low',
                'Recommendation': 'Review capital instrument mix',
                'Impact': 'Optimize cost of capital',
                'Timeline': '12 months',
                'Owner': 'Treasury'
            }
        ]

        rec_df = pd.DataFrame(recommendations)

        # Color code by priority
        def highlight_priority(row):
            if row['Priority'] == 'High':
                return ['background-color: #fef2f2; color: #991b1b'] * len(row)
            elif row['Priority'] == 'Medium':
                return ['background-color: #fffbeb; color: #92400e'] * len(row)
            else:
                return ['background-color: #f0fdf4; color: #166534'] * len(row)

        styled_rec_df = rec_df.style.apply(highlight_priority, axis=1)
        st.dataframe(styled_rec_df, use_container_width=True)

    def render_factor_analysis(self):
        """Render factor analysis page with heat maps"""
        st.markdown("## 🔍 Risk Factor Analysis")

        # Get risk factors data
        risk_factors = self.data_provider.get_risk_factors()

        # Risk factor overview
        st.markdown("### 🌡️ Risk Factor Heat Map")

        # Create interactive risk factor heatmap
        fig_heatmap = self.viz_engine.create_risk_factor_heatmap(risk_factors)
        st.plotly_chart(fig_heatmap, use_container_width=True)

        # Factor performance metrics
        st.markdown("### 📈 Risk Factor Performance")

        col1, col2, col3, col4 = st.columns(4)

        # Display key factor metrics
        factors_list = list(risk_factors.keys())

        with col1:
            if 'VIX' in factors_list:
                vix_data = risk_factors['VIX']
                st.metric(
                    "VIX (Volatility Index)",
                    f"{vix_data.current_value:.1f}",
                    delta=f"{vix_data.change_1d:.1f}%"
                )
            else:
                st.metric("VIX", "18.5", delta="+2.1%")

        with col2:
            if 'US_10Y_YIELD' in factors_list:
                yield_data = risk_factors['US_10Y_YIELD']
                st.metric(
                    "US 10Y Yield",
                    f"{yield_data.current_value:.2f}%",
                    delta=f"{yield_data.change_1d:.1f}bp"
                )
            else:
                st.metric("US 10Y Yield", "4.25%", delta="-5bp")

        with col3:
            if 'USD_INDEX' in factors_list:
                usd_data = risk_factors['USD_INDEX']
                st.metric(
                    "USD Index",
                    f"{usd_data.current_value:.1f}",
                    delta=f"{usd_data.change_1d:.1f}%"
                )
            else:
                st.metric("USD Index", "103.2", delta="+0.3%")

        with col4:
            if 'OIL_PRICE' in factors_list:
                oil_data = risk_factors['OIL_PRICE']
                st.metric(
                    "Oil Price (WTI)",
                    f"${oil_data.current_value:.1f}",
                    delta=f"{oil_data.change_1d:.1f}%"
                )
            else:
                st.metric("Oil Price", "$78.5", delta="-1.2%")

        # Factor correlation analysis
        st.markdown("### 🔗 Factor Correlation Matrix")

        # Generate correlation data
        factor_symbols = ['SPY', 'TLT', 'GLD', 'DXY', 'VIX']
        correlation_data = self.data_provider.get_correlation_matrix(factor_symbols)

        fig_corr = self.viz_engine.create_correlation_heatmap(correlation_data)
        st.plotly_chart(fig_corr, use_container_width=True)

        # Factor exposure analysis
        st.markdown("### 📋 Portfolio Factor Exposure")

        # Simulate factor exposures
        factor_exposures = {
            'Risk Factor': ['Equity Risk', 'Interest Rate Risk', 'Credit Risk', 'FX Risk', 'Commodity Risk', 'Volatility Risk'],
            'Exposure ($M)': [1250, 380, 220, 185, 125, 340],
            'Beta': [1.15, 0.85, 0.65, 0.45, 0.75, 1.25],
            '1% Factor Move P&L': [14.4, -3.2, -1.4, 0.8, 0.9, 4.3],
            'VaR Contribution (%)': [52.3, 15.8, 9.2, 7.7, 5.2, 14.2],
            'Hedge Ratio': ['75%', '90%', '60%', '85%', '40%', '30%']
        }

        exposure_df = pd.DataFrame(factor_exposures)

        # Style based on VaR contribution
        def highlight_var_contribution(val):
            if isinstance(val, str) and '%' in val:
                numeric_val = float(val.replace('%', ''))
                if numeric_val > 20:
                    return 'background-color: #fef2f2; color: #991b1b'
                elif numeric_val > 10:
                    return 'background-color: #fffbeb; color: #92400e'
            return ''

        styled_exposure_df = exposure_df.style.applymap(highlight_var_contribution, subset=['VaR Contribution (%)'])
        st.dataframe(styled_exposure_df, use_container_width=True)

        # Factor scenario analysis
        st.markdown("### 🎭 Factor Scenario Analysis")

        scenario_tabs = st.tabs(["Market Regime Changes", "Factor Shocks", "Historical Scenarios"])

        with scenario_tabs[0]:
            st.markdown("#### Market Regime Transition Impact")

            regime_scenarios = {
                'Current Regime': ['Risk-On', 'Risk-On', 'Risk-Off', 'Risk-Off'],
                'Target Regime': ['Risk-Off', 'High Vol', 'Risk-On', 'Low Vol'],
                'Equity Impact': ['-15%', '-25%', '+12%', '+8%'],
                'Bond Impact': ['+8%', '+12%', '-5%', '+3%'],
                'FX Impact': ['+5%', '+15%', '-8%', '-2%'],
                'Portfolio P&L': ['-$85M', '-$145M', '+$65M', '+$35M'],
                'Probability': ['25%', '10%', '30%', '15%']
            }

            regime_df = pd.DataFrame(regime_scenarios)
            st.dataframe(regime_df, use_container_width=True)

        with scenario_tabs[1]:
            st.markdown("#### Factor Shock Analysis")

            shock_scenarios = {
                'Factor': ['VIX Spike (+50%)', 'Yield Curve Steepening (+100bp)', 'USD Strength (+10%)', 'Credit Spreads (+200bp)', 'Oil Shock (+30%)'],
                'Direct Impact': ['-$35M', '-$28M', '+$18M', '-$22M', '+$8M'],
                'Correlation Impact': ['-$15M', '-$8M', '-$12M', '-$18M', '-$3M'],
                'Total Impact': ['-$50M', '-$36M', '+$6M', '-$40M', '+$5M'],
                'Hedge Effectiveness': ['65%', '85%', '90%', '70%', '45%'],
                'Recovery Time': ['1-2 weeks', '1-3 months', '2-4 weeks', '3-6 months', '1-2 weeks']
            }

            shock_df = pd.DataFrame(shock_scenarios)
            st.dataframe(shock_df, use_container_width=True)

        with scenario_tabs[2]:
            st.markdown("#### Historical Event Replication")

            historical_events = {
                'Historical Event': ['2008 Financial Crisis', '2020 COVID-19', '2016 Brexit', '2018 Vol Target Crisis', '2022 Russia-Ukraine'],
                'Duration': ['18 months', '6 months', '3 months', '2 weeks', '6 months'],
                'Max Drawdown': ['-42%', '-35%', '-12%', '-18%', '-28%'],
                'Recovery Time': ['36 months', '12 months', '8 months', '4 months', 'Ongoing'],
                'Portfolio Impact': ['-$1.2B', '-$850M', '-$280M', '-$420M', '-$650M'],
                'Key Factors': ['Credit/Liquidity', 'Flight to Quality', 'GBP/EU Risk', 'Volatility/Correlation', 'Commodities/Energy']
            }

            hist_df = pd.DataFrame(historical_events)
            st.dataframe(hist_df, use_container_width=True)

        # Factor model performance
        st.markdown("### 🏆 Factor Model Performance")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Model Validation Metrics")

            validation_metrics = {
                'Metric': ['R-squared', 'Tracking Error', 'Information Ratio', 'Active Share', 'Factor Coverage'],
                'Current Value': ['0.87', '2.8%', '1.45', '0.65', '0.92'],
                'Target': ['>0.80', '<3.5%', '>1.0', '<0.80', '>0.85'],
                'Status': ['✅ Good', '✅ Good', '✅ Good', '✅ Good', '✅ Good']
            }

            val_df = pd.DataFrame(validation_metrics)
            st.dataframe(val_df, use_container_width=True)

        with col2:
            st.markdown("#### Factor Attribution (MTD)")

            attribution = {
                'Factor': ['Selection Effect', 'Allocation Effect', 'Interaction Effect', 'Currency Effect', 'Timing Effect'],
                'Contribution (bp)': [45, -12, 8, -5, 23],
                'Weight (%)': [65, 20, 5, 5, 5],
                'Impact': ['Positive', 'Negative', 'Positive', 'Negative', 'Positive']
            }

            attr_df = pd.DataFrame(attribution)

            # Color code by impact
            def highlight_impact(row):
                if row['Impact'] == 'Positive':
                    return ['background-color: #f0fdf4; color: #166534'] * len(row)
                else:
                    return ['background-color: #fef2f2; color: #991b1b'] * len(row)

            styled_attr_df = attr_df.style.apply(highlight_impact, axis=1)
            st.dataframe(styled_attr_df, use_container_width=True)

        # Factor-based hedging recommendations
        st.markdown("### 🛡️ Factor-Based Hedging Recommendations")

        hedging_recs = {
            'Factor': ['Equity Risk', 'Interest Rate Risk', 'Credit Risk', 'FX Risk', 'Volatility Risk'],
            'Current Hedge': ['75%', '90%', '60%', '85%', '30%'],
            'Recommended Hedge': ['80%', '85%', '70%', '90%', '50%'],
            'Instrument': ['SPY Put Options', 'TLT/IEF Spread', 'CDX IG Index', 'EUR/USD Forward', 'VIX Call Options'],
            'Cost (bp)': [25, 15, 35, 10, 45],
            'Effectiveness': ['High', 'High', 'Medium', 'High', 'Medium'],
            'Priority': ['Medium', 'Low', 'High', 'Low', 'High']
        }

        hedging_df = pd.DataFrame(hedging_recs)

        # Color code by priority
        def highlight_hedge_priority(row):
            if row['Priority'] == 'High':
                return ['background-color: #fef2f2; color: #991b1b'] * len(row)
            elif row['Priority'] == 'Medium':
                return ['background-color: #fffbeb; color: #92400e'] * len(row)
            else:
                return ['background-color: #f0fdf4; color: #166534'] * len(row)

        styled_hedging_df = hedging_df.style.apply(highlight_hedge_priority, axis=1)
        st.dataframe(styled_hedging_df, use_container_width=True)

    def render_mobile_view(self):
        """Render mobile-optimized view"""
        st.markdown("## 📱 Mobile Risk Dashboard")

        # Mobile-optimized layout with single column
        # Key metrics in compact format
        st.markdown("### 📈 Quick Risk Overview")

        metrics = self.get_current_risk_metrics()

        # Compact metric cards
        st.metric("Portfolio Value", f"${metrics['portfolio_value']:,.0f}M",
                 f"{metrics['portfolio_value_change']:+.1f}%")

        st.metric("1-Day VaR", f"${metrics['var_1d']:,.1f}M",
                 f"{metrics['var_1d_change']:+.1f}%")

        st.metric("Expected Shortfall", f"${metrics['expected_shortfall']:,.1f}M",
                 f"{metrics['es_change']:+.1f}%")

        # Quick alerts
        st.markdown("### ⚠️ Alerts")
        alerts = self.get_active_alerts()
        if alerts:
            for alert in alerts[:3]:  # Show top 3
                severity_icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
                st.write(f"{severity_icon.get(alert['severity'], '⚪')} {alert['message']}")
        else:
            st.success("✅ No active alerts")

        # Simple portfolio composition chart
        st.markdown("### 📋 Portfolio Breakdown")
        composition_data = {
            'Asset Class': ['Equities', 'Bonds', 'Alternatives', 'Cash'],
            'Percentage': [45, 30, 20, 5]
        }
        fig = px.pie(pd.DataFrame(composition_data), values='Percentage', names='Asset Class')
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

        # Quick actions
        st.markdown("### ⚡ Quick Actions")

        if st.button("🔄 Refresh Data", use_container_width=True):
            st.success("Data refreshed!")

        if st.button("📧 Send Alert Summary", use_container_width=True):
            st.success("Alert summary sent!")

        if st.button("📄 Generate Mobile Report", use_container_width=True):
            st.success("Mobile report generated!")

    def render_settings(self):
        """Render settings and configuration page"""
        st.markdown("## ⚙️ Dashboard Settings")

        # User preferences
        st.markdown("### 👤 User Preferences")

        col1, col2 = st.columns(2)

        with col1:
            # Display settings
            st.markdown("#### Display Settings")

            theme = st.selectbox("Dashboard Theme",
                               ["Professional Blue", "Dark Mode", "Light Mode", "High Contrast"])

            currency = st.selectbox("Base Currency", ["USD", "EUR", "GBP", "JPY"])

            timezone = st.selectbox("Timezone", ["UTC", "EST", "PST", "GMT", "CET"])

            decimal_places = st.slider("Decimal Places", 0, 4, 2)

        with col2:
            # Notification settings
            st.markdown("#### Notification Settings")

            email_alerts = st.checkbox("Email Alerts", value=True)

            alert_frequency = st.selectbox("Alert Frequency",
                                         ["Immediate", "Every 15 minutes", "Hourly", "Daily"])

            sound_alerts = st.checkbox("Sound Notifications", value=False)

            mobile_push = st.checkbox("Mobile Push Notifications", value=True)

        # Risk settings
        st.markdown("### ⚠️ Risk Settings")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### VaR Configuration")

            var_confidence = st.selectbox("VaR Confidence Level", ["95%", "99%", "99.9%"], index=0)

            var_horizon = st.selectbox("VaR Time Horizon", ["1 Day", "5 Days", "10 Days"], index=0)

            var_method = st.selectbox("Primary VaR Method",
                                    ["Historical Simulation", "Parametric", "Monte Carlo"], index=0)

        with col2:
            st.markdown("#### Alert Thresholds")

            var_warning = st.number_input("VaR Warning Threshold ($M)", value=20.0, step=0.5)

            var_breach = st.number_input("VaR Breach Threshold ($M)", value=25.0, step=0.5)

            concentration_limit = st.number_input("Concentration Limit (%)", value=10.0, step=0.5)

        # Data settings
        st.markdown("### 📊 Data Settings")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Data Sources")

            primary_data_source = st.selectbox("Primary Data Source",
                                              ["Yahoo Finance", "Bloomberg", "Refinitiv", "Alpha Vantage"])

            backup_data_source = st.selectbox("Backup Data Source",
                                             ["Yahoo Finance", "Bloomberg", "Refinitiv", "Alpha Vantage"], index=1)

            refresh_interval = st.slider("Auto Refresh Interval (seconds)", 5, 300, 30)

        with col2:
            st.markdown("#### Data Quality")

            data_validation = st.checkbox("Enable Data Validation", value=True)

            outlier_detection = st.checkbox("Outlier Detection", value=True)

            data_backup = st.checkbox("Daily Data Backup", value=True)

        # Export settings
        st.markdown("### 📎 Export Settings")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Report Configuration")

            default_format = st.selectbox("Default Export Format", ["PDF", "Excel", "CSV"], index=0)

            include_charts = st.checkbox("Include Charts in Reports", value=True)

            watermark = st.checkbox("Add Confidential Watermark", value=True)

        with col2:
            st.markdown("#### Scheduled Reports")

            daily_report = st.checkbox("Daily Risk Report", value=True)

            weekly_summary = st.checkbox("Weekly Risk Summary", value=True)

            monthly_detailed = st.checkbox("Monthly Detailed Report", value=False)

        # Advanced settings (for administrators)
        if st.session_state.get('user_role') == 'Administrator':
            st.markdown("### 🔧 Advanced Settings")

            with st.expander("System Configuration"):
                st.markdown("#### Database Settings")

                db_host = st.text_input("Database Host", value="localhost")
                db_port = st.number_input("Database Port", value=5432)
                connection_pool = st.slider("Connection Pool Size", 5, 50, 20)

                st.markdown("#### Performance Settings")

                cache_size = st.slider("Cache Size (MB)", 100, 1000, 500)
                max_workers = st.slider("Max Worker Threads", 1, 16, 4)

            with st.expander("Security Settings"):
                st.markdown("#### Authentication")

                session_timeout = st.slider("Session Timeout (hours)", 1, 24, 8)
                max_login_attempts = st.slider("Max Login Attempts", 3, 10, 5)
                password_expiry = st.slider("Password Expiry (days)", 30, 365, 90)

                two_factor = st.checkbox("Require Two-Factor Authentication", value=False)

        # Save settings
        st.markdown("---")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("💾 Save Settings", use_container_width=True):
                # Update session state
                st.session_state.dashboard_config.update({
                    'currency': currency,
                    'time_zone': timezone,
                    'auto_refresh': True,
                    'refresh_interval': refresh_interval,
                    'default_var_confidence': float(var_confidence.replace('%', '')) / 100,
                    'charts_per_row': 2
                })
                st.success("✅ Settings saved successfully!")

        with col2:
            if st.button("🔄 Reset to Defaults", use_container_width=True):
                st.session_state.dashboard_config = self.get_default_config()
                st.info("🔄 Settings reset to defaults")

        with col3:
            if st.button("📥 Export Settings", use_container_width=True):
                config_json = json.dumps(st.session_state.dashboard_config, indent=2)
                st.download_button(
                    label="Download Config",
                    data=config_json,
                    file_name="dashboard_config.json",
                    mime="application/json"
                )

    def generate_risk_report(self):
        """Generate comprehensive risk report"""
        try:
            with st.spinner("Generating risk report..."):
                # Get sample data
                sample_portfolio = self.risk_engine.generate_sample_portfolio()
                risk_metrics = self.risk_engine.calculate_risk_metrics_summary(sample_portfolio)

                # Generate PDF report
                portfolio_data = {'name': 'Total Portfolio', 'total_value': 2500}
                metrics_dict = {
                    'var_1d': risk_metrics.var_1d,
                    'expected_shortfall': risk_metrics.expected_shortfall,
                    'sharpe_ratio': risk_metrics.sharpe_ratio,
                    'max_drawdown': risk_metrics.max_drawdown
                }

                pdf_data = self.export_manager.generate_executive_summary_pdf(portfolio_data, metrics_dict)

                if pdf_data:
                    st.download_button(
                        label="📄 Download Risk Report",
                        data=pdf_data,
                        file_name=f"risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf"
                    )
                    st.success("✅ Risk report generated successfully!")
                else:
                    st.error("❌ Failed to generate risk report")
        except Exception as e:
            st.error(f"Error generating report: {e}")

    def send_risk_alerts(self):
        """Send risk alerts to stakeholders"""
        try:
            # Simulate sending alerts
            active_alerts = self.alert_manager.get_active_alerts()

            if active_alerts:
                # Generate demo alerts for testing
                self.alert_manager.simulate_demo_alerts()

                # Send email alerts (would be real in production)
                for alert in active_alerts[:3]:  # Send top 3 alerts
                    if st.session_state.get('user_role') in ['Administrator', 'Risk Manager']:
                        # Simulate email sending
                        st.success(f"📧 Alert sent: {alert.title}")

                st.success(f"✅ {len(active_alerts)} risk alerts sent to stakeholders")
            else:
                st.info("🔵 No active alerts to send")

        except Exception as e:
            st.error(f"Error sending alerts: {e}")

    def export_portfolio_data(self):
        """Export portfolio data"""
        try:
            with st.spinner("Exporting portfolio data..."):
                # Get sample portfolio data
                sample_portfolio = self.risk_engine.generate_sample_portfolio()

                # Convert to list of dictionaries
                portfolio_data = []
                for pos in sample_portfolio:
                    portfolio_data.append({
                        'symbol': pos.symbol,
                        'quantity': pos.quantity,
                        'market_value': pos.market_value,
                        'asset_class': pos.asset_class,
                        'sector': pos.sector,
                        'region': pos.region,
                        'currency': pos.currency,
                        'delta': pos.delta,
                        'gamma': pos.gamma,
                        'theta': pos.theta,
                        'vega': pos.vega,
                        'rho': pos.rho
                    })

                # Get risk metrics
                risk_metrics = self.risk_engine.calculate_risk_metrics_summary(sample_portfolio)
                metrics_dict = {
                    'var_1d': risk_metrics.var_1d,
                    'var_10d': risk_metrics.var_10d,
                    'expected_shortfall': risk_metrics.expected_shortfall,
                    'volatility': risk_metrics.volatility,
                    'beta': risk_metrics.beta,
                    'sharpe_ratio': risk_metrics.sharpe_ratio,
                    'max_drawdown': risk_metrics.max_drawdown,
                    'tracking_error': risk_metrics.tracking_error,
                    'information_ratio': risk_metrics.information_ratio
                }

                # Generate Excel export
                excel_data = self.export_manager.export_portfolio_to_excel(portfolio_data, metrics_dict)

                if excel_data:
                    st.download_button(
                        label="📈 Download Portfolio Data (Excel)",
                        data=excel_data,
                        file_name=f"portfolio_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    st.success("✅ Portfolio data exported successfully!")
                else:
                    st.error("❌ Failed to export portfolio data")

        except Exception as e:
            st.error(f"Error exporting data: {e}")

    def render_regulatory_compliance(self):
        """Render regulatory compliance dashboard"""
        st.header("⚖️ Regulatory Compliance")
        st.markdown("### Real-time regulatory compliance monitoring and validation")

        # Import here to avoid circular imports
        from src.regulatory_compliance import generate_sample_portfolio_data

        try:
            # Get sample portfolio data for compliance testing
            portfolio_data = generate_sample_portfolio_data()

            # Create tabs for different compliance frameworks
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🏛️ Basel III", "📈 FRTB", "📋 Reports"])

            with tab1:
                st.subheader("Compliance Overview")

                # Run full compliance check
                with st.spinner("Running compliance checks..."):
                    compliance_results = self.compliance_engine.run_full_compliance_check(portfolio_data)

                # Display summary metrics
                if compliance_results.get('summary'):
                    col1, col2, col3, col4 = st.columns(4)

                    for i, summary_item in enumerate(compliance_results['summary'][:4]):
                        col = [col1, col2, col3, col4][i]

                        with col:
                            if summary_item['format'] == 'percentage':
                                value_display = f"{summary_item['value']*100:.1f}%"
                                color = "green" if summary_item['status'] == 'good' else "orange" if summary_item['status'] == 'warning' else "red"
                            else:
                                value_display = str(summary_item['value'])
                                color = "green" if summary_item['status'] == 'good' else "red"

                            st.metric(
                                label=summary_item['metric'],
                                value=value_display
                            )
                            st.markdown(f"<div style='color: {color}'>● {summary_item['status'].title()}</div>",
                                      unsafe_allow_html=True)

                # Compliance status visualization
                st.subheader("Compliance Status by Framework")

                all_results = compliance_results.get('basel_iii', []) + compliance_results.get('frtb', [])
                if all_results:
                    # Create compliance status dataframe
                    status_data = []
                    for result in all_results:
                        status_data.append({
                            'Framework': result.framework.value,
                            'Rule ID': result.rule_id,
                            'Description': result.rule_description,
                            'Status': result.status.value,
                            'Value': result.value,
                            'Threshold': result.threshold,
                            'Deviation': result.deviation
                        })

                    df_status = pd.DataFrame(status_data)

                    # Color code by status
                    def color_status(val):
                        if val == 'Compliant':
                            return 'background-color: #d4edda; color: #155724'
                        elif val == 'Non-Compliant':
                            return 'background-color: #f8d7da; color: #721c24'
                        else:
                            return 'background-color: #fff3cd; color: #856404'

                    styled_df = df_status.style.applymap(color_status, subset=['Status'])
                    st.dataframe(styled_df, use_container_width=True)

            with tab2:
                st.subheader("Basel III Compliance")

                basel_results = compliance_results.get('basel_iii', [])
                if basel_results:
                    # Capital adequacy metrics
                    st.markdown("#### Capital Adequacy")

                    col1, col2 = st.columns(2)

                    # Find Tier 1 and Total Capital ratios
                    tier1_result = next((r for r in basel_results if r.rule_id == 'CAR_T1'), None)
                    total_result = next((r for r in basel_results if r.rule_id == 'CAR_TOTAL'), None)

                    if tier1_result:
                        with col1:
                            st.metric(
                                "Tier 1 Capital Ratio",
                                f"{tier1_result.value*100:.2f}%",
                                f"{tier1_result.deviation*100:.2f}%" if tier1_result.deviation != 0 else None
                            )
                            st.markdown(f"**Threshold:** {tier1_result.threshold*100:.1f}%")
                            st.markdown(f"**Status:** {tier1_result.status.value}")

                    if total_result:
                        with col2:
                            st.metric(
                                "Total Capital Ratio",
                                f"{total_result.value*100:.2f}%",
                                f"{total_result.deviation*100:.2f}%" if total_result.deviation != 0 else None
                            )
                            st.markdown(f"**Threshold:** {total_result.threshold*100:.1f}%")
                            st.markdown(f"**Status:** {total_result.status.value}")

                    # Leverage ratio
                    leverage_result = next((r for r in basel_results if r.rule_id == 'LR'), None)
                    if leverage_result:
                        st.markdown("#### Leverage Ratio")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(
                                "Leverage Ratio",
                                f"{leverage_result.value*100:.2f}%",
                                f"{leverage_result.deviation*100:.2f}%" if leverage_result.deviation != 0 else None
                            )
                        with col2:
                            st.markdown(f"**Threshold:** {leverage_result.threshold*100:.1f}%")
                            st.markdown(f"**Status:** {leverage_result.status.value}")

                    # Liquidity ratios
                    lcr_result = next((r for r in basel_results if r.rule_id == 'LCR'), None)
                    nsfr_result = next((r for r in basel_results if r.rule_id == 'NSFR'), None)

                    if lcr_result or nsfr_result:
                        st.markdown("#### Liquidity Ratios")
                        col1, col2 = st.columns(2)

                        if lcr_result:
                            with col1:
                                st.metric(
                                    "Liquidity Coverage Ratio (LCR)",
                                    f"{lcr_result.value*100:.1f}%",
                                    f"{lcr_result.deviation*100:.1f}%" if lcr_result.deviation != 0 else None
                                )
                                st.markdown(f"**Status:** {lcr_result.status.value}")

                        if nsfr_result:
                            with col2:
                                st.metric(
                                    "Net Stable Funding Ratio (NSFR)",
                                    f"{nsfr_result.value*100:.1f}%",
                                    f"{nsfr_result.deviation*100:.1f}%" if nsfr_result.deviation != 0 else None
                                )
                                st.markdown(f"**Status:** {nsfr_result.status.value}")

            with tab3:
                st.subheader("FRTB Compliance")

                frtb_results = compliance_results.get('frtb', [])
                if frtb_results:
                    # Sensitivities-based approach
                    sba_result = next((r for r in frtb_results if r.rule_id == 'SBA_TOTAL'), None)
                    if sba_result:
                        st.markdown("#### Sensitivities-Based Approach (SBA)")

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric("Delta Charge", f"${sba_result.details.get('delta_charge', 0):,.0f}")
                        with col2:
                            st.metric("Vega Charge", f"${sba_result.details.get('vega_charge', 0):,.0f}")
                        with col3:
                            st.metric("Curvature Charge", f"${sba_result.details.get('curvature_charge', 0):,.0f}")

                        st.metric("Total SBA Capital Charge", f"${sba_result.value:,.0f}")

                    # Internal models approach
                    es_result = next((r for r in frtb_results if r.rule_id == 'IMA_ES'), None)
                    pla_result = next((r for r in frtb_results if r.rule_id == 'IMA_PLA'), None)

                    if es_result or pla_result:
                        st.markdown("#### Internal Models Approach (IMA)")

                        col1, col2 = st.columns(2)

                        if es_result:
                            with col1:
                                st.metric("ES Capital Charge", f"${es_result.value:,.0f}")
                                st.markdown(f"**ES Multiplier:** {es_result.details.get('es_multiplier', 3.0)}")

                        if pla_result:
                            with col2:
                                st.metric("P&L Attribution Test", f"{pla_result.value*100:.1f}%")
                                status_color = "green" if pla_result.status.value == 'Compliant' else "red"
                                st.markdown(f"<div style='color: {status_color}'>**Status:** {pla_result.status.value}</div>",
                                          unsafe_allow_html=True)

            with tab4:
                st.subheader("Compliance Reports")

                # Generate and display compliance report
                if st.button("Generate Compliance Report"):
                    with st.spinner("Generating comprehensive compliance report..."):
                        report_text = self.compliance_engine.generate_compliance_report(compliance_results)

                        st.text_area("Compliance Report", report_text, height=400)

                        # Download button for report
                        st.download_button(
                            label="Download Report",
                            data=report_text,
                            file_name=f"compliance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )

                # Historical compliance data
                st.markdown("#### Historical Compliance Trends")

                # Get compliance history
                history_df = self.compliance_engine.get_compliance_history(days=30)

                if not history_df.empty:
                    # Create compliance trend chart
                    fig = go.Figure()

                    # Group by rule_id and plot trends
                    for rule_id in history_df['rule_id'].unique()[:5]:  # Show top 5 rules
                        rule_data = history_df[history_df['rule_id'] == rule_id]
                        fig.add_trace(go.Scatter(
                            x=rule_data['timestamp'],
                            y=rule_data['value'],
                            mode='lines+markers',
                            name=rule_id,
                            line=dict(width=2),
                            marker=dict(size=6)
                        ))

                    fig.update_layout(
                        title="Compliance Metrics Trends (Last 30 Days)",
                        xaxis_title="Date",
                        yaxis_title="Metric Value",
                        height=400,
                        showlegend=True,
                        hovermode='x unified'
                    )

                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No historical compliance data available. Run compliance checks regularly to build history.")

        except Exception as e:
            st.error(f"Error in regulatory compliance dashboard: {str(e)}")
            st.exception(e)

# Initialize and run dashboard
if __name__ == "__main__":
    dashboard = RiskDashboard()
    dashboard.run()