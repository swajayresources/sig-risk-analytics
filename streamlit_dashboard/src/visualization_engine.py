"""
Visualization Engine Module
============================

Advanced visualization engine for the professional risk management dashboard.
Provides 3D plots, interactive charts, heat maps, and real-time streaming visualizations.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

class VisualizationEngine:
    """Advanced visualization engine for risk analytics"""

    def __init__(self):
        """Initialize visualization engine"""
        self.color_schemes = {
            'risk': ['#10b981', '#f59e0b', '#ef4444', '#dc2626'],  # Green to Red
            'performance': ['#3b82f6', '#1e40af', '#1e3a8a'],      # Blues
            'correlation': 'RdBu_r',                                # Red-Blue
            'heatmap': 'viridis',                                   # Viridis
            'diverging': 'RdYlBu'                                   # Red-Yellow-Blue
        }

        self.chart_configs = {
            'default_height': 400,
            'default_width': 800,
            'font_family': 'Arial, sans-serif',
            'title_size': 16,
            'axis_size': 12,
            'legend_size': 10
        }

    def create_3d_risk_surface(self, portfolio_data: List[Dict]) -> go.Figure:
        """Create 3D portfolio risk surface plot"""
        try:
            # Prepare data for 3D surface
            sectors = list(set([pos.get('sector', 'Other') for pos in portfolio_data]))
            regions = list(set([pos.get('region', 'Unknown') for pos in portfolio_data]))

            # Create meshgrid for surface
            x = np.linspace(0, len(sectors)-1, len(sectors))
            y = np.linspace(0, len(regions)-1, len(regions))
            X, Y = np.meshgrid(x, y)

            # Calculate risk surface (VaR contribution by sector/region)
            Z = np.zeros((len(regions), len(sectors)))

            for i, region in enumerate(regions):
                for j, sector in enumerate(sectors):
                    # Calculate risk for this sector/region combination
                    positions = [p for p in portfolio_data
                               if p.get('sector') == sector and p.get('region') == region]
                    total_value = sum([p.get('market_value', 0) for p in positions])
                    # Simple risk approximation
                    Z[i, j] = total_value * 0.02  # 2% risk factor

            # Create 3D surface plot
            fig = go.Figure(data=[
                go.Surface(
                    x=sectors,
                    y=regions,
                    z=Z,
                    colorscale='Viridis',
                    opacity=0.8,
                    contours={
                        "x": {"show": True, "start": 0, "end": len(sectors), "size": 1},
                        "y": {"show": True, "start": 0, "end": len(regions), "size": 1},
                        "z": {"show": True, "start": Z.min(), "end": Z.max(), "size": Z.max()/10}
                    }
                )
            ])

            fig.update_layout(
                title={
                    'text': '3D Portfolio Risk Surface by Sector and Region',
                    'x': 0.5,
                    'font': {'size': 16}
                },
                scene=dict(
                    xaxis_title='Sectors',
                    yaxis_title='Regions',
                    zaxis_title='Risk Contribution ($M)',
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                ),
                width=800,
                height=600,
                font=dict(family=self.chart_configs['font_family'])
            )

            return fig

        except Exception as e:
            st.error(f"Error creating 3D risk surface: {e}")
            return go.Figure()

    def create_correlation_heatmap(self, correlation_data: pd.DataFrame) -> go.Figure:
        """Create interactive correlation matrix heatmap"""
        try:
            fig = go.Figure(data=go.Heatmap(
                z=correlation_data.values,
                x=correlation_data.columns,
                y=correlation_data.index,
                colorscale='RdBu',
                zmid=0,
                text=np.round(correlation_data.values, 2),
                texttemplate="%{text}",
                textfont={"size": 10},
                hoverongaps=False,
                hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
            ))

            fig.update_layout(
                title={
                    'text': 'Asset Correlation Matrix Heat Map',
                    'x': 0.5,
                    'font': {'size': 16}
                },
                xaxis_title='Assets',
                yaxis_title='Assets',
                width=700,
                height=600,
                font=dict(family=self.chart_configs['font_family'])
            )

            return fig

        except Exception as e:
            st.error(f"Error creating correlation heatmap: {e}")
            return go.Figure()

    def create_monte_carlo_distribution(self, simulation_results: np.ndarray,
                                      var_level: float = 0.95) -> go.Figure:
        """Create Monte Carlo simulation results distribution"""
        try:
            # Calculate VaR and ES
            var_threshold = np.percentile(simulation_results, (1 - var_level) * 100)
            es_threshold = np.mean(simulation_results[simulation_results <= var_threshold])

            # Create histogram
            fig = go.Figure()

            # Main distribution
            fig.add_trace(go.Histogram(
                x=simulation_results,
                nbinsx=50,
                name='P&L Distribution',
                opacity=0.7,
                marker_color='lightblue'
            ))

            # VaR line
            fig.add_vline(
                x=var_threshold,
                line_dash="dash",
                line_color="red",
                annotation_text=f"VaR ({var_level:.0%}): ${var_threshold:.1f}M",
                annotation_position="top"
            )

            # Expected Shortfall line
            fig.add_vline(
                x=es_threshold,
                line_dash="dot",
                line_color="darkred",
                annotation_text=f"Expected Shortfall: ${es_threshold:.1f}M",
                annotation_position="bottom"
            )

            # Tail area
            tail_data = simulation_results[simulation_results <= var_threshold]
            if len(tail_data) > 0:
                fig.add_trace(go.Histogram(
                    x=tail_data,
                    nbinsx=20,
                    name='Tail Risk',
                    opacity=0.8,
                    marker_color='red'
                ))

            fig.update_layout(
                title={
                    'text': 'Monte Carlo Simulation - P&L Distribution',
                    'x': 0.5,
                    'font': {'size': 16}
                },
                xaxis_title='P&L ($M)',
                yaxis_title='Frequency',
                showlegend=True,
                width=800,
                height=500,
                font=dict(family=self.chart_configs['font_family'])
            )

            return fig

        except Exception as e:
            st.error(f"Error creating Monte Carlo distribution: {e}")
            return go.Figure()

    def create_risk_waterfall_chart(self, risk_contributions: Dict[str, float]) -> go.Figure:
        """Create risk contribution waterfall chart"""
        try:
            categories = list(risk_contributions.keys())
            values = list(risk_contributions.values())

            # Calculate cumulative values for waterfall
            cumulative = [0]
            for val in values[:-1]:
                cumulative.append(cumulative[-1] + val)

            fig = go.Figure(go.Waterfall(
                name="Risk Contribution",
                orientation="v",
                measure=["relative"] * (len(categories) - 1) + ["total"],
                x=categories,
                textposition="outside",
                text=[f"${v:.1f}M" for v in values],
                y=values,
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                increasing={"marker": {"color": "#ef4444"}},
                decreasing={"marker": {"color": "#10b981"}},
                totals={"marker": {"color": "#3b82f6"}}
            ))

            fig.update_layout(
                title={
                    'text': 'Risk Contribution Waterfall',
                    'x': 0.5,
                    'font': {'size': 16}
                },
                xaxis_title='Risk Factors',
                yaxis_title='VaR Contribution ($M)',
                showlegend=False,
                width=800,
                height=500,
                font=dict(family=self.chart_configs['font_family'])
            )

            return fig

        except Exception as e:
            st.error(f"Error creating waterfall chart: {e}")
            return go.Figure()

    def create_scenario_analysis_chart(self, scenario_results: Dict[str, float]) -> go.Figure:
        """Create scenario analysis impact chart"""
        try:
            scenarios = list(scenario_results.keys())
            impacts = list(scenario_results.values())

            # Color code by impact severity
            colors = []
            for impact in impacts:
                if impact < -50:
                    colors.append('#dc2626')  # Dark red - severe
                elif impact < -20:
                    colors.append('#ef4444')  # Red - high
                elif impact < -5:
                    colors.append('#f59e0b')  # Orange - medium
                else:
                    colors.append('#10b981')  # Green - low

            fig = go.Figure(data=[
                go.Bar(
                    x=scenarios,
                    y=impacts,
                    marker_color=colors,
                    text=[f"${impact:.1f}M" for impact in impacts],
                    textposition='auto',
                    hovertemplate='%{x}<br>Impact: %{y:.1f}M<extra></extra>'
                )
            ])

            fig.update_layout(
                title={
                    'text': 'Stress Test Scenario Analysis',
                    'x': 0.5,
                    'font': {'size': 16}
                },
                xaxis_title='Stress Scenarios',
                yaxis_title='Portfolio Impact ($M)',
                xaxis_tickangle=-45,
                width=800,
                height=500,
                font=dict(family=self.chart_configs['font_family'])
            )

            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)

            return fig

        except Exception as e:
            st.error(f"Error creating scenario analysis chart: {e}")
            return go.Figure()

    def create_time_series_risk_evolution(self, historical_data: pd.DataFrame) -> go.Figure:
        """Create time series risk metric evolution with zoom/pan"""
        try:
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('VaR Evolution', 'Portfolio Value'),
                vertical_spacing=0.1,
                shared_xaxes=True
            )

            # VaR evolution
            fig.add_trace(
                go.Scatter(
                    x=historical_data['timestamp'],
                    y=historical_data['var_1d'],
                    mode='lines+markers',
                    name='1-Day VaR',
                    line=dict(color='#ef4444', width=2),
                    marker=dict(size=4)
                ),
                row=1, col=1
            )

            # VaR limit line
            fig.add_hline(
                y=25.0,
                line_dash="dash",
                line_color="red",
                annotation_text="VaR Limit",
                row=1
            )

            # Portfolio value
            fig.add_trace(
                go.Scatter(
                    x=historical_data['timestamp'],
                    y=historical_data['total_value'],
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color='#3b82f6', width=2),
                    fill='tonexty'
                ),
                row=2, col=1
            )

            fig.update_layout(
                title={
                    'text': 'Historical Risk Metrics Evolution',
                    'x': 0.5,
                    'font': {'size': 16}
                },
                height=600,
                showlegend=True,
                hovermode='x unified',
                font=dict(family=self.chart_configs['font_family'])
            )

            # Update axes
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="VaR ($M)", row=1, col=1)
            fig.update_yaxes(title_text="Portfolio Value ($M)", row=2, col=1)

            # Add range selector
            fig.update_layout(
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=7, label="7d", step="day", stepmode="backward"),
                            dict(count=30, label="30d", step="day", stepmode="backward"),
                            dict(count=90, label="3m", step="day", stepmode="backward"),
                            dict(step="all")
                        ])
                    ),
                    rangeslider=dict(visible=True),
                    type="date"
                )
            )

            return fig

        except Exception as e:
            st.error(f"Error creating time series chart: {e}")
            return go.Figure()

    def create_geographic_risk_map(self, geographic_data: Dict[str, float]) -> go.Figure:
        """Create geographic risk concentration using choropleth maps"""
        try:
            # Map country names to ISO codes
            country_mapping = {
                'United States': 'USA',
                'United Kingdom': 'GBR',
                'Germany': 'DEU',
                'France': 'FRA',
                'Japan': 'JPN',
                'China': 'CHN',
                'Canada': 'CAN',
                'Australia': 'AUS',
                'Switzerland': 'CHE',
                'Netherlands': 'NLD'
            }

            countries = []
            risk_values = []

            for country, risk in geographic_data.items():
                if country in country_mapping:
                    countries.append(country_mapping[country])
                    risk_values.append(risk)

            fig = go.Figure(data=go.Choropleth(
                locations=countries,
                z=risk_values,
                text=[f"{country}: ${risk:.1f}M" for country, risk in zip(countries, risk_values)],
                colorscale='Reds',
                reversescale=False,
                marker_line_color='darkgray',
                marker_line_width=0.5,
                colorbar_title="Risk Exposure ($M)",
                hovertemplate='%{text}<extra></extra>'
            ))

            fig.update_layout(
                title={
                    'text': 'Geographic Risk Concentration',
                    'x': 0.5,
                    'font': {'size': 16}
                },
                geo=dict(
                    showframe=False,
                    showcoastlines=True,
                    projection_type='equirectangular'
                ),
                width=900,
                height=500,
                font=dict(family=self.chart_configs['font_family'])
            )

            return fig

        except Exception as e:
            st.error(f"Error creating geographic risk map: {e}")
            return go.Figure()

    def create_real_time_pnl_stream(self, streaming_data: pd.DataFrame) -> go.Figure:
        """Create real-time P&L chart with streaming updates"""
        try:
            fig = go.Figure()

            # P&L line
            fig.add_trace(go.Scatter(
                x=streaming_data.index,
                y=streaming_data['pnl'],
                mode='lines',
                name='Real-time P&L',
                line=dict(color='#3b82f6', width=2),
                fill='tonexty'
            ))

            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)

            fig.update_layout(
                title={
                    'text': 'Real-time P&L Evolution',
                    'x': 0.5,
                    'font': {'size': 16}
                },
                xaxis_title='Time',
                yaxis_title='P&L ($M)',
                hovermode='x',
                width=800,
                height=400,
                font=dict(family=self.chart_configs['font_family'])
            )

            return fig

        except Exception as e:
            st.error(f"Error creating real-time P&L chart: {e}")
            return go.Figure()

    def create_greeks_ladder_chart(self, greeks_data: Dict[str, Dict[str, float]]) -> go.Figure:
        """Create Greeks ladder visualization"""
        try:
            # Prepare data for grouped bar chart
            greeks = ['delta', 'gamma', 'theta', 'vega', 'rho']
            categories = list(greeks_data.keys())

            fig = go.Figure()

            colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6']

            for i, greek in enumerate(greeks):
                values = [greeks_data[cat].get(greek, 0) for cat in categories]
                fig.add_trace(go.Bar(
                    name=greek.capitalize(),
                    x=categories,
                    y=values,
                    marker_color=colors[i],
                    text=[f"{val:.0f}" for val in values],
                    textposition='auto'
                ))

            fig.update_layout(
                title={
                    'text': 'Portfolio Greeks by Category',
                    'x': 0.5,
                    'font': {'size': 16}
                },
                xaxis_title='Portfolio Categories',
                yaxis_title='Greeks Value',
                barmode='group',
                width=800,
                height=500,
                font=dict(family=self.chart_configs['font_family'])
            )

            return fig

        except Exception as e:
            st.error(f"Error creating Greeks ladder chart: {e}")
            return go.Figure()

    def create_risk_factor_heatmap(self, risk_factors: Dict[str, Dict[str, float]]) -> go.Figure:
        """Create interactive risk factor heat map"""
        try:
            # Prepare data for heatmap
            factors = list(risk_factors.keys())
            metrics = ['current_value', 'change_1d', 'change_1w', 'change_1m', 'volatility']
            metric_labels = ['Current', '1D Change', '1W Change', '1M Change', 'Volatility']

            z_data = []
            for metric in metrics:
                row = [risk_factors[factor].get(metric, 0) for factor in factors]
                z_data.append(row)

            # Normalize data for better visualization
            z_normalized = []
            for i, row in enumerate(z_data):
                if i == 0:  # Current value - use as is
                    z_normalized.append(row)
                else:  # Changes and volatility - normalize
                    max_val = max(abs(min(row)), abs(max(row)))
                    if max_val > 0:
                        z_normalized.append([val / max_val for val in row])
                    else:
                        z_normalized.append(row)

            fig = go.Figure(data=go.Heatmap(
                z=z_normalized,
                x=factors,
                y=metric_labels,
                colorscale='RdYlBu',
                zmid=0,
                hoverongaps=False,
                hovertemplate='%{x}<br>%{y}: %{z:.2f}<extra></extra>'
            ))

            fig.update_layout(
                title={
                    'text': 'Risk Factor Heat Map',
                    'x': 0.5,
                    'font': {'size': 16}
                },
                xaxis_title='Risk Factors',
                yaxis_title='Metrics',
                width=800,
                height=400,
                font=dict(family=self.chart_configs['font_family'])
            )

            return fig

        except Exception as e:
            st.error(f"Error creating risk factor heatmap: {e}")
            return go.Figure()

    def create_var_breakdown_chart(self, var_breakdown: Dict[str, Dict[str, float]]) -> go.Figure:
        """Create VaR breakdown by asset class and region"""
        try:
            # Create subplots for asset class and region breakdown
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('VaR by Asset Class', 'VaR by Region'),
                specs=[[{"type": "pie"}, {"type": "pie"}]]
            )

            # Asset class breakdown
            asset_classes = list(var_breakdown.get('by_asset_class', {}).keys())
            asset_values = [var_breakdown['by_asset_class'][ac].get('total_var', 0)
                          for ac in asset_classes]

            fig.add_trace(go.Pie(
                labels=asset_classes,
                values=asset_values,
                name="Asset Class",
                marker_colors=px.colors.qualitative.Set3
            ), 1, 1)

            # Region breakdown
            regions = list(var_breakdown.get('by_region', {}).keys())
            region_values = [var_breakdown['by_region'][region].get('total_var', 0)
                           for region in regions]

            fig.add_trace(go.Pie(
                labels=regions,
                values=region_values,
                name="Region",
                marker_colors=px.colors.qualitative.Set2
            ), 1, 2)

            fig.update_layout(
                title={
                    'text': 'VaR Breakdown Analysis',
                    'x': 0.5,
                    'font': {'size': 16}
                },
                width=900,
                height=500,
                font=dict(family=self.chart_configs['font_family'])
            )

            return fig

        except Exception as e:
            st.error(f"Error creating VaR breakdown chart: {e}")
            return go.Figure()

    def render_advanced_charts_interface(self):
        """Render interface for advanced chart configurations"""
        st.markdown("### 🎨 Chart Customization")

        col1, col2, col3 = st.columns(3)

        with col1:
            chart_theme = st.selectbox(
                "Chart Theme",
                ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn"]
            )

        with col2:
            color_scheme = st.selectbox(
                "Color Scheme",
                ["Default", "Risk", "Performance", "Correlation"]
            )

        with col3:
            chart_height = st.slider("Chart Height", 300, 800, 500)

        return {
            'theme': chart_theme,
            'color_scheme': color_scheme.lower(),
            'height': chart_height
        }