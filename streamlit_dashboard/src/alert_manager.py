"""
Alert Manager Module
====================

Handles risk alerts, notifications, and limit monitoring for the risk management dashboard.
Provides real-time alert generation, email notifications, and alert history tracking.
"""

import streamlit as st
import pandas as pd
import numpy as np
import smtplib
import json
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import sqlite3
import warnings
warnings.filterwarnings('ignore')

class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class AlertType(Enum):
    """Alert types"""
    VAR_BREACH = "VAR_BREACH"
    LIMIT_BREACH = "LIMIT_BREACH"
    CONCENTRATION_RISK = "CONCENTRATION_RISK"
    LIQUIDITY_RISK = "LIQUIDITY_RISK"
    MODEL_VALIDATION = "MODEL_VALIDATION"
    MARKET_STRESS = "MARKET_STRESS"
    OPERATIONAL_RISK = "OPERATIONAL_RISK"
    REGULATORY_BREACH = "REGULATORY_BREACH"

@dataclass
class RiskAlert:
    """Risk alert data structure"""
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    metric_name: str
    current_value: float
    threshold_value: float
    breach_percentage: float
    portfolio_name: str
    timestamp: datetime
    acknowledged: bool = False
    resolved: bool = False
    assigned_to: str = ""
    resolution_notes: str = ""

@dataclass
class RiskLimit:
    """Risk limit configuration"""
    limit_id: str
    limit_name: str
    metric_type: str
    portfolio_name: str
    warning_threshold: float
    breach_threshold: float
    critical_threshold: float
    currency: str
    active: bool = True
    description: str = ""

class AlertManager:
    """Risk alert and notification management system"""

    def __init__(self):
        """Initialize alert manager"""
        self.active_alerts = []
        self.alert_history = []
        self.risk_limits = self._load_default_risk_limits()
        self.notification_settings = self._load_notification_settings()

        # Database for alert persistence
        self.db_path = "risk_alerts.db"
        self.init_database()

        # Email configuration
        self.email_config = {
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'email_address': 'risk.alerts@company.com',
            'email_password': 'your_app_password',  # Use app password for Gmail
            'enabled': False  # Set to True to enable email alerts
        }

    def init_database(self):
        """Initialize SQLite database for alert storage"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Create alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id TEXT PRIMARY KEY,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    title TEXT NOT NULL,
                    message TEXT NOT NULL,
                    metric_name TEXT,
                    current_value REAL,
                    threshold_value REAL,
                    breach_percentage REAL,
                    portfolio_name TEXT,
                    timestamp DATETIME,
                    acknowledged BOOLEAN DEFAULT FALSE,
                    resolved BOOLEAN DEFAULT FALSE,
                    assigned_to TEXT,
                    resolution_notes TEXT
                )
            ''')

            # Create limits table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS risk_limits (
                    limit_id TEXT PRIMARY KEY,
                    limit_name TEXT NOT NULL,
                    metric_type TEXT NOT NULL,
                    portfolio_name TEXT,
                    warning_threshold REAL,
                    breach_threshold REAL,
                    critical_threshold REAL,
                    currency TEXT,
                    active BOOLEAN DEFAULT TRUE,
                    description TEXT
                )
            ''')

            conn.commit()
            conn.close()

        except Exception as e:
            st.error(f"Alert database initialization failed: {e}")

    def _load_default_risk_limits(self) -> List[RiskLimit]:
        """Load default risk limits configuration"""
        return [
            RiskLimit(
                limit_id="VAR_1D_TOTAL",
                limit_name="1-Day VaR - Total Portfolio",
                metric_type="var_1d",
                portfolio_name="Total Portfolio",
                warning_threshold=20.0,
                breach_threshold=25.0,
                critical_threshold=30.0,
                currency="USD",
                description="Daily Value at Risk for entire portfolio"
            ),
            RiskLimit(
                limit_id="VAR_1D_EQUITY",
                limit_name="1-Day VaR - Equity Portfolio",
                metric_type="var_1d",
                portfolio_name="Equity Portfolio",
                warning_threshold=15.0,
                breach_threshold=20.0,
                critical_threshold=25.0,
                currency="USD",
                description="Daily Value at Risk for equity positions"
            ),
            RiskLimit(
                limit_id="EXPECTED_SHORTFALL",
                limit_name="Expected Shortfall - Total",
                metric_type="expected_shortfall",
                portfolio_name="Total Portfolio",
                warning_threshold=30.0,
                breach_threshold=40.0,
                critical_threshold=50.0,
                currency="USD",
                description="Expected Shortfall (Conditional VaR)"
            ),
            RiskLimit(
                limit_id="CONCENTRATION_SINGLE_NAME",
                limit_name="Single Name Concentration",
                metric_type="concentration",
                portfolio_name="Total Portfolio",
                warning_threshold=5.0,
                breach_threshold=8.0,
                critical_threshold=10.0,
                currency="%",
                description="Maximum concentration in single security"
            ),
            RiskLimit(
                limit_id="SECTOR_CONCENTRATION",
                limit_name="Sector Concentration",
                metric_type="sector_concentration",
                portfolio_name="Total Portfolio",
                warning_threshold=25.0,
                breach_threshold=35.0,
                critical_threshold=40.0,
                currency="%",
                description="Maximum concentration in single sector"
            ),
            RiskLimit(
                limit_id="LEVERAGE_RATIO",
                limit_name="Portfolio Leverage",
                metric_type="leverage",
                portfolio_name="Total Portfolio",
                warning_threshold=2.0,
                breach_threshold=3.0,
                critical_threshold=4.0,
                currency="X",
                description="Portfolio leverage ratio"
            ),
            RiskLimit(
                limit_id="LIQUIDITY_COVERAGE",
                limit_name="Liquidity Coverage Ratio",
                metric_type="liquidity_coverage",
                portfolio_name="Total Portfolio",
                warning_threshold=100.0,
                breach_threshold=80.0,
                critical_threshold=60.0,
                currency="%",
                description="Liquidity coverage ratio"
            )
        ]

    def _load_notification_settings(self) -> Dict:
        """Load notification settings"""
        return {
            'email_enabled': True,
            'sms_enabled': False,
            'dashboard_popup': True,
            'sound_alerts': False,
            'recipients': {
                'critical': ['risk.manager@company.com', 'cro@company.com'],
                'high': ['risk.manager@company.com', 'portfolio.manager@company.com'],
                'medium': ['risk.analyst@company.com'],
                'low': ['risk.analyst@company.com']
            },
            'escalation_rules': {
                'unacknowledged_time_minutes': 30,
                'escalation_levels': ['supervisor', 'manager', 'cro']
            }
        }

    def check_risk_limits(self, portfolio_metrics: Dict[str, float],
                         portfolio_name: str = "Total Portfolio") -> List[RiskAlert]:
        """Check portfolio metrics against risk limits and generate alerts"""
        new_alerts = []

        try:
            for limit in self.risk_limits:
                if not limit.active or limit.portfolio_name != portfolio_name:
                    continue

                metric_value = portfolio_metrics.get(limit.metric_type, 0.0)

                # Check if limit is breached
                severity = None
                threshold_breached = None

                if metric_value >= limit.critical_threshold:
                    severity = AlertSeverity.CRITICAL
                    threshold_breached = limit.critical_threshold
                elif metric_value >= limit.breach_threshold:
                    severity = AlertSeverity.HIGH
                    threshold_breached = limit.breach_threshold
                elif metric_value >= limit.warning_threshold:
                    severity = AlertSeverity.MEDIUM
                    threshold_breached = limit.warning_threshold

                if severity and threshold_breached:
                    # Calculate breach percentage
                    breach_percentage = ((metric_value - threshold_breached) / threshold_breached) * 100

                    # Create alert
                    alert = RiskAlert(
                        alert_id=f"{limit.limit_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        alert_type=self._get_alert_type_from_metric(limit.metric_type),
                        severity=severity,
                        title=f"{limit.limit_name} Breach",
                        message=f"{limit.limit_name} has breached the {severity.value.lower()} threshold. "
                               f"Current value: {metric_value:.2f} {limit.currency}, "
                               f"Threshold: {threshold_breached:.2f} {limit.currency} "
                               f"({breach_percentage:.1f}% over limit)",
                        metric_name=limit.metric_type,
                        current_value=metric_value,
                        threshold_value=threshold_breached,
                        breach_percentage=breach_percentage,
                        portfolio_name=portfolio_name,
                        timestamp=datetime.now()
                    )

                    new_alerts.append(alert)

            # Add alerts to active list and save to database
            for alert in new_alerts:
                self.active_alerts.append(alert)
                self._save_alert_to_db(alert)

            return new_alerts

        except Exception as e:
            st.error(f"Error checking risk limits: {e}")
            return []

    def _get_alert_type_from_metric(self, metric_type: str) -> AlertType:
        """Map metric type to alert type"""
        mapping = {
            'var_1d': AlertType.VAR_BREACH,
            'expected_shortfall': AlertType.VAR_BREACH,
            'concentration': AlertType.CONCENTRATION_RISK,
            'sector_concentration': AlertType.CONCENTRATION_RISK,
            'leverage': AlertType.LIMIT_BREACH,
            'liquidity_coverage': AlertType.LIQUIDITY_RISK
        }
        return mapping.get(metric_type, AlertType.LIMIT_BREACH)

    def generate_market_stress_alerts(self, market_indicators: Dict[str, float]) -> List[RiskAlert]:
        """Generate alerts based on market stress indicators"""
        stress_alerts = []

        try:
            # VIX spike alert
            if market_indicators.get('vix', 0) > 30:
                alert = RiskAlert(
                    alert_id=f"VIX_SPIKE_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    alert_type=AlertType.MARKET_STRESS,
                    severity=AlertSeverity.HIGH,
                    title="High Market Volatility Detected",
                    message=f"VIX has spiked to {market_indicators['vix']:.1f}, indicating high market stress. "
                           f"Consider reviewing portfolio risk exposure and hedging strategies.",
                    metric_name="vix",
                    current_value=market_indicators['vix'],
                    threshold_value=30.0,
                    breach_percentage=((market_indicators['vix'] - 30) / 30) * 100,
                    portfolio_name="Total Portfolio",
                    timestamp=datetime.now()
                )
                stress_alerts.append(alert)

            # Credit spread widening
            if market_indicators.get('credit_spreads', 0) > 500:  # 5% spreads
                alert = RiskAlert(
                    alert_id=f"CREDIT_SPREAD_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    alert_type=AlertType.MARKET_STRESS,
                    severity=AlertSeverity.MEDIUM,
                    title="Credit Spreads Widening",
                    message=f"Credit spreads have widened to {market_indicators['credit_spreads']:.0f}bp, "
                           f"indicating potential credit stress in the market.",
                    metric_name="credit_spreads",
                    current_value=market_indicators['credit_spreads'],
                    threshold_value=500.0,
                    breach_percentage=((market_indicators['credit_spreads'] - 500) / 500) * 100,
                    portfolio_name="Total Portfolio",
                    timestamp=datetime.now()
                )
                stress_alerts.append(alert)

            # Add to active alerts
            for alert in stress_alerts:
                self.active_alerts.append(alert)
                self._save_alert_to_db(alert)

            return stress_alerts

        except Exception as e:
            st.error(f"Error generating market stress alerts: {e}")
            return []

    def _save_alert_to_db(self, alert: RiskAlert):
        """Save alert to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT OR REPLACE INTO alerts
                (alert_id, alert_type, severity, title, message, metric_name,
                 current_value, threshold_value, breach_percentage, portfolio_name,
                 timestamp, acknowledged, resolved, assigned_to, resolution_notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert.alert_id,
                alert.alert_type.value,
                alert.severity.value,
                alert.title,
                alert.message,
                alert.metric_name,
                alert.current_value,
                alert.threshold_value,
                alert.breach_percentage,
                alert.portfolio_name,
                alert.timestamp.isoformat(),
                alert.acknowledged,
                alert.resolved,
                alert.assigned_to,
                alert.resolution_notes
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            st.error(f"Error saving alert to database: {e}")

    def send_email_alert(self, alert: RiskAlert):
        """Send email notification for alert"""
        if not self.email_config['enabled']:
            return

        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.email_config['email_address']
            msg['Subject'] = f"[RISK ALERT - {alert.severity.value}] {alert.title}"

            # Get recipients based on severity
            recipients = self.notification_settings['recipients'].get(
                alert.severity.value.lower(), ['risk.analyst@company.com']
            )
            msg['To'] = ', '.join(recipients)

            # Create email body
            body = f"""
            Risk Alert Notification
            =======================

            Alert ID: {alert.alert_id}
            Severity: {alert.severity.value}
            Type: {alert.alert_type.value}
            Portfolio: {alert.portfolio_name}
            Timestamp: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

            Description:
            {alert.message}

            Metric Details:
            - Metric: {alert.metric_name}
            - Current Value: {alert.current_value:.2f}
            - Threshold: {alert.threshold_value:.2f}
            - Breach: {alert.breach_percentage:.1f}% over limit

            Please acknowledge this alert in the risk dashboard and take appropriate action.

            Best regards,
            Risk Management System
            """

            msg.attach(MIMEText(body, 'plain'))

            # Send email
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['email_address'], self.email_config['email_password'])
            text = msg.as_string()
            server.sendmail(self.email_config['email_address'], recipients, text)
            server.quit()

            st.success(f"Email alert sent for {alert.alert_id}")

        except Exception as e:
            st.error(f"Failed to send email alert: {e}")

    def acknowledge_alert(self, alert_id: str, user: str, notes: str = ""):
        """Acknowledge an alert"""
        try:
            # Update in active alerts
            for alert in self.active_alerts:
                if alert.alert_id == alert_id:
                    alert.acknowledged = True
                    alert.assigned_to = user
                    alert.resolution_notes = notes
                    break

            # Update in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                UPDATE alerts
                SET acknowledged = ?, assigned_to = ?, resolution_notes = ?
                WHERE alert_id = ?
            ''', (True, user, notes, alert_id))

            conn.commit()
            conn.close()

            return True

        except Exception as e:
            st.error(f"Error acknowledging alert: {e}")
            return False

    def resolve_alert(self, alert_id: str, user: str, resolution_notes: str):
        """Resolve an alert"""
        try:
            # Update in active alerts
            for i, alert in enumerate(self.active_alerts):
                if alert.alert_id == alert_id:
                    alert.resolved = True
                    alert.assigned_to = user
                    alert.resolution_notes = resolution_notes
                    # Move to history
                    self.alert_history.append(self.active_alerts.pop(i))
                    break

            # Update in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                UPDATE alerts
                SET resolved = ?, assigned_to = ?, resolution_notes = ?
                WHERE alert_id = ?
            ''', (True, user, resolution_notes, alert_id))

            conn.commit()
            conn.close()

            return True

        except Exception as e:
            st.error(f"Error resolving alert: {e}")
            return False

    def get_active_alerts(self, severity_filter: Optional[AlertSeverity] = None) -> List[RiskAlert]:
        """Get list of active alerts"""
        if severity_filter:
            return [alert for alert in self.active_alerts
                   if alert.severity == severity_filter and not alert.resolved]
        return [alert for alert in self.active_alerts if not alert.resolved]

    def get_alert_statistics(self) -> Dict[str, int]:
        """Get alert statistics"""
        try:
            total_alerts = len(self.active_alerts)
            critical_alerts = len([a for a in self.active_alerts if a.severity == AlertSeverity.CRITICAL])
            high_alerts = len([a for a in self.active_alerts if a.severity == AlertSeverity.HIGH])
            medium_alerts = len([a for a in self.active_alerts if a.severity == AlertSeverity.MEDIUM])
            low_alerts = len([a for a in self.active_alerts if a.severity == AlertSeverity.LOW])

            unacknowledged = len([a for a in self.active_alerts if not a.acknowledged])

            return {
                'total': total_alerts,
                'critical': critical_alerts,
                'high': high_alerts,
                'medium': medium_alerts,
                'low': low_alerts,
                'unacknowledged': unacknowledged
            }

        except Exception as e:
            return {'total': 0, 'critical': 0, 'high': 0, 'medium': 0, 'low': 0, 'unacknowledged': 0}

    def render_alert_dashboard(self):
        """Render alert management dashboard"""
        st.markdown("## 🚨 Risk Alert Management")

        # Alert statistics
        stats = self.get_alert_statistics()

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Total Alerts", stats['total'])

        with col2:
            st.metric("Critical", stats['critical'], delta=None,
                     delta_color="inverse" if stats['critical'] > 0 else "normal")

        with col3:
            st.metric("High", stats['high'], delta=None,
                     delta_color="inverse" if stats['high'] > 0 else "normal")

        with col4:
            st.metric("Medium", stats['medium'])

        with col5:
            st.metric("Unacknowledged", stats['unacknowledged'],
                     delta=None, delta_color="inverse" if stats['unacknowledged'] > 0 else "normal")

        # Active alerts table
        st.markdown("### 📋 Active Alerts")

        if self.active_alerts:
            # Create DataFrame for display
            alert_data = []
            for alert in self.active_alerts:
                if not alert.resolved:
                    alert_data.append({
                        'ID': alert.alert_id,
                        'Severity': alert.severity.value,
                        'Type': alert.alert_type.value,
                        'Title': alert.title,
                        'Portfolio': alert.portfolio_name,
                        'Current Value': f"{alert.current_value:.2f}",
                        'Threshold': f"{alert.threshold_value:.2f}",
                        'Breach %': f"{alert.breach_percentage:.1f}%",
                        'Timestamp': alert.timestamp.strftime('%Y-%m-%d %H:%M'),
                        'Acknowledged': "✅" if alert.acknowledged else "❌",
                        'Assigned To': alert.assigned_to or "Unassigned"
                    })

            if alert_data:
                df = pd.DataFrame(alert_data)

                # Color code by severity
                def highlight_severity(row):
                    if row['Severity'] == 'CRITICAL':
                        return ['background-color: #fee2e2; color: #991b1b'] * len(row)
                    elif row['Severity'] == 'HIGH':
                        return ['background-color: #fef3c7; color: #92400e'] * len(row)
                    elif row['Severity'] == 'MEDIUM':
                        return ['background-color: #dbeafe; color: #1e40af'] * len(row)
                    else:
                        return ['background-color: #f0fdf4; color: #166534'] * len(row)

                styled_df = df.style.apply(highlight_severity, axis=1)
                st.dataframe(styled_df, use_container_width=True)

                # Alert actions
                st.markdown("### ⚡ Alert Actions")

                col1, col2 = st.columns(2)

                with col1:
                    selected_alert = st.selectbox(
                        "Select Alert ID",
                        [alert.alert_id for alert in self.active_alerts if not alert.resolved]
                    )

                with col2:
                    action = st.selectbox("Action", ["Acknowledge", "Resolve"])

                if action == "Acknowledge":
                    notes = st.text_area("Acknowledgment Notes", placeholder="Enter any notes...")
                    if st.button("Acknowledge Alert"):
                        if self.acknowledge_alert(selected_alert, st.session_state.get('user_name', 'User'), notes):
                            st.success("Alert acknowledged successfully")
                            st.rerun()

                elif action == "Resolve":
                    resolution_notes = st.text_area("Resolution Notes", placeholder="Describe the resolution...")
                    if st.button("Resolve Alert"):
                        if resolution_notes:
                            if self.resolve_alert(selected_alert, st.session_state.get('user_name', 'User'), resolution_notes):
                                st.success("Alert resolved successfully")
                                st.rerun()
                        else:
                            st.warning("Please provide resolution notes")

            else:
                st.info("No active alerts")

        else:
            st.success("🎉 No active alerts - All systems operating within limits")

    def simulate_demo_alerts(self):
        """Generate some demo alerts for testing"""
        demo_alerts = [
            RiskAlert(
                alert_id=f"DEMO_VAR_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                alert_type=AlertType.VAR_BREACH,
                severity=AlertSeverity.HIGH,
                title="VaR Limit Breach - Equity Desk",
                message="1-Day VaR has exceeded the warning threshold. Current VaR: $28.5M, Limit: $25.0M",
                metric_name="var_1d",
                current_value=28.5,
                threshold_value=25.0,
                breach_percentage=14.0,
                portfolio_name="Equity Portfolio",
                timestamp=datetime.now() - timedelta(minutes=15)
            ),
            RiskAlert(
                alert_id=f"DEMO_CONC_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                alert_type=AlertType.CONCENTRATION_RISK,
                severity=AlertSeverity.MEDIUM,
                title="Sector Concentration Warning",
                message="Technology sector concentration has reached 38% of total portfolio value",
                metric_name="sector_concentration",
                current_value=38.0,
                threshold_value=35.0,
                breach_percentage=8.6,
                portfolio_name="Total Portfolio",
                timestamp=datetime.now() - timedelta(minutes=45)
            ),
            RiskAlert(
                alert_id=f"DEMO_MODEL_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                alert_type=AlertType.MODEL_VALIDATION,
                severity=AlertSeverity.LOW,
                title="Model Validation Due",
                message="VaR model validation is due for quarterly review",
                metric_name="model_validation",
                current_value=90.0,
                threshold_value=90.0,
                breach_percentage=0.0,
                portfolio_name="Total Portfolio",
                timestamp=datetime.now() - timedelta(hours=2)
            )
        ]

        for alert in demo_alerts:
            if not any(a.alert_id == alert.alert_id for a in self.active_alerts):
                self.active_alerts.append(alert)
                self._save_alert_to_db(alert)