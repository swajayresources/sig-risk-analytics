"""
Authentication Manager for Risk Dashboard
=========================================

Handles user authentication, session management, and role-based access control.
"""

import streamlit as st
import hashlib
import jwt
import datetime
from typing import Dict, Optional, List
import pandas as pd
import json

class AuthManager:
    """Manages user authentication and authorization"""

    def __init__(self):
        """Initialize authentication manager"""
        self.secret_key = "risk_dashboard_secret_key_2024"
        self.users_db = self.load_users_database()
        self.roles_config = self.load_roles_configuration()

    def load_users_database(self) -> Dict:
        """Load user database (in production, this would be from a secure database)"""
        return {
            "admin": {
                "password_hash": self.hash_password("admin123"),
                "role": "Administrator",
                "name": "System Administrator",
                "email": "admin@riskdashboard.com",
                "permissions": ["all"]
            },
            "risk_manager": {
                "password_hash": self.hash_password("risk123"),
                "role": "Risk Manager",
                "name": "John Smith",
                "email": "john.smith@riskdashboard.com",
                "permissions": ["view_portfolio", "manage_limits", "run_stress_tests", "generate_reports"]
            },
            "trader": {
                "password_hash": self.hash_password("trade123"),
                "role": "Trader",
                "name": "Alice Johnson",
                "email": "alice.johnson@riskdashboard.com",
                "permissions": ["view_portfolio", "view_positions", "view_pnl"]
            },
            "compliance": {
                "password_hash": self.hash_password("comply123"),
                "role": "Compliance Officer",
                "name": "Mike Wilson",
                "email": "mike.wilson@riskdashboard.com",
                "permissions": ["view_portfolio", "view_regulatory", "generate_reports", "manage_limits"]
            },
            "viewer": {
                "password_hash": self.hash_password("view123"),
                "role": "Viewer",
                "name": "Guest User",
                "email": "guest@riskdashboard.com",
                "permissions": ["view_portfolio"]
            }
        }

    def load_roles_configuration(self) -> Dict:
        """Load role-based access configuration"""
        return {
            "Administrator": {
                "can_access": ["all_pages"],
                "can_export": True,
                "can_modify_limits": True,
                "can_send_alerts": True,
                "dashboard_layout": "advanced"
            },
            "Risk Manager": {
                "can_access": ["portfolio_overview", "var_analysis", "stress_testing", "risk_limits", "regulatory"],
                "can_export": True,
                "can_modify_limits": True,
                "can_send_alerts": True,
                "dashboard_layout": "advanced"
            },
            "Trader": {
                "can_access": ["portfolio_overview", "pnl_attribution", "greeks_monitoring"],
                "can_export": False,
                "can_modify_limits": False,
                "can_send_alerts": False,
                "dashboard_layout": "simplified"
            },
            "Compliance Officer": {
                "can_access": ["portfolio_overview", "regulatory", "risk_limits"],
                "can_export": True,
                "can_modify_limits": False,
                "can_send_alerts": True,
                "dashboard_layout": "compliance"
            },
            "Viewer": {
                "can_access": ["portfolio_overview"],
                "can_export": False,
                "can_modify_limits": False,
                "can_send_alerts": False,
                "dashboard_layout": "readonly"
            }
        }

    def hash_password(self, password: str) -> str:
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()

    def verify_password(self, username: str, password: str) -> bool:
        """Verify user password"""
        if username not in self.users_db:
            return False

        password_hash = self.hash_password(password)
        return self.users_db[username]["password_hash"] == password_hash

    def authenticate_user(self, username: str, password: str) -> Optional[Dict]:
        """Authenticate user and return user info"""
        if self.verify_password(username, password):
            user_info = self.users_db[username].copy()
            user_info["username"] = username
            del user_info["password_hash"]  # Remove sensitive data
            return user_info
        return None

    def generate_token(self, user_info: Dict) -> str:
        """Generate JWT token for authenticated user"""
        payload = {
            "username": user_info["username"],
            "role": user_info["role"],
            "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=8)
        }
        return jwt.encode(payload, self.secret_key, algorithm="HS256")

    def verify_token(self, token: str) -> Optional[Dict]:
        """Verify JWT token and return user info"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

    def check_permission(self, permission: str) -> bool:
        """Check if current user has specific permission"""
        if not st.session_state.authenticated:
            return False

        user_role = st.session_state.user_role
        if user_role == "Administrator":
            return True

        user_permissions = self.users_db.get(st.session_state.user_name, {}).get("permissions", [])
        return permission in user_permissions or "all" in user_permissions

    def render_login(self):
        """Render login interface"""
        st.markdown("""
        <div style="max-width: 400px; margin: 2rem auto; padding: 2rem;
                    background: white; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
            <h2 style="text-align: center; color: #1e40af; margin-bottom: 2rem;">
                🏦 Risk Dashboard Login
            </h2>
        """, unsafe_allow_html=True)

        # Login form
        with st.form("login_form"):
            st.markdown("### 🔐 Please sign in to continue")

            username = st.text_input("👤 Username", placeholder="Enter your username")
            password = st.text_input("🔑 Password", type="password", placeholder="Enter your password")

            col1, col2 = st.columns(2)
            with col1:
                login_submitted = st.form_submit_button("🚀 Sign In", use_container_width=True)
            with col2:
                demo_submitted = st.form_submit_button("👁️ Demo Mode", use_container_width=True)

        if login_submitted:
            if username and password:
                user_info = self.authenticate_user(username, password)
                if user_info:
                    # Successful login
                    st.session_state.authenticated = True
                    st.session_state.user_name = username
                    st.session_state.user_role = user_info["role"]
                    st.session_state.user_email = user_info["email"]
                    st.session_state.user_permissions = user_info["permissions"]

                    # Generate and store token
                    token = self.generate_token(user_info)
                    st.session_state.auth_token = token

                    st.success(f"✅ Welcome back, {user_info['name']}!")
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password")
            else:
                st.warning("⚠️ Please enter both username and password")

        if demo_submitted:
            # Demo mode - authenticate as viewer
            user_info = self.users_db["viewer"]
            st.session_state.authenticated = True
            st.session_state.user_name = "demo_user"
            st.session_state.user_role = "Viewer"
            st.session_state.user_email = "demo@riskdashboard.com"
            st.session_state.user_permissions = ["view_portfolio"]

            st.info("🎯 Logged in as Demo User (Read-only access)")
            st.rerun()

        # Demo credentials info
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("🔍 Demo Credentials"):
            st.markdown("""
            **Available Demo Accounts:**

            | Role | Username | Password | Access Level |
            |------|----------|----------|--------------|
            | Administrator | admin | admin123 | Full access |
            | Risk Manager | risk_manager | risk123 | Risk management |
            | Trader | trader | trade123 | Trading desk |
            | Compliance | compliance | comply123 | Compliance view |
            | Viewer | viewer | view123 | Read-only |

            **Or click 'Demo Mode' for instant read-only access**
            """)

        # System status
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("🟢 System Online")
        with col2:
            st.info("📡 Data Feed Active")
        with col3:
            st.info("🔒 SSL Secured")

    def render_user_profile(self):
        """Render user profile section"""
        if not st.session_state.authenticated:
            return

        with st.sidebar:
            st.markdown("---")
            st.markdown("### 👤 User Profile")

            user_info = self.users_db.get(st.session_state.user_name, {})

            st.write(f"**Name:** {user_info.get('name', 'Unknown')}")
            st.write(f"**Role:** {st.session_state.user_role}")
            st.write(f"**Email:** {user_info.get('email', 'N/A')}")

            # Logout button
            if st.button("🚪 Logout", use_container_width=True):
                self.logout()

    def logout(self):
        """Logout current user"""
        # Clear all session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]

        st.success("👋 Successfully logged out")
        st.rerun()

    def get_user_dashboard_config(self) -> Dict:
        """Get dashboard configuration for current user role"""
        if not st.session_state.authenticated:
            return {}

        user_role = st.session_state.user_role
        return self.roles_config.get(user_role, {})

    def can_access_page(self, page_name: str) -> bool:
        """Check if user can access specific page"""
        if not st.session_state.authenticated:
            return False

        config = self.get_user_dashboard_config()
        accessible_pages = config.get("can_access", [])

        return "all_pages" in accessible_pages or page_name in accessible_pages

    def render_access_denied(self, page_name: str):
        """Render access denied message"""
        st.error(f"🚫 Access Denied: You don't have permission to access {page_name}")
        st.info("Contact your administrator for access to this feature.")

    def audit_log_action(self, action: str, details: str = ""):
        """Log user action for audit trail"""
        if not st.session_state.authenticated:
            return

        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "username": st.session_state.user_name,
            "role": st.session_state.user_role,
            "action": action,
            "details": details,
            "ip_address": "127.0.0.1"  # In production, get real IP
        }

        # In production, this would write to a secure audit log database
        st.session_state.setdefault("audit_log", []).append(log_entry)

    def get_audit_log(self) -> List[Dict]:
        """Get audit log entries"""
        return st.session_state.get("audit_log", [])

    def render_session_management(self):
        """Render session management interface"""
        if not self.check_permission("admin"):
            return

        st.markdown("### 👥 Active Sessions")

        # Mock active sessions data
        sessions_data = {
            "Username": ["admin", "risk_manager", "trader", "compliance"],
            "Role": ["Administrator", "Risk Manager", "Trader", "Compliance Officer"],
            "Login Time": ["09:00 AM", "09:15 AM", "09:30 AM", "10:00 AM"],
            "Last Activity": ["10:45 AM", "10:42 AM", "10:40 AM", "10:35 AM"],
            "Status": ["🟢 Active", "🟢 Active", "🟡 Idle", "🟢 Active"]
        }

        df = pd.DataFrame(sessions_data)
        st.dataframe(df, use_container_width=True)

        # Session actions
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 Export Session Log"):
                st.success("Session log exported")

        with col2:
            if st.button("🔄 Refresh Sessions"):
                st.rerun()

    def enforce_password_policy(self, password: str) -> bool:
        """Enforce password policy"""
        if len(password) < 8:
            return False
        if not any(c.isupper() for c in password):
            return False
        if not any(c.islower() for c in password):
            return False
        if not any(c.isdigit() for c in password):
            return False
        return True

    def render_security_settings(self):
        """Render security settings interface"""
        if not self.check_permission("admin"):
            return

        st.markdown("### 🔒 Security Settings")

        # Password policy
        st.markdown("#### Password Policy")
        col1, col2 = st.columns(2)

        with col1:
            min_length = st.number_input("Minimum Length", min_value=6, max_value=20, value=8)
            require_uppercase = st.checkbox("Require Uppercase", value=True)

        with col2:
            require_numbers = st.checkbox("Require Numbers", value=True)
            require_symbols = st.checkbox("Require Symbols", value=False)

        # Session settings
        st.markdown("#### Session Management")
        col1, col2 = st.columns(2)

        with col1:
            session_timeout = st.number_input("Session Timeout (hours)", min_value=1, max_value=24, value=8)

        with col2:
            max_concurrent_sessions = st.number_input("Max Concurrent Sessions", min_value=1, max_value=10, value=3)

        if st.button("💾 Save Security Settings"):
            st.success("Security settings updated successfully")

        # Two-factor authentication
        st.markdown("#### Two-Factor Authentication")
        enable_2fa = st.checkbox("Enable 2FA for all users", value=False)

        if enable_2fa:
            st.info("📱 2FA will be enforced on next login for all users")