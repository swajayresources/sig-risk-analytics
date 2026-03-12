# Professional Risk Management Dashboard

A comprehensive, professional-grade risk management dashboard built with Streamlit that rivals Bloomberg/Refinitiv terminals. Features real-time portfolio risk monitoring, advanced analytics, interactive visualizations, and automated reporting capabilities.

## 🚀 Features

### Core Risk Analytics
- **Real-time Portfolio Risk Metrics**: Live VaR, Expected Shortfall, and risk factor monitoring
- **Advanced VaR Calculations**: Historical simulation, parametric, and Monte Carlo methods
- **Greeks Monitoring**: Comprehensive options risk management with delta, gamma, theta, vega, and rho tracking
- **Stress Testing**: Scenario analysis with historical and custom stress scenarios
- **Factor Analysis**: Risk factor heat maps, correlation matrices, and factor attribution

### Professional Visualizations
- **3D Risk Surface Plots**: Interactive 3D visualizations of portfolio risk by sector and region
- **Real-time Streaming Charts**: Live P&L and risk metric evolution with auto-refresh
- **Interactive Heat Maps**: Correlation matrices and risk factor analysis
- **Geographic Risk Maps**: Choropleth maps showing global risk concentration
- **Monte Carlo Distributions**: Visualization of simulation results with VaR overlays

### Risk Management Tools
- **Risk Limit Monitoring**: Real-time limit tracking with breach alerts
- **Alert Management**: Configurable risk alerts with email notifications
- **Regulatory Capital**: Basel III compliance monitoring and reporting
- **Portfolio Attribution**: P&L and risk contribution analysis
- **Backtesting**: Model validation and performance analysis

### Professional Features
- **Multi-user Authentication**: Role-based access control with different permission levels
- **Export Capabilities**: PDF reports, Excel exports, CSV downloads
- **Automated Reporting**: Scheduled risk reports with email delivery
- **Mobile-Responsive**: Optimized views for mobile and tablet access
- **Audit Trail**: Complete user action logging for compliance

## 📋 Requirements

### System Requirements
- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- 2GB free disk space
- Internet connection for real-time data

### Dependencies
See `requirements.txt` for complete list. Key dependencies include:
- `streamlit>=1.28.0`
- `pandas>=2.0.0`
- `plotly>=5.15.0`
- `yfinance>=0.2.20`
- `reportlab>=4.0.0`

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd streamlit_dashboard
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Optional: Install Redis (for real-time features)
```bash
# Windows (using Chocolatey)
choco install redis-64

# Linux
sudo apt-get install redis-server

# Mac
brew install redis
```

## 🚀 Quick Start

### Method 1: Using the Startup Script (Recommended)
```bash
python run_dashboard.py
```

### Method 2: Direct Streamlit Launch
```bash
streamlit run main_dashboard.py
```

### Command Line Options
```bash
# Run on custom port
python run_dashboard.py --port 8502

# Enable debug mode
python run_dashboard.py --debug

# Run in demo mode with simulated data
python run_dashboard.py --demo

# Check requirements only
python run_dashboard.py --check-only
```

## 🔐 Authentication

### Demo Accounts
The dashboard comes with pre-configured demo accounts:

| Role | Username | Password | Access Level |
|------|----------|----------|--------------|
| Administrator | admin | admin123 | Full access to all features |
| Risk Manager | risk_manager | risk123 | Risk management and reporting |
| Trader | trader | trade123 | Trading desk view |
| Compliance | compliance | comply123 | Compliance and regulatory |
| Viewer | viewer | view123 | Read-only access |

### Demo Mode
Click "Demo Mode" for instant read-only access without credentials.

## 📊 Dashboard Pages

### 1. Portfolio Overview
- Real-time portfolio metrics and key risk indicators
- Portfolio composition charts and P&L attribution
- Risk metrics table with conditional formatting
- Interactive risk evolution charts

### 2. VaR Analysis
- Multi-method VaR calculations and comparison
- Monte Carlo simulation results with distributions
- VaR breakdown by asset class and region
- Component VaR and marginal risk contributions
- Model validation and backtesting results

### 3. Stress Testing
- Comprehensive scenario analysis with multiple stress tests
- Market risk scenarios (equity crash, interest rate shock, etc.)
- Operational risk scenarios (liquidity crisis, cyber attacks)
- Custom scenario builder with real-time impact calculation
- Historical event replication (2008 crisis, COVID-19, etc.)

### 4. Greeks Monitoring
- Real-time options Greeks tracking (delta, gamma, theta, vega, rho)
- Greeks ladder charts by asset class and strategy
- Scenario analysis for volatility, time decay, and interest rates
- Risk limits monitoring with utilization tracking
- Hedging recommendations based on Greeks exposure

### 5. Risk Limits
- Real-time limit monitoring with traffic light system
- Detailed limits table with breach tracking
- Alert management and resolution workflow
- Historical breach analysis and patterns
- Limit effectiveness metrics and calibration

### 6. Regulatory Capital
- Basel III capital adequacy ratios (CET1, Tier 1, Total Capital)
- Risk-weighted assets breakdown and analysis
- Regulatory stress test results and projections
- Capital planning and forecasting tools
- Regulatory reporting status and deadlines

### 7. Factor Analysis
- Interactive risk factor heat maps
- Factor correlation matrices and analysis
- Portfolio factor exposure and attribution
- Market regime analysis and transition impacts
- Factor-based hedging recommendations

### 8. Export & Reports
- Professional PDF report generation
- Excel exports with multiple worksheets
- CSV data downloads for external analysis
- Scheduled automated reporting
- Email delivery and distribution lists

## 🎨 Customization

### Theme Configuration
The dashboard supports multiple themes and can be customized through the Settings page:
- Professional Blue (default)
- Dark Mode
- Light Mode
- High Contrast

### Chart Customization
- Multiple color schemes for different chart types
- Configurable chart heights and layouts
- Interactive features with zoom, pan, and hover details
- Export charts as images for external use

### Risk Model Configuration
- Adjustable VaR confidence levels and time horizons
- Customizable risk limits and alert thresholds
- Flexible portfolio groupings and attributions
- User-defined stress scenarios and factor models

## 📱 Mobile Support

The dashboard includes a mobile-optimized view accessible through the "Mobile View" page:
- Compact metric displays optimized for small screens
- Touch-friendly interface elements
- Essential risk information and quick actions
- Responsive charts that adapt to screen size

## 🔧 Configuration

### Environment Variables
Create a `.env` file for production configuration:
```bash
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=risk_dashboard

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379

# Email Configuration
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
EMAIL_ADDRESS=alerts@company.com
EMAIL_PASSWORD=your_app_password

# API Keys
ALPHA_VANTAGE_API_KEY=your_api_key
QUANDL_API_KEY=your_api_key
```

### Database Setup
For production use, configure a PostgreSQL database:
```sql
CREATE DATABASE risk_dashboard;
CREATE USER risk_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE risk_dashboard TO risk_user;
```

## 🐳 Docker Deployment

### Build Docker Image
```bash
docker build -t risk-dashboard .
```

### Run with Docker Compose
```bash
docker-compose up -d
```

This will start:
- Streamlit dashboard on port 8501
- Redis for caching on port 6379
- PostgreSQL database on port 5432

## 🔍 Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   python run_dashboard.py --port 8502
   ```

2. **Missing Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Redis Connection Failed**
   - Dashboard will work with simulated data if Redis is unavailable
   - Install and start Redis service for real-time features

4. **Data Loading Issues**
   - Check internet connection for Yahoo Finance data
   - Dashboard includes fallback simulated data

### Debug Mode
Enable debug mode for detailed logging:
```bash
python run_dashboard.py --debug
```

### Log Files
Check log files in the `logs/` directory for detailed error information.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the debug logs

## 🔮 Roadmap

### Upcoming Features
- Machine learning-based risk models
- Real-time market data integration
- Advanced portfolio optimization tools
- ESG risk analytics
- Cryptocurrency risk monitoring
- API for external integrations

### Performance Improvements
- Caching optimization
- Asynchronous data loading
- Database query optimization
- Chart rendering improvements

---

**Built with ❤️ using Streamlit and modern Python tools**