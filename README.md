# SIG Risk Analytics Engine

Production-grade quantitative risk analytics platform built for the SIG internship application. Real-time portfolio risk monitoring with VaR, Greeks, Monte Carlo simulations, and stress testing.

## What It Does

- Real-time VaR and Expected Shortfall calculations
- Options Greeks (Delta, Gamma, Vega, Theta, Rho + higher-order)
- Portfolio optimization (Mean-Variance, Risk Parity, Black-Litterman)
- Monte Carlo stress testing with historical scenarios
- Compliance and regulatory reporting
- Export to PDF / Excel

## Quick Start

### Prerequisites
- Python 3.9+
- No Redis or database setup needed — SQLite is used automatically

### Minimal install & run

```bash
pip install streamlit pandas numpy plotly scipy yfinance reportlab scikit-learn statsmodels sqlalchemy
cd streamlit_dashboard
streamlit run main_dashboard.py
```

Open http://localhost:8501

### Full install

```bash
pip install -r streamlit_dashboard/requirements.txt
cd streamlit_dashboard
streamlit run main_dashboard.py
```

> **Note**: Some optional packages in `requirements.txt` (QuantLib, TensorFlow, PyTorch, geopandas) can be skipped. If the full install fails, use the minimal install above.

## Dependency Notes

- **Redis**: Not required. Dashboard uses SQLite automatically.
- **PostgreSQL / MongoDB**: Not required. All storage is SQLite-based locally.
- **QuantLib**: Optional — advanced pricing falls back to built-in implementations.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Dashboard | Streamlit, Plotly |
| Risk engine | Python, NumPy, SciPy |
| Storage | SQLite (local), Redis (optional) |
| Exports | ReportLab (PDF), OpenPyXL (Excel) |

## Project Structure

```
SIG2/
├── streamlit_dashboard/
│   ├── main_dashboard.py          # Entry point — run this
│   ├── src/
│   │   ├── risk_engine.py         # VaR, Greeks, Monte Carlo
│   │   ├── data_provider.py       # Market data feeds
│   │   ├── visualization_engine.py
│   │   ├── alert_manager.py
│   │   ├── export_manager.py
│   │   └── auth_manager.py
│   └── requirements.txt
├── web/                           # React frontend (optional)
└── src/                           # C++ engine (optional)
```
