#!/usr/bin/env python3
"""
Risk Management Dashboard Startup Script
=========================================

This script launches the professional risk management dashboard with proper
configuration and error handling.

Usage:
    python run_dashboard.py [--port PORT] [--debug] [--demo]

Arguments:
    --port PORT    Port number to run the dashboard (default: 8501)
    --debug        Enable debug mode with detailed logging
    --demo         Run in demo mode with simulated data
"""

import sys
import os
import argparse
import subprocess
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit',
        'pandas',
        'numpy',
        'plotly',
        'yfinance',
        'reportlab'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        logger.error(f"Missing required packages: {missing_packages}")
        logger.error("Please install requirements: pip install -r requirements.txt")
        return False

    return True

def setup_environment():
    """Setup environment variables and configuration"""
    # Set default environment variables
    os.environ.setdefault('STREAMLIT_SERVER_PORT', '8501')
    os.environ.setdefault('STREAMLIT_SERVER_ADDRESS', 'localhost')
    os.environ.setdefault('STREAMLIT_BROWSER_GATHER_USAGE_STATS', 'false')
    os.environ.setdefault('STREAMLIT_SERVER_ENABLE_CORS', 'false')

    # Create necessary directories
    dashboard_dir = Path(__file__).parent

    # Create data directory if it doesn't exist
    data_dir = dashboard_dir / 'data'
    data_dir.mkdir(exist_ok=True)

    # Create logs directory if it doesn't exist
    logs_dir = dashboard_dir / 'logs'
    logs_dir.mkdir(exist_ok=True)

    logger.info("Environment setup completed")

def create_streamlit_config():
    """Create Streamlit configuration file"""
    config_dir = Path.home() / '.streamlit'
    config_dir.mkdir(exist_ok=True)

    config_file = config_dir / 'config.toml'

    config_content = """
[global]
developmentMode = false
showWarningOnDirectExecution = false

[server]
enableCORS = false
enableXsrfProtection = true
maxUploadSize = 200
headless = false

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#3b82f6"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f8fafc"
textColor = "#1f2937"
font = "sans serif"
"""

    with open(config_file, 'w') as f:
        f.write(config_content)

    logger.info(f"Streamlit config created at {config_file}")

def validate_dashboard_files():
    """Validate that all required dashboard files exist"""
    dashboard_dir = Path(__file__).parent

    required_files = [
        'main_dashboard.py',
        'src/auth_manager.py',
        'src/risk_engine.py',
        'src/data_provider.py',
        'src/alert_manager.py',
        'src/export_manager.py',
        'src/visualization_engine.py',
        'requirements.txt'
    ]

    missing_files = []

    for file_path in required_files:
        if not (dashboard_dir / file_path).exists():
            missing_files.append(file_path)

    if missing_files:
        logger.error(f"Missing required files: {missing_files}")
        return False

    logger.info("All required files validated")
    return True

def run_dashboard(port=8501, debug=False, demo=False):
    """Launch the Streamlit dashboard"""
    dashboard_dir = Path(__file__).parent
    main_file = dashboard_dir / 'main_dashboard.py'

    # Build streamlit command
    cmd = [
        sys.executable, '-m', 'streamlit', 'run',
        str(main_file),
        '--server.port', str(port),
        '--server.address', 'localhost'
    ]

    if debug:
        cmd.extend(['--logger.level', 'debug'])

    # Set environment variables for demo mode
    if demo:
        os.environ['DASHBOARD_DEMO_MODE'] = 'true'
        logger.info("Running in demo mode with simulated data")

    logger.info(f"Starting Risk Management Dashboard on port {port}")
    logger.info(f"Dashboard will be available at: http://localhost:{port}")

    try:
        # Launch Streamlit
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start dashboard: {e}")
        return False
    except KeyboardInterrupt:
        logger.info("Dashboard shutdown requested")
        return True

    return True

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Launch the Professional Risk Management Dashboard'
    )

    parser.add_argument(
        '--port',
        type=int,
        default=8501,
        help='Port number to run the dashboard (default: 8501)'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode with detailed logging'
    )

    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run in demo mode with simulated data'
    )

    parser.add_argument(
        '--check-only',
        action='store_true',
        help='Only check requirements and configuration, do not start dashboard'
    )

    args = parser.parse_args()

    # Print banner
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║           Professional Risk Management Dashboard             ║
    ║                                                              ║
    ║  A comprehensive risk analytics platform built with         ║
    ║  Streamlit, featuring real-time monitoring, advanced        ║
    ║  visualizations, and professional reporting capabilities.   ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    # Setup logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")

    # Validate environment
    logger.info("Validating environment...")

    if not check_requirements():
        logger.error("Requirements check failed")
        sys.exit(1)

    if not validate_dashboard_files():
        logger.error("File validation failed")
        sys.exit(1)

    # Setup environment
    setup_environment()
    create_streamlit_config()

    if args.check_only:
        logger.info("Environment check completed successfully")
        return

    # Launch dashboard
    success = run_dashboard(
        port=args.port,
        debug=args.debug,
        demo=args.demo
    )

    if not success:
        sys.exit(1)

if __name__ == '__main__':
    main()