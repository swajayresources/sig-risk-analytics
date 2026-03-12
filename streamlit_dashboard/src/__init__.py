"""
Risk Management Dashboard Package
=================================

This package contains all the core modules for the professional risk management dashboard.
"""

__version__ = "1.0.0"
__author__ = "Risk Management Team"

# Make imports available at package level
from .risk_engine import RiskEngine
from .data_provider import DataProvider
from .alert_manager import AlertManager
from .export_manager import ExportManager
from .visualization_engine import VisualizationEngine

try:
    from .auth_manager import AuthManager
except ImportError:
    AuthManager = None

try:
    from .model_validation import ModelValidationFramework
except ImportError:
    ModelValidationFramework = None

try:
    from .backtesting_framework import BacktestingEngine
except ImportError:
    BacktestingEngine = None

try:
    from .data_quality import DataQualityValidator
except ImportError:
    DataQualityValidator = None

try:
    from .model_documentation import ModelDocumentationSystem
except ImportError:
    ModelDocumentationSystem = None

try:
    from .regulatory_compliance import RegulatoryComplianceEngine
except ImportError:
    RegulatoryComplianceEngine = None

__all__ = [
    'RiskEngine',
    'DataProvider',
    'AlertManager',
    'AuthManager',
    'ExportManager',
    'VisualizationEngine',
    'ModelValidationFramework',
    'BacktestingEngine',
    'DataQualityValidator',
    'ModelDocumentationSystem',
    'RegulatoryComplianceEngine'
]