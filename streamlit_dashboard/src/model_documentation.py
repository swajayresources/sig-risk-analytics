"""
Model Documentation and Audit Trail System
==========================================

Comprehensive model documentation system for maintaining audit trails,
model governance, and regulatory compliance documentation.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib
import pickle
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import sqlite3
import uuid

logger = logging.getLogger(__name__)

class ModelType(Enum):
 """Model type enumeration"""
 VAR_MODEL = "VAR_MODEL"
 STRESS_TEST = "STRESS_TEST"
 MONTE_CARLO = "MONTE_CARLO"
 GREEKS_MODEL = "GREEKS_MODEL"
 RISK_ATTRIBUTION = "RISK_ATTRIBUTION"
 PORTFOLIO_OPTIMIZATION = "PORTFOLIO_OPTIMIZATION"

class ModelStatus(Enum):
 """Model status enumeration"""
 DEVELOPMENT = "DEVELOPMENT"
 TESTING = "TESTING"
 VALIDATION = "VALIDATION"
 APPROVED = "APPROVED"
 PRODUCTION = "PRODUCTION"
 DEPRECATED = "DEPRECATED"
 RETIRED = "RETIRED"

class ChangeType(Enum):
 """Change type enumeration"""
 CREATED = "CREATED"
 UPDATED = "UPDATED"
 VALIDATED = "VALIDATED"
 APPROVED = "APPROVED"
 DEPLOYED = "DEPLOYED"
 RETIRED = "RETIRED"
 PARAMETER_CHANGE = "PARAMETER_CHANGE"
 BUG_FIX = "BUG_FIX"

@dataclass
class ModelMetadata:
 """Model metadata structure"""
 model_id: str
 model_name: str
 model_type: ModelType
 version: str
 status: ModelStatus
 description: str
 owner: str
 created_date: datetime
 last_modified: datetime
 production_date: Optional[datetime] = None
 retirement_date: Optional[datetime] = None
 business_justification: str = ""
 regulatory_approval: bool = False
 model_complexity: str = "MEDIUM" # LOW, MEDIUM, HIGH
 materiality: str = "MEDIUM" # LOW, MEDIUM, HIGH
 tags: List[str] = field(default_factory=list)

@dataclass
class ModelParameters:
 """Model parameters structure"""
 confidence_level: float = 0.95
 time_horizon: int = 1
 simulation_count: int = 10000
 lookback_window: int = 252
 decay_factor: float = 0.94
 correlation_method: str = "pearson"
 volatility_model: str = "ewma"
 custom_parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ValidationResult:
 """Model validation result"""
 validation_id: str
 model_id: str
 validation_date: datetime
 validator_name: str
 validation_type: str
 passed: bool
 score: float
 findings: List[str]
 recommendations: List[str]
 next_validation_date: datetime
 validation_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ChangeRecord:
 """Model change record"""
 change_id: str
 model_id: str
 change_type: ChangeType
 change_date: datetime
 changed_by: str
 description: str
 impact_assessment: str
 approval_required: bool
 approved_by: Optional[str] = None
 approval_date: Optional[datetime] = None
 rollback_plan: str = ""
 affected_components: List[str] = field(default_factory=list)
 before_state: Dict[str, Any] = field(default_factory=dict)
 after_state: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceMetrics:
 """Model performance tracking"""
 metric_id: str
 model_id: str
 measurement_date: datetime
 accuracy_score: float
 precision_score: float
 recall_score: float
 f1_score: float
 execution_time: float
 memory_usage: float
 data_quality_score: float
 custom_metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class ModelDocumentation:
 """Complete model documentation"""
 metadata: ModelMetadata
 parameters: ModelParameters
 methodology: str
 assumptions: List[str]
 limitations: List[str]
 data_requirements: Dict[str, str]
 performance_benchmarks: Dict[str, float]
 validation_history: List[ValidationResult]
 change_history: List[ChangeRecord]
 performance_history: List[PerformanceMetrics]
 regulatory_notes: str = ""
 technical_specification: str = ""
 user_guide: str = ""

class ModelDocumentationDB:
 """Database interface for model documentation"""

 def __init__(self, db_path: str = "model_documentation.db"):
 self.db_path = db_path
 self.init_database()

 def init_database(self):
 """Initialize database schema"""
 try:
 conn = sqlite3.connect(self.db_path)
 cursor = conn.cursor()

 # Models table
 cursor.execute('''
 CREATE TABLE IF NOT EXISTS models (
 model_id TEXT PRIMARY KEY,
 model_name TEXT NOT NULL,
 model_type TEXT NOT NULL,
 version TEXT NOT NULL,
 status TEXT NOT NULL,
 description TEXT,
 owner TEXT NOT NULL,
 created_date DATETIME NOT NULL,
 last_modified DATETIME NOT NULL,
 production_date DATETIME,
 retirement_date DATETIME,
 business_justification TEXT,
 regulatory_approval BOOLEAN DEFAULT FALSE,
 model_complexity TEXT DEFAULT 'MEDIUM',
 materiality TEXT DEFAULT 'MEDIUM',
 tags TEXT,
 parameters TEXT,
 methodology TEXT,
 assumptions TEXT,
 limitations TEXT,
 data_requirements TEXT,
 performance_benchmarks TEXT,
 regulatory_notes TEXT,
 technical_specification TEXT,
 user_guide TEXT
 )
 ''')

 # Validation results table
 cursor.execute('''
 CREATE TABLE IF NOT EXISTS validation_results (
 validation_id TEXT PRIMARY KEY,
 model_id TEXT NOT NULL,
 validation_date DATETIME NOT NULL,
 validator_name TEXT NOT NULL,
 validation_type TEXT NOT NULL,
 passed BOOLEAN NOT NULL,
 score REAL NOT NULL,
 findings TEXT,
 recommendations TEXT,
 next_validation_date DATETIME,
 validation_data TEXT,
 FOREIGN KEY (model_id) REFERENCES models (model_id)
 )
 ''')

 # Change records table
 cursor.execute('''
 CREATE TABLE IF NOT EXISTS change_records (
 change_id TEXT PRIMARY KEY,
 model_id TEXT NOT NULL,
 change_type TEXT NOT NULL,
 change_date DATETIME NOT NULL,
 changed_by TEXT NOT NULL,
 description TEXT NOT NULL,
 impact_assessment TEXT,
 approval_required BOOLEAN DEFAULT FALSE,
 approved_by TEXT,
 approval_date DATETIME,
 rollback_plan TEXT,
 affected_components TEXT,
 before_state TEXT,
 after_state TEXT,
 FOREIGN KEY (model_id) REFERENCES models (model_id)
 )
 ''')

 # Performance metrics table
 cursor.execute('''
 CREATE TABLE IF NOT EXISTS performance_metrics (
 metric_id TEXT PRIMARY KEY,
 model_id TEXT NOT NULL,
 measurement_date DATETIME NOT NULL,
 accuracy_score REAL,
 precision_score REAL,
 recall_score REAL,
 f1_score REAL,
 execution_time REAL,
 memory_usage REAL,
 data_quality_score REAL,
 custom_metrics TEXT,
 FOREIGN KEY (model_id) REFERENCES models (model_id)
 )
 ''')

 # Model usage log table
 cursor.execute('''
 CREATE TABLE IF NOT EXISTS model_usage_log (
 log_id TEXT PRIMARY KEY,
 model_id TEXT NOT NULL,
 usage_date DATETIME NOT NULL,
 user_id TEXT NOT NULL,
 operation TEXT NOT NULL,
 parameters TEXT,
 execution_time REAL,
 success BOOLEAN,
 error_message TEXT,
 FOREIGN KEY (model_id) REFERENCES models (model_id)
 )
 ''')

 conn.commit()
 conn.close()
 logger.info("Model documentation database initialized")

 except Exception as e:
 logger.error(f"Error initializing database: {e}")
 raise

 def save_model_documentation(self, doc: ModelDocumentation) -> bool:
 """Save model documentation to database"""
 try:
 conn = sqlite3.connect(self.db_path)
 cursor = conn.cursor()

 # Convert complex objects to JSON
 tags_json = json.dumps(doc.metadata.tags)
 parameters_json = json.dumps(asdict(doc.parameters))
 assumptions_json = json.dumps(doc.assumptions)
 limitations_json = json.dumps(doc.limitations)
 data_requirements_json = json.dumps(doc.data_requirements)
 performance_benchmarks_json = json.dumps(doc.performance_benchmarks)

 # Insert or update model
 cursor.execute('''
 INSERT OR REPLACE INTO models (
 model_id, model_name, model_type, version, status, description,
 owner, created_date, last_modified, production_date, retirement_date,
 business_justification, regulatory_approval, model_complexity,
 materiality, tags, parameters, methodology, assumptions,
 limitations, data_requirements, performance_benchmarks,
 regulatory_notes, technical_specification, user_guide
 ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
 ''', (
 doc.metadata.model_id,
 doc.metadata.model_name,
 doc.metadata.model_type.value,
 doc.metadata.version,
 doc.metadata.status.value,
 doc.metadata.description,
 doc.metadata.owner,
 doc.metadata.created_date.isoformat(),
 doc.metadata.last_modified.isoformat(),
 doc.metadata.production_date.isoformat() if doc.metadata.production_date else None,
 doc.metadata.retirement_date.isoformat() if doc.metadata.retirement_date else None,
 doc.metadata.business_justification,
 doc.metadata.regulatory_approval,
 doc.metadata.model_complexity,
 doc.metadata.materiality,
 tags_json,
 parameters_json,
 doc.methodology,
 assumptions_json,
 limitations_json,
 data_requirements_json,
 performance_benchmarks_json,
 doc.regulatory_notes,
 doc.technical_specification,
 doc.user_guide
 ))

 # Save validation history
 for validation in doc.validation_history:
 self._save_validation_result(cursor, validation)

 # Save change history
 for change in doc.change_history:
 self._save_change_record(cursor, change)

 # Save performance history
 for performance in doc.performance_history:
 self._save_performance_metrics(cursor, performance)

 conn.commit()
 conn.close()
 return True

 except Exception as e:
 logger.error(f"Error saving model documentation: {e}")
 return False

 def _save_validation_result(self, cursor, validation: ValidationResult):
 """Save validation result"""
 cursor.execute('''
 INSERT OR REPLACE INTO validation_results (
 validation_id, model_id, validation_date, validator_name,
 validation_type, passed, score, findings, recommendations,
 next_validation_date, validation_data
 ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
 ''', (
 validation.validation_id,
 validation.model_id,
 validation.validation_date.isoformat(),
 validation.validator_name,
 validation.validation_type,
 validation.passed,
 validation.score,
 json.dumps(validation.findings),
 json.dumps(validation.recommendations),
 validation.next_validation_date.isoformat(),
 json.dumps(validation.validation_data)
 ))

 def _save_change_record(self, cursor, change: ChangeRecord):
 """Save change record"""
 cursor.execute('''
 INSERT OR REPLACE INTO change_records (
 change_id, model_id, change_type, change_date, changed_by,
 description, impact_assessment, approval_required, approved_by,
 approval_date, rollback_plan, affected_components, before_state, after_state
 ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
 ''', (
 change.change_id,
 change.model_id,
 change.change_type.value,
 change.change_date.isoformat(),
 change.changed_by,
 change.description,
 change.impact_assessment,
 change.approval_required,
 change.approved_by,
 change.approval_date.isoformat() if change.approval_date else None,
 change.rollback_plan,
 json.dumps(change.affected_components),
 json.dumps(change.before_state),
 json.dumps(change.after_state)
 ))

 def _save_performance_metrics(self, cursor, performance: PerformanceMetrics):
 """Save performance metrics"""
 cursor.execute('''
 INSERT OR REPLACE INTO performance_metrics (
 metric_id, model_id, measurement_date, accuracy_score,
 precision_score, recall_score, f1_score, execution_time,
 memory_usage, data_quality_score, custom_metrics
 ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
 ''', (
 performance.metric_id,
 performance.model_id,
 performance.measurement_date.isoformat(),
 performance.accuracy_score,
 performance.precision_score,
 performance.recall_score,
 performance.f1_score,
 performance.execution_time,
 performance.memory_usage,
 performance.data_quality_score,
 json.dumps(performance.custom_metrics)
 ))

 def get_model_documentation(self, model_id: str) -> Optional[ModelDocumentation]:
 """Retrieve model documentation"""
 try:
 conn = sqlite3.connect(self.db_path)
 cursor = conn.cursor()

 # Get model data
 cursor.execute('SELECT * FROM models WHERE model_id = ?', (model_id,))
 model_row = cursor.fetchone()

 if not model_row:
 return None

 # Parse model data
 columns = [desc[0] for desc in cursor.description]
 model_data = dict(zip(columns, model_row))

 # Create metadata
 metadata = ModelMetadata(
 model_id=model_data['model_id'],
 model_name=model_data['model_name'],
 model_type=ModelType(model_data['model_type']),
 version=model_data['version'],
 status=ModelStatus(model_data['status']),
 description=model_data['description'],
 owner=model_data['owner'],
 created_date=datetime.fromisoformat(model_data['created_date']),
 last_modified=datetime.fromisoformat(model_data['last_modified']),
 production_date=datetime.fromisoformat(model_data['production_date']) if model_data['production_date'] else None,
 retirement_date=datetime.fromisoformat(model_data['retirement_date']) if model_data['retirement_date'] else None,
 business_justification=model_data['business_justification'] or "",
 regulatory_approval=bool(model_data['regulatory_approval']),
 model_complexity=model_data['model_complexity'],
 materiality=model_data['materiality'],
 tags=json.loads(model_data['tags']) if model_data['tags'] else []
 )

 # Parse parameters
 parameters_data = json.loads(model_data['parameters']) if model_data['parameters'] else {}
 parameters = ModelParameters(**parameters_data)

 # Get validation history
 cursor.execute('SELECT * FROM validation_results WHERE model_id = ? ORDER BY validation_date DESC', (model_id,))
 validation_rows = cursor.fetchall()
 validation_columns = [desc[0] for desc in cursor.description]
 validation_history = []

 for row in validation_rows:
 val_data = dict(zip(validation_columns, row))
 validation = ValidationResult(
 validation_id=val_data['validation_id'],
 model_id=val_data['model_id'],
 validation_date=datetime.fromisoformat(val_data['validation_date']),
 validator_name=val_data['validator_name'],
 validation_type=val_data['validation_type'],
 passed=bool(val_data['passed']),
 score=float(val_data['score']),
 findings=json.loads(val_data['findings']) if val_data['findings'] else [],
 recommendations=json.loads(val_data['recommendations']) if val_data['recommendations'] else [],
 next_validation_date=datetime.fromisoformat(val_data['next_validation_date']),
 validation_data=json.loads(val_data['validation_data']) if val_data['validation_data'] else {}
 )
 validation_history.append(validation)

 # Get change history
 cursor.execute('SELECT * FROM change_records WHERE model_id = ? ORDER BY change_date DESC', (model_id,))
 change_rows = cursor.fetchall()
 change_columns = [desc[0] for desc in cursor.description]
 change_history = []

 for row in change_rows:
 change_data = dict(zip(change_columns, row))
 change = ChangeRecord(
 change_id=change_data['change_id'],
 model_id=change_data['model_id'],
 change_type=ChangeType(change_data['change_type']),
 change_date=datetime.fromisoformat(change_data['change_date']),
 changed_by=change_data['changed_by'],
 description=change_data['description'],
 impact_assessment=change_data['impact_assessment'] or "",
 approval_required=bool(change_data['approval_required']),
 approved_by=change_data['approved_by'],
 approval_date=datetime.fromisoformat(change_data['approval_date']) if change_data['approval_date'] else None,
 rollback_plan=change_data['rollback_plan'] or "",
 affected_components=json.loads(change_data['affected_components']) if change_data['affected_components'] else [],
 before_state=json.loads(change_data['before_state']) if change_data['before_state'] else {},
 after_state=json.loads(change_data['after_state']) if change_data['after_state'] else {}
 )
 change_history.append(change)

 # Get performance history
 cursor.execute('SELECT * FROM performance_metrics WHERE model_id = ? ORDER BY measurement_date DESC', (model_id,))
 perf_rows = cursor.fetchall()
 perf_columns = [desc[0] for desc in cursor.description]
 performance_history = []

 for row in perf_rows:
 perf_data = dict(zip(perf_columns, row))
 performance = PerformanceMetrics(
 metric_id=perf_data['metric_id'],
 model_id=perf_data['model_id'],
 measurement_date=datetime.fromisoformat(perf_data['measurement_date']),
 accuracy_score=float(perf_data['accuracy_score'] or 0),
 precision_score=float(perf_data['precision_score'] or 0),
 recall_score=float(perf_data['recall_score'] or 0),
 f1_score=float(perf_data['f1_score'] or 0),
 execution_time=float(perf_data['execution_time'] or 0),
 memory_usage=float(perf_data['memory_usage'] or 0),
 data_quality_score=float(perf_data['data_quality_score'] or 0),
 custom_metrics=json.loads(perf_data['custom_metrics']) if perf_data['custom_metrics'] else {}
 )
 performance_history.append(performance)

 conn.close()

 # Create documentation object
 documentation = ModelDocumentation(
 metadata=metadata,
 parameters=parameters,
 methodology=model_data['methodology'] or "",
 assumptions=json.loads(model_data['assumptions']) if model_data['assumptions'] else [],
 limitations=json.loads(model_data['limitations']) if model_data['limitations'] else [],
 data_requirements=json.loads(model_data['data_requirements']) if model_data['data_requirements'] else {},
 performance_benchmarks=json.loads(model_data['performance_benchmarks']) if model_data['performance_benchmarks'] else {},
 validation_history=validation_history,
 change_history=change_history,
 performance_history=performance_history,
 regulatory_notes=model_data['regulatory_notes'] or "",
 technical_specification=model_data['technical_specification'] or "",
 user_guide=model_data['user_guide'] or ""
 )

 return documentation

 except Exception as e:
 logger.error(f"Error retrieving model documentation: {e}")
 return None

 def log_model_usage(self, model_id: str, user_id: str, operation: str,
 parameters: Dict[str, Any], execution_time: float,
 success: bool, error_message: str = ""):
 """Log model usage for audit trail"""
 try:
 conn = sqlite3.connect(self.db_path)
 cursor = conn.cursor()

 log_id = str(uuid.uuid4())
 cursor.execute('''
 INSERT INTO model_usage_log (
 log_id, model_id, usage_date, user_id, operation,
 parameters, execution_time, success, error_message
 ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
 ''', (
 log_id,
 model_id,
 datetime.now().isoformat(),
 user_id,
 operation,
 json.dumps(parameters),
 execution_time,
 success,
 error_message
 ))

 conn.commit()
 conn.close()

 except Exception as e:
 logger.error(f"Error logging model usage: {e}")

class ModelGovernanceManager:
 """Model governance and lifecycle management"""

 def __init__(self, db: ModelDocumentationDB):
 self.db = db
 self.approval_workflows = {}

 def create_model_documentation(self, model_name: str, model_type: ModelType,
 owner: str, description: str) -> ModelDocumentation:
 """Create new model documentation"""
 model_id = str(uuid.uuid4())

 metadata = ModelMetadata(
 model_id=model_id,
 model_name=model_name,
 model_type=model_type,
 version="1.0.0",
 status=ModelStatus.DEVELOPMENT,
 description=description,
 owner=owner,
 created_date=datetime.now(),
 last_modified=datetime.now(),
 business_justification="Initial model development",
 tags=[]
 )

 parameters = ModelParameters()

 # Initial change record
 change_record = ChangeRecord(
 change_id=str(uuid.uuid4()),
 model_id=model_id,
 change_type=ChangeType.CREATED,
 change_date=datetime.now(),
 changed_by=owner,
 description="Initial model creation",
 impact_assessment="New model development",
 approval_required=False
 )

 documentation = ModelDocumentation(
 metadata=metadata,
 parameters=parameters,
 methodology="",
 assumptions=[],
 limitations=[],
 data_requirements={},
 performance_benchmarks={},
 validation_history=[],
 change_history=[change_record],
 performance_history=[]
 )

 self.db.save_model_documentation(documentation)
 return documentation

 def update_model_status(self, model_id: str, new_status: ModelStatus,
 changed_by: str, reason: str) -> bool:
 """Update model status with proper governance"""
 try:
 doc = self.db.get_model_documentation(model_id)
 if not doc:
 return False

 old_status = doc.metadata.status
 doc.metadata.status = new_status
 doc.metadata.last_modified = datetime.now()

 # Create change record
 change_record = ChangeRecord(
 change_id=str(uuid.uuid4()),
 model_id=model_id,
 change_type=ChangeType.UPDATED,
 change_date=datetime.now(),
 changed_by=changed_by,
 description=f"Status changed from {old_status.value} to {new_status.value}",
 impact_assessment=reason,
 approval_required=new_status in [ModelStatus.PRODUCTION, ModelStatus.RETIRED],
 before_state={'status': old_status.value},
 after_state={'status': new_status.value}
 )

 doc.change_history.append(change_record)
 return self.db.save_model_documentation(doc)

 except Exception as e:
 logger.error(f"Error updating model status: {e}")
 return False

 def add_validation_result(self, model_id: str, validator_name: str,
 validation_type: str, passed: bool, score: float,
 findings: List[str], recommendations: List[str]) -> bool:
 """Add validation result to model documentation"""
 try:
 doc = self.db.get_model_documentation(model_id)
 if not doc:
 return False

 validation = ValidationResult(
 validation_id=str(uuid.uuid4()),
 model_id=model_id,
 validation_date=datetime.now(),
 validator_name=validator_name,
 validation_type=validation_type,
 passed=passed,
 score=score,
 findings=findings,
 recommendations=recommendations,
 next_validation_date=datetime.now() + timedelta(days=90) # Quarterly validation
 )

 doc.validation_history.append(validation)
 doc.metadata.last_modified = datetime.now()

 # Create change record for validation
 change_record = ChangeRecord(
 change_id=str(uuid.uuid4()),
 model_id=model_id,
 change_type=ChangeType.VALIDATED,
 change_date=datetime.now(),
 changed_by=validator_name,
 description=f"Model validation: {validation_type}",
 impact_assessment=f"Validation {'passed' if passed else 'failed'} with score {score}",
 approval_required=False
 )

 doc.change_history.append(change_record)
 return self.db.save_model_documentation(doc)

 except Exception as e:
 logger.error(f"Error adding validation result: {e}")
 return False

 def track_model_performance(self, model_id: str, performance_metrics: Dict[str, float]) -> bool:
 """Track model performance metrics"""
 try:
 doc = self.db.get_model_documentation(model_id)
 if not doc:
 return False

 performance = PerformanceMetrics(
 metric_id=str(uuid.uuid4()),
 model_id=model_id,
 measurement_date=datetime.now(),
 accuracy_score=performance_metrics.get('accuracy', 0.0),
 precision_score=performance_metrics.get('precision', 0.0),
 recall_score=performance_metrics.get('recall', 0.0),
 f1_score=performance_metrics.get('f1_score', 0.0),
 execution_time=performance_metrics.get('execution_time', 0.0),
 memory_usage=performance_metrics.get('memory_usage', 0.0),
 data_quality_score=performance_metrics.get('data_quality', 0.0),
 custom_metrics={k: v for k, v in performance_metrics.items()
 if k not in ['accuracy', 'precision', 'recall', 'f1_score',
 'execution_time', 'memory_usage', 'data_quality']}
 )

 doc.performance_history.append(performance)
 doc.metadata.last_modified = datetime.now()

 return self.db.save_model_documentation(doc)

 except Exception as e:
 logger.error(f"Error tracking model performance: {e}")
 return False

 def get_models_requiring_validation(self) -> List[str]:
 """Get models that require validation"""
 try:
 conn = sqlite3.connect(self.db.db_path)
 cursor = conn.cursor()

 # Find models that haven't been validated recently
 cursor.execute('''
 SELECT DISTINCT m.model_id, m.model_name
 FROM models m
 LEFT JOIN validation_results v ON m.model_id = v.model_id
 WHERE m.status = 'PRODUCTION'
 AND (v.next_validation_date IS NULL OR v.next_validation_date <= ?)
 ORDER BY m.model_name
 ''', (datetime.now().isoformat(),))

 results = cursor.fetchall()
 conn.close()

 return [row[0] for row in results]

 except Exception as e:
 logger.error(f"Error getting models requiring validation: {e}")
 return []

 def generate_model_inventory_report(self) -> pd.DataFrame:
 """Generate model inventory report"""
 try:
 conn = sqlite3.connect(self.db.db_path)

 query = '''
 SELECT
 model_id,
 model_name,
 model_type,
 version,
 status,
 owner,
 created_date,
 last_modified,
 model_complexity,
 materiality,
 regulatory_approval
 FROM models
 ORDER BY model_name
 '''

 df = pd.read_sql_query(query, conn)
 conn.close()

 return df

 except Exception as e:
 logger.error(f"Error generating model inventory report: {e}")
 return pd.DataFrame()

 def generate_validation_status_report(self) -> pd.DataFrame:
 """Generate validation status report"""
 try:
 conn = sqlite3.connect(self.db.db_path)

 query = '''
 SELECT
 m.model_id,
 m.model_name,
 m.status,
 v.validation_date,
 v.validation_type,
 v.passed,
 v.score,
 v.next_validation_date,
 CASE
 WHEN v.next_validation_date <= datetime('now') THEN 'OVERDUE'
 WHEN v.next_validation_date <= datetime('now', '+30 days') THEN 'DUE_SOON'
 ELSE 'CURRENT'
 END as validation_status
 FROM models m
 LEFT JOIN (
 SELECT model_id, validation_date, validation_type, passed, score, next_validation_date,
 ROW_NUMBER() OVER (PARTITION BY model_id ORDER BY validation_date DESC) as rn
 FROM validation_results
 ) v ON m.model_id = v.model_id AND v.rn = 1
 ORDER BY m.model_name
 '''

 df = pd.read_sql_query(query, conn)
 conn.close()

 return df

 except Exception as e:
 logger.error(f"Error generating validation status report: {e}")
 return pd.DataFrame()

class ModelDocumentationReporter:
 """Generate various model documentation reports"""

 def __init__(self, governance_manager: ModelGovernanceManager):
 self.governance_manager = governance_manager

 def generate_model_summary_report(self, model_id: str) -> str:
 """Generate model summary report"""
 doc = self.governance_manager.db.get_model_documentation(model_id)
 if not doc:
 return "Model not found"

 report = f"""
MODEL SUMMARY REPORT
===================

Model Information:
- ID: {doc.metadata.model_id}
- Name: {doc.metadata.model_name}
- Type: {doc.metadata.model_type.value}
- Version: {doc.metadata.version}
- Status: {doc.metadata.status.value}
- Owner: {doc.metadata.owner}
- Complexity: {doc.metadata.model_complexity}
- Materiality: {doc.metadata.materiality}

Description:
{doc.metadata.description}

Business Justification:
{doc.metadata.business_justification}

Key Dates:
- Created: {doc.metadata.created_date.strftime('%Y-%m-%d')}
- Last Modified: {doc.metadata.last_modified.strftime('%Y-%m-%d')}
- Production: {doc.metadata.production_date.strftime('%Y-%m-%d') if doc.metadata.production_date else 'Not in production'}

Regulatory Information:
- Regulatory Approval: {'Yes' if doc.metadata.regulatory_approval else 'No'}
- Regulatory Notes: {doc.regulatory_notes or 'None'}

Parameters:
- Confidence Level: {doc.parameters.confidence_level}
- Time Horizon: {doc.parameters.time_horizon} day(s)
- Simulation Count: {doc.parameters.simulation_count:,}
- Lookback Window: {doc.parameters.lookback_window} day(s)

Methodology:
{doc.methodology or 'Not documented'}

Key Assumptions:
"""
 for i, assumption in enumerate(doc.assumptions, 1):
 report += f"{i}. {assumption}\n"

 if not doc.assumptions:
 report += "None documented\n"

 report += f"""
Known Limitations:
"""
 for i, limitation in enumerate(doc.limitations, 1):
 report += f"{i}. {limitation}\n"

 if not doc.limitations:
 report += "None documented\n"

 # Latest validation
 if doc.validation_history:
 latest_validation = doc.validation_history[0]
 report += f"""
Latest Validation:
- Date: {latest_validation.validation_date.strftime('%Y-%m-%d')}
- Type: {latest_validation.validation_type}
- Result: {'PASSED' if latest_validation.passed else 'FAILED'}
- Score: {latest_validation.score:.2f}
- Next Due: {latest_validation.next_validation_date.strftime('%Y-%m-%d')}
"""

 # Performance summary
 if doc.performance_history:
 latest_performance = doc.performance_history[0]
 report += f"""
Latest Performance Metrics:
- Accuracy: {latest_performance.accuracy_score:.3f}
- Execution Time: {latest_performance.execution_time:.3f}s
- Memory Usage: {latest_performance.memory_usage:.1f}MB
- Data Quality Score: {latest_performance.data_quality_score:.3f}
"""

 return report

 def generate_change_log_report(self, model_id: str) -> str:
 """Generate change log report"""
 doc = self.governance_manager.db.get_model_documentation(model_id)
 if not doc:
 return "Model not found"

 report = f"""
CHANGE LOG REPORT - {doc.metadata.model_name}
{'=' * (25 + len(doc.metadata.model_name))}

"""
 for change in doc.change_history:
 approval_status = ""
 if change.approval_required:
 if change.approved_by:
 approval_status = f" (Approved by {change.approved_by} on {change.approval_date.strftime('%Y-%m-%d')})"
 else:
 approval_status = " (PENDING APPROVAL)"

 report += f"""
Change ID: {change.change_id}
Date: {change.change_date.strftime('%Y-%m-%d %H:%M:%S')}
Type: {change.change_type.value}
Changed By: {change.changed_by}{approval_status}

Description:
{change.description}

Impact Assessment:
{change.impact_assessment}

Affected Components:
{', '.join(change.affected_components) if change.affected_components else 'None specified'}

{'=' * 50}
"""

 return report

 def generate_compliance_report(self) -> str:
 """Generate regulatory compliance report"""
 inventory_df = self.governance_manager.generate_model_inventory_report()
 validation_df = self.governance_manager.generate_validation_status_report()

 if inventory_df.empty:
 return "No models found in inventory"

 total_models = len(inventory_df)
 production_models = len(inventory_df[inventory_df['status'] == 'PRODUCTION'])
 approved_models = len(inventory_df[inventory_df['regulatory_approval'] == True])

 # Validation status counts
 overdue_validations = len(validation_df[validation_df['validation_status'] == 'OVERDUE'])
 due_soon_validations = len(validation_df[validation_df['validation_status'] == 'DUE_SOON'])

 report = f"""
REGULATORY COMPLIANCE REPORT
============================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

MODEL INVENTORY SUMMARY:
- Total Models: {total_models}
- Production Models: {production_models}
- Regulatory Approved: {approved_models}
- Approval Rate: {(approved_models/total_models*100):.1f}%

VALIDATION STATUS:
- Overdue Validations: {overdue_validations}
- Due Soon (30 days): {due_soon_validations}
- Compliance Rate: {((total_models-overdue_validations)/total_models*100):.1f}%

MODELS BY COMPLEXITY:
"""
 complexity_counts = inventory_df['model_complexity'].value_counts()
 for complexity, count in complexity_counts.items():
 report += f"- {complexity}: {count}\n"

 report += f"""
MODELS BY MATERIALITY:
"""
 materiality_counts = inventory_df['materiality'].value_counts()
 for materiality, count in materiality_counts.items():
 report += f"- {materiality}: {count}\n"

 if overdue_validations > 0:
 report += f"""
OVERDUE VALIDATIONS:
"""
 overdue_models = validation_df[validation_df['validation_status'] == 'OVERDUE']
 for _, model in overdue_models.iterrows():
 report += f"- {model['model_name']} (Due: {model['next_validation_date']})\n"

 return report