"""
Data Quality Validation and Cleansing Framework
==============================================

Comprehensive data quality assurance tools for financial data validation,
outlier detection, missing data handling, and data integrity checks.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
import warnings
from abc import ABC, abstractmethod
import json
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)

class DataQualityIssue(Enum):
 """Types of data quality issues"""
 MISSING_VALUES = "MISSING_VALUES"
 OUTLIERS = "OUTLIERS"
 DUPLICATES = "DUPLICATES"
 INCONSISTENT_FORMAT = "INCONSISTENT_FORMAT"
 INVALID_RANGE = "INVALID_RANGE"
 TEMPORAL_INCONSISTENCY = "TEMPORAL_INCONSISTENCY"
 STATISTICAL_ANOMALY = "STATISTICAL_ANOMALY"
 SCHEMA_VIOLATION = "SCHEMA_VIOLATION"

class Severity(Enum):
 """Issue severity levels"""
 CRITICAL = "CRITICAL"
 HIGH = "HIGH"
 MEDIUM = "MEDIUM"
 LOW = "LOW"
 INFO = "INFO"

@dataclass
class DataQualityResult:
 """Data quality check result"""
 check_name: str
 issue_type: DataQualityIssue
 severity: Severity
 passed: bool
 details: Dict[str, Any]
 affected_rows: Optional[List[int]] = None
 affected_columns: Optional[List[str]] = None
 recommendations: List[str] = field(default_factory=list)
 timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class DataQualityReport:
 """Comprehensive data quality report"""
 dataset_name: str
 total_rows: int
 total_columns: int
 checks_performed: int
 issues_found: int
 critical_issues: int
 high_issues: int
 medium_issues: int
 low_issues: int
 overall_score: float
 results: List[DataQualityResult]
 summary_statistics: Dict[str, Any]
 timestamp: datetime = field(default_factory=datetime.now)

class BaseDataQualityCheck(ABC):
 """Base class for data quality checks"""

 def __init__(self, name: str, description: str, severity: Severity = Severity.MEDIUM):
 self.name = name
 self.description = description
 self.severity = severity

 @abstractmethod
 def check(self, data: pd.DataFrame, **kwargs) -> DataQualityResult:
 """Perform data quality check"""
 pass

class MissingValueCheck(BaseDataQualityCheck):
 """Check for missing values"""

 def __init__(self, threshold: float = 0.05):
 super().__init__("Missing Value Check", "Check for missing values in dataset")
 self.threshold = threshold

 def check(self, data: pd.DataFrame, **kwargs) -> DataQualityResult:
 """Check for missing values"""
 try:
 missing_counts = data.isnull().sum()
 missing_percentages = missing_counts / len(data)

 total_missing = missing_counts.sum()
 total_cells = data.size
 overall_missing_rate = total_missing / total_cells

 # Identify columns with high missing rates
 high_missing_columns = missing_percentages[missing_percentages > self.threshold].index.tolist()

 # Determine severity
 if overall_missing_rate > 0.2:
 severity = Severity.CRITICAL
 elif overall_missing_rate > 0.1:
 severity = Severity.HIGH
 elif overall_missing_rate > self.threshold:
 severity = Severity.MEDIUM
 else:
 severity = Severity.LOW

 passed = overall_missing_rate <= self.threshold

 recommendations = []
 if not passed:
 if overall_missing_rate > 0.2:
 recommendations.append("Critical missing data - consider data source quality")
 recommendations.append("Implement robust data collection procedures")

 recommendations.append("Consider imputation strategies for missing values")
 recommendations.append("Evaluate impact of missing data on model performance")

 if high_missing_columns:
 recommendations.append(f"Consider removing columns: {high_missing_columns}")

 return DataQualityResult(
 check_name=self.name,
 issue_type=DataQualityIssue.MISSING_VALUES,
 severity=severity,
 passed=passed,
 details={
 'total_missing_values': int(total_missing),
 'missing_rate': float(overall_missing_rate),
 'threshold': self.threshold,
 'missing_by_column': missing_counts.to_dict(),
 'missing_percentages': missing_percentages.to_dict(),
 'high_missing_columns': high_missing_columns
 },
 affected_columns=high_missing_columns,
 recommendations=recommendations
 )

 except Exception as e:
 logger.error(f"Error in missing value check: {e}")
 return DataQualityResult(
 check_name=self.name,
 issue_type=DataQualityIssue.MISSING_VALUES,
 severity=Severity.CRITICAL,
 passed=False,
 details={'error': str(e)}
 )

class OutlierDetectionCheck(BaseDataQualityCheck):
 """Detect statistical outliers"""

 def __init__(self, method: str = 'iqr', threshold: float = 0.05):
 super().__init__("Outlier Detection", "Detect statistical outliers in numerical data")
 self.method = method # 'iqr', 'zscore', 'isolation_forest'
 self.threshold = threshold

 def check(self, data: pd.DataFrame, **kwargs) -> DataQualityResult:
 """Check for outliers"""
 try:
 numeric_columns = data.select_dtypes(include=[np.number]).columns
 outlier_results = {}
 total_outliers = 0
 affected_columns = []

 for column in numeric_columns:
 column_data = data[column].dropna()

 if len(column_data) == 0:
 continue

 if self.method == 'iqr':
 outliers = self._detect_iqr_outliers(column_data)
 elif self.method == 'zscore':
 outliers = self._detect_zscore_outliers(column_data)
 elif self.method == 'isolation_forest':
 outliers = self._detect_isolation_forest_outliers(column_data)
 else:
 outliers = pd.Series(dtype=bool)

 outlier_count = outliers.sum()
 outlier_rate = outlier_count / len(column_data)

 outlier_results[column] = {
 'count': int(outlier_count),
 'rate': float(outlier_rate),
 'indices': outliers[outliers].index.tolist()
 }

 total_outliers += outlier_count

 if outlier_rate > self.threshold:
 affected_columns.append(column)

 overall_outlier_rate = total_outliers / len(data) if len(data) > 0 else 0

 # Determine severity
 if overall_outlier_rate > 0.1:
 severity = Severity.HIGH
 elif overall_outlier_rate > 0.05:
 severity = Severity.MEDIUM
 else:
 severity = Severity.LOW

 passed = overall_outlier_rate <= self.threshold

 recommendations = []
 if not passed:
 recommendations.append("Investigate outliers for data quality issues")
 recommendations.append("Consider robust statistical methods")
 recommendations.append("Validate outliers with domain experts")

 if len(affected_columns) > 0:
 recommendations.append(f"Focus on columns: {affected_columns}")

 return DataQualityResult(
 check_name=self.name,
 issue_type=DataQualityIssue.OUTLIERS,
 severity=severity,
 passed=passed,
 details={
 'method': self.method,
 'total_outliers': int(total_outliers),
 'outlier_rate': float(overall_outlier_rate),
 'threshold': self.threshold,
 'outliers_by_column': outlier_results,
 'numeric_columns_checked': list(numeric_columns)
 },
 affected_columns=affected_columns,
 recommendations=recommendations
 )

 except Exception as e:
 logger.error(f"Error in outlier detection: {e}")
 return DataQualityResult(
 check_name=self.name,
 issue_type=DataQualityIssue.OUTLIERS,
 severity=Severity.CRITICAL,
 passed=False,
 details={'error': str(e)}
 )

 def _detect_iqr_outliers(self, data: pd.Series) -> pd.Series:
 """Detect outliers using IQR method"""
 Q1 = data.quantile(0.25)
 Q3 = data.quantile(0.75)
 IQR = Q3 - Q1
 lower_bound = Q1 - 1.5 * IQR
 upper_bound = Q3 + 1.5 * IQR

 return (data < lower_bound) | (data > upper_bound)

 def _detect_zscore_outliers(self, data: pd.Series, threshold: float = 3) -> pd.Series:
 """Detect outliers using Z-score method"""
 z_scores = np.abs((data - data.mean()) / data.std())
 return z_scores > threshold

 def _detect_isolation_forest_outliers(self, data: pd.Series) -> pd.Series:
 """Detect outliers using Isolation Forest"""
 try:
 from sklearn.ensemble import IsolationForest

 if len(data) < 10:
 return pd.Series(False, index=data.index)

 model = IsolationForest(contamination=0.1, random_state=42)
 outliers = model.fit_predict(data.values.reshape(-1, 1))

 return pd.Series(outliers == -1, index=data.index)

 except ImportError:
 logger.warning("scikit-learn not available for isolation forest outliers")
 return self._detect_iqr_outliers(data)

class DuplicateCheck(BaseDataQualityCheck):
 """Check for duplicate records"""

 def __init__(self, subset: Optional[List[str]] = None):
 super().__init__("Duplicate Records Check", "Check for duplicate records")
 self.subset = subset

 def check(self, data: pd.DataFrame, **kwargs) -> DataQualityResult:
 """Check for duplicate records"""
 try:
 if self.subset:
 # Check duplicates based on specific columns
 duplicates = data.duplicated(subset=self.subset, keep='first')
 duplicate_columns = self.subset
 else:
 # Check for completely duplicate rows
 duplicates = data.duplicated(keep='first')
 duplicate_columns = list(data.columns)

 duplicate_count = duplicates.sum()
 duplicate_rate = duplicate_count / len(data) if len(data) > 0 else 0

 # Find duplicate groups
 if self.subset:
 duplicate_groups = data[data.duplicated(subset=self.subset, keep=False)].groupby(self.subset).size()
 else:
 duplicate_groups = data[data.duplicated(keep=False)].groupby(list(data.columns)).size()

 # Determine severity
 if duplicate_rate > 0.1:
 severity = Severity.HIGH
 elif duplicate_rate > 0.05:
 severity = Severity.MEDIUM
 elif duplicate_rate > 0:
 severity = Severity.LOW
 else:
 severity = Severity.INFO

 passed = duplicate_count == 0

 recommendations = []
 if not passed:
 recommendations.append("Remove duplicate records")
 recommendations.append("Investigate data source for duplication causes")
 recommendations.append("Implement unique constraints in data pipeline")

 return DataQualityResult(
 check_name=self.name,
 issue_type=DataQualityIssue.DUPLICATES,
 severity=severity,
 passed=passed,
 details={
 'duplicate_count': int(duplicate_count),
 'duplicate_rate': float(duplicate_rate),
 'subset_columns': self.subset,
 'checked_columns': duplicate_columns,
 'duplicate_groups': duplicate_groups.to_dict() if len(duplicate_groups) > 0 else {}
 },
 affected_rows=duplicates[duplicates].index.tolist(),
 recommendations=recommendations
 )

 except Exception as e:
 logger.error(f"Error in duplicate check: {e}")
 return DataQualityResult(
 check_name=self.name,
 issue_type=DataQualityIssue.DUPLICATES,
 severity=Severity.CRITICAL,
 passed=False,
 details={'error': str(e)}
 )

class RangeValidationCheck(BaseDataQualityCheck):
 """Validate data ranges for numerical columns"""

 def __init__(self, range_rules: Dict[str, Tuple[float, float]]):
 super().__init__("Range Validation", "Validate numerical data ranges")
 self.range_rules = range_rules

 def check(self, data: pd.DataFrame, **kwargs) -> DataQualityResult:
 """Check data ranges"""
 try:
 violations = {}
 total_violations = 0
 affected_columns = []

 for column, (min_val, max_val) in self.range_rules.items():
 if column not in data.columns:
 continue

 column_data = data[column].dropna()

 if len(column_data) == 0:
 continue

 # Check for values outside range
 out_of_range = (column_data < min_val) | (column_data > max_val)
 violation_count = out_of_range.sum()
 violation_rate = violation_count / len(column_data)

 if violation_count > 0:
 violations[column] = {
 'count': int(violation_count),
 'rate': float(violation_rate),
 'min_allowed': min_val,
 'max_allowed': max_val,
 'actual_min': float(column_data.min()),
 'actual_max': float(column_data.max()),
 'violating_indices': out_of_range[out_of_range].index.tolist()
 }

 total_violations += violation_count
 affected_columns.append(column)

 violation_rate = total_violations / len(data) if len(data) > 0 else 0

 # Determine severity
 if violation_rate > 0.05:
 severity = Severity.HIGH
 elif violation_rate > 0.01:
 severity = Severity.MEDIUM
 elif violation_rate > 0:
 severity = Severity.LOW
 else:
 severity = Severity.INFO

 passed = total_violations == 0

 recommendations = []
 if not passed:
 recommendations.append("Investigate values outside expected ranges")
 recommendations.append("Validate data entry procedures")
 recommendations.append("Consider data transformation or capping")

 return DataQualityResult(
 check_name=self.name,
 issue_type=DataQualityIssue.INVALID_RANGE,
 severity=severity,
 passed=passed,
 details={
 'total_violations': int(total_violations),
 'violation_rate': float(violation_rate),
 'range_rules': self.range_rules,
 'violations_by_column': violations
 },
 affected_columns=affected_columns,
 recommendations=recommendations
 )

 except Exception as e:
 logger.error(f"Error in range validation: {e}")
 return DataQualityResult(
 check_name=self.name,
 issue_type=DataQualityIssue.INVALID_RANGE,
 severity=Severity.CRITICAL,
 passed=False,
 details={'error': str(e)}
 )

class TemporalConsistencyCheck(BaseDataQualityCheck):
 """Check temporal consistency for time series data"""

 def __init__(self, date_column: str, frequency: str = 'D'):
 super().__init__("Temporal Consistency", "Check temporal consistency in time series")
 self.date_column = date_column
 self.frequency = frequency

 def check(self, data: pd.DataFrame, **kwargs) -> DataQualityResult:
 """Check temporal consistency"""
 try:
 if self.date_column not in data.columns:
 return DataQualityResult(
 check_name=self.name,
 issue_type=DataQualityIssue.TEMPORAL_INCONSISTENCY,
 severity=Severity.CRITICAL,
 passed=False,
 details={'error': f'Date column {self.date_column} not found'}
 )

 # Convert to datetime if not already
 try:
 dates = pd.to_datetime(data[self.date_column])
 except Exception:
 return DataQualityResult(
 check_name=self.name,
 issue_type=DataQualityIssue.TEMPORAL_INCONSISTENCY,
 severity=Severity.CRITICAL,
 passed=False,
 details={'error': f'Cannot convert {self.date_column} to datetime'}
 )

 # Sort by date
 sorted_dates = dates.sort_values()

 # Check for gaps
 expected_dates = pd.date_range(
 start=sorted_dates.min(),
 end=sorted_dates.max(),
 freq=self.frequency
 )

 missing_dates = expected_dates.difference(sorted_dates)
 missing_count = len(missing_dates)

 # Check for duplicates
 duplicate_dates = dates[dates.duplicated()]
 duplicate_count = len(duplicate_dates)

 # Check chronological order
 chronological_violations = 0
 if not dates.equals(dates.sort_values()):
 chronological_violations = len(data) - len(dates.sort_values())

 total_issues = missing_count + duplicate_count + chronological_violations

 # Determine severity
 if total_issues > len(data) * 0.1:
 severity = Severity.HIGH
 elif total_issues > len(data) * 0.05:
 severity = Severity.MEDIUM
 elif total_issues > 0:
 severity = Severity.LOW
 else:
 severity = Severity.INFO

 passed = total_issues == 0

 recommendations = []
 if not passed:
 if missing_count > 0:
 recommendations.append("Fill missing dates or adjust frequency")
 if duplicate_count > 0:
 recommendations.append("Remove or aggregate duplicate dates")
 if chronological_violations > 0:
 recommendations.append("Sort data chronologically")

 return DataQualityResult(
 check_name=self.name,
 issue_type=DataQualityIssue.TEMPORAL_INCONSISTENCY,
 severity=severity,
 passed=passed,
 details={
 'date_column': self.date_column,
 'frequency': self.frequency,
 'missing_dates_count': missing_count,
 'duplicate_dates_count': duplicate_count,
 'chronological_violations': chronological_violations,
 'total_issues': total_issues,
 'date_range': (sorted_dates.min().isoformat(), sorted_dates.max().isoformat()),
 'missing_dates': missing_dates.strftime('%Y-%m-%d').tolist()[:10] if missing_count > 0 else []
 },
 recommendations=recommendations
 )

 except Exception as e:
 logger.error(f"Error in temporal consistency check: {e}")
 return DataQualityResult(
 check_name=self.name,
 issue_type=DataQualityIssue.TEMPORAL_INCONSISTENCY,
 severity=Severity.CRITICAL,
 passed=False,
 details={'error': str(e)}
 )

class StatisticalAnomalyCheck(BaseDataQualityCheck):
 """Check for statistical anomalies"""

 def __init__(self, threshold_std: float = 4.0):
 super().__init__("Statistical Anomaly Detection", "Detect statistical anomalies")
 self.threshold_std = threshold_std

 def check(self, data: pd.DataFrame, **kwargs) -> DataQualityResult:
 """Check for statistical anomalies"""
 try:
 numeric_columns = data.select_dtypes(include=[np.number]).columns
 anomalies = {}
 total_anomalies = 0

 for column in numeric_columns:
 column_data = data[column].dropna()

 if len(column_data) < 10: # Need sufficient data
 continue

 # Calculate statistics
 mean = column_data.mean()
 std = column_data.std()

 if std == 0: # Constant values
 continue

 # Detect extreme values
 z_scores = np.abs((column_data - mean) / std)
 extreme_values = z_scores > self.threshold_std

 # Check distribution properties
 skewness = column_data.skew()
 kurtosis = column_data.kurtosis()

 # Shapiro-Wilk test for normality (for small samples)
 if len(column_data) <= 5000:
 try:
 from scipy.stats import shapiro
 _, normality_p_value = shapiro(column_data.sample(min(5000, len(column_data))))
 except ImportError:
 normality_p_value = None
 else:
 normality_p_value = None

 anomaly_count = extreme_values.sum()

 if anomaly_count > 0 or abs(skewness) > 2 or abs(kurtosis) > 7:
 anomalies[column] = {
 'extreme_values_count': int(anomaly_count),
 'extreme_values_rate': float(anomaly_count / len(column_data)),
 'mean': float(mean),
 'std': float(std),
 'skewness': float(skewness),
 'kurtosis': float(kurtosis),
 'normality_p_value': float(normality_p_value) if normality_p_value else None,
 'extreme_indices': extreme_values[extreme_values].index.tolist()[:100] # Limit output
 }

 total_anomalies += anomaly_count

 anomaly_rate = total_anomalies / len(data) if len(data) > 0 else 0

 # Determine severity
 if anomaly_rate > 0.05:
 severity = Severity.HIGH
 elif anomaly_rate > 0.01:
 severity = Severity.MEDIUM
 elif len(anomalies) > 0:
 severity = Severity.LOW
 else:
 severity = Severity.INFO

 passed = len(anomalies) == 0

 recommendations = []
 if not passed:
 recommendations.append("Investigate statistical anomalies")
 recommendations.append("Consider data transformation for skewed distributions")
 recommendations.append("Validate extreme values with domain experts")

 return DataQualityResult(
 check_name=self.name,
 issue_type=DataQualityIssue.STATISTICAL_ANOMALY,
 severity=severity,
 passed=passed,
 details={
 'threshold_std': self.threshold_std,
 'total_anomalies': int(total_anomalies),
 'anomaly_rate': float(anomaly_rate),
 'anomalies_by_column': anomalies,
 'columns_checked': list(numeric_columns)
 },
 affected_columns=list(anomalies.keys()),
 recommendations=recommendations
 )

 except Exception as e:
 logger.error(f"Error in statistical anomaly check: {e}")
 return DataQualityResult(
 check_name=self.name,
 issue_type=DataQualityIssue.STATISTICAL_ANOMALY,
 severity=Severity.CRITICAL,
 passed=False,
 details={'error': str(e)}
 )

class DataQualityValidator:
 """Main data quality validation framework"""

 def __init__(self):
 self.checks: List[BaseDataQualityCheck] = []
 self.custom_checks: Dict[str, Callable] = {}

 def add_check(self, check: BaseDataQualityCheck):
 """Add a data quality check"""
 self.checks.append(check)

 def add_custom_check(self, name: str, check_function: Callable):
 """Add custom data quality check function"""
 self.custom_checks[name] = check_function

 def validate(self, data: pd.DataFrame, dataset_name: str = "Unknown") -> DataQualityReport:
 """Run comprehensive data quality validation"""

 start_time = datetime.now()
 results = []

 # Run standard checks
 for check in self.checks:
 try:
 logger.info(f"Running check: {check.name}")
 result = check.check(data)
 results.append(result)
 except Exception as e:
 logger.error(f"Error running check {check.name}: {e}")
 error_result = DataQualityResult(
 check_name=check.name,
 issue_type=DataQualityIssue.SCHEMA_VIOLATION,
 severity=Severity.CRITICAL,
 passed=False,
 details={'error': str(e)}
 )
 results.append(error_result)

 # Run custom checks
 for check_name, check_function in self.custom_checks.items():
 try:
 logger.info(f"Running custom check: {check_name}")
 result = check_function(data)
 if isinstance(result, DataQualityResult):
 results.append(result)
 else:
 logger.warning(f"Custom check {check_name} did not return DataQualityResult")
 except Exception as e:
 logger.error(f"Error running custom check {check_name}: {e}")

 # Calculate summary statistics
 summary_stats = self._calculate_summary_statistics(data)

 # Generate quality score
 overall_score = self._calculate_quality_score(results)

 # Count issues by severity
 issue_counts = self._count_issues_by_severity(results)

 execution_time = (datetime.now() - start_time).total_seconds()

 return DataQualityReport(
 dataset_name=dataset_name,
 total_rows=len(data),
 total_columns=len(data.columns),
 checks_performed=len(results),
 issues_found=len([r for r in results if not r.passed]),
 critical_issues=issue_counts[Severity.CRITICAL],
 high_issues=issue_counts[Severity.HIGH],
 medium_issues=issue_counts[Severity.MEDIUM],
 low_issues=issue_counts[Severity.LOW],
 overall_score=overall_score,
 results=results,
 summary_statistics=summary_stats
 )

 def _calculate_summary_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
 """Calculate summary statistics for the dataset"""

 numeric_columns = data.select_dtypes(include=[np.number]).columns
 categorical_columns = data.select_dtypes(include=['object', 'category']).columns
 datetime_columns = data.select_dtypes(include=['datetime64[ns]']).columns

 stats = {
 'basic_info': {
 'shape': data.shape,
 'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024 / 1024,
 'dtypes': data.dtypes.value_counts().to_dict()
 },
 'numeric_summary': {},
 'categorical_summary': {},
 'datetime_summary': {}
 }

 # Numeric columns summary
 if len(numeric_columns) > 0:
 stats['numeric_summary'] = {
 'count': len(numeric_columns),
 'columns': list(numeric_columns),
 'describe': data[numeric_columns].describe().to_dict()
 }

 # Categorical columns summary
 if len(categorical_columns) > 0:
 categorical_info = {}
 for col in categorical_columns:
 unique_count = data[col].nunique()
 most_frequent = data[col].mode().iloc[0] if len(data[col].mode()) > 0 else None

 categorical_info[col] = {
 'unique_count': unique_count,
 'most_frequent': most_frequent,
 'cardinality_ratio': unique_count / len(data) if len(data) > 0 else 0
 }

 stats['categorical_summary'] = {
 'count': len(categorical_columns),
 'columns': list(categorical_columns),
 'details': categorical_info
 }

 # Datetime columns summary
 if len(datetime_columns) > 0:
 datetime_info = {}
 for col in datetime_columns:
 datetime_info[col] = {
 'min_date': data[col].min().isoformat() if pd.notna(data[col].min()) else None,
 'max_date': data[col].max().isoformat() if pd.notna(data[col].max()) else None,
 'date_range_days': (data[col].max() - data[col].min()).days if pd.notna(data[col].min()) and pd.notna(data[col].max()) else None
 }

 stats['datetime_summary'] = {
 'count': len(datetime_columns),
 'columns': list(datetime_columns),
 'details': datetime_info
 }

 return stats

 def _calculate_quality_score(self, results: List[DataQualityResult]) -> float:
 """Calculate overall data quality score (0-100)"""

 if not results:
 return 100.0

 total_weight = 0
 weighted_score = 0

 severity_weights = {
 Severity.CRITICAL: 4,
 Severity.HIGH: 3,
 Severity.MEDIUM: 2,
 Severity.LOW: 1,
 Severity.INFO: 0.5
 }

 for result in results:
 weight = severity_weights.get(result.severity, 1)
 score = 100 if result.passed else 0

 weighted_score += score * weight
 total_weight += weight

 return weighted_score / total_weight if total_weight > 0 else 100.0

 def _count_issues_by_severity(self, results: List[DataQualityResult]) -> Dict[Severity, int]:
 """Count issues by severity level"""

 counts = {severity: 0 for severity in Severity}

 for result in results:
 if not result.passed:
 counts[result.severity] += 1

 return counts

 def generate_report_html(self, report: DataQualityReport) -> str:
 """Generate HTML report"""

 html = f"""
 <!DOCTYPE html>
 <html>
 <head>
 <title>Data Quality Report - {report.dataset_name}</title>
 <style>
 body {{ font-family: Arial, sans-serif; margin: 20px; }}.header {{ background-color: #f0f8ff; padding: 20px; border-radius: 5px; }}.summary {{ display: flex; justify-content: space-around; margin: 20px 0; }}.metric {{ text-align: center; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }}.critical {{ background-color: #ffebee; border-color: #f44336; }}.high {{ background-color: #fff3e0; border-color: #ff9800; }}.medium {{ background-color: #f3e5f5; border-color: #9c27b0; }}.low {{ background-color: #e8f5e8; border-color: #4caf50; }}.results {{ margin-top: 30px; }}.result-item {{ margin: 10px 0; padding: 15px; border-left: 4px solid #ddd; }}.passed {{ border-left-color: #4caf50; background-color: #f1f8e9; }}.failed {{ border-left-color: #f44336; background-color: #ffebee; }}
 table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
 th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
 th {{ background-color: #f2f2f2; }}
 </style>
 </head>
 <body>
 <div class="header">
 <h1>Data Quality Report</h1>
 <p><strong>Dataset:</strong> {report.dataset_name}</p>
 <p><strong>Generated:</strong> {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
 <p><strong>Overall Quality Score:</strong> {report.overall_score:.1f}/100</p>
 </div>

 <div class="summary">
 <div class="metric">
 <h3>{report.total_rows:,}</h3>
 <p>Total Rows</p>
 </div>
 <div class="metric">
 <h3>{report.total_columns}</h3>
 <p>Total Columns</p>
 </div>
 <div class="metric">
 <h3>{report.checks_performed}</h3>
 <p>Checks Performed</p>
 </div>
 <div class="metric">
 <h3>{report.issues_found}</h3>
 <p>Issues Found</p>
 </div>
 </div>

 <div class="summary">
 <div class="metric critical">
 <h3>{report.critical_issues}</h3>
 <p>Critical Issues</p>
 </div>
 <div class="metric high">
 <h3>{report.high_issues}</h3>
 <p>High Issues</p>
 </div>
 <div class="metric medium">
 <h3>{report.medium_issues}</h3>
 <p>Medium Issues</p>
 </div>
 <div class="metric low">
 <h3>{report.low_issues}</h3>
 <p>Low Issues</p>
 </div>
 </div>

 <div class="results">
 <h2>Detailed Results</h2>
 """

 for result in report.results:
 status_class = "passed" if result.passed else "failed"
 severity_class = result.severity.value.lower()

 html += f"""
 <div class="result-item {status_class}">
 <h3>{result.check_name} - {result.severity.value}</h3>
 <p><strong>Status:</strong> {'PASSED' if result.passed else 'FAILED'}</p>
 <p><strong>Issue Type:</strong> {result.issue_type.value}</p>

 <h4>Details:</h4>
 <table>
 """

 for key, value in result.details.items():
 if isinstance(value, (dict, list)) and len(str(value)) > 100:
 value = f"{str(value)[:100]}..."
 html += f"<tr><td><strong>{key}</strong></td><td>{value}</td></tr>"

 html += "</table>"

 if result.recommendations:
 html += "<h4>Recommendations:</h4><ul>"
 for rec in result.recommendations:
 html += f"<li>{rec}</li>"
 html += "</ul>"

 html += "</div>"

 html += """
 </div>
 </body>
 </html>
 """

 return html

def create_standard_validator() -> DataQualityValidator:
 """Create validator with standard checks"""

 validator = DataQualityValidator()

 # Add standard checks
 validator.add_check(MissingValueCheck(threshold=0.05))
 validator.add_check(OutlierDetectionCheck(method='iqr', threshold=0.05))
 validator.add_check(DuplicateCheck())
 validator.add_check(StatisticalAnomalyCheck(threshold_std=4.0))

 return validator

def create_financial_data_validator() -> DataQualityValidator:
 """Create validator specifically for financial data"""

 validator = DataQualityValidator()

 # Financial data specific checks
 validator.add_check(MissingValueCheck(threshold=0.01)) # Stricter for financial data
 validator.add_check(OutlierDetectionCheck(method='iqr', threshold=0.02))
 validator.add_check(DuplicateCheck())

 # Financial data ranges
 financial_ranges = {
 'price': (0, float('inf')),
 'volume': (0, float('inf')),
 'return': (-1, 5), # -100% to 500% daily return
 'volatility': (0, 5), # 0% to 500% volatility
 'correlation': (-1, 1)
 }
 validator.add_check(RangeValidationCheck(financial_ranges))

 validator.add_check(StatisticalAnomalyCheck(threshold_std=3.0)) # Stricter for financial data

 return validator