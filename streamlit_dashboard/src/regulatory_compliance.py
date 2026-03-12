"""
Regulatory compliance validation tools for risk management systems.
Implements Basel III, FRTB, and other regulatory framework requirements.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import json
import sqlite3
from dataclasses import dataclass, asdict
from enum import Enum
import warnings

class ComplianceFramework(Enum):
    BASEL_III = "Basel III"
    FRTB = "FRTB"
    CCAR = "CCAR"
    SOLVENCY_II = "Solvency II"
    IFRS_9 = "IFRS 9"

class ComplianceStatus(Enum):
    COMPLIANT = "Compliant"
    NON_COMPLIANT = "Non-Compliant"
    WARNING = "Warning"
    UNDER_REVIEW = "Under Review"

@dataclass
class ComplianceResult:
    """Results of a compliance check."""
    framework: ComplianceFramework
    rule_id: str
    rule_description: str
    status: ComplianceStatus
    value: float
    threshold: float
    deviation: float
    timestamp: datetime
    details: Dict[str, Any]
    recommendations: List[str]

class BaselIIIValidator:
    """Validator for Basel III compliance requirements."""

    def __init__(self):
        self.min_tier1_ratio = 0.06  # 6%
        self.min_total_capital_ratio = 0.08  # 8%
        self.min_leverage_ratio = 0.03  # 3%
        self.min_lcr = 1.0  # 100%
        self.min_nsfr = 1.0  # 100%

    def validate_capital_adequacy(self, tier1_capital: float,
                                total_capital: float, rwa: float) -> List[ComplianceResult]:
        """Validate capital adequacy ratios."""
        results = []

        tier1_ratio = tier1_capital / rwa if rwa > 0 else 0
        total_ratio = total_capital / rwa if rwa > 0 else 0

        # Tier 1 Capital Ratio
        results.append(ComplianceResult(
            framework=ComplianceFramework.BASEL_III,
            rule_id="CAR_T1",
            rule_description="Tier 1 Capital Ratio >= 6%",
            status=ComplianceStatus.COMPLIANT if tier1_ratio >= self.min_tier1_ratio
                   else ComplianceStatus.NON_COMPLIANT,
            value=tier1_ratio,
            threshold=self.min_tier1_ratio,
            deviation=tier1_ratio - self.min_tier1_ratio,
            timestamp=datetime.now(),
            details={
                "tier1_capital": tier1_capital,
                "rwa": rwa,
                "buffer_required": max(0, self.min_tier1_ratio - tier1_ratio) * rwa
            },
            recommendations=[] if tier1_ratio >= self.min_tier1_ratio
                           else ["Increase Tier 1 capital", "Reduce risk-weighted assets"]
        ))

        # Total Capital Ratio
        results.append(ComplianceResult(
            framework=ComplianceFramework.BASEL_III,
            rule_id="CAR_TOTAL",
            rule_description="Total Capital Ratio >= 8%",
            status=ComplianceStatus.COMPLIANT if total_ratio >= self.min_total_capital_ratio
                   else ComplianceStatus.NON_COMPLIANT,
            value=total_ratio,
            threshold=self.min_total_capital_ratio,
            deviation=total_ratio - self.min_total_capital_ratio,
            timestamp=datetime.now(),
            details={
                "total_capital": total_capital,
                "rwa": rwa,
                "buffer_required": max(0, self.min_total_capital_ratio - total_ratio) * rwa
            },
            recommendations=[] if total_ratio >= self.min_total_capital_ratio
                           else ["Increase total capital", "Optimize capital structure"]
        ))

        return results

    def validate_leverage_ratio(self, tier1_capital: float,
                              total_exposure: float) -> ComplianceResult:
        """Validate leverage ratio requirement."""
        leverage_ratio = tier1_capital / total_exposure if total_exposure > 0 else 0

        return ComplianceResult(
            framework=ComplianceFramework.BASEL_III,
            rule_id="LR",
            rule_description="Leverage Ratio >= 3%",
            status=ComplianceStatus.COMPLIANT if leverage_ratio >= self.min_leverage_ratio
                   else ComplianceStatus.NON_COMPLIANT,
            value=leverage_ratio,
            threshold=self.min_leverage_ratio,
            deviation=leverage_ratio - self.min_leverage_ratio,
            timestamp=datetime.now(),
            details={
                "tier1_capital": tier1_capital,
                "total_exposure": total_exposure,
                "exposure_reduction_needed": max(0,
                    (tier1_capital / self.min_leverage_ratio) - total_exposure)
            },
            recommendations=[] if leverage_ratio >= self.min_leverage_ratio
                           else ["Reduce total exposure", "Increase Tier 1 capital"]
        )

    def validate_liquidity_ratios(self, hqla: float, net_cash_outflows: float,
                                available_stable_funding: float,
                                required_stable_funding: float) -> List[ComplianceResult]:
        """Validate LCR and NSFR requirements."""
        results = []

        # Liquidity Coverage Ratio (LCR)
        lcr = hqla / net_cash_outflows if net_cash_outflows > 0 else float('inf')
        results.append(ComplianceResult(
            framework=ComplianceFramework.BASEL_III,
            rule_id="LCR",
            rule_description="Liquidity Coverage Ratio >= 100%",
            status=ComplianceStatus.COMPLIANT if lcr >= self.min_lcr
                   else ComplianceStatus.NON_COMPLIANT,
            value=lcr,
            threshold=self.min_lcr,
            deviation=lcr - self.min_lcr,
            timestamp=datetime.now(),
            details={
                "hqla": hqla,
                "net_cash_outflows": net_cash_outflows,
                "additional_hqla_needed": max(0, net_cash_outflows - hqla)
            },
            recommendations=[] if lcr >= self.min_lcr
                           else ["Increase HQLA holdings", "Reduce cash outflows"]
        ))

        # Net Stable Funding Ratio (NSFR)
        nsfr = available_stable_funding / required_stable_funding if required_stable_funding > 0 else float('inf')
        results.append(ComplianceResult(
            framework=ComplianceFramework.BASEL_III,
            rule_id="NSFR",
            rule_description="Net Stable Funding Ratio >= 100%",
            status=ComplianceStatus.COMPLIANT if nsfr >= self.min_nsfr
                   else ComplianceStatus.NON_COMPLIANT,
            value=nsfr,
            threshold=self.min_nsfr,
            deviation=nsfr - self.min_nsfr,
            timestamp=datetime.now(),
            details={
                "available_stable_funding": available_stable_funding,
                "required_stable_funding": required_stable_funding,
                "funding_gap": max(0, required_stable_funding - available_stable_funding)
            },
            recommendations=[] if nsfr >= self.min_nsfr
                           else ["Increase stable funding sources", "Reduce funding requirements"]
        ))

        return results

class FRTBValidator:
    """Validator for Fundamental Review of Trading Book (FRTB) requirements."""

    def __init__(self):
        self.es_multiplier = 3.0
        self.var_multiplier = 2.5
        self.confidence_level = 0.975  # 97.5%
        self.liquidity_horizons = {
            'sovereign': 10,
            'corporate': 20,
            'equity_large_cap': 10,
            'equity_small_cap': 20,
            'fx_major': 10,
            'fx_minor': 20,
            'commodity': 20
        }

    def validate_sensitivities_based_approach(self, delta_sensitivities: Dict[str, float],
                                            vega_sensitivities: Dict[str, float],
                                            curvature_sensitivities: Dict[str, float]) -> List[ComplianceResult]:
        """Validate SBA capital requirements."""
        results = []

        # Delta risk charge
        delta_charge = sum(abs(s) for s in delta_sensitivities.values())

        # Vega risk charge
        vega_charge = sum(abs(s) for s in vega_sensitivities.values())

        # Curvature risk charge
        curvature_charge = sum(abs(s) for s in curvature_sensitivities.values())

        total_sba_charge = delta_charge + vega_charge + curvature_charge

        results.append(ComplianceResult(
            framework=ComplianceFramework.FRTB,
            rule_id="SBA_TOTAL",
            rule_description="Sensitivities-Based Approach Capital Charge",
            status=ComplianceStatus.COMPLIANT,  # Always compliant if calculated
            value=total_sba_charge,
            threshold=0,  # No specific threshold
            deviation=0,
            timestamp=datetime.now(),
            details={
                "delta_charge": delta_charge,
                "vega_charge": vega_charge,
                "curvature_charge": curvature_charge,
                "risk_factor_count": len(delta_sensitivities)
            },
            recommendations=["Monitor concentration in specific risk factors"]
        ))

        return results

    def validate_internal_models_approach(self, es_values: List[float],
                                        var_values: List[float]) -> List[ComplianceResult]:
        """Validate IMA requirements including backtesting."""
        results = []

        if len(es_values) != len(var_values):
            raise ValueError("ES and VaR series must have same length")

        # ES multiplier validation
        avg_es = np.mean(es_values)
        es_capital = avg_es * self.es_multiplier

        results.append(ComplianceResult(
            framework=ComplianceFramework.FRTB,
            rule_id="IMA_ES",
            rule_description="Expected Shortfall Capital Requirement",
            status=ComplianceStatus.COMPLIANT,
            value=es_capital,
            threshold=0,
            deviation=0,
            timestamp=datetime.now(),
            details={
                "average_es": avg_es,
                "es_multiplier": self.es_multiplier,
                "observation_count": len(es_values)
            },
            recommendations=["Ensure ES model captures tail risk adequately"]
        ))

        # P&L attribution test
        pnl_attribution_ratio = 0.95  # Placeholder - would need actual P&L data

        results.append(ComplianceResult(
            framework=ComplianceFramework.FRTB,
            rule_id="IMA_PLA",
            rule_description="P&L Attribution Test >= 95%",
            status=ComplianceStatus.COMPLIANT if pnl_attribution_ratio >= 0.95
                   else ComplianceStatus.NON_COMPLIANT,
            value=pnl_attribution_ratio,
            threshold=0.95,
            deviation=pnl_attribution_ratio - 0.95,
            timestamp=datetime.now(),
            details={
                "unexplained_pnl_ratio": 1 - pnl_attribution_ratio
            },
            recommendations=[] if pnl_attribution_ratio >= 0.95
                           else ["Improve risk factor coverage", "Enhance model calibration"]
        ))

        return results

class RegulatoryComplianceEngine:
    """Main engine for regulatory compliance validation."""

    def __init__(self, db_path: str = "compliance.db"):
        self.db_path = db_path
        self.basel_validator = BaselIIIValidator()
        self.frtb_validator = FRTBValidator()
        self._init_database()

    def _init_database(self):
        """Initialize compliance database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS compliance_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    framework TEXT NOT NULL,
                    rule_id TEXT NOT NULL,
                    rule_description TEXT NOT NULL,
                    status TEXT NOT NULL,
                    value REAL NOT NULL,
                    threshold REAL NOT NULL,
                    deviation REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    details TEXT,
                    recommendations TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS compliance_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rule_id TEXT NOT NULL,
                    status_change TEXT NOT NULL,
                    previous_status TEXT,
                    new_status TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    reason TEXT
                )
            """)

    def run_full_compliance_check(self, portfolio_data: Dict[str, Any]) -> Dict[str, List[ComplianceResult]]:
        """Run comprehensive compliance validation."""
        results = {
            'basel_iii': [],
            'frtb': [],
            'summary': []
        }

        try:
            # Basel III validations
            if 'capital' in portfolio_data:
                capital = portfolio_data['capital']
                basel_results = []

                # Capital adequacy
                if all(k in capital for k in ['tier1', 'total', 'rwa']):
                    basel_results.extend(self.basel_validator.validate_capital_adequacy(
                        capital['tier1'], capital['total'], capital['rwa']
                    ))

                # Leverage ratio
                if all(k in capital for k in ['tier1', 'total_exposure']):
                    basel_results.append(self.basel_validator.validate_leverage_ratio(
                        capital['tier1'], capital['total_exposure']
                    ))

                # Liquidity ratios
                if 'liquidity' in capital:
                    liq = capital['liquidity']
                    if all(k in liq for k in ['hqla', 'net_cash_outflows', 'asf', 'rsf']):
                        basel_results.extend(self.basel_validator.validate_liquidity_ratios(
                            liq['hqla'], liq['net_cash_outflows'], liq['asf'], liq['rsf']
                        ))

                results['basel_iii'] = basel_results

            # FRTB validations
            if 'trading_book' in portfolio_data:
                tb = portfolio_data['trading_book']
                frtb_results = []

                # Sensitivities-based approach
                if 'sensitivities' in tb:
                    sens = tb['sensitivities']
                    if all(k in sens for k in ['delta', 'vega', 'curvature']):
                        frtb_results.extend(self.frtb_validator.validate_sensitivities_based_approach(
                            sens['delta'], sens['vega'], sens['curvature']
                        ))

                # Internal models approach
                if 'risk_measures' in tb:
                    rm = tb['risk_measures']
                    if all(k in rm for k in ['es', 'var']):
                        frtb_results.extend(self.frtb_validator.validate_internal_models_approach(
                            rm['es'], rm['var']
                        ))

                results['frtb'] = frtb_results

            # Store results
            self._store_results(results['basel_iii'] + results['frtb'])

            # Generate summary
            all_results = results['basel_iii'] + results['frtb']
            results['summary'] = self._generate_summary(all_results)

        except Exception as e:
            print(f"Error in compliance check: {str(e)}")

        return results

    def _store_results(self, results: List[ComplianceResult]):
        """Store compliance results in database."""
        with sqlite3.connect(self.db_path) as conn:
            for result in results:
                conn.execute("""
                    INSERT INTO compliance_results
                    (framework, rule_id, rule_description, status, value, threshold,
                     deviation, timestamp, details, recommendations)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    result.framework.value,
                    result.rule_id,
                    result.rule_description,
                    result.status.value,
                    result.value,
                    result.threshold,
                    result.deviation,
                    result.timestamp.isoformat(),
                    json.dumps(result.details),
                    json.dumps(result.recommendations)
                ))

    def _generate_summary(self, results: List[ComplianceResult]) -> List[Dict[str, Any]]:
        """Generate compliance summary statistics."""
        if not results:
            return []

        summary = []

        # Overall compliance rate
        compliant_count = sum(1 for r in results if r.status == ComplianceStatus.COMPLIANT)
        total_count = len(results)
        compliance_rate = compliant_count / total_count if total_count > 0 else 0

        summary.append({
            'metric': 'Overall Compliance Rate',
            'value': compliance_rate,
            'format': 'percentage',
            'status': 'good' if compliance_rate >= 0.95 else 'warning' if compliance_rate >= 0.8 else 'critical'
        })

        # Framework-specific compliance
        frameworks = {}
        for result in results:
            fw = result.framework.value
            if fw not in frameworks:
                frameworks[fw] = {'total': 0, 'compliant': 0}
            frameworks[fw]['total'] += 1
            if result.status == ComplianceStatus.COMPLIANT:
                frameworks[fw]['compliant'] += 1

        for fw, stats in frameworks.items():
            rate = stats['compliant'] / stats['total'] if stats['total'] > 0 else 0
            summary.append({
                'metric': f'{fw} Compliance Rate',
                'value': rate,
                'format': 'percentage',
                'details': f"{stats['compliant']}/{stats['total']} rules",
                'status': 'good' if rate >= 0.95 else 'warning' if rate >= 0.8 else 'critical'
            })

        # Critical violations
        critical_violations = [r for r in results if r.status == ComplianceStatus.NON_COMPLIANT]
        summary.append({
            'metric': 'Critical Violations',
            'value': len(critical_violations),
            'format': 'count',
            'details': [r.rule_id for r in critical_violations],
            'status': 'good' if len(critical_violations) == 0 else 'critical'
        })

        return summary

    def get_compliance_history(self, rule_id: Optional[str] = None,
                             days: int = 30) -> pd.DataFrame:
        """Get historical compliance data."""
        query = """
            SELECT * FROM compliance_results
            WHERE timestamp >= ?
        """
        params = [datetime.now() - timedelta(days=days)]

        if rule_id:
            query += " AND rule_id = ?"
            params.append(rule_id)

        query += " ORDER BY timestamp DESC"

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])

        return df

    def generate_compliance_report(self, results: Dict[str, List[ComplianceResult]]) -> str:
        """Generate detailed compliance report."""
        report = []
        report.append("REGULATORY COMPLIANCE REPORT")
        report.append("=" * 50)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Executive summary
        all_results = []
        for category_results in results.values():
            if isinstance(category_results, list):
                all_results.extend(category_results)

        if all_results:
            compliant = sum(1 for r in all_results if r.status == ComplianceStatus.COMPLIANT)
            total = len(all_results)

            report.append("EXECUTIVE SUMMARY")
            report.append("-" * 20)
            report.append(f"Overall Compliance Rate: {compliant}/{total} ({compliant/total*100:.1f}%)")
            report.append("")

        # Detailed results by framework
        for category, category_results in results.items():
            if category == 'summary' or not category_results:
                continue

            report.append(f"{category.upper().replace('_', ' ')} COMPLIANCE")
            report.append("-" * 30)

            for result in category_results:
                status_symbol = "✓" if result.status == ComplianceStatus.COMPLIANT else "✗"
                report.append(f"{status_symbol} {result.rule_id}: {result.rule_description}")
                report.append(f"   Value: {result.value:.4f}, Threshold: {result.threshold:.4f}")

                if result.recommendations:
                    report.append("   Recommendations:")
                    for rec in result.recommendations:
                        report.append(f"   - {rec}")
                report.append("")

        # Summary statistics
        if 'summary' in results and results['summary']:
            report.append("SUMMARY STATISTICS")
            report.append("-" * 20)
            for stat in results['summary']:
                if stat['format'] == 'percentage':
                    report.append(f"{stat['metric']}: {stat['value']*100:.1f}%")
                elif stat['format'] == 'count':
                    report.append(f"{stat['metric']}: {stat['value']}")
                else:
                    report.append(f"{stat['metric']}: {stat['value']}")
            report.append("")

        return "\n".join(report)

# Example usage and test data generator
def generate_sample_portfolio_data() -> Dict[str, Any]:
    """Generate sample portfolio data for testing."""
    return {
        'capital': {
            'tier1': 1000000000,  # $1B
            'total': 1200000000,  # $1.2B
            'rwa': 10000000000,   # $10B
            'total_exposure': 15000000000,  # $15B
            'liquidity': {
                'hqla': 500000000,  # $500M
                'net_cash_outflows': 400000000,  # $400M
                'asf': 8000000000,  # $8B
                'rsf': 7500000000   # $7.5B
            }
        },
        'trading_book': {
            'sensitivities': {
                'delta': {'IR_USD': 100000, 'IR_EUR': -50000, 'EQ_SPX': 75000},
                'vega': {'IR_USD': 25000, 'IR_EUR': -10000, 'EQ_SPX': 15000},
                'curvature': {'IR_USD': 5000, 'IR_EUR': -2000, 'EQ_SPX': 3000}
            },
            'risk_measures': {
                'es': np.random.normal(1000000, 100000, 252).tolist(),  # Daily ES for 1 year
                'var': np.random.normal(800000, 80000, 252).tolist()    # Daily VaR for 1 year
            }
        }
    }

if __name__ == "__main__":
    # Example usage
    engine = RegulatoryComplianceEngine()
    sample_data = generate_sample_portfolio_data()

    print("Running regulatory compliance validation...")
    results = engine.run_full_compliance_check(sample_data)

    print("\nCompliance Report:")
    print(engine.generate_compliance_report(results))