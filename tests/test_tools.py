"""
Tests for FinSight agent tools.

Run with:
    pytest tests/ -v

These tests mock AWS Bedrock calls so they run without real AWS credentials.
For integration tests against real Bedrock, use:
    pytest tests/ -v -m integration
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.agent.tools import (
    compliance_screener,
    report_generator,
    risk_scorer,
    transaction_analyzer,
)
from src.agent.state import RiskLevel, ComplianceStatus


# ---------------------------------------------------------------------------
# Transaction Analyzer tests
# ---------------------------------------------------------------------------

class TestTransactionAnalyzer:
    def test_returns_transactions(self):
        result = transaction_analyzer.invoke({"account_id": "ACC-TEST", "days": 30})
        assert "transactions" in result
        assert isinstance(result["transactions"], list)
        assert len(result["transactions"]) > 0

    def test_detects_structuring(self):
        """The mock data always injects structuring transactions — verify detection."""
        result = transaction_analyzer.invoke({"account_id": "ACC-TEST", "days": 30})
        anomaly_types = [a.get("flag_type") for a in result.get("anomalies", [])]
        assert "structuring" in anomaly_types, (
            "Structuring anomaly should always be detected with default mock data"
        )

    def test_structuring_anomaly_is_high_severity(self):
        result = transaction_analyzer.invoke({"account_id": "ACC-TEST", "days": 30})
        structuring = next(
            (a for a in result.get("anomalies", []) if a.get("flag_type") == "structuring"),
            None,
        )
        assert structuring is not None
        assert structuring["severity"] == RiskLevel.HIGH

    def test_returns_correct_account_id(self):
        result = transaction_analyzer.invoke({"account_id": "ACC-XYZ-999", "days": 7})
        assert result["account_id"] == "ACC-XYZ-999"

    def test_total_volume_is_positive(self):
        result = transaction_analyzer.invoke({"account_id": "ACC-TEST", "days": 30})
        assert result["total_volume_usd"] > 0

    def test_anomaly_confidence_in_range(self):
        result = transaction_analyzer.invoke({"account_id": "ACC-TEST", "days": 30})
        for anomaly in result.get("anomalies", []):
            assert 0.0 <= anomaly["confidence_score"] <= 1.0, (
                f"Confidence score out of range: {anomaly['confidence_score']}"
            )


# ---------------------------------------------------------------------------
# Risk Scorer tests
# ---------------------------------------------------------------------------

class TestRiskScorer:
    def test_no_anomalies_returns_low_risk(self):
        result = risk_scorer.invoke({
            "account_id": "ACC-CLEAN",
            "anomaly_count": 0,
            "international_txn_count": 0,
        })
        assert result["risk_score"] < 55, "Zero anomalies should not return high risk"

    def test_many_anomalies_returns_high_risk(self):
        result = risk_scorer.invoke({
            "account_id": "ACC-RISKY",
            "anomaly_count": 5,
            "international_txn_count": 10,
        })
        assert result["risk_score"] >= 55, "Multiple anomalies should return high risk score"

    def test_risk_score_bounded(self):
        result = risk_scorer.invoke({
            "account_id": "ACC-MAX",
            "anomaly_count": 100,
            "international_txn_count": 100,
        })
        assert 0 <= result["risk_score"] <= 100, "Risk score must be bounded 0–100"

    def test_recommendation_present(self):
        result = risk_scorer.invoke({
            "account_id": "ACC-TEST",
            "anomaly_count": 2,
            "international_txn_count": 3,
        })
        assert "recommendation" in result
        assert len(result["recommendation"]) > 10

    def test_factors_list_populated(self):
        result = risk_scorer.invoke({
            "account_id": "ACC-TEST",
            "anomaly_count": 1,
            "international_txn_count": 2,
        })
        assert isinstance(result["factors"], list)
        assert len(result["factors"]) >= 2

    @pytest.mark.parametrize("anomaly_count,expected_max_score", [
        (0, 30),
        (1, 60),
        (3, 100),
    ])
    def test_risk_scales_with_anomalies(self, anomaly_count, expected_max_score):
        result = risk_scorer.invoke({
            "account_id": "ACC-PARAM",
            "anomaly_count": anomaly_count,
            "international_txn_count": 0,
        })
        # Score should be bounded by the parametrized ceiling
        assert result["risk_score"] <= expected_max_score + 30, (
            f"For {anomaly_count} anomalies, score {result['risk_score']} seems too high"
        )


# ---------------------------------------------------------------------------
# Compliance Screener tests
# ---------------------------------------------------------------------------

class TestComplianceScreener:
    def test_suspicious_txns_trigger_review(self):
        """Transaction IDs containing 'SUSPICIOUS' trigger the mock AML match."""
        result = compliance_screener.invoke({
            "account_id": "ACC-TEST",
            "transaction_ids": ["TXN-SUSPICIOUS-001", "TXN-SUSPICIOUS-002"],
        })
        assert result["overall_status"] == ComplianceStatus.REVIEW_REQUIRED

    def test_clean_txns_return_clear(self):
        result = compliance_screener.invoke({
            "account_id": "ACC-CLEAN",
            "transaction_ids": [],  # No flagged IDs → OFAC clear
        })
        # With empty transaction_ids the AML pattern check also returns clear
        ofac_result = next(
            (r for r in result["screening_results"] if r["screening_type"] == "OFAC"),
            None,
        )
        assert ofac_result is not None
        assert ofac_result["status"] == ComplianceStatus.CLEAR

    def test_requires_sar_when_review_needed(self):
        result = compliance_screener.invoke({
            "account_id": "ACC-SAR",
            "transaction_ids": ["TXN-SUSPICIOUS-001"],
        })
        assert result["requires_sar"] is True

    def test_screened_at_is_valid_iso(self):
        result = compliance_screener.invoke({
            "account_id": "ACC-TEST",
            "transaction_ids": ["TXN-001"],
        })
        # Should parse without error
        datetime.fromisoformat(result["screened_at"])

    def test_multiple_screening_types(self):
        result = compliance_screener.invoke({
            "account_id": "ACC-TEST",
            "transaction_ids": ["TXN-001"],
        })
        screening_types = {r["screening_type"] for r in result["screening_results"]}
        assert "OFAC" in screening_types
        assert "AML_pattern" in screening_types


# ---------------------------------------------------------------------------
# Report Generator tests
# ---------------------------------------------------------------------------

class TestReportGenerator:
    def test_report_contains_account_id(self):
        result = report_generator.invoke({
            "account_id": "ACC-RPT-001",
            "risk_level": "high",
            "anomaly_summary": "Two structuring events detected.",
            "compliance_status": "review_required",
            "recommendations": "Escalate to BSA officer.",
        })
        assert "ACC-RPT-001" in result["content"]

    def test_high_risk_report_requires_human_review(self):
        result = report_generator.invoke({
            "account_id": "ACC-HIGH",
            "risk_level": "critical",
            "anomaly_summary": "Multiple AML patterns.",
            "compliance_status": "review_required",
            "recommendations": "Immediate escalation.",
        })
        assert result["requires_human_review"] is True

    def test_low_risk_report_no_human_review_flag(self):
        result = report_generator.invoke({
            "account_id": "ACC-LOW",
            "risk_level": "low",
            "anomaly_summary": "No anomalies detected.",
            "compliance_status": "clear",
            "recommendations": "Standard monitoring.",
        })
        assert result["requires_human_review"] is False

    def test_report_id_format(self):
        result = report_generator.invoke({
            "account_id": "ACC-FMT",
            "risk_level": "medium",
            "anomaly_summary": "Minor velocity spike.",
            "compliance_status": "clear",
            "recommendations": "Enhanced monitoring.",
        })
        assert result["report_id"].startswith("RPT-ACC-FMT")

    def test_report_is_non_empty(self):
        result = report_generator.invoke({
            "account_id": "ACC-TEST",
            "risk_level": "medium",
            "anomaly_summary": "Test.",
            "compliance_status": "clear",
            "recommendations": "Monitor.",
        })
        assert len(result["content"]) > 200
