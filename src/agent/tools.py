"""
FinSight agent tools.

Each tool is a pure function decorated with @tool from LangChain.
In production, swap the mock implementations for real DB/API calls:
  - transaction_analyzer → DynamoDB / Redshift query
  - compliance_screener  → OFAC SDN list API / internal AML engine
  - risk_scorer          → Capital One internal risk model endpoint
  - report_generator     → Bedrock Knowledge Bases + Claude
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta

from langchain_core.tools import tool

from src.agent.state import (
    AnomalyFlag,
    ComplianceResult,
    ComplianceStatus,
    RiskAssessment,
    RiskLevel,
    Transaction,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Tool 1 — Transaction Analyzer
# ---------------------------------------------------------------------------

@tool
def transaction_analyzer(account_id: str, days: int = 30) -> dict:
    """
    Fetch and analyze recent transactions for an account.

    Detects anomalies including:
    - Velocity spikes (unusual number of transactions in a short window)
    - Geographic mismatches (transactions from unexpected locations)
    - Amount anomalies (transactions significantly above historical average)
    - Structuring patterns (multiple transactions just below CTR thresholds)

    Args:
        account_id: The account identifier to analyze.
        days: Lookback window in days (default 30).

    Returns:
        Dict containing transactions list and detected anomaly flags.
    """
    logger.info("transaction_analyzer called", account_id=account_id, days=days)

    # ---- Mock data (replace with DynamoDB / Redshift in production) ----
    base_date = datetime.utcnow()
    transactions = [
        Transaction(
            transaction_id=f"TXN-{i:04d}",
            account_id=account_id,
            amount=round(random.uniform(10, 5000), 2),
            merchant=random.choice(
                ["Amazon", "Whole Foods", "Shell Gas", "Netflix", "Unknown Merchant XJ"]
            ),
            category=random.choice(
                ["retail", "grocery", "fuel", "entertainment", "international_wire"]
            ),
            timestamp=base_date - timedelta(days=random.randint(0, days)),
            country_code=random.choice(["US", "US", "US", "RU", "NG"]),
            is_international=random.random() < 0.1,
            channel=random.choice(["online", "pos", "atm", "wire"]),
        )
        for i in range(25)
    ]

    # Inject suspicious patterns for demo realism
    transactions.append(
        Transaction(
            transaction_id="TXN-SUSPICIOUS-001",
            account_id=account_id,
            amount=9_800.00,  # structuring — just under $10k CTR threshold
            merchant="Cash Advance",
            category="cash",
            timestamp=base_date - timedelta(hours=6),
            country_code="US",
            channel="atm",
        )
    )
    transactions.append(
        Transaction(
            transaction_id="TXN-SUSPICIOUS-002",
            account_id=account_id,
            amount=9_750.00,
            merchant="Cash Advance",
            category="cash",
            timestamp=base_date - timedelta(hours=4),
            country_code="US",
            channel="atm",
        )
    )

    # ---- Anomaly detection logic ----
    anomalies: list[AnomalyFlag] = []

    # Structuring check: multiple large-ish cash txns just under $10k
    cash_txns = [t for t in transactions if t.category == "cash"]
    near_threshold = [t for t in cash_txns if 9_000 <= t.amount < 10_000]
    if len(near_threshold) >= 2:
        anomalies.append(
            AnomalyFlag(
                flag_type="structuring",
                severity=RiskLevel.HIGH,
                description=(
                    f"Detected {len(near_threshold)} cash transactions between $9,000–$9,999 "
                    "within a short window. Possible BSA structuring to avoid CTR filing."
                ),
                transaction_ids=[t.transaction_id for t in near_threshold],
                confidence_score=0.91,
            )
        )

    # Geographic anomaly: transactions from high-risk countries
    high_risk_countries = {"RU", "NG", "KP", "IR"}
    intl_txns = [t for t in transactions if t.country_code in high_risk_countries]
    if intl_txns:
        anomalies.append(
            AnomalyFlag(
                flag_type="geo_risk",
                severity=RiskLevel.MEDIUM,
                description=(
                    f"Transactions from high-risk jurisdictions detected: "
                    f"{set(t.country_code for t in intl_txns)}"
                ),
                transaction_ids=[t.transaction_id for t in intl_txns],
                confidence_score=0.78,
            )
        )

    return {
        "account_id": account_id,
        "period_days": days,
        "total_transactions": len(transactions),
        "total_volume_usd": round(sum(t.amount for t in transactions), 2),
        "anomalies_detected": len(anomalies),
        "transactions": [t.model_dump() for t in transactions[:10]],  # sample
        "anomalies": [a.model_dump() for a in anomalies],
    }


# ---------------------------------------------------------------------------
# Tool 2 — Risk Scorer
# ---------------------------------------------------------------------------

@tool
def risk_scorer(account_id: str, anomaly_count: int, international_txn_count: int) -> dict:
    """
    Compute a multi-factor risk score for an account.

    Factors considered:
    - Number and severity of detected anomalies
    - International transaction frequency
    - Account age and tenure
    - Historical dispute rate
    - Velocity relative to peer group

    Args:
        account_id: Account to score.
        anomaly_count: Number of anomalies detected (from transaction_analyzer).
        international_txn_count: Number of international transactions.

    Returns:
        RiskAssessment with score (0–100), level, and explainable factors.
    """
    logger.info("risk_scorer called", account_id=account_id)

    # Weighted scoring model (replace with internal ML model endpoint in prod)
    factors = []
    score = 0.0

    if anomaly_count == 0:
        score += 5
        factors.append({"factor": "anomaly_count", "weight": 30, "contribution": 5, "note": "No anomalies"})
    elif anomaly_count <= 2:
        score += 35
        factors.append({"factor": "anomaly_count", "weight": 30, "contribution": 35, "note": f"{anomaly_count} anomalies"})
    else:
        score += 65
        factors.append({"factor": "anomaly_count", "weight": 30, "contribution": 65, "note": f"{anomaly_count} anomalies — elevated"})

    if international_txn_count > 5:
        score += 20
        factors.append({"factor": "international_exposure", "weight": 20, "contribution": 20, "note": "High international activity"})
    else:
        score += 5
        factors.append({"factor": "international_exposure", "weight": 20, "contribution": 5, "note": "Normal international activity"})

    # Simulated peer comparison (replace with Redshift aggregation)
    peer_deviation = random.uniform(0, 30)
    score += peer_deviation * 0.5
    factors.append({
        "factor": "peer_deviation",
        "weight": 25,
        "contribution": round(peer_deviation * 0.5, 1),
        "note": f"Spending {peer_deviation:.0f}% above peer group average",
    })

    score = min(score, 100.0)

    if score < 30:
        level = RiskLevel.LOW
        recommendation = "Standard monitoring. No immediate action required."
    elif score < 55:
        level = RiskLevel.MEDIUM
        recommendation = "Enhanced monitoring. Flag for quarterly review."
    elif score < 75:
        level = RiskLevel.HIGH
        recommendation = "Escalate to BSA officer. Consider account restrictions pending review."
    else:
        level = RiskLevel.CRITICAL
        recommendation = "Immediate escalation required. Suspend outbound wires pending SAR review."

    assessment = RiskAssessment(
        account_id=account_id,
        overall_risk=level,
        risk_score=round(score, 1),
        factors=factors,
        recommendation=recommendation,
    )

    return assessment.model_dump()


# ---------------------------------------------------------------------------
# Tool 3 — Compliance Screener
# ---------------------------------------------------------------------------

@tool
def compliance_screener(account_id: str, transaction_ids: list[str]) -> dict:
    """
    Screen transactions against AML and OFAC watchlists.

    Checks:
    - OFAC SDN (Specially Designated Nationals) list
    - PEP (Politically Exposed Persons) database
    - Internal AML pattern library (structuring, layering, integration)
    - FinCEN 314(a) list matches

    Args:
        account_id: Account to screen.
        transaction_ids: Specific transaction IDs to screen.

    Returns:
        ComplianceResult with status and any matched watchlist entries.
    """
    logger.info("compliance_screener called", account_id=account_id, txn_count=len(transaction_ids))

    # Mock OFAC/AML screening — replace with real OFAC API or internal AML engine
    results: list[ComplianceResult] = []

    # Simulate OFAC check
    ofac_match = any("SUSPICIOUS" in txn_id for txn_id in transaction_ids)
    results.append(
        ComplianceResult(
            account_id=account_id,
            status=ComplianceStatus.REVIEW_REQUIRED if ofac_match else ComplianceStatus.CLEAR,
            matches=[
                {
                    "list": "OFAC_SDN",
                    "match_score": 0.87,
                    "matched_name": "SAMPLE ENTITY LLC",
                    "match_type": "entity",
                }
            ] if ofac_match else [],
            screening_type="OFAC",
            notes="Possible OFAC entity match on counterparty — requires human review" if ofac_match else "",
        )
    )

    # Simulate AML pattern check
    results.append(
        ComplianceResult(
            account_id=account_id,
            status=ComplianceStatus.REVIEW_REQUIRED if len(transaction_ids) > 1 else ComplianceStatus.CLEAR,
            matches=[
                {
                    "pattern": "structuring",
                    "confidence": 0.91,
                    "rule": "BSA_STRUCT_001",
                    "description": "Multiple sub-threshold cash transactions within 24-hour window",
                }
            ] if len(transaction_ids) > 1 else [],
            screening_type="AML_pattern",
            notes="BSA structuring pattern detected — SAR filing may be required" if len(transaction_ids) > 1 else "",
        )
    )

    overall_status = (
        ComplianceStatus.REVIEW_REQUIRED
        if any(r.status != ComplianceStatus.CLEAR for r in results)
        else ComplianceStatus.CLEAR
    )

    return {
        "account_id": account_id,
        "overall_status": overall_status,
        "screening_results": [r.model_dump() for r in results],
        "requires_sar": overall_status == ComplianceStatus.REVIEW_REQUIRED,
        "screened_at": datetime.utcnow().isoformat(),
    }


# ---------------------------------------------------------------------------
# Tool 4 — Report Generator
# ---------------------------------------------------------------------------

@tool
def report_generator(
    account_id: str,
    risk_level: str,
    anomaly_summary: str,
    compliance_status: str,
    recommendations: str,
) -> dict:
    """
    Generate a structured compliance and risk report for an account.

    Produces a report suitable for BSA officers, risk committees,
    or regulatory submissions. Includes all required audit fields.

    Args:
        account_id: The subject account.
        risk_level: Overall risk level (low/medium/high/critical).
        anomaly_summary: Plain-text summary of detected anomalies.
        compliance_status: AML/OFAC screening outcome.
        recommendations: Recommended actions.

    Returns:
        Formatted report with metadata for audit trail.
    """
    logger.info("report_generator called", account_id=account_id, risk_level=risk_level)

    report = f"""
FINSIGHT AI — FINANCIAL INTELLIGENCE REPORT
{'='*60}
Report ID:         RPT-{account_id}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}
Account:           {account_id}
Generated:         {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}
Generated By:      FinSight AI Agent v1.0 (AWS Bedrock / Claude 3.5 Sonnet)
Classification:    CONFIDENTIAL — BSA/AML Internal Use Only
{'='*60}

EXECUTIVE SUMMARY
-----------------
Overall Risk Level: {risk_level.upper()}
Compliance Status:  {compliance_status.upper()}

ANOMALY FINDINGS
----------------
{anomaly_summary}

RECOMMENDED ACTIONS
-------------------
{recommendations}

AUDIT TRAIL
-----------
This report was generated by an AI agent and must be reviewed
by a qualified BSA/Compliance Officer before any action is taken.
All AI outputs are subject to human oversight per internal policy.

AI systems used: AWS Bedrock (Claude 3.5 Sonnet), LangGraph 0.2
Data sources:    Transaction DB (DynamoDB), OFAC SDN API, AML Engine
{'='*60}
END OF REPORT — REVIEW AND FILE AS REQUIRED
""".strip()

    return {
        "report_id": f"RPT-{account_id}-{datetime.utcnow().strftime('%Y%m%d')}",
        "account_id": account_id,
        "content": report,
        "generated_at": datetime.utcnow().isoformat(),
        "requires_human_review": risk_level.lower() in ("high", "critical"),
    }


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

TOOLS = [
    transaction_analyzer,
    risk_scorer,
    compliance_screener,
    report_generator,
]

TOOL_MAP = {tool.name: tool for tool in TOOLS}
