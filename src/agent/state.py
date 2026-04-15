"""
Agent state definitions for FinSight.

Using TypedDict + Pydantic for type-safe state management across
the LangGraph graph. Each node reads from and writes to this state.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Any
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComplianceStatus(str, Enum):
    CLEAR = "clear"
    REVIEW_REQUIRED = "review_required"
    BLOCKED = "blocked"


class Transaction(BaseModel):
    """Represents a single financial transaction."""

    transaction_id: str
    account_id: str
    amount: float
    currency: str = "USD"
    merchant: str
    category: str
    timestamp: datetime
    country_code: str = "US"
    is_international: bool = False
    channel: str  # e.g. "online", "pos", "atm", "wire"

    @property
    def is_large(self) -> bool:
        """Flag transactions over $10k (CTR threshold)."""
        return self.amount >= 10_000


class AnomalyFlag(BaseModel):
    """An anomaly detected in transaction analysis."""

    flag_type: str  # e.g. "velocity", "geo_mismatch", "amount_spike"
    severity: RiskLevel
    description: str
    transaction_ids: list[str]
    confidence_score: float = Field(ge=0.0, le=1.0)


class RiskAssessment(BaseModel):
    """Output of the risk scorer tool."""

    account_id: str
    overall_risk: RiskLevel
    risk_score: float = Field(ge=0.0, le=100.0)
    factors: list[dict[str, Any]]
    recommendation: str
    assessed_at: datetime = Field(default_factory=datetime.utcnow)


class ComplianceResult(BaseModel):
    """Output of AML/OFAC compliance screening."""

    account_id: str
    status: ComplianceStatus
    matches: list[dict[str, Any]] = Field(default_factory=list)
    screening_type: str  # "OFAC", "AML_pattern", "PEP"
    notes: str = ""
    screened_at: datetime = Field(default_factory=datetime.utcnow)


class AgentState(BaseModel):
    """
    Central state object that flows through the LangGraph graph.

    Every node receives this state, performs its work, and returns
    a partial update. LangGraph merges updates automatically.

    The `messages` field uses add_messages reducer — this appends
    rather than overwriting, preserving full conversation history.
    """

    # Core conversation
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
    query: str = ""
    account_id: str | None = None

    # Parsed intent from the intent_parser node
    intent: str | None = None
    tools_needed: list[str] = Field(default_factory=list)

    # Tool outputs
    transactions: list[Transaction] = Field(default_factory=list)
    anomalies: list[AnomalyFlag] = Field(default_factory=list)
    risk_assessment: RiskAssessment | None = None
    compliance_results: list[ComplianceResult] = Field(default_factory=list)

    # Final outputs
    report: str | None = None
    final_response: str | None = None

    # Metadata
    guardrails_passed: bool = True
    guardrails_flags: list[str] = Field(default_factory=list)
    execution_path: list[str] = Field(default_factory=list)
    error: str | None = None

    class Config:
        arbitrary_types_allowed = True
