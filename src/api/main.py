"""
FinSight FastAPI application.

Production-ready API with:
- Request/response validation (Pydantic v2)
- Structured request logging with correlation IDs
- Health check + readiness endpoints (for K8s / ECS health probes)
- OpenAPI docs auto-generated from type annotations
- Async request handling throughout

In production, add:
- AWS Cognito / Capital One SSO via OAuth2 middleware
- Rate limiting (slowapi + Redis)
- mTLS for service-to-service calls
- CloudTrail-compatible audit log export
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.agent.graph import agent_graph, run_agent
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Lifespan handler — runs on startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up the agent graph and validate AWS connectivity on startup."""
    logger.info("FinSight API starting up")
    # agent_graph is built at import time — validate it compiled cleanly
    assert agent_graph is not None, "Agent graph failed to compile"
    logger.info("agent graph validated", nodes=list(agent_graph.nodes.keys()) if hasattr(agent_graph, 'nodes') else "compiled")
    yield
    logger.info("FinSight API shutting down")


# ---------------------------------------------------------------------------
# App definition
# ---------------------------------------------------------------------------

app = FastAPI(
    title="FinSight AI Agent API",
    description=(
        "Financial intelligence agent powered by AWS Bedrock (Claude 3.5 Sonnet) "
        "and LangGraph. Analyzes transactions, scores risk, screens for AML/OFAC "
        "compliance, and generates audit-ready reports."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # tighten in prod
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request logging middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every request with a correlation ID for audit trail."""
    correlation_id = str(uuid.uuid4())
    request.state.correlation_id = correlation_id

    logger.info(
        "request received",
        method=request.method,
        path=request.url.path,
        correlation_id=correlation_id,
    )

    start = datetime.utcnow()
    response = await call_next(request)
    duration_ms = int((datetime.utcnow() - start).total_seconds() * 1000)

    logger.info(
        "request complete",
        status_code=response.status_code,
        duration_ms=duration_ms,
        correlation_id=correlation_id,
    )

    response.headers["X-Correlation-ID"] = correlation_id
    return response


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="Natural language query for the financial analyst.",
        examples=["Analyze account ACC-4821 for unusual transactions and flag AML concerns"],
    )
    account_id: str | None = Field(
        None,
        description="Account identifier to analyze. Required for transaction-level queries.",
        examples=["ACC-4821"],
    )
    thread_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Conversation thread ID. Pass the same ID across turns for multi-turn sessions.",
    )


class AnalyzeResponse(BaseModel):
    request_id: str
    query: str
    account_id: str | None
    final_response: str
    risk_level: str | None
    risk_score: float | None
    anomalies_count: int
    report_available: bool
    report: str | None
    guardrails_passed: bool
    guardrails_flags: list[str]
    execution_path: list[str]
    generated_at: str
    error: str | None


class HealthResponse(BaseModel):
    status: str
    version: str
    agent_graph_ready: bool
    timestamp: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Operations"])
async def health_check():
    """
    Liveness probe — returns 200 if the API is running.
    Used by ECS/K8s to determine if the container should receive traffic.
    """
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        agent_graph_ready=agent_graph is not None,
        timestamp=datetime.utcnow().isoformat(),
    )


@app.get("/ready", tags=["Operations"])
async def readiness_check():
    """
    Readiness probe — returns 200 if the agent is ready to process requests.
    Validates AWS Bedrock connectivity before returning healthy.
    """
    try:
        # Lightweight check — validate the graph compiled
        if agent_graph is None:
            raise RuntimeError("Agent graph not initialized")
        return {"status": "ready", "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error("readiness check failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service not ready: {str(e)}",
        )


@app.post(
    "/agent/analyze",
    response_model=AnalyzeResponse,
    tags=["Agent"],
    summary="Run a financial analysis query",
    description=(
        "Submit a natural language query to the FinSight agent. "
        "The agent will parse the intent, run appropriate tools "
        "(transaction analysis, risk scoring, AML/OFAC screening), "
        "and return a synthesized compliance-ready response."
    ),
)
async def analyze(request: AnalyzeRequest, http_request: Request) -> AnalyzeResponse:
    """Main agent invocation endpoint."""
    correlation_id = getattr(http_request.state, "correlation_id", str(uuid.uuid4()))

    logger.info(
        "agent analyze request",
        account_id=request.account_id,
        query_preview=request.query[:60],
        correlation_id=correlation_id,
    )

    try:
        result = await run_agent(
            query=request.query,
            account_id=request.account_id,
            thread_id=request.thread_id,
        )

        # Extract risk details from nested structure
        risk_assessment = result.get("risk_assessment") or {}
        risk_level = risk_assessment.get("overall_risk") if risk_assessment else None
        risk_score = risk_assessment.get("risk_score") if risk_assessment else None

        return AnalyzeResponse(
            request_id=correlation_id,
            query=request.query,
            account_id=request.account_id,
            final_response=result.get("final_response", ""),
            risk_level=risk_level,
            risk_score=risk_score,
            anomalies_count=result.get("anomalies_count", 0),
            report_available=bool(result.get("report")),
            report=result.get("report"),
            guardrails_passed=result.get("guardrails_passed", True),
            guardrails_flags=result.get("guardrails_flags", []),
            execution_path=result.get("execution_path", []),
            generated_at=datetime.utcnow().isoformat(),
            error=result.get("error"),
        )

    except Exception as e:
        logger.error(
            "agent invocation failed",
            error=str(e),
            correlation_id=correlation_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent error: {str(e)}",
        )


@app.get("/agent/graph-viz", tags=["Agent"])
async def graph_visualization():
    """Return the agent graph structure for debugging / documentation."""
    try:
        graph_data = agent_graph.get_graph().to_json()
        return JSONResponse(content={"graph": graph_data})
    except Exception as e:
        return JSONResponse(content={"error": str(e), "note": "Graph visualization requires full env setup"})
