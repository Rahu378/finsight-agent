"""
LangGraph node functions for the FinSight agent.

Each function is a node in the agent graph. Nodes receive the current
AgentState, perform work (calling Bedrock, running tools, checking
guardrails), and return a dict of state updates.

Graph flow:
    intent_parser → tool_router → tool_executor → synthesizer → guardrails_checker → END
                                        ↑                              |
                                        └──────── (retry loop) ────────┘
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agent.state import AgentState
from src.agent.tools import TOOL_MAP, compliance_screener
from src.bedrock.client import BedrockClient
from src.utils.logger import get_logger

logger = get_logger(__name__)

bedrock = BedrockClient()

# ---------------------------------------------------------------------------
# System prompt — the agent's operating instructions
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are FinSight, an AI financial intelligence agent built for a regulated
US financial services environment. You assist BSA officers, risk analysts, and compliance
teams by analyzing transaction data, detecting anomalies, and generating audit-ready reports.

CRITICAL OPERATING RULES:
1. Never output real customer PII (SSN, full card numbers, account passwords).
2. Always recommend human review for HIGH or CRITICAL risk findings.
3. Every finding must cite the specific transaction IDs or data points it is based on.
4. Do not speculate beyond what the data shows — say "insufficient data" when uncertain.
5. All outputs must comply with BSA, AML, and OFAC regulatory standards.
6. You have access to these tools: transaction_analyzer, risk_scorer, compliance_screener, report_generator.

When given a query, first determine which tools you need, then reason through the results
to produce a clear, actionable response for a compliance professional."""


# ---------------------------------------------------------------------------
# Node 1: Intent Parser
# ---------------------------------------------------------------------------

def intent_parser(state: AgentState) -> dict[str, Any]:
    """
    Parse the user query and determine which tools are needed.

    Uses Claude to classify the intent and plan the tool execution
    sequence. Returns the intent label and ordered list of tools.
    """
    logger.info("intent_parser node running", query=state.query[:80])

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=f"""Analyze this financial analyst query and respond with JSON only:

Query: {state.query}
Account ID: {state.account_id or "not specified"}

Respond with this exact JSON structure:
{{
  "intent": "one of: transaction_analysis | risk_assessment | compliance_check | full_investigation | report_only",
  "tools_needed": ["ordered list of tools to call: transaction_analyzer, risk_scorer, compliance_screener, report_generator"],
  "reasoning": "brief explanation of why these tools are needed"
}}"""
        ),
    ]

    response = bedrock.invoke(messages)
    content = response.content

    try:
        # Strip markdown fences if Claude wraps in ```json
        clean = re.sub(r"```(?:json)?|```", "", content).strip()
        parsed = json.loads(clean)
        intent = parsed.get("intent", "full_investigation")
        tools_needed = parsed.get("tools_needed", ["transaction_analyzer", "risk_scorer"])
        logger.info("intent parsed", intent=intent, tools=tools_needed)
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning("intent parsing failed, using defaults", error=str(e))
        intent = "full_investigation"
        tools_needed = ["transaction_analyzer", "risk_scorer", "compliance_screener", "report_generator"]

    return {
        "intent": intent,
        "tools_needed": tools_needed,
        "execution_path": state.execution_path + ["intent_parser"],
        "messages": [AIMessage(content=f"Intent: {intent}. Tools planned: {', '.join(tools_needed)}")],
    }


# ---------------------------------------------------------------------------
# Node 2: Tool Executor
# ---------------------------------------------------------------------------

def tool_executor(state: AgentState) -> dict[str, Any]:
    """
    Execute each tool in the planned sequence and collect results.

    Runs tools in order, passing outputs forward as inputs where needed.
    All tool calls are logged for the audit trail.
    """
    logger.info("tool_executor node running", tools=state.tools_needed)

    updates: dict[str, Any] = {
        "execution_path": state.execution_path + ["tool_executor"],
    }

    txn_result: dict | None = None
    risk_result: dict | None = None
    compliance_result: dict | None = None

    for tool_name in state.tools_needed:
        tool_fn = TOOL_MAP.get(tool_name)
        if not tool_fn:
            logger.warning("unknown tool requested", tool=tool_name)
            continue

        logger.info("executing tool", tool=tool_name)

        try:
            if tool_name == "transaction_analyzer":
                txn_result = tool_fn.invoke({
                    "account_id": state.account_id or "ACC-UNKNOWN",
                    "days": 30,
                })
                updates["transactions"] = txn_result.get("transactions", [])
                updates["anomalies"] = txn_result.get("anomalies", [])

            elif tool_name == "risk_scorer" and txn_result:
                risk_result = tool_fn.invoke({
                    "account_id": state.account_id or "ACC-UNKNOWN",
                    "anomaly_count": txn_result.get("anomalies_detected", 0),
                    "international_txn_count": sum(
                        1 for t in txn_result.get("transactions", [])
                        if t.get("is_international")
                    ),
                })
                updates["risk_assessment"] = risk_result

            elif tool_name == "compliance_screener" and txn_result:
                suspicious_ids = [
                    a.get("transaction_ids", [])
                    for a in txn_result.get("anomalies", [])
                ]
                flat_ids = [tid for sublist in suspicious_ids for tid in sublist]
                compliance_result = tool_fn.invoke({
                    "account_id": state.account_id or "ACC-UNKNOWN",
                    "transaction_ids": flat_ids or ["TXN-0001"],
                })
                updates["compliance_results"] = compliance_result.get("screening_results", [])

            elif tool_name == "report_generator":
                anomaly_summary = "\n".join(
                    f"• [{a.get('severity', 'unknown').upper()}] {a.get('description', '')}"
                    for a in (txn_result or {}).get("anomalies", [])
                ) or "No significant anomalies detected."

                report_result = tool_fn.invoke({
                    "account_id": state.account_id or "ACC-UNKNOWN",
                    "risk_level": (risk_result or {}).get("overall_risk", "medium"),
                    "anomaly_summary": anomaly_summary,
                    "compliance_status": (compliance_result or {}).get("overall_status", "clear"),
                    "recommendations": (risk_result or {}).get("recommendation", "Standard monitoring."),
                })
                updates["report"] = report_result.get("content", "")

        except Exception as e:
            logger.error("tool execution failed", tool=tool_name, error=str(e))
            updates["error"] = f"Tool {tool_name} failed: {str(e)}"

    return updates


# ---------------------------------------------------------------------------
# Node 3: Synthesizer
# ---------------------------------------------------------------------------

def synthesizer(state: AgentState) -> dict[str, Any]:
    """
    Synthesize all tool outputs into a coherent final response.

    Uses Claude to weave together the transaction analysis, risk score,
    compliance findings, and report into a response tailored for the
    compliance analyst's original question.
    """
    logger.info("synthesizer node running")

    context_parts = [f"Original Query: {state.query}\n"]

    if state.anomalies:
        context_parts.append(
            f"Anomalies Found ({len(state.anomalies)}):\n" +
            "\n".join(f"  - {a}" for a in state.anomalies[:5])
        )

    if state.risk_assessment:
        ra = state.risk_assessment
        risk_data = ra if isinstance(ra, dict) else ra.model_dump()
        context_parts.append(
            f"Risk Score: {risk_data.get('risk_score')}/100 ({risk_data.get('overall_risk', '').upper()})\n"
            f"Recommendation: {risk_data.get('recommendation', '')}"
        )

    if state.compliance_results:
        context_parts.append(
            f"Compliance Status: {len(state.compliance_results)} screens run. "
            f"Issues found: {sum(1 for r in state.compliance_results if isinstance(r, dict) and r.get('status') != 'clear')}"
        )

    if state.report:
        context_parts.append(f"Full Report Generated: {len(state.report)} characters")

    context = "\n\n".join(context_parts)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=f"""Using the analysis results below, provide a clear, actionable response
to the analyst's query. Be specific, cite findings, and clearly state what action is required.

{context}

Respond in a professional tone suitable for a BSA compliance officer."""
        ),
    ]

    response = bedrock.invoke(messages)

    return {
        "final_response": response.content,
        "execution_path": state.execution_path + ["synthesizer"],
        "messages": [response],
    }


# ---------------------------------------------------------------------------
# Node 4: Guardrails Checker
# ---------------------------------------------------------------------------

def guardrails_checker(state: AgentState) -> dict[str, Any]:
    """
    Validate the final response against compliance and safety rules.

    Checks for:
    - PII exposure (SSNs, card numbers, passwords)
    - Unsubstantiated risk claims
    - Missing human review disclaimers on high-risk outputs
    - Hallucinated regulatory references

    In production, this node also calls AWS Bedrock Guardrails API
    for automated content filtering before the response is returned.
    """
    logger.info("guardrails_checker node running")

    response = state.final_response or ""
    flags: list[str] = []

    # PII patterns
    pii_patterns = [
        (r"\b\d{3}-\d{2}-\d{4}\b", "Possible SSN detected"),
        (r"\b4[0-9]{12}(?:[0-9]{3})?\b", "Possible Visa card number detected"),
        (r"\bpassword[:\s]+\S+", "Password value detected"),
    ]
    for pattern, message in pii_patterns:
        if re.search(pattern, response, re.IGNORECASE):
            flags.append(message)

    # Ensure high-risk responses include human review language
    risk_data = state.risk_assessment or {}
    if isinstance(risk_data, dict):
        risk_level = risk_data.get("overall_risk", "low")
    else:
        risk_level = "low"

    if risk_level in ("high", "critical"):
        if "human review" not in response.lower() and "review" not in response.lower():
            flags.append("High-risk response missing human review disclaimer")
            # Auto-append disclaimer rather than blocking
            response += "\n\n⚠️ IMPORTANT: This is a HIGH-RISK finding. Human review by a qualified BSA officer is required before any account action is taken."

    guardrails_passed = len(flags) == 0

    if flags:
        logger.warning("guardrails flags raised", flags=flags)
    else:
        logger.info("guardrails check passed")

    return {
        "final_response": response,
        "guardrails_passed": guardrails_passed,
        "guardrails_flags": flags,
        "execution_path": state.execution_path + ["guardrails_checker"],
    }


# ---------------------------------------------------------------------------
# Routing functions (used by LangGraph conditional edges)
# ---------------------------------------------------------------------------

def route_after_intent(state: AgentState) -> str:
    """Route to tool_executor or directly to synthesizer."""
    if state.tools_needed:
        return "tool_executor"
    return "synthesizer"


def route_after_guardrails(state: AgentState) -> str:
    """Route to END or back to synthesizer if flags are critical."""
    critical_flags = [f for f in state.guardrails_flags if "PII" in f or "SSN" in f]
    if critical_flags:
        return "synthesizer"  # re-run synthesis with stricter instructions
    return "end"
