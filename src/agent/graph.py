"""
FinSight LangGraph agent graph.

This module assembles the StateGraph — connecting nodes with edges
and conditional routing. The graph is compiled once at startup and
reused across requests (thread-safe via LangGraph's checkpointer).

Graph topology:
    [START]
       │
       ▼
  intent_parser ──────────────────────────────────────────────────────┐
       │ (tools_needed?)                                               │
       │ yes                                                           │ no
       ▼                                                               ▼
  tool_executor                                                   synthesizer
       │                                                               │
       ▼                                                               │
  synthesizer ◀──────────────────────────────────────────────────────┘
       │
       ▼
  guardrails_checker
       │ (critical PII?)
       │ yes → back to synthesizer (max 1 retry)
       │ no
       ▼
     [END]
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from src.agent.state import AgentState
from src.agent.nodes import (
    guardrails_checker,
    intent_parser,
    route_after_guardrails,
    route_after_intent,
    synthesizer,
    tool_executor,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_graph():
    """
    Construct and compile the FinSight agent graph.

    Returns a compiled graph ready for invocation. The MemorySaver
    checkpointer enables conversation continuity across turns —
    swap for SqliteSaver or RedisSaver for persistent sessions in prod.
    """
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("intent_parser", intent_parser)
    graph.add_node("tool_executor", tool_executor)
    graph.add_node("synthesizer", synthesizer)
    graph.add_node("guardrails_checker", guardrails_checker)

    # Entry point
    graph.add_edge(START, "intent_parser")

    # After intent parsing: route based on whether tools are needed
    graph.add_conditional_edges(
        "intent_parser",
        route_after_intent,
        {
            "tool_executor": "tool_executor",
            "synthesizer": "synthesizer",
        },
    )

    # Tool executor always feeds into synthesizer
    graph.add_edge("tool_executor", "synthesizer")

    # Synthesizer always feeds into guardrails
    graph.add_edge("synthesizer", "guardrails_checker")

    # After guardrails: end normally or retry synthesis on critical PII flag
    graph.add_conditional_edges(
        "guardrails_checker",
        route_after_guardrails,
        {
            "synthesizer": "synthesizer",
            "end": END,
        },
    )

    # Compile with in-memory checkpointer (swap for persistent in prod)
    checkpointer = MemorySaver()
    compiled = graph.compile(checkpointer=checkpointer)

    logger.info("agent graph compiled successfully")
    return compiled


# Module-level compiled graph — built once, reused per request
agent_graph = build_graph()


async def run_agent(query: str, account_id: str | None = None, thread_id: str = "default") -> dict:
    """
    Run the FinSight agent for a given query.

    Args:
        query: The analyst's natural language question.
        account_id: Optional account to analyze.
        thread_id: Conversation thread ID for checkpointing (enables multi-turn).

    Returns:
        Dict with final_response, risk_assessment, report, execution_path,
        guardrails_passed, and any flags.
    """
    from langchain_core.messages import HumanMessage

    initial_state = AgentState(
        messages=[HumanMessage(content=query)],
        query=query,
        account_id=account_id,
    )

    config = {"configurable": {"thread_id": thread_id}}

    logger.info("agent invocation started", account_id=account_id, thread_id=thread_id)

    final_state = await agent_graph.ainvoke(initial_state.model_dump(), config=config)

    logger.info(
        "agent invocation complete",
        path=final_state.get("execution_path", []),
        guardrails_passed=final_state.get("guardrails_passed", True),
    )

    return {
        "query": query,
        "account_id": account_id,
        "final_response": final_state.get("final_response", ""),
        "risk_assessment": final_state.get("risk_assessment"),
        "report": final_state.get("report"),
        "anomalies_count": len(final_state.get("anomalies", [])),
        "execution_path": final_state.get("execution_path", []),
        "guardrails_passed": final_state.get("guardrails_passed", True),
        "guardrails_flags": final_state.get("guardrails_flags", []),
        "error": final_state.get("error"),
    }
