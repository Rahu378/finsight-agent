"""
FinSight interactive demo.

Runs the agent against several example queries to demonstrate
its capabilities. No real AWS account needed — tools use mock data.

Usage:
    python examples/demo.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import os

# Make src importable from the examples directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Patch Bedrock for demo (remove this when you have real AWS creds)
from unittest.mock import MagicMock, patch

MOCK_RESPONSE = MagicMock()
MOCK_RESPONSE.content = """
Based on the transaction analysis and compliance screening for this account:

**Risk Summary:** HIGH RISK — Score 72/100

**Key Findings:**
1. Structuring Pattern Detected (Confidence: 91%) — Two cash transactions of $9,800 and 
   $9,750 occurred within a 6-hour window, a classic BSA structuring pattern to avoid 
   the $10,000 Currency Transaction Report (CTR) filing threshold.

2. Geographic Risk Exposure — Transactions flagged from high-risk jurisdictions (RU, NG) 
   which require enhanced due diligence under OFAC and FinCEN guidelines.

3. AML Pattern Match — Rule BSA_STRUCT_001 triggered on the sub-threshold cash sequence. 
   SAR (Suspicious Activity Report) filing may be required within 30 days per 31 CFR 1020.320.

**Recommended Actions:**
- Escalate to BSA Officer immediately
- Place outbound wire restrictions pending review  
- Prepare SAR documentation referencing TXN-SUSPICIOUS-001 and TXN-SUSPICIOUS-002
- Schedule Enhanced Due Diligence (EDD) review within 5 business days

⚠️ IMPORTANT: This is a HIGH-RISK finding. Human review by a qualified BSA officer is 
required before any account action is taken.
""".strip()
MOCK_RESPONSE.usage_metadata = {"input_tokens": 450, "output_tokens": 280}


DEMO_QUERIES = [
    {
        "title": "🔍 Query 1 — Full AML Investigation",
        "query": "Analyze account ACC-4821 for unusual transactions in the last 30 days and flag any AML or OFAC concerns. Generate a compliance report.",
        "account_id": "ACC-4821",
    },
    {
        "title": "📊 Query 2 — Risk Score Only",
        "query": "What is the current risk level for account ACC-0042? Focus on transaction anomalies and peer comparison.",
        "account_id": "ACC-0042",
    },
    {
        "title": "📋 Query 3 — Report Generation",
        "query": "Generate an executive compliance report for account ACC-9900 summarizing all findings from this week.",
        "account_id": "ACC-9900",
    },
]


async def run_demo():
    print("\n" + "=" * 65)
    print("  FinSight AI Agent — Demo")
    print("  AWS Bedrock + LangGraph | FinTech Compliance Intelligence")
    print("=" * 65)

    with patch("src.bedrock.client.ChatBedrockConverse") as mock_llm_class:
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = MOCK_RESPONSE
        mock_instance.bind_tools.return_value = mock_instance
        mock_llm_class.return_value = mock_instance

        from src.agent.graph import run_agent

        for demo in DEMO_QUERIES:
            print(f"\n{demo['title']}")
            print("-" * 55)
            print(f"Query: {demo['query']}\n")

            try:
                result = await run_agent(
                    query=demo["query"],
                    account_id=demo["account_id"],
                    thread_id=f"demo-{demo['account_id']}",
                )

                print(f"✅ Execution Path: {' → '.join(result['execution_path'])}")
                print(f"🛡️  Guardrails Passed: {result['guardrails_passed']}")
                print(f"⚠️  Anomalies Found: {result['anomalies_count']}")

                if result.get("risk_assessment"):
                    ra = result["risk_assessment"]
                    print(f"📊 Risk Score: {ra.get('risk_score', 'N/A')}/100 ({ra.get('overall_risk', '').upper()})")

                print(f"\n🤖 Agent Response:")
                print("-" * 40)
                response = result.get("final_response", "")
                # Show first 600 chars for demo brevity
                print(response[:600] + ("..." if len(response) > 600 else ""))

                if result.get("report"):
                    print(f"\n📄 Full report generated ({len(result['report'])} chars) — available via API")

                if result.get("error"):
                    print(f"\n❌ Error: {result['error']}")

            except Exception as e:
                print(f"❌ Demo query failed: {e}")
                import traceback
                traceback.print_exc()

            print()

    print("=" * 65)
    print("Demo complete. To run against real AWS Bedrock:")
    print("  1. Ensure AWS credentials are configured (aws configure)")
    print("  2. Enable Claude 3.5 Sonnet in AWS Bedrock console (us-east-1)")
    print("  3. Remove the mock patch in this demo file")
    print("  4. Run: uvicorn src.api.main:app --reload")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    asyncio.run(run_demo())
