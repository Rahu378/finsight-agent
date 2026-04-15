# FinSight AI Agent 🏦

> A production-grade financial intelligence agent built with **LangGraph** + **AWS Bedrock (Claude)** — designed for regulated fintech environments.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![AWS Bedrock](https://img.shields.io/badge/AWS-Bedrock-orange.svg)](https://aws.amazon.com/bedrock/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2-green.svg)](https://github.com/langchain-ai/langgraph)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-teal.svg)](https://fastapi.tiangolo.com/)

---

## What This Does

FinSight is an **agentic AI system** that handles complex financial analysis queries through a multi-step reasoning graph. It combines:

- **Real-time transaction anomaly detection** — flags unusual spending patterns
- **AML/OFAC compliance screening** — checks transactions against watchlists
- **Risk scoring** — dynamic risk assessment with explainable outputs
- **Natural language financial reporting** — generates executive-ready summaries
- **Guardrails enforcement** — blocks PII exposure and hallucinated financial data

Built for **Capital One-style regulated environments** where data governance, auditability, and compliance are non-negotiable.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────┐
│                  FastAPI Gateway                    │
│              (auth, rate limiting, logging)         │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│              LangGraph Agent Graph                  │
│                                                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────────┐   │
│  │  Intent  │───▶│  Route   │───▶│   Execute    │   │
│  │  Parser  │    │  Tools   │    │    Tools     │   │
│  └──────────┘    └──────────┘    └──────┬───────┘   │
│                                         │           │
│  ┌──────────────┐    ┌─────────────┐    │           │
│  │  Guardrails  │◀───│  Synthesize │◀───┘           │
│  │   Checker    │    │  Response   │                │
│  └──────┬───────┘    └─────────────┘                │
│         │                                           │
└─────────┼───────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────┐
│           AWS Bedrock (Claude 3.5 Sonnet)           │
│                                                     │
│  ┌──────────────┐  ┌─────────────┐  ┌───────────┐   │
│  │  Knowledge   │  │  Guardrails │  │   Model   │   │
│  │    Bases     │  │  (PII/harm) │  │Evaluation │   │
│  └──────────────┘  └─────────────┘  └───────────┘   │
└─────────────────────────────────────────────────────┘
```

### Agent Tools

| Tool | Purpose | Compliance Relevance |
|---|---|---|
| `transaction_analyzer` | Detect anomalies in spending patterns | Fraud Prevention |
| `risk_scorer` | Dynamic multi-factor risk assessment | Credit Risk / BSA |
| `compliance_screener` | AML/OFAC watchlist screening | Regulatory (BSA/AML) |
| `report_generator` | NL → structured executive reports | Audit Trail |
| `account_profiler` | Behavioral baseline + deviation scoring | KYC / CDD |

---

## Quick Start

### Prerequisites
- Python 3.11+
- AWS account with Bedrock access (Claude 3.5 Sonnet enabled)
- AWS CLI configured (`aws configure`)

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/finsight-agent.git
cd finsight-agent

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your AWS region and settings
```

### 3. Run the API

```bash
uvicorn src.api.main:app --reload --port 8000
```

### 4. Try It Out

```bash
# Analyze a transaction
curl -X POST http://localhost:8000/agent/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Analyze account ACC-4821 for unusual transactions in the last 30 days and flag any AML concerns",
    "account_id": "ACC-4821"
  }'
```

Or run the interactive demo:

```bash
python examples/demo.py
```

### 5. Docker

```bash
docker compose up --build
```

---

## Project Structure

```
finsight-agent/
├── src/
│   ├── agent/
│   │   ├── graph.py       # LangGraph graph definition
│   │   ├── state.py       # Typed agent state (Pydantic)
│   │   ├── nodes.py       # Graph node functions
│   │   └── tools.py       # Agent tool definitions
│   ├── bedrock/
│   │   └── client.py      # AWS Bedrock client wrapper
│   ├── api/
│   │   └── main.py        # FastAPI application
│   └── utils/
│       └── logger.py      # Structured logging
├── tests/
│   ├── test_agent.py
│   └── test_tools.py
├── examples/
│   └── demo.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Key Design Decisions

**Why LangGraph over vanilla LangChain?**
LangGraph gives you explicit control over agent flow — cycles, conditional edges, checkpointing. For regulated environments, you need to know exactly what path the agent took and why. LangGraph's state graph makes every decision auditable.

**Why AWS Bedrock over direct Anthropic API?**
In a Capital One-style environment, customer data can never leave your VPC. Bedrock runs inside your AWS account — no data retention, FedRAMP High compliant, full CloudTrail audit logs. The Guardrails integration also blocks PII before it reaches the model.

**Why Pydantic v2 for state?**
Type-safe agent state catches bugs at development time, not in production. Pydantic's validation also documents the contract between graph nodes — readable as spec, enforced at runtime.

---

## Extending This

- Swap mock data for real Snowflake / DynamoDB calls
- Add Bedrock Knowledge Bases for internal policy RAG
- Wire in AWS EventBridge for async agent triggers
- Add LangSmith tracing for production observability

---

## License

MIT — use freely, attribute appreciated.
