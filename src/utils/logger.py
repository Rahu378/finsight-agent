"""
Structured logging for FinSight.

Uses structlog to produce JSON-formatted logs compatible with:
- AWS CloudWatch Logs Insights (query structured fields)
- Capital One's internal SIEM ingestion
- Splunk / Datadog log pipelines

Every log entry includes:
- timestamp (ISO 8601)
- level
- logger name
- message
- any kwargs passed to the log call (structured fields)
"""

from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(level: str = "INFO") -> None:
    """Configure structlog for JSON output. Call once at app startup."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(level)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a structured logger bound to a module name.

    Usage:
        logger = get_logger(__name__)
        logger.info("processing started", account_id="ACC-123", amount=5000)
    """
    return structlog.get_logger(name)


# Configure on import
configure_logging()
