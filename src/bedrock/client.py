"""
AWS Bedrock client wrapper for FinSight.

Wraps the LangChain Bedrock integration with:
- Retry logic with exponential backoff (handles Bedrock throttling)
- Structured logging for every API call (audit trail)
- Cost tracking per invocation
- Streaming support for long-form report generation
- Model fallback chain (Sonnet → Haiku on throttle)

In production, wire in CloudWatch metrics for:
  - bedrock.invocations (count)
  - bedrock.latency_ms (histogram)
  - bedrock.input_tokens / bedrock.output_tokens (for cost attribution)
"""

from __future__ import annotations

import time
from typing import Iterator

from botocore.exceptions import ClientError
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import BaseMessage

from src.utils.logger import get_logger

logger = get_logger(__name__)

# AWS Bedrock model IDs (us-east-1)
MODEL_PRIMARY = "anthropic.claude-3-5-sonnet-20241022-v2:0"
MODEL_FALLBACK = "anthropic.claude-3-haiku-20240307-v1:0"

# Approximate cost per 1M tokens (USD) — update when AWS changes pricing
COST_PER_1M_INPUT = 3.00   # Claude 3.5 Sonnet via Bedrock
COST_PER_1M_OUTPUT = 15.00


class BedrockClient:
    """
    Thread-safe AWS Bedrock client with retry logic and cost tracking.

    Usage:
        client = BedrockClient()
        response = client.invoke(messages)
        print(response.content)
    """

    def __init__(
        self,
        region: str = "us-east-1",
        model_id: str = MODEL_PRIMARY,
        max_tokens: int = 4096,
        temperature: float = 0.1,  # Low temp for consistent compliance outputs
        max_retries: int = 3,
    ):
        self.region = region
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries

        # Initialize primary and fallback LLMs
        self._llm = self._build_llm(model_id)
        self._llm_fallback = self._build_llm(MODEL_FALLBACK)

        logger.info(
            "BedrockClient initialized",
            model=model_id,
            region=region,
            temperature=temperature,
        )

    def _build_llm(self, model_id: str) -> ChatBedrockConverse:
        """Build a ChatBedrockConverse instance for a given model."""
        return ChatBedrockConverse(
            model=model_id,
            region_name=self.region,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

    def invoke(self, messages: list[BaseMessage]) -> BaseMessage:
        """
        Invoke the Bedrock model with retry and fallback logic.

        Handles:
        - ThrottlingException → exponential backoff + retry
        - ModelNotReadyException → immediate fallback to Haiku
        - ServiceUnavailableException → backoff + retry

        Args:
            messages: List of LangChain messages (System, Human, AI).

        Returns:
            AIMessage containing the model's response.
        """
        start_time = time.time()
        attempt = 0
        last_error = None

        while attempt < self.max_retries:
            try:
                llm = self._llm if attempt < 2 else self._llm_fallback
                model_used = self.model_id if attempt < 2 else MODEL_FALLBACK

                response = llm.invoke(messages)

                # Log usage for cost tracking
                latency_ms = int((time.time() - start_time) * 1000)
                usage = getattr(response, "usage_metadata", {}) or {}
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)
                estimated_cost = (
                    (input_tokens / 1_000_000) * COST_PER_1M_INPUT
                    + (output_tokens / 1_000_000) * COST_PER_1M_OUTPUT
                )

                logger.info(
                    "bedrock invocation successful",
                    model=model_used,
                    latency_ms=latency_ms,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    estimated_cost_usd=round(estimated_cost, 6),
                    attempt=attempt + 1,
                )

                return response

            except ClientError as e:
                error_code = e.response["Error"]["Code"]
                last_error = e

                if error_code in ("ThrottlingException", "ServiceUnavailableException"):
                    backoff = 2 ** attempt  # 1s, 2s, 4s
                    logger.warning(
                        "bedrock throttled, backing off",
                        error_code=error_code,
                        backoff_seconds=backoff,
                        attempt=attempt + 1,
                    )
                    time.sleep(backoff)
                    attempt += 1
                    continue

                elif error_code == "ModelNotReadyException":
                    logger.warning(
                        "primary model not ready, falling back to Haiku",
                        model=self.model_id,
                    )
                    attempt = 2  # Skip straight to fallback
                    continue

                else:
                    logger.error(
                        "bedrock invocation failed",
                        error_code=error_code,
                        error_message=str(e),
                    )
                    raise

        raise RuntimeError(
            f"Bedrock invocation failed after {self.max_retries} attempts. "
            f"Last error: {last_error}"
        )

    def invoke_with_tools(
        self,
        messages: list[BaseMessage],
        tools: list,
    ) -> BaseMessage:
        """
        Invoke Bedrock with tool bindings (function calling).

        Used when we want the model to select and call tools itself
        rather than the graph routing calling them explicitly.

        Args:
            messages: Conversation history.
            tools: LangChain tool objects to bind.

        Returns:
            AIMessage, potentially with tool_calls populated.
        """
        llm_with_tools = self._llm.bind_tools(tools)
        return llm_with_tools.invoke(messages)

    def stream(self, messages: list[BaseMessage]) -> Iterator[str]:
        """
        Stream a response from Bedrock token by token.

        Use for long-form report generation where you want to show
        progress rather than waiting for the full response.

        Args:
            messages: Conversation history.

        Yields:
            String chunks as they arrive from Bedrock.
        """
        for chunk in self._llm.stream(messages):
            if hasattr(chunk, "content") and chunk.content:
                yield str(chunk.content)
