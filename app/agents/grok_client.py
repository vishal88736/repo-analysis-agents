"""
Production-grade Grok (xAI) API client wrapper.

Features:
  - Async support via openai AsyncOpenAI
  - Retry with exponential backoff (tenacity)
  - Rate limit handling via asyncio.Semaphore
  - Token usage logging
  - Structured JSON output parsing with Pydantic validation
"""

import json
import logging
from typing import TypeVar, Type

from openai import AsyncOpenAI, RateLimitError, APITimeoutError, APIConnectionError
from pydantic import BaseModel
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
import asyncio

from app.config import settings
from app.core.exceptions import LLMError, LLMRateLimitError, LLMResponseValidationError

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class TokenUsageTracker:
    """Thread-safe token usage accumulator."""

    def __init__(self):
        self._lock = asyncio.Lock()
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_requests: int = 0

    async def record(self, prompt_tokens: int, completion_tokens: int):
        async with self._lock:
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_requests += 1

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    def summary(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
        }


class GrokClient:
    """
    Async wrapper around the xAI Grok API (OpenAI-compatible).

    Usage:
        client = GrokClient()
        result = await client.chat("Explain this code", system="You are a code analyst.")
        parsed = await client.structured_chat(prompt, system, MyPydanticModel)
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        max_concurrent: int | None = None,
    ):
        self._api_key = api_key or settings.xai_api_key
        self._base_url = base_url or settings.xai_base_url
        self._model = model or settings.xai_model
        self._max_concurrent = max_concurrent or settings.max_concurrent_llm_calls

        if not self._api_key:
            raise LLMError("XAI_API_KEY is not configured. Set it in .env file.")

        self._client = AsyncOpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
        )
        self._semaphore = asyncio.Semaphore(self._max_concurrent)
        self.token_usage = TokenUsageTracker()

    @retry(
        retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIConnectionError)),
        stop=stop_after_attempt(settings.llm_retry_attempts),
        wait=wait_exponential(multiplier=settings.llm_retry_delay, min=2, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def _call_api(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict | None = None,
    ) -> tuple[str, int, int]:
        """
        Core API call with retry logic and rate-limit semaphore.
        Returns (content, prompt_tokens, completion_tokens).
        """
        async with self._semaphore:
            logger.debug(
                "Grok API call | model=%s | messages=%d | temp=%.2f",
                self._model,
                len(messages),
                temperature or settings.llm_temperature,
            )
            try:
                kwargs = dict(
                    model=self._model,
                    messages=messages,
                    temperature=temperature or settings.llm_temperature,
                    max_tokens=max_tokens or settings.llm_max_tokens,
                )
                if response_format:
                    kwargs["response_format"] = response_format

                response = await self._client.chat.completions.create(**kwargs)

                content = response.choices[0].message.content or ""
                prompt_tokens = response.usage.prompt_tokens if response.usage else 0
                completion_tokens = response.usage.completion_tokens if response.usage else 0

                await self.token_usage.record(prompt_tokens, completion_tokens)

                logger.info(
                    "Grok API success | prompt_tokens=%d | completion_tokens=%d",
                    prompt_tokens,
                    completion_tokens,
                )

                return content, prompt_tokens, completion_tokens

            except RateLimitError as e:
                logger.warning("Grok API rate limit hit: %s", e)
                raise
            except (APITimeoutError, APIConnectionError) as e:
                logger.warning("Grok API connection issue: %s", e)
                raise
            except Exception as e:
                logger.error("Grok API unexpected error: %s", e)
                raise LLMError(f"Unexpected Grok API error: {e}")

    async def chat(
        self,
        prompt: str,
        system: str = "You are a helpful assistant.",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Simple chat completion. Returns raw string."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        content, _, _ = await self._call_api(messages, temperature, max_tokens)
        return content

    async def structured_chat(
        self,
        prompt: str,
        system: str,
        response_model: Type[T],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> T:
        """
        Chat completion with structured JSON output validated against a Pydantic model.
        Requests JSON mode from the API and parses + validates the response.
        """
        schema_description = json.dumps(
            response_model.model_json_schema(), indent=2
        )
        enhanced_system = (
            f"{system}\n\n"
            f"You MUST respond with valid JSON that matches this exact schema:\n"
            f"```json\n{schema_description}\n```\n"
            f"Do NOT include any text outside the JSON object. "
            f"Do NOT wrap it in markdown code fences."
        )

        messages = [
            {"role": "system", "content": enhanced_system},
            {"role": "user", "content": prompt},
        ]

        content, _, _ = await self._call_api(
            messages,
            temperature,
            max_tokens,
            response_format={"type": "json_object"},
        )

        # Parse and validate
        try:
            # Strip potential markdown fences
            cleaned = content.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                # Remove first and last lines (fences)
                cleaned = "\n".join(lines[1:-1])

            data = json.loads(cleaned)
            return response_model.model_validate(data)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse LLM JSON response: %s\nRaw: %s", e, content[:500])
            raise LLMResponseValidationError(
                f"LLM returned invalid JSON: {e}",
                details=content[:1000],
            )
        except Exception as e:
            logger.error("Failed to validate LLM response against schema: %s", e)
            raise LLMResponseValidationError(
                f"LLM response failed Pydantic validation: {e}",
                details=content[:1000],
            )

    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings from Grok API for a list of texts."""
        async with self._semaphore:
            try:
                response = await self._client.embeddings.create(
                    model=settings.xai_embedding_model,
                    input=texts,
                )
                return [item.embedding for item in response.data]
            except Exception as e:
                logger.error("Embedding API error: %s", e)
                raise LLMError(f"Embedding API error: {e}")