"""
Production-grade Groq API client wrapper.

Uses the OFFICIAL groq Python SDK (pip install groq).
Runs open-source models (Llama 3.3, Mixtral, Gemma, etc.) on Groq LPU hardware.

Features:
  - AsyncGroq client for non-blocking calls
  - Retry with exponential backoff (tenacity) for rate limits & transient errors
  - asyncio.Semaphore for concurrency control
  - Token usage tracking across all calls
  - Structured JSON output with Pydantic validation
  - Model selection (heavy model for analysis, fast model for simple tasks)
"""

import json
import logging
import asyncio
from typing import TypeVar, Type

from groq import AsyncGroq, RateLimitError, APITimeoutError, APIConnectionError, APIStatusError
from pydantic import BaseModel
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

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


class GroqClient:
    """
    Async wrapper around the Groq API using the official groq-python SDK.

    Groq runs open-source models (Llama, Mixtral, Gemma, etc.) at extreme speed
    on their custom LPU (Language Processing Unit) hardware.

    Usage:
        client = GroqClient()
        result = await client.chat("Explain this code", system="You are a code analyst.")
        parsed = await client.structured_chat(prompt, system, MyPydanticModel)
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        fast_model: str | None = None,
        max_concurrent: int | None = None,
    ):
        self._api_key = api_key or settings.groq_api_key
        self._model = model or settings.groq_model
        self._fast_model = fast_model or settings.groq_fast_model
        self._max_concurrent = max_concurrent or settings.max_concurrent_llm_calls

        if not self._api_key:
            raise LLMError("GROQ_API_KEY is not configured. Get one at https://console.groq.com")

        # Official AsyncGroq client
        self._client = AsyncGroq(api_key=self._api_key)
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
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict | None = None,
    ) -> tuple[str, int, int]:
        """
        Core API call with retry logic and rate-limit semaphore.

        The semaphore ensures we never exceed max_concurrent_llm_calls parallel
        requests to Groq, even when processing a full batch concurrently.

        Returns: (content, prompt_tokens, completion_tokens)
        """
        used_model = model or self._model

        async with self._semaphore:
            logger.debug(
                "Groq API call | model=%s | messages=%d | temp=%.2f",
                used_model,
                len(messages),
                temperature or settings.groq_temperature,
            )
            try:
                kwargs: dict = {
                    "model": used_model,
                    "messages": messages,
                    "temperature": temperature or settings.groq_temperature,
                    "max_tokens": max_tokens or settings.groq_max_tokens,
                }
                if response_format:
                    kwargs["response_format"] = response_format

                response = await self._client.chat.completions.create(**kwargs)

                content = response.choices[0].message.content or ""
                prompt_tokens = response.usage.prompt_tokens if response.usage else 0
                completion_tokens = response.usage.completion_tokens if response.usage else 0

                await self.token_usage.record(prompt_tokens, completion_tokens)

                logger.info(
                    "Groq API success | model=%s | prompt_tok=%d | completion_tok=%d",
                    used_model,
                    prompt_tokens,
                    completion_tokens,
                )

                return content, prompt_tokens, completion_tokens

            except RateLimitError as e:
                logger.warning("Groq rate limit hit (will retry): %s", e)
                raise  # tenacity will retry
            except (APITimeoutError, APIConnectionError) as e:
                logger.warning("Groq connection issue (will retry): %s", e)
                raise  # tenacity will retry
            except APIStatusError as e:
                logger.error("Groq API status error: %s", e)
                raise LLMError(f"Groq API error: {e.status_code} - {e.message}")
            except Exception as e:
                logger.error("Groq API unexpected error: %s", e)
                raise LLMError(f"Unexpected Groq API error: {e}")

    async def chat(
        self,
        prompt: str,
        system: str = "You are a helpful assistant.",
        temperature: float | None = None,
        max_tokens: int | None = None,
        use_fast_model: bool = False,
    ) -> str:
        """
        Simple chat completion. Returns raw string.

        Args:
            use_fast_model: If True, uses the smaller/faster model (e.g., llama-3.1-8b-instant)
                           for simpler tasks like Mermaid generation.
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        model = self._fast_model if use_fast_model else self._model
        content, _, _ = await self._call_api(messages, model=model, temperature=temperature, max_tokens=max_tokens)
        return content

    async def structured_chat(
        self,
        prompt: str,
        system: str,
        response_model: Type[T],
        temperature: float | None = None,
        max_tokens: int | None = None,
        use_fast_model: bool = False,
    ) -> T:
        """
        Chat completion with structured JSON output validated against a Pydantic model.

        Uses Groq's JSON mode (response_format={"type": "json_object"}) and
        validates the output against the provided Pydantic schema.
        """
        schema_description = json.dumps(
            response_model.model_json_schema(), indent=2
        )
        enhanced_system = (
            f"{system}\n\n"
            f"You MUST respond with valid JSON that conforms to this schema:\n"
            f"```json\n{schema_description}\n```\n"
            f"Respond with ONLY the JSON object. No markdown fences, no explanations."
        )

        messages = [
            {"role": "system", "content": enhanced_system},
            {"role": "user", "content": prompt},
        ]

        model = self._fast_model if use_fast_model else self._model

        content, _, _ = await self._call_api(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )

        # Parse and validate against Pydantic model
        try:
            cleaned = content.strip()
            # Strip markdown fences if the model accidentally includes them
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].strip()

            data = json.loads(cleaned)
            return response_model.model_validate(data)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse Groq JSON response: %s\nRaw: %s", e, content[:500])
            raise LLMResponseValidationError(
                f"LLM returned invalid JSON: {e}",
                details=content[:1000],
            )
        except Exception as e:
            logger.error("Pydantic validation failed on Groq response: %s", e)
            raise LLMResponseValidationError(
                f"LLM response failed Pydantic validation: {e}",
                details=content[:1000],
            )