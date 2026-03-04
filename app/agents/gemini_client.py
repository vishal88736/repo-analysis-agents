"""
Feature 13: Google Gemini API client wrapper.

Gemini is used for tasks requiring large context windows (1M+ tokens):
- Architecture analysis
- Query planning with full repo context

This client is optional — if GEMINI_API_KEY is not set, the system falls back to Groq.
"""

import json
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar, Type

from pydantic import BaseModel

from app.config import settings
from app.core.exceptions import LLMError, LLMResponseValidationError

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class GeminiTokenUsageTracker:
    """Simple token usage tracker for Gemini (synchronous, no async needed)."""

    def __init__(self):
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_requests: int = 0

    def record(self, prompt_tokens: int, completion_tokens: int) -> None:
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


class GeminiClient:
    """
    Async wrapper around the Google Generative AI SDK.

    The google-generativeai library is synchronous, so we use run_in_executor
    to avoid blocking the event loop.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ):
        self._api_key = api_key or settings.gemini_api_key
        self._model_name = model or settings.gemini_model
        self._executor = ThreadPoolExecutor(max_workers=3)
        self.token_usage = GeminiTokenUsageTracker()

        if not self._api_key:
            raise LLMError(
                "GEMINI_API_KEY is not configured. "
                "Get one at https://aistudio.google.com/app/apikey"
            )

        try:
            import google.generativeai as genai
            genai.configure(api_key=self._api_key)
            self._model = genai.GenerativeModel(self._model_name)
            self._genai = genai
        except ImportError:
            raise LLMError(
                "google-generativeai is not installed. "
                "Run: pip install google-generativeai>=0.8.0"
            )

    def _sync_generate(self, prompt: str, system: str = "", temperature: float = 0.3) -> tuple[str, int, int]:
        """Synchronous generation call (runs in executor thread)."""
        import google.generativeai as genai
        from google.generativeai.types import GenerationConfig

        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        config = GenerationConfig(temperature=temperature)

        response = self._model.generate_content(full_prompt, generation_config=config)
        text = response.text or ""

        # Extract token counts if available
        prompt_tokens = 0
        completion_tokens = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            prompt_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
            completion_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0
        else:
            logger.debug("Gemini response did not include usage_metadata; token counts unavailable")

        return text, prompt_tokens, completion_tokens

    async def generate(
        self, prompt: str, system: str = "", temperature: float = 0.3
    ) -> str:
        """Async text generation via Gemini."""
        loop = asyncio.get_event_loop()
        try:
            text, prompt_tok, completion_tok = await loop.run_in_executor(
                self._executor,
                lambda: self._sync_generate(prompt, system, temperature),
            )
            self.token_usage.record(prompt_tok, completion_tok)
            logger.info(
                "Gemini success | model=%s | prompt_tok=%d | completion_tok=%d",
                self._model_name, prompt_tok, completion_tok,
            )
            return text
        except Exception as e:
            logger.error("Gemini generation failed: %s", e)
            raise LLMError(f"Gemini API error: {e}")

    async def structured_generate(
        self,
        prompt: str,
        system: str,
        response_model: Type[T],
        temperature: float = 0.3,
    ) -> T:
        """Async structured JSON generation with Pydantic validation."""
        schema_description = json.dumps(response_model.model_json_schema(), indent=2)
        enhanced_system = (
            f"{system}\n\n"
            f"You MUST respond with valid JSON that conforms to this schema:\n"
            f"```json\n{schema_description}\n```\n"
            f"Respond with ONLY the JSON object. No markdown fences, no explanations."
        )

        raw = await self.generate(prompt, enhanced_system, temperature)

        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].strip()

            data = json.loads(cleaned)
            return response_model.model_validate(data)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse Gemini JSON response: %s\nRaw: %s", e, raw[:500])
            raise LLMResponseValidationError(
                f"Gemini returned invalid JSON: {e}",
                details=raw[:1000],
            )
        except Exception as e:
            logger.error("Pydantic validation failed on Gemini response: %s", e)
            raise LLMResponseValidationError(
                f"Gemini response failed Pydantic validation: {e}",
                details=raw[:1000],
            )
