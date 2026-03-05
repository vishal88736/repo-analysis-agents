"""
Feature 14: LLM Router — routes tasks to appropriate provider.
Feature 7: Model routing for cost optimization.
Feature 18: Automatic fallback (Groq → Gemini).

All agents call the router instead of calling Groq/Gemini directly.
"""

import logging
from enum import Enum
from typing import TypeVar, Type

from pydantic import BaseModel

from app.config import settings
from app.agents.groq_client import GroqClient
from app.core.exceptions import LLMError

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class TaskType(str, Enum):
    FILE_ANALYSIS = "file_analysis"
    ARCHITECTURE = "architecture"
    MERMAID = "mermaid"
    QUERY_PLANNING = "query_planning"
    QUERY_ANSWER = "query_answer"
    RAG_ANSWER = "rag_answer"


# Default routing: task → (provider, model_hint)
# "groq_fast" = llama-3.1-8b-instant
# "groq_heavy" = llama-3.3-70b-versatile
# "gemini" = gemini-2.0-flash (large context)
DEFAULT_ROUTING = {
    TaskType.FILE_ANALYSIS: "groq_fast",
    TaskType.ARCHITECTURE: "gemini",
    TaskType.MERMAID: "groq_fast",
    TaskType.QUERY_PLANNING: "gemini",
    TaskType.QUERY_ANSWER: "groq_heavy",
    TaskType.RAG_ANSWER: "groq_heavy",
}


class LLMRouter:
    """
    Central LLM dispatch — routes by task type with automatic fallback.

    Priority chain for each call:
      1. Primary provider for task (from routing table)
      2. If Gemini unavailable or fails → Groq heavy
      3. If Groq heavy fails (rate limit) → Groq fast
    """

    def __init__(self, groq: GroqClient):
        self._groq = groq
        self._gemini = None
        self._gemini_init_attempted = False

    def _get_gemini(self):
        """Lazy init Gemini — only created when first needed."""
        if self._gemini is not None:
            return self._gemini
        if self._gemini_init_attempted:
            return None

        self._gemini_init_attempted = True
        if not settings.gemini_available:
            logger.info("Gemini not configured — all tasks will use Groq")
            return None

        try:
            from app.agents.gemini_client import GeminiClient
            self._gemini = GeminiClient()
            logger.info("Gemini client initialized: %s", settings.gemini_model)
            return self._gemini
        except Exception as e:
            logger.warning("Gemini init failed, using Groq only: %s", e)
            return None

    async def chat(
        self,
        task: TaskType,
        prompt: str,
        system: str = "You are a helpful assistant.",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Route a chat completion to the appropriate provider."""
        route = DEFAULT_ROUTING.get(task, "groq_heavy")
        temp = temperature or settings.groq_temperature

        # Try primary route
        try:
            if route == "gemini":
                gemini = self._get_gemini()
                if gemini:
                    logger.debug("Routing %s → Gemini", task.value)
                    return await gemini.generate(prompt, system, temp)
                else:
                    # Fallback to Groq heavy
                    logger.debug("Routing %s → Groq heavy (Gemini unavailable)", task.value)
                    return await self._groq.chat(prompt, system, temp, max_tokens)

            elif route == "groq_fast":
                logger.debug("Routing %s → Groq fast", task.value)
                return await self._groq.chat(prompt, system, temp, max_tokens, use_fast_model=True)

            else:  # groq_heavy
                logger.debug("Routing %s → Groq heavy", task.value)
                return await self._groq.chat(prompt, system, temp, max_tokens)

        except Exception as primary_error:
            logger.warning("Primary route failed for %s: %s", task.value, primary_error)
            return await self._fallback_chat(task, prompt, system, temp, max_tokens, primary_error)

    async def _fallback_chat(
        self, task, prompt, system, temperature, max_tokens, original_error
    ) -> str:
        """Fallback chain: Gemini → Groq heavy → Groq fast."""
        # Try Gemini if not already tried
        gemini = self._get_gemini()
        if gemini:
            try:
                logger.info("Fallback %s → Gemini", task.value)
                return await gemini.generate(prompt, system, temperature)
            except Exception as e:
                logger.warning("Gemini fallback failed: %s", e)

        # Try Groq heavy
        try:
            logger.info("Fallback %s → Groq heavy", task.value)
            return await self._groq.chat(prompt, system, temperature, max_tokens)
        except Exception:
            pass

        # Last resort: Groq fast
        try:
            logger.info("Fallback %s → Groq fast", task.value)
            return await self._groq.chat(prompt, system, temperature, max_tokens, use_fast_model=True)
        except Exception as e:
            raise LLMError(f"All LLM providers failed for {task.value}: {e}") from original_error

    async def structured_chat(
        self,
        task: TaskType,
        prompt: str,
        system: str,
        response_model: Type[T],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> T:
        """Route a structured JSON completion with Pydantic validation."""
        route = DEFAULT_ROUTING.get(task, "groq_heavy")
        temp = temperature or settings.groq_temperature

        try:
            if route == "gemini":
                gemini = self._get_gemini()
                if gemini:
                    return await gemini.structured_generate(prompt, system, response_model, temp)

            # Default: Groq
            use_fast = route == "groq_fast"
            return await self._groq.structured_chat(
                prompt=prompt,
                system=system,
                response_model=response_model,
                temperature=temp,
                max_tokens=max_tokens,
                use_fast_model=use_fast,
            )
        except Exception as primary_error:
            logger.warning("Structured route failed for %s: %s", task.value, primary_error)
            # Fallback to Groq heavy
            try:
                return await self._groq.structured_chat(
                    prompt=prompt,
                    system=system,
                    response_model=response_model,
                    temperature=temp,
                    max_tokens=max_tokens,
                )
            except Exception:
                # Last fallback: Groq fast
                return await self._groq.structured_chat(
                    prompt=prompt,
                    system=system,
                    response_model=response_model,
                    temperature=temp,
                    max_tokens=max_tokens,
                    use_fast_model=True,
                )