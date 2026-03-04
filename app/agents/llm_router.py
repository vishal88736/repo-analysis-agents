"""
Feature 14 & 18: Multi-LLM task router with automatic fallback.

Routes LLM tasks to the appropriate provider (Groq or Gemini) based on
task type, with automatic fallback when a provider fails.

Routing rules:
  FILE_ANALYSIS        → Groq llama-3.1-8b-instant
  ARCHITECTURE_REASON  → Gemini (if available) else Groq llama-3.3-70b-versatile
  QUERY_PLANNING       → Gemini (if available) else Groq llama-3.3-70b-versatile
  RAG_ANSWER           → Groq llama-3.3-70b-versatile
  MERMAID_GENERATION   → Groq llama-3.1-8b-instant
  CONTEXT_COMPRESSION  → Groq llama-3.1-8b-instant
"""

import logging
from typing import TypeVar, Type

from pydantic import BaseModel

from app.agents.groq_client import GroqClient, TaskType, get_model_for_task
from app.core.exceptions import LLMError

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

# Tasks that prefer Gemini when available (large context)
_GEMINI_PREFERRED_TASKS = {
    TaskType.ARCHITECTURE_REASONING,
    TaskType.QUERY_PLANNING,
}


class LLMRouter:
    """
    Routes LLM calls to Groq or Gemini based on task type.

    If Gemini is not configured, all tasks fall back to Groq transparently.
    Feature 18: generate_with_fallback handles provider-level failures.
    """

    def __init__(self, groq: GroqClient, gemini=None):
        self.groq = groq
        self.gemini = gemini  # GeminiClient | None

    def _use_gemini_for(self, task: TaskType) -> bool:
        """Return True if Gemini is available and preferred for this task."""
        return self.gemini is not None and task in _GEMINI_PREFERRED_TASKS

    async def generate(
        self,
        task: TaskType,
        prompt: str,
        system: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Route a text generation task to the appropriate provider."""
        return await self.generate_with_fallback(task, prompt, system, temperature, max_tokens)

    async def structured_generate(
        self,
        task: TaskType,
        prompt: str,
        system: str,
        response_model: Type[T],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> T:
        """Route a structured generation task to the appropriate provider."""
        return await self._structured_with_fallback(
            task, prompt, system, response_model, temperature, max_tokens
        )

    async def generate_with_fallback(
        self,
        task: TaskType,
        prompt: str,
        system: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Feature 18: Generate with automatic provider fallback."""
        if self._use_gemini_for(task):
            try:
                return await self.gemini.generate(
                    prompt, system, temperature=temperature or 0.3
                )
            except Exception as e:
                logger.warning(
                    "Gemini failed for task %s, falling back to Groq: %s", task.value, e
                )
                # Fall through to Groq

        # Groq path (primary or fallback)
        try:
            model = get_model_for_task(task)
            return await self.groq.chat(
                prompt=prompt,
                system=system or "You are a helpful assistant.",
                temperature=temperature,
                max_tokens=max_tokens,
                task=task,
            )
        except Exception as e:
            logger.error("Groq also failed for task %s: %s", task.value, e)
            raise LLMError(f"All providers failed for task {task.value}: {e}")

    async def _structured_with_fallback(
        self,
        task: TaskType,
        prompt: str,
        system: str,
        response_model: Type[T],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> T:
        """Structured generation with automatic provider fallback."""
        if self._use_gemini_for(task):
            try:
                return await self.gemini.structured_generate(
                    prompt, system, response_model, temperature=temperature or 0.3
                )
            except Exception as e:
                logger.warning(
                    "Gemini structured gen failed for task %s, falling back to Groq: %s",
                    task.value, e,
                )
                # Fall through to Groq

        # Groq path
        try:
            return await self.groq.structured_chat(
                prompt=prompt,
                system=system,
                response_model=response_model,
                temperature=temperature,
                max_tokens=max_tokens,
                task=task,
            )
        except Exception as e:
            logger.error("Groq structured gen also failed for task %s: %s", task.value, e)
            raise LLMError(f"All providers failed for task {task.value}: {e}")
