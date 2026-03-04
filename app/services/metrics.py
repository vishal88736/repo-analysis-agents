"""
Feature 12 & 17: Token usage metrics with multi-provider support.

Tracks prompt tokens, completion tokens, and durations per pipeline stage
across Groq and Gemini providers.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class StageMetrics:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    duration_seconds: float = 0.0
    calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "duration_seconds": round(self.duration_seconds, 3),
        }


@dataclass
class ProviderMetrics:
    """Per-provider token usage tracker."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_requests: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def record(self, prompt: int, completion: int) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_requests += 1

    def to_dict(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


class AnalysisMetrics:
    """Per-stage token usage and duration tracker."""

    def __init__(self):
        self.stages: dict[str, StageMetrics] = {}
        self._lock = asyncio.Lock()

    async def record(
        self,
        stage: str,
        prompt_tokens: int,
        completion_tokens: int,
        duration: float,
    ) -> None:
        async with self._lock:
            if stage not in self.stages:
                self.stages[stage] = StageMetrics()
            m = self.stages[stage]
            m.prompt_tokens += prompt_tokens
            m.completion_tokens += completion_tokens
            m.duration_seconds += duration
            m.calls += 1

    def summary(self) -> dict:
        total_prompt = sum(s.prompt_tokens for s in self.stages.values())
        total_completion = sum(s.completion_tokens for s in self.stages.values())
        total_duration = sum(s.duration_seconds for s in self.stages.values())
        return {
            "stages": {name: s.to_dict() for name, s in self.stages.items()},
            "totals": {
                "prompt_tokens": total_prompt,
                "completion_tokens": total_completion,
                "total_tokens": total_prompt + total_completion,
                "duration_seconds": round(total_duration, 3),
            },
        }


class MultiProviderMetrics:
    """Feature 17: Tracks usage across both Groq and Gemini providers."""

    def __init__(self):
        self.groq_usage = ProviderMetrics()
        self.gemini_usage = ProviderMetrics()

    def total_cost_estimate(self) -> dict:
        """
        Estimate costs. Groq free tier has no per-token cost.
        Gemini flash: ~$0.075/M input tokens, ~$0.30/M output tokens (approximate).
        """
        # Gemini 2.0 Flash pricing as of 2024 (may vary by model/region; see https://ai.google.dev/pricing)
        # Input: $0.075 / 1M tokens, Output: $0.30 / 1M tokens (standard tier)
        gemini_input_cost = self.gemini_usage.prompt_tokens * 0.075 / 1_000_000
        gemini_output_cost = self.gemini_usage.completion_tokens * 0.30 / 1_000_000
        return {
            "groq": {
                **self.groq_usage.to_dict(),
                "estimated_cost_usd": 0.0,  # Free tier
            },
            "gemini": {
                **self.gemini_usage.to_dict(),
                "estimated_cost_usd": round(gemini_input_cost + gemini_output_cost, 6),
            },
        }


class _Timer:
    """Simple context manager for measuring wall-clock duration."""

    def __init__(self):
        self.elapsed: float = 0.0

    def __enter__(self):
        self._start = time.monotonic()
        return self

    def __exit__(self, *_):
        self.elapsed = time.monotonic() - self._start
