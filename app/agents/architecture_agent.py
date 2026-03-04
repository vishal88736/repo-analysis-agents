"""
Architecture Summary Agent — global codebase overview.

Feature 14: Uses LLMRouter for multi-provider support.
Feature 15: When Gemini is available, sends full repo map + all summaries
            to leverage Gemini's large context window.
"""

import json
import logging
from typing import TYPE_CHECKING

from app.agents.groq_client import GroqClient, TaskType
from app.schemas.analysis import ArchitectureSummary, FileAnalysisResult, RepoMap, CompactFileSummary

logger = logging.getLogger(__name__)

ARCHITECTURE_SYSTEM_PROMPT = """\
You are a senior software architect. Given file analysis reports from a codebase, \
produce a structured architecture summary.

Include:
1. overview: 2-4 paragraph high-level description of the codebase purpose, \
architecture style, and component interactions.
2. key_components: Most important modules/files and their roles.
3. design_patterns: Any patterns observed (MVC, microservices, pub-sub, etc.).
4. entry_points: Where execution begins (main(), app startup, CLI). Each must have \
file_path, function_name, reason.
5. technology_stack: Languages, frameworks, libraries used.

RULES:
- Base analysis ONLY on provided file reports. Do NOT guess.
- Reference actual file paths and function names.
- Respond with ONLY valid JSON.\
"""


async def generate_architecture_summary(
    groq: GroqClient,
    file_analyses: list[FileAnalysisResult],
    router=None,
    repo_map: RepoMap | None = None,
    compact_summaries: list[CompactFileSummary] | None = None,
) -> ArchitectureSummary:
    """
    Generate architecture summary using best available LLM.

    Feature 15: If router with Gemini is available, sends full context (repo map
    + all compact summaries + full file analyses). Otherwise falls back to Groq
    with truncated context.
    """
    summaries = []
    all_deps = set()
    for fa in file_analyses:
        summaries.append({
            "file": fa.file_path,
            "summary": fa.summary[:200],
            "functions": [f.name for f in fa.functions],
            "classes": [c.name for c in fa.classes],
            "deps": fa.external_dependencies,
        })
        all_deps.update(fa.external_dependencies)

    if router is not None:
        # Feature 15: Use LLM router (may use Gemini for large context)
        return await _generate_with_router(
            router, file_analyses, summaries, all_deps, repo_map, compact_summaries
        )
    else:
        # Backward-compatible path: Groq only
        return await _generate_with_groq(groq, file_analyses, summaries, all_deps)


async def _generate_with_router(
    router,
    file_analyses: list[FileAnalysisResult],
    summaries: list[dict],
    all_deps: set,
    repo_map: RepoMap | None,
    compact_summaries: list[CompactFileSummary] | None,
) -> ArchitectureSummary:
    """Feature 15: Build rich context for Gemini when available."""
    # Build the prompt with as much context as possible
    repo_map_text = ""
    if repo_map and repo_map.files:
        entries = [
            f"  {path}: {info.language} ({info.size}B)"
            for path, info in list(repo_map.files.items())[:300]
        ]
        repo_map_text = "REPOSITORY MAP:\n" + "\n".join(entries) + "\n\n"

    compact_text = ""
    if compact_summaries:
        items = [
            f"  {s.file_path}: {s.purpose}"
            for s in compact_summaries[:200]
        ]
        compact_text = "COMPACT FILE SUMMARIES:\n" + "\n".join(items) + "\n\n"

    max_files = 100
    note = ""
    if len(summaries) > max_files:
        summaries_to_show = summaries[:max_files]
        note = f"\n[Showing {max_files} of {len(file_analyses)} files for brevity]"
    else:
        summaries_to_show = summaries

    prompt = f"""\
Analyze this codebase and generate an architecture summary.

{repo_map_text}{compact_text}Total files: {len(file_analyses)}
External dependencies: {', '.join(sorted(all_deps)[:40])}

DETAILED FILE REPORTS:
{json.dumps(summaries_to_show, indent=1)}
{note}

Produce JSON with: overview, key_components, design_patterns, entry_points, technology_stack
"""

    try:
        return await router.structured_generate(
            task=TaskType.ARCHITECTURE_REASONING,
            prompt=prompt,
            system=ARCHITECTURE_SYSTEM_PROMPT,
            response_model=ArchitectureSummary,
            temperature=0.3,
        )
    except Exception as e:
        logger.error("Architecture summary (router) failed: %s", e)
        return ArchitectureSummary(overview=f"Generation failed: {e}")


async def _generate_with_groq(
    groq: GroqClient,
    file_analyses: list[FileAnalysisResult],
    summaries: list[dict],
    all_deps: set,
) -> ArchitectureSummary:
    """Backward-compatible Groq-only path with truncation."""
    max_files = 60
    note = ""
    if len(summaries) > max_files:
        summaries = summaries[:max_files]
        note = f"\n[Showing {max_files} of {len(file_analyses)} files for brevity]"

    prompt = f"""\
Analyze these codebase file reports and generate an architecture summary.

Total files: {len(file_analyses)}
External dependencies: {', '.join(sorted(all_deps)[:40])}

FILE REPORTS:
{json.dumps(summaries, indent=1)}
{note}

Produce JSON with: overview, key_components, design_patterns, entry_points, technology_stack
"""

    try:
        return await groq.structured_chat(
            prompt=prompt,
            system=ARCHITECTURE_SYSTEM_PROMPT,
            response_model=ArchitectureSummary,
            temperature=0.3,
            max_tokens=4096,
            task=TaskType.ARCHITECTURE_REASONING,
        )
    except Exception as e:
        logger.error("Architecture summary failed: %s", e)
        return ArchitectureSummary(overview=f"Generation failed: {e}")
