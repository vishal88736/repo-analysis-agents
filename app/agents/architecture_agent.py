"""
Architecture Summary Agent — global codebase overview via Groq (Llama 3.3 70B).
"""

import json
import logging

from app.agents.groq_client import GroqClient
from app.schemas.analysis import ArchitectureSummary, FileAnalysisResult

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
) -> ArchitectureSummary:
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

    # Limit to avoid context overflow on Groq models
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
        )
    except Exception as e:
        logger.error("Architecture summary failed: %s", e)
        return ArchitectureSummary(overview=f"Generation failed: {e}")