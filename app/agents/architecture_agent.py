"""
Architecture Summary Agent — generates a system-wide architecture report
from all combined file analyses.
"""

import json
import logging

from app.agents.grok_client import GrokClient
from app.schemas.analysis import ArchitectureSummary, FileAnalysisResult

logger = logging.getLogger(__name__)

ARCHITECTURE_SYSTEM_PROMPT = """\
You are a senior software architect. Given a collection of file analysis reports from a \
codebase, produce a structured architecture summary.

Your report must include:
1. overview: A 2-4 paragraph high-level description of what this codebase does, its \
architecture style, and how components interact.
2. key_components: List of the most important modules/files and what they do.
3. design_patterns: Any design patterns observed (MVC, microservices, pub-sub, etc.).
4. entry_points: Where execution begins — main(), app startup, CLI entry, etc. Each \
entry point should have file_path, function_name, and reason.
5. technology_stack: Languages, frameworks, and libraries used.

RULES:
- Base your analysis ONLY on the provided file reports. Do NOT guess or hallucinate.
- Be specific — reference actual file paths and function names.
- Respond with ONLY a valid JSON object.\
"""


async def generate_architecture_summary(
    grok: GrokClient,
    file_analyses: list[FileAnalysisResult],
) -> ArchitectureSummary:
    """Generate a global architecture summary from all file analyses."""
    # Build a condensed representation to fit in context
    summaries = []
    all_deps = set()
    for fa in file_analyses:
        entry = {
            "file": fa.file_path,
            "summary": fa.summary[:200],
            "functions": [f.name for f in fa.functions],
            "classes": [c.name for c in fa.classes],
            "dependencies": fa.external_dependencies,
        }
        summaries.append(entry)
        all_deps.update(fa.external_dependencies)

    # Truncate if too many files
    max_files = 80
    if len(summaries) > max_files:
        summaries = summaries[:max_files]
        note = f"\n\n[NOTE: Showing {max_files} of {len(file_analyses)} files for brevity]"
    else:
        note = ""

    prompt = f"""\
Analyze the following codebase file reports and generate a comprehensive architecture summary.

Total files analyzed: {len(file_analyses)}
All external dependencies: {', '.join(sorted(all_deps)[:50])}

FILE REPORTS:
{json.dumps(summaries, indent=1)}
{note}

Produce a JSON object with these exact fields:
- overview (string: 2-4 paragraphs)
- key_components (array of strings)
- design_patterns (array of strings)
- entry_points (array of objects with: file_path, function_name, reason)
- technology_stack (array of strings)
"""

    try:
        result = await grok.structured_chat(
            prompt=prompt,
            system=ARCHITECTURE_SYSTEM_PROMPT,
            response_model=ArchitectureSummary,
            temperature=0.3,
            max_tokens=4096,
        )
        logger.info("Architecture summary generated successfully")
        return result
    except Exception as e:
        logger.error("Architecture summary generation failed: %s", e)
        return ArchitectureSummary(
            overview=f"Architecture summary generation failed: {e}",
        )