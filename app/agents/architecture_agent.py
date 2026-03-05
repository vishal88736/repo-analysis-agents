"""
Architecture Summary Agent — strict anti-hallucination version.
"""

import json
import logging

from app.agents.llm_router import LLMRouter, TaskType
from app.schemas.analysis import ArchitectureSummary, FileAnalysisResult

logger = logging.getLogger(__name__)

ARCHITECTURE_SYSTEM_PROMPT = """\
You are a senior software architect. Given file analysis reports from a codebase, \
produce a structured architecture summary.

CRITICAL ANTI-HALLUCINATION RULES:
1. ONLY reference files, functions, classes, and dependencies that appear in the \
provided file reports. Do NOT invent or assume anything.
2. If a file is listed in the reports, you may reference it. If not listed, it does NOT exist.
3. For entry_points: ONLY list files and functions that actually exist in the reports. \
Each entry point must have a file_path that matches an actual file from the reports.
4. For technology_stack: ONLY list technologies that are actually imported or used in the code. \
Do NOT add frameworks just because they seem likely.
5. For key_components: ONLY list files/modules that are actually in the reports.
6. For design_patterns: ONLY identify patterns you can actually see in the provided data.

Include:
1. overview: 2-4 paragraph description of what this codebase ACTUALLY does based on the reports.
2. key_components: The actual files/modules and their roles (from the reports only).
3. design_patterns: Patterns visible in the provided data.
4. entry_points: Where execution begins. Each must have file_path (must match an actual file), \
function_name (must be a real function), reason.
5. technology_stack: Languages, frameworks, libraries ACTUALLY imported in the code.

Respond with ONLY valid JSON. No markdown fences, no explanations.\
"""


async def generate_architecture_summary(
    router: LLMRouter,
    file_analyses: list[FileAnalysisResult],
) -> ArchitectureSummary:
    summaries = []
    all_deps = set()
    actual_files = set()

    for fa in file_analyses:
        actual_files.add(fa.file_path)
        summaries.append({
            "file": fa.file_path,
            "summary": fa.summary[:300],
            "functions": [f.name for f in fa.functions],
            "classes": [c.name for c in fa.classes],
            "deps": fa.external_dependencies,
        })
        all_deps.update(fa.external_dependencies)

    max_files = 120
    note = ""
    if len(summaries) > max_files:
        summaries = summaries[:max_files]
        note = f"\n[Showing {max_files} of {len(file_analyses)} files for brevity]"

    # Explicitly list all actual files to prevent hallucination
    file_list = json.dumps(sorted(actual_files), indent=1)

    prompt = f"""\
Analyze these codebase file reports and generate an architecture summary.

COMPLETE LIST OF ACTUAL FILES (reference ONLY these):
{file_list}

Total files: {len(file_analyses)}
External dependencies actually imported: {', '.join(sorted(all_deps)[:40])}

FILE REPORTS:
{json.dumps(summaries, indent=1)}
{note}

REMINDER: entry_points must use file_path values from the actual file list above.
Do NOT reference files like "main.js", "utils.js", "server.js" unless they are in the list.

Produce JSON with: overview, key_components, design_patterns, entry_points, technology_stack
"""

    try:
        result = await router.structured_chat(
            task=TaskType.ARCHITECTURE,
            prompt=prompt,
            system=ARCHITECTURE_SYSTEM_PROMPT,
            response_model=ArchitectureSummary,
            temperature=0.2,  # Low temperature
            max_tokens=4096,
        )

        # POST-VALIDATION: Remove entry points that reference non-existent files
        validated_eps = []
        for ep in result.entry_points:
            if ep.file_path in actual_files:
                validated_eps.append(ep)
            else:
                logger.warning("Removed hallucinated entry point: %s", ep.file_path)
        result.entry_points = validated_eps

        # Validate key_components: should reference actual files
        validated_components = []
        for comp in result.key_components:
            # Keep it if it mentions an actual file or is a general description
            if any(f in comp for f in actual_files) or len(comp) > 30:
                validated_components.append(comp)
            else:
                # Check if it's a reasonable short description
                validated_components.append(comp)
        result.key_components = validated_components

        return result

    except Exception as e:
        logger.error("Architecture summary failed: %s", e)
        return ArchitectureSummary(overview=f"Generation failed: {e}")