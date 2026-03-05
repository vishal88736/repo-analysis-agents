"""
Architecture Agent — generates execution flow, data flow, tech profile,
component interactions, and the standard architecture summary.
"""

import json
import logging

from app.agents.llm_router import LLMRouter, TaskType
from app.schemas.analysis import (
    ArchitectureSummary, FileAnalysisResult, FileInteraction,
    ExecutionFlow, ExecutionStep, DataFlow, DataFlowStep, TechnologyProfile,
)

logger = logging.getLogger(__name__)

ARCHITECTURE_SYSTEM_PROMPT = """\
You are a senior software architect. Given file analysis reports from a codebase, \
produce a comprehensive architecture summary.

CRITICAL RULES:
1. ONLY reference files, functions, and dependencies from the provided reports.
2. Do NOT invent files or components that aren't listed.
3. Every entry_point must have a file_path matching an actual file.
4. technology_stack lists raw library names.
5. technology_profile classifies the PLATFORM (e.g., "Chrome Extension", "Flask API").
6. execution_flow describes the RUNTIME WORKFLOW step by step.
7. data_flow describes how DATA moves through the system.
8. file_interactions describes how files depend on each other.
9. component_interaction_summary is a prose description of how all pieces fit together.

Respond with ONLY valid JSON.\
"""


def _collect_all_interactions(file_analyses: list[FileAnalysisResult]) -> list[FileInteraction]:
    """Merge all file interactions from individual analyses."""
    all_interactions = []
    seen = set()
    for fa in file_analyses:
        for inter in fa.file_interactions:
            key = (inter.source_file, inter.target_file, inter.interaction_type)
            if key not in seen:
                seen.add(key)
                all_interactions.append(inter)
    return all_interactions


def _detect_platform(
    file_analyses: list[FileAnalysisResult],
    all_deps: set[str],
) -> dict:
    """
    Heuristic-based platform detection.
    Returns hints for the LLM to build TechnologyProfile.
    """
    hints = {
        "platform": "unknown",
        "platform_category": "unknown",
        "runtime_environment": "unknown",
        "apis_used": [],
    }

    all_files = {fa.file_path for fa in file_analyses}
    all_content_hints = set()
    for fa in file_analyses:
        for func in fa.functions:
            all_content_hints.update(func.calls)
            all_content_hints.update(func.imports_used)
        all_content_hints.update(fa.external_dependencies)

    # Chrome Extension detection
    if "manifest.json" in all_files:
        for fa in file_analyses:
            if fa.file_path == "manifest.json":
                if "manifest_version" in fa.summary.lower() or "chrome" in fa.summary.lower():
                    hints["platform"] = "Chrome Extension"
                    hints["platform_category"] = "Browser Extension"
                    hints["runtime_environment"] = "Browser (Chrome)"
                    hints["apis_used"].append("Chrome Extension API")
                    break

    # Check for chrome API usage
    chrome_apis = [h for h in all_content_hints if "chrome." in h.lower()]
    if chrome_apis:
        hints["platform"] = hints.get("platform", "") or "Chrome Extension"
        hints["platform_category"] = "Browser Extension"
        hints["runtime_environment"] = "Browser (Chrome)"
        for api in chrome_apis:
            if api not in hints["apis_used"]:
                hints["apis_used"].append(api)

    # Flask/FastAPI/Django detection
    if "fastapi" in all_deps or "FastAPI" in all_deps:
        hints["platform"] = "FastAPI Web Server"
        hints["platform_category"] = "Web Application / API"
        hints["runtime_environment"] = "Python Runtime (Server)"
    elif "flask" in all_deps or "Flask" in all_deps:
        hints["platform"] = "Flask Web Server"
        hints["platform_category"] = "Web Application / API"
        hints["runtime_environment"] = "Python Runtime (Server)"
    elif "django" in all_deps:
        hints["platform"] = "Django Web Application"
        hints["platform_category"] = "Web Application"
        hints["runtime_environment"] = "Python Runtime (Server)"

    # React/Next.js/Vue detection
    if "react" in all_deps:
        hints["platform"] = "React Application"
        hints["platform_category"] = "Single Page Application"
        hints["runtime_environment"] = "Browser"
    if "next" in all_deps:
        hints["platform"] = "Next.js Application"
        hints["platform_category"] = "Full-Stack Web Application"

    # Node.js / Express detection
    if "express" in all_deps:
        hints["platform"] = "Express.js Server"
        hints["platform_category"] = "Web Application / API"
        hints["runtime_environment"] = "Node.js"

    # Fetch API / jsPDF etc.
    if any("fetch" in h.lower() for h in all_content_hints):
        if "Fetch API" not in hints["apis_used"]:
            hints["apis_used"].append("Fetch API")
    if "jsPDF" in all_deps or "jspdf" in str(all_deps).lower():
        if "jsPDF" not in hints["apis_used"]:
            hints["apis_used"].append("jsPDF PDF Generation")

    return hints


async def generate_architecture_summary(
    router: LLMRouter,
    file_analyses: list[FileAnalysisResult],
) -> ArchitectureSummary:
    summaries = []
    all_deps = set()
    actual_files = set()
    all_interactions = _collect_all_interactions(file_analyses)

    for fa in file_analyses:
        actual_files.add(fa.file_path)
        summaries.append({
            "file": fa.file_path,
            "summary": fa.summary[:300],
            "functions": [
                {"name": f.name, "description": f.description, "calls": f.calls}
                for f in fa.functions
            ],
            "classes": [c.name for c in fa.classes],
            "deps": fa.external_dependencies,
            "internal_refs": fa.internal_file_references,
        })
        all_deps.update(fa.external_dependencies)

    # Platform detection
    platform_hints = _detect_platform(file_analyses, all_deps)

    # Interaction data for the LLM
    interactions_data = [
        {
            "source": i.source_file,
            "target": i.target_file,
            "type": i.interaction_type,
            "description": i.description,
        }
        for i in all_interactions
    ]

    max_files = 120
    note = ""
    if len(summaries) > max_files:
        summaries = summaries[:max_files]
        note = f"\n[Showing {max_files} of {len(file_analyses)} files]"

    file_list = json.dumps(sorted(actual_files), indent=1)

    prompt = f"""\
Analyze these codebase file reports and generate a COMPREHENSIVE architecture summary.

ACTUAL FILES (reference ONLY these):
{file_list}

Total files: {len(file_analyses)}
External dependencies: {', '.join(sorted(all_deps)[:40])}

DETECTED PLATFORM HINTS:
{json.dumps(platform_hints, indent=1)}

FILE INTERACTIONS DETECTED:
{json.dumps(interactions_data, indent=1)}

FILE REPORTS:
{json.dumps(summaries, indent=1)}
{note}

Produce JSON with ALL of these fields:

1. "overview" (string): 2-4 paragraphs about what this codebase does and how it's structured.

2. "key_components" (list of strings): actual files/modules and their roles.

3. "design_patterns" (list of strings): patterns you can see in the data.

4. "entry_points" (list of {{"file_path": str, "function_name": str, "reason": str}}): \
where execution begins. file_path MUST be from the actual file list.

5. "technology_stack" (list of strings): raw library/language names.

6. "technology_profile" (object): {{
    "platform": str (e.g., "Chrome Extension", "FastAPI Web Server"),
    "platform_category": str (e.g., "Browser Extension", "Web Application"),
    "primary_language": str,
    "runtime_environment": str (e.g., "Browser (Chrome)", "Node.js", "Python"),
    "apis_used": [str] (e.g., ["Chrome Extension API", "Fetch API", "jsPDF"]),
    "libraries": [str],
    "build_tools": [str],
    "summary": str (1-2 sentence description of the tech stack)
}}

7. "file_interactions" (list of {{"source_file": str, "target_file": str, \
"interaction_type": str, "description": str}}): \
how files depend on each other. Use the detected interactions above plus any you infer.

8. "execution_flow" (object): {{
    "trigger": str (what starts the workflow, e.g., "User clicks extension icon"),
    "steps": [{{
        "step_number": int,
        "actor": str (who/what performs this step),
        "action": str (what happens),
        "target": str (which file/component),
        "data_involved": str (what data is passed),
        "description": str
    }}],
    "output": str (final result),
    "summary": str (1 paragraph end-to-end description)
}}

9. "data_flow" (object): {{
    "steps": [{{
        "source": str (where data comes from),
        "transform": str (what happens to it),
        "destination": str (where it goes),
        "data_type": str (e.g., "HTML DOM", "JSON", "PDF bytes")
    }}],
    "summary": str (how data moves through the system)
}}

10. "component_interaction_summary" (string): \
A 2-3 paragraph prose description explaining how all components interact \
with each other at runtime. Include the flow from user trigger to final output.
"""

    try:
        result = await router.structured_chat(
            task=TaskType.ARCHITECTURE,
            prompt=prompt,
            system=ARCHITECTURE_SYSTEM_PROMPT,
            response_model=ArchitectureSummary,
            temperature=0.2,
            max_tokens=6000,
        )

        # Validate entry points
        validated_eps = [ep for ep in result.entry_points if ep.file_path in actual_files]
        if not validated_eps and result.entry_points:
            logger.warning("Removed %d hallucinated entry points", len(result.entry_points))
        result.entry_points = validated_eps

        # Merge static interactions with LLM-generated ones
        static_pairs = {(i.source_file, i.target_file) for i in all_interactions}
        merged = list(all_interactions)
        for inter in result.file_interactions:
            if (inter.source_file, inter.target_file) not in static_pairs:
                if inter.source_file in actual_files and inter.target_file in actual_files:
                    merged.append(inter)
        result.file_interactions = merged

        # Merge platform hints into profile if LLM missed them
        if not result.technology_profile.platform and platform_hints.get("platform"):
            result.technology_profile.platform = platform_hints["platform"]
        if not result.technology_profile.platform_category and platform_hints.get("platform_category"):
            result.technology_profile.platform_category = platform_hints["platform_category"]
        if not result.technology_profile.runtime_environment and platform_hints.get("runtime_environment"):
            result.technology_profile.runtime_environment = platform_hints["runtime_environment"]
        for api in platform_hints.get("apis_used", []):
            if api not in result.technology_profile.apis_used:
                result.technology_profile.apis_used.append(api)

        return result

    except Exception as e:
        logger.error("Architecture summary failed: %s", e)
        # Return a minimal result with at least the static data
        return ArchitectureSummary(
            overview=f"Generation failed: {e}",
            file_interactions=all_interactions,
            technology_profile=TechnologyProfile(
                platform=platform_hints.get("platform", ""),
                platform_category=platform_hints.get("platform_category", ""),
                apis_used=platform_hints.get("apis_used", []),
            ),
        )