"""
Mermaid Flowchart Generator Agent — creates diagrams via Groq.

Feature 7: Uses MERMAID_GENERATION task type (fast model).
Feature 14: Accepts optional LLM router.
"""

import json
import logging

from app.agents.groq_client import GroqClient, TaskType
from app.schemas.analysis import MermaidDiagram, FileAnalysisResult
from app.schemas.graph_models import DependencyGraph

logger = logging.getLogger(__name__)

MERMAID_SYSTEM_PROMPT = """\
You are a diagram specialist. Given dependency data, produce clean Mermaid flowchart syntax.

RULES:
1. Use `graph TD` for file-level (top-down). Use `graph LR` for function-level (left-right).
2. Keep diagrams READABLE — max 50 nodes.
3. If >50 nodes, select the most important (entry points, core modules, highly-connected).
4. Use SHORT but descriptive labels.
5. Sanitize node IDs: replace dots, slashes, special chars with underscores.
6. Output ONLY raw Mermaid syntax. No markdown fences, no explanations.\
"""


def _clean_mermaid(content: str) -> str:
    """Strip accidental markdown fences from LLM output."""
    cleaned = content.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:])
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()
    if cleaned.startswith("mermaid"):
        cleaned = cleaned[7:].strip()
    return cleaned


async def generate_file_flow_diagram(
    groq: GroqClient,
    graph: DependencyGraph,
    file_analyses: list[FileAnalysisResult],
) -> MermaidDiagram:
    edges = [{"from": e.source, "to": e.target}
             for e in graph.edges if e.relationship == "imports"][:100]

    prompt = f"""\
Generate a Mermaid file dependency flowchart.

File dependency edges (source → target):
{json.dumps(edges, indent=1)}

Total files: {len(graph.nodes)}

Create a `graph TD` diagram showing key file dependencies. Max 50 nodes. Short file names.\
"""

    try:
        content = await groq.chat(
            prompt, MERMAID_SYSTEM_PROMPT, task=TaskType.MERMAID_GENERATION
        )
        return MermaidDiagram(
            title="File Dependency Flow",
            diagram_type="file_flow",
            mermaid_syntax=_clean_mermaid(content),
        )
    except Exception as e:
        logger.error("File flow diagram failed: %s", e)
        return MermaidDiagram(
            title="File Dependency Flow",
            diagram_type="file_flow",
            mermaid_syntax=f'graph TD\n    Error["Diagram generation failed: {e}"]',
        )


async def generate_function_flow_diagram(
    groq: GroqClient,
    file_analyses: list[FileAnalysisResult],
) -> MermaidDiagram:
    call_data = []
    for fa in file_analyses:
        for func in fa.functions:
            for called in func.calls:
                call_data.append({"caller": f"{fa.file_path}::{func.name}", "callee": called})

    call_data = call_data[:80]

    if not call_data:
        return MermaidDiagram(
            title="Function Call Flow",
            diagram_type="function_flow",
            mermaid_syntax='graph LR\n    NoData["No function call relationships detected"]',
        )

    prompt = f"""\
Generate a Mermaid function call flow diagram.

Function calls:
{json.dumps(call_data, indent=1)}

Create a `graph LR` diagram. Max 40 nodes. Use function names as labels.\
"""

    try:
        content = await groq.chat(
            prompt, MERMAID_SYSTEM_PROMPT, task=TaskType.MERMAID_GENERATION
        )
        return MermaidDiagram(
            title="Function Call Flow",
            diagram_type="function_flow",
            mermaid_syntax=_clean_mermaid(content),
        )
    except Exception as e:
        return MermaidDiagram(
            title="Function Call Flow",
            diagram_type="function_flow",
            mermaid_syntax=f'graph LR\n    Error["Failed: {e}"]',
        )


async def generate_entry_point_diagram(
    groq: GroqClient,
    file_analyses: list[FileAnalysisResult],
    entry_points: list[dict],
) -> MermaidDiagram:
    if not entry_points:
        return MermaidDiagram(
            title="Entry Point Flow",
            diagram_type="entry_point_flow",
            mermaid_syntax='graph TD\n    NoEntry["No entry points detected"]',
        )

    entry_files = {ep.get("file_path", "") for ep in entry_points}
    relevant = []
    for fa in file_analyses:
        if fa.file_path in entry_files:
            relevant.append({
                "file": fa.file_path,
                "functions": [{"name": f.name, "calls": f.calls} for f in fa.functions],
            })

    prompt = f"""\
Generate a Mermaid entry point flow diagram.

Entry points:
{json.dumps(entry_points, indent=1)}

Related files:
{json.dumps(relevant, indent=1)}

Create a `graph TD` diagram from each entry point showing 2-3 levels of calls. Max 30 nodes.\
"""

    try:
        content = await groq.chat(
            prompt, MERMAID_SYSTEM_PROMPT, task=TaskType.MERMAID_GENERATION
        )
        return MermaidDiagram(
            title="Entry Point Flow",
            diagram_type="entry_point_flow",
            mermaid_syntax=_clean_mermaid(content),
        )
    except Exception as e:
        return MermaidDiagram(
            title="Entry Point Flow",
            diagram_type="entry_point_flow",
            mermaid_syntax=f'graph TD\n    Error["Failed: {e}"]',
        )
