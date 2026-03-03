"""
Mermaid Flowchart Generator Agent — creates visual diagrams from dependency data.
"""

import json
import logging

from app.agents.grok_client import GrokClient
from app.schemas.analysis import MermaidDiagram, FileAnalysisResult
from app.schemas.graph_models import DependencyGraph

logger = logging.getLogger(__name__)

MERMAID_SYSTEM_PROMPT = """\
You are a diagram generation specialist. Given dependency graph data, you produce \
clean Mermaid flowchart syntax.

RULES:
1. Use `graph TD` (top-down) for file-level diagrams.
2. Use `graph LR` (left-right) for function-level diagrams.
3. Keep diagrams READABLE — never exceed 50 nodes.
4. If there are more than 50 nodes, select the most important ones \
(entry points, core modules, highly connected nodes).
5. Use descriptive but SHORT labels.
6. Use proper Mermaid syntax — no errors.
7. Sanitize node IDs: replace special characters with underscores.

Respond with ONLY the raw Mermaid syntax. No markdown fences, no explanations.\
"""


async def generate_file_flow_diagram(
    grok: GrokClient,
    graph: DependencyGraph,
    file_analyses: list[FileAnalysisResult],
) -> MermaidDiagram:
    """Generate a high-level file dependency flow diagram."""
    # Condense edges for the prompt
    edges = [{"from": e.source, "to": e.target} for e in graph.edges if e.relationship == "imports"]
    if len(edges) > 100:
        edges = edges[:100]

    prompt = f"""\
Generate a Mermaid file dependency flowchart.

File dependency edges (source imports/depends-on target):
{json.dumps(edges, indent=1)}

Total files: {len(graph.nodes)}

Create a `graph TD` Mermaid diagram showing the most important file-to-file \
dependency relationships. Keep it under 50 nodes. Use short file names as labels.\
"""

    try:
        content = await grok.chat(
            prompt=prompt,
            system=MERMAID_SYSTEM_PROMPT,
            temperature=0.2,
        )
        # Clean up any accidental fences
        cleaned = content.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:])
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
        if cleaned.startswith("mermaid"):
            cleaned = cleaned[7:].strip()

        return MermaidDiagram(
            title="File Dependency Flow",
            diagram_type="file_flow",
            mermaid_syntax=cleaned,
        )
    except Exception as e:
        logger.error("File flow diagram generation failed: %s", e)
        return MermaidDiagram(
            title="File Dependency Flow",
            diagram_type="file_flow",
            mermaid_syntax=f"graph TD\n    Error[\"Diagram generation failed: {e}\"]",
        )


async def generate_function_flow_diagram(
    grok: GrokClient,
    file_analyses: list[FileAnalysisResult],
) -> MermaidDiagram:
    """Generate a function-level call flow diagram."""
    # Build function call data
    call_data = []
    for fa in file_analyses:
        for func in fa.functions:
            if func.calls:
                for called in func.calls:
                    call_data.append({
                        "caller": f"{fa.file_path}::{func.name}",
                        "callee": called,
                    })

    if len(call_data) > 80:
        call_data = call_data[:80]

    if not call_data:
        return MermaidDiagram(
            title="Function Call Flow",
            diagram_type="function_flow",
            mermaid_syntax='graph LR\n    NoData["No function call relationships detected"]',
        )

    prompt = f"""\
Generate a Mermaid function call flow diagram.

Function call relationships:
{json.dumps(call_data, indent=1)}

Create a `graph LR` Mermaid diagram showing function-to-function call flows. \
Keep it under 40 nodes. Use function names as labels. Group by file if possible.\
"""

    try:
        content = await grok.chat(
            prompt=prompt,
            system=MERMAID_SYSTEM_PROMPT,
            temperature=0.2,
        )
        cleaned = content.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:])
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
        if cleaned.startswith("mermaid"):
            cleaned = cleaned[7:].strip()

        return MermaidDiagram(
            title="Function Call Flow",
            diagram_type="function_flow",
            mermaid_syntax=cleaned,
        )
    except Exception as e:
        logger.error("Function flow diagram generation failed: %s", e)
        return MermaidDiagram(
            title="Function Call Flow",
            diagram_type="function_flow",
            mermaid_syntax=f"graph LR\n    Error[\"Diagram generation failed: {e}\"]",
        )


async def generate_entry_point_diagram(
    grok: GrokClient,
    file_analyses: list[FileAnalysisResult],
    entry_points: list[dict],
) -> MermaidDiagram:
    """Generate a diagram showing entry point flows."""
    if not entry_points:
        return MermaidDiagram(
            title="Entry Point Flow",
            diagram_type="entry_point_flow",
            mermaid_syntax='graph TD\n    NoEntry["No entry points detected"]',
        )

    # Gather relevant file data for entry point files
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

Related file details:
{json.dumps(relevant, indent=1)}

Create a `graph TD` Mermaid diagram starting from each entry point and showing \
the first 2-3 levels of function calls / file dependencies. Keep it under 30 nodes.\
"""

    try:
        content = await grok.chat(
            prompt=prompt,
            system=MERMAID_SYSTEM_PROMPT,
            temperature=0.2,
        )
        cleaned = content.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:])
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
        if cleaned.startswith("mermaid"):
            cleaned = cleaned[7:].strip()

        return MermaidDiagram(
            title="Entry Point Flow",
            diagram_type="entry_point_flow",
            mermaid_syntax=cleaned,
        )
    except Exception as e:
        logger.error("Entry point diagram generation failed: %s", e)
        return MermaidDiagram(
            title="Entry Point Flow",
            diagram_type="entry_point_flow",
            mermaid_syntax=f"graph TD\n    Error[\"Diagram generation failed: {e}\"]",
        )