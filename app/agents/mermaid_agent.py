"""
Mermaid Flowchart Generator — STRICT anti-hallucination version.
Only uses data explicitly provided in the prompt. Never invents nodes.
"""

import json
import logging

from app.agents.llm_router import LLMRouter, TaskType
from app.schemas.analysis import MermaidDiagram, FileAnalysisResult
from app.schemas.graph_models import DependencyGraph

logger = logging.getLogger(__name__)

MERMAID_SYSTEM_PROMPT = """\
You are a diagram specialist. Given ACTUAL dependency data, produce clean Mermaid flowchart syntax.

CRITICAL RULES:
1. Use ONLY the nodes and edges provided in the data below. Do NOT invent ANY files, \
functions, variables, or connections that are not explicitly listed.
2. If the data is empty or has no edges, return a single-node diagram saying "No dependencies found".
3. Use `graph TD` for file-level (top-down). Use `graph LR` for function-level (left-right).
4. Keep diagrams READABLE — max 50 nodes.
5. Use SHORT but descriptive labels derived from the actual file/function names given.
6. Sanitize node IDs: replace dots, slashes, special chars with underscores.
7. Output ONLY raw Mermaid syntax. No markdown fences, no explanations, no notes.
8. NEVER add nodes that are not in the provided data.
9. NEVER explain the diagram or add comments after the syntax.\
"""


def _clean_mermaid(content: str) -> str:
    """Strip accidental markdown fences and trailing notes from LLM output."""
    cleaned = content.strip()

    # Remove opening markdown fences
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:])
    if cleaned.startswith("mermaid"):
        cleaned = cleaned[7:].strip()

    # Remove closing markdown fences
    if "```" in cleaned:
        cleaned = cleaned[:cleaned.index("```")].strip()

    # Remove any trailing notes/explanations after the diagram
    # Mermaid diagrams end when lines stop having diagram syntax
    lines = cleaned.split("\n")
    diagram_lines = []
    for line in lines:
        stripped = line.strip()
        # Stop at empty lines followed by non-diagram text
        if stripped and not stripped.startswith("graph ") and not stripped.startswith("flowchart "):
            # Check if it looks like a diagram line (contains -->, ---, [, (, etc.)
            is_diagram = any(c in stripped for c in ["-->", "---", "---|", "[", "]", "(", ")", "{", "}"])
            is_indent = line.startswith(" ") or line.startswith("\t")
            is_graph_def = stripped.startswith("graph ") or stripped.startswith("flowchart ")
            is_subgraph = stripped.startswith("subgraph") or stripped == "end"
            is_style = stripped.startswith("style ") or stripped.startswith("classDef ")
            is_node_def = is_indent and not stripped.startswith("Note")

            if not (is_diagram or is_indent or is_graph_def or is_subgraph or is_style or is_node_def):
                # Looks like a note/explanation — stop here
                if stripped.startswith("Note") or stripped.startswith("I've") or stripped.startswith("This"):
                    break
        diagram_lines.append(line)

    cleaned = "\n".join(diagram_lines).strip()

    # Final cleanup: remove trailing empty lines
    while cleaned.endswith("\n"):
        cleaned = cleaned[:-1]

    return cleaned


def _build_file_flow_from_data(graph: DependencyGraph, file_analyses: list[FileAnalysisResult]) -> str:
    """
    Build Mermaid diagram DIRECTLY from data — no LLM needed for small graphs.
    This is the fallback for when the LLM would hallucinate.
    """
    actual_files = {fa.file_path for fa in file_analyses}
    import_edges = [
        e for e in graph.edges
        if e.relationship == "imports" and e.source in actual_files and e.target in actual_files
    ]

    if not import_edges and len(actual_files) <= 20:
        # No edges — just show the file list
        lines = ["graph TD"]
        for i, fp in enumerate(sorted(actual_files)):
            node_id = fp.replace("/", "_").replace(".", "_").replace("-", "_")
            short_name = fp.split("/")[-1] if "/" in fp else fp
            lines.append(f"    {node_id}[\"{short_name}\"]")
        return "\n".join(lines)

    if not import_edges:
        return 'graph TD\n    NoEdges["No file dependencies detected"]'

    lines = ["graph TD"]
    for e in import_edges[:50]:
        src_id = e.source.replace("/", "_").replace(".", "_").replace("-", "_")
        tgt_id = e.target.replace("/", "_").replace(".", "_").replace("-", "_")
        src_label = e.source.split("/")[-1]
        tgt_label = e.target.split("/")[-1]
        lines.append(f"    {src_id}[\"{src_label}\"] --> {tgt_id}[\"{tgt_label}\"]")

    return "\n".join(lines)


def _build_function_flow_from_data(file_analyses: list[FileAnalysisResult]) -> str:
    """Build function call diagram DIRECTLY from analysis data."""
    call_data = []
    for fa in file_analyses:
        for func in fa.functions:
            for called in func.calls:
                caller_label = f"{fa.file_path.split('/')[-1]}::{func.name}"
                call_data.append((caller_label, called))

    if not call_data:
        return 'graph LR\n    NoData["No function call relationships detected"]'

    lines = ["graph LR"]
    seen_edges = set()
    for caller, callee in call_data[:50]:
        edge_key = f"{caller}→{callee}"
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)

        caller_id = caller.replace("/", "_").replace(".", "_").replace("-", "_").replace(":", "_").replace(" ", "_")
        callee_id = callee.replace("/", "_").replace(".", "_").replace("-", "_").replace(":", "_").replace(" ", "_")
        lines.append(f"    {caller_id}[\"{caller}\"] --> {callee_id}[\"{callee}\"]")

    return "\n".join(lines)


async def generate_file_flow_diagram(
    router: LLMRouter,
    graph: DependencyGraph,
    file_analyses: list[FileAnalysisResult],
) -> MermaidDiagram:
    """
    Generate file dependency diagram.
    For small repos (≤20 files), build directly from data (no LLM = no hallucination).
    For larger repos, use LLM but with strict constraints.
    """
    actual_files = {fa.file_path for fa in file_analyses}
    edges = [
        {"from": e.source, "to": e.target}
        for e in graph.edges
        if e.relationship == "imports" and e.source in actual_files and e.target in actual_files
    ][:100]

    # For small repos: build directly — no LLM needed
    if len(actual_files) <= 20:
        syntax = _build_file_flow_from_data(graph, file_analyses)
        return MermaidDiagram(
            title="File Dependency Flow",
            diagram_type="file_flow",
            mermaid_syntax=syntax,
        )

    # For larger repos: use LLM with strict prompt
    actual_file_list = json.dumps(sorted(actual_files), indent=1)

    prompt = f"""\
Generate a Mermaid file dependency flowchart using ONLY these actual files and edges.

ACTUAL FILES IN THIS REPOSITORY (use ONLY these):
{actual_file_list}

ACTUAL DEPENDENCY EDGES (use ONLY these):
{json.dumps(edges, indent=1)}

CRITICAL: Do NOT add any files or edges that are not listed above.
If there are no edges, just show the files as standalone nodes.

Create a `graph TD` diagram. Max 50 nodes. Use short file names as labels.\
"""

    try:
        content = await router.chat(TaskType.MERMAID, prompt, MERMAID_SYSTEM_PROMPT)
        syntax = _clean_mermaid(content)

        # VALIDATION: Check that the LLM didn't invent fake files
        # If it did, fall back to data-driven diagram
        if _diagram_has_fake_nodes(syntax, actual_files):
            logger.warning("LLM hallucinated nodes in file flow — using data-driven fallback")
            syntax = _build_file_flow_from_data(graph, file_analyses)

        return MermaidDiagram(
            title="File Dependency Flow",
            diagram_type="file_flow",
            mermaid_syntax=syntax,
        )
    except Exception as e:
        logger.error("File flow diagram failed: %s — using fallback", e)
        return MermaidDiagram(
            title="File Dependency Flow",
            diagram_type="file_flow",
            mermaid_syntax=_build_file_flow_from_data(graph, file_analyses),
        )


def _diagram_has_fake_nodes(syntax: str, actual_files: set[str]) -> bool:
    """
    Check if the generated Mermaid diagram contains nodes not in the actual file list.
    Returns True if hallucinated nodes are detected.
    """
    # Extract node labels from Mermaid — look for ["..."] patterns
    import re
    labels = re.findall(r'\["([^"]+)"\]', syntax)

    # Build set of acceptable labels (full paths and short names)
    acceptable = set()
    for fp in actual_files:
        acceptable.add(fp)
        acceptable.add(fp.split("/")[-1])  # short name
        acceptable.add(fp.replace(".", "_").replace("/", "_"))  # sanitized

    # Check each label
    fake_count = 0
    for label in labels:
        label_clean = label.strip()
        if label_clean and label_clean not in acceptable:
            # Check fuzzy — maybe the label is slightly different
            if not any(label_clean.lower() in a.lower() or a.lower() in label_clean.lower() for a in acceptable):
                fake_count += 1
                logger.debug("Possible hallucinated node: '%s'", label_clean)

    # Allow some tolerance (LLM might shorten names)
    # But if >30% of labels are fake, it's hallucinating
    if labels and fake_count / len(labels) > 0.3:
        return True
    return False


async def generate_function_flow_diagram(
    router: LLMRouter,
    file_analyses: list[FileAnalysisResult],
) -> MermaidDiagram:
    """Generate function call flow — uses data-driven approach for small repos."""
    call_data = []
    for fa in file_analyses:
        for func in fa.functions:
            for called in func.calls:
                call_data.append({
                    "caller": f"{fa.file_path}::{func.name}",
                    "callee": called,
                })

    # For small repos or few calls: build directly
    if len(call_data) <= 30:
        syntax = _build_function_flow_from_data(file_analyses)
        return MermaidDiagram(
            title="Function Call Flow",
            diagram_type="function_flow",
            mermaid_syntax=syntax,
        )

    call_data = call_data[:80]

    # Build list of actual function names for validation
    actual_functions = set()
    for fa in file_analyses:
        for func in fa.functions:
            actual_functions.add(func.name)
            actual_functions.add(f"{fa.file_path}::{func.name}")
            for called in func.calls:
                actual_functions.add(called)

    prompt = f"""\
Generate a Mermaid function call flow diagram using ONLY these actual function calls.

ACTUAL FUNCTION CALLS (use ONLY these):
{json.dumps(call_data, indent=1)}

CRITICAL: Do NOT add any functions or calls that are not listed above.
Do NOT invent intermediate steps, callbacks, or connections.

Create a `graph LR` diagram. Max 40 nodes. Use function names as labels.\
"""

    try:
        content = await router.chat(TaskType.MERMAID, prompt, MERMAID_SYSTEM_PROMPT)
        syntax = _clean_mermaid(content)
        return MermaidDiagram(
            title="Function Call Flow",
            diagram_type="function_flow",
            mermaid_syntax=syntax,
        )
    except Exception as e:
        return MermaidDiagram(
            title="Function Call Flow",
            diagram_type="function_flow",
            mermaid_syntax=_build_function_flow_from_data(file_analyses),
        )


async def generate_entry_point_diagram(
    router: LLMRouter,
    file_analyses: list[FileAnalysisResult],
    entry_points: list[dict],
) -> MermaidDiagram:
    """Generate entry point flow — data-driven for small repos."""
    if not entry_points:
        return MermaidDiagram(
            title="Entry Point Flow",
            diagram_type="entry_point_flow",
            mermaid_syntax='graph TD\n    NoEntry["No entry points detected"]',
        )

    # Build actual data from entry points
    entry_files = {ep.get("file_path", "") for ep in entry_points}
    relevant = []
    actual_functions = set()
    for fa in file_analyses:
        if fa.file_path in entry_files:
            funcs_data = []
            for f in fa.functions:
                funcs_data.append({"name": f.name, "calls": f.calls})
                actual_functions.add(f.name)
                for c in f.calls:
                    actual_functions.add(c)
            relevant.append({"file": fa.file_path, "functions": funcs_data})

    # For small repos: build directly
    if len(actual_functions) <= 20:
        lines = ["graph TD"]
        for ep in entry_points:
            ep_file = ep.get("file_path", "unknown")
            ep_func = ep.get("function_name", "entry")
            ep_id = f"{ep_file}__{ep_func}".replace("/", "_").replace(".", "_").replace("-", "_").replace("(", "").replace(")", "")
            lines.append(f"    {ep_id}[\"{ep_file}::{ep_func}\"]")

            # Add direct calls from this entry point
            for fa in file_analyses:
                if fa.file_path == ep_file:
                    for func in fa.functions:
                        if func.name == ep_func or ep_func in ("(anonymous)", ""):
                            for called in func.calls:
                                called_id = called.replace(".", "_").replace("/", "_").replace("-", "_")
                                lines.append(f"    {ep_id} --> {called_id}[\"{called}\"]")

        return MermaidDiagram(
            title="Entry Point Flow",
            diagram_type="entry_point_flow",
            mermaid_syntax="\n".join(lines),
        )

    prompt = f"""\
Generate a Mermaid entry point flow diagram using ONLY these actual entry points and functions.

ACTUAL ENTRY POINTS:
{json.dumps(entry_points, indent=1)}

ACTUAL RELATED FILES AND THEIR FUNCTIONS:
{json.dumps(relevant, indent=1)}

CRITICAL: Do NOT invent any functions, API calls, or connections not listed above.
Show ONLY the actual entry point → function call chains from the data.

Create a `graph TD` diagram. Max 30 nodes.\
"""

    try:
        content = await router.chat(TaskType.MERMAID, prompt, MERMAID_SYSTEM_PROMPT)
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