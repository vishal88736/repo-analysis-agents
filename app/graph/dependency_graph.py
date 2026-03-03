"""
Dependency graph builder — constructs file-level and function-level
dependency graphs from file analysis results.
"""

import logging
import os
from collections import defaultdict

from app.schemas.analysis import FileAnalysisResult
from app.schemas.graph_models import DependencyGraph, DependencyNode, DependencyEdge

logger = logging.getLogger(__name__)


def _normalize_import_to_file(import_module: str, all_file_paths: set[str]) -> str | None:
    """
    Attempt to resolve an import module name to an actual file path in the repo.
    E.g., 'app.services.scanner' → 'app/services/scanner.py'
    """
    # Try direct path mapping (Python style)
    potential = import_module.replace(".", "/")
    for ext in [".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".java", ".rs"]:
        candidate = potential + ext
        if candidate in all_file_paths:
            return candidate
    # Try as directory with __init__
    init_candidate = potential + "/__init__.py"
    if init_candidate in all_file_paths:
        return init_candidate
    # Try relative-style
    if "/" in import_module:
        for ext in [".js", ".ts", ".jsx", ".tsx"]:
            candidate = import_module.lstrip("./") + ext
            if candidate in all_file_paths:
                return candidate
    return None


def build_dependency_graph(
    file_analyses: list[FileAnalysisResult],
) -> DependencyGraph:
    """
    Build a dependency graph from all file analysis results.

    Creates:
    - File-level nodes and import edges
    - Function-level nodes and call edges
    - Adjacency list for traversal
    """
    nodes: list[DependencyNode] = []
    edges: list[DependencyEdge] = []
    adjacency: dict[str, list[str]] = defaultdict(list)

    all_file_paths = {fa.file_path for fa in file_analyses}
    file_functions: dict[str, set[str]] = {}  # file_path → set of function names

    # Build file nodes and function registry
    for fa in file_analyses:
        nodes.append(DependencyNode(
            id=fa.file_path,
            type="file",
            label=os.path.basename(fa.file_path),
            file_path=fa.file_path,
        ))
        funcs = set()
        for func in fa.functions:
            func_id = f"{fa.file_path}::{func.name}"
            nodes.append(DependencyNode(
                id=func_id,
                type="function",
                label=func.name,
                file_path=fa.file_path,
            ))
            funcs.add(func.name)
        file_functions[fa.file_path] = funcs

        for cls in fa.classes:
            cls_id = f"{fa.file_path}::{cls.name}"
            nodes.append(DependencyNode(
                id=cls_id,
                type="class",
                label=cls.name,
                file_path=fa.file_path,
            ))

    # Build edges
    for fa in file_analyses:
        # File → File import edges
        for dep in fa.external_dependencies:
            target = _normalize_import_to_file(dep, all_file_paths)
            if target and target != fa.file_path:
                edges.append(DependencyEdge(
                    source=fa.file_path,
                    target=target,
                    relationship="imports",
                ))
                adjacency[fa.file_path].append(target)

        # Function → Function call edges
        for func in fa.functions:
            caller_id = f"{fa.file_path}::{func.name}"
            for called_name in func.calls:
                # Try to find the target function in any file
                for other_path, other_funcs in file_functions.items():
                    if called_name in other_funcs:
                        callee_id = f"{other_path}::{called_name}"
                        edges.append(DependencyEdge(
                            source=caller_id,
                            target=callee_id,
                            relationship="calls",
                        ))
                        adjacency[caller_id].append(callee_id)
                        break

    logger.info(
        "Dependency graph built: %d nodes, %d edges",
        len(nodes),
        len(edges),
    )

    return DependencyGraph(
        nodes=nodes,
        edges=edges,
        adjacency_list=dict(adjacency),
    )


def get_dependency_context_for_file(
    graph: DependencyGraph,
    file_path: str,
    depth: int = 2,
) -> str:
    """Get a text description of dependencies around a specific file."""
    visited = set()
    lines = []

    def _walk(node_id: str, current_depth: int):
        if current_depth > depth or node_id in visited:
            return
        visited.add(node_id)
        neighbors = graph.adjacency_list.get(node_id, [])
        for neighbor in neighbors:
            lines.append(f"  {'  ' * current_depth}{node_id} → {neighbor}")
            _walk(neighbor, current_depth + 1)

    _walk(file_path, 0)
    return "\n".join(lines) if lines else f"No dependencies found for {file_path}"