"""Dependency graph builder from file analysis results."""

import logging
import os
from collections import defaultdict

from app.schemas.analysis import FileAnalysisResult
from app.schemas.graph_models import DependencyGraph, DependencyNode, DependencyEdge

logger = logging.getLogger(__name__)


def _resolve_import(import_module: str, all_paths: set[str]) -> str | None:
    potential = import_module.replace(".", "/")
    for ext in (".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".java", ".rs"):
        candidate = potential + ext
        if candidate in all_paths:
            return candidate
    init = potential + "/__init__.py"
    if init in all_paths:
        return init
    if "/" in import_module:
        for ext in (".js", ".ts", ".jsx", ".tsx"):
            candidate = import_module.lstrip("./") + ext
            if candidate in all_paths:
                return candidate
    return None


def build_dependency_graph(file_analyses: list[FileAnalysisResult]) -> DependencyGraph:
    nodes, edges = [], []
    adjacency: dict[str, list[str]] = defaultdict(list)
    all_paths = {fa.file_path for fa in file_analyses}
    file_funcs: dict[str, set[str]] = {}

    for fa in file_analyses:
        nodes.append(DependencyNode(id=fa.file_path, type="file",
                                     label=os.path.basename(fa.file_path), file_path=fa.file_path))
        funcs = set()
        for func in fa.functions:
            fid = f"{fa.file_path}::{func.name}"
            nodes.append(DependencyNode(id=fid, type="function", label=func.name, file_path=fa.file_path))
            funcs.add(func.name)
        file_funcs[fa.file_path] = funcs

        for cls in fa.classes:
            cid = f"{fa.file_path}::{cls.name}"
            nodes.append(DependencyNode(id=cid, type="class", label=cls.name, file_path=fa.file_path))

    for fa in file_analyses:
        for dep in fa.external_dependencies:
            target = _resolve_import(dep, all_paths)
            if target and target != fa.file_path:
                edges.append(DependencyEdge(source=fa.file_path, target=target, relationship="imports"))
                adjacency[fa.file_path].append(target)

        for func in fa.functions:
            caller_id = f"{fa.file_path}::{func.name}"
            for called in func.calls:
                for other_path, other_funcs in file_funcs.items():
                    if called in other_funcs:
                        callee_id = f"{other_path}::{called}"
                        edges.append(DependencyEdge(source=caller_id, target=callee_id, relationship="calls"))
                        adjacency[caller_id].append(callee_id)
                        break

    logger.info("Dependency graph: %d nodes, %d edges", len(nodes), len(edges))
    return DependencyGraph(nodes=nodes, edges=edges, adjacency_list=dict(adjacency))


def get_dependency_context_for_file(graph: DependencyGraph, file_path: str, depth: int = 2) -> str:
    visited = set()
    lines = []

    def _walk(node_id: str, d: int):
        if d > depth or node_id in visited:
            return
        visited.add(node_id)
        for neighbor in graph.adjacency_list.get(node_id, []):
            lines.append(f"{'  ' * d}{node_id} → {neighbor}")
            _walk(neighbor, d + 1)

    _walk(file_path, 0)
    return "\n".join(lines) if lines else f"No dependencies found for {file_path}"