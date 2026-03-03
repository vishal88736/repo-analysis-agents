"""
Report Combiner — merges all file analyses, builds dependency graph,
generates architecture summary and Mermaid diagrams.
"""

import logging

from app.agents.grok_client import GrokClient
from app.agents.architecture_agent import generate_architecture_summary
from app.agents.mermaid_agent import (
    generate_file_flow_diagram,
    generate_function_flow_diagram,
    generate_entry_point_diagram,
)
from app.graph.dependency_graph import build_dependency_graph
from app.schemas.analysis import (
    FileAnalysisResult,
    ArchitectureSummary,
    MermaidDiagram,
    FullAnalysisReport,
)
from app.schemas.graph_models import DependencyGraph

logger = logging.getLogger(__name__)


async def combine_and_generate_report(
    grok: GrokClient,
    analysis_id: str,
    repository_url: str,
    file_analyses: list[FileAnalysisResult],
) -> tuple[FullAnalysisReport, DependencyGraph]:
    """
    Combine all file analyses into a full report with:
    1. Dependency graph
    2. Architecture summary
    3. Mermaid diagrams

    Returns (full_report, dependency_graph).
    """
    logger.info("Combining reports for %d files", len(file_analyses))

    # Phase 4: Build dependency graph
    dep_graph = build_dependency_graph(file_analyses)
    logger.info("Dependency graph: %d nodes, %d edges", len(dep_graph.nodes), len(dep_graph.edges))

    # Phase 4: Generate architecture summary
    arch_summary = await generate_architecture_summary(grok, file_analyses)

    # Phase 6: Generate Mermaid diagrams (run concurrently)
    import asyncio

    entry_points_dicts = [
        ep.model_dump() for ep in arch_summary.entry_points
    ]

    file_flow_task = generate_file_flow_diagram(grok, dep_graph, file_analyses)
    func_flow_task = generate_function_flow_diagram(grok, file_analyses)
    entry_flow_task = generate_entry_point_diagram(grok, file_analyses, entry_points_dicts)

    diagrams = await asyncio.gather(
        file_flow_task,
        func_flow_task,
        entry_flow_task,
        return_exceptions=True,
    )

    mermaid_diagrams: list[MermaidDiagram] = []
    for d in diagrams:
        if isinstance(d, MermaidDiagram):
            mermaid_diagrams.append(d)
        elif isinstance(d, Exception):
            logger.error("Diagram generation error: %s", d)

    # Build final report
    report = FullAnalysisReport(
        analysis_id=analysis_id,
        repository_url=repository_url,
        total_files=len(file_analyses),
        file_analyses=file_analyses,
        architecture_summary=arch_summary,
        mermaid_diagrams=mermaid_diagrams,
        status="completed",
    )

    logger.info("Full report generated for analysis %s", analysis_id)
    return report, dep_graph