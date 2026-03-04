"""Report Combiner — merges analyses, builds graph, generates architecture + diagrams."""

import logging
import asyncio

from app.agents.groq_client import GroqClient
from app.agents.architecture_agent import generate_architecture_summary
from app.agents.mermaid_agent import (
    generate_file_flow_diagram,
    generate_function_flow_diagram,
    generate_entry_point_diagram,
)
from app.graph.dependency_graph import build_dependency_graph
from app.schemas.analysis import (
    FileAnalysisResult, MermaidDiagram, FullAnalysisReport,
    RepoMap, CompactFileSummary,
)
from app.schemas.graph_models import DependencyGraph

logger = logging.getLogger(__name__)


async def combine_and_generate_report(
    groq: GroqClient,
    analysis_id: str,
    repository_url: str,
    file_analyses: list[FileAnalysisResult],
    router=None,
    repo_map: RepoMap | None = None,
    compact_summaries: list[CompactFileSummary] | None = None,
) -> tuple[FullAnalysisReport, DependencyGraph]:
    """
    Merge file analyses, build dependency graph, generate architecture summary + diagrams.

    Feature 14/15: Passes router + repo map + compact summaries to architecture agent
    so Gemini (when available) can use full context.
    """
    logger.info("Combining %d file reports", len(file_analyses))

    # Build dependency graph
    dep_graph = build_dependency_graph(file_analyses)

    # Architecture summary — uses LLM router (Gemini if available)
    arch_summary = await generate_architecture_summary(
        groq=groq,
        file_analyses=file_analyses,
        router=router,
        repo_map=repo_map,
        compact_summaries=compact_summaries,
    )

    # Mermaid diagrams (use fast model, run concurrently)
    ep_dicts = [ep.model_dump() for ep in arch_summary.entry_points]

    diagrams_raw = await asyncio.gather(
        generate_file_flow_diagram(groq, dep_graph, file_analyses),
        generate_function_flow_diagram(groq, file_analyses),
        generate_entry_point_diagram(groq, file_analyses, ep_dicts),
        return_exceptions=True,
    )

    diagrams = [d for d in diagrams_raw if isinstance(d, MermaidDiagram)]

    report = FullAnalysisReport(
        analysis_id=analysis_id,
        repository_url=repository_url,
        total_files=len(file_analyses),
        file_analyses=file_analyses,
        architecture_summary=arch_summary,
        mermaid_diagrams=diagrams,
        status="completed",
    )

    logger.info("Report generated for %s", analysis_id)
    return report, dep_graph
