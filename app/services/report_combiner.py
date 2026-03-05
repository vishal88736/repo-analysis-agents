"""Report Combiner — uses LLM router for multi-provider support."""

import logging
import asyncio

from app.agents.groq_client import GroqClient
from app.agents.llm_router import LLMRouter
from app.agents.architecture_agent import generate_architecture_summary
from app.agents.mermaid_agent import (
    generate_file_flow_diagram,
    generate_function_flow_diagram,
    generate_entry_point_diagram,
)
from app.graph.dependency_graph import build_dependency_graph
from app.schemas.analysis import (
    FileAnalysisResult, MermaidDiagram, FullAnalysisReport,
    CompactFileSummary, RepoMap,
)
from app.schemas.graph_models import DependencyGraph

logger = logging.getLogger(__name__)


async def combine_and_generate_report(
    groq: GroqClient,
    router: LLMRouter,
    analysis_id: str,
    repository_url: str,
    file_analyses: list[FileAnalysisResult],
    compact_summaries: list[CompactFileSummary] | None = None,
    repo_map: RepoMap | None = None,
) -> tuple[FullAnalysisReport, DependencyGraph]:
    logger.info("Combining %d file reports", len(file_analyses))

    dep_graph = build_dependency_graph(file_analyses)

    # Architecture summary via router (may use Gemini for large context)
    arch_summary = await generate_architecture_summary(router, file_analyses)

    ep_dicts = [ep.model_dump() for ep in arch_summary.entry_points]

    diagrams_raw = await asyncio.gather(
        generate_file_flow_diagram(router, dep_graph, file_analyses),
        generate_function_flow_diagram(router, file_analyses),
        generate_entry_point_diagram(router, file_analyses, ep_dicts),
        return_exceptions=True,
    )

    diagrams = [d for d in diagrams_raw if isinstance(d, MermaidDiagram)]

    report = FullAnalysisReport(
        analysis_id=analysis_id,
        repository_url=repository_url,
        total_files=len(file_analyses),
        file_analyses=file_analyses,
        compact_summaries=compact_summaries or [],
        architecture_summary=arch_summary,
        mermaid_diagrams=diagrams,
        repo_map=repo_map,
        status="completed",
    )

    logger.info("Report generated for %s", analysis_id)
    return report, dep_graph