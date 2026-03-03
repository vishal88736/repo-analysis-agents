"""
API Routes — REST endpoints for the analysis system.

POST /analyze       → Start analysis, returns analysis_id
GET  /report/{id}   → Get full analysis report
POST /query         → RAG-based question answering
GET  /status/{id}   → Check analysis status
"""

import logging
import asyncio

from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException, status

from app.api.dependencies import get_analysis_store, get_vector_store
from app.schemas.api_models import (
    AnalyzeRequest,
    AnalyzeResponse,
    QueryRequest,
    QueryResponse,
    ReportResponse,
    ErrorResponse,
)
from app.services.analysis_store import AnalysisStore
from app.services.orchestrator import run_analysis_pipeline, create_analysis_id
from app.agents.grok_client import GrokClient
from app.agents.rag_answer_agent import answer_query
from app.graph.dependency_graph import get_dependency_context_for_file
from app.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Analysis"])


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={400: {"model": ErrorResponse}},
)
async def start_analysis(
    request: AnalyzeRequest,
    background_tasks: BackgroundTasks,
    store: AnalysisStore = Depends(get_analysis_store),
):
    """
    Start a new repository analysis.

    Clones the repository, processes files in batches, generates reports,
    builds RAG index. All processing happens in the background.

    Returns an analysis_id to track progress.
    """
    # Basic URL validation
    url = request.repository_url.strip()
    if not url.startswith("https://github.com/"):
        raise HTTPException(
            status_code=400,
            detail="Only GitHub URLs are supported (https://github.com/owner/repo)",
        )

    analysis_id = create_analysis_id()
    await store.set_status(analysis_id, "pending")

    # Launch pipeline in background
    background_tasks.add_task(
        run_analysis_pipeline,
        analysis_id=analysis_id,
        repository_url=url,
        store=store,
    )

    logger.info("Analysis started: %s for %s", analysis_id, url)

    return AnalyzeResponse(
        analysis_id=analysis_id,
        status="pending",
        message=f"Analysis started. Use GET /api/v1/report/{analysis_id} to check results.",
    )


@router.get(
    "/status/{analysis_id}",
    responses={404: {"model": ErrorResponse}},
)
async def get_status(
    analysis_id: str,
    store: AnalysisStore = Depends(get_analysis_store),
):
    """Check the status of an ongoing analysis."""
    current_status = await store.get_status(analysis_id)
    if current_status is None:
        raise HTTPException(status_code=404, detail=f"Analysis '{analysis_id}' not found.")
    return {"analysis_id": analysis_id, "status": current_status}


@router.get(
    "/report/{analysis_id}",
    response_model=ReportResponse,
    responses={404: {"model": ErrorResponse}, 202: {"model": dict}},
)
async def get_report(
    analysis_id: str,
    store: AnalysisStore = Depends(get_analysis_store),
):
    """
    Get the full analysis report.

    Returns global summary, file summaries, and Mermaid diagrams.
    Returns 202 if analysis is still in progress.
    """
    current_status = await store.get_status(analysis_id)
    if current_status is None:
        raise HTTPException(status_code=404, detail=f"Analysis '{analysis_id}' not found.")

    if current_status not in ("completed", "failed"):
        return HTTPException(
            status_code=202,
            detail={"analysis_id": analysis_id, "status": current_status, "message": "Analysis still in progress."},
        )

    report = await store.load_report(analysis_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report data not found on disk.")

    if report.status == "failed":
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {report.error_message}",
        )

    # Build response
    file_summaries = [
        {
            "file_path": fa.file_path,
            "summary": fa.summary,
            "functions": len(fa.functions),
            "classes": len(fa.classes),
            "dependencies": fa.external_dependencies,
        }
        for fa in report.file_analyses
    ]

    mermaid_data = [
        {
            "title": d.title,
            "type": d.diagram_type,
            "syntax": d.mermaid_syntax,
        }
        for d in report.mermaid_diagrams
    ]

    entry_points = [
        ep.model_dump() for ep in report.architecture_summary.entry_points
    ]

    return ReportResponse(
        analysis_id=report.analysis_id,
        repository_url=report.repository_url,
        status=report.status,
        total_files=report.total_files,
        global_summary=report.architecture_summary.overview,
        key_components=report.architecture_summary.key_components,
        design_patterns=report.architecture_summary.design_patterns,
        technology_stack=report.architecture_summary.technology_stack,
        entry_points=entry_points,
        file_summaries=file_summaries,
        mermaid_diagrams=mermaid_data,
    )


@router.post(
    "/query",
    response_model=QueryResponse,
    responses={404: {"model": ErrorResponse}},
)
async def query_codebase(
    request: QueryRequest,
    store: AnalysisStore = Depends(get_analysis_store),
    vector_store: VectorStore = Depends(get_vector_store),
):
    """
    Ask a natural language question about the analyzed codebase.

    Uses RAG: vector search → retrieve context → augment with dependency graph → LLM answer.
    """
    analysis_id = request.analysis_id

    # Verify analysis exists and is complete
    current_status = await store.get_status(analysis_id)
    if current_status is None:
        raise HTTPException(status_code=404, detail=f"Analysis '{analysis_id}' not found.")
    if current_status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Analysis not yet complete. Current status: {current_status}",
        )

    # Load report for context
    report = await store.load_report(analysis_id)
    graph = await store.load_graph(analysis_id)

    # Step 1: Vector search
    retrieved_chunks = await vector_store.search(
        analysis_id=analysis_id,
        query=request.question,
    )

    # Step 2: Augment with dependency context
    dep_context = ""
    if graph and retrieved_chunks:
        # Get dependency info for the most relevant files
        relevant_files = set()
        for chunk in retrieved_chunks[:5]:
            fp = chunk.get("metadata", {}).get("file_path", "")
            if fp and fp != "__global__":
                relevant_files.add(fp)
        dep_parts = []
        for fp in relevant_files:
            dep_parts.append(get_dependency_context_for_file(graph, fp))
        dep_context = "\n".join(dep_parts)

    # Step 3: LLM answer generation
    arch_summary = report.architecture_summary.overview if report else ""

    grok = GrokClient()
    answer = await answer_query(
        grok=grok,
        question=request.question,
        retrieved_chunks=retrieved_chunks,
        dependency_context=dep_context,
        architecture_summary=arch_summary,
    )

    # Gather source file paths
    sources = list(set(
        chunk.get("metadata", {}).get("file_path", "")
        for chunk in retrieved_chunks
        if chunk.get("metadata", {}).get("file_path", "__global__") != "__global__"
    ))

    return QueryResponse(
        analysis_id=analysis_id,
        question=request.question,
        answer=answer,
        sources=sources,
    )