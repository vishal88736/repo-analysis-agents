"""
REST API Endpoints:
  POST /analyze       → Start analysis (returns analysis_id)
  GET  /status/{id}   → Check pipeline status
  GET  /report/{id}   → Full report with summaries + diagrams
  POST /query         → RAG question answering
"""

import logging
import asyncio

from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException, status

from app.api.dependencies import get_analysis_store, get_vector_store
from app.schemas.api_models import (
    AnalyzeRequest, AnalyzeResponse, QueryRequest, QueryResponse,
    ReportResponse, StatusResponse, ErrorResponse,
)
from app.services.analysis_store import AnalysisStore
from app.services.orchestrator import run_analysis_pipeline, create_analysis_id
from app.agents.groq_client import GroqClient
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
    """Start repository analysis. Processing happens in background."""
    url = request.repository_url.strip()
    if not url.startswith("https://github.com/"):
        raise HTTPException(status_code=400, detail="Only GitHub URLs supported (https://github.com/owner/repo)")

    analysis_id = create_analysis_id()
    await store.set_status(analysis_id, "pending")

    background_tasks.add_task(run_analysis_pipeline, analysis_id, url, store)

    return AnalyzeResponse(
        analysis_id=analysis_id,
        status="pending",
        message=f"Analysis started. Track: GET /api/v1/status/{analysis_id}",
    )


@router.get("/status/{analysis_id}", response_model=StatusResponse)
async def get_status(analysis_id: str, store: AnalysisStore = Depends(get_analysis_store)):
    """Check analysis pipeline status."""
    s = await store.get_status(analysis_id)
    if s is None:
        raise HTTPException(status_code=404, detail=f"Analysis '{analysis_id}' not found.")
    return StatusResponse(analysis_id=analysis_id, status=s)


@router.get(
    "/report/{analysis_id}",
    response_model=ReportResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_report(analysis_id: str, store: AnalysisStore = Depends(get_analysis_store)):
    """Get the full analysis report with summaries, diagrams, etc."""
    s = await store.get_status(analysis_id)
    if s is None:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    if s not in ("completed", "failed"):
        raise HTTPException(status_code=202, detail=f"Still processing. Status: {s}")

    report = await store.load_report(analysis_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report file not found.")
    if report.status == "failed":
        raise HTTPException(status_code=500, detail=f"Analysis failed: {report.error_message}")

    return ReportResponse(
        analysis_id=report.analysis_id,
        repository_url=report.repository_url,
        status=report.status,
        total_files=report.total_files,
        global_summary=report.architecture_summary.overview,
        key_components=report.architecture_summary.key_components,
        design_patterns=report.architecture_summary.design_patterns,
        technology_stack=report.architecture_summary.technology_stack,
        entry_points=[ep.model_dump() for ep in report.architecture_summary.entry_points],
        file_summaries=[
            {
                "file_path": fa.file_path,
                "summary": fa.summary,
                "functions": len(fa.functions),
                "classes": len(fa.classes),
                "dependencies": fa.external_dependencies,
            }
            for fa in report.file_analyses
        ],
        mermaid_diagrams=[
            {"title": d.title, "type": d.diagram_type, "syntax": d.mermaid_syntax}
            for d in report.mermaid_diagrams
        ],
    )


@router.post("/query", response_model=QueryResponse, responses={404: {"model": ErrorResponse}})
async def query_codebase(
    request: QueryRequest,
    store: AnalysisStore = Depends(get_analysis_store),
    vector_store: VectorStore = Depends(get_vector_store),
):
    """RAG-based question answering: vector search → context → Groq LLM answer."""
    s = await store.get_status(request.analysis_id)
    if s is None:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    if s != "completed":
        raise HTTPException(status_code=400, detail=f"Not ready. Status: {s}")

    report = await store.load_report(request.analysis_id)
    graph = await store.load_graph(request.analysis_id)

    # Step 1: Vector search
    chunks = await vector_store.search(analysis_id=request.analysis_id, query=request.question)

    # Step 2: Dependency context augmentation
    dep_context = ""
    if graph and chunks:
        relevant_files = {
            c.get("metadata", {}).get("file_path", "")
            for c in chunks[:5]
            if c.get("metadata", {}).get("file_path", "__global__") != "__global__"
        }
        dep_context = "\n".join(
            get_dependency_context_for_file(graph, fp) for fp in relevant_files
        )

    # Step 3: Groq LLM answer
    groq = GroqClient()
    arch = report.architecture_summary.overview if report else ""
    answer = await answer_query(groq, request.question, chunks, dep_context, arch)

    sources = list({
        c.get("metadata", {}).get("file_path", "")
        for c in chunks
        if c.get("metadata", {}).get("file_path", "__global__") != "__global__"
    })

    return QueryResponse(
        analysis_id=request.analysis_id,
        question=request.question,
        answer=answer,
        sources=sources,
    )