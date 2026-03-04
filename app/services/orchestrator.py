"""
Orchestrator — chains all pipeline phases as a background task.

Phase 1: Clone & Scan → repo map generation
Phase 2-3: Batch Process (Groq) → compact summaries + full analyses
Phase 4: Combine & Graph → architecture summary (via LLM router) + dependency graph
Phase 5: RAG Index (smart chunking)
Phase 6: Mermaid Diagrams

Integrates:
  Feature 1: Repo map generation + persistence
  Feature 2: Compact file summaries
  Feature 8: Analysis cache
  Feature 12: Pipeline metrics
  Feature 13/14: LLM router (Gemini + Groq)
  Feature 15: Architecture with large context via Gemini
"""

import logging
import uuid
import asyncio

from app.agents.groq_client import GroqClient
from app.services.scanner import clone_repository, scan_repository, build_repo_map
from app.services.batch_processor import (
    process_files_in_batches,
    generate_compact_summaries_in_batches,
)
from app.services.report_combiner import combine_and_generate_report
from app.services.analysis_store import AnalysisStore
from app.services.cache import AnalysisCache
from app.services.metrics import AnalysisMetrics
from app.agents.llm_router import LLMRouter
from app.rag.vector_store import VectorStore
from app.config import settings

logger = logging.getLogger(__name__)


def _build_router(groq: GroqClient) -> LLMRouter:
    """Build an LLMRouter, adding Gemini if GEMINI_API_KEY is configured."""
    gemini = None
    if settings.gemini_api_key:
        try:
            from app.agents.gemini_client import GeminiClient
            gemini = GeminiClient()
            logger.info("Gemini client initialized for large-context tasks")
        except Exception as e:
            logger.warning("Gemini init failed (Groq-only mode): %s", e)
    return LLMRouter(groq=groq, gemini=gemini)


async def run_analysis_pipeline(
    analysis_id: str,
    repository_url: str,
    store: AnalysisStore,
) -> None:
    """Full pipeline — runs in background after POST /analyze returns."""
    groq = GroqClient()
    router = _build_router(groq)
    vector_store = VectorStore()
    cache = AnalysisCache(settings.cache_path)
    metrics = AnalysisMetrics()

    try:
        # PHASE 1: Clone & Scan + Repo Map
        await store.set_status(analysis_id, "cloning")
        logger.info("[%s] Phase 1: Cloning %s", analysis_id, repository_url)
        repo_path = await asyncio.to_thread(clone_repository, repository_url)
        files = await asyncio.to_thread(scan_repository, repo_path)

        if not files:
            await store.save_error(analysis_id, repository_url, "No processable files found.")
            return

        # Feature 1: Build and save repo map
        repo_map = build_repo_map(files)
        await store.save_repo_map(analysis_id, repo_map)
        logger.info(
            "[%s] Phase 1 done: %d files | %d tokens estimated",
            analysis_id, len(files), repo_map.total_tokens_estimate,
        )

        # PHASE 2-3: Batch Process (full analyses + compact summaries)
        await store.set_status(analysis_id, "analyzing_files")
        logger.info("[%s] Phase 2-3: Batch analysis", analysis_id)

        # Full analyses (with cache + token budgeting)
        file_analyses = await process_files_in_batches(
            groq=groq, repo_path=repo_path, files=files, cache=cache
        )

        if not file_analyses:
            await store.save_error(analysis_id, repository_url, "All file analyses failed.")
            return

        # Feature 2: Compact summaries (subset to avoid too many LLM calls)
        # Only generate for files up to a reasonable limit
        summary_files = files[:min(len(files), 100)]
        compact_summaries = await generate_compact_summaries_in_batches(
            groq=groq, repo_path=repo_path, files=summary_files
        )
        await store.save_compact_summaries(analysis_id, compact_summaries)
        logger.info("[%s] Generated %d compact summaries", analysis_id, len(compact_summaries))

        # PHASE 4 + 6: Combine + Graph + Architecture + Diagrams
        await store.set_status(analysis_id, "generating_report")
        logger.info("[%s] Phase 4+6: Report + Diagrams", analysis_id)
        report, dep_graph = await combine_and_generate_report(
            groq=groq,
            analysis_id=analysis_id,
            repository_url=repository_url,
            file_analyses=file_analyses,
            router=router,
            repo_map=repo_map,
            compact_summaries=compact_summaries,
        )
        await store.save_report(report, dep_graph)

        # PHASE 5: RAG Index (smart chunking)
        await store.set_status(analysis_id, "building_rag")
        logger.info("[%s] Phase 5: RAG indexing", analysis_id)
        doc_count = await vector_store.index_analysis(
            analysis_id=analysis_id,
            file_analyses=file_analyses,
            architecture_summary=report.architecture_summary,
        )
        logger.info("[%s] Indexed %d docs", analysis_id, doc_count)

        # DONE
        await store.set_status(analysis_id, "completed")
        logger.info(
            "[%s] ✅ Pipeline complete | tokens=%s",
            analysis_id, groq.token_usage.summary(),
        )

    except Exception as e:
        logger.exception("[%s] Pipeline FAILED: %s", analysis_id, e)
        await store.save_error(analysis_id, repository_url, str(e))


def create_analysis_id() -> str:
    return str(uuid.uuid4())
