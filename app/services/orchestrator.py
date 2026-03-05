"""
Orchestrator — full pipeline with all 18 features integrated.
"""

import logging
import uuid
import asyncio

from app.agents.groq_client import GroqClient
from app.agents.llm_router import LLMRouter
from app.services.scanner import clone_repository, scan_repository, generate_repo_map
from app.services.batch_processor import process_files_in_batches
from app.services.report_combiner import combine_and_generate_report
from app.services.analysis_store import AnalysisStore
from app.services.cache import AnalysisCache
from app.services.metrics import PipelineMetrics
from app.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


async def run_analysis_pipeline(
    analysis_id: str,
    repository_url: str,
    store: AnalysisStore,
) -> None:
    """Full pipeline with all features."""
    groq = GroqClient()
    router = LLMRouter(groq)
    vector_store = VectorStore()
    cache = AnalysisCache()
    metrics = PipelineMetrics()

    try:
        # PHASE 1: Clone & Scan + Repo Map (F1, F6)
        await store.set_status(analysis_id, "cloning")
        logger.info("[%s] Phase 1: Cloning %s", analysis_id, repository_url)
        repo_path = await asyncio.to_thread(clone_repository, repository_url)
        files = await asyncio.to_thread(scan_repository, repo_path)

        if not files:
            await store.save_error(analysis_id, repository_url, "No processable files found.")
            return

        # Feature 1: Generate repo map
        repo_map = await asyncio.to_thread(generate_repo_map, repo_path, files)
        await store.save_repo_map(analysis_id, repo_map)
        logger.info("[%s] Phase 1 done: %d files, ~%d tokens", analysis_id, len(files), repo_map.total_tokens_estimate)

        # PHASE 2-3: Batch Process with cache + token budgets (F2, F3, F7, F8, F10)
        await store.set_status(analysis_id, "analyzing_files")
        logger.info("[%s] Phase 2-3: Batch analysis", analysis_id)
        file_analyses, compact_summaries = await process_files_in_batches(
            groq=groq,
            repo_path=repo_path,
            files=files,
            cache=cache,
        )

        if not file_analyses:
            await store.save_error(analysis_id, repository_url, "All file analyses failed.")
            return

        # Feature 2: Save compact summaries
        await store.save_compact_summaries(analysis_id, compact_summaries)

        # PHASE 4 + 6: Combine + Graph + Diagrams (F14)
        await store.set_status(analysis_id, "generating_report")
        logger.info("[%s] Phase 4+6: Report + Diagrams", analysis_id)
        report, dep_graph = await combine_and_generate_report(
            groq=groq,
            router=router,
            analysis_id=analysis_id,
            repository_url=repository_url,
            file_analyses=file_analyses,
            compact_summaries=compact_summaries,
            repo_map=repo_map,
        )
        await store.save_report(report, dep_graph)

        # PHASE 5: RAG Index with smart chunking (F9)
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
        logger.info("[%s] ✅ Pipeline complete | cache=%s | tokens=%s",
                     analysis_id, cache.stats(), groq.token_usage.summary())

    except Exception as e:
        logger.exception("[%s] Pipeline FAILED: %s", analysis_id, e)
        await store.save_error(analysis_id, repository_url, str(e))


def create_analysis_id() -> str:
    return str(uuid.uuid4())