"""
Orchestrator — chains all 6 phases as a background pipeline.

Phase 1: Clone & Scan → Phase 2-3: Batch Process (Groq) → Phase 4: Combine & Graph →
Phase 5: RAG Index → Phase 6: Mermaid Diagrams
"""

import logging
import uuid
import asyncio

from app.agents.groq_client import GroqClient
from app.services.scanner import clone_repository, scan_repository
from app.services.batch_processor import process_files_in_batches
from app.services.report_combiner import combine_and_generate_report
from app.services.analysis_store import AnalysisStore
from app.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


async def run_analysis_pipeline(
    analysis_id: str,
    repository_url: str,
    store: AnalysisStore,
) -> None:
    """Full pipeline — runs in background after POST /analyze returns."""
    groq = GroqClient()
    vector_store = VectorStore()

    try:
        # PHASE 1: Clone & Scan
        await store.set_status(analysis_id, "cloning")
        logger.info("[%s] Phase 1: Cloning %s", analysis_id, repository_url)
        repo_path = await asyncio.to_thread(clone_repository, repository_url)
        files = await asyncio.to_thread(scan_repository, repo_path)

        if not files:
            await store.save_error(analysis_id, repository_url, "No processable files found.")
            return

        logger.info("[%s] Phase 1 done: %d files", analysis_id, len(files))

        # PHASE 2-3: Batch Process
        await store.set_status(analysis_id, "analyzing_files")
        logger.info("[%s] Phase 2-3: Batch analysis", analysis_id)
        file_analyses = await process_files_in_batches(groq=groq, repo_path=repo_path, files=files)

        if not file_analyses:
            await store.save_error(analysis_id, repository_url, "All file analyses failed.")
            return

        # PHASE 4 + 6: Combine + Graph + Diagrams
        await store.set_status(analysis_id, "generating_report")
        logger.info("[%s] Phase 4+6: Report + Diagrams", analysis_id)
        report, dep_graph = await combine_and_generate_report(
            groq=groq, analysis_id=analysis_id,
            repository_url=repository_url, file_analyses=file_analyses,
        )
        await store.save_report(report, dep_graph)

        # PHASE 5: RAG Index
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
        logger.info("[%s] ✅ Pipeline complete | tokens=%s", analysis_id, groq.token_usage.summary())

    except Exception as e:
        logger.exception("[%s] Pipeline FAILED: %s", analysis_id, e)
        await store.save_error(analysis_id, repository_url, str(e))


def create_analysis_id() -> str:
    return str(uuid.uuid4())