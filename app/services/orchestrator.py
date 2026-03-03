"""
Orchestrator — coordinates the entire analysis pipeline.

Phases:
  1. Clone & scan repository
  2. Batch process files with worker pool
  3. Combine reports + build graph
  4. Generate architecture summary & Mermaid diagrams
  5. Build RAG vector store

All phases run asynchronously in the background after the API returns the analysis_id.
"""

import logging
import uuid
import asyncio
from pathlib import Path

from app.config import settings
from app.agents.grok_client import GrokClient
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
    """
    Full analysis pipeline — runs in the background.

    This is the main orchestration function that chains all phases together.
    """
    grok = GrokClient()
    vector_store = VectorStore()

    try:
        # ── PHASE 1: Clone & Scan ──
        await store.set_status(analysis_id, "cloning")
        logger.info("[%s] Phase 1: Cloning repository %s", analysis_id, repository_url)

        repo_path = await asyncio.to_thread(clone_repository, repository_url)
        files = await asyncio.to_thread(scan_repository, repo_path)

        if not files:
            await store.save_error(analysis_id, repository_url, "No processable files found in repository.")
            return

        logger.info("[%s] Phase 1 complete: %d files found", analysis_id, len(files))

        # ── PHASE 2 & 3: Batch Process Files ──
        await store.set_status(analysis_id, "analyzing_files")
        logger.info("[%s] Phase 2-3: Batch processing %d files", analysis_id, len(files))

        file_analyses = await process_files_in_batches(
            grok=grok,
            repo_path=repo_path,
            files=files,
        )

        if not file_analyses:
            await store.save_error(analysis_id, repository_url, "All file analyses failed.")
            return

        logger.info("[%s] Phase 2-3 complete: %d files analyzed", analysis_id, len(file_analyses))

        # ── PHASE 4 & 6: Combine Reports + Architecture + Diagrams ──
        await store.set_status(analysis_id, "generating_report")
        logger.info("[%s] Phase 4-6: Combining reports and generating diagrams", analysis_id)

        report, dep_graph = await combine_and_generate_report(
            grok=grok,
            analysis_id=analysis_id,
            repository_url=repository_url,
            file_analyses=file_analyses,
        )

        # Save report
        await store.save_report(report, dep_graph)

        # ── PHASE 5: Build RAG Vector Store ──
        await store.set_status(analysis_id, "building_rag")
        logger.info("[%s] Phase 5: Building RAG vector store", analysis_id)

        doc_count = await vector_store.index_analysis(
            analysis_id=analysis_id,
            file_analyses=file_analyses,
            architecture_summary=report.architecture_summary,
        )

        logger.info("[%s] Phase 5 complete: %d documents indexed", analysis_id, doc_count)

        # ── DONE ──
        await store.set_status(analysis_id, "completed")
        logger.info("[%s] ✅ Analysis pipeline completed successfully", analysis_id)

        # Log final token usage
        usage = grok.token_usage.summary()
        logger.info("[%s] Final token usage: %s", analysis_id, usage)

    except Exception as e:
        logger.exception("[%s] Pipeline failed: %s", analysis_id, e)
        await store.save_error(analysis_id, repository_url, str(e))


def create_analysis_id() -> str:
    """Generate a unique analysis ID."""
    return str(uuid.uuid4())