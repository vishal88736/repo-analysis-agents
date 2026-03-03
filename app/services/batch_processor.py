"""
Batch Processor — processes files in configurable batches using an async worker pool.

=== HOW BATCHING WORKS ===

  1. All files are split into batches of `batch_size` (default: 10 files per batch).
  2. Each batch is processed concurrently using `asyncio.gather()`.
  3. Within each batch, a global `asyncio.Semaphore` (inside GroqClient)
     limits actual concurrent Groq API calls to `max_concurrent_llm_calls` (default: 5).
  4. After a batch completes (all files done or errored), the next batch starts.
  5. This creates a "round-robin" effect — controlled throughput, not flood.

  Example: 100 files, batch_size=10, max_concurrent=5:
    → 10 batches, each with 10 files
    → Within each batch, at most 5 Groq API calls run in parallel
    → Batch 1 finishes → Batch 2 starts → ... → Batch 10 finishes

=== HOW RATE LIMITING IS HANDLED ===

  Layer 1: asyncio.Semaphore(5) — caps parallel Groq API requests globally
  Layer 2: tenacity @retry — catches Groq 429/timeout errors, retries with 2s→4s→8s→...60s backoff
  Layer 3: Batch rounds — only N files process at once, natural throttle
  Layer 4: TokenUsageTracker — logs all token consumption for monitoring/budgeting

  This 4-layer approach ensures we never overwhelm Groq's free tier or paid rate limits.
"""

import logging
import asyncio
from pathlib import Path

from app.config import settings
from app.agents.groq_client import GroqClient
from app.agents.file_analysis_agent import analyze_file
from app.parsers.treesitter_parser import parse_file
from app.schemas.analysis import FileMetadata, FileAnalysisResult

logger = logging.getLogger(__name__)


async def _process_single_file(
    groq: GroqClient,
    repo_path: Path,
    metadata: FileMetadata,
) -> FileAnalysisResult | None:
    """Read → Parse (tree-sitter) → Analyze (Groq LLM) a single file."""
    file_path = repo_path / metadata.path

    try:
        content_bytes = file_path.read_bytes()
        content_text = content_bytes.decode("utf-8", errors="replace")
    except Exception as e:
        logger.warning("Failed to read %s: %s", metadata.path, e)
        return None

    parsed = parse_file(metadata.path, content_bytes, metadata.extension)

    return await analyze_file(
        groq=groq,
        file_path=metadata.path,
        content=content_text,
        metadata=metadata,
        parsed=parsed,
    )


async def process_files_in_batches(
    groq: GroqClient,
    repo_path: Path,
    files: list[FileMetadata],
    batch_size: int | None = None,
) -> list[FileAnalysisResult]:
    """
    Process all files in batches. Worker pool pattern, NOT one agent per file.

    The GroqClient's semaphore is the actual concurrency limiter.
    Batching adds a second layer of flow control.
    """
    batch_size = batch_size or settings.batch_size
    total = len(files)
    results: list[FileAnalysisResult] = []
    failed = 0

    logger.info(
        "Batch processing: %d files | batch_size=%d | max_concurrent_llm=%d",
        total, batch_size, settings.max_concurrent_llm_calls,
    )

    batches = [files[i:i + batch_size] for i in range(0, total, batch_size)]

    for idx, batch in enumerate(batches):
        batch_num = idx + 1
        logger.info("Batch %d/%d (%d files)", batch_num, len(batches), len(batch))

        tasks = [_process_single_file(groq, repo_path, m) for m in batch]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(batch_results):
            if isinstance(result, Exception):
                logger.error("Batch %d file %s failed: %s", batch_num, batch[i].path, result)
                failed += 1
            elif result is not None:
                results.append(result)
            else:
                failed += 1

        logger.info("Batch %d done | processed=%d | failed=%d", batch_num, len(results), failed)

    logger.info("All batches complete | total=%d | success=%d | failed=%d | tokens=%s",
                total, len(results), failed, groq.token_usage.summary())
    return results