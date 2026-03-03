"""
Batch Processor — processes files in configurable batches using an async worker pool.

KEY DESIGN:
  - Does NOT create one agent per file.
  - Creates a fixed-size worker pool (configurable).
  - Files are queued and processed in rounds/batches.
  - Concurrency is limited by asyncio.Semaphore.
  - Each batch is processed concurrently within the pool limit.

HOW BATCHING WORKS:
  1. All files are split into batches of `batch_size` (e.g., 10 files).
  2. Each batch is processed concurrently using asyncio.gather.
  3. Within each batch, a semaphore limits concurrent Grok API calls
     (e.g., max 5 concurrent calls even within a batch of 10).
  4. After a batch completes, the next batch starts.
  5. This prevents API overload while maintaining throughput.

HOW RATE LIMITING IS HANDLED:
  1. asyncio.Semaphore in GrokClient limits concurrent API calls globally.
  2. tenacity retry decorator handles 429 (rate limit) responses with exponential backoff.
  3. Batching naturally throttles throughput — only N files at a time.
  4. Token usage is tracked for monitoring/budgeting.
"""

import logging
import asyncio
from pathlib import Path

from app.config import settings
from app.agents.grok_client import GrokClient
from app.agents.file_analysis_agent import analyze_file
from app.parsers.treesitter_parser import parse_file
from app.schemas.analysis import FileMetadata, FileAnalysisResult

logger = logging.getLogger(__name__)


async def _process_single_file(
    grok: GrokClient,
    repo_path: Path,
    metadata: FileMetadata,
) -> FileAnalysisResult | None:
    """Process a single file: read → parse → analyze."""
    file_path = repo_path / metadata.path

    try:
        content_bytes = file_path.read_bytes()
        content_text = content_bytes.decode("utf-8", errors="replace")
    except Exception as e:
        logger.warning("Failed to read file %s: %s", metadata.path, e)
        return None

    # Tree-sitter parsing
    parsed = parse_file(metadata.path, content_bytes, metadata.extension)

    # LLM analysis
    result = await analyze_file(
        grok=grok,
        file_path=metadata.path,
        content=content_text,
        metadata=metadata,
        parsed=parsed,
    )

    return result


async def process_files_in_batches(
    grok: GrokClient,
    repo_path: Path,
    files: list[FileMetadata],
    batch_size: int | None = None,
) -> list[FileAnalysisResult]:
    """
    Process all files in batches with limited concurrency.

    Batching Strategy:
      - Split files into batches of `batch_size`.
      - Process each batch with asyncio.gather (concurrent within batch).
      - GrokClient's semaphore limits actual concurrent API calls.
      - Move to next batch only after current batch completes.

    Args:
        grok: Shared GrokClient instance (with built-in rate limiting).
        repo_path: Path to cloned repository.
        files: List of file metadata to process.
        batch_size: Number of files per batch (default from settings).

    Returns:
        List of FileAnalysisResult for all successfully processed files.
    """
    batch_size = batch_size or settings.batch_size
    total = len(files)
    results: list[FileAnalysisResult] = []
    failed_count = 0

    logger.info(
        "Starting batch processing: %d files, batch_size=%d, max_concurrent_llm=%d",
        total,
        batch_size,
        settings.max_concurrent_llm_calls,
    )

    # Split into batches
    batches = [files[i:i + batch_size] for i in range(0, total, batch_size)]

    for batch_idx, batch in enumerate(batches):
        batch_num = batch_idx + 1
        logger.info(
            "Processing batch %d/%d (%d files)",
            batch_num,
            len(batches),
            len(batch),
        )

        # Process all files in this batch concurrently
        tasks = [
            _process_single_file(grok, repo_path, metadata)
            for metadata in batch
        ]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(batch_results):
            if isinstance(result, Exception):
                logger.error(
                    "Batch %d, file %s failed: %s",
                    batch_num,
                    batch[i].path,
                    result,
                )
                failed_count += 1
            elif result is not None:
                results.append(result)
            else:
                failed_count += 1

        logger.info(
            "Batch %d/%d completed. Total processed: %d, failed: %d",
            batch_num,
            len(batches),
            len(results),
            failed_count,
        )

    # Log token usage
    usage = grok.token_usage.summary()
    logger.info(
        "Batch processing complete. Token usage: %s",
        usage,
    )

    return results