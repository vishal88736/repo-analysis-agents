"""
Batch Processor — processes files in configurable batches using an async worker pool.

Feature 3: Token budgeting — estimates tokens before LLM call, splits large files.
Feature 8: Cache integration — checks cache before calling Groq.

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
from app.agents.file_analysis_agent import analyze_file, generate_compact_summary
from app.parsers.treesitter_parser import parse_file
from app.schemas.analysis import FileMetadata, FileAnalysisResult, CompactFileSummary
from app.services.token_utils import estimate_tokens, split_by_boundaries, combine_chunk_results
from app.services.cache import AnalysisCache

logger = logging.getLogger(__name__)


async def _process_single_file(
    groq: GroqClient,
    repo_path: Path,
    metadata: FileMetadata,
    cache: AnalysisCache | None = None,
) -> FileAnalysisResult | None:
    """Read → Parse (tree-sitter) → [Cache check] → Analyze (Groq LLM) a single file."""
    file_path = repo_path / metadata.path

    try:
        content_bytes = file_path.read_bytes()
        content_text = content_bytes.decode("utf-8", errors="replace")
    except Exception as e:
        logger.warning("Failed to read %s: %s", metadata.path, e)
        return None

    # Feature 8: Check cache before LLM call
    if cache is not None:
        content_hash = cache.compute_hash(content_bytes)
        cached = await cache.get_cached_analysis(metadata.path, content_hash)
        if cached is not None:
            logger.debug("Cache HIT (skipping LLM): %s", metadata.path)
            return cached

    parsed = parse_file(metadata.path, content_bytes, metadata.extension)

    # Feature 3: Token budgeting — split large files
    token_count = estimate_tokens(content_text)
    if token_count > settings.max_file_tokens:
        logger.debug(
            "Large file %s (%d tokens > %d) — splitting into chunks",
            metadata.path, token_count, settings.max_file_tokens,
        )
        chunks = split_by_boundaries(content_text, parsed, settings.max_file_tokens)
        chunk_results = []
        for chunk_text in chunks:
            chunk_result = await analyze_file(
                groq=groq,
                file_path=metadata.path,
                content=chunk_text,
                metadata=metadata,
                parsed=parsed,
            )
            chunk_results.append(chunk_result)
        result = combine_chunk_results(chunk_results)
    else:
        result = await analyze_file(
            groq=groq,
            file_path=metadata.path,
            content=content_text,
            metadata=metadata,
            parsed=parsed,
        )

    # Feature 8: Cache the result
    if cache is not None and result is not None:
        await cache.cache_analysis(metadata.path, content_hash, result)

    return result


async def _process_compact_summary(
    groq: GroqClient,
    repo_path: Path,
    metadata: FileMetadata,
) -> CompactFileSummary | None:
    """Feature 2: Generate a compact summary for a single file."""
    file_path = repo_path / metadata.path
    try:
        content_bytes = file_path.read_bytes()
        content_text = content_bytes.decode("utf-8", errors="replace")
    except Exception as e:
        logger.warning("Failed to read %s for compact summary: %s", metadata.path, e)
        return None

    parsed = parse_file(metadata.path, content_bytes, metadata.extension)
    return await generate_compact_summary(
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
    cache: AnalysisCache | None = None,
) -> list[FileAnalysisResult]:
    """
    Process all files in batches. Worker pool pattern, NOT one agent per file.

    Feature 3: Token budgeting applied per file.
    Feature 8: Cache checked before each LLM call.

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

        tasks = [_process_single_file(groq, repo_path, m, cache) for m in batch]
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


async def generate_compact_summaries_in_batches(
    groq: GroqClient,
    repo_path: Path,
    files: list[FileMetadata],
    batch_size: int | None = None,
) -> list[CompactFileSummary]:
    """Feature 2: Generate compact summaries for all files in batches."""
    batch_size = batch_size or settings.batch_size
    results: list[CompactFileSummary] = []

    batches = [files[i:i + batch_size] for i in range(0, len(files), batch_size)]
    for idx, batch in enumerate(batches):
        tasks = [_process_compact_summary(groq, repo_path, m) for m in batch]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in batch_results:
            if isinstance(result, CompactFileSummary):
                results.append(result)

        logger.info("Compact summaries batch %d/%d done", idx + 1, len(batches))

    return results
