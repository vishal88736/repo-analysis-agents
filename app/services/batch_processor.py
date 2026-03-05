"""
Batch Processor — with token budgeting (F3) and caching (F8).
"""

import logging
import asyncio
from pathlib import Path

from app.config import settings
from app.agents.groq_client import GroqClient
from app.agents.file_analysis_agent import analyze_file
from app.parsers.treesitter_parser import parse_file
from app.schemas.analysis import FileMetadata, FileAnalysisResult, CompactFileSummary
from app.services.cache import AnalysisCache
from app.services.token_utils import estimate_tokens, compress_code, split_by_token_budget

logger = logging.getLogger(__name__)


def _build_compact_summary(fa: FileAnalysisResult) -> CompactFileSummary:
    """Feature 2: Build compact summary from full analysis."""
    return CompactFileSummary(
        file_path=fa.file_path,
        purpose=fa.summary[:200] if fa.summary else "",
        functions=[f.name for f in fa.functions],
        classes=[c.name for c in fa.classes],
        imports=[d for d in fa.external_dependencies],
        key_dependencies=fa.external_dependencies[:10],
        entry_point=any(
            f.name in ("main", "__main__", "app", "run", "start", "handler")
            for f in fa.functions
        ),
    )


async def _process_single_file(
    groq: GroqClient,
    repo_path: Path,
    metadata: FileMetadata,
    cache: AnalysisCache,
) -> tuple[FileAnalysisResult | None, CompactFileSummary | None]:
    """Read → Cache check → Compress → Parse → Analyze."""
    file_path = repo_path / metadata.path

    try:
        content_bytes = file_path.read_bytes()
        content_text = content_bytes.decode("utf-8", errors="replace")
    except Exception as e:
        logger.warning("Failed to read %s: %s", metadata.path, e)
        return None, None

    # Feature 8: Check cache
    content_hash = AnalysisCache.hash_content(content_bytes)
    cached = cache.get(content_hash)
    if cached:
        logger.info("♻ Cache hit: %s", metadata.path)
        summary = cache.get_summary(content_hash)
        if not summary:
            summary = _build_compact_summary(cached)
        return cached, summary

    # Feature 10: Compress before analysis
    compressed = compress_code(content_text, metadata.language)

    # Feature 3: Token budgeting
    tokens = estimate_tokens(compressed)
    if tokens > settings.max_file_tokens:
        # Split into chunks and analyze separately
        chunks = split_by_token_budget(compressed, settings.max_file_tokens)
        logger.info("Splitting %s into %d chunks (%d tokens)", metadata.path, len(chunks), tokens)

        # Analyze first chunk (most important — imports, class defs)
        parsed = parse_file(metadata.path, content_bytes, metadata.extension)
        result = await analyze_file(
            groq=groq,
            file_path=metadata.path,
            content=chunks[0],
            metadata=metadata,
            parsed=parsed,
        )

        # Add note about truncation
        if result and len(chunks) > 1:
            result.summary += f" [Analyzed first {settings.max_file_tokens} tokens of {tokens} total]"
    else:
        parsed = parse_file(metadata.path, content_bytes, metadata.extension)
        result = await analyze_file(
            groq=groq,
            file_path=metadata.path,
            content=compressed,
            metadata=metadata,
            parsed=parsed,
        )

    if result:
        summary = _build_compact_summary(result)
        # Feature 8: Store in cache
        cache.put(content_hash, result, summary)
        return result, summary

    return None, None


async def process_files_in_batches(
    groq: GroqClient,
    repo_path: Path,
    files: list[FileMetadata],
    batch_size: int | None = None,
    cache: AnalysisCache | None = None,
) -> tuple[list[FileAnalysisResult], list[CompactFileSummary]]:
    """
    Process files with caching + token budgets.
    Returns both full analyses and compact summaries.
    """
    batch_size = batch_size or settings.batch_size
    if cache is None:
        cache = AnalysisCache()

    total = len(files)
    results: list[FileAnalysisResult] = []
    summaries: list[CompactFileSummary] = []
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
                analysis, summary = result
                if analysis:
                    results.append(analysis)
                if summary:
                    summaries.append(summary)
            else:
                failed += 1

        logger.info("Batch %d done | processed=%d | failed=%d", batch_num, len(results), failed)

    logger.info(
        "All batches complete | total=%d | success=%d | failed=%d | cache=%s | tokens=%s",
        total, len(results), failed, cache.stats(), groq.token_usage.summary(),
    )
    return results, summaries