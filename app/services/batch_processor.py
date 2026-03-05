"""
Batch Processor — passes project file list for cross-file dependency detection.
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
    all_project_files: list[str],
) -> tuple[FileAnalysisResult | None, CompactFileSummary | None]:
    file_path = repo_path / metadata.path

    try:
        content_bytes = file_path.read_bytes()
        content_text = content_bytes.decode("utf-8", errors="replace")
    except Exception as e:
        logger.warning("Failed to read %s: %s", metadata.path, e)
        return None, None

    # Check cache
    content_hash = AnalysisCache.hash_content(content_bytes)
    cached = cache.get(content_hash)
    if cached:
        logger.info("♻ Cache hit: %s", metadata.path)
        summary = cache.get_summary(content_hash)
        if not summary:
            summary = _build_compact_summary(cached)
        return cached, summary

    # Compress
    compressed = compress_code(content_text, metadata.language)

    # Token budgeting
    tokens = estimate_tokens(compressed)
    if tokens > settings.max_file_tokens:
        chunks = split_by_token_budget(compressed, settings.max_file_tokens)
        logger.info("Splitting %s into %d chunks (%d tokens)", metadata.path, len(chunks), tokens)
        parsed = parse_file(metadata.path, content_bytes, metadata.extension)
        result = await analyze_file(
            groq=groq,
            file_path=metadata.path,
            content=chunks[0],
            metadata=metadata,
            parsed=parsed,
            all_project_files=all_project_files,
        )
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
            all_project_files=all_project_files,
        )

    if result:
        summary = _build_compact_summary(result)
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
    batch_size = batch_size or settings.batch_size
    if cache is None:
        cache = AnalysisCache()

    # Build complete project file list for cross-reference detection
    all_project_files = [f.path for f in files]

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

        tasks = [
            _process_single_file(groq, repo_path, m, cache, all_project_files)
            for m in batch
        ]
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

    # Log inter-file dependency stats
    total_refs = sum(len(r.internal_file_references) for r in results)
    total_interactions = sum(len(r.file_interactions) for r in results)
    logger.info(
        "All batches complete | total=%d | success=%d | failed=%d | "
        "file_refs=%d | interactions=%d | cache=%s",
        total, len(results), failed, total_refs, total_interactions, cache.stats(),
    )
    return results, summaries