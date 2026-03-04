"""
Feature 8: File-hash based caching for LLM analysis results.

Uses SHA-256 content hashes to detect file changes and skip re-analysis.
"""

import hashlib
import logging
from pathlib import Path

import orjson

from app.schemas.analysis import FileAnalysisResult, ArchitectureSummary

logger = logging.getLogger(__name__)


class AnalysisCache:
    """Persistent cache for file analysis and architecture results keyed by content hash."""

    def __init__(self, cache_dir: Path):
        self._cache_dir = cache_dir
        self._file_cache_dir = cache_dir / "files"
        self._arch_cache_dir = cache_dir / "architecture"
        self._file_cache_dir.mkdir(parents=True, exist_ok=True)
        self._arch_cache_dir.mkdir(parents=True, exist_ok=True)

    def _file_hash(self, content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    def _file_cache_path(self, content_hash: str) -> Path:
        return self._file_cache_dir / f"{content_hash}.json"

    def _arch_cache_path(self, repo_hash: str) -> Path:
        return self._arch_cache_dir / f"{repo_hash}.json"

    async def get_cached_analysis(
        self, file_path: str, content_hash: str
    ) -> FileAnalysisResult | None:
        """Return cached FileAnalysisResult if it exists for the given hash."""
        path = self._file_cache_path(content_hash)
        if not path.exists():
            return None
        try:
            data = orjson.loads(path.read_bytes())
            result = FileAnalysisResult.model_validate(data)
            logger.debug("Cache HIT: %s (%s)", file_path, content_hash[:8])
            return result
        except Exception as e:
            logger.warning("Cache read error for %s: %s", file_path, e)
            return None

    async def cache_analysis(
        self, file_path: str, content_hash: str, result: FileAnalysisResult
    ) -> None:
        """Persist a FileAnalysisResult under its content hash."""
        path = self._file_cache_path(content_hash)
        try:
            path.write_bytes(orjson.dumps(result.model_dump(), option=orjson.OPT_INDENT_2))
            logger.debug("Cache WRITE: %s (%s)", file_path, content_hash[:8])
        except Exception as e:
            logger.warning("Cache write error for %s: %s", file_path, e)

    async def get_cached_architecture(
        self, repo_hash: str
    ) -> ArchitectureSummary | None:
        """Return cached ArchitectureSummary if it exists."""
        path = self._arch_cache_path(repo_hash)
        if not path.exists():
            return None
        try:
            data = orjson.loads(path.read_bytes())
            return ArchitectureSummary.model_validate(data)
        except Exception as e:
            logger.warning("Cache read error for architecture %s: %s", repo_hash, e)
            return None

    async def cache_architecture(
        self, repo_hash: str, summary: ArchitectureSummary
    ) -> None:
        """Persist an ArchitectureSummary."""
        path = self._arch_cache_path(repo_hash)
        try:
            path.write_bytes(orjson.dumps(summary.model_dump(), option=orjson.OPT_INDENT_2))
        except Exception as e:
            logger.warning("Cache write error for architecture: %s", e)

    def compute_hash(self, content: bytes) -> str:
        """Public helper to compute SHA-256 hash of content."""
        return self._file_hash(content)
