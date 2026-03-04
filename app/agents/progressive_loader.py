"""
Feature 5: Progressive Context Loading.

Reasons in stages to minimize token usage while maximising relevance:
  Stage 1: Send repo map → identify relevant areas
  Stage 2: Send compact summaries of candidates → narrow down to ~10 files
  Stage 3: Load raw code for top 3-5 files
"""

import logging
from pathlib import Path

from app.agents.groq_client import GroqClient, TaskType
from app.schemas.analysis import RepoMap, CompactFileSummary
from app.services.token_utils import estimate_tokens

logger = logging.getLogger(__name__)

_STAGE1_SYSTEM = """\
You are a code navigation assistant. Given a repository file map, identify which
directories and files are most likely relevant to a user's question.
Return a JSON object with key "candidate_paths" (list of file paths).\
"""

_STAGE2_SYSTEM = """\
You are a code navigation assistant. Given compact file summaries, identify which
files are most relevant to answer the user's question.
Return a JSON object with key "top_files" (list of up to 5 file paths).\
"""


class ProgressiveContextLoader:
    """
    Loads context for RAG in three progressive stages to minimise token waste.
    """

    def __init__(self, groq: GroqClient, repo_path: Path | None = None):
        self._groq = groq
        self._repo_path = repo_path

    async def load_context(
        self,
        query: str,
        repo_map: RepoMap | None,
        summaries: list[CompactFileSummary],
        repo_path: Path | None = None,
    ) -> dict:
        """
        Progressively load context for the given query.

        Returns:
            {
                "context": str,         # assembled context text
                "sources": list[str],   # file paths used
            }
        """
        path = repo_path or self._repo_path

        # Stage 1: Repo map → candidate files
        candidates = await self._stage1_repo_map(query, repo_map)
        if not candidates and summaries:
            candidates = [s.file_path for s in summaries]

        # Stage 2: Filter candidates using compact summaries
        relevant_summaries = [
            s for s in summaries if s.file_path in candidates
        ] if candidates else summaries
        top_files = await self._stage2_filter_summaries(query, relevant_summaries)

        # Stage 3: Load raw code for top files
        context_parts = []
        sources = []
        files_loaded = 0
        max_files = 5

        for file_path in top_files[:max_files]:
            raw_code = self._load_file(file_path, path)
            if raw_code:
                token_count = estimate_tokens(raw_code)
                context_parts.append(f"### {file_path}\n```\n{raw_code}\n```\n")
                sources.append(file_path)
                files_loaded += 1
                logger.debug("Stage 3 loaded %s (%d tokens)", file_path, token_count)

        return {
            "context": "\n".join(context_parts),
            "sources": sources,
        }

    async def _stage1_repo_map(
        self, query: str, repo_map: RepoMap | None
    ) -> list[str]:
        """Stage 1: Use repo map to identify candidate directories/files."""
        if not repo_map or not repo_map.files:
            return []

        file_list = "\n".join(
            f"  {path} ({info.language}, {info.size} bytes)"
            for path, info in list(repo_map.files.items())[:300]
        )
        prompt = f"""\
USER QUESTION: {query}

REPOSITORY FILE MAP:
{file_list}

Which file paths are most likely relevant to answer this question?
Return JSON: {{"candidate_paths": ["path1", "path2", ...]}}
"""
        try:
            import json as _json
            raw = await self._groq.chat(
                prompt, _STAGE1_SYSTEM, temperature=0.2,
                task=TaskType.QUERY_PLANNING,
            )
            cleaned = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            data = _json.loads(cleaned)
            return data.get("candidate_paths", [])
        except Exception as e:
            logger.debug("Stage 1 failed, skipping: %s", e)
            return []

    async def _stage2_filter_summaries(
        self, query: str, summaries: list[CompactFileSummary]
    ) -> list[str]:
        """Stage 2: Use compact summaries to narrow down to top files."""
        if not summaries:
            return []

        summary_text = "\n".join(
            f"  {s.file_path}: {s.purpose}"
            for s in summaries[:50]
        )
        prompt = f"""\
USER QUESTION: {query}

COMPACT FILE SUMMARIES:
{summary_text}

Which files (at most 5) are most relevant?
Return JSON: {{"top_files": ["path1", "path2", ...]}}
"""
        try:
            import json as _json
            raw = await self._groq.chat(
                prompt, _STAGE2_SYSTEM, temperature=0.2,
                task=TaskType.QUERY_PLANNING,
            )
            cleaned = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            data = _json.loads(cleaned)
            return data.get("top_files", [s.file_path for s in summaries[:5]])
        except Exception as e:
            logger.debug("Stage 2 failed, using top summaries: %s", e)
            return [s.file_path for s in summaries[:5]]

    def _load_file(self, file_path: str, base_path: Path | None) -> str:
        """Load raw file content from disk."""
        if not base_path:
            return ""
        full_path = base_path / file_path
        if not full_path.exists():
            return ""
        try:
            return full_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.debug("Failed to load %s: %s", file_path, e)
            return ""
