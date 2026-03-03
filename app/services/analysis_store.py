"""
Analysis Store — persists analysis reports to disk as JSON.
Thread-safe in-memory status tracking + JSON file persistence.
"""

import json
import logging
import asyncio
from pathlib import Path

import orjson

from app.schemas.analysis import FullAnalysisReport
from app.schemas.graph_models import DependencyGraph

logger = logging.getLogger(__name__)


class AnalysisStore:
    """
    Simple file-based analysis store.
    For production, replace with PostgreSQL/Redis.
    """

    def __init__(self, store_dir: Path):
        self._store_dir = store_dir
        self._store_dir.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._status: dict[str, str] = {}  # analysis_id → status

    def _report_path(self, analysis_id: str) -> Path:
        return self._store_dir / f"{analysis_id}_report.json"

    def _graph_path(self, analysis_id: str) -> Path:
        return self._store_dir / f"{analysis_id}_graph.json"

    async def set_status(self, analysis_id: str, status: str):
        async with self._lock:
            self._status[analysis_id] = status

    async def get_status(self, analysis_id: str) -> str | None:
        async with self._lock:
            return self._status.get(analysis_id)

    async def save_report(
        self,
        report: FullAnalysisReport,
        graph: DependencyGraph | None = None,
    ):
        """Save report and optionally the dependency graph to disk."""
        path = self._report_path(report.analysis_id)
        data = orjson.dumps(report.model_dump(), option=orjson.OPT_INDENT_2)
        path.write_bytes(data)

        if graph:
            graph_path = self._graph_path(report.analysis_id)
            graph_data = orjson.dumps(graph.model_dump(), option=orjson.OPT_INDENT_2)
            graph_path.write_bytes(graph_data)

        await self.set_status(report.analysis_id, report.status)
        logger.info("Saved report for analysis %s", report.analysis_id)

    async def load_report(self, analysis_id: str) -> FullAnalysisReport | None:
        """Load a report from disk."""
        path = self._report_path(analysis_id)
        if not path.exists():
            return None
        try:
            data = orjson.loads(path.read_bytes())
            return FullAnalysisReport.model_validate(data)
        except Exception as e:
            logger.error("Failed to load report %s: %s", analysis_id, e)
            return None

    async def load_graph(self, analysis_id: str) -> DependencyGraph | None:
        """Load a dependency graph from disk."""
        path = self._graph_path(analysis_id)
        if not path.exists():
            return None
        try:
            data = orjson.loads(path.read_bytes())
            return DependencyGraph.model_validate(data)
        except Exception as e:
            logger.error("Failed to load graph %s: %s", analysis_id, e)
            return None

    async def save_error(self, analysis_id: str, repository_url: str, error: str):
        """Save a failed analysis report."""
        report = FullAnalysisReport(
            analysis_id=analysis_id,
            repository_url=repository_url,
            status="failed",
            error_message=error,
        )
        await self.save_report(report)