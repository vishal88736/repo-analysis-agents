"""Analysis Store — JSON-based persistence for reports and graphs."""

import logging
import asyncio
from pathlib import Path

import orjson

from app.schemas.analysis import FullAnalysisReport
from app.schemas.graph_models import DependencyGraph

logger = logging.getLogger(__name__)


class AnalysisStore:
    def __init__(self, store_dir: Path):
        self._store_dir = store_dir
        self._store_dir.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._status: dict[str, str] = {}

    def _report_path(self, aid: str) -> Path:
        return self._store_dir / f"{aid}_report.json"

    def _graph_path(self, aid: str) -> Path:
        return self._store_dir / f"{aid}_graph.json"

    async def set_status(self, aid: str, status: str):
        async with self._lock:
            self._status[aid] = status

    async def get_status(self, aid: str) -> str | None:
        async with self._lock:
            return self._status.get(aid)

    async def save_report(self, report: FullAnalysisReport, graph: DependencyGraph | None = None):
        self._report_path(report.analysis_id).write_bytes(
            orjson.dumps(report.model_dump(), option=orjson.OPT_INDENT_2)
        )
        if graph:
            self._graph_path(report.analysis_id).write_bytes(
                orjson.dumps(graph.model_dump(), option=orjson.OPT_INDENT_2)
            )
        await self.set_status(report.analysis_id, report.status)

    async def load_report(self, aid: str) -> FullAnalysisReport | None:
        path = self._report_path(aid)
        if not path.exists():
            return None
        try:
            return FullAnalysisReport.model_validate(orjson.loads(path.read_bytes()))
        except Exception as e:
            logger.error("Load report %s failed: %s", aid, e)
            return None

    async def load_graph(self, aid: str) -> DependencyGraph | None:
        path = self._graph_path(aid)
        if not path.exists():
            return None
        try:
            return DependencyGraph.model_validate(orjson.loads(path.read_bytes()))
        except Exception as e:
            logger.error("Load graph %s failed: %s", aid, e)
            return None

    async def save_error(self, aid: str, url: str, error: str):
        report = FullAnalysisReport(analysis_id=aid, repository_url=url, status="failed", error_message=error)
        await self.save_report(report)