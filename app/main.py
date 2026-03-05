"""FastAPI application entry point."""

import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.config import settings
from app.core.logging_config import setup_logging
from app.api.routes import router
from app.services.analysis_store import AnalysisStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(settings.log_level)
    settings.clone_path.mkdir(parents=True, exist_ok=True)
    settings.vector_store_path.mkdir(parents=True, exist_ok=True)
    settings.analysis_store_path.mkdir(parents=True, exist_ok=True)
    settings.cache_path.mkdir(parents=True, exist_ok=True)
    app.state.analysis_store = AnalysisStore(settings.analysis_store_path)
    yield


app = FastAPI(
    title="Multi-Agent Repo Analyzer (Groq + Gemini)",
    description=(
        "Analyzes GitHub repositories using Groq + Gemini with "
        "token optimization, caching, hybrid RAG, progressive loading, "
        "and multi-LLM routing."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.include_router(router, prefix="/api/v1")


if __name__ == "__main__":
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)