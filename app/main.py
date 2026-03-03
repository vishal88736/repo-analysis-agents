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
    app.state.analysis_store = AnalysisStore(settings.analysis_store_path)
    yield


app = FastAPI(
    title="Multi-Agent Repo Analyzer (Groq + Open-Source LLMs)",
    description=(
        "Analyzes GitHub repositories using Groq-hosted open-source models "
        "(Llama 3.3, Mixtral, Gemma) with batch agent processing, "
        "tree-sitter parsing, dependency graphs, and RAG."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router, prefix="/api/v1")


if __name__ == "__main__":
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)