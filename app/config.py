"""
Centralized configuration using Pydantic Settings.
All values loaded from .env file with sane defaults.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # --- Grok API ---
    xai_api_key: str = ""
    xai_base_url: str = "https://api.x.ai/v1"
    xai_model: str = "grok-3"
    xai_embedding_model: str = "grok-embedding"

    # --- Processing ---
    batch_size: int = 10
    max_concurrent_llm_calls: int = 5
    max_file_size_kb: int = 500
    worker_pool_size: int = 10
    llm_temperature: float = 0.2
    llm_max_tokens: int = 4096
    llm_retry_attempts: int = 3
    llm_retry_delay: float = 2.0

    # --- Storage ---
    clone_base_dir: str = "./repos"
    vector_store_dir: str = "./vector_stores"
    analysis_store_dir: str = "./analysis_data"

    # --- RAG ---
    rag_top_k: int = 10
    rag_chunk_size: int = 1000
    rag_chunk_overlap: int = 200

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    @property
    def clone_path(self) -> Path:
        return Path(self.clone_base_dir)

    @property
    def vector_store_path(self) -> Path:
        return Path(self.vector_store_dir)

    @property
    def analysis_store_path(self) -> Path:
        return Path(self.analysis_store_dir)


settings = Settings()