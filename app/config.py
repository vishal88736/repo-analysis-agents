"""
Centralized configuration — Groq API + processing settings.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # --- Groq API ---
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    groq_fast_model: str = "llama-3.1-8b-instant"
    groq_max_tokens: int = 4096
    groq_temperature: float = 0.2

    # --- Gemini API (optional) ---
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    # --- Processing ---
    batch_size: int = 10
    max_concurrent_llm_calls: int = 5
    max_file_size_kb: int = 500
    llm_retry_attempts: int = 3
    llm_retry_delay: float = 2.0

    # --- Token budgeting ---
    max_file_tokens: int = 1500
    max_prompt_tokens: int = 3000

    # --- Storage ---
    clone_base_dir: str = "./repos"
    vector_store_dir: str = "./vector_stores"
    analysis_store_dir: str = "./analysis_data"
    cache_dir: str = "./cache"

    # --- RAG ---
    rag_top_k: int = 10
    rag_chunk_size: int = 1000
    rag_chunk_overlap: int = 200
    embedding_model: str = "all-MiniLM-L6-v2"

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

    @property
    def cache_path(self) -> Path:
        return Path(self.cache_dir)


settings = Settings()
