"""
RAG vector store using ChromaDB with local sentence-transformer embeddings.

Groq doesn't provide a general-purpose embedding endpoint like OpenAI,
so we use sentence-transformers (all-MiniLM-L6-v2) locally for embeddings.
This is fast, free, and works offline.

Feature 9: Uses SmartChunker for code-aware chunking instead of naive text splitting.
"""

import logging
import hashlib
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from app.config import settings
from app.schemas.analysis import FileAnalysisResult, ArchitectureSummary
from app.services.smart_chunker import SmartChunker
from app.schemas.analysis import ParsedStructure

logger = logging.getLogger(__name__)

_smart_chunker = SmartChunker()


class VectorStore:
    def __init__(self, persist_dir: Path | None = None):
        self._persist_dir = persist_dir or settings.vector_store_path
        self._client = chromadb.PersistentClient(
            path=str(self._persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        # Local embedding function — no API calls, runs on CPU
        self._embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=settings.embedding_model,
        )

    def _collection_name(self, analysis_id: str) -> str:
        safe = analysis_id.replace("-", "_")[:60]
        return f"a_{safe}"

    def _get_or_create_collection(self, analysis_id: str):
        return self._client.get_or_create_collection(
            name=self._collection_name(analysis_id),
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def _chunk_text(self, text: str, max_size: int | None = None) -> list[str]:
        """Naive text chunking (fallback for non-code content)."""
        max_size = max_size or settings.rag_chunk_size
        overlap = settings.rag_chunk_overlap
        if len(text) <= max_size:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = start + max_size
            chunks.append(text[start:end])
            start = end - overlap
        return chunks

    def _make_id(self, *parts: str) -> str:
        return hashlib.sha256("::".join(parts).encode()).hexdigest()[:16]

    async def index_analysis(
        self,
        analysis_id: str,
        file_analyses: list[FileAnalysisResult],
        architecture_summary: ArchitectureSummary,
        parsed_structures: dict[str, ParsedStructure] | None = None,
        raw_contents: dict[str, str] | None = None,
    ) -> int:
        """
        Index analysis results into ChromaDB.

        Feature 9: Uses SmartChunker for code-aware chunking when
        parsed structures and raw contents are provided.
        """
        collection = self._get_or_create_collection(analysis_id)

        documents, metadatas, ids = [], [], []

        # Architecture summary
        if architecture_summary.overview:
            for i, chunk in enumerate(self._chunk_text(architecture_summary.overview)):
                ids.append(self._make_id("arch", str(i)))
                documents.append(chunk)
                metadatas.append({"type": "architecture_summary", "file_path": "__global__", "chunk_index": i})

        # File summaries + functions + classes
        for fa in file_analyses:
            # Feature 9: Smart chunking for raw code when available
            if raw_contents and fa.file_path in raw_contents:
                parsed = (parsed_structures or {}).get(fa.file_path, ParsedStructure())
                raw = raw_contents[fa.file_path]
                smart_chunks = _smart_chunker.chunk_file(
                    content=raw,
                    parsed=parsed,
                    file_path=fa.file_path,
                    # rag_chunk_size is in chars; divide by 4 to convert to token estimate (~4 chars/token)
                    max_chunk_tokens=settings.rag_chunk_size // 4,
                )
                for i, chunk in enumerate(smart_chunks):
                    chunk_text = f"File: {fa.file_path}\n{chunk['text']}"
                    ids.append(self._make_id("smart", fa.file_path, str(i)))
                    documents.append(chunk_text)
                    metadatas.append({
                        "type": chunk.get("type", "module"),
                        "file_path": fa.file_path,
                        "chunk_index": i,
                        "chunk_name": chunk.get("name", ""),
                    })
            else:
                # Fallback: naive summary chunking
                if fa.summary:
                    for i, chunk in enumerate(self._chunk_text(fa.summary)):
                        ids.append(self._make_id("file", fa.file_path, str(i)))
                        documents.append(f"File: {fa.file_path}\n{chunk}")
                        metadatas.append({"type": "file_summary", "file_path": fa.file_path, "chunk_index": i})

            for func in fa.functions:
                text = f"Function `{func.name}` in {fa.file_path}: {func.description}"
                if func.calls:
                    text += f"\nCalls: {', '.join(func.calls)}"
                if func.imports_used:
                    text += f"\nUses: {', '.join(func.imports_used)}"
                ids.append(self._make_id("func", fa.file_path, func.name))
                documents.append(text)
                metadatas.append({"type": "function", "file_path": fa.file_path, "function_name": func.name})

            for cls in fa.classes:
                text = f"Class `{cls.name}` in {fa.file_path}. Methods: {', '.join(cls.methods)}"
                ids.append(self._make_id("cls", fa.file_path, cls.name))
                documents.append(text)
                metadatas.append({"type": "class", "file_path": fa.file_path, "class_name": cls.name})

        # Batch upsert
        if documents:
            batch_size = 500
            for i in range(0, len(documents), batch_size):
                end = min(i + batch_size, len(documents))
                collection.upsert(ids=ids[i:end], documents=documents[i:end], metadatas=metadatas[i:end])

        logger.info("Indexed %d documents for analysis %s", len(documents), analysis_id)
        return len(documents)

    async def search(
        self,
        analysis_id: str,
        query: str,
        top_k: int | None = None,
        filter_type: str | None = None,
    ) -> list[dict]:
        top_k = top_k or settings.rag_top_k
        collection = self._get_or_create_collection(analysis_id)

        where = {"type": filter_type} if filter_type else None

        try:
            results = collection.query(query_texts=[query], n_results=top_k, where=where)
        except Exception as e:
            logger.error("Vector search failed: %s", e)
            return []

        chunks = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                dist = results["distances"][0][i] if results["distances"] else 0.0
                chunks.append({"text": doc, "metadata": meta, "distance": dist})
        return chunks

    def delete_collection(self, analysis_id: str):
        try:
            self._client.delete_collection(self._collection_name(analysis_id))
        except Exception:
            pass
