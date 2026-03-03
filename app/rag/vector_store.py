"""
RAG vector store using ChromaDB.

Embeds file summaries, function descriptions, and architecture summary.
Supports top-k similarity search with metadata filtering.
"""

import logging
import hashlib
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import settings
from app.schemas.analysis import FileAnalysisResult, ArchitectureSummary

logger = logging.getLogger(__name__)


class VectorStore:
    """
    ChromaDB-backed vector store for codebase RAG.

    Each analysis gets its own collection keyed by analysis_id.
    Uses ChromaDB's built-in embedding function (default: all-MiniLM-L6-v2)
    for fast local embeddings. For production, swap to Grok embeddings.
    """

    def __init__(self, persist_dir: Path | None = None):
        self._persist_dir = persist_dir or settings.vector_store_path
        self._client = chromadb.PersistentClient(
            path=str(self._persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

    def _collection_name(self, analysis_id: str) -> str:
        """Generate a valid collection name from analysis_id."""
        # ChromaDB collection names: 3-63 chars, alphanumeric + underscores/hyphens
        safe = analysis_id.replace("-", "_")[:60]
        return f"a_{safe}"

    def _get_or_create_collection(self, analysis_id: str):
        """Get or create a ChromaDB collection for an analysis."""
        name = self._collection_name(analysis_id)
        return self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    def _chunk_text(self, text: str, max_size: int | None = None) -> list[str]:
        """Split text into chunks with overlap."""
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
        """Create a deterministic document ID."""
        combined = "::".join(parts)
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    async def index_analysis(
        self,
        analysis_id: str,
        file_analyses: list[FileAnalysisResult],
        architecture_summary: ArchitectureSummary,
    ) -> int:
        """
        Index all analysis data into the vector store.

        Returns the total number of documents indexed.
        """
        collection = self._get_or_create_collection(analysis_id)

        documents: list[str] = []
        metadatas: list[dict] = []
        ids: list[str] = []

        # Index architecture summary
        if architecture_summary.overview:
            for i, chunk in enumerate(self._chunk_text(architecture_summary.overview)):
                doc_id = self._make_id("arch", str(i))
                documents.append(chunk)
                metadatas.append({
                    "type": "architecture_summary",
                    "file_path": "__global__",
                    "chunk_index": i,
                })
                ids.append(doc_id)

        # Index file summaries and function descriptions
        for fa in file_analyses:
            # File summary
            if fa.summary:
                for i, chunk in enumerate(self._chunk_text(fa.summary)):
                    doc_id = self._make_id("file", fa.file_path, str(i))
                    documents.append(f"File: {fa.file_path}\n{chunk}")
                    metadatas.append({
                        "type": "file_summary",
                        "file_path": fa.file_path,
                        "chunk_index": i,
                    })
                    ids.append(doc_id)

            # Function descriptions
            for func in fa.functions:
                text = (
                    f"Function `{func.name}` in {fa.file_path}: "
                    f"{func.description}"
                )
                if func.calls:
                    text += f"\nCalls: {', '.join(func.calls)}"
                if func.imports_used:
                    text += f"\nUses: {', '.join(func.imports_used)}"

                doc_id = self._make_id("func", fa.file_path, func.name)
                documents.append(text)
                metadatas.append({
                    "type": "function",
                    "file_path": fa.file_path,
                    "function_name": func.name,
                })
                ids.append(doc_id)

            # Class descriptions
            for cls in fa.classes:
                text = (
                    f"Class `{cls.name}` in {fa.file_path}. "
                    f"Methods: {', '.join(cls.methods)}"
                )
                doc_id = self._make_id("cls", fa.file_path, cls.name)
                documents.append(text)
                metadatas.append({
                    "type": "class",
                    "file_path": fa.file_path,
                    "class_name": cls.name,
                })
                ids.append(doc_id)

        # Batch upsert to ChromaDB
        if documents:
            batch_size = 500
            for i in range(0, len(documents), batch_size):
                batch_end = min(i + batch_size, len(documents))
                collection.upsert(
                    ids=ids[i:batch_end],
                    documents=documents[i:batch_end],
                    metadatas=metadatas[i:batch_end],
                )

        total = len(documents)
        logger.info("Indexed %d documents for analysis %s", total, analysis_id)
        return total

    async def search(
        self,
        analysis_id: str,
        query: str,
        top_k: int | None = None,
        filter_type: str | None = None,
    ) -> list[dict]:
        """
        Search the vector store for relevant chunks.

        Returns list of dicts with 'text', 'metadata', 'distance'.
        """
        top_k = top_k or settings.rag_top_k
        collection = self._get_or_create_collection(analysis_id)

        where_filter = None
        if filter_type:
            where_filter = {"type": filter_type}

        try:
            results = collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where_filter,
            )
        except Exception as e:
            logger.error("Vector search failed: %s", e)
            return []

        chunks = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 0.0
                chunks.append({
                    "text": doc,
                    "metadata": meta,
                    "distance": distance,
                })

        return chunks

    def delete_collection(self, analysis_id: str):
        """Delete the vector store collection for an analysis."""
        name = self._collection_name(analysis_id)
        try:
            self._client.delete_collection(name)
            logger.info("Deleted vector collection: %s", name)
        except Exception:
            pass