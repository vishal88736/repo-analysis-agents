"""
Feature 4: Hybrid Retrieval — combines vector similarity, keyword search,
and dependency graph traversal for more relevant RAG context.

Scoring formula: vector_score * 0.5 + keyword_score * 0.3 + graph_score * 0.2
"""

import logging
import re
from collections import defaultdict

from app.rag.vector_store import VectorStore
from app.schemas.graph_models import DependencyGraph

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Combines three retrieval strategies and merges results with weighted scoring.
    """

    def __init__(self, vector_store: VectorStore):
        self._vector_store = vector_store

    async def retrieve(
        self,
        analysis_id: str,
        query: str,
        top_k: int = 10,
        graph: DependencyGraph | None = None,
        file_summaries: list | None = None,
    ) -> list[dict]:
        """
        Retrieve relevant chunks using hybrid scoring.

        Args:
            analysis_id: ID of the analysis to search
            query: User query
            top_k: Number of final results to return
            graph: Optional dependency graph for graph traversal
            file_summaries: Optional compact file summaries for keyword search
        """
        # --- Step 1: Vector similarity search ---
        vector_results = await self._vector_store.search(
            analysis_id=analysis_id,
            query=query,
            top_k=top_k * 2,  # Fetch more to merge
        )

        # Build a scored dict: file_path → list of chunks with scores
        scored: dict[str, list[dict]] = defaultdict(list)

        for chunk in vector_results:
            meta = chunk.get("metadata", {})
            file_path = meta.get("file_path", "__global__")
            # ChromaDB returns cosine distance (0=identical, 2=opposite)
            # Convert to similarity score [0, 1]
            distance = float(chunk.get("distance", 1.0))
            vector_score = max(0.0, 1.0 - distance / 2.0)
            chunk["_vector_score"] = vector_score
            chunk["_keyword_score"] = 0.0
            chunk["_graph_score"] = 0.0
            scored[file_path].append(chunk)

        # --- Step 2: Keyword search over file names and summaries ---
        keywords = self._extract_keywords(query)
        if keywords and file_summaries:
            for summary in file_summaries:
                fp = summary.file_path
                text_to_search = " ".join([
                    fp,
                    summary.purpose,
                    " ".join(summary.functions),
                    " ".join(summary.classes),
                    " ".join(summary.imports),
                ])
                kw_score = self._keyword_score(text_to_search, keywords)
                if kw_score > 0 and fp not in scored:
                    # Add a synthetic chunk for keyword-matched files with no vector hit
                    scored[fp].append({
                        "text": f"File: {fp}\nPurpose: {summary.purpose}",
                        "metadata": {"file_path": fp, "type": "file_summary"},
                        "distance": 1.0,
                        "_vector_score": 0.0,
                        "_keyword_score": kw_score,
                        "_graph_score": 0.0,
                    })
                elif kw_score > 0:
                    for chunk in scored[fp]:
                        chunk["_keyword_score"] = max(chunk["_keyword_score"], kw_score)

        # --- Step 3: Dependency graph traversal ---
        if graph and graph.adjacency_list:
            graph_scores = self._graph_score_files(
                set(scored.keys()) - {"__global__"},
                graph,
            )
            for fp, g_score in graph_scores.items():
                for chunk in scored.get(fp, []):
                    chunk["_graph_score"] = g_score

        # --- Step 4: Merge and rank by composite score ---
        all_chunks: list[dict] = []
        for fp, chunks in scored.items():
            # Take best chunk per file (max composite score)
            for chunk in chunks:
                composite = (
                    chunk["_vector_score"] * 0.5
                    + chunk["_keyword_score"] * 0.3
                    + chunk["_graph_score"] * 0.2
                )
                chunk["_composite_score"] = composite
            all_chunks.extend(chunks)

        # Sort by composite score descending
        all_chunks.sort(key=lambda c: c["_composite_score"], reverse=True)

        # Deduplicate: one chunk per file (best scoring)
        seen_files: set[str] = set()
        final: list[dict] = []
        for chunk in all_chunks:
            fp = chunk.get("metadata", {}).get("file_path", "__global__")
            if fp not in seen_files:
                seen_files.add(fp)
                final.append(chunk)
            if len(final) >= top_k:
                break

        logger.info(
            "Hybrid retrieval: query=%r | vector=%d | final=%d",
            query[:50], len(vector_results), len(final),
        )
        return final

    def _extract_keywords(self, query: str) -> list[str]:
        """Extract meaningful keywords from the query (simple tokenization)."""
        # Remove punctuation, lowercase, split, filter short words
        words = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", query.lower())
        stopwords = {"the", "a", "an", "is", "in", "of", "to", "and", "or",
                     "for", "with", "how", "what", "where", "does", "do"}
        return [w for w in words if len(w) > 2 and w not in stopwords]

    def _keyword_score(self, text: str, keywords: list[str]) -> float:
        """Score a text based on keyword presence (0.0 - 1.0)."""
        if not keywords:
            return 0.0
        text_lower = text.lower()
        hits = sum(1 for kw in keywords if kw in text_lower)
        return hits / len(keywords)

    def _graph_score_files(
        self, file_paths: set[str], graph: DependencyGraph, depth: int = 2
    ) -> dict[str, float]:
        """
        Score files based on their connectivity in the dependency graph.

        Files connected to already-relevant files get a boost.
        """
        scores: dict[str, float] = {fp: 0.0 for fp in file_paths}
        # BFS from each relevant file up to `depth` hops
        for start_fp in file_paths:
            visited: dict[str, int] = {}  # node → distance
            queue = [(start_fp, 0)]
            while queue:
                node, dist = queue.pop(0)
                if dist > depth or node in visited:
                    continue
                visited[node] = dist
                for neighbor in graph.adjacency_list.get(node, []):
                    if neighbor not in visited:
                        queue.append((neighbor, dist + 1))
                        # Connected files that are already in our set get a boost
                        if neighbor in file_paths:
                            # Higher score for closer connections
                            boost = 1.0 / (dist + 1)
                            scores[neighbor] = min(1.0, scores.get(neighbor, 0.0) + boost)

        return scores
