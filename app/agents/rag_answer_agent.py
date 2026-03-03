"""
RAG Answer Agent — answers user questions about the codebase using
retrieved context from the vector store + dependency graph.
"""

import logging

from app.agents.grok_client import GrokClient

logger = logging.getLogger(__name__)

RAG_ANSWER_SYSTEM_PROMPT = """\
You are an expert codebase assistant. You answer questions about a software repository \
using the provided context.

RULES:
1. Answer ONLY based on the provided context. If the context doesn't contain enough \
information, say so clearly.
2. Reference specific file paths and function names when relevant.
3. Be concise but thorough.
4. If asked about architecture, reference the global summary and dependency information.
5. Do NOT make up code that isn't in the context.
6. Format your response in clear markdown with code blocks where appropriate.\
"""


async def answer_query(
    grok: GrokClient,
    question: str,
    retrieved_chunks: list[dict],
    dependency_context: str = "",
    architecture_summary: str = "",
) -> str:
    """
    Answer a user query using RAG context.

    Args:
        grok: Grok client instance.
        question: User's natural language question.
        retrieved_chunks: List of dicts with 'text' and 'metadata' from vector search.
        dependency_context: Formatted dependency graph context.
        architecture_summary: Global architecture overview.

    Returns:
        LLM-generated answer string.
    """
    # Build context from retrieved chunks
    context_parts = []

    if architecture_summary:
        context_parts.append(f"## Architecture Overview\n{architecture_summary}\n")

    if dependency_context:
        context_parts.append(f"## Dependency Context\n{dependency_context}\n")

    context_parts.append("## Relevant Code Sections\n")
    for i, chunk in enumerate(retrieved_chunks, 1):
        metadata = chunk.get("metadata", {})
        file_path = metadata.get("file_path", "unknown")
        chunk_type = metadata.get("type", "code")
        text = chunk.get("text", "")
        context_parts.append(
            f"### [{i}] {file_path} ({chunk_type})\n{text}\n"
        )

    full_context = "\n".join(context_parts)

    prompt = f"""\
Using the following context from a codebase analysis, answer the user's question.

{full_context}

---
USER QUESTION: {question}
---

Provide a clear, well-structured answer. Reference specific files and functions where applicable.\
"""

    try:
        answer = await grok.chat(
            prompt=prompt,
            system=RAG_ANSWER_SYSTEM_PROMPT,
            temperature=0.3,
            max_tokens=2048,
        )
        return answer
    except Exception as e:
        logger.error("RAG answer generation failed: %s", e)
        return f"Failed to generate answer: {e}"