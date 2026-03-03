"""
RAG Answer Agent — answers codebase questions using retrieved context via Groq.
"""

import logging
from app.agents.groq_client import GroqClient

logger = logging.getLogger(__name__)

RAG_ANSWER_SYSTEM_PROMPT = """\
You are an expert codebase assistant powered by Groq. You answer questions about \
a software repository using the provided context.

RULES:
1. Answer ONLY based on provided context. If insufficient, say so.
2. Reference specific file paths and function names.
3. Be concise but thorough.
4. Do NOT invent code not in the context.
5. Format response in clear markdown with code blocks where appropriate.\
"""


async def answer_query(
    groq: GroqClient,
    question: str,
    retrieved_chunks: list[dict],
    dependency_context: str = "",
    architecture_summary: str = "",
) -> str:
    """Answer a user query using RAG context."""
    context_parts = []

    if architecture_summary:
        context_parts.append(f"## Architecture Overview\n{architecture_summary}\n")
    if dependency_context:
        context_parts.append(f"## Dependency Context\n{dependency_context}\n")

    context_parts.append("## Relevant Code Sections\n")
    for i, chunk in enumerate(retrieved_chunks, 1):
        meta = chunk.get("metadata", {})
        fp = meta.get("file_path", "unknown")
        ctype = meta.get("type", "code")
        text = chunk.get("text", "")
        context_parts.append(f"### [{i}] {fp} ({ctype})\n{text}\n")

    full_context = "\n".join(context_parts)

    prompt = f"""\
Using the following context from a codebase analysis, answer the user's question.

{full_context}

---
USER QUESTION: {question}
---

Provide a clear, well-structured answer referencing specific files and functions.\
"""

    try:
        return await groq.chat(prompt, RAG_ANSWER_SYSTEM_PROMPT, temperature=0.3, max_tokens=2048)
    except Exception as e:
        logger.error("RAG answer failed: %s", e)
        return f"Failed to generate answer: {e}"