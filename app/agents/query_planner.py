"""
Feature 11 & 16: Query Planning Agent.

Before answering a query, the planner identifies relevant files, modules,
and search keywords. Uses Gemini when available (large context window) to
process the full repo map + all compact summaries.
"""

import json
import logging

from app.agents.llm_router import LLMRouter
from app.agents.groq_client import TaskType
from app.schemas.analysis import QueryPlan, CompactFileSummary, RepoMap
from app.schemas.graph_models import DependencyGraph

logger = logging.getLogger(__name__)

QUERY_PLAN_SYSTEM = """\
You are a code navigation expert. Given a user question and information about a repository,
identify the most relevant files and keywords needed to answer the question.

Produce a JSON plan that helps narrow down which files to examine.
Be specific — list actual file paths from the repo map, not generic descriptions.

Respond with ONLY valid JSON. No markdown, no extra text.\
"""


async def plan_query(
    router: LLMRouter,
    question: str,
    repo_map: RepoMap | None,
    file_summaries: list[CompactFileSummary],
    graph: DependencyGraph | None = None,
) -> QueryPlan:
    """
    Feature 11 & 16: Plan which files/modules are relevant before retrieval.

    Sends repo map + compact summaries to the LLM (Gemini if available,
    else Groq) to identify relevant areas of the codebase.
    """
    # Build context for the planner
    repo_map_text = ""
    if repo_map and repo_map.files:
        # Provide a compact view of the repo map
        entries = [
            f"  {path}: {info.language} ({info.size} bytes)"
            for path, info in list(repo_map.files.items())[:200]
        ]
        repo_map_text = "REPOSITORY FILES:\n" + "\n".join(entries)

    summaries_text = ""
    if file_summaries:
        summary_items = []
        for s in file_summaries[:100]:
            item = f"  {s.file_path}: {s.purpose}"
            if s.classes:
                item += f" | classes: {', '.join(s.classes[:3])}"
            if s.functions:
                item += f" | functions: {', '.join(s.functions[:3])}"
            summary_items.append(item)
        summaries_text = "FILE SUMMARIES:\n" + "\n".join(summary_items)

    graph_text = ""
    if graph and graph.adjacency_list:
        # Show a subset of the dependency graph
        entries = [
            f"  {src} → {', '.join(targets[:3])}"
            for src, targets in list(graph.adjacency_list.items())[:50]
            if targets
        ]
        if entries:
            graph_text = "DEPENDENCY GRAPH (sample):\n" + "\n".join(entries)

    prompt = f"""\
Given this repository information, identify what is relevant to answer the user question.

USER QUESTION: {question}

{repo_map_text}

{summaries_text}

{graph_text}

Produce a JSON object with:
- "relevant_files": list of specific file paths most likely relevant to the question
- "relevant_modules": list of module/directory names to focus on
- "search_keywords": list of keywords to use for text search
- "graph_entry_points": list of file paths to start graph traversal from
"""

    try:
        return await router.structured_generate(
            task=TaskType.QUERY_PLANNING,
            prompt=prompt,
            system=QUERY_PLAN_SYSTEM,
            response_model=QueryPlan,
            temperature=0.2,
        )
    except Exception as e:
        logger.error("Query planning failed: %s", e)
        # Fallback: return empty plan so retrieval proceeds normally
        return QueryPlan()
