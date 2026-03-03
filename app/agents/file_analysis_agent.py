"""
File Analysis Agent — analyzes a single file via Groq (Llama 3.3 70B).
"""

import logging

from app.agents.groq_client import GroqClient
from app.schemas.analysis import FileAnalysisResult, FileMetadata, ParsedStructure

logger = logging.getLogger(__name__)

FILE_ANALYSIS_SYSTEM_PROMPT = """\
You are an expert code analysis agent running on Groq infrastructure. \
Your job is to analyze a single source code file and produce a structured JSON report.

STRICT RULES:
1. ONLY describe what is ACTUALLY present in the provided code. \
Do NOT hallucinate or invent functions, classes, imports, or relationships.
2. For each function: describe its purpose, list the function calls it makes \
(visible in code), and list imports/modules it uses.
3. For each class: list its methods (only those actually defined).
4. List all exports (module-level names importable by other files).
5. List all external dependencies (third-party packages imported, NOT stdlib).
6. Write a concise 1-3 sentence summary of the file's purpose.
7. If the file is config/data with no functions or classes, still provide a summary.

Respond with ONLY a valid JSON object. No markdown, no extra text.\
"""


def _build_file_analysis_prompt(
    file_path: str,
    content: str,
    metadata: FileMetadata,
    parsed: ParsedStructure,
) -> str:
    # Truncate large files to stay within Groq context limits
    max_len = 10000  # ~3k tokens
    truncated = content[:max_len]
    if len(content) > max_len:
        truncated += "\n\n... [TRUNCATED — first 10000 chars shown] ..."

    parsed_info = ""
    if parsed.functions:
        parsed_info += "TREE-SITTER EXTRACTED FUNCTIONS:\n"
        for f in parsed.functions:
            parsed_info += f"  - {f.name}({', '.join(f.parameters)}) lines {f.start_line}-{f.end_line}\n"
    if parsed.classes:
        parsed_info += "TREE-SITTER EXTRACTED CLASSES:\n"
        for c in parsed.classes:
            parsed_info += f"  - {c.name} methods=[{', '.join(c.methods)}] lines {c.start_line}-{c.end_line}\n"
    if parsed.imports:
        parsed_info += "TREE-SITTER EXTRACTED IMPORTS:\n"
        for i in parsed.imports:
            parsed_info += f"  - from {i.module} import {', '.join(i.names)}\n"

    return f"""\
Analyze this file and produce a structured JSON report.

FILE PATH: {file_path}
LANGUAGE: {metadata.language}
SIZE: {metadata.size_bytes} bytes

{parsed_info}

--- FILE CONTENT START ---
{truncated}
--- FILE CONTENT END ---

Produce a JSON object with exactly these fields:
- "file_path" (string)
- "summary" (string)
- "functions" (array of {{"name": str, "description": str, "calls": [str], "imports_used": [str]}})
- "classes" (array of {{"name": str, "methods": [str]}})
- "exports" (array of strings)
- "external_dependencies" (array of strings)
"""


async def analyze_file(
    groq: GroqClient,
    file_path: str,
    content: str,
    metadata: FileMetadata,
    parsed: ParsedStructure,
) -> FileAnalysisResult:
    """Analyze a single file using Groq LLM."""
    prompt = _build_file_analysis_prompt(file_path, content, metadata, parsed)

    try:
        result = await groq.structured_chat(
            prompt=prompt,
            system=FILE_ANALYSIS_SYSTEM_PROMPT,
            response_model=FileAnalysisResult,
            temperature=0.2,
        )
        result.file_path = file_path  # Ensure path matches
        logger.info("✓ Analyzed: %s", file_path)
        return result

    except Exception as e:
        logger.error("✗ Analysis failed for %s: %s", file_path, e)
        return FileAnalysisResult(
            file_path=file_path,
            summary=f"Analysis failed: {str(e)}",
        )