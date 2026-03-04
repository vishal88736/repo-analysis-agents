"""
File Analysis Agent — analyzes a single file via Groq (Llama 3.1 8B).

Feature 2: Produces compact structured summaries (CompactFileSummary).
Feature 10: Uses ContextCompressor to reduce token usage before sending to LLM.
Feature 7: Uses FILE_ANALYSIS task type for model routing.
"""

import logging

from app.config import settings

from app.agents.groq_client import GroqClient, TaskType
from app.schemas.analysis import (
    FileAnalysisResult, FileMetadata, ParsedStructure, CompactFileSummary,
)
from app.services.compressor import ContextCompressor

logger = logging.getLogger(__name__)

_compressor = ContextCompressor()

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

COMPACT_SUMMARY_SYSTEM_PROMPT = """\
You are a code analysis agent. Analyze this source file and produce a compact structured summary.

STRICT RULES:
1. Be concise — purpose is 1 sentence, lists are short.
2. Only include items actually present in the code.
3. Respond with ONLY valid JSON. No markdown, no extra text.\
"""


def _build_file_analysis_prompt(
    file_path: str,
    content: str,
    metadata: FileMetadata,
    parsed: ParsedStructure,
) -> str:
    # Feature 10: Compress code before sending to LLM
    compressed = _compressor.compress_for_prompt(content, metadata.language, max_tokens=settings.max_file_tokens)

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
{compressed}
--- FILE CONTENT END ---

Produce a JSON object with exactly these fields:
- "file_path" (string)
- "summary" (string)
- "functions" (array of {{"name": str, "description": str, "calls": [str], "imports_used": [str]}})
- "classes" (array of {{"name": str, "methods": [str]}})
- "exports" (array of strings)
- "external_dependencies" (array of strings)
"""


def _build_compact_summary_prompt(
    file_path: str,
    content: str,
    metadata: FileMetadata,
    parsed: ParsedStructure,
) -> str:
    # Compress content for compact summary generation
    compressed = _compressor.compress_for_prompt(content, metadata.language, max_tokens=settings.max_file_tokens // 2)

    function_names = [f.name for f in parsed.functions[:20]]
    class_names = [c.name for c in parsed.classes[:10]]
    import_modules = [i.module for i in parsed.imports[:15]]

    return f"""\
Produce a compact JSON summary of this file.

FILE PATH: {file_path}
LANGUAGE: {metadata.language}
PARSED FUNCTIONS: {function_names}
PARSED CLASSES: {class_names}
PARSED IMPORTS: {import_modules}

--- CODE EXCERPT ---
{compressed}
--- END ---

Produce JSON with these exact fields:
- "file_path" (string — the file path)
- "purpose" (string — one sentence describing what this file does)
- "functions" (array of strings — function names)
- "classes" (array of strings — class names)
- "imports" (array of strings — imported module names)
- "key_dependencies" (array of strings — external packages used)
- "entry_points" (array of strings — function names that are entry points, e.g., main, app startup)
"""


async def analyze_file(
    groq: GroqClient,
    file_path: str,
    content: str,
    metadata: FileMetadata,
    parsed: ParsedStructure,
) -> FileAnalysisResult:
    """Analyze a single file using Groq LLM. Feature 7: uses FILE_ANALYSIS task type."""
    prompt = _build_file_analysis_prompt(file_path, content, metadata, parsed)

    try:
        result = await groq.structured_chat(
            prompt=prompt,
            system=FILE_ANALYSIS_SYSTEM_PROMPT,
            response_model=FileAnalysisResult,
            temperature=0.2,
            task=TaskType.FILE_ANALYSIS,
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


async def generate_compact_summary(
    groq: GroqClient,
    file_path: str,
    content: str,
    metadata: FileMetadata,
    parsed: ParsedStructure,
) -> CompactFileSummary:
    """Feature 2: Generate a compact structured summary for a file."""
    prompt = _build_compact_summary_prompt(file_path, content, metadata, parsed)

    try:
        result = await groq.structured_chat(
            prompt=prompt,
            system=COMPACT_SUMMARY_SYSTEM_PROMPT,
            response_model=CompactFileSummary,
            temperature=0.2,
            task=TaskType.FILE_ANALYSIS,
        )
        result.file_path = file_path
        logger.debug("✓ Compact summary: %s", file_path)
        return result
    except Exception as e:
        logger.warning("Compact summary failed for %s: %s", file_path, e)
        # Fallback: build from parsed structure
        return CompactFileSummary(
            file_path=file_path,
            purpose=f"File analysis failed: {str(e)}",
            functions=[f.name for f in parsed.functions],
            classes=[c.name for c in parsed.classes],
            imports=[i.module for i in parsed.imports],
        )
