"""
File Analysis Agent — analyzes a single file using Grok API.

Takes file content + tree-sitter structure and produces a structured FileAnalysisResult.
"""

import logging

from app.agents.grok_client import GrokClient
from app.schemas.analysis import FileAnalysisResult, FileMetadata, ParsedStructure

logger = logging.getLogger(__name__)

FILE_ANALYSIS_SYSTEM_PROMPT = """\
You are an expert code analysis agent. Your job is to analyze a single source code file \
and produce a structured JSON report.

STRICT RULES:
1. Only describe what is ACTUALLY in the provided code. Do NOT hallucinate or invent \
functions, classes, imports, or relationships that are not present.
2. For each function: describe what it does, list function calls it makes (that are \
visible in the code), and list imports/modules it uses.
3. For each class: list its methods.
4. List all exports (module-level names that could be imported by others).
5. List all external dependencies (third-party packages imported).
6. Write a concise summary of the file's purpose.
7. If the file is a config/data file with no functions or classes, still provide a summary \
and list dependencies if any.

Respond with ONLY a valid JSON object. No markdown, no explanation outside JSON.\
"""


def _build_file_analysis_prompt(
    file_path: str,
    content: str,
    metadata: FileMetadata,
    parsed: ParsedStructure,
) -> str:
    """Build the user prompt for the file analysis agent."""
    # Truncate very large files to avoid token overflow
    max_content_len = 12000
    truncated = content[:max_content_len]
    if len(content) > max_content_len:
        truncated += "\n\n... [FILE TRUNCATED — showing first 12000 chars] ..."

    parsed_info = ""
    if parsed.functions:
        parsed_info += "TREE-SITTER EXTRACTED FUNCTIONS:\n"
        for f in parsed.functions:
            parsed_info += f"  - {f.name}(params: {', '.join(f.parameters)}) lines {f.start_line}-{f.end_line}\n"
    if parsed.classes:
        parsed_info += "TREE-SITTER EXTRACTED CLASSES:\n"
        for c in parsed.classes:
            parsed_info += f"  - {c.name} methods: {', '.join(c.methods)} lines {c.start_line}-{c.end_line}\n"
    if parsed.imports:
        parsed_info += "TREE-SITTER EXTRACTED IMPORTS:\n"
        for i in parsed.imports:
            parsed_info += f"  - {i.module} names: {', '.join(i.names)}\n"

    return f"""\
Analyze this file and produce a structured JSON report.

FILE PATH: {file_path}
LANGUAGE: {metadata.language}
SIZE: {metadata.size_bytes} bytes

{parsed_info}

--- FILE CONTENT START ---
{truncated}
--- FILE CONTENT END ---

Produce a JSON object with these exact fields:
- file_path (string)
- summary (string)
- functions (array of objects with: name, description, calls, imports_used)
- classes (array of objects with: name, methods)
- exports (array of strings)
- external_dependencies (array of strings)
"""


async def analyze_file(
    grok: GrokClient,
    file_path: str,
    content: str,
    metadata: FileMetadata,
    parsed: ParsedStructure,
) -> FileAnalysisResult:
    """
    Analyze a single file using the Grok LLM and return structured output.
    """
    prompt = _build_file_analysis_prompt(file_path, content, metadata, parsed)

    try:
        result = await grok.structured_chat(
            prompt=prompt,
            system=FILE_ANALYSIS_SYSTEM_PROMPT,
            response_model=FileAnalysisResult,
            temperature=0.2,
        )
        # Ensure file_path matches actual path
        result.file_path = file_path
        logger.info("File analysis completed: %s", file_path)
        return result

    except Exception as e:
        logger.error("File analysis failed for %s: %s", file_path, e)
        # Return a minimal result on failure so we don't lose the file
        return FileAnalysisResult(
            file_path=file_path,
            summary=f"Analysis failed: {str(e)}",
        )