"""
File Analysis Agent — strict anti-hallucination version.
"""

import logging

from app.agents.groq_client import GroqClient
from app.schemas.analysis import FileAnalysisResult, FileMetadata, ParsedStructure
from app.services.token_utils import compress_code, estimate_tokens

logger = logging.getLogger(__name__)

FILE_ANALYSIS_SYSTEM_PROMPT = """\
You are an expert code analysis agent. \
Your job is to analyze a single source code file and produce a structured JSON report.

STRICT RULES — VIOLATIONS WILL CAUSE ERRORS:
1. ONLY describe what is ACTUALLY present in the provided code. \
Do NOT hallucinate or invent functions, classes, imports, or relationships that are not in the code.
2. For each function: describe its purpose, list ONLY the function calls that are \
VISIBLE in the code (not imagined ones), and list imports/modules it actually uses.
3. For each class: list ONLY methods that are actually defined in the code.
4. List ONLY exports that actually exist as module-level names.
5. List ONLY external dependencies that are actually imported in the code (NOT standard library).
6. Write a concise 1-3 sentence summary of the file's ACTUAL purpose.
7. If the file is config/data with no functions or classes, set functions and classes to empty arrays.
8. For the "calls" field: list ONLY direct function/method calls visible in the function body. \
Do NOT add framework callbacks, event handlers, or inferred calls that aren't explicitly written.
9. If the file uses Chrome APIs like chrome.scripting.executeScript, list "chrome" as a dependency \
and list the actual API call (e.g., "chrome.scripting.executeScript") in the calls array.

Respond with ONLY a valid JSON object. No markdown, no extra text, no explanations.\
"""


def _build_file_analysis_prompt(
    file_path: str,
    content: str,
    metadata: FileMetadata,
    parsed: ParsedStructure,
) -> str:
    # Use compressed content for token efficiency
    compressed = compress_code(content, metadata.language)
    tokens = estimate_tokens(compressed)

    # Truncate if still too large
    max_len = 12000  # ~3k tokens
    truncated = compressed[:max_len]
    if len(compressed) > max_len:
        truncated += "\n\n... [TRUNCATED — first 12000 chars shown] ..."

    # Build tree-sitter info as ground truth
    parsed_info = ""
    if parsed.functions:
        parsed_info += "TREE-SITTER VERIFIED FUNCTIONS (these actually exist):\n"
        for f in parsed.functions:
            parsed_info += f"  - {f.name}({', '.join(f.parameters)}) lines {f.start_line}-{f.end_line}\n"
    if parsed.classes:
        parsed_info += "TREE-SITTER VERIFIED CLASSES (these actually exist):\n"
        for c in parsed.classes:
            parsed_info += f"  - {c.name} methods=[{', '.join(c.methods)}] lines {c.start_line}-{c.end_line}\n"
    if parsed.imports:
        parsed_info += "TREE-SITTER VERIFIED IMPORTS (these actually exist):\n"
        for i in parsed.imports:
            parsed_info += f"  - from {i.module} import {', '.join(i.names)}\n"

    validation_note = ""
    if parsed.functions or parsed.classes:
        validation_note = """
IMPORTANT VALIDATION:
- The tree-sitter data above is GROUND TRUTH extracted by a parser.
- Your response MUST be consistent with it.
- Do NOT add functions/classes that tree-sitter didn't find.
- You MAY add details tree-sitter missed (descriptions, calls, etc.) \
but ONLY if they are visible in the code.
"""

    return f"""\
Analyze this file and produce a structured JSON report.

FILE PATH: {file_path}
LANGUAGE: {metadata.language}
SIZE: {metadata.size_bytes} bytes ({tokens} estimated tokens)

{parsed_info}
{validation_note}

--- FILE CONTENT START ---
{truncated}
--- FILE CONTENT END ---

Produce a JSON object with exactly these fields:
- "file_path" (string): must be exactly "{file_path}"
- "summary" (string): 1-3 sentences about what this file ACTUALLY does
- "functions" (array of {{"name": str, "description": str, "calls": [str], "imports_used": [str]}}): \
ONLY functions that exist in the code
- "classes" (array of {{"name": str, "methods": [str]}}): ONLY classes that exist
- "exports" (array of strings): ONLY actual exportable names
- "external_dependencies" (array of strings): ONLY actually imported third-party packages
"""


async def analyze_file(
    groq: GroqClient,
    file_path: str,
    content: str,
    metadata: FileMetadata,
    parsed: ParsedStructure,
) -> FileAnalysisResult:
    """Analyze a single file using Groq LLM with anti-hallucination."""
    prompt = _build_file_analysis_prompt(file_path, content, metadata, parsed)

    try:
        result = await groq.structured_chat(
            prompt=prompt,
            system=FILE_ANALYSIS_SYSTEM_PROMPT,
            response_model=FileAnalysisResult,
            temperature=0.1,  # Lower temperature = less creative = less hallucination
        )
        result.file_path = file_path

        # POST-VALIDATION: Cross-check with tree-sitter data
        result = _validate_against_parsed(result, parsed)

        logger.info("✓ Analyzed: %s", file_path)
        return result

    except Exception as e:
        logger.error("✗ Analysis failed for %s: %s", file_path, e)
        # Fallback: build result from tree-sitter data only
        return _build_fallback_result(file_path, parsed)


def _validate_against_parsed(
    result: FileAnalysisResult,
    parsed: ParsedStructure,
) -> FileAnalysisResult:
    """
    Post-validation: remove hallucinated functions/classes
    that tree-sitter didn't find.
    """
    if not parsed.functions and not parsed.classes:
        # No tree-sitter data to validate against
        return result

    # If tree-sitter found functions, validate LLM's function list
    if parsed.functions:
        ts_func_names = {f.name for f in parsed.functions}
        validated_functions = []
        for func in result.functions:
            if func.name in ts_func_names:
                validated_functions.append(func)
            else:
                # Check if it's a method inside a class (tree-sitter might not catch it at top level)
                ts_methods = set()
                for cls in parsed.classes:
                    ts_methods.update(cls.methods)
                if func.name not in ts_methods:
                    logger.debug("Removed hallucinated function: %s", func.name)
        result.functions = validated_functions

    # Validate classes
    if parsed.classes:
        ts_class_names = {c.name for c in parsed.classes}
        validated_classes = []
        for cls in result.classes:
            if cls.name in ts_class_names:
                # Validate methods too
                ts_cls = next((c for c in parsed.classes if c.name == cls.name), None)
                if ts_cls:
                    valid_methods = [m for m in cls.methods if m in ts_cls.methods]
                    cls.methods = valid_methods if valid_methods else ts_cls.methods
                validated_classes.append(cls)
            else:
                logger.debug("Removed hallucinated class: %s", cls.name)
        result.classes = validated_classes

    return result


def _build_fallback_result(file_path: str, parsed: ParsedStructure) -> FileAnalysisResult:
    """Build a minimal result from tree-sitter data when LLM fails."""
    from app.schemas.analysis import FunctionAnalysis, ClassAnalysis

    functions = [
        FunctionAnalysis(
            name=f.name,
            description=f"Function at lines {f.start_line}-{f.end_line}",
            calls=[],
            imports_used=[],
        )
        for f in parsed.functions
    ]

    classes = [
        ClassAnalysis(name=c.name, methods=c.methods)
        for c in parsed.classes
    ]

    imports = [i.module for i in parsed.imports]

    return FileAnalysisResult(
        file_path=file_path,
        summary=f"File analysis via tree-sitter only (LLM unavailable). "
                f"Contains {len(functions)} functions, {len(classes)} classes.",
        functions=functions,
        classes=classes,
        exports=[f.name for f in parsed.functions] + [c.name for c in parsed.classes],
        external_dependencies=imports,
    )