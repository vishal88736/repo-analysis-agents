"""
File Analysis Agent — detects inter-file dependencies and interactions.
"""

import re
import logging

from app.agents.groq_client import GroqClient
from app.schemas.analysis import (
    FileAnalysisResult, FileMetadata, ParsedStructure,
    FileInteraction, FunctionAnalysis, ClassAnalysis,
)
from app.services.token_utils import compress_code, estimate_tokens

logger = logging.getLogger(__name__)

FILE_ANALYSIS_SYSTEM_PROMPT = """\
You are an expert code analysis agent. \
Your job is to analyze a single source code file and produce a structured JSON report.

STRICT RULES:
1. ONLY describe what is ACTUALLY present in the provided code. \
Do NOT hallucinate or invent functions, classes, imports, or relationships.
2. For each function: describe its purpose, list ONLY function calls VISIBLE in code, \
and list imports/modules it uses.
3. For each class: list ONLY methods actually defined.
4. List ONLY exports that actually exist as module-level names.
5. List ONLY external dependencies actually imported (NOT standard library).
6. Write a concise 1-3 sentence summary of the file's ACTUAL purpose.
7. For "internal_file_references": list OTHER PROJECT FILES referenced in this code. \
Look for: string literals containing filenames (e.g., "content.js", "./utils"), \
dynamic imports, script injection calls (executeScript), require(), import statements \
pointing to local files. Only list filenames, not external packages.
8. For "file_interactions": describe HOW this file uses each referenced file. \
For example: "background.js injects content.js via chrome.scripting.executeScript" \
or "app.py imports routes from routes.py". Each interaction must have: \
source_file (this file), target_file, interaction_type (one of: injects, imports, \
loads, calls_into, configures, extends, includes), description.

Respond with ONLY a valid JSON object. No markdown, no extra text.\
"""


def _extract_file_references(content: str, all_project_files: list[str]) -> list[str]:
    """
    Static extraction: find references to other project files in code.
    Catches string literals, imports, require(), executeScript, etc.
    """
    refs = set()

    # All project file basenames and paths
    file_basenames = {}
    for f in all_project_files:
        basename = f.split("/")[-1]
        file_basenames[basename] = f
        file_basenames[f] = f

    # Pattern 1: String literals containing filenames — "content.js", './utils.py', etc.
    string_patterns = re.findall(r"""['"]([^'"]*?\.(?:js|ts|py|go|rs|java|jsx|tsx|css|html|json))['"]\s*""", content)
    for match in string_patterns:
        clean = match.lstrip("./")
        if clean in file_basenames:
            refs.add(file_basenames[clean])
        # Also check basename
        basename = clean.split("/")[-1]
        if basename in file_basenames:
            refs.add(file_basenames[basename])

    # Pattern 2: import/require statements with local paths
    import_patterns = re.findall(r"""(?:import|require|from)\s*\(?['"](\.[^'"]+)['"]\)?""", content)
    for match in import_patterns:
        clean = match.lstrip("./")
        for ext in ("", ".js", ".ts", ".py", ".jsx", ".tsx"):
            candidate = clean + ext
            if candidate in file_basenames:
                refs.add(file_basenames[candidate])

    # Pattern 3: Chrome extension specific — executeScript with files
    exec_patterns = re.findall(r"""files\s*:\s*\[(.*?)\]""", content, re.DOTALL)
    for match in exec_patterns:
        file_refs = re.findall(r"""['"]([^'"]+)['"]""", match)
        for fref in file_refs:
            clean = fref.lstrip("./")
            if clean in file_basenames:
                refs.add(file_basenames[clean])
            basename = clean.split("/")[-1]
            if basename in file_basenames:
                refs.add(file_basenames[basename])

    # Pattern 4: HTML script/link tags
    tag_patterns = re.findall(r"""(?:src|href)\s*=\s*['"]([^'"]+)['"]""", content)
    for match in tag_patterns:
        clean = match.lstrip("./")
        if clean in file_basenames:
            refs.add(file_basenames[clean])

    # Pattern 5: manifest.json specific — background scripts, content_scripts, etc.
    manifest_patterns = re.findall(r"""['"]([^'"]+\.js)['"]""", content)
    for match in manifest_patterns:
        clean = match.lstrip("./")
        if clean in file_basenames:
            refs.add(file_basenames[clean])

    return sorted(refs)


def _detect_interactions(
    file_path: str,
    content: str,
    internal_refs: list[str],
    language: str,
) -> list[FileInteraction]:
    """Detect HOW this file interacts with referenced files."""
    interactions = []

    for ref in internal_refs:
        ref_basename = ref.split("/")[-1]

        # Chrome extension: executeScript injection
        if "executeScript" in content and ref_basename in content:
            interactions.append(FileInteraction(
                source_file=file_path,
                target_file=ref,
                interaction_type="injects",
                description=f"{file_path} injects {ref} via chrome.scripting.executeScript",
            ))
        # manifest.json: configures/declares files
        elif file_path.endswith("manifest.json"):
            if "background" in content and ref_basename in content:
                interactions.append(FileInteraction(
                    source_file=file_path,
                    target_file=ref,
                    interaction_type="configures",
                    description=f"manifest.json declares {ref} as a background/content script",
                ))
            else:
                interactions.append(FileInteraction(
                    source_file=file_path,
                    target_file=ref,
                    interaction_type="configures",
                    description=f"manifest.json references {ref}",
                ))
        # Python imports
        elif language == "Python" and ("import " in content or "from " in content):
            interactions.append(FileInteraction(
                source_file=file_path,
                target_file=ref,
                interaction_type="imports",
                description=f"{file_path} imports from {ref}",
            ))
        # JS/TS imports or require
        elif language in ("JavaScript", "TypeScript"):
            if f"require(" in content or f"import " in content:
                interactions.append(FileInteraction(
                    source_file=file_path,
                    target_file=ref,
                    interaction_type="imports",
                    description=f"{file_path} imports/requires {ref}",
                ))
            elif "files:" in content or "src=" in content:
                interactions.append(FileInteraction(
                    source_file=file_path,
                    target_file=ref,
                    interaction_type="loads",
                    description=f"{file_path} loads {ref}",
                ))
            else:
                interactions.append(FileInteraction(
                    source_file=file_path,
                    target_file=ref,
                    interaction_type="calls_into",
                    description=f"{file_path} references {ref}",
                ))
        else:
            interactions.append(FileInteraction(
                source_file=file_path,
                target_file=ref,
                interaction_type="calls_into",
                description=f"{file_path} references {ref}",
            ))

    return interactions


def _build_file_analysis_prompt(
    file_path: str,
    content: str,
    metadata: FileMetadata,
    parsed: ParsedStructure,
    all_project_files: list[str],
    static_refs: list[str],
    static_interactions: list[FileInteraction],
) -> str:
    compressed = compress_code(content, metadata.language)
    tokens = estimate_tokens(compressed)

    max_len = 12000
    truncated = compressed[:max_len]
    if len(compressed) > max_len:
        truncated += "\n\n... [TRUNCATED — first 12000 chars shown] ..."

    parsed_info = ""
    if parsed.functions:
        parsed_info += "TREE-SITTER VERIFIED FUNCTIONS:\n"
        for f in parsed.functions:
            parsed_info += f"  - {f.name}({', '.join(f.parameters)}) lines {f.start_line}-{f.end_line}\n"
    if parsed.classes:
        parsed_info += "TREE-SITTER VERIFIED CLASSES:\n"
        for c in parsed.classes:
            parsed_info += f"  - {c.name} methods=[{', '.join(c.methods)}] lines {c.start_line}-{c.end_line}\n"
    if parsed.imports:
        parsed_info += "TREE-SITTER VERIFIED IMPORTS:\n"
        for i in parsed.imports:
            parsed_info += f"  - from {i.module} import {', '.join(i.names)}\n"

    # Show statically detected file references
    refs_info = ""
    if static_refs:
        refs_info = "\nSTATICALLY DETECTED FILE REFERENCES (confirmed by parser):\n"
        for ref in static_refs:
            refs_info += f"  - {ref}\n"

    interactions_info = ""
    if static_interactions:
        interactions_info = "\nSTATICALLY DETECTED INTERACTIONS:\n"
        for inter in static_interactions:
            interactions_info += f"  - {inter.source_file} --[{inter.interaction_type}]--> {inter.target_file}: {inter.description}\n"

    all_files_str = "\n".join(f"  - {f}" for f in all_project_files)

    return f"""\
Analyze this file and produce a structured JSON report.

FILE PATH: {file_path}
LANGUAGE: {metadata.language}
SIZE: {metadata.size_bytes} bytes ({tokens} estimated tokens)

ALL FILES IN THIS PROJECT:
{all_files_str}

{parsed_info}
{refs_info}
{interactions_info}

--- FILE CONTENT START ---
{truncated}
--- FILE CONTENT END ---

Produce a JSON object with exactly these fields:
- "file_path" (string): must be exactly "{file_path}"
- "summary" (string): 1-3 sentences about what this file does
- "functions" (array of {{"name": str, "description": str, "calls": [str], "imports_used": [str]}}): \
ONLY functions in the code
- "classes" (array of {{"name": str, "methods": [str]}}): ONLY classes in the code
- "exports" (array of strings): actual exportable names
- "external_dependencies" (array of strings): third-party packages imported
- "internal_file_references" (array of strings): other project files this file references \
(use the statically detected ones above as a baseline, add any others you find in the code)
- "file_interactions" (array of {{"source_file": str, "target_file": str, \
"interaction_type": str, "description": str}}): how this file uses each referenced file. \
interaction_type must be one of: injects, imports, loads, calls_into, configures, extends, includes
"""


async def analyze_file(
    groq: GroqClient,
    file_path: str,
    content: str,
    metadata: FileMetadata,
    parsed: ParsedStructure,
    all_project_files: list[str] | None = None,
) -> FileAnalysisResult:
    """Analyze a single file with dependency detection."""
    if all_project_files is None:
        all_project_files = []

    # Static analysis: extract file references before LLM
    static_refs = _extract_file_references(content, all_project_files)
    static_interactions = _detect_interactions(
        file_path, content, static_refs, metadata.language,
    )

    prompt = _build_file_analysis_prompt(
        file_path, content, metadata, parsed,
        all_project_files, static_refs, static_interactions,
    )

    try:
        result = await groq.structured_chat(
            prompt=prompt,
            system=FILE_ANALYSIS_SYSTEM_PROMPT,
            response_model=FileAnalysisResult,
            temperature=0.1,
        )
        result.file_path = file_path

        # Merge static refs with LLM-detected refs
        all_refs = set(static_refs)
        all_refs.update(result.internal_file_references)
        # Remove self-references
        all_refs.discard(file_path)
        result.internal_file_references = sorted(all_refs)

        # Merge static interactions with LLM interactions
        existing_pairs = {(i.source_file, i.target_file) for i in static_interactions}
        merged_interactions = list(static_interactions)
        for inter in result.file_interactions:
            if (inter.source_file, inter.target_file) not in existing_pairs:
                merged_interactions.append(inter)
        result.file_interactions = merged_interactions

        # Cross-validate with tree-sitter
        result = _validate_against_parsed(result, parsed)

        logger.info("✓ Analyzed: %s (refs=%d, interactions=%d)",
                     file_path, len(result.internal_file_references), len(result.file_interactions))
        return result

    except Exception as e:
        logger.error("✗ Analysis failed for %s: %s", file_path, e)
        return _build_fallback_result(file_path, parsed, static_refs, static_interactions)


def _validate_against_parsed(result: FileAnalysisResult, parsed: ParsedStructure) -> FileAnalysisResult:
    if not parsed.functions and not parsed.classes:
        return result

    if parsed.functions:
        ts_func_names = {f.name for f in parsed.functions}
        ts_methods = set()
        for cls in parsed.classes:
            ts_methods.update(cls.methods)

        validated = []
        for func in result.functions:
            if func.name in ts_func_names or func.name in ts_methods:
                validated.append(func)
            else:
                logger.debug("Removed hallucinated function: %s", func.name)
        result.functions = validated

    if parsed.classes:
        ts_class_names = {c.name for c in parsed.classes}
        validated = []
        for cls in result.classes:
            if cls.name in ts_class_names:
                ts_cls = next((c for c in parsed.classes if c.name == cls.name), None)
                if ts_cls:
                    valid_methods = [m for m in cls.methods if m in ts_cls.methods]
                    cls.methods = valid_methods if valid_methods else ts_cls.methods
                validated.append(cls)
        result.classes = validated

    return result


def _build_fallback_result(
    file_path: str,
    parsed: ParsedStructure,
    static_refs: list[str],
    static_interactions: list[FileInteraction],
) -> FileAnalysisResult:
    functions = [
        FunctionAnalysis(
            name=f.name,
            description=f"Function at lines {f.start_line}-{f.end_line}",
        )
        for f in parsed.functions
    ]
    classes = [
        ClassAnalysis(name=c.name, methods=c.methods)
        for c in parsed.classes
    ]
    return FileAnalysisResult(
        file_path=file_path,
        summary=f"File analysis via tree-sitter (LLM unavailable). "
                f"{len(functions)} functions, {len(classes)} classes.",
        functions=functions,
        classes=classes,
        exports=[f.name for f in parsed.functions] + [c.name for c in parsed.classes],
        external_dependencies=[i.module for i in parsed.imports],
        internal_file_references=static_refs,
        file_interactions=static_interactions,
    )