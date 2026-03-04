"""
Feature 9: Smart code-aware chunking for RAG.

Chunks files by function/class/module boundaries rather than naive character splitting.
"""

import logging

from app.schemas.analysis import ParsedStructure
from app.services.token_utils import estimate_tokens

logger = logging.getLogger(__name__)


class SmartChunker:
    """Chunks source files respecting code structure boundaries."""

    def chunk_file(
        self,
        content: str,
        parsed: ParsedStructure,
        file_path: str = "",
        max_chunk_tokens: int = 500,
    ) -> list[dict]:
        """
        Split a file into semantically meaningful chunks.

        Returns list of dicts with keys:
          - text: chunk content
          - type: "function" | "class" | "imports" | "module"
          - name: function/class name or empty string
          - file_path: the source file path
        """
        lines = content.splitlines(keepends=True)
        chunks: list[dict] = []

        # Collect all structural boundaries
        blocks: list[dict] = []

        # Imports block: first continuous block of import lines
        import_lines: list[int] = []
        for imp in parsed.imports:
            # ParsedImport doesn't have line info, scan for import lines
            pass

        # Functions
        for func in parsed.functions:
            block_text = "".join(lines[func.start_line - 1:func.end_line])
            blocks.append({
                "type": "function",
                "name": func.name,
                "start": func.start_line,
                "end": func.end_line,
                "text": block_text,
            })

        # Classes
        for cls in parsed.classes:
            block_text = "".join(lines[cls.start_line - 1:cls.end_line])
            blocks.append({
                "type": "class",
                "name": cls.name,
                "start": cls.start_line,
                "end": cls.end_line,
                "text": block_text,
            })

        # Sort blocks by start line
        blocks.sort(key=lambda b: b["start"])

        # Build set of covered line ranges
        covered: set[int] = set()
        for b in blocks:
            covered.update(range(b["start"], b["end"] + 1))

        # Module-level lines (not covered by any function/class)
        module_lines = [i + 1 for i in range(len(lines)) if (i + 1) not in covered]

        # Add module-level chunk
        if module_lines:
            # Group consecutive module lines
            groups = _group_consecutive(module_lines)
            for group in groups:
                text = "".join(lines[ln - 1] for ln in group)
                if text.strip():
                    chunks.append({
                        "text": text,
                        "type": "module",
                        "name": "",
                        "file_path": file_path,
                    })

        # Add function/class blocks, splitting large ones if needed
        for block in blocks:
            if estimate_tokens(block["text"]) <= max_chunk_tokens:
                chunks.append({
                    "text": block["text"],
                    "type": block["type"],
                    "name": block["name"],
                    "file_path": file_path,
                })
            else:
                # Split large block into sub-chunks
                sub_chunks = _split_text(block["text"], max_chunk_tokens)
                for i, sub in enumerate(sub_chunks):
                    chunks.append({
                        "text": sub,
                        "type": block["type"],
                        "name": f"{block['name']}[{i}]",
                        "file_path": file_path,
                    })

        # If no structured chunks, fall back to naive chunking
        if not chunks:
            sub_chunks = _split_text(content, max_chunk_tokens)
            for i, sub in enumerate(sub_chunks):
                chunks.append({
                    "text": sub,
                    "type": "module",
                    "name": f"chunk_{i}",
                    "file_path": file_path,
                })

        return chunks


def _group_consecutive(lines: list[int]) -> list[list[int]]:
    """Group consecutive line numbers into segments."""
    if not lines:
        return []
    groups = []
    current = [lines[0]]
    for ln in lines[1:]:
        if ln == current[-1] + 1:
            current.append(ln)
        else:
            groups.append(current)
            current = [ln]
    groups.append(current)
    return groups


def _split_text(text: str, max_tokens: int) -> list[str]:
    """Naive character-based split respecting token budget."""
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + max_chars])
        start += max_chars
    return chunks
