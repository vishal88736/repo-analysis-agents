"""
Feature 3: Token estimation and splitting utilities.

Uses ~4 chars per token heuristic for estimation.
Splits files by function/class boundaries from tree-sitter parsed structure.
"""

from app.schemas.analysis import ParsedStructure, FileAnalysisResult


def estimate_tokens(text: str) -> int:
    """Estimate token count using ~4 chars per token heuristic."""
    return max(1, len(text) // 4)


def split_by_boundaries(
    content: str,
    parsed: ParsedStructure,
    max_tokens: int,
) -> list[str]:
    """
    Split file content into chunks respecting function/class boundaries.

    Uses tree-sitter parsed boundaries when available; falls back to
    naive line-based splitting.
    """
    lines = content.splitlines(keepends=True)

    # Collect all boundaries: (start_line, end_line) 1-based
    boundaries: list[tuple[int, int]] = []
    for func in parsed.functions:
        boundaries.append((func.start_line, func.end_line))
    for cls in parsed.classes:
        boundaries.append((cls.start_line, cls.end_line))

    if not boundaries:
        # No boundaries: naive chunking by token count
        return _naive_split(content, max_tokens)

    # Sort boundaries by start line
    boundaries.sort(key=lambda x: x[0])

    chunks: list[str] = []
    current_chunk_lines: list[str] = []
    current_tokens = 0
    last_end = 0

    for start, end in boundaries:
        # Add lines between last boundary and this one (e.g. module-level code)
        preamble = lines[last_end:start - 1]
        preamble_text = "".join(preamble)
        preamble_tokens = estimate_tokens(preamble_text)

        block_lines = lines[start - 1:end]
        block_text = "".join(block_lines)
        block_tokens = estimate_tokens(block_text)

        # If adding preamble + block would exceed limit, flush current chunk
        if current_tokens + preamble_tokens + block_tokens > max_tokens and current_chunk_lines:
            chunks.append("".join(current_chunk_lines))
            current_chunk_lines = []
            current_tokens = 0

        current_chunk_lines.extend(preamble)
        current_chunk_lines.extend(block_lines)
        current_tokens += preamble_tokens + block_tokens
        last_end = end

    # Remaining lines after last boundary
    tail = lines[last_end:]
    if tail:
        current_chunk_lines.extend(tail)

    if current_chunk_lines:
        # If the accumulated chunk is too large, split it naively
        accumulated = "".join(current_chunk_lines)
        if estimate_tokens(accumulated) > max_tokens:
            chunks.extend(_naive_split(accumulated, max_tokens))
        else:
            chunks.append(accumulated)

    return chunks if chunks else [content]


def _naive_split(content: str, max_tokens: int) -> list[str]:
    """Split content naively by token budget."""
    max_chars = max_tokens * 4
    if len(content) <= max_chars:
        return [content]
    chunks = []
    start = 0
    while start < len(content):
        chunks.append(content[start:start + max_chars])
        start += max_chars
    return chunks


def combine_chunk_results(results: list[FileAnalysisResult]) -> FileAnalysisResult:
    """Combine multiple chunk analysis results into a single FileAnalysisResult."""
    if not results:
        raise ValueError("No results to combine")
    if len(results) == 1:
        return results[0]

    base = results[0]
    combined_summary = " ".join(r.summary for r in results if r.summary)
    all_functions = []
    seen_funcs: set[str] = set()
    all_classes = []
    seen_classes: set[str] = set()
    all_exports: set[str] = set()
    all_deps: set[str] = set()

    for r in results:
        for f in r.functions:
            if f.name not in seen_funcs:
                all_functions.append(f)
                seen_funcs.add(f.name)
        for c in r.classes:
            if c.name not in seen_classes:
                all_classes.append(c)
                seen_classes.add(c.name)
        all_exports.update(r.exports)
        all_deps.update(r.external_dependencies)

    return FileAnalysisResult(
        file_path=base.file_path,
        summary=combined_summary,
        functions=all_functions,
        classes=all_classes,
        exports=list(all_exports),
        external_dependencies=list(all_deps),
    )
