"""
Feature 10: Context compression — removes noise from code before sending to LLM.

Removes multiline comments/docstrings, excessive whitespace, while keeping
function bodies, signatures, variable assignments, and control flow.
"""

import re
import logging

from app.services.token_utils import estimate_tokens

logger = logging.getLogger(__name__)


class ContextCompressor:
    """Removes comment noise from code to reduce token count."""

    def compress(self, code: str, language: str) -> str:
        """
        Compress code by removing comments and excessive whitespace.

        Keeps function bodies, signatures, variable assignments, and control flow.
        """
        lang = language.lower()
        if lang == "python":
            return self._compress_python(code)
        elif lang in ("javascript", "typescript"):
            return self._compress_js_ts(code)
        else:
            return self._compress_generic(code)

    def compress_for_prompt(
        self, code: str, language: str, max_tokens: int = 1500
    ) -> str:
        """Compress code and truncate to fit within a token budget."""
        compressed = self.compress(code, language)
        max_chars = max_tokens * 4
        if len(compressed) > max_chars:
            compressed = compressed[:max_chars] + "\n... [truncated for token budget] ..."
        return compressed

    def _compress_python(self, code: str) -> str:
        """Remove Python docstrings and comments."""
        # Remove triple-quoted strings (docstrings)
        code = re.sub(r'"""[\s\S]*?"""', '', code)
        code = re.sub(r"'''[\s\S]*?'''", '', code)
        # Remove single-line comments
        code = re.sub(r'#[^\n]*', '', code)
        return self._clean_whitespace(code)

    def _compress_js_ts(self, code: str) -> str:
        """Remove JS/TS block comments and single-line comments."""
        # Remove block comments
        code = re.sub(r'/\*[\s\S]*?\*/', '', code)
        # Remove single-line comments
        code = re.sub(r'//[^\n]*', '', code)
        return self._clean_whitespace(code)

    def _compress_generic(self, code: str) -> str:
        """Generic: remove C-style block comments and # comments."""
        code = re.sub(r'/\*[\s\S]*?\*/', '', code)
        code = re.sub(r'//[^\n]*', '', code)
        code = re.sub(r'#[^\n]*', '', code)
        return self._clean_whitespace(code)

    def _clean_whitespace(self, code: str) -> str:
        """Remove excessive blank lines (max 1 consecutive blank line)."""
        # Replace 3+ consecutive newlines with 2
        code = re.sub(r'\n{3,}', '\n\n', code)
        # Remove trailing whitespace on each line
        lines = [ln.rstrip() for ln in code.splitlines()]
        return "\n".join(lines).strip()
