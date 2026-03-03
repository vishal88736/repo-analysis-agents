"""Custom exception hierarchy."""

from fastapi import HTTPException, status


class AnalysisError(Exception):
    def __init__(self, message: str, details: str | None = None):
        self.message = message
        self.details = details
        super().__init__(self.message)


class RepositoryCloneError(AnalysisError):
    """Failed to clone repository."""
    pass


class LLMError(AnalysisError):
    """Error communicating with Groq API."""
    pass


class LLMRateLimitError(LLMError):
    """Rate limit exceeded on Groq API."""
    pass


class LLMResponseValidationError(LLMError):
    """LLM returned invalid/unparseable structured output."""
    pass


class ParsingError(AnalysisError):
    """Tree-sitter or file parsing error."""
    pass


class AnalysisNotFoundError(AnalysisError):
    """Requested analysis ID not found."""
    pass