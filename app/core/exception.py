"""
Custom exception hierarchy for the application.
"""

from fastapi import HTTPException, status


class AnalysisError(Exception):
    """Base exception for analysis errors."""

    def __init__(self, message: str, details: str | None = None):
        self.message = message
        self.details = details
        super().__init__(self.message)


class RepositoryCloneError(AnalysisError):
    """Failed to clone repository."""
    pass


class LLMError(AnalysisError):
    """Error communicating with the Grok LLM API."""
    pass


class LLMRateLimitError(LLMError):
    """Rate limit exceeded on Grok API."""
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


def analysis_not_found_exception(analysis_id: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Analysis with ID '{analysis_id}' not found.",
    )