"""
Pydantic models for API request/response.
"""

from pydantic import BaseModel, Field, HttpUrl


class AnalyzeRequest(BaseModel):
    repository_url: str = Field(
        ...,
        description="Full GitHub repository URL",
        examples=["https://github.com/owner/repo"],
    )


class AnalyzeResponse(BaseModel):
    analysis_id: str
    status: str
    message: str


class QueryRequest(BaseModel):
    analysis_id: str = Field(..., description="The analysis ID returned from /analyze")
    question: str = Field(..., min_length=3, description="Natural language question about the codebase")


class QueryResponse(BaseModel):
    analysis_id: str
    question: str
    answer: str
    sources: list[str] = Field(default_factory=list, description="Relevant file paths used")


class ReportResponse(BaseModel):
    analysis_id: str
    repository_url: str
    status: str
    total_files: int
    global_summary: str
    key_components: list[str]
    design_patterns: list[str]
    technology_stack: list[str]
    entry_points: list[dict]
    file_summaries: list[dict]
    mermaid_diagrams: list[dict]


class ErrorResponse(BaseModel):
    detail: str