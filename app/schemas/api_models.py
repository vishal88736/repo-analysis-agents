"""Pydantic models for API request/response."""

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    repository_url: str = Field(
        ...,
        description="Full GitHub repository URL",
        examples=["https://github.com/pallets/flask"],
    )


class AnalyzeResponse(BaseModel):
    analysis_id: str
    status: str
    message: str


class QueryRequest(BaseModel):
    analysis_id: str
    question: str = Field(..., min_length=3)


class QueryResponse(BaseModel):
    analysis_id: str
    question: str
    answer: str
    sources: list[str] = Field(default_factory=list)


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
    # NEW fields
    technology_profile: dict = Field(default_factory=dict)
    file_interactions: list[dict] = Field(default_factory=list)
    execution_flow: dict = Field(default_factory=dict)
    data_flow: dict = Field(default_factory=dict)
    component_interaction_summary: str = ""


class StatusResponse(BaseModel):
    analysis_id: str
    status: str


class ErrorResponse(BaseModel):
    detail: str