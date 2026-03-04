"""Pydantic models for file analysis, agent outputs, and reports."""

from __future__ import annotations
from pydantic import BaseModel, Field


class FileMetadata(BaseModel):
    path: str = Field(..., description="Relative file path from repo root")
    size_bytes: int = Field(..., ge=0)
    language: str = Field(default="unknown")
    extension: str = Field(default="")


class ParsedFunction(BaseModel):
    name: str
    start_line: int
    end_line: int
    parameters: list[str] = Field(default_factory=list)


class ParsedClass(BaseModel):
    name: str
    start_line: int
    end_line: int
    methods: list[str] = Field(default_factory=list)


class ParsedImport(BaseModel):
    module: str
    names: list[str] = Field(default_factory=list)


class ParsedStructure(BaseModel):
    functions: list[ParsedFunction] = Field(default_factory=list)
    classes: list[ParsedClass] = Field(default_factory=list)
    imports: list[ParsedImport] = Field(default_factory=list)


class FunctionAnalysis(BaseModel):
    name: str
    description: str = ""
    calls: list[str] = Field(default_factory=list)
    imports_used: list[str] = Field(default_factory=list)


class ClassAnalysis(BaseModel):
    name: str
    methods: list[str] = Field(default_factory=list)


class FileAnalysisResult(BaseModel):
    file_path: str
    summary: str = ""
    functions: list[FunctionAnalysis] = Field(default_factory=list)
    classes: list[ClassAnalysis] = Field(default_factory=list)
    exports: list[str] = Field(default_factory=list)
    external_dependencies: list[str] = Field(default_factory=list)


class EntryPoint(BaseModel):
    file_path: str
    function_name: str = ""
    reason: str = ""


class ArchitectureSummary(BaseModel):
    overview: str = ""
    key_components: list[str] = Field(default_factory=list)
    design_patterns: list[str] = Field(default_factory=list)
    entry_points: list[EntryPoint] = Field(default_factory=list)
    technology_stack: list[str] = Field(default_factory=list)


class MermaidDiagram(BaseModel):
    title: str
    diagram_type: str = "file_flow"
    mermaid_syntax: str = ""


class FullAnalysisReport(BaseModel):
    analysis_id: str
    repository_url: str
    total_files: int = 0
    file_analyses: list[FileAnalysisResult] = Field(default_factory=list)
    architecture_summary: ArchitectureSummary = Field(default_factory=ArchitectureSummary)
    mermaid_diagrams: list[MermaidDiagram] = Field(default_factory=list)
    status: str = "pending"
    error_message: str | None = None


# --- Feature 1: Repo Map ---

class RepoFileInfo(BaseModel):
    language: str = "unknown"
    size: int = 0
    tokens_estimate: int = 0
    file_type: str = ""


class RepoMap(BaseModel):
    """Repository map: file paths → metadata for context-efficient reasoning."""
    files: dict[str, RepoFileInfo] = Field(default_factory=dict)
    total_files: int = 0
    total_tokens_estimate: int = 0


# --- Feature 2: Compact File Summaries ---

class CompactFileSummary(BaseModel):
    """Compact structured summary for progressive context loading."""
    file_path: str
    purpose: str = ""
    functions: list[str] = Field(default_factory=list)
    classes: list[str] = Field(default_factory=list)
    imports: list[str] = Field(default_factory=list)
    key_dependencies: list[str] = Field(default_factory=list)
    entry_points: list[str] = Field(default_factory=list)


# --- Feature 11: Query Plan ---

class QueryPlan(BaseModel):
    """Plan produced by the query planner before retrieval."""
    relevant_files: list[str] = Field(default_factory=list)
    relevant_modules: list[str] = Field(default_factory=list)
    search_keywords: list[str] = Field(default_factory=list)
    graph_entry_points: list[str] = Field(default_factory=list)
