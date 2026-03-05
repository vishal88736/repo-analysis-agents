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


# === Feature 2: Compact File Summary ===

class CompactFileSummary(BaseModel):
    """Lightweight summary — stored and reused instead of full analysis text."""
    file_path: str
    purpose: str = ""
    functions: list[str] = Field(default_factory=list)
    classes: list[str] = Field(default_factory=list)
    imports: list[str] = Field(default_factory=list)
    key_dependencies: list[str] = Field(default_factory=list)
    entry_point: bool = False


# === Feature 1: Repository Map ===

class RepoMapEntry(BaseModel):
    language: str = "unknown"
    size_bytes: int = 0
    tokens_estimate: int = 0
    extension: str = ""
    directory: str = ""


class RepoMap(BaseModel):
    """Lightweight repo structure — no code content, just metadata."""
    files: dict[str, RepoMapEntry] = Field(default_factory=dict)
    total_files: int = 0
    total_tokens_estimate: int = 0
    languages: dict[str, int] = Field(default_factory=dict)
    directory_tree: list[str] = Field(default_factory=list)


# === Feature 11/16: Query Plan ===

class QueryPlan(BaseModel):
    """Output of the query planning agent."""
    relevant_files: list[str] = Field(default_factory=list)
    relevant_modules: list[str] = Field(default_factory=list)
    reasoning: str = ""
    needs_raw_code: bool = False


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
    compact_summaries: list[CompactFileSummary] = Field(default_factory=list)
    architecture_summary: ArchitectureSummary = Field(default_factory=ArchitectureSummary)
    mermaid_diagrams: list[MermaidDiagram] = Field(default_factory=list)
    repo_map: RepoMap | None = None
    status: str = "pending"
    error_message: str | None = None