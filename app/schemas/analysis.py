"""
Pydantic models for file analysis, agent outputs, and reports.
"""

from __future__ import annotations
from pydantic import BaseModel, Field


# --- File Metadata ---

class FileMetadata(BaseModel):
    path: str = Field(..., description="Relative file path from repo root")
    size_bytes: int = Field(..., ge=0)
    language: str = Field(default="unknown")
    extension: str = Field(default="")


# --- Tree-sitter Parsed Structure ---

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


# --- LLM Agent Output Models ---

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


# --- Combined Report ---

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
    diagram_type: str = "file_flow"  # file_flow | function_flow | entry_point_flow
    mermaid_syntax: str = ""


class FullAnalysisReport(BaseModel):
    analysis_id: str
    repository_url: str
    total_files: int = 0
    file_analyses: list[FileAnalysisResult] = Field(default_factory=list)
    architecture_summary: ArchitectureSummary = Field(default_factory=ArchitectureSummary)
    mermaid_diagrams: list[MermaidDiagram] = Field(default_factory=list)
    status: str = "pending"  # pending | processing | completed | failed
    error_message: str | None = None