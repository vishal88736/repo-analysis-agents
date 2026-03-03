"""
Pydantic models for the dependency graph.
"""

from __future__ import annotations
from pydantic import BaseModel, Field


class DependencyEdge(BaseModel):
    source: str
    target: str
    relationship: str = "imports"  # imports | calls | extends


class DependencyNode(BaseModel):
    id: str
    type: str = "file"  # file | function | class
    label: str = ""
    file_path: str = ""


class DependencyGraph(BaseModel):
    nodes: list[DependencyNode] = Field(default_factory=list)
    edges: list[DependencyEdge] = Field(default_factory=list)
    adjacency_list: dict[str, list[str]] = Field(default_factory=dict)