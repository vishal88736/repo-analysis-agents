"""Pydantic models for the dependency graph."""

from __future__ import annotations
from pydantic import BaseModel, Field
from collections import defaultdict


class DependencyEdge(BaseModel):
    source: str
    target: str
    relationship: str = "imports"


class DependencyNode(BaseModel):
    id: str
    type: str = "file"
    label: str = ""
    file_path: str = ""


class DependencyGraph(BaseModel):
    nodes: list[DependencyNode] = Field(default_factory=list)
    edges: list[DependencyEdge] = Field(default_factory=list)
    adjacency_list: dict[str, list[str]] = Field(default_factory=dict)