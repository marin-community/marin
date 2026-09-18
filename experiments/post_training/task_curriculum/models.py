# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Data exchanged by curriculum generation, review, and task mapping."""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SampleTask(StrictModel):
    kind: Literal["entry", "representative"]
    instruction: str = Field(min_length=1)


class CurriculumSection(StrictModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    parent_id: str | None
    name: str = Field(min_length=1)
    outcome: str = Field(min_length=1)
    includes: list[str] = Field(min_length=1)
    excludes: list[str] = Field(min_length=1)
    prerequisites: list[str]
    sample_tasks: list[SampleTask]


def _reject_cycles(edges: dict[str, Iterable[str]], label: str) -> None:
    visited: set[str] = set()
    visiting: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            raise ValueError(f"{label} cycle includes {node}")
        if node in visited:
            return
        visiting.add(node)
        for dependency in edges[node]:
            visit(dependency)
        visiting.remove(node)
        visited.add(node)

    for node in edges:
        visit(node)


class Curriculum(StrictModel):
    version: str
    subject_id: str
    subject_name: str
    sections: list[CurriculumSection] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_graph(self) -> Curriculum:
        ids = [section.id for section in self.sections]
        if len(ids) != len(set(ids)):
            raise ValueError("section IDs must be unique")

        known = set(ids)
        parent_edges: dict[str, list[str]] = {}
        prerequisite_edges: dict[str, list[str]] = {}
        for section in self.sections:
            parent_edges[section.id] = [] if section.parent_id is None else [section.parent_id]
            prerequisite_edges[section.id] = section.prerequisites
            references = set(parent_edges[section.id]) | set(section.prerequisites)
            unknown = sorted(references - known)
            if unknown:
                raise ValueError(f"section {section.id} has unknown references: {unknown}")
            if section.id in references:
                raise ValueError(f"section {section.id} refers to itself")

        _reject_cycles(parent_edges, "parent")
        _reject_cycles(prerequisite_edges, "prerequisite")
        return self

    def depths(self) -> dict[str, int]:
        by_id = {section.id: section for section in self.sections}
        depths: dict[str, int] = {}

        def depth(section_id: str) -> int:
            if section_id not in depths:
                parent_id = by_id[section_id].parent_id
                depths[section_id] = 1 if parent_id is None else depth(parent_id) + 1
            return depths[section_id]

        for section_id in by_id:
            depth(section_id)
        return depths

    def check_generation_contract(self, maximum_depth: int) -> None:
        actual_depth = max(self.depths().values())
        if actual_depth > maximum_depth:
            raise ValueError(f"maximum depth {maximum_depth}, generated depth {actual_depth}")
        wrong_probes = [
            section.id
            for section in self.sections
            if [task.kind for task in section.sample_tasks] != ["entry", "representative"]
        ]
        if wrong_probes:
            raise ValueError(f"sections must have one entry and one representative probe in order: {wrong_probes}")


class RoutingFacet(StrEnum):
    SUBJECT_DOMAIN = "subject_domain"
    TASK_MECHANIC = "task_mechanic"


class AnchorKind(StrEnum):
    GRAPH = "graph"
    SECTION = "section"


class CatalogCurriculum(StrictModel):
    routing_facet: RoutingFacet
    curriculum: Curriculum


class CurriculumCatalog(StrictModel):
    catalog_version: str
    curricula: list[CatalogCurriculum] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_catalog(self) -> CurriculumCatalog:
        subject_ids = [entry.curriculum.subject_id for entry in self.curricula]
        if len(subject_ids) != len(set(subject_ids)):
            raise ValueError("subject IDs must be unique across curricula")
        section_ids = [section.id for entry in self.curricula for section in entry.curriculum.sections]
        if len(section_ids) != len(set(section_ids)):
            raise ValueError("section IDs must be unique across curricula")
        return self


class AssignmentAnchor(StrictModel):
    kind: AnchorKind
    subject_id: str
    section_id: str | None
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_target(self) -> AssignmentAnchor:
        if self.kind == AnchorKind.GRAPH and self.section_id is not None:
            raise ValueError("graph anchors cannot name a section")
        if self.kind == AnchorKind.SECTION and self.section_id is None:
            raise ValueError("section anchors must name a section")
        return self


class SemanticKey(StrictModel):
    subject_domain: str = Field(
        min_length=1,
        description="Operational knowledge domain needed to solve the task, excluding narrative subject matter",
    )
    task_mechanic: str = Field(
        min_length=1,
        description="Domain-independent transformation or reasoning action required by the task",
    )
    summary: str = Field(min_length=1, description="Requested result without source or curriculum labels")
    hardest_operation: str = Field(min_length=1)
    required_operations: list[str] = Field(min_length=1)
    answer_form: str = Field(min_length=1)

    def membership_text(self, facet: RoutingFacet) -> str:
        if facet == RoutingFacet.SUBJECT_DOMAIN:
            return f"Subject domain: {self.subject_domain}."
        return f"Task mechanic: {self.task_mechanic}."

    def operation_text(self) -> str:
        return (
            f"Task: {self.summary}. "
            f"Hardest operation: {self.hardest_operation}. "
            f"Required operations: {'; '.join(self.required_operations)}. "
            f"Answer form: {self.answer_form}."
        )


class TaskAnnotation(StrictModel):
    task_id: str
    task_hash: str
    model: str
    prompt_version: str
    key: SemanticKey


class MappingCandidate(StrictModel):
    section_id: str
    similarity: float = Field(ge=-1.0, le=1.0)


class GraphMapping(StrictModel):
    subject_id: str
    routing_facet: RoutingFacet
    membership_similarity: float = Field(ge=-1.0, le=1.0)
    candidates: list[MappingCandidate]


class TaskMapping(StrictModel):
    task_id: str
    task_hash: str
    annotation_model: str
    annotation_prompt_version: str
    catalog_version: str
    embedding_model: str
    graphs: list[GraphMapping]
