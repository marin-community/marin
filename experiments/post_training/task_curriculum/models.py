# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Data exchanged by curriculum generation, review, and task mapping."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SubjectGuidepost(StrictModel):
    id: str
    name: str


class SubjectArea(StrictModel):
    id: str
    name: str
    guideposts: list[SubjectGuidepost]


class SubjectInventory(StrictModel):
    version: str
    source_url: str
    areas: list[SubjectArea]

    def area(self, area_id: str) -> SubjectArea:
        matches = [area for area in self.areas if area.id == area_id]
        if len(matches) != 1:
            raise ValueError(f"expected one subject area {area_id}, found {len(matches)}")
        return matches[0]


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


class CurriculumReview(StrictModel):
    score: int = Field(ge=0, le=100)
    verdict: str = Field(min_length=1)
    findings: list[str]
    rubric_changes: list[str]


class CurriculumReviewV2(StrictModel):
    score: int = Field(ge=0, le=100)
    verdict: str = Field(min_length=1)
    blocking_findings: list[str]
    mutual_confidence_findings: list[str]
    continuity_findings: list[str]
    other_findings: list[str]
    rubric_changes: list[str]


class EvaluationProbe(StrictModel):
    task_id: str
    benchmark: str
    source_revision: str
    split: str
    item_id: str
    task_hash: str
    applicable_subject_ids: list[str] = Field(min_length=1)


class EvaluationPolicySource(StrictModel):
    benchmark: str
    policy_class: Literal["in_distribution", "out_of_distribution"]
    allowed_use: Literal["held_out_examples", "metadata_only"]
    source_url: str
    source_revision: str | None


class SemanticKey(StrictModel):
    subject: str = Field(
        min_length=1,
        description="Operational knowledge domain needed to solve the task, excluding narrative subject matter",
    )
    summary: str = Field(min_length=1, description="Requested result without source or curriculum labels")
    hardest_operation: str = Field(min_length=1)
    required_operations: list[str] = Field(min_length=1)
    answer_form: str = Field(min_length=1)

    def embedding_text(self) -> str:
        return (
            f"Subject: {self.subject}. Task: {self.summary}. "
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


class TaskMechanicAnnotation(StrictModel):
    task_id: str
    task_hash: str
    model: str
    prompt_version: str
    task_mechanic: str = Field(min_length=1)


class MappingCandidate(StrictModel):
    section_id: str
    similarity: float = Field(ge=-1.0, le=1.0)


class TaskMapping(StrictModel):
    task_id: str
    task_hash: str
    annotation_model: str
    annotation_prompt_version: str
    curriculum_versions: dict[str, str]
    embedding_model: str
    candidates: list[MappingCandidate]


class MappingReference(StrictModel):
    task_id: str
    status: Literal["exact", "ambiguous", "out_of_scope"]
    acceptable_section_ids: list[str]
    rationale: str

    @model_validator(mode="after")
    def validate_sections(self) -> MappingReference:
        if self.status == "out_of_scope" and self.acceptable_section_ids:
            raise ValueError("out-of-scope references cannot name acceptable sections")
        if self.status != "out_of_scope" and not self.acceptable_section_ids:
            raise ValueError("in-scope references must name an acceptable section")
        return self
