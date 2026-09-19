# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Structured inputs and outputs for task-to-curriculum mapping."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from experiments.post_training.task_curriculum.models import RoutingFacet, StrictModel

MIN_COSINE_SIMILARITY = -1.0
MAX_COSINE_SIMILARITY = 1.0


class AnchorKind(StrEnum):
    """Level of the curriculum targeted by an embedding anchor."""

    GRAPH = "graph"
    SECTION = "section"


class AssignmentAnchor(StrictModel):
    """Human-authored embedding anchor for a graph or capability."""

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
    """Curriculum-independent semantic annotation embedded once per task."""

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
        """Render the semantic projection used for graph membership."""
        if facet == RoutingFacet.SUBJECT_DOMAIN:
            return f"Subject domain: {self.subject_domain}."
        return f"Task mechanic: {self.task_mechanic}."

    def operation_text(self) -> str:
        """Render the operation projection used for capability ranking."""
        return (
            f"Task: {self.summary}. "
            f"Hardest operation: {self.hardest_operation}. "
            f"Required operations: {'; '.join(self.required_operations)}. "
            f"Answer form: {self.answer_form}."
        )


class TaskAnnotation(StrictModel):
    """Versioned semantic key for one model-visible task."""

    task_id: str
    task_hash: str
    model: str
    prompt_version: str
    key: SemanticKey


class MappingCandidate(StrictModel):
    """One ranked capability candidate within a curriculum graph."""

    section_id: str
    similarity: float = Field(ge=MIN_COSINE_SIMILARITY, le=MAX_COSINE_SIMILARITY)


class GraphMapping(StrictModel):
    """Graph membership score and its ranked capability candidates."""

    subject_id: str
    routing_facet: RoutingFacet
    membership_similarity: float = Field(ge=MIN_COSINE_SIMILARITY, le=MAX_COSINE_SIMILARITY)
    candidates: list[MappingCandidate]


class TaskMapping(StrictModel):
    """Versioned graph and capability rankings for one task."""

    task_id: str
    task_hash: str
    annotation_model: str
    annotation_prompt_version: str
    catalog_version: str
    embedding_model: str
    graphs: list[GraphMapping]
