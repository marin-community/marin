# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exact, faceted curriculum retrieval with a shared cosine-distance cutoff."""

from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property

import numpy as np
from pydantic import BaseModel, ConfigDict

from taskcompendium.pipeline.semantics import SemanticProjection
from taskcompendium.pipeline.task_embeddings import normalized_vectors

MAX_DISTANCE = 0.2
TOP_K = 3
CANDIDATE_SUBJECTS_PER_FACET = 3


class AnchorKind(StrEnum):
    SUBJECT = "subject"
    CAPABILITY = "capability"


class CurriculumAnchor(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    anchor_id: str
    kind: AnchorKind
    subject_id: str
    capability_id: str | None
    projection: SemanticProjection
    text: str


class AssignmentStatus(StrEnum):
    ASSIGNED = "assigned"
    SUBJECT_ONLY = "subject_only"
    OUT_OF_SCOPE = "out_of_scope"


class CapabilityCandidate(BaseModel):
    capability_id: str
    similarity: float
    anchor_id: str


class SubjectAssignment(BaseModel):
    subject_id: str
    projection: SemanticProjection
    status: AssignmentStatus
    membership_similarity: float
    membership_anchor_id: str
    capability_id: str | None
    margin: float
    candidates: list[CapabilityCandidate]


def _top_positions(scores: np.ndarray, count: int) -> np.ndarray:
    count = min(count, len(scores))
    cutoff = np.partition(scores, len(scores) - count)[len(scores) - count]
    greater = np.flatnonzero(scores > cutoff)
    equal = np.flatnonzero(scores == cutoff)[: count - len(greater)]
    selected = np.concatenate([greater, equal])
    return selected[np.lexsort((selected, -scores[selected]))]


def _cosine(value: float) -> float:
    return float(np.clip(value, -1, 1))


@dataclass(frozen=True)
class _CapabilityReference:
    capability_id: str
    anchor_positions: tuple[int, ...]


@dataclass(frozen=True)
class _SubjectReference:
    subject_id: str
    projection: SemanticProjection
    anchor_positions: tuple[int, ...]
    capability_positions: tuple[int, ...]


@dataclass(frozen=True)
class _CurriculumIndex:
    subjects: tuple[_SubjectReference, ...]
    capabilities: tuple[_CapabilityReference, ...]
    references: np.ndarray
    operation_positions: np.ndarray
    capability_starts: np.ndarray


@dataclass(frozen=True)
class CurriculumVectors:
    """Small reference matrices loaded once for a task shard."""

    anchors: tuple[CurriculumAnchor, ...]
    vectors: np.ndarray

    def __post_init__(self) -> None:
        if not self.anchors:
            raise ValueError("Curriculum must contain anchors")
        normalized_vectors(self.vectors, len(self.anchors), self.vectors.shape[1])
        identifiers = [anchor.anchor_id for anchor in self.anchors]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Curriculum anchor IDs must be unique")

    @cached_property
    def index(self) -> _CurriculumIndex:
        subjects = []
        capabilities = []
        for subject_id in sorted({anchor.subject_id for anchor in self.anchors}):
            membership = tuple(
                index
                for index, anchor in enumerate(self.anchors)
                if anchor.subject_id == subject_id and anchor.kind == AnchorKind.SUBJECT
            )
            projections = {self.anchors[index].projection for index in membership}
            if len(projections) != 1 or SemanticProjection.OPERATIONS in projections:
                raise ValueError(f"Subject {subject_id} must declare one membership projection")
            by_capability: dict[str, list[int]] = {}
            for index, anchor in enumerate(self.anchors):
                if anchor.subject_id != subject_id or anchor.kind != AnchorKind.CAPABILITY:
                    continue
                if anchor.capability_id is None or anchor.projection != SemanticProjection.OPERATIONS:
                    raise ValueError("Capability anchors require an ID and the operations projection")
                by_capability.setdefault(anchor.capability_id, []).append(index)
            if not by_capability:
                raise ValueError(f"Subject {subject_id} has no capability anchors")
            positions = []
            for capability_id in sorted(by_capability):
                positions.append(len(capabilities))
                anchors = tuple(sorted(by_capability[capability_id], key=lambda index: self.anchors[index].anchor_id))
                capabilities.append(_CapabilityReference(capability_id, anchors))
            subjects.append(_SubjectReference(subject_id, next(iter(projections)), membership, tuple(positions)))
        operation_positions = np.asarray(
            [position for capability in capabilities for position in capability.anchor_positions]
        )
        starts = np.cumsum([0, *(len(capability.anchor_positions) for capability in capabilities[:-1])])
        return _CurriculumIndex(
            tuple(subjects),
            tuple(capabilities),
            normalized_vectors(self.vectors, len(self.anchors), self.vectors.shape[1]),
            operation_positions,
            starts,
        )


def rank_task_batch(
    curriculum: CurriculumVectors,
    task_vectors: dict[SemanticProjection, np.ndarray],
) -> list[list[SubjectAssignment]]:
    """Rank all anchors exactly and accept subjects and capabilities within MAX_DISTANCE."""
    row_count = len(task_vectors[SemanticProjection.OPERATIONS])
    dimensions = curriculum.vectors.shape[1]
    tasks = {
        projection: normalized_vectors(task_vectors[projection], row_count, dimensions)
        for projection in SemanticProjection
    }
    index = curriculum.index
    subjects = [subject.subject_id for subject in index.subjects]
    # Contiguous anchor groups reduce once per batch; membership never scores capability anchors.
    operation_scores = tasks[SemanticProjection.OPERATIONS] @ index.references[index.operation_positions].T
    capability_scores = np.maximum.reduceat(operation_scores, index.capability_starts, axis=1)
    membership_scores = [
        tasks[subject.projection] @ index.references[list(subject.anchor_positions)].T for subject in index.subjects
    ]
    graph_scores = np.stack([scores.max(axis=1) for scores in membership_scores], axis=1)
    operation_columns = {int(position): column for column, position in enumerate(index.operation_positions)}
    rows = []
    for row in range(row_count):
        selected = set()
        for projection in (SemanticProjection.SUBJECT_DOMAIN, SemanticProjection.TASK_MECHANIC):
            indices = [position for position, subject in enumerate(index.subjects) if subject.projection == projection]
            if indices:
                selected.update(
                    indices[position]
                    for position in _top_positions(graph_scores[row, indices], CANDIDATE_SUBJECTS_PER_FACET)
                )
        selected.update(index for index, subject in enumerate(subjects) if 1 - graph_scores[row, index] <= MAX_DISTANCE)
        assignments = []
        for subject_index in sorted(selected):
            subject = subjects[subject_index]
            reference = index.subjects[subject_index]
            positions = reference.capability_positions
            scores = capability_scores[row, list(positions)]
            winners = _top_positions(scores, TOP_K)
            margin = _cosine(scores[winners[0]]) - _cosine(scores[winners[1]]) if len(winners) > 1 else 2.0
            candidates = []
            for winner in winners:
                capability = index.capabilities[positions[winner]]
                anchors = capability.anchor_positions
                columns = [operation_columns[position] for position in anchors]
                anchor = anchors[int(np.argmax(operation_scores[row, columns]))]
                candidates.append(
                    CapabilityCandidate(
                        capability_id=capability.capability_id,
                        similarity=_cosine(scores[winner]),
                        anchor_id=curriculum.anchors[anchor].anchor_id,
                    )
                )
            member = _cosine(graph_scores[row, subject_index])
            if 1 - member > MAX_DISTANCE:
                status = AssignmentStatus.OUT_OF_SCOPE
            elif 1 - candidates[0].similarity > MAX_DISTANCE:
                status = AssignmentStatus.SUBJECT_ONLY
            else:
                status = AssignmentStatus.ASSIGNED
            membership_anchors = reference.anchor_positions
            best_anchor = membership_anchors[int(np.argmax(membership_scores[subject_index][row]))]
            assignments.append(
                SubjectAssignment(
                    subject_id=subject,
                    projection=reference.projection,
                    status=status,
                    membership_similarity=member,
                    membership_anchor_id=curriculum.anchors[best_anchor].anchor_id,
                    capability_id=candidates[0].capability_id if status == AssignmentStatus.ASSIGNED else None,
                    margin=margin,
                    candidates=candidates,
                )
            )
        rows.append(assignments)
    return rows
