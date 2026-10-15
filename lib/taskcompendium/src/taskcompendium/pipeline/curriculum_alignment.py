# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Threshold-independent curriculum retrieval with primary subjects and scored candidates."""

from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property

import numpy as np
from pydantic import BaseModel, ConfigDict

from taskcompendium.pipeline.task_embeddings import normalized_vectors

TOP_K = 3


class CurriculumFacet(StrEnum):
    SUBJECT_DOMAIN = "subject_domain"
    TASK_MECHANIC = "task_mechanic"


class AnchorKind(StrEnum):
    SUBJECT = "subject"
    CAPABILITY = "capability"


class CurriculumAnchor(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    anchor_id: str
    kind: AnchorKind
    subject_id: str
    capability_id: str | None
    facet: CurriculumFacet
    text: str


class CapabilityCandidate(BaseModel):
    capability_id: str
    similarity: float
    anchor_id: str


class SubjectCandidate(BaseModel):
    subject_id: str
    similarity: float
    anchor_id: str


class SubjectMatch(BaseModel):
    subject_id: str
    facet: CurriculumFacet
    membership_similarity: float
    membership_anchor_id: str
    capability_id: str
    margin: float
    candidates: list[CapabilityCandidate]
    subject_candidates: list[SubjectCandidate]


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
    facet: CurriculumFacet
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
                sorted(
                    (
                        index
                        for index, anchor in enumerate(self.anchors)
                        if anchor.subject_id == subject_id and anchor.kind == AnchorKind.SUBJECT
                    ),
                    key=lambda index: self.anchors[index].anchor_id,
                )
            )
            facets = {self.anchors[index].facet for index in membership}
            if len(facets) != 1:
                raise ValueError(f"Subject {subject_id} must declare one routing facet")
            by_capability: dict[str, list[int]] = {}
            for index, anchor in enumerate(self.anchors):
                if anchor.subject_id != subject_id or anchor.kind != AnchorKind.CAPABILITY:
                    continue
                if anchor.capability_id is None or anchor.facet not in facets:
                    raise ValueError("Capability anchors require an ID and their subject's routing facet")
                by_capability.setdefault(anchor.capability_id, []).append(index)
            if not by_capability:
                raise ValueError(f"Subject {subject_id} has no capability anchors")
            positions = []
            for capability_id in sorted(by_capability):
                positions.append(len(capabilities))
                anchors = tuple(sorted(by_capability[capability_id], key=lambda index: self.anchors[index].anchor_id))
                capabilities.append(_CapabilityReference(capability_id, anchors))
            subjects.append(_SubjectReference(subject_id, next(iter(facets)), membership, tuple(positions)))
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
    task_vectors: np.ndarray,
) -> list[list[SubjectMatch]]:
    """Score primary subjects and nearest capabilities without applying acceptance thresholds."""
    row_count = len(task_vectors)
    dimensions = curriculum.vectors.shape[1]
    tasks = normalized_vectors(task_vectors, row_count, dimensions)
    index = curriculum.index
    subjects = [subject.subject_id for subject in index.subjects]
    anchor_scores = tasks @ index.references.T
    # Membership scores only root scopes. Capability anchors reduce within each capability.
    operation_scores = anchor_scores[:, index.operation_positions]
    capability_scores = np.maximum.reduceat(operation_scores, index.capability_starts, axis=1)
    membership_scores = [anchor_scores[:, list(subject.anchor_positions)] for subject in index.subjects]
    graph_scores = np.stack([scores.max(axis=1) for scores in membership_scores], axis=1)
    operation_columns = {int(position): column for column, position in enumerate(index.operation_positions)}
    by_facet = {
        facet: [position for position, subject in enumerate(index.subjects) if subject.facet == facet]
        for facet in CurriculumFacet
    }
    rows = []
    for row in range(row_count):
        assignments = []
        for indices in by_facet.values():
            if not indices:
                continue
            nearest_subjects = [indices[position] for position in _top_positions(graph_scores[row, indices], TOP_K)]
            subject_candidates = []
            for position in nearest_subjects:
                reference = index.subjects[position]
                anchor = reference.anchor_positions[int(np.argmax(membership_scores[position][row]))]
                subject_candidates.append(
                    SubjectCandidate(
                        subject_id=reference.subject_id,
                        similarity=_cosine(graph_scores[row, position]),
                        anchor_id=curriculum.anchors[anchor].anchor_id,
                    )
                )
            subject_index = nearest_subjects[0]
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
            membership_anchors = reference.anchor_positions
            best_anchor = membership_anchors[int(np.argmax(membership_scores[subject_index][row]))]
            assignments.append(
                SubjectMatch(
                    subject_id=subject,
                    facet=reference.facet,
                    membership_similarity=member,
                    membership_anchor_id=curriculum.anchors[best_anchor].anchor_id,
                    capability_id=candidates[0].capability_id,
                    margin=margin,
                    candidates=candidates,
                    subject_candidates=subject_candidates,
                )
            )
        rows.append(assignments)
    return rows
