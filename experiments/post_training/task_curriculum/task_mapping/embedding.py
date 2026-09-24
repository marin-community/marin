# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Vectorized task-to-curriculum assignment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from experiments.post_training.task_curriculum.models import CurriculumCatalog, RoutingFacet
from experiments.post_training.task_curriculum.task_mapping.models import (
    MAX_COSINE_SIMILARITY,
    MIN_COSINE_SIMILARITY,
    AnchorKind,
    AssignmentAnchor,
    GraphMapping,
    MappingCandidate,
    TaskAnnotation,
    TaskMapping,
)


@dataclass(frozen=True)
class GraphAnchor:
    subject_id: str
    facet: RoutingFacet
    text: str


@dataclass(frozen=True)
class SectionAnchor:
    subject_id: str
    section_id: str
    text: str


@dataclass(frozen=True)
class MappingInputs:
    annotations: Sequence[TaskAnnotation]
    membership_vectors: Mapping[RoutingFacet, np.ndarray]
    operation_vectors: np.ndarray
    catalog: CurriculumCatalog
    graph_anchor_rows: Sequence[GraphAnchor]
    graph_anchor_vectors: np.ndarray
    section_anchor_rows: Sequence[SectionAnchor]
    section_anchor_vectors: np.ndarray
    embedding_model: str
    top_k: int
    row_batch_size: int


def graph_anchors(
    catalog: CurriculumCatalog,
    assignment_anchors: Sequence[AssignmentAnchor] = (),
) -> list[GraphAnchor]:
    """Return membership anchors for each curriculum graph."""
    anchors: list[GraphAnchor] = []
    for entry in catalog.curricula:
        curriculum = entry.curriculum
        roots = [section for section in curriculum.sections if section.parent_id is None]
        for root in roots:
            anchors.append(
                GraphAnchor(
                    subject_id=curriculum.subject_id,
                    facet=entry.routing_facet,
                    text=f"{curriculum.subject_name}. {root.name}. {root.scope_text()} "
                    f"Includes: {'; '.join(root.includes)}.",
                )
            )
    facets_by_subject = {entry.curriculum.subject_id: entry.routing_facet for entry in catalog.curricula}
    for anchor in assignment_anchors:
        if anchor.kind != AnchorKind.GRAPH:
            continue
        if anchor.subject_id not in facets_by_subject:
            raise ValueError(f"unknown graph anchor subject {anchor.subject_id}")
        anchors.append(GraphAnchor(anchor.subject_id, facets_by_subject[anchor.subject_id], anchor.text))
    return anchors


def section_anchors(
    catalog: CurriculumCatalog,
    assignment_anchors: Sequence[AssignmentAnchor] = (),
) -> list[SectionAnchor]:
    """Return section anchors that exclude task-generation probes."""
    anchors: list[SectionAnchor] = []
    for entry in catalog.curricula:
        curriculum = entry.curriculum
        for section in curriculum.capability_sections():
            base = f"{section.name}. {section.outcome}"
            for included in section.includes:
                anchors.append(
                    SectionAnchor(
                        subject_id=curriculum.subject_id,
                        section_id=section.id,
                        text=f"{base}. Includes: {included}.",
                    )
                )
    subject_by_section = {
        section.id: entry.curriculum.subject_id
        for entry in catalog.curricula
        for section in entry.curriculum.capability_sections()
    }
    for anchor in assignment_anchors:
        if anchor.kind != AnchorKind.SECTION:
            continue
        if anchor.section_id not in subject_by_section:
            raise ValueError(f"unknown section anchor target {anchor.section_id}")
        expected_subject = subject_by_section[anchor.section_id]
        if anchor.subject_id != expected_subject:
            raise ValueError(f"section {anchor.section_id} belongs to {expected_subject}, not {anchor.subject_id}")
        anchors.append(SectionAnchor(anchor.subject_id, anchor.section_id, anchor.text))
    return anchors


def normalized(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize a matrix of row vectors."""
    values = np.asarray(vectors, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"expected a matrix, got shape {values.shape}")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("cannot normalize zero-length embedding")
    return values / norms


def _maximum_by_group(
    similarities: np.ndarray,
    anchor_groups: Sequence[str],
) -> tuple[list[str], np.ndarray]:
    groups = list(dict.fromkeys(anchor_groups))
    anchor_ids = np.asarray(anchor_groups)
    scores = np.stack(
        [similarities[:, anchor_ids == group].max(axis=1) for group in groups],
        axis=1,
    )
    return groups, scores


def _grouped_cosine_scores(
    task_vectors: np.ndarray,
    anchor_vectors: np.ndarray,
    anchor_groups: Sequence[str],
) -> tuple[list[str], np.ndarray]:
    similarities = normalized(task_vectors) @ normalized(anchor_vectors).T
    return _maximum_by_group(similarities, anchor_groups)


def _validate_mapping_inputs(inputs: MappingInputs) -> None:
    task_count = len(inputs.annotations)
    if len(inputs.operation_vectors) != task_count:
        raise ValueError("task annotation and operation-vector counts differ")
    if any(len(vectors) != task_count for vectors in inputs.membership_vectors.values()):
        raise ValueError("task annotation and membership-vector counts differ")
    if set(inputs.membership_vectors) != set(RoutingFacet):
        raise ValueError("membership vectors must cover every routing facet")
    task_ids = [annotation.task_id for annotation in inputs.annotations]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("task annotation IDs must be unique")
    if inputs.top_k < 1:
        raise ValueError("top_k must be positive")
    if inputs.row_batch_size < 1:
        raise ValueError("row_batch_size must be positive")
    if len(inputs.graph_anchor_rows) != len(inputs.graph_anchor_vectors):
        raise ValueError("graph anchor metadata and vector counts differ")
    if len(inputs.section_anchor_rows) != len(inputs.section_anchor_vectors):
        raise ValueError("section anchor metadata and vector counts differ")


def _graph_membership_scores(
    membership_vectors: Mapping[RoutingFacet, np.ndarray],
    anchor_rows: Sequence[GraphAnchor],
    anchor_vectors: np.ndarray,
) -> dict[RoutingFacet, tuple[list[str], np.ndarray]]:
    scores: dict[RoutingFacet, tuple[list[str], np.ndarray]] = {}
    for facet in {anchor.facet for anchor in anchor_rows}:
        indices = [index for index, anchor in enumerate(anchor_rows) if anchor.facet == facet]
        scores[facet] = _grouped_cosine_scores(
            membership_vectors[facet],
            anchor_vectors[indices],
            [anchor_rows[index].subject_id for index in indices],
        )
    return scores


def _rank_graph_sections(
    subject_id: str,
    routing_facet: RoutingFacet,
    task_index: int,
    graph_scores: Mapping[RoutingFacet, tuple[list[str], np.ndarray]],
    section_scores: np.ndarray,
    section_positions: Mapping[str, list[int]],
    section_ids: Sequence[str],
    top_k: int,
) -> GraphMapping:
    graph_subjects, membership_scores = graph_scores[routing_facet]
    graph_index = graph_subjects.index(subject_id)
    positions = section_positions[subject_id]
    if top_k > len(positions):
        raise ValueError(f"top_k exceeds section count for {subject_id}")
    winners = sorted(positions, key=lambda position: section_scores[position], reverse=True)[:top_k]
    return GraphMapping(
        subject_id=subject_id,
        routing_facet=routing_facet,
        membership_similarity=float(
            np.clip(membership_scores[task_index, graph_index], MIN_COSINE_SIMILARITY, MAX_COSINE_SIMILARITY)
        ),
        candidates=[
            MappingCandidate(
                section_id=section_ids[position],
                similarity=float(np.clip(section_scores[position], MIN_COSINE_SIMILARITY, MAX_COSINE_SIMILARITY)),
            )
            for position in winners
        ],
    )


def map_task_vectors(inputs: MappingInputs) -> list[TaskMapping]:
    """Rank graph membership and sections independently for each curriculum."""
    _validate_mapping_inputs(inputs)
    task_count = len(inputs.annotations)
    graph_scores = _graph_membership_scores(
        inputs.membership_vectors,
        inputs.graph_anchor_rows,
        inputs.graph_anchor_vectors,
    )
    task_operations = normalized(inputs.operation_vectors)
    section_values = normalized(inputs.section_anchor_vectors)
    anchor_section_ids = [anchor.section_id for anchor in inputs.section_anchor_rows]
    unique_section_ids = list(dict.fromkeys(anchor_section_ids))
    subject_by_section = {anchor.section_id: anchor.subject_id for anchor in inputs.section_anchor_rows}
    section_positions = {
        entry.curriculum.subject_id: [
            index
            for index, section_id in enumerate(unique_section_ids)
            if subject_by_section[section_id] == entry.curriculum.subject_id
        ]
        for entry in inputs.catalog.curricula
    }

    mappings: list[TaskMapping] = []
    for start in range(0, task_count, inputs.row_batch_size):
        stop = min(start + inputs.row_batch_size, task_count)
        anchor_scores = task_operations[start:stop] @ section_values.T
        batch_section_ids, section_scores = _maximum_by_group(anchor_scores, anchor_section_ids)
        assert batch_section_ids == unique_section_ids
        for row, annotation in enumerate(inputs.annotations[start:stop]):
            graph_mappings = [
                _rank_graph_sections(
                    entry.curriculum.subject_id,
                    entry.routing_facet,
                    start + row,
                    graph_scores,
                    section_scores[row],
                    section_positions,
                    unique_section_ids,
                    inputs.top_k,
                )
                for entry in inputs.catalog.curricula
            ]
            graph_mappings.sort(key=lambda graph: graph.membership_similarity, reverse=True)
            mappings.append(
                TaskMapping(
                    task_id=annotation.task_id,
                    task_hash=annotation.task_hash,
                    annotation_model=annotation.model,
                    annotation_prompt_version=annotation.prompt_version,
                    catalog_version=inputs.catalog.catalog_version,
                    embedding_model=inputs.embedding_model,
                    graphs=graph_mappings,
                )
            )
    return mappings
