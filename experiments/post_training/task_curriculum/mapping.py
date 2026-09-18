# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Vectorized task-to-curriculum assignment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from experiments.post_training.task_curriculum.models import (
    AnchorKind,
    AssignmentAnchor,
    CurriculumCatalog,
    GraphMapping,
    MappingCandidate,
    RoutingFacet,
    TaskAnnotation,
    TaskMapping,
)


def graph_anchors(
    catalog: CurriculumCatalog,
    assignment_anchors: Sequence[AssignmentAnchor] = (),
) -> tuple[list[str], list[RoutingFacet], list[str]]:
    """Return membership anchors for each curriculum graph."""
    subject_ids: list[str] = []
    facets: list[RoutingFacet] = []
    anchors: list[str] = []
    for entry in catalog.curricula:
        curriculum = entry.curriculum
        roots = [section for section in curriculum.sections if section.parent_id is None]
        for root in roots:
            subject_ids.append(curriculum.subject_id)
            facets.append(entry.routing_facet)
            anchors.append(
                f"{curriculum.subject_name}. {root.name}. {root.outcome} " f"Includes: {'; '.join(root.includes)}."
            )
    facets_by_subject = {entry.curriculum.subject_id: entry.routing_facet for entry in catalog.curricula}
    for anchor in assignment_anchors:
        if anchor.kind != AnchorKind.GRAPH:
            continue
        if anchor.subject_id not in facets_by_subject:
            raise ValueError(f"unknown graph anchor subject {anchor.subject_id}")
        subject_ids.append(anchor.subject_id)
        facets.append(facets_by_subject[anchor.subject_id])
        anchors.append(anchor.text)
    return subject_ids, facets, anchors


def section_anchors(
    catalog: CurriculumCatalog,
    assignment_anchors: Sequence[AssignmentAnchor] = (),
) -> tuple[list[str], list[str], list[str]]:
    """Return section anchors that exclude task-generation probes."""
    subject_ids: list[str] = []
    section_ids: list[str] = []
    anchors: list[str] = []
    for entry in catalog.curricula:
        curriculum = entry.curriculum
        for section in curriculum.sections:
            base = f"{section.name}. {section.outcome}"
            for included in section.includes:
                subject_ids.append(curriculum.subject_id)
                section_ids.append(section.id)
                anchors.append(f"{base}. Includes: {included}.")
    subject_by_section = {
        section.id: entry.curriculum.subject_id for entry in catalog.curricula for section in entry.curriculum.sections
    }
    for anchor in assignment_anchors:
        if anchor.kind != AnchorKind.SECTION:
            continue
        if anchor.section_id not in subject_by_section:
            raise ValueError(f"unknown section anchor target {anchor.section_id}")
        expected_subject = subject_by_section[anchor.section_id]
        if anchor.subject_id != expected_subject:
            raise ValueError(f"section {anchor.section_id} belongs to {expected_subject}, not {anchor.subject_id}")
        subject_ids.append(anchor.subject_id)
        section_ids.append(anchor.section_id)
        anchors.append(anchor.text)
    return subject_ids, section_ids, anchors


def normalized(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize a matrix of row vectors."""
    values = np.asarray(vectors, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"expected a matrix, got shape {values.shape}")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("cannot normalize zero-length embedding")
    return values / norms


def _maximum_scores(
    task_vectors: np.ndarray,
    anchor_vectors: np.ndarray,
    anchor_groups: Sequence[str],
) -> tuple[list[str], np.ndarray]:
    groups = list(dict.fromkeys(anchor_groups))
    anchor_ids = np.asarray(anchor_groups)
    similarities = normalized(task_vectors) @ normalized(anchor_vectors).T
    scores = np.stack(
        [similarities[:, anchor_ids == group].max(axis=1) for group in groups],
        axis=1,
    )
    return groups, scores


def map_task_vectors(
    annotations: Sequence[TaskAnnotation],
    membership_vectors: Mapping[RoutingFacet, np.ndarray],
    operation_vectors: np.ndarray,
    catalog: CurriculumCatalog,
    graph_anchor_subject_ids: Sequence[str],
    graph_anchor_facets: Sequence[RoutingFacet],
    graph_anchor_vectors: np.ndarray,
    section_anchor_subject_ids: Sequence[str],
    section_anchor_ids: Sequence[str],
    section_anchor_vectors: np.ndarray,
    embedding_model: str,
    top_k: int,
    row_batch_size: int,
) -> list[TaskMapping]:
    """Rank graph membership and sections independently for each curriculum."""
    task_count = len(annotations)
    if len(operation_vectors) != task_count:
        raise ValueError("task annotation and operation-vector counts differ")
    if any(len(vectors) != task_count for vectors in membership_vectors.values()):
        raise ValueError("task annotation and membership-vector counts differ")
    if set(membership_vectors) != set(RoutingFacet):
        raise ValueError("membership vectors must cover every routing facet")
    task_ids = [annotation.task_id for annotation in annotations]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("task annotation IDs must be unique")
    if not 1 <= top_k:
        raise ValueError("top_k must be positive")
    if row_batch_size < 1:
        raise ValueError("row_batch_size must be positive")
    if not (len(graph_anchor_subject_ids) == len(graph_anchor_facets) == len(graph_anchor_vectors)):
        raise ValueError("graph anchor metadata and vector counts differ")
    if not (len(section_anchor_subject_ids) == len(section_anchor_ids) == len(section_anchor_vectors)):
        raise ValueError("section anchor metadata and vector counts differ")

    graph_scores: dict[RoutingFacet, tuple[list[str], np.ndarray]] = {}
    for facet in RoutingFacet:
        indices = [index for index, value in enumerate(graph_anchor_facets) if value == facet]
        graph_scores[facet] = _maximum_scores(
            membership_vectors[facet],
            graph_anchor_vectors[indices],
            [graph_anchor_subject_ids[index] for index in indices],
        )

    task_operations = normalized(operation_vectors)
    section_values = normalized(section_anchor_vectors)
    section_subjects = np.asarray(section_anchor_subject_ids)
    section_ids = np.asarray(section_anchor_ids)
    unique_section_ids = list(dict.fromkeys(section_anchor_ids))
    section_positions = {
        subject_id: [
            index
            for index, section_id in enumerate(unique_section_ids)
            if section_subjects[section_ids == section_id][0] == subject_id
        ]
        for subject_id in {entry.curriculum.subject_id for entry in catalog.curricula}
    }
    entries = {entry.curriculum.subject_id: entry for entry in catalog.curricula}
    mappings: list[TaskMapping] = []
    for start in range(0, task_count, row_batch_size):
        stop = min(start + row_batch_size, task_count)
        anchor_scores = task_operations[start:stop] @ section_values.T
        section_scores = np.stack(
            [anchor_scores[:, section_ids == section_id].max(axis=1) for section_id in unique_section_ids],
            axis=1,
        )
        for row, annotation in enumerate(annotations[start:stop]):
            graph_mappings: list[GraphMapping] = []
            for subject_id, entry in entries.items():
                graph_subjects, scores = graph_scores[entry.routing_facet]
                graph_index = graph_subjects.index(subject_id)
                graph_positions = section_positions[subject_id]
                if top_k > len(graph_positions):
                    raise ValueError(f"top_k exceeds section count for {subject_id}")
                winner_positions = sorted(
                    graph_positions,
                    key=lambda position: section_scores[row, position],
                    reverse=True,
                )[:top_k]
                candidates = [
                    MappingCandidate(
                        section_id=unique_section_ids[position],
                        similarity=float(np.clip(section_scores[row, position], -1.0, 1.0)),
                    )
                    for position in winner_positions
                ]
                graph_mappings.append(
                    GraphMapping(
                        subject_id=subject_id,
                        routing_facet=entry.routing_facet,
                        membership_similarity=float(np.clip(scores[start + row, graph_index], -1.0, 1.0)),
                        candidates=candidates[:top_k],
                    )
                )
            graph_mappings.sort(key=lambda graph: graph.membership_similarity, reverse=True)
            mappings.append(
                TaskMapping(
                    task_id=annotation.task_id,
                    task_hash=annotation.task_hash,
                    annotation_model=annotation.model,
                    annotation_prompt_version=annotation.prompt_version,
                    catalog_version=catalog.catalog_version,
                    embedding_model=embedding_model,
                    graphs=graph_mappings,
                )
            )
    return mappings
