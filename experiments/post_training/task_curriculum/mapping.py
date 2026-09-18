# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Vectorized task-to-curriculum assignment."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from experiments.post_training.task_curriculum.models import (
    Curriculum,
    MappingCandidate,
    TaskAnnotation,
    TaskMapping,
)


def curriculum_anchors(curricula: Sequence[Curriculum]) -> tuple[list[str], list[str]]:
    """Return globally unique section IDs and model-independent anchor sentences."""
    subject_ids = [curriculum.subject_id for curriculum in curricula]
    if len(subject_ids) != len(set(subject_ids)):
        raise ValueError("subject IDs must be unique across curricula")
    section_ids: list[str] = []
    anchors: list[str] = []
    for curriculum in curricula:
        for section in curriculum.sections:
            base = f"{curriculum.subject_name}. {section.name}. {section.outcome}"
            for task in section.sample_tasks:
                section_ids.append(section.id)
                anchors.append(f"{base}. Example task: {task.instruction}")

    section_curricula = {
        section.id: curriculum.subject_id for curriculum in curricula for section in curriculum.sections
    }
    if len(section_curricula) != sum(len(curriculum.sections) for curriculum in curricula):
        raise ValueError("section IDs must be globally unique across curricula")
    return section_ids, anchors


def normalized(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize a matrix of row vectors."""
    values = np.asarray(vectors, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"expected a matrix, got shape {values.shape}")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("cannot normalize zero-length embedding")
    return values / norms


def map_task_vectors(
    annotations: Sequence[TaskAnnotation],
    task_vectors: np.ndarray,
    curricula: Sequence[Curriculum],
    anchor_section_ids: Sequence[str],
    anchor_vectors: np.ndarray,
    embedding_model: str,
    top_k: int,
    row_batch_size: int,
) -> list[TaskMapping]:
    """Rank curriculum sections by each section's closest anchor."""
    if len(annotations) != len(task_vectors):
        raise ValueError("task annotation and vector counts differ")
    task_ids = [annotation.task_id for annotation in annotations]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("task annotation IDs must be unique")
    if len(anchor_section_ids) != len(anchor_vectors):
        raise ValueError("anchor ID and vector counts differ")

    unique_sections = list(dict.fromkeys(anchor_section_ids))
    if not 1 <= top_k <= len(unique_sections):
        raise ValueError(f"top_k must be between 1 and {len(unique_sections)}")
    if row_batch_size < 1:
        raise ValueError("row_batch_size must be positive")

    task_values = normalized(task_vectors)
    anchor_values = normalized(anchor_vectors)
    anchor_ids = np.asarray(anchor_section_ids)
    mappings: list[TaskMapping] = []
    for start in range(0, len(annotations), row_batch_size):
        batch_annotations = annotations[start : start + row_batch_size]
        anchor_scores = task_values[start : start + row_batch_size] @ anchor_values.T
        section_scores = np.stack(
            [anchor_scores[:, anchor_ids == section_id].max(axis=1) for section_id in unique_sections],
            axis=1,
        )
        winner_indices = np.argsort(-section_scores, axis=1)[:, :top_k]
        for row, (annotation, indices) in enumerate(zip(batch_annotations, winner_indices, strict=True)):
            candidates = [
                MappingCandidate(
                    section_id=unique_sections[index],
                    similarity=float(np.clip(section_scores[row, index], -1.0, 1.0)),
                )
                for index in indices
            ]
            mappings.append(
                TaskMapping(
                    task_id=annotation.task_id,
                    task_hash=annotation.task_hash,
                    annotation_model=annotation.model,
                    annotation_prompt_version=annotation.prompt_version,
                    curriculum_versions={curriculum.subject_id: curriculum.version for curriculum in curricula},
                    embedding_model=embedding_model,
                    candidates=candidates,
                )
            )
    return mappings
