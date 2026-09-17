# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Embedding-only hierarchical curriculum assignment."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from experiments.post_training.task_curriculum.catalog import CurriculumUnit

Vector = Sequence[float]


class AssignmentStatus(StrEnum):
    ASSIGNED = "assigned"
    AMBIGUOUS = "ambiguous"
    COVERAGE_GAP = "coverage_gap"
    OUTSIDE_INVENTORY = "outside_inventory"


@dataclass(frozen=True)
class Candidate:
    id: str
    distance: float


@dataclass(frozen=True)
class AssignmentThresholds:
    max_macro_distance: float
    min_macro_margin: float
    max_unit_distance: float
    min_unit_margin: float


@dataclass(frozen=True)
class CurriculumAssignment:
    status: AssignmentStatus
    macro_area_id: str | None
    unit_id: str | None
    macro_candidates: tuple[Candidate, ...]
    unit_candidates: tuple[Candidate, ...]


def cosine_distance(left: Vector, right: Vector) -> float:
    """Return cosine distance for two nonzero vectors of the same length."""

    if len(left) != len(right) or not left:
        raise ValueError("vectors must have the same nonzero length")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        raise ValueError("vectors must be nonzero")
    similarity = sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)
    return 1.0 - max(-1.0, min(1.0, similarity))


def rank_anchors(task_vector: Vector, anchors: Mapping[str, Sequence[Vector]]) -> tuple[Candidate, ...]:
    """Rank labels by their nearest anchor vector."""

    candidates = []
    for label, vectors in anchors.items():
        if not vectors:
            raise ValueError(f"anchor set {label} is empty")
        candidates.append(Candidate(label, min(cosine_distance(task_vector, vector) for vector in vectors)))
    return tuple(sorted(candidates, key=lambda candidate: (candidate.distance, candidate.id)))


def _margin(candidates: Sequence[Candidate]) -> float:
    if len(candidates) < 2:
        return math.inf
    return candidates[1].distance - candidates[0].distance


def assign_hierarchically(
    macro_vector: Vector,
    unit_vector: Vector,
    macro_anchors: Mapping[str, Sequence[Vector]],
    unit_anchors: Mapping[str, Sequence[Vector]],
    units_by_id: Mapping[str, CurriculumUnit],
    thresholds: AssignmentThresholds,
) -> CurriculumAssignment:
    """Assign a task through macro inventory, local coverage, and unit boundaries."""

    macro_candidates = rank_anchors(macro_vector, macro_anchors)
    if not macro_candidates:
        raise ValueError("macro anchors are required")
    nearest_macro = macro_candidates[0]
    if nearest_macro.distance > thresholds.max_macro_distance:
        return CurriculumAssignment(
            AssignmentStatus.OUTSIDE_INVENTORY,
            None,
            None,
            macro_candidates,
            (),
        )
    if _margin(macro_candidates) < thresholds.min_macro_margin:
        return CurriculumAssignment(
            AssignmentStatus.AMBIGUOUS,
            None,
            None,
            macro_candidates,
            (),
        )

    macro_units = {
        unit_id: unit_anchors[unit_id]
        for unit_id, unit in units_by_id.items()
        if unit.macro_area_id == nearest_macro.id and unit_id in unit_anchors
    }
    if not macro_units:
        return CurriculumAssignment(
            AssignmentStatus.COVERAGE_GAP,
            nearest_macro.id,
            None,
            macro_candidates,
            (),
        )

    unit_candidates = rank_anchors(unit_vector, macro_units)
    nearest_unit = unit_candidates[0]
    if nearest_unit.distance > thresholds.max_unit_distance:
        return CurriculumAssignment(
            AssignmentStatus.COVERAGE_GAP,
            nearest_macro.id,
            None,
            macro_candidates,
            unit_candidates,
        )
    if _margin(unit_candidates) < thresholds.min_unit_margin:
        return CurriculumAssignment(
            AssignmentStatus.AMBIGUOUS,
            nearest_macro.id,
            None,
            macro_candidates,
            unit_candidates,
        )
    return CurriculumAssignment(
        AssignmentStatus.ASSIGNED,
        nearest_macro.id,
        nearest_unit.id,
        macro_candidates,
        unit_candidates,
    )
