# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evidence rubric for deciding whether a curriculum unit is ready to use."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UnitEvidence:
    unit_id: str
    observable_outcome: int
    distinct_boundary: int
    generation_feasible: int
    reviewed_examples: int
    source_count: int
    generated_valid: int
    generated_total: int
    solve_successes: int
    solve_attempts: int
    difficulty_ordering_plausible: bool


@dataclass(frozen=True)
class UnitRubricResult:
    unit_id: str
    scores: dict[str, int]
    total: int
    fatal_flaws: tuple[str, ...]
    ready: bool


def _review_score(value: int, field: str) -> int:
    if value not in {0, 1, 2}:
        raise ValueError(f"{field} must be 0, 1, or 2")
    return value


def evaluate_unit(evidence: UnitEvidence) -> UnitRubricResult:
    """Score one unit on six criteria and apply readiness gates."""

    observable = _review_score(evidence.observable_outcome, "observable_outcome")
    boundary = _review_score(evidence.distinct_boundary, "distinct_boundary")
    generation_feasible = _review_score(evidence.generation_feasible, "generation_feasible")
    if evidence.reviewed_examples >= 3 and evidence.source_count >= 2:
        support = 2
    elif evidence.reviewed_examples >= 2 and evidence.source_count >= 1:
        support = 1
    else:
        support = 0

    if evidence.generated_total <= 0 or evidence.generated_valid < 0:
        generation_validity = 0
    else:
        validity = evidence.generated_valid / evidence.generated_total
        generation_validity = 2 if validity >= 0.95 else 1 if validity >= 0.8 else 0

    if evidence.solve_attempts <= 0 or not 0 <= evidence.solve_successes <= evidence.solve_attempts:
        difficulty_gradient = 0
    elif evidence.solve_successes in {0, evidence.solve_attempts}:
        difficulty_gradient = 0
    elif evidence.difficulty_ordering_plausible:
        difficulty_gradient = 2
    else:
        difficulty_gradient = 1

    scores = {
        "observable_outcome": observable,
        "distinct_boundary": boundary,
        "evidence_support": support,
        "generation_feasible": generation_feasible,
        "generation_validity": generation_validity,
        "difficulty_gradient": difficulty_gradient,
    }
    fatal_flaws = []
    if observable == 0:
        fatal_flaws.append("outcome is not observable")
    if boundary == 0:
        fatal_flaws.append("unit has no usable boundary")
    if generation_feasible == 0:
        fatal_flaws.append("unit cannot generate verifiable tasks")
    if evidence.generated_total > 0 and evidence.generated_valid == 0:
        fatal_flaws.append("unit produced no valid generated task")

    total = sum(scores.values())
    ready = not fatal_flaws and total >= 10 and support == 2 and generation_validity == 2
    return UnitRubricResult(
        unit_id=evidence.unit_id,
        scores=scores,
        total=total,
        fatal_flaws=tuple(fatal_flaws),
        ready=ready,
    )
