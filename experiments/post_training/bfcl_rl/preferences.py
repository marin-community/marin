# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select verifier-discriminated teacher/student trajectories for BFCL recovery."""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum


class RolloutOutcome(StrEnum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    UNSCORED = "unscored"


class PairDisposition(StrEnum):
    PREFERENCE = "preference"
    BOTH_CORRECT = "both_correct"
    BOTH_INCORRECT = "both_incorrect"
    UNSCORED = "unscored"


@dataclass(frozen=True)
class VerifiedRollout:
    task_source_id: str
    task_digest: str
    harness: str
    repetition: int
    model_revision: str
    outcome: RolloutOutcome
    trajectory_uri: str


@dataclass(frozen=True)
class PreferencePair:
    chosen: VerifiedRollout
    rejected: VerifiedRollout


@dataclass(frozen=True)
class PairSelection:
    disposition: PairDisposition
    pair: PreferencePair | None


def select_pair(teacher: VerifiedRollout, student: VerifiedRollout) -> PairSelection:
    """Prefer the sole correct trajectory; discard ties and unscored pairs."""
    teacher_key = (teacher.task_source_id, teacher.task_digest, teacher.harness, teacher.repetition)
    student_key = (student.task_source_id, student.task_digest, student.harness, student.repetition)
    if teacher_key != student_key:
        raise ValueError("teacher and student must solve the same pinned task with the same harness and repetition")
    if RolloutOutcome.UNSCORED in (teacher.outcome, student.outcome):
        return PairSelection(PairDisposition.UNSCORED, None)
    if teacher.outcome == student.outcome:
        disposition = (
            PairDisposition.BOTH_CORRECT if teacher.outcome is RolloutOutcome.CORRECT else PairDisposition.BOTH_INCORRECT
        )
        return PairSelection(disposition, None)
    chosen, rejected = (teacher, student) if teacher.outcome is RolloutOutcome.CORRECT else (student, teacher)
    return PairSelection(PairDisposition.PREFERENCE, PreferencePair(chosen, rejected))


def select_training_pairs(
    teachers: Sequence[VerifiedRollout],
    students: Sequence[VerifiedRollout],
    *,
    complement_source_ids: frozenset[str],
    parity_source_ids: frozenset[str],
) -> tuple[PairSelection, ...]:
    """Join paired rollouts while rejecting holdout leakage and missing counterparts."""
    if complement_source_ids & parity_source_ids:
        raise ValueError("training and parity source IDs overlap")

    def indexed(rollouts: Sequence[VerifiedRollout]) -> dict[tuple[str, str, int], VerifiedRollout]:
        records = {}
        for rollout in rollouts:
            if rollout.task_source_id in parity_source_ids or rollout.task_source_id not in complement_source_ids:
                raise ValueError(f"rollout is outside the BFCL training complement: {rollout.task_source_id}")
            key = (rollout.task_source_id, rollout.harness, rollout.repetition)
            if key in records:
                raise ValueError(f"duplicate paired rollout: {key}")
            records[key] = rollout
        return records

    teacher_records = indexed(teachers)
    student_records = indexed(students)
    if teacher_records.keys() != student_records.keys():
        raise ValueError("teacher and student rollout keys differ; missing counterparts cannot form preferences")
    return tuple(select_pair(teacher_records[key], student_records[key]) for key in sorted(teacher_records))
