# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import pytest

from experiments.post_training.bfcl_rl.preferences import (
    PairDisposition,
    RolloutOutcome,
    VerifiedRollout,
    select_training_pairs,
)


def test_preferences_follow_verifier_results_and_exclude_ties_and_unscored_rollouts():
    outcomes = (
        (RolloutOutcome.CORRECT, RolloutOutcome.INCORRECT),
        (RolloutOutcome.INCORRECT, RolloutOutcome.CORRECT),
        (RolloutOutcome.CORRECT, RolloutOutcome.CORRECT),
        (RolloutOutcome.INCORRECT, RolloutOutcome.INCORRECT),
        (RolloutOutcome.CORRECT, RolloutOutcome.UNSCORED),
    )
    teachers = [
        VerifiedRollout(f"simple_python_{index}", "task-hash", "pi", 0, "teacher-revision", outcome, f"teacher/{index}")
        for index, (outcome, _) in enumerate(outcomes)
    ]
    students = [
        replace(teacher, model_revision="student-revision", outcome=outcome, trajectory_uri=f"student/{index}")
        for index, (teacher, (_, outcome)) in enumerate(zip(teachers, outcomes, strict=True))
    ]
    selections = select_training_pairs(
        teachers,
        list(reversed(students)),
        complement_source_ids=frozenset(teacher.task_source_id for teacher in teachers),
        parity_source_ids=frozenset({"simple_python_holdout"}),
    )
    assert [selection.disposition for selection in selections] == [
        PairDisposition.PREFERENCE,
        PairDisposition.PREFERENCE,
        PairDisposition.BOTH_CORRECT,
        PairDisposition.BOTH_INCORRECT,
        PairDisposition.UNSCORED,
    ]
    pairs = [selection.pair for selection in selections if selection.pair is not None]
    assert [(pair.chosen.trajectory_uri, pair.rejected.trajectory_uri) for pair in pairs] == [
        ("teacher/0", "student/0"),
        ("student/1", "teacher/1"),
    ]


def test_parity_rollout_cannot_enter_training_even_if_teacher_is_correct():
    teacher = VerifiedRollout("simple_python_12", "task-hash", "pi", 0, "teacher", RolloutOutcome.CORRECT, "teacher/0")
    student = replace(teacher, model_revision="student", outcome=RolloutOutcome.INCORRECT, trajectory_uri="student/0")
    with pytest.raises(ValueError, match="outside the BFCL training complement"):
        select_training_pairs(
            [teacher],
            [student],
            complement_source_ids=frozenset({"simple_python_13"}),
            parity_source_ids=frozenset({"simple_python_12"}),
        )


def test_pairing_rejects_changed_task_content_and_missing_model_rollouts():
    teacher = VerifiedRollout("simple_python_13", "task-hash", "pi", 0, "teacher", RolloutOutcome.CORRECT, "teacher/0")
    student = replace(teacher, model_revision="student", task_digest="changed", outcome=RolloutOutcome.INCORRECT)
    settings = {"complement_source_ids": frozenset({teacher.task_source_id}), "parity_source_ids": frozenset()}
    with pytest.raises(ValueError, match="same pinned task"):
        select_training_pairs([teacher], [student], **settings)
    with pytest.raises(ValueError, match="missing counterparts"):
        select_training_pairs([teacher], [], **settings)
