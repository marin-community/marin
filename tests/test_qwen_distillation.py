# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.qwen_distillation import (
    BATCH_SIZE,
    EXTENDED_ARMS,
    EXTENDED_STEPS,
    EXTENDED_TOKENS,
    SEQ_LEN,
    ZERO_SHOT_ACCURACY_TOLERANCES,
    ZERO_SHOT_TASKS,
    Arm,
    extended_checkpoint,
    screen_checkpoint_subpath,
)


@pytest.mark.parametrize(
    ("arm", "expected"),
    [
        (Arm.CE_SCRATCH, "model"),
        (Arm.FOUR_B_TEACHER, "model/student"),
    ],
)
def test_screen_checkpoint_subpath(arm: Arm, expected: str):
    assert screen_checkpoint_subpath(arm) == expected


def test_winogrande_uses_namespaced_dataset():
    winogrande = next(task for task in ZERO_SHOT_TASKS if task.task == "winogrande")
    assert winogrande.dataset_path == "allenai/winogrande"


def test_zero_shot_tolerances_cover_every_task():
    assert set(ZERO_SHOT_ACCURACY_TOLERANCES) == {task.task for task in ZERO_SHOT_TASKS}


def test_extended_stage_uses_shared_token_cap_and_promoted_arm():
    assert EXTENDED_STEPS * SEQ_LEN * BATCH_SIZE >= EXTENDED_TOKENS
    assert (EXTENDED_STEPS - 1) * SEQ_LEN * BATCH_SIZE < EXTENDED_TOKENS
    assert EXTENDED_ARMS == (
        Arm.CE_SCRATCH,
        Arm.KL_SCRATCH,
        Arm.CE_BASE,
        Arm.KL_BASE,
        Arm.FACTORIZED,
    )


def test_extended_checkpoint_uses_frozen_training_version():
    checkpoint = extended_checkpoint(Arm.FACTORIZED, 1)
    assert checkpoint.version == "2026.07.26.17"
