# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.qwen_distillation import Arm, screen_checkpoint_subpath


@pytest.mark.parametrize(
    ("arm", "expected"),
    [
        (Arm.CE_SCRATCH, "model"),
        (Arm.FOUR_B_TEACHER, "model/student"),
    ],
)
def test_screen_checkpoint_subpath(arm: Arm, expected: str):
    assert screen_checkpoint_subpath(arm) == expected
