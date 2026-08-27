# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.grug.moe_hero_fsdp_constant_lr_tpu.launch import D512_STEPS, constant_lr_optimizer
from experiments.grug.moe_hero_fsdp_constant_lr_tpu.launch_lower_lr import (
    D512_NEW_LOW_LR_POINTS,
    LOW_LR_MULTIPLIERS,
    NEW_LOW_LR_MULTIPLIERS,
    select_d512_lower_lr_points,
)


def test_lower_lr_sweep_reuses_completed_point_seven_cells():
    expected = {
        (token_multiple, lr_multiplier, D512_STEPS[token_multiple])
        for token_multiple in (30, 60, 150, 300)
        for lr_multiplier in NEW_LOW_LR_MULTIPLIERS
    }
    expected.update((600, lr_multiplier, D512_STEPS[600]) for lr_multiplier in LOW_LR_MULTIPLIERS)

    assert len(D512_NEW_LOW_LR_POINTS) == 21
    assert len({point.experiment_id for point in D512_NEW_LOW_LR_POINTS}) == 21
    assert len({point.run_id for point in D512_NEW_LOW_LR_POINTS}) == 21
    assert {(point.token_multiple, point.lr_multiplier, point.num_train_steps) for point in D512_NEW_LOW_LR_POINTS} == expected


def test_lower_lr_sweep_brackets_the_shared_fit():
    assert LOW_LR_MULTIPLIERS == (0.1, 0.2, 0.32, 0.45, 0.7)
    assert min(LOW_LR_MULTIPLIERS) < 0.20
    assert 0.32 in LOW_LR_MULTIPLIERS
    assert max(LOW_LR_MULTIPLIERS) == 0.7


def test_lower_lr_optimizer_scales_the_matched_july_peak():
    point = select_d512_lower_lr_points(token_multiples=(300,), lr_multipliers=(0.32,))[0]
    optimizer = constant_lr_optimizer(point)

    assert optimizer.learning_rate == pytest.approx(0.0115098911435345 * 0.32, rel=1e-12)
    assert optimizer.lr_schedule == "constant"
    assert optimizer.warmup == 0.01
