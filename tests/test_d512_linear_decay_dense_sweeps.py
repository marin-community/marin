# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.grug.dense_one_layer_muonh.launch import (
    CONSTANT_LR_EXPERIMENT as MUONH_CONSTANT_EXPERIMENT,
)
from experiments.grug.dense_one_layer_muonh.launch import (
    SWEEP_POINTS as MUONH_CONSTANT_POINTS,
)
from experiments.grug.dense_one_layer_muonh.launch import TRAIN_RESOURCES as MUONH_TRAIN_RESOURCES
from experiments.grug.dense_one_layer_muonh.launch import (
    dense_muonh_optimizer,
)
from experiments.grug.dense_one_layer_muonh.launch_linear_decay import (
    EXPERIMENT as MUONH_LINEAR_EXPERIMENT,
)
from experiments.grug.dense_one_layer_muonh.launch_linear_decay import (
    SWEEP_POINTS as MUONH_LINEAR_POINTS,
)
from experiments.grug.dense_one_layer_sgdh.launch import (
    CONSTANT_LR_EXPERIMENT as SGDH_CONSTANT_EXPERIMENT,
)
from experiments.grug.dense_one_layer_sgdh.launch import (
    SWEEP_POINTS as SGDH_CONSTANT_POINTS,
)
from experiments.grug.dense_one_layer_sgdh.launch import TRAIN_RESOURCES as SGDH_TRAIN_RESOURCES
from experiments.grug.dense_one_layer_sgdh.launch import (
    dense_sgdh_optimizer,
)
from experiments.grug.dense_one_layer_sgdh.launch_linear_decay import (
    EXPERIMENT as SGDH_LINEAR_EXPERIMENT,
)
from experiments.grug.dense_one_layer_sgdh.launch_linear_decay import (
    SWEEP_POINTS as SGDH_LINEAR_POINTS,
)
from experiments.grug.dense_one_layer_sgdmh.launch import EXPERIMENT as SGDMH_CONSTANT_EXPERIMENT
from experiments.grug.dense_one_layer_sgdmh.launch import SWEEP_POINTS as SGDMH_CONSTANT_POINTS
from experiments.grug.dense_one_layer_sgdmh.launch_linear_decay import EXPERIMENT as SGDMH_LINEAR_EXPERIMENT
from experiments.grug.dense_one_layer_sgdmh.launch_linear_decay import SWEEP_POINTS as SGDMH_LINEAR_POINTS


@pytest.mark.parametrize(
    "constant_points,linear_points,constant_experiment,linear_experiment",
    [
        (MUONH_CONSTANT_POINTS, MUONH_LINEAR_POINTS, MUONH_CONSTANT_EXPERIMENT, MUONH_LINEAR_EXPERIMENT),
        (SGDH_CONSTANT_POINTS, SGDH_LINEAR_POINTS, SGDH_CONSTANT_EXPERIMENT, SGDH_LINEAR_EXPERIMENT),
        (SGDMH_CONSTANT_POINTS, SGDMH_LINEAR_POINTS, SGDMH_CONSTANT_EXPERIMENT, SGDMH_LINEAR_EXPERIMENT),
    ],
)
def test_linear_decay_dense_sweep_matches_grid_without_reusing_identity(
    constant_points, linear_points, constant_experiment, linear_experiment
):
    constant_grid = {(point.token_multiple, point.lr_multiplier, point.num_train_steps) for point in constant_points}
    linear_grid = {(point.token_multiple, point.lr_multiplier, point.num_train_steps) for point in linear_points}

    assert linear_grid == constant_grid
    assert len(linear_points) == 25
    assert {point.run_id for point in linear_points}.isdisjoint(point.run_id for point in constant_points)
    assert linear_experiment.wandb_group != constant_experiment.wandb_group


@pytest.mark.parametrize(
    "optimizer_factory,point,experiment",
    [
        (dense_muonh_optimizer, MUONH_LINEAR_POINTS[0], MUONH_LINEAR_EXPERIMENT),
        (dense_sgdh_optimizer, SGDH_LINEAR_POINTS[0], SGDH_LINEAR_EXPERIMENT),
        (dense_sgdh_optimizer, SGDMH_LINEAR_POINTS[0], SGDMH_LINEAR_EXPERIMENT),
    ],
)
def test_linear_decay_dense_schedule_warms_up_then_reaches_five_percent_floor(optimizer_factory, point, experiment):
    optimizer = optimizer_factory(point, experiment)
    schedule = optimizer.lr_scheduler(point.num_train_steps)
    warmup_steps = int(optimizer.warmup * point.num_train_steps)

    assert float(schedule(0)) == pytest.approx(0.0)
    assert float(schedule(warmup_steps)) == pytest.approx(optimizer.learning_rate)
    assert float(schedule(point.num_train_steps)) == pytest.approx(optimizer.learning_rate * optimizer.min_lr_ratio)


def test_dense_sweeps_fit_current_v4_host_memory_limit():
    assert MUONH_TRAIN_RESOURCES.ram == "190g"
    assert SGDH_TRAIN_RESOURCES.ram == "190g"
