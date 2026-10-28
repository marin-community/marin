# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures: a synthetic swarm drawn from a known MARINER model, and the local mariner-data snapshot."""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mariner.model import Anchor, Head, Objective, ObjectiveFit, Shape, Swarm, TaskFit

SYNTHETIC_SEED = 20261028
EPOCHS_PER_UNIT_WEIGHT = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
SAMPLED_RUNS = 160  # at 80, t1's floor search ends flat and fresh mixtures leave the swarm's coverage
TRUE_SHAPES = (Shape(0.5, 1.0, 2.0), Shape(1.0, 0.7, 3.0), Shape(0.25, 1.0, 1.0))
TRUE_INTERCEPT = math.log(0.3)
TRUE_PHI = 0.8
TASK_WEIGHTS = (0.5, 0.3, 0.2)
NOISE_SD = 0.002


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--mariner-data", default=None, help="local mariner-data snapshot read by data_integration tests")


@pytest.fixture(scope="session")
def snapshot(request: pytest.FixtureRequest) -> Path:
    root = request.config.getoption("--mariner-data")
    if root is None:
        pytest.fail("pass --mariner-data <snapshot dir>; see lib/mariner/README.md")
    return Path(root)


@pytest.fixture
def synthetic_problem() -> tuple[Swarm, Objective, dict[str, Anchor], ObjectiveFit]:
    """A swarm whose outcomes follow a known three-task MARINER model plus Gaussian noise.

    Six buckets; row 0 is the proportional mixture and the calibration row, rows 1..160 are Dirichlet draws, and every
    pool fraction is one. Returns the swarm, its objective, anchors at the model's value for the proportional mixture
    (repeat SD the noise SD) and the generating model.
    """
    rng = np.random.default_rng(SYNTHETIC_SEED)
    epochs = np.asarray(EPOCHS_PER_UNIT_WEIGHT)
    buckets = tuple(f"b{index}" for index in range(len(epochs)))
    proportional = (1 / epochs) / np.sum(1 / epochs)
    weights = np.vstack([proportional, rng.dirichlet(np.ones(len(epochs)), size=SAMPLED_RUNS)])
    task_names = tuple(f"t{index}" for index in range(len(TRUE_SHAPES)))
    tasks = []
    for task, shape in zip(task_names, TRUE_SHAPES, strict=True):
        benefit_amplitudes = rng.uniform(0.2, 1.0, len(epochs))
        harm_amplitudes = rng.uniform(0.0, 0.3, len(epochs))
        head = Head(TRUE_INTERCEPT, np.concatenate([benefit_amplitudes, harm_amplitudes]), TRUE_PHI)
        # gamma, nu and the inner-CV RMSE describe a fit; a generating model has none.
        tasks.append(TaskFit(task, shape, math.nan, math.nan, False, math.nan, head))
    truth = ObjectiveFit("synthetic", buckets, epochs, np.asarray(TASK_WEIGHTS), tuple(tasks), {})
    clean = truth.predict_tasks(weights)
    outcomes = pd.DataFrame(clean + rng.normal(0.0, NOISE_SD, size=clean.shape), columns=list(task_names))
    swarm = Swarm(
        runs=tuple(f"run_{row:03d}" for row in range(len(weights))),
        buckets=buckets,
        weights=weights,
        epochs_per_unit_weight=epochs,
        pool_fractions=np.ones_like(weights),
        outcomes=outcomes,
        calibration=np.arange(len(weights)) == 0,
    )
    objective = Objective("synthetic", task_names, np.asarray(TASK_WEIGHTS))
    at_proportional = truth.predict_tasks(proportional)[0]
    anchors = {task: Anchor(float(value), NOISE_SD) for task, value in zip(task_names, at_proportional, strict=True)}
    return swarm, objective, anchors, truth
