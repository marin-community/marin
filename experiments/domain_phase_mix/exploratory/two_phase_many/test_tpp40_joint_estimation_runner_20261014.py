# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from experiments.domain_phase_mix.exploratory.two_phase_many import run_tpp40_joint_estimation_round3_20261014 as runner


def test_common_absolute_floor_recovers_exponential_law_and_fold_errors() -> None:
    matrix = np.linspace(-2, 2, 30)[:, None]
    observed = 0.6 + np.exp(-0.7 + 0.8 * matrix[:, 0])
    labels = np.arange(len(observed)) % 3
    with threadpool_limits(limits=1):
        result = runner.search_absolute_floor(matrix, observed, labels, 0.0)
    assert result["floor"] == pytest.approx(0.6, abs=2e-7)
    assert result["inner_rmse"] < 1e-8
    selected = min(
        (trial for trial in result["trace"] if trial["inner_rmse"] is not None),
        key=lambda trial: (trial["inner_rmse"], trial["floor"]),
    )
    for fold in selected["folds"]:
        train, held = labels != fold["fold_index"], labels == fold["fold_index"]
        design = np.column_stack([np.ones(np.sum(train)), matrix[train, 0]])
        intercept, coefficient = np.linalg.lstsq(design, np.log(observed[train] - result["floor"]), rcond=None)[0]
        prediction = result["floor"] + np.exp(intercept + coefficient * matrix[held, 0])
        assert fold["validation_sse"] == pytest.approx(np.sum((prediction - observed[held]) ** 2), abs=1e-21)
    assert any(not fold["admissible"] for trial in result["trace"] for fold in trial["folds"])
    assert all(
        fold["validation_sse"] is None
        for trial in result["trace"]
        if trial["inner_rmse"] is None
        for fold in trial["folds"]
    )
