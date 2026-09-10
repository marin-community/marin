# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import numpy as np

from experiments.datakit.mixprior.objective import fit_objective


def test_objective_rewards_targets_penalizes_regressions_and_propagates_noise(data):
    calibration = replace(
        data,
        weights=np.repeat(data.weights[:2], 2, axis=0),
        outcomes=np.array([[1, 1], [3, 3], [3, 1], [5, 3]], dtype=float),
        groups=["proportional_baseline"] * 2 + ["search"] * 2,
    )
    objective = fit_objective(calibration, metrics=("reward", "guard"), targets=("reward",))
    values, variances = objective(np.array([[1, 1], [3, 3], [3, 1], [1, 3], [100, 100]], dtype=float))
    np.testing.assert_allclose(values, np.array([1, -3, -2, 0, -30 * np.sqrt(2)]) / np.sqrt(2), atol=1e-12)
    np.testing.assert_allclose(variances[:4], [1, 9, 4, 4], atol=1e-12)
    assert variances[-1] == np.finfo(float).eps
