# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import math

import numpy as np
from scipy.spatial.distance import euclidean

from experiments.datakit.mixprior.model import fit, kernel


def test_matern_kernel_matches_scalar_formula():
    x = np.array([[0.0, 1.0], [0.3, 0.7], [1.0, 0.0]])
    expected = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            radius = math.sqrt(5) * euclidean(x[i], x[j]) / 0.7
            expected[i, j] = 0.1 * (1 + radius + radius**2 / 3) * math.exp(-radius)
    np.testing.assert_allclose(kernel(x, x, 0.7), expected, rtol=1e-12, atol=1e-12)


def test_gp_conditions_on_observations_and_is_uncertain_away_from_them(data):
    y = np.sin(12 * data.weights[:, 0, 0]) - np.cos(8 * data.weights[:, 1, 2])
    gp = fit(data, y, np.full(len(y), 1e-10))
    mean, variance = gp.predict(data.weights)
    np.testing.assert_allclose(mean, y, atol=0.003)
    far = np.array([[[1, 0, 0], [0, 0, 1]]], dtype=float)
    _, far_variance = gp.predict(far)
    assert far_variance[0] > variance.max()
    assert np.all(variance >= 0)
