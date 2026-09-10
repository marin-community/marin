# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from scipy.integrate import quad

from experiments.datakit.mixprior.metric import expected_score


def metric_loss_density(normal, mean, deviation, column):
    metric = np.clip(mean + deviation * normal, -10, 10)
    loss = max(metric, 0) + (metric if column == 0 else 0)
    return loss * np.exp(-(normal**2) / 2) / np.sqrt(2 * np.pi)


def test_expected_acquisition_matches_integrated_clipped_losses():
    means = np.array([[mean, -mean / 2] for mean in [-15.0, 0.0, 15.0] for _ in range(2)])
    sd = np.tile([[0.2, 0.4], [4.0, 3.0]], (3, 1))
    actual = expected_score(means, sd**2, np.array([1.0, 0.0]), np.array([1.0, 1.0]))
    reference = []
    for row, deviations in zip(means, sd, strict=True):
        expected = 0.0
        for column, (mean, deviation) in enumerate(zip(row, deviations, strict=True)):
            points = [(threshold - mean) / deviation for threshold in [-10, 0, 10]]
            points = [point for point in points if -12 < point < 12]

            expected -= quad(metric_loss_density, -12, 12, args=(mean, deviation, column), points=points, epsabs=1e-10)[
                0
            ]
        reference.append(expected)
    np.testing.assert_allclose(actual, reference, atol=1e-9, rtol=1e-9)
