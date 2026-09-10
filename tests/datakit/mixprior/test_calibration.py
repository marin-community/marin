# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import numpy as np
from scipy.stats import norm

from experiments.datakit.mixprior.calibration import new_design_indices, residual_calibration


def test_residual_calibration_removes_bias_and_covers_unseen_noisy_outcomes():
    rng = np.random.default_rng(37)
    mean = rng.normal(size=(2000, 2))
    variance = np.full_like(mean, 0.75)
    noise = np.full(2, 0.25)
    bias = np.array([-3.0, 1.5])
    measured = mean + bias + rng.normal(scale=2.0, size=mean.shape)
    correction = residual_calibration(mean, variance, measured, noise, np.ones(len(mean)))
    test_mean = rng.normal(size=(10000, 2))
    test_measured = test_mean + bias + rng.normal(scale=2.0, size=test_mean.shape)
    corrected = test_mean + correction.bias
    z = (test_measured - corrected) / np.sqrt(correction.variance_multiplier)
    np.testing.assert_allclose((test_measured - corrected).mean(0), 0, atol=0.12)
    coverage = (np.abs(z) < norm.ppf(0.975)).mean(0)
    assert np.all((coverage > 0.935) & (coverage < 0.965))


def test_new_design_indices_excludes_new_replicates_of_previous_designs(data):
    previous = replace(data, weights=data.weights[:5])
    current = replace(data, weights=np.concatenate([data.weights, data.weights[:2]]))
    np.testing.assert_array_equal(new_design_indices(current, previous), np.arange(5, 20))


def test_calibration_matches_grouped_observation_variance_with_replicates():
    count = 100
    residual = np.tile([-1.0, 1.0], count // 2)[:, None]
    residual *= np.sqrt(5.0 * (count - 1) / (count + 1))
    correction = residual_calibration(
        np.zeros_like(residual), np.ones_like(residual), residual, np.ones(1), np.full(count, 4.0)
    )
    corrected_latent_variance = correction.variance_multiplier * 2 - 1
    np.testing.assert_allclose(corrected_latent_variance + 1 / 4, 5.0, rtol=1e-12)
