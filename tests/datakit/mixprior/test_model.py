# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import math

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial.distance import cdist, euclidean

from experiments.datakit.mixprior.model import kernel, squared_distances


def test_distances_match_reference_with_repeated_profiles_and_partial_batches():
    rng = np.random.default_rng(18)
    x = rng.normal(size=(35, 7))
    y = np.concatenate([x[:3], rng.normal(size=(34, 7))])
    actual = jax.jit(squared_distances)(jnp.asarray(x), jnp.asarray(y))
    np.testing.assert_allclose(actual, cdist(x, y, metric="sqeuclidean"), rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(np.diag(actual[:3, :3]), 0)


def test_matern_kernel_matches_scalar_formula():
    x = np.array([[0.0, 1.0], [0.3, 0.7], [1.0, 0.0]])
    expected = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            radius = math.sqrt(5) * euclidean(x[i], x[j]) / 0.7
            expected[i, j] = 0.1 * (1 + radius + radius**2 / 3) * math.exp(-radius)
    np.testing.assert_allclose(kernel(x, x, 0.7), expected, rtol=1e-12, atol=1e-12)
