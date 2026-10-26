# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check grouped products across row-partition and expert boundaries."""

from itertools import pairwise

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.grug.belay_tpu.row_partitioned_gmm import row_partitioned_gmm


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("groups", [(63, 0, 271, 178), (0, 512, 0, 0)])
def test_row_partitions_preserve_experts_spanning_the_core_boundary(dtype, transpose, groups):
    rng = np.random.default_rng(423)
    x = jnp.asarray(rng.normal(scale=0.1, size=(512, 128)), dtype)
    weights = jnp.asarray(rng.normal(scale=0.1, size=(4, 128, 256)), dtype)
    argument = weights.transpose(0, 2, 1) if transpose else weights
    with jax.default_matmul_precision("highest"):
        actual = row_partitioned_gmm(
            x, argument, jnp.asarray(groups, jnp.int32), tile_rows=128, transpose_rhs=transpose, interpret=True
        )
    host_x, host_weights = np.asarray(x, np.float64), np.asarray(weights, np.float64)
    bounds = np.cumsum([0, *groups])
    expected = np.concatenate([host_x[lo:hi] @ host_weights[i] for i, (lo, hi) in enumerate(pairwise(bounds))])
    tolerance = 2e-2 if dtype == jnp.bfloat16 else 1e-4
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(np.asarray(actual, np.float32), expected, rtol=0, atol=tolerance * np.abs(expected).max())
