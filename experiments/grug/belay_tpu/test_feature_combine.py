# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check weighted expert output reduction without optimizer updates."""

import jax.numpy as jnp
import numpy as np
import pytest

from experiments.grug.belay_tpu.feature_combine_tpu import combine_with_transpose


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("topk", [3, 10])
def test_feature_major_combine_matches_independent_weighted_sum(dtype, topk):
    rng = np.random.default_rng(422)
    tokens, features = 128, 16
    order = rng.permutation(tokens * topk).astype(np.int32)
    inverse = np.argsort(order).astype(np.int32)
    rows = jnp.asarray(rng.normal(size=(tokens * topk, features)), dtype).at[-31:].set(0)
    weights = jnp.asarray(rng.random((tokens, topk)), jnp.float32).at[:, 0].set(0)
    actual = combine_with_transpose(rows, jnp.asarray(order), jnp.asarray(inverse), weights, interpret=True)
    expected = np.zeros((tokens, features), np.float64)
    host_rows, host_weights = np.asarray(rows, np.float64), np.asarray(weights, np.float64)
    for token in range(tokens):
        for route in range(topk):
            expected[token] += host_rows[inverse[token * topk + route]] * host_weights[token, route]
    tolerance = 2e-2 if dtype == jnp.bfloat16 else 1e-5
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(np.asarray(actual, np.float32), expected, rtol=0, atol=tolerance * np.abs(expected).max())
