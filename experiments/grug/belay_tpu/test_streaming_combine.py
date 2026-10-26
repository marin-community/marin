# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check route sums and custom pullbacks against direct grouped outputs."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.grug.belay_tpu.local_experts import combine_local
from experiments.grug.belay_tpu.streaming_combine import combine_streaming, sum_routes


@pytest.mark.parametrize("implementation", ["unrolled", "scan", "flat"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("empty", [False, True])
def test_streaming_values_and_pullbacks(implementation, dtype, empty):
    rng = np.random.default_rng(424)
    tokens, topk, width = 31, 3, 16
    order = jnp.asarray(rng.permutation(tokens * topk), jnp.int32)
    inverse = jnp.argsort(order)
    rows = jnp.asarray(rng.normal(size=(tokens * topk, width)), dtype)
    rows = rows.at[0 if empty else 73 :].set(0)
    weights = jnp.asarray(rng.random((tokens, topk)), jnp.float32).at[:, 0].set(0)
    gradient = jnp.asarray(rng.normal(size=(tokens, width)), dtype)

    def evaluate(combine):
        output, backward = jax.vjp(lambda x, w: combine(x, order, inverse, w), rows, weights)
        return output, backward(gradient)

    expected = evaluate(combine_local)
    actual = evaluate(functools.partial(combine_streaming, implementation=implementation))
    tolerance = 1e-4 if dtype == jnp.float32 else 2e-2
    for got, want in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        assert np.isfinite(got).all()
        np.testing.assert_allclose(
            np.asarray(got, np.float32), np.asarray(want, np.float32), atol=tolerance, rtol=tolerance
        )
    # Dispatch's pullback is the unweighted route sum, including rejected rows.
    expected_sum = np.asarray(rows, np.float32)[np.asarray(inverse)].reshape(tokens, topk, width).sum(axis=1)
    observed = sum_routes(rows, inverse.reshape(tokens, topk), implementation=implementation)
    np.testing.assert_allclose(np.asarray(observed, np.float32), expected_sum, atol=tolerance, rtol=tolerance)
