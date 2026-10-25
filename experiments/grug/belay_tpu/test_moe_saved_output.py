# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Analytical routing derivatives without collectives or optimizer updates."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.grug.belay_tpu.moe_saved_output import routing_cotangents


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("case", ["full", "dropped", "zero_weight", "inactive_nan"])
def test_saved_returns_produce_the_weighted_sum_derivatives(dtype, case):
    rng = np.random.RandomState(311)
    tokens, topk, width = 7, 3, 16
    returned = jnp.asarray(rng.normal(size=(tokens * topk, width)), dtype)
    weights = jnp.asarray(rng.uniform(0.1, 1, size=(tokens, topk)), dtype)
    cotangent = jnp.asarray(rng.normal(size=(tokens, width)), dtype)
    indices = rng.permutation(tokens * topk).astype(np.int32)
    accepted = np.ones((tokens, topk), bool)
    if case in ("dropped", "inactive_nan"):
        accepted[::2, ::2] = False
        weights = jnp.where(accepted, weights, 0)
    if case == "zero_weight":
        weights = weights.at[1, 2].set(0)
    if case == "inactive_nan":
        returned = jnp.where(accepted.reshape(-1)[indices, None], returned, jnp.nan)

    dx, dw = jax.jit(routing_cotangents)(returned, weights, indices, accepted, cotangent)
    values = np.asarray(returned, np.float32)
    w = np.asarray(weights, np.float32)
    dy = np.asarray(cotangent, np.float32)
    expected_dx = np.empty_like(values)
    expected_dw = np.zeros((tokens, topk), np.float32)
    for row, assignment in enumerate(indices):
        token, slot = divmod(int(assignment), topk)
        expected_dx[row] = w[token, slot] * dy[token]
        if accepted[token, slot] and w[token, slot] != 0:
            expected_dw[token, slot] = np.dot(values[row], dy[token])
    expected_dx = np.asarray(jnp.asarray(expected_dx, dtype), np.float32)
    expected_dw = np.asarray(jnp.asarray(expected_dw, dtype), np.float32)
    tolerance = 1e-5 if dtype == jnp.float32 else 2e-2
    np.testing.assert_allclose(np.asarray(dx, np.float32), expected_dx, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(np.asarray(dw, np.float32), expected_dw, rtol=tolerance, atol=tolerance)
    assert np.isfinite(np.asarray(dw, np.float32)).all()
