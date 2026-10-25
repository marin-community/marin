# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check indexed projection outputs without optimizer updates."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.grug.belay_tpu.indexed_up_tpu import indexed_layout, indexed_up


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("concentrated", [False, True])
def test_indexed_up_matches_each_selected_expert(dtype, concentrated):
    rng = np.random.default_rng(421)
    tokens, topk, experts, width, hidden = 32, 2, 4, 128, 128
    selected = np.stack([rng.choice(experts, topk, replace=False) for _ in range(tokens)])
    if concentrated:
        selected[:] = [0, 3]
    indices, _, tile_experts, restore = indexed_layout(selected, experts)
    x = jnp.asarray(rng.normal(scale=0.1, size=(tokens, width)), dtype)
    weights = jnp.asarray(rng.normal(scale=0.1, size=(experts, width, hidden)), dtype)
    with jax.default_matmul_precision("highest"):
        padded = indexed_up(x, weights, jnp.asarray(indices), jnp.asarray(tile_experts), interpret=True)
        actual = padded.reshape(-1, hidden)[restore].reshape(tokens, topk, hidden)
    host_x, host_weights = np.asarray(x, np.float64), np.asarray(weights, np.float64)
    expected = np.einsum("td,tkdh->tkh", host_x, host_weights[selected])
    tolerance = 1e-4 if dtype == jnp.float32 else 2e-2
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual.astype(jnp.float32), expected, rtol=0, atol=tolerance * np.abs(expected).max())
    np.testing.assert_array_equal(np.asarray(padded)[indices < 0], 0)
