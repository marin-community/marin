# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare saved and recomputed expert pullbacks without optimizer updates."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from levanter.grug._moe.ep_ragged_all_to_all import _RaggedDotExpertMlp, _SavedRaggedDotExpertMlp


def evaluate(expert_mlp, inputs, cotangent):
    out, residuals = expert_mlp.forward(*inputs)
    return out, expert_mlp.backward(residuals, cotangent)


def relu2(x):
    return jnp.square(jax.nn.relu(x))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("active_sizes", [(7, 0, 11), (0, 0, 0), (0, 32, 0)])
def test_saved_expert_pullback_matches_source_with_inactive_rows(dtype, active_sizes):
    rng = np.random.RandomState(161)
    active = jnp.asarray(active_sizes, jnp.int32)
    physical = active.at[-1].add(32 - active.sum())
    x = jnp.asarray(rng.normal(size=(32, 16)), dtype)
    x = x.at[sum(active_sizes) :].set(jnp.nan)
    up = jnp.asarray(rng.normal(size=(3, 16, 32)), dtype)
    down = jnp.asarray(rng.normal(size=(3, 32, 16)), dtype)
    inputs = (x, up, down, physical, active)
    cotangent = jnp.asarray(rng.normal(size=(32, 16)), dtype)
    source = _RaggedDotExpertMlp(relu2)
    saved = _SavedRaggedDotExpertMlp(relu2)
    expected = jax.jit(functools.partial(evaluate, source))(inputs, cotangent)
    actual = jax.jit(functools.partial(evaluate, saved))(inputs, cotangent)
    for got, want in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_array_equal(got, want)
