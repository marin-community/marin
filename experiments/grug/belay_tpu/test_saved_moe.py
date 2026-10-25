# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare saved and recomputed expert pullbacks without optimizer updates."""

import functools
from itertools import pairwise

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from levanter.grug._moe.ep_ragged_all_to_all import _RaggedDotExpertMlp, _SavedRaggedDotExpertMlp

from experiments.grug.belay_tpu.expert_gradient_tpu import expert_weight_gradient
from experiments.grug.belay_tpu.saved_relu2_adapter import SavedRelu2ExpertMlp


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


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("active_sizes", [(1, 0, 95), (0, 0, 0), (0, 128, 0)])
def test_fused_saved_expert_masks_padding_and_returns_router_gradient(dtype, active_sizes):
    rng = np.random.default_rng(171)
    active = jnp.asarray(active_sizes, jnp.int32)
    physical = active.at[-1].add(128 - active.sum())
    x = jnp.asarray(rng.normal(size=(128, 128)), dtype)
    x = x.at[sum(active_sizes) :].set(jnp.nan)
    up = jnp.asarray(rng.normal(scale=0.1, size=(3, 128, 128)), dtype)
    down = jnp.asarray(rng.normal(scale=0.1, size=(3, 128, 128)), dtype)
    cotangent = jnp.asarray(rng.normal(size=(128, 128)), dtype)
    inputs = x, up, down, physical, active
    expected = jax.jit(functools.partial(evaluate, _RaggedDotExpertMlp(relu2)))(inputs, cotangent)
    # Padding may be uninitialized in native transport, including its cotangent.
    cotangent = cotangent.at[sum(active_sizes) :].set(jnp.nan)
    actual = jax.jit(functools.partial(evaluate, SavedRelu2ExpertMlp(interpret=True)))(inputs, cotangent)
    tolerance = 2e-2 if dtype == jnp.bfloat16 else 2e-5
    for got, want in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        got, want = np.asarray(got, np.float32), np.asarray(want, np.float32)
        assert np.isfinite(got).all()
        np.testing.assert_allclose(got, want, rtol=0, atol=tolerance * max(float(np.abs(want).max()), 1e-6))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("sizes", [(1, 0, 31, 96), (0, 0, 64, 64), (128, 0, 0, 0)])
def test_partitioned_expert_weight_gradients_match_independent_products(dtype, sizes):
    rng = np.random.default_rng(417)
    x = jnp.asarray(rng.normal(scale=0.1, size=(128, 128)), dtype)
    dy = jnp.asarray(rng.normal(scale=0.1, size=(128, 128)), dtype)
    groups = jnp.asarray(sizes, jnp.int32)
    with jax.default_matmul_precision("highest"):
        actual = expert_weight_gradient(x, dy, groups, tile_rows=128, interpret=True)
    host_x, host_dy = np.asarray(x, np.float64), np.asarray(dy, np.float64)
    boundaries = np.cumsum([0, *sizes])
    expected = np.stack([host_x[lo:hi].T @ host_dy[lo:hi] for lo, hi in pairwise(boundaries)])
    tolerance = 1e-4 if dtype == jnp.float32 else 2e-2
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
