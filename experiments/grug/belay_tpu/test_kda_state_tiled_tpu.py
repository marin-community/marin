# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Interpret the tiled state operator on CPU; no model or optimizer steps."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.grug.belay_tpu.kda_state_tiled_tpu import state_pass_tiled
from experiments.grug.moe.kda import _parallel_chunk_recurrence
from experiments.grug.moe.kda_prep_pallas import ChunkPrep


def source(q, a, k, w, v, decay, initial, *, dtype):
    prep = ChunkPrep(q, w, v, a, k, decay[..., 0, :])

    def mm(spec, left, right):
        return jnp.einsum(spec, left.astype(dtype), right.astype(dtype)).astype(jnp.float32)

    return _parallel_chunk_recurrence(prep, initial, mm, q.shape[:1], 128, 128)


def evaluate(function, inputs, cotangents):
    out, backward = jax.vjp(function, *inputs)
    return (*out, *backward(cotangents))


@pytest.mark.parametrize("dtype,tolerance", [(jnp.float32, 1e-4), (jnp.bfloat16, 2e-2)])
@pytest.mark.parametrize("tile", [1, 2, 4])
def test_tiled_state_and_all_gradients_match_source(dtype, tolerance, tile):
    rng = np.random.RandomState(131)
    shape = (2, 8, 16, 128)
    q, k, w, v = (jnp.asarray(0.1 * rng.normal(size=shape), jnp.float32) for _ in range(4))
    a = jnp.tril(jnp.asarray(0.1 * rng.normal(size=(2, 8, 16, 16)), jnp.float32))
    decay = jnp.asarray(rng.uniform(0.5, 1.0, size=(2, 8, 1, 128)), jnp.float32)
    # A zero-decay chunk separates streams and checks carried-state initialization.
    decay = decay.at[:, 3].set(0)
    initial = jnp.asarray(0.1 * rng.normal(size=(2, 128, 128)), jnp.float32)
    inputs = (q, a, k, w, v, decay, initial)
    cotangents = (
        jnp.asarray(rng.normal(size=shape), jnp.float32),
        jnp.asarray(rng.normal(size=initial.shape), jnp.float32),
    )
    reference = functools.partial(source, dtype=dtype)
    candidate = functools.partial(state_pass_tiled, matmul_dtype=dtype, tile=tile, interpret=True)
    with jax.default_matmul_precision("highest"):
        expected = jax.jit(functools.partial(evaluate, reference))(inputs, cotangents)
        actual = jax.jit(functools.partial(evaluate, candidate))(inputs, cotangents)
    for got, want in zip(actual, expected, strict=True):
        np.testing.assert_allclose(got, want, atol=tolerance, rtol=tolerance)
