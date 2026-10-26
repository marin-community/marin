# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Non-training numerical and causality checks for saved KDA solves."""

import functools
import json
from unittest.mock import patch

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import pytest

from experiments.grug.belay_tpu.kda_triangular_tpu import chunk_prep_solve, solve_unit_lower
from experiments.grug.belay_tpu.kernel_check import evaluate, measure_errors
from experiments.grug.moe.kda import chunk_kda, recurrent_kda


@pytest.mark.parametrize("chunk", [16, 64])
@pytest.mark.parametrize("repeated", [False, True])
def test_saved_solve_values_and_gradients(chunk, repeated):
    rng = np.random.default_rng(428)
    a = rng.normal(size=(5, chunk, chunk)).astype(np.float32) * 0.04
    if repeated:
        a = np.broadcast_to(-rng.uniform(0.8, 1.0, (5, chunk, 1)), a.shape).astype(np.float32)
    a = jnp.asarray(a)
    rhs = jnp.asarray(rng.normal(size=(5, chunk, 128)), jnp.float32)
    dy = jnp.asarray(rng.normal(size=rhs.shape), jnp.float32)

    def reference(a, rhs):
        return jsp.linalg.solve_triangular(jnp.eye(chunk) - jnp.tril(a, -1), rhs, lower=True, unit_diagonal=True)

    def with_gradients(fn, a, rhs):
        y, backward = jax.vjp(fn, a, rhs)
        return (y, *backward(dy))

    expected = jax.jit(functools.partial(with_gradients, reference))(a, rhs)
    candidate = functools.partial(solve_unit_lower, dtype=jnp.float32, interpret=True)
    actual = jax.jit(functools.partial(with_gradients, candidate))(a, rhs)
    deviations = measure_errors(actual, expected, 1e-4)
    assert all(row["normalized_max"] <= 1 for row in deviations), json.dumps(deviations)


@pytest.mark.parametrize("chunk", [16, 32, 64])
@pytest.mark.parametrize("case", ["packed", "strong_packed", "weak_long"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.timeout(180)
def test_saved_solve_kda_recurrence_and_gradients(chunk, case, dtype, request):
    if dtype == jnp.bfloat16 and case == "weak_long" and chunk > 16:
        request.applymarker(
            pytest.mark.xfail(
                strict=True, reason="Recorded BF16 weak-decay gradient drift; TPU loss qualification pending"
            )
        )
    rng = np.random.default_rng(428)
    length = 57 if chunk == 16 else 129
    shape = (1, 2, length, 128)
    q, k, v = (jnp.asarray(rng.normal(size=shape), dtype) for _ in range(3))
    scale = 5.0 if case.startswith("strong") else 0.001 if case.startswith("weak") else 0.1
    g = jnp.asarray(-scale * rng.random(shape), jnp.float32)
    beta = jnp.asarray(rng.random(shape[:-1]), jnp.float32)
    initial = jnp.asarray(0.01 * rng.normal(size=(1, 2, 128, 128)), jnp.float32)
    erase = jnp.asarray(0.5 + rng.random(shape), jnp.float32)
    ids = np.searchsorted([7, 16, 17, 33, 56, 65, 95, 110, 127], np.arange(length), side="right")
    segments = jnp.asarray(ids[None, None, :], jnp.int32) if case.endswith("packed") else None
    inputs = (q, k, v, g, beta, initial, erase)
    cotangents = (
        jnp.asarray(rng.normal(size=shape), jnp.float32),
        jnp.asarray(rng.normal(size=initial.shape), jnp.float32),
    )
    reference = (
        recurrent_kda if dtype == jnp.float32 else functools.partial(chunk_kda, chunk_size=16, matmul_dtype=dtype)
    )
    candidate = functools.partial(chunk_kda, chunk_size=chunk, matmul_dtype=dtype)
    with jax.default_matmul_precision("highest"):
        expected = jax.jit(functools.partial(evaluate, reference))(inputs, segments, cotangents)
        with patch("experiments.grug.moe.kda._chunk_prep_xla", functools.partial(chunk_prep_solve, interpret=True)):
            function = jax.jit(functools.partial(evaluate, candidate)).lower(inputs, segments, cotangents).compile()
        actual = function(inputs, segments, cotangents)
    deviations = measure_errors(actual, expected, 1e-4 if dtype == jnp.float32 else 2e-2)
    assert all(row["normalized_max"] <= 1 for row in deviations), json.dumps(deviations)


@pytest.mark.parametrize("chunk", [32, 64])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.timeout(180)
def test_saved_solve_preserves_prefix_and_document_independence(chunk, dtype):
    rng = np.random.default_rng(429)
    shape = (1, 2, 129, 128)
    q, k, v = (jnp.asarray(rng.normal(size=shape), dtype) for _ in range(3))
    g = jnp.asarray(-5 * rng.random(shape), jnp.float32)
    beta = jnp.asarray(rng.random(shape[:-1]), jnp.float32)
    ids = np.searchsorted([7, 17, 33, 65, 95], np.arange(129), side="right")
    segments = jnp.asarray(ids[None, None, :], jnp.int32)

    def forward(q, k, v, g, beta):
        return chunk_kda(q, k, v, g, beta, chunk_size=chunk, matmul_dtype=dtype, segment_ids=segments)[0]

    def derivative(q, k, v, g, beta):
        return jax.grad(lambda g: jnp.sum(forward(q, k, v, g, beta)[..., :7, :]))(g)

    inputs = (q, k, v, g, beta)
    with jax.default_matmul_precision("highest"):
        with patch("experiments.grug.moe.kda._chunk_prep_xla", functools.partial(chunk_prep_solve, interpret=True)):
            function = jax.jit(forward).lower(*inputs).compile()
            gradient = jax.jit(derivative).lower(*inputs).compile()
    expected = np.asarray(function(*inputs))
    future = function(q, k, v.at[..., 7:, :].add(2), g.at[..., 7:, :].set(-0.01), beta)
    previous_doc = function(q, k, v.at[..., :7, :].add(2), g.at[..., :7, :].set(-0.01), beta)
    np.testing.assert_array_equal(expected[..., :7, :], np.asarray(future)[..., :7, :])
    np.testing.assert_array_equal(expected[..., 7:, :], np.asarray(previous_doc)[..., 7:, :])
    np.testing.assert_array_equal(np.asarray(gradient(*inputs))[..., 7:, :], 0)
