# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Non-training checks of larger KDA chunks against the token recurrence."""

import functools
import json
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.grug.belay_tpu.kda_chunk_prep import chunk_prep_midpoint
from experiments.grug.belay_tpu.kernel_check import evaluate, measure_errors
from experiments.grug.moe.kda import chunk_kda, recurrent_kda


@pytest.mark.parametrize("chunk_size", [32, 64, 128])
@pytest.mark.parametrize("case", ["packed", "strong_packed", "strong_long", "weak_long"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_midpoint_recurrence_and_gradients(chunk_size, case, dtype, request):
    if dtype == jnp.bfloat16 and case == "weak_long":
        request.applymarker(
            pytest.mark.xfail(
                strict=True,
                reason="Known BF16 weak-decay gradient drift from C16; full-update qualification pending",
            )
        )
    rng = np.random.default_rng(426)
    shape = (1, 2, 129, 8)
    q, k, v = (jnp.asarray(rng.normal(size=shape), dtype) for _ in range(3))
    scale = 5.0 if case.startswith("strong") else 0.001 if case.startswith("weak") else 0.1
    g = jnp.asarray(-scale * rng.random(shape), jnp.float32)
    beta = jnp.asarray(rng.random(shape[:-1]), jnp.float32)
    initial = jnp.asarray(0.01 * rng.normal(size=(*shape[:2], 8, 8)), jnp.float32)
    erase = jnp.asarray(0.5 + rng.random(shape), jnp.float32)
    ids = np.searchsorted([7, 16, 17, 33, 56, 65, 95, 110, 127], np.arange(shape[-2]), side="right")
    segments = jnp.asarray(ids[None, None, :], jnp.int32) if case.endswith("packed") else None
    inputs = (q, k, v, g, beta, initial, erase)
    cotangents = (
        jnp.asarray(rng.normal(size=shape), jnp.float32),
        jnp.asarray(rng.normal(size=initial.shape), jnp.float32),
    )
    reference = (
        recurrent_kda if dtype == jnp.float32 else functools.partial(chunk_kda, chunk_size=16, matmul_dtype=dtype)
    )
    expected = jax.jit(functools.partial(evaluate, reference))(inputs, segments, cotangents)
    candidate = functools.partial(chunk_kda, chunk_size=chunk_size, matmul_dtype=dtype)
    with patch("experiments.grug.moe.kda._chunk_prep_xla", chunk_prep_midpoint):
        compiled = jax.jit(functools.partial(evaluate, candidate)).lower(inputs, segments, cotangents).compile()
    actual = compiled(inputs, segments, cotangents)
    deviations = measure_errors(actual, expected, 1e-4 if dtype == jnp.float32 else 2e-2)
    assert all(np.isfinite(row["normalized_max"]) and row["normalized_max"] <= 1 for row in deviations), json.dumps(
        deviations
    )
