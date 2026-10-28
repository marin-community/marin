# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare causal-bucket values and all derivatives against dense attention."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.grug.belay_tpu.attention_kernel_check import dense, evaluate
from experiments.grug.belay_tpu.causal_attention import causal_attention_local
from experiments.grug.belay_tpu.kernel_check import measure_errors


@pytest.mark.parametrize("dtype,tolerance", [(jnp.float32, 1e-4), (jnp.bfloat16, 2e-2)])
@pytest.mark.parametrize(
    "length,bucket,width,scenario",
    [
        (128, 512, 256, "full"),
        (512, 128, 256, "full"),
        (256, 128, 256, "packed"),
        (384, 256, 512, "last_bucket"),
        (384, 128, 1152, "strong"),
        (256, 128, 256, "zero_gradient"),
        (768, 512, 1152, "repeated_ids"),
    ],
)
def test_causal_attention_values_and_derivatives(length, bucket, width, scenario, dtype, tolerance):
    rng = np.random.default_rng(452)
    shape = (1, length, 128)
    scale = 4 if scenario == "strong" else 1
    q, k = (jnp.asarray(scale * rng.standard_normal(shape), dtype) for _ in range(2))
    v = jnp.asarray(rng.standard_normal(shape), dtype)
    band = jnp.asarray(0.2 * rng.standard_normal((1, length, width)), dtype)
    ids = np.arange(length)[None, :] // 37
    if scenario == "full":
        ids[:] = 0
    elif scenario == "repeated_ids":
        ids %= 3
    ids = jnp.asarray(ids, jnp.int32)
    cotangent = jnp.asarray(rng.standard_normal(shape), dtype)
    if scenario == "zero_gradient":
        cotangent = jnp.zeros_like(cotangent)
    inputs = q, k, v, band

    def candidate(q, k, v, band, ids):
        return causal_attention_local(q, k, v, band, ids, ids, bucket, True)

    # The frozen TPU run disables excess precision. Default CPU cast folding
    # otherwise changes saturated BF16 logits in both the old and new kernels.
    options = {"xla_allow_excess_precision": False}
    expected = (
        jax.jit(functools.partial(evaluate, dense))
        .lower(inputs, ids, cotangent)
        .compile(compiler_options=options)(inputs, ids, cotangent)
    )
    actual = (
        jax.jit(functools.partial(evaluate, candidate))
        .lower(inputs, ids, cotangent)
        .compile(compiler_options=options)(inputs, ids, cotangent)
    )
    errors = measure_errors(actual, expected, tolerance)
    assert all(np.isfinite(row["normalized_max"]) and row["normalized_max"] <= 1 for row in errors), errors
