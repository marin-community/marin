# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Numerical operator tests only; no model training or optimizer updates."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from levanter.grug.attention import AttentionMask, reference_attention

from experiments.grug.belay_tpu.inkling_tpu import saved_attention_local


def attention_reference(q, k, v, band, qids, kids):
    return reference_attention(
        q[:, :, None],
        k[:, :, None],
        v[:, :, None],
        AttentionMask(is_causal=True, segment_ids=(qids, kids)),
        logits_dtype=jnp.float32,
        rel_bias=band[:, None],
    )[:, :, 0]


def attention_values_grads(fn, inputs, cotangent):
    out, backward = jax.vjp(lambda *xs: fn(*xs, *inputs[4:]), *inputs[:4])
    return (out, *backward(cotangent))


@pytest.mark.parametrize("length,width", [(128, 256), (256, 256), (256, 1152)])
@pytest.mark.parametrize("dtype,tolerance", [(jnp.float32, 1e-5), (jnp.bfloat16, 2e-2)])
def test_saved_attention_outputs_and_gradients(length, width, dtype, tolerance):
    rng = np.random.RandomState(61)
    q, k, v = (jnp.asarray(rng.randn(2, length, 128), dtype) for _ in range(3))
    band = jnp.asarray(0.2 * rng.randn(2, length, width), dtype)
    ids = jnp.asarray(np.broadcast_to(np.arange(length) // 37, (2, length)), jnp.int32)
    inputs = (q, k, v, band, ids, ids)
    cotangent = jnp.asarray(rng.randn(*q.shape), dtype)
    with jax.default_matmul_precision("highest"):
        expected = jax.jit(functools.partial(attention_values_grads, attention_reference))(inputs, cotangent)
        actual = jax.jit(functools.partial(attention_values_grads, saved_attention_local))(inputs, cotangent)
    for got, want in zip(actual, expected, strict=True):
        np.testing.assert_allclose(
            np.asarray(got, np.float32), np.asarray(want, np.float32), atol=tolerance, rtol=tolerance
        )


def test_saved_attention_preserves_bf16_gradient_rounding():
    rng = np.random.RandomState(71)
    shape = (2, 256, 128)
    q, k = (jnp.asarray(4 * rng.randn(*shape), jnp.bfloat16) for _ in range(2))
    v = jnp.asarray(rng.randn(*shape), jnp.bfloat16)
    band = jnp.asarray(0.2 * rng.randn(2, 256, 1152), jnp.bfloat16)
    ids = jnp.asarray(np.broadcast_to(np.arange(256) // 37, (2, 256)), jnp.int32)
    cotangent = jnp.asarray(rng.randn(*shape), jnp.bfloat16)
    inputs = (q, k, v, band, ids, ids)
    with jax.softmax_custom_jvp(True), jax.default_matmul_precision("highest"):
        expected = jax.jit(functools.partial(attention_values_grads, attention_reference))(inputs, cotangent)
        actual = jax.jit(functools.partial(attention_values_grads, saved_attention_local))(inputs, cotangent)
    for got, want in zip(actual, expected, strict=True):
        np.testing.assert_allclose(np.asarray(got, np.float32), np.asarray(want, np.float32), atol=2e-2, rtol=2e-2)
