# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Numerical coverage for the QuACK epilogue API used by expert training and Muon."""

import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _require_sm100():
    if jax.default_backend() != "gpu":
        pytest.skip("QuACK expert GEMMs require an SM100 GPU")
    if float(jax.devices("gpu")[0].compute_capability) < 10.0:
        pytest.skip("QuACK expert GEMMs require SM100")
    pytest.importorskip("quack")


def _assert_bfloat16_close(actual, expected):
    actual = np.asarray(actual, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    scale = max(float(np.max(np.abs(expected))), 1e-6)
    # Independent GEMMs round bf16 intermediates differently; normalize by the reference
    # magnitude, as in the grouped-wgrad tests, while checking every output element.
    np.testing.assert_allclose(actual / scale, expected / scale, atol=2e-2, rtol=1e-4)


@pytest.mark.parametrize("use_clc", [False, True])
def test_gated_grouped_gemm_keeps_preact_and_interleaved_swiglu(use_clc):
    _require_sm100()
    kernels = importlib.import_module("levanter.grug._moe.quack_moe_cute")
    rng = np.random.default_rng(42)
    x = jnp.asarray(rng.normal(0, 0.2, (224, 64)), dtype=jnp.bfloat16)
    w = jnp.asarray(rng.normal(0, 0.2, (3, 64, 128)), dtype=jnp.bfloat16)
    cu = jnp.asarray([0, 97, 97, 224], dtype=jnp.int32)
    preact, postact = jax.jit(
        lambda a, b: kernels.quack_gated_grouped_gemm(a, b, cu, return_preact=True, use_clc_persistence=use_clc)
    )(x, w)
    expected = jnp.concatenate(
        [x[:97].astype(jnp.float32) @ w[0].astype(jnp.float32), x[97:].astype(jnp.float32) @ w[2].astype(jnp.float32)]
    )
    _assert_bfloat16_close(preact, expected)
    _assert_bfloat16_close(postact, jax.nn.silu(expected[:, 0::2]) * expected[:, 1::2])


def test_expert_mlp_forward_and_all_gradients_match_reference():
    _require_sm100()
    sonic = importlib.import_module("levanter.grug._moe.sonic_cute")
    rng = np.random.default_rng(7)
    x = jnp.asarray(rng.normal(0, 0.2, (224, 64)), dtype=jnp.bfloat16)
    w13 = jnp.asarray(rng.normal(0, 0.2, (3, 64, 128)), dtype=jnp.bfloat16)
    w2 = jnp.asarray(rng.normal(0, 0.2, (3, 64, 64)), dtype=jnp.bfloat16)
    dy = jnp.asarray(rng.normal(0, 0.2, (224, 64)), dtype=jnp.bfloat16)
    cu = jnp.asarray([0, 97, 97, 224], dtype=jnp.int32)

    def reference(a, b, c):
        outputs = []
        for expert, start, stop in [(0, 0, 97), (2, 97, 224)]:
            gu = a[start:stop] @ b[expert]
            h = jax.nn.silu(gu[:, 0::2]) * gu[:, 1::2]
            outputs.append(h @ c[expert])
        return jnp.concatenate(outputs)

    actual, actual_pullback = jax.vjp(jax.jit(lambda a, b, c: sonic._expert_mlp_quack_wgrad(a, b, c, cu)), x, w13, w2)
    expected, expected_pullback = jax.vjp(jax.jit(reference), x, w13, w2)
    _assert_bfloat16_close(actual, expected)
    for got, want in zip(actual_pullback(dy), expected_pullback(dy), strict=True):
        _assert_bfloat16_close(got, want)


def test_muon_symmetric_gemm_matches_gram_matrix():
    _require_sm100()
    kernels = importlib.import_module("levanter.grug._moe.quack_symmetric_cute")
    rng = np.random.default_rng(3)
    x = jnp.asarray(rng.normal(0, 0.2, (3, 256, 128)), dtype=jnp.bfloat16)
    got = jax.jit(kernels.quack_symmetric_gemm)(x)
    expected = x.astype(jnp.float32) @ jnp.swapaxes(x.astype(jnp.float32), -1, -2)
    _assert_bfloat16_close(got, expected)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(jnp.swapaxes(got, -1, -2)))
