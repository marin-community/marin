# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh

from levanter.grug.mxfp8 import MX_BLOCK, mx_batched, mx_dense, mx_quantize_dequantize


@pytest.fixture(autouse=True)
def _mesh():
    mesh = Mesh(np.array(jax.devices()[:1]).reshape((1,)), ("data",), axis_types=(AxisType.Explicit,))
    with jax.set_mesh(mesh):
        yield


def _e4m3_reference(x: np.ndarray) -> np.ndarray:
    """MXFP8 along the last axis in numpy: e8m0 scale = 2^ceil(log2(amax / 448)) per 32-block, e4m3 RNE."""
    blocks = x.astype(np.float32).reshape(*x.shape[:-1], -1, MX_BLOCK)
    amax = np.abs(blocks).max(-1, keepdims=True)
    scale = np.where(amax > 0, np.exp2(np.ceil(np.log2(amax / 448.0))), 1.0)
    q = (blocks / scale).astype(ml_dtypes.float8_e4m3fn).astype(np.float32)
    return (q * scale).reshape(x.shape)


def test_quantize_matches_numpy_reference_under_jit_and_keeps_blocks_independent():
    x = np.array(jax.random.normal(jax.random.PRNGKey(0), (4, 128), jnp.float32))
    x[:, 5] *= 1e4  # an outlier in the first block of each row
    got = np.asarray(jax.jit(lambda a: mx_quantize_dequantize(a, -1))(jnp.asarray(x)))
    np.testing.assert_array_equal(got, _e4m3_reference(x))
    # The rounding really happens (XLA must not fold the f8 round trip away) ...
    assert np.abs(got - x).max() > 0
    # ... and the outlier only coarsens its own block: other blocks keep e4m3's ~2^-4 relative precision.
    rel = np.abs(got[:, MX_BLOCK:] - x[:, MX_BLOCK:]) / np.abs(x[:, MX_BLOCK:])
    assert rel.max() <= 2.0**-4 + 1e-6


def test_scales_are_exact_powers_of_two_so_rounding_ties_stay_ties():
    """bf16 inputs over a power-of-two scale often land exactly on an e4m3 rounding tie; an inexact scale
    (GPU exp2/log2) would tip every such tie the same way. Small-magnitude weights hit this most."""
    x = jax.random.normal(jax.random.PRNGKey(4), (64, 256), jnp.float32).astype(jnp.bfloat16) * 0.02
    got = np.asarray(mx_quantize_dequantize(x, -1).astype(jnp.float32))
    np.testing.assert_array_equal(got, _e4m3_reference(np.asarray(x.astype(jnp.float32))))


def test_quantize_along_a_middle_axis_with_padding():
    x = jax.random.normal(jax.random.PRNGKey(1), (3, 40, 8), jnp.float32)
    got = np.asarray(mx_quantize_dequantize(x, 1))
    moved = np.moveaxis(np.asarray(x), 1, -1)
    padded = np.concatenate([moved, np.zeros((*moved.shape[:-1], 24), np.float32)], axis=-1)
    expected = np.moveaxis(_e4m3_reference(padded)[..., :40], -1, 1)
    np.testing.assert_array_equal(got, expected)


def test_dense_gradients_quantize_each_gemm_along_its_own_contraction_axis():
    kx, kw, kg = jax.random.split(jax.random.PRNGKey(2), 3)
    x = jax.random.normal(kx, (2, 64, 64), jnp.float32)
    w = jax.random.normal(kw, (64, 96), jnp.float32)
    g = jax.random.normal(kg, (2, 64, 96), jnp.float32)
    y, vjp = jax.vjp(lambda a, b: mx_dense(a, b), x, w)
    dx, dw = vjp(g)
    q = mx_quantize_dequantize
    np.testing.assert_allclose(y, jnp.einsum("bsk,kn->bsn", q(x, -1), q(w, 0)), rtol=1e-6)
    np.testing.assert_allclose(dx, jnp.einsum("bsn,kn->bsk", q(g, -1), q(w, 1)), rtol=1e-6)
    np.testing.assert_allclose(dw, jnp.einsum("bsk,bsn->kn", q(x, 1), q(g, 1)), rtol=1e-5)
    # Overall the gradients stay within MXFP8's few-percent error of the exact ones.
    exact_dw = jnp.einsum("bsk,bsn->kn", x, g)
    assert 0 < float(jnp.linalg.norm(dw - exact_dw) / jnp.linalg.norm(exact_dw)) < 0.06


def test_batched_gradients_match_per_expert_dense():
    kx, kw, kg = jax.random.split(jax.random.PRNGKey(3), 3)
    x = jax.random.normal(kx, (3, 64, 32), jnp.float32)
    w = jax.random.normal(kw, (3, 32, 64), jnp.float32)
    g = jax.random.normal(kg, (3, 64, 64), jnp.float32)
    _, vjp = jax.vjp(mx_batched, x, w)
    dx, dw = vjp(g)
    for e in range(3):
        _, dense_vjp = jax.vjp(lambda a, b: mx_dense(a, b), x[e], w[e])
        dx_e, dw_e = dense_vjp(g[e])
        np.testing.assert_allclose(dx[e], dx_e, rtol=1e-6)
        np.testing.assert_allclose(dw[e], dw_e, rtol=1e-5)
