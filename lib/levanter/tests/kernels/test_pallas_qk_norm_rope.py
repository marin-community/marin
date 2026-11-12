# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from levanter.kernels.pallas.qk_norm_rope import qk_norm_rope, qk_norm_rope_reference, rope_tables

EPS = 1e-6
# (batch, seq, heads, head_dim), with a row count that is not a multiple of the kernel's row block.
SHAPES = [(2, 8, 3, 64), (1, 16, 5, 128)]
# (shape, per_batch_positions, rope, scales): each shape, table batch, rope setting and scale chain at least once.
CASES = [
    ((2, 8, 3, 64), False, True, (1.3, 1.5)),
    ((1, 16, 5, 128), True, True, (1.3, 1.0)),
    ((2, 8, 3, 64), True, False, (1.3,)),
    ((1, 16, 5, 128), False, False, ()),
]


def _inputs(shape, dtype, *, per_batch_positions, seed=0):
    batch, seq, _, head_dim = shape
    k_x, k_g, k_s, k_p = jax.random.split(jax.random.key(seed), 4)
    # Rows of different scales, so eps matters for some of them.
    scale = jnp.exp(jax.random.uniform(k_s, (*shape[:-1], 1), minval=-6.0, maxval=3.0))
    x = (jax.random.normal(k_x, shape) * scale).astype(dtype)
    g = jax.random.normal(k_g, shape).astype(dtype)
    if per_batch_positions:
        positions = jax.random.randint(k_p, (batch, seq), 0, 4 * seq)
    else:
        positions = jnp.arange(seq)[None]
    cos, sin = rope_tables(positions, head_dim // 4, theta=10000.0)
    return x, g, cos, sin


def _value_and_grad(fn, x, g):
    y, vjp = jax.vjp(fn, x)
    (dx,) = vjp(g)
    return np.asarray(y, np.float32), np.asarray(dx, np.float32)


def _ulps(actual, expected, dtype):
    # Differences in units of the output dtype's epsilon times each element's magnitude (at least the tensor's
    # typical magnitude, so tiny elements do not dominate).
    floor = np.median(np.abs(expected))
    return np.abs(actual - expected) / (np.maximum(np.abs(expected), floor) * float(jnp.finfo(dtype).eps))


@pytest.mark.parametrize("shape,per_batch_positions,rope,scales", CASES)
def test_fused_qk_norm_rope_matches_the_separate_ops_in_bfloat16(shape, per_batch_positions, rope, scales):
    # The kernels round where the separate ops do; only the float32 statistic is summed in another order, so
    # an element differs by at most one bfloat16 ulp, and rarely. The backward rounds at the same points as JAX's
    # differentiation of the separate ops; its statistic and projection sums also run in another order, so dx stays
    # within two bfloat16 ulps of the element's scale.
    x, g, cos, sin = _inputs(shape, jnp.bfloat16, per_batch_positions=per_batch_positions)
    kwargs = dict(eps=EPS, rope=rope, scales=scales)
    want_y, want_dx = _value_and_grad(lambda v: qk_norm_rope_reference(v, cos, sin, **kwargs), x, g)
    fn = jax.jit(lambda v: qk_norm_rope(v, cos, sin, impl="pallas", **kwargs))
    y, dx = _value_and_grad(fn, x, g)
    y_ulps = _ulps(y, want_y, jnp.bfloat16)
    assert y_ulps.max() <= 1.0, y_ulps.max()
    assert np.mean(y_ulps > 0) < 0.01, np.mean(y_ulps > 0)
    dx_ulps = _ulps(dx, want_dx, jnp.bfloat16)
    assert dx_ulps.max() <= 2.0, dx_ulps.max()
    assert np.mean(dx_ulps > 0) < 0.05, np.mean(dx_ulps > 0)


@pytest.mark.parametrize("rope", [True, False])
@pytest.mark.parametrize("shape", SHAPES)
def test_fused_qk_norm_rope_matches_a_float64_reference_in_float32(shape, rope):
    x, g, cos, sin = _inputs(shape, jnp.float32, per_batch_positions=True, seed=1)
    scales = (1.3, 1.5)
    x64, g64 = np.asarray(x, np.float64), np.asarray(g, np.float64)
    c64, s64 = np.asarray(cos, np.float64)[:, :, None], np.asarray(sin, np.float64)[:, :, None]
    q = shape[-1] // 4
    r = 1 / np.sqrt(np.mean(x64 * x64, axis=-1, keepdims=True) + EPS)
    n = x64 * r
    y = n.copy()
    gn = g64 * np.prod(scales)
    if rope:
        a, b = n[..., :q], n[..., q : 2 * q]
        y[..., :q], y[..., q : 2 * q] = a * c64 - b * s64, b * c64 + a * s64
        ga, gb = gn[..., :q].copy(), gn[..., q : 2 * q].copy()
        gn[..., :q], gn[..., q : 2 * q] = ga * c64 + gb * s64, gb * c64 - ga * s64
    y = y * np.prod(scales)
    dx = r * gn - x64 * r**3 * np.mean(gn * x64, axis=-1, keepdims=True)
    for impl in ("reference", "pallas"):
        fn = jax.jit(lambda v, impl=impl: qk_norm_rope(v, cos, sin, eps=EPS, rope=rope, scales=scales, impl=impl))
        got_y, got_dx = _value_and_grad(fn, x, g)
        np.testing.assert_allclose(got_y, y, rtol=0, atol=1e-5 * np.abs(y).max(), err_msg=impl)
        np.testing.assert_allclose(got_dx, dx, rtol=0, atol=1e-5 * np.abs(dx).max(), err_msg=impl)
