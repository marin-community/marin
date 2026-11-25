# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The AMD June model's Pallas kernels, in the Pallas interpreter, against their JAX formulations and float64."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.amd.june_67b_a2b.kernels.add_rms_norm import (
    add_rms_norm_bwd,
    add_rms_norm_bwd_reference,
    add_rms_norm_fwd,
    add_rms_norm_fwd_reference,
)
from experiments.amd.june_67b_a2b.kernels.gated_norm import (
    gated_norm_gate_bwd,
    gated_norm_gate_bwd_reference,
    gated_norm_gate_fwd,
    gated_norm_gate_fwd_reference,
    gated_norm_silu,
)
from experiments.amd.june_67b_a2b.kernels.qk_norm_rope import qk_norm_rope, qk_norm_rope_reference, rope_tables

QK_EPS = 1e-6
NORM_EPS = 1e-5
# Row counts that are not multiples of the kernels' row blocks.
NORM_ROWS = 200


def _bf16_ulps(actual, expected):
    """Elementwise distance in bfloat16 units in the last place of the larger magnitude."""
    a, b = np.asarray(actual, np.float32), np.asarray(expected, np.float32)
    ulp = np.exp2(np.floor(np.log2(np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-30))) - 7)
    return np.abs(a - b) / ulp


def _qk_inputs(shape, dtype, *, per_batch_positions):
    batch, seq, _, head_dim = shape
    k_x, k_g, k_s, k_p = jax.random.split(jax.random.key(0), 4)
    # Rows of different scales, so eps matters for some of them.
    scale = jnp.exp(jax.random.uniform(k_s, (*shape[:-1], 1), minval=-6.0, maxval=3.0))
    x = (jax.random.normal(k_x, shape) * scale).astype(dtype)
    g = jax.random.normal(k_g, shape).astype(dtype)
    positions = jax.random.randint(k_p, (batch, seq), 0, 4 * seq) if per_batch_positions else jnp.arange(seq)[None]
    cos, sin = rope_tables(positions, head_dim // 4, theta=10000.0)
    return x, g, cos, sin


def _value_and_grad(fn, x, g):
    y, vjp = jax.vjp(fn, x)
    (dx,) = vjp(g)
    return np.asarray(y, np.float32), np.asarray(dx, np.float32)


@pytest.mark.parametrize(
    "shape,per_batch_positions,rope,scales",
    [((2, 8, 3, 64), False, True, (1.3, 1.5)), ((1, 16, 5, 128), True, False, (1.3,))],
)
def test_qk_norm_rope_kernels_match_the_separate_ops_in_bfloat16(shape, per_batch_positions, rope, scales):
    # The kernels round where the separate ops do; only the float32 statistic and projection are summed in another
    # order, so y differs by at most one bfloat16 ulp in a few elements and dx by at most two.
    x, g, cos, sin = _qk_inputs(shape, jnp.bfloat16, per_batch_positions=per_batch_positions)
    kwargs = dict(eps=QK_EPS, rope=rope, scales=scales)
    want_y, want_dx = _value_and_grad(lambda v: qk_norm_rope_reference(v, cos, sin, **kwargs), x, g)
    y, dx = _value_and_grad(jax.jit(lambda v: qk_norm_rope(v, cos, sin, **kwargs)), x, g)
    y_ulps, dx_ulps = _bf16_ulps(y, want_y), _bf16_ulps(dx, want_dx)
    assert y_ulps.max() <= 1.0 and np.mean(y_ulps > 0) < 0.01, (y_ulps.max(), np.mean(y_ulps > 0))
    assert dx_ulps.max() <= 2.0 and np.mean(dx_ulps > 0) < 0.05, (dx_ulps.max(), np.mean(dx_ulps > 0))


def test_qk_norm_rope_kernels_match_float64_in_float32():
    shape, scales = (2, 8, 3, 64), (1.3, 1.5)
    x, g, cos, sin = _qk_inputs(shape, jnp.float32, per_batch_positions=True)
    x64, g64 = np.asarray(x, np.float64), np.asarray(g, np.float64)
    c64, s64 = np.asarray(cos, np.float64)[:, :, None], np.asarray(sin, np.float64)[:, :, None]
    q = shape[-1] // 4
    r = 1 / np.sqrt(np.mean(x64 * x64, axis=-1, keepdims=True) + QK_EPS)
    n = x64 * r
    y = n.copy()
    y[..., :q], y[..., q : 2 * q] = (
        n[..., :q] * c64 - n[..., q : 2 * q] * s64,
        n[..., q : 2 * q] * c64 + n[..., :q] * s64,
    )
    y = y * np.prod(scales)
    gn = g64 * np.prod(scales)
    ga, gb = gn[..., :q].copy(), gn[..., q : 2 * q].copy()
    gn[..., :q], gn[..., q : 2 * q] = ga * c64 + gb * s64, gb * c64 - ga * s64
    dx = r * gn - x64 * r**3 * np.mean(gn * x64, axis=-1, keepdims=True)
    got_y, got_dx = _value_and_grad(jax.jit(lambda v: qk_norm_rope(v, cos, sin, eps=QK_EPS, scales=scales)), x, g)
    np.testing.assert_allclose(got_y, y, rtol=0, atol=1e-5 * np.abs(y).max())
    np.testing.assert_allclose(got_dx, dx, rtol=0, atol=1e-5 * np.abs(dx).max())


def _norm_rows(key, scale=1.0, hidden=96):
    return (scale * jax.random.normal(key, (NORM_ROWS, hidden), jnp.float32)).astype(jnp.bfloat16)


def test_add_rms_norm_kernels_match_jax_and_float64():
    # Hidden 96 is tiled in 32-column chunks.
    keys = jax.random.split(jax.random.key(1), 7)
    x, residual = _norm_rows(keys[0], scale=3.0), _norm_rows(keys[1])
    weight = (1.0 + 0.1 * jax.random.normal(keys[2], (96,))).astype(jnp.bfloat16)
    h_ref, n_ref, inv_ref = add_rms_norm_fwd_reference(x, residual, weight, eps=NORM_EPS)
    h, n, inv = add_rms_norm_fwd(x, residual, weight, eps=NORM_EPS)
    # h is the sum rounded once: exact. The statistic differs only in float32 summation order.
    np.testing.assert_array_equal(np.asarray(h, np.float32), np.asarray(h_ref, np.float32))
    np.testing.assert_allclose(inv, inv_ref, rtol=2e-6)
    assert _bf16_ulps(n, n_ref).max() <= 1
    s = np.asarray(x, np.float64) + np.asarray(residual, np.float64)
    n64 = s / np.sqrt(np.mean(s * s, axis=-1, keepdims=True) + NORM_EPS) * np.asarray(weight, np.float64)
    assert _bf16_ulps(n_ref, n64).max() <= 0.5 + 1e-3

    d_n, gate, d_n_extra, d_h = (_norm_rows(keys[i]) for i in (3, 4, 5, 6))
    gate = jax.nn.sigmoid(gate)
    # The statistic of the rounded h, which the autodiff check below differentiates.
    _, _, inv_h = add_rms_norm_fwd_reference(h, None, weight, eps=NORM_EPS)
    dx_ref, dw_ref = add_rms_norm_bwd_reference(h, inv_h, weight, d_n, gate=gate, d_n_extra=d_n_extra, d_h=d_h)
    dx, dw = add_rms_norm_bwd(h, inv_h, weight, d_n, gate=gate, d_n_extra=d_n_extra, d_h=d_h)
    assert _bf16_ulps(dx, dx_ref).max() <= 1
    np.testing.assert_allclose(dw, dw_ref, rtol=1e-5, atol=1e-5 * float(jnp.max(jnp.abs(dw_ref))))

    # The reference backward against autodiff of the float32 forward from h, with the same cotangent.
    def f(hf, w):
        return hf * jax.lax.rsqrt(jnp.mean(jnp.square(hf), axis=-1, keepdims=True) + NORM_EPS) * w

    _, vjp = jax.vjp(f, h.astype(jnp.float32), weight.astype(jnp.float32))
    g = d_n.astype(jnp.float32) * gate.astype(jnp.float32) + d_n_extra.astype(jnp.float32)
    dh_ad, dw_ad = vjp(g)
    dh_ad = dh_ad + d_h.astype(jnp.float32)
    np.testing.assert_allclose(np.asarray(dx_ref, np.float32), dh_ad, atol=float(jnp.max(jnp.abs(dh_ad))) * 2**-8)
    np.testing.assert_allclose(dw_ref, dw_ad, rtol=1e-4, atol=1e-4 * float(jnp.max(jnp.abs(dw_ad))))


def test_gated_norm_kernels_match_jax_and_autodiff_in_float32():
    # Two 128-column tiles; rank 16.
    k = jax.random.split(jax.random.key(2), 4)
    n = jax.random.normal(k[0], (NORM_ROWS, 256))
    w_down = 0.2 * jax.random.normal(k[1], (256, 16))
    w_up = 0.5 * jax.random.normal(k[2], (16, 256))
    d_out = jax.random.normal(k[3], (NORM_ROWS, 256))
    a = n @ w_down
    out_ref, g_ref = gated_norm_gate_fwd_reference(n, a, w_up)
    dz_ref, da_ref = gated_norm_gate_bwd_reference(n, a, g_ref, w_up, d_out)
    out, g = gated_norm_gate_fwd(n, a, w_up)
    dz, da = gated_norm_gate_bwd(n, a, g, w_up, d_out)
    for got, want in [(out, out_ref), (g, g_ref), (dz, dz_ref), (da, da_ref)]:
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5 * float(jnp.max(jnp.abs(want))))

    # GatedNorm as autodiff sees it; the caller's part: d_n = d_out * g + da w_down^T, d_w_down = n^T da,
    # d_w_up = silu(a)^T dz.
    want_out, vjp = jax.vjp(lambda n, wd, wu: n * jax.nn.sigmoid(jax.nn.silu(n @ wd) @ wu), n, w_down, w_up)
    want_dn, want_dwd, want_dwu = vjp(d_out)
    for got, want in [
        (out, want_out),
        (d_out * g + da @ w_down.T, want_dn),
        (n.T @ da, want_dwd),
        (gated_norm_silu(a).T @ dz, want_dwu),
    ]:
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4 * float(jnp.max(jnp.abs(want))))
