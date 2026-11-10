# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from levanter.kernels.pallas.add_rms_norm import add_rms_norm_bwd, add_rms_norm_fwd

EPS = 1e-5
ROWS, HIDDEN = 200, 96  # rows not a multiple of the row blocks; hidden tiled in 32-column chunks


def _rows(key, dtype=jnp.bfloat16, scale=1.0):
    return (scale * jax.random.normal(key, (ROWS, HIDDEN), jnp.float32)).astype(dtype)


def _weight(key):
    return (1.0 + 0.1 * jax.random.normal(key, (HIDDEN,), jnp.float32)).astype(jnp.bfloat16)


def _bf16_ulps(a, b):
    """Elementwise distance in bfloat16 units in the last place of the larger magnitude."""
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    ulp = np.exp2(np.floor(np.log2(np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-30))) - 7)
    return np.abs(a - b) / ulp


def _f64_forward(x, residual, weight):
    # The statistic and n from the unrounded sum; h rounded.
    s = np.asarray(x, np.float64)
    if residual is not None:
        s = s + np.asarray(residual, np.float64)
    inv = 1.0 / np.sqrt(np.mean(s * s, axis=-1, keepdims=True) + EPS)
    h = np.asarray(jnp.asarray(s, jnp.float32).astype(jnp.bfloat16), np.float64)
    return h, s * inv * np.asarray(weight, np.float64), inv[:, 0]


@pytest.mark.parametrize("with_residual", [False, True])
def test_forward_kernel_matches_xla_and_float64(with_residual):
    kx, kr, kw = jax.random.split(jax.random.key(0), 3)
    x, weight = _rows(kx, scale=3.0), _weight(kw)
    residual = _rows(kr) if with_residual else None
    h_x, n_x, inv_x = add_rms_norm_fwd(x, residual, weight, eps=EPS, impl="xla")
    h_p, n_p, inv_p = add_rms_norm_fwd(x, residual, weight, eps=EPS, impl="pallas", interpret=True)
    h64, n64, inv64 = _f64_forward(x, residual, weight)
    # h is the sum rounded once: exact.
    np.testing.assert_array_equal(np.asarray(h_p, np.float32), np.asarray(h_x, np.float32))
    np.testing.assert_array_equal(np.asarray(h_x, np.float64), h64)
    # The statistic differs only in float32 summation order; n then rounds to bf16 (at most one ulp apart).
    np.testing.assert_allclose(inv_p, inv_x, rtol=2e-6)
    np.testing.assert_allclose(inv_x, inv64, rtol=2e-6)
    assert _bf16_ulps(n_p, n_x).max() <= 1
    assert _bf16_ulps(n_x, n64).max() <= 0.5 + 1e-3


@pytest.mark.parametrize(
    ("gated", "extra", "with_dh"), [(False, False, False), (True, True, True), (True, False, True)]
)
def test_backward_kernel_matches_xla_and_autodiff(gated, extra, with_dh):
    keys = jax.random.split(jax.random.key(1), 7)
    x, r, weight = _rows(keys[0], scale=2.0), _rows(keys[1]), _weight(keys[2])
    d_n = _rows(keys[3])
    gate = jax.nn.sigmoid(_rows(keys[4])) if gated else None
    d_n_extra = _rows(keys[5], scale=0.5) if extra else None
    d_h = _rows(keys[6]) if with_dh else None
    h, _, _ = add_rms_norm_fwd(x, r, weight, eps=EPS, impl="xla")
    # The statistic of the rounded h, which the autodiff reference below differentiates.
    _, _, inv = add_rms_norm_fwd(h, None, weight, eps=EPS, impl="xla")
    dx_x, dw_x = add_rms_norm_bwd(h, inv, weight, d_n, gate=gate, d_n_extra=d_n_extra, d_h=d_h, impl="xla")
    dx_p, dw_p = add_rms_norm_bwd(
        h, inv, weight, d_n, gate=gate, d_n_extra=d_n_extra, d_h=d_h, impl="pallas", interpret=True
    )

    # Autodiff of the float32 forward from h, with the same cotangent.
    g = d_n.astype(jnp.float32)
    if gate is not None:
        g = g * gate.astype(jnp.float32)
    if d_n_extra is not None:
        g = g + d_n_extra.astype(jnp.float32)

    def f(hf, w):
        inv_rms = jax.lax.rsqrt(jnp.mean(jnp.square(hf), axis=-1, keepdims=True) + EPS)
        return hf * inv_rms * w

    _, vjp = jax.vjp(f, h.astype(jnp.float32), weight.astype(jnp.float32))
    dh_ad, dw_ad = vjp(g)
    if d_h is not None:
        dh_ad = dh_ad + d_h.astype(jnp.float32)

    assert dx_p.dtype == dx_x.dtype == h.dtype
    assert _bf16_ulps(dx_p, dx_x).max() <= 1
    scale = float(jnp.max(jnp.abs(dh_ad)))
    np.testing.assert_allclose(np.asarray(dx_x, np.float32), dh_ad, atol=scale * 2**-8)
    np.testing.assert_allclose(dw_p, dw_x, rtol=1e-5, atol=1e-5 * float(jnp.max(jnp.abs(dw_x))))
    np.testing.assert_allclose(dw_x, dw_ad, rtol=1e-4, atol=1e-4 * float(jnp.max(jnp.abs(dw_ad))))
