# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from levanter.kernels.pallas.gated_norm import gated_norm_gate_bwd, gated_norm_gate_fwd, gated_norm_silu

ROWS, HIDDEN, RANK = 200, 256, 16  # rows not a multiple of the row blocks; two 128-column tiles


def _inputs(dtype):
    k = jax.random.split(jax.random.key(0), 5)
    n = jax.random.normal(k[0], (ROWS, HIDDEN), jnp.float32).astype(dtype)
    w_down = (0.2 * jax.random.normal(k[1], (HIDDEN, RANK), jnp.float32)).astype(dtype)
    w_up = (0.5 * jax.random.normal(k[2], (RANK, HIDDEN), jnp.float32)).astype(dtype)
    d_out = jax.random.normal(k[3], (ROWS, HIDDEN), jnp.float32).astype(dtype)
    a = jnp.dot(n, w_down, preferred_element_type=jnp.float32).astype(dtype)
    return n, a, w_down, w_up, d_out


def _gated_norm(n, w_down, w_up):
    """GatedNorm as autodiff sees it, in float32."""
    return n * jax.nn.sigmoid(jax.nn.silu(n @ w_down) @ w_up)


def _to_np(*xs):
    return [np.asarray(x, np.float32) for x in xs]


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_kernels_match_xla(dtype):
    n, a, _, w_up, d_out = _inputs(dtype)
    out_x, g_x = gated_norm_gate_fwd(n, a, w_up, impl="xla")
    out_p, g_p = gated_norm_gate_fwd(n, a, w_up, impl="pallas", interpret=True)
    dz_x, da_x = gated_norm_gate_bwd(n, a, g_x, w_up, d_out, impl="xla")
    dz_p, da_p = gated_norm_gate_bwd(n, a, g_x, w_up, d_out, impl="pallas", interpret=True)
    # Same expressions, same roundings; only the float32 order of the GEMM sums differs.
    tol = 1e-5 if dtype == jnp.float32 else 2**-7
    for got, want in [(out_p, out_x), (g_p, g_x), (dz_p, dz_x), (da_p, da_x)]:
        got, want = _to_np(got, want)
        np.testing.assert_allclose(got, want, rtol=tol, atol=tol * np.abs(want).max())


def test_gradients_match_autodiff_float32():
    n, a, w_down, w_up, d_out = _inputs(jnp.float32)
    want_out, vjp = jax.vjp(_gated_norm, n, w_down, w_up)
    want_dn, want_dwd, want_dwu = vjp(d_out)
    out, g = gated_norm_gate_fwd(n, a, w_up, impl="pallas", interpret=True)
    dz, da = gated_norm_gate_bwd(n, a, g, w_up, d_out, impl="pallas", interpret=True)
    # The caller's part: d_n = d_out * g + da w_down^T, d_w_down = n^T da, d_w_up = silu(a)^T dz.
    d_n = d_out * g + da @ w_down.T
    for got, want in [
        (out, want_out),
        (d_n, want_dn),
        (n.T @ da, want_dwd),
        (gated_norm_silu(a).T @ dz, want_dwu),
    ]:
        got, want = _to_np(got, want)
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4 * np.abs(want).max())
