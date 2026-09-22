# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0
# Modifications Copyright The Levanter Authors, licensed Apache-2.0.
# Derived from Atomic Ops, Copyright (c) 2026 Omirbay Akseleu, MIT licensed.
# The original MIT copyright and permission notice are retained in ../LICENSE.

"""
Trainable wrapper: custom_vjp with fused Pallas backward.
Backward uses residuals from forward_with_residuals — NO recomputation.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from .configs import KernelConfig, DEFAULT_CONFIG, validate_inputs
from .gdn2_fwd import (
    gdn2_pallas_forward,
    gdn2_pallas_forward_with_residuals,
    _reshape_to_chunks as _r2c,
    _reshape_from_chunks as _r2f,
)
from .gdn2_bwd import (
    gdn2_dhu_backward,
    dav_backward_pallas,
    wy_dqkg_backward_pallas,
    intra_backward_pallas,
    reverse_cumsum_bwd,
)

_HIGHEST = jax.lax.Precision.HIGHEST


def _build_dh_next_all(dh_all, dht):
    shifted = dh_all[:, :, 1:]
    dht_expanded = dht[:, :, None]
    return jnp.concatenate([shifted, dht_expanded], axis=2)


@partial(jax.custom_vjp, nondiff_argnums=(6, 7))
def _gdn2_core(q, k, v, w, b, g, scale, config, h0):
    return gdn2_pallas_forward(q, k, v, w, b, g, scale, h0=h0, config=config)


def _gdn2_core_fwd(q, k, v, w, b, g, scale, config, h0):
    out, h_final, fwd_res = gdn2_pallas_forward_with_residuals(q, k, v, w, b, g, scale, h0=h0, config=config)
    residuals = {
        "q": q,
        "k": k,
        "v": v,
        "w": w,
        "b": b,
        "g": g,
        "h0": h0,
        **fwd_res,
    }
    return (out, h_final), residuals


def _gdn2_core_bwd(scale, config, residuals, cotangents):
    q = residuals["q"]
    k = residuals["k"]
    v = residuals["v"]
    w = residuals["w"]
    b = residuals["b"]
    g = residuals["g"]
    h0 = residuals["h0"]
    Aqk = residuals["Aqk"]
    Akk = residuals["Akk"]
    A = residuals["A"]
    h_pre_all = residuals["h_pre_all"]
    v_new_all = residuals["v_new_all"]
    w_pseudo = residuals["w_pseudo"]
    kg = residuals["kg"]
    qg = residuals["qg"]
    gc_last = residuals["gc_last"]

    do, dh_final = cotangents
    bsz, L, H, D = q.shape
    n_chunks = L // config.bt

    # Reshape do into chunk form
    do_r = _r2c(do, bsz, n_chunks, H, D, config.bt)

    # Build gc from g (needed for B3 and B4)
    g_r = _r2c(g, bsz, n_chunks, H, D, config.bt)
    idx = jnp.arange(config.bt)
    tril_ones_bt = (idx[:, None] >= idx[None, :]).astype(jnp.float32)
    gc = jnp.einsum("ij,bhcjd->bhcid", tril_ones_bt, g_r, precision=_HIGHEST)

    # B2
    dAqk, dv_partial = dav_backward_pallas(Aqk, v_new_all, do_r, config)

    # B1
    dh_all, dh0, dv_all = gdn2_dhu_backward(
        do_r, dv_partial, w_pseudo, qg, kg, gc_last, scale, dht=dh_final, config=config
    )
    dh_next_all = _build_dh_next_all(dh_all, dh_final)

    # B3
    q_r = _r2c(q, bsz, n_chunks, H, D, config.bt)
    k_r = _r2c(k, bsz, n_chunks, H, D, config.bt)
    b_r = _r2c(b, bsz, n_chunks, H, D, config.bt)
    w_r = _r2c(w, bsz, n_chunks, H, D, config.bt)
    v_r = _r2c(v, bsz, n_chunks, H, D, config.bt)

    b3_out = wy_dqkg_backward_pallas(
        q_r,
        k_r,
        b_r,
        w_r,
        v_r,
        gc,
        A,
        Akk,
        h_pre_all,
        v_new_all,
        do_r,
        dv_all,
        dh_next_all,
        scale,
        config,
    )

    dq4, dk4, db4, dgc4 = intra_backward_pallas(dAqk, b3_out["dAkk"], q, k, b, g, scale, config)
    # B5
    dgc_total = b3_out["dgc"] + dgc4
    dg_raw = reverse_cumsum_bwd(dgc_total, chunk_size=config.bt, config=config)

    # Assemble
    dq = _r2f(b3_out["dq"] + dq4, bsz, n_chunks, config.bt, H, D)
    dk = _r2f(b3_out["dk"] + dk4, bsz, n_chunks, config.bt, H, D)
    db = _r2f(b3_out["db"] + db4, bsz, n_chunks, config.bt, H, D)
    dw = _r2f(b3_out["dw"], bsz, n_chunks, config.bt, H, D)
    dv = _r2f(b3_out["dv_raw"], bsz, n_chunks, config.bt, H, D)
    dg = _r2f(dg_raw, bsz, n_chunks, config.bt, H, D)

    # Match cotangent dtypes without hiding non-finite arithmetic.
    dq = dq.astype(q.dtype)
    dk = dk.astype(k.dtype)
    db = db.astype(b.dtype)
    dw = dw.astype(w.dtype)
    dv = dv.astype(v.dtype)
    dg = dg.astype(g.dtype)
    dh0 = dh0.astype(h0.dtype)

    return dq, dk, dv, dw, db, dg, dh0


# Register corrected VJP
_gdn2_core.defvjp(_gdn2_core_fwd, _gdn2_core_bwd)


def gdn2_pallas_forward_trainable(q, k, v, w, b, g, scale, h0=None, config: KernelConfig = DEFAULT_CONFIG):
    """Apply GDN-2 to single-device arrays with finite log-decay ``g <= 0``.

    The optional initial state ``h0`` must be FP32; omission initializes zeros.
    Outputs and final state are FP32. No clipping or non-finite replacement is
    performed, and callers must keep all inputs local to the same device.
    """
    bsz, L, H, D = q.shape
    if h0 is None:
        h0 = jnp.zeros((bsz, H, D, D), dtype=jnp.float32)
    validate_inputs(q, k, v, w, b, g, scale, h0, config)
    return _gdn2_core(q, k, v, w, b, g, scale, config, h0)
