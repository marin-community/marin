# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0
# Modifications Copyright The Levanter Authors, licensed Apache-2.0.
# Derived from Atomic Ops, Copyright (c) 2026 Omirbay Akseleu, MIT licensed.
# The original MIT copyright and permission notice are retained in ../LICENSE.

"""Differentiable GDN-2 for single-device arrays with FP32 state."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from .configs import KernelConfig, DEFAULT_CONFIG, _reshape_from_chunks, _reshape_to_chunks, validate_inputs
from .gdn2_fwd import (
    ForwardResiduals,
    gdn2_pallas_forward,
    gdn2_pallas_forward_with_residuals,
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
    out, h_final, forward = gdn2_pallas_forward_with_residuals(q, k, v, w, b, g, scale, h0=h0, config=config)
    return (out, h_final), ((q, k, v, w, b, g, h0), forward)


def _gdn2_core_bwd(scale, config, residuals: tuple[tuple[jax.Array, ...], ForwardResiduals], cotangents):
    (q, k, v, w, b, g, h0), forward = residuals
    do, dh_final = cotangents
    bsz, L, H, D = q.shape
    n_chunks = L // config.bt

    do_r = _reshape_to_chunks(do, bsz, n_chunks, H, D, config.bt)

    # Build gc from g (needed for B3 and B4)
    g_r = _reshape_to_chunks(g, bsz, n_chunks, H, D, config.bt)
    idx = jnp.arange(config.bt)
    tril_ones_bt = (idx[:, None] >= idx[None, :]).astype(jnp.float32)
    gc = jnp.einsum("ij,bhcjd->bhcid", tril_ones_bt, g_r, precision=_HIGHEST)

    # B2
    dAqk, dv_partial = dav_backward_pallas(forward.Aqk, forward.v_new_all, do_r, config)

    # B1
    dh_all, dh0, dv_all = gdn2_dhu_backward(
        do_r, dv_partial, forward.w_pseudo, forward.qg, forward.kg, forward.gc_last, scale, dht=dh_final
    )
    dh_next_all = _build_dh_next_all(dh_all, dh_final)

    # B3
    q_r = _reshape_to_chunks(q, bsz, n_chunks, H, D, config.bt)
    k_r = _reshape_to_chunks(k, bsz, n_chunks, H, D, config.bt)
    b_r = _reshape_to_chunks(b, bsz, n_chunks, H, D, config.bt)
    w_r = _reshape_to_chunks(w, bsz, n_chunks, H, D, config.bt)
    v_r = _reshape_to_chunks(v, bsz, n_chunks, H, D, config.bt)

    b3_out = wy_dqkg_backward_pallas(
        q_r,
        k_r,
        b_r,
        w_r,
        v_r,
        gc,
        forward.A,
        forward.h_pre_all,
        forward.v_new_all,
        do_r,
        dv_all,
        dh_next_all,
        scale,
        config,
    )

    score_grads = intra_backward_pallas(dAqk, b3_out.dAkk, q, k, b, g, scale, config)
    # B5
    dgc_total = b3_out.dgc + score_grads.dgc
    dg_raw = reverse_cumsum_bwd(dgc_total, chunk_size=config.bt)

    dq = _reshape_from_chunks(b3_out.dq + score_grads.dq, bsz, n_chunks, config.bt, H, D)
    dk = _reshape_from_chunks(b3_out.dk + score_grads.dk, bsz, n_chunks, config.bt, H, D)
    db = _reshape_from_chunks(b3_out.db + score_grads.db, bsz, n_chunks, config.bt, H, D)
    dw = _reshape_from_chunks(b3_out.dw, bsz, n_chunks, config.bt, H, D)
    dv = _reshape_from_chunks(b3_out.dv_raw, bsz, n_chunks, config.bt, H, D)
    dg = _reshape_from_chunks(dg_raw, bsz, n_chunks, config.bt, H, D)

    # Match cotangent dtypes without hiding non-finite arithmetic.
    dq = dq.astype(q.dtype)
    dk = dk.astype(k.dtype)
    db = db.astype(b.dtype)
    dw = dw.astype(w.dtype)
    dv = dv.astype(v.dtype)
    dg = dg.astype(g.dtype)
    dh0 = dh0.astype(h0.dtype)

    return dq, dk, dv, dw, db, dg, dh0


_gdn2_core.defvjp(_gdn2_core_fwd, _gdn2_core_bwd)


def gdn2_pallas_forward_trainable(q, k, v, w, b, g, scale, h0=None, config: KernelConfig = DEFAULT_CONFIG):
    """Apply GDN-2 to single-device arrays with finite log-decay ``g <= 0``.

    The optional initial state ``h0`` must be FP32; omission initializes zeros.
    Outputs and final state are FP32. No clipping or non-finite replacement is
    performed, and callers must keep all inputs local to the same device.
    """
    validate_inputs(q, k, v, w, b, g, h0, config)
    if h0 is None:
        bsz, _, heads, dim = q.shape
        h0 = jnp.zeros((bsz, heads, dim, dim), dtype=jnp.float32)
    return _gdn2_core(q, k, v, w, b, g, scale, config, h0)
