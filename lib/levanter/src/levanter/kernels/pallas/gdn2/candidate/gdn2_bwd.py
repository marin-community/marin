# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0
# Modifications Copyright The Levanter Authors, licensed Apache-2.0.
# Derived from Atomic Ops, Copyright (c) 2026 Omirbay Akseleu, MIT licensed.
# The original MIT copyright and permission notice are retained in ../LICENSE.

"""
Backward kernels: B1 (state) -> B2 (dAqk/dv) -> B3 (WY/dqkg) -> B4 (intra) -> B5 (reverse cumsum).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .configs import (
    KernelConfig,
    DEFAULT_CONFIG,
    ScoreLayout,
    _reshape_to_chunks as _r2c,
)
from .gdn2_fwd import _kernel_cost

_HIGHEST = jax.lax.Precision.HIGHEST


# ---------- B5 ----------
def reverse_cumsum_bwd(dgc, chunk_size: int, config: KernelConfig = DEFAULT_CONFIG):
    C = chunk_size
    idx = jnp.arange(C)
    triu_ones = (idx[:, None] <= idx[None, :]).astype(jnp.float32)
    dg_raw = jnp.einsum("ij,...jd->...id", triu_ones, dgc.astype(jnp.float32), precision=_HIGHEST)
    return dg_raw


# ---------- B1 ----------
def gdn2_dhu_backward(
    do, dv_partial, w_pseudo, qg, kg, gc_last, scale, dht=None, config: KernelConfig = DEFAULT_CONFIG
):
    bsz, H, n_chunks, BT, D = qg.shape
    if dht is None:
        dht = jnp.zeros((bsz, H, D, D), dtype=jnp.float32)

    to_scan = tuple(jnp.moveaxis(x, 2, 0) for x in (do, dv_partial, w_pseudo, qg, kg, gc_last))

    def step(dh_carry, inputs):
        do_c, dvp_c, wp_c, qg_c, kg_c, gclast_c = inputs
        decay_c = jnp.exp(gclast_c)[..., None]

        dqh = scale * do_c
        contrib_from_output = jnp.einsum("bhid,bhiv->bhdv", qg_c, dqh, precision=_HIGHEST)
        contrib_from_state = dh_carry * decay_c

        dv_write = jnp.einsum("bhid,bhdv->bhiv", kg_c, dh_carry, precision=_HIGHEST)
        dv_new_c = dvp_c + dv_write

        contrib_from_vnew = -jnp.einsum("bhjd,bhjv->bhdv", wp_c, dv_new_c, precision=_HIGHEST)

        dh_pre_c = contrib_from_output + contrib_from_state + contrib_from_vnew
        return dh_pre_c, (dh_pre_c, dv_new_c)

    dh0, (dh_all_rev, dv_all_rev) = jax.lax.scan(step, dht, to_scan, reverse=True)
    dh_all = jnp.moveaxis(dh_all_rev, 0, 2)
    dv_all = jnp.moveaxis(dv_all_rev, 0, 2)
    return dh_all, dh0, dv_all


# ---------- B2 ----------
def _kernel_b2_body(aqk_ref, vnew_ref, do_ref, daqk_ref, dvnew_ref, *, bt: int, config: KernelConfig):
    Aqk = aqk_ref[0, 0, 0].astype(jnp.float32)
    v_new = vnew_ref[0, 0, 0].astype(jnp.float32)
    do = do_ref[0, 0, 0].astype(jnp.float32)

    idx = jnp.arange(bt)
    causal = (idx[:, None] >= idx[None, :]).astype(jnp.float32)

    dAqk = jnp.dot(do, v_new.T, precision=_HIGHEST) * causal
    dv_new = jnp.dot(Aqk.T, do, precision=_HIGHEST)

    daqk_ref[0, 0, 0] = dAqk
    dvnew_ref[0, 0, 0] = dv_new


def dav_backward_pallas(Aqk, v_new, do, config: KernelConfig = DEFAULT_CONFIG):
    bsz, H, n_chunks, _BT, D = v_new.shape
    grid = (bsz, H, n_chunks)
    aqk_spec = pl.BlockSpec((1, 1, 1, config.bt, config.bt), lambda i, h, c: (i, h, c, 0, 0))
    io_spec = pl.BlockSpec((1, 1, 1, config.bt, D), lambda i, h, c: (i, h, c, 0, 0))

    outputs = [jax.ShapeDtypeStruct(Aqk.shape, jnp.float32), jax.ShapeDtypeStruct(v_new.shape, jnp.float32)]
    dAqk, dv_new = pl.pallas_call(
        lambda *refs: _kernel_b2_body(*refs, bt=config.bt, config=config),
        name="gdn2_backward_output",
        grid=grid,
        in_specs=[aqk_spec, io_spec, io_spec],
        out_specs=[aqk_spec, io_spec],
        out_shape=outputs,
        cost_estimate=_kernel_cost(_kernel_b2_body, (Aqk, v_new, do), outputs, bt=config.bt, config=config),
        compiler_params=pltpu.CompilerParams(),
        interpret=config.interpret,
    )(Aqk, v_new, do)
    return dAqk, dv_new


# ---------- B3 ----------
def _kernel_b3_body(
    q_ref,
    k_ref,
    b_ref,
    w_ref,
    v_ref,
    gc_ref,
    a_ref,
    akk_ref,
    hpre_ref,
    vnew_ref,
    do_ref,
    dv_ref,
    dhnext_ref,
    dq_ref,
    dk_ref,
    db_ref,
    dw_ref,
    dvraw_ref,
    dgc_ref,
    dakk_ref,
    *,
    scale: float,
    bt: int,
    wy_eps: float,
    config: KernelConfig,
):
    q_c = q_ref[0, 0, 0].astype(jnp.float32)
    k_c = k_ref[0, 0, 0].astype(jnp.float32)
    b_c = b_ref[0, 0, 0].astype(jnp.float32)
    w_c = w_ref[0, 0, 0].astype(jnp.float32)
    v_c = v_ref[0, 0, 0].astype(jnp.float32)
    gc = gc_ref[0, 0, 0].astype(jnp.float32)
    A = a_ref[0, 0, 0].astype(jnp.float32)
    h_pre = hpre_ref[0, 0, 0].astype(jnp.float32)
    v_new = vnew_ref[0, 0, 0].astype(jnp.float32)
    do = do_ref[0, 0, 0].astype(jnp.float32)
    dv = dv_ref[0, 0, 0].astype(jnp.float32)
    dh_next = dhnext_ref[0, 0, 0].astype(jnp.float32)

    C = bt
    gc_last = gc[C - 1]

    kb_decayed = b_c * k_c * jnp.exp(gc)
    kg = k_c * jnp.exp(gc_last[None, :] - gc)
    qg = q_c * jnp.exp(gc)
    wv = w_c * v_c

    dqh_up = scale * do
    dqg = jnp.dot(dqh_up, h_pre.T, precision=_HIGHEST)

    dwh = -dv
    dw_pseudo = jnp.dot(dwh, h_pre.T, precision=_HIGHEST)
    du = dv

    dkg = jnp.dot(v_new, dh_next.T, precision=_HIGHEST)

    dA_from_w = jnp.dot(dw_pseudo, kb_decayed.T, precision=_HIGHEST)
    dkb_decayed = jnp.dot(A.T, dw_pseudo, precision=_HIGHEST)

    dA_from_u = jnp.dot(du, wv.T, precision=_HIGHEST)
    dwv = jnp.dot(A.T, du, precision=_HIGHEST)

    dA_total = dA_from_w + dA_from_u

    idx = jnp.arange(C)
    strict = (idx[:, None] > idx[None, :]).astype(jnp.float32)

    tmp = jnp.dot(dA_total, A.T, precision=_HIGHEST)
    dAkk_raw = -jnp.dot(A.T, tmp, precision=_HIGHEST)
    dAkk_raw = dAkk_raw * (1.0 - wy_eps)
    dAkk = dAkk_raw * strict

    dk_from_kb = dkb_decayed * jnp.exp(gc) * b_c
    db = dkb_decayed * jnp.exp(gc) * k_c
    dgc_from_kb = dkb_decayed * kb_decayed

    dx = dkg * kg
    dk_from_kg = dkg * jnp.exp(gc_last[None, :] - gc)
    dgc_from_kg = -dx
    dgc_last_contrib = jnp.sum(dx, axis=0)

    dq = dqg * jnp.exp(gc)
    dgc_from_qg = dqg * qg

    dw = dwv * v_c
    dv_raw = dwv * w_c

    dk = dk_from_kb + dk_from_kg
    dgc = dgc_from_kb + dgc_from_qg + dgc_from_kg

    decay_h_row = jnp.exp(gc_last)
    if config.score_layout == ScoreLayout.FEATURE_FIRST:
        # Keep the reduction result in lanes, matching the cumulative-decay
        # gradient. The row-reduced layout requires unsupported v4 gathers.
        dgc_last_from_decay = decay_h_row * jnp.sum((dh_next * h_pre).T, axis=0)
    else:
        dgc_last_from_decay = decay_h_row * jnp.sum(dh_next * h_pre, axis=-1)
    dgc_last_total = dgc_last_contrib + dgc_last_from_decay

    row_mask = (idx == (C - 1)).astype(jnp.float32)[:, None]
    dgc = dgc + row_mask * dgc_last_total[None, :]

    dq_ref[0, 0, 0] = dq
    dk_ref[0, 0, 0] = dk
    db_ref[0, 0, 0] = db
    dw_ref[0, 0, 0] = dw
    dvraw_ref[0, 0, 0] = dv_raw
    dakk_ref[0, 0, 0] = dAkk
    dgc_ref[0, 0, 0] = dgc


def wy_dqkg_backward_pallas(
    q, k, b, w, v, gc, A, Akk, h_pre_all, v_new_all, do, dv, dh_next_all, scale, config: KernelConfig = DEFAULT_CONFIG
):
    bsz, H, n_chunks, _BT, D = q.shape
    grid = (bsz, H, n_chunks)

    io_spec = pl.BlockSpec((1, 1, 1, config.bt, D), lambda i, h, c: (i, h, c, 0, 0))
    score_spec = pl.BlockSpec((1, 1, 1, config.bt, config.bt), lambda i, h, c: (i, h, c, 0, 0))
    h_spec = pl.BlockSpec((1, 1, 1, D, D), lambda i, h, c: (i, h, c, 0, 0))

    outputs = [jax.ShapeDtypeStruct((bsz, H, n_chunks, config.bt, D), jnp.float32)] * 6 + [
        jax.ShapeDtypeStruct((bsz, H, n_chunks, config.bt, config.bt), jnp.float32)
    ]
    dq, dk, db, dw, dv_raw, dgc, dAkk = pl.pallas_call(
        lambda *refs: _kernel_b3_body(*refs, scale=scale, bt=config.bt, wy_eps=config.wy_eps, config=config),
        name="gdn2_backward_wy",
        grid=grid,
        in_specs=[
            io_spec,
            io_spec,
            io_spec,
            io_spec,
            io_spec,
            io_spec,
            score_spec,
            score_spec,
            h_spec,
            io_spec,
            io_spec,
            io_spec,
            h_spec,
        ],
        out_specs=[io_spec, io_spec, io_spec, io_spec, io_spec, io_spec, score_spec],
        out_shape=outputs,
        cost_estimate=_kernel_cost(
            _kernel_b3_body,
            (q, k, b, w, v, gc, A, Akk, h_pre_all, v_new_all, do, dv, dh_next_all),
            outputs,
            scale=scale,
            bt=config.bt,
            wy_eps=config.wy_eps,
            config=config,
        ),
        compiler_params=pltpu.CompilerParams(),
        interpret=config.interpret,
    )(q, k, b, w, v, gc, A, Akk, h_pre_all, v_new_all, do, dv, dh_next_all)

    return dict(dq=dq, dk=dk, db=db, dw=dw, dv_raw=dv_raw, dgc=dgc, dAkk=dAkk)


# ---------- B4 ----------
def _dL_pair_sum(dM, edecay, R):
    tmp = dM[:, :, None] * edecay
    tmp = tmp * R[None, :, :]
    return jnp.sum(tmp, axis=1)


def _dR_pair_sum(dM, edecay, L):
    tmp = dM[:, :, None] * edecay
    tmp = tmp * L[:, None, :]
    return jnp.sum(tmp, axis=0)


def _dgc_pair_sum(dM, edecay, L, R):
    weight = dM[:, :, None] * L[:, None, :] * R[None, :, :] * edecay
    dgc_i = jnp.sum(weight, axis=1)
    dgc_j = -jnp.sum(weight, axis=0)
    return dgc_i, dgc_j


def _diagonal_intra_grads(dM_qk, dM_kk, L_qk, L_kk, R, gc):
    """Return query/key derivatives for a causal diagonal score block."""
    bc = gc.shape[0]
    # Bound pairwise temporaries to 16 query rows for v4's smaller VMEM.
    rows_per_tile = min(16, bc)
    dL_qk_tiles = []
    dL_kk_tiles = []
    dR_qk = jnp.zeros_like(R)
    dR_kk = jnp.zeros_like(R)
    for start in range(0, bc, rows_per_tile):
        end = start + rows_per_tile
        decay_diff = gc[start:end, None, :] - gc[None, :, :]
        row = jax.lax.broadcasted_iota(jnp.int32, decay_diff.shape, 0) + start
        col = jax.lax.broadcasted_iota(jnp.int32, decay_diff.shape, 1)
        edecay = jnp.exp(jnp.where(row >= col, decay_diff, -jnp.inf))
        dL_qk_tiles.append(_dL_pair_sum(dM_qk[start:end], edecay, R))
        dL_kk_tiles.append(_dL_pair_sum(dM_kk[start:end], edecay, R))
        dR_qk = dR_qk + _dR_pair_sum(dM_qk[start:end], edecay, L_qk[start:end])
        dR_kk = dR_kk + _dR_pair_sum(dM_kk[start:end], edecay, L_kk[start:end])
    return jnp.concatenate(dL_qk_tiles), dR_qk, jnp.concatenate(dL_kk_tiles), dR_kk


def _kernel_b4_body(
    q_ref,
    k_ref,
    b_ref,
    g_ref,
    daqk_ref,
    dakk_ref,
    dq_ref,
    dk_ref,
    db_ref,
    dgc_ref,
    *,
    scale: float,
    bt: int,
    bc: int,
    n_sub: int,
    config: KernelConfig,
):
    q_full = q_ref[0, 0, 0].astype(jnp.float32)
    k_full = k_ref[0, 0, 0].astype(jnp.float32)
    b_full = b_ref[0, 0, 0].astype(jnp.float32)
    g_raw = g_ref[0, 0, 0].astype(jnp.float32)
    dAqk = daqk_ref[0, 0, 0].astype(jnp.float32)
    dAkk = dakk_ref[0, 0, 0].astype(jnp.float32)

    bt_idx = jnp.arange(bt)
    tril_ones_bt = (bt_idx[:, None] >= bt_idx[None, :]).astype(jnp.float32)
    gc = jnp.dot(tril_ones_bt, g_raw, precision=_HIGHEST)

    bk_full = b_full * k_full

    dq_ref[0, 0, 0] = jnp.zeros_like(q_full)
    dk_ref[0, 0, 0] = jnp.zeros_like(k_full)
    db_ref[0, 0, 0] = jnp.zeros_like(k_full)
    dgc_ref[0, 0, 0] = jnp.zeros_like(g_raw)

    for si in range(n_sub):
        for sj in range(si + 1):
            i0, i1 = si * bc, (si + 1) * bc
            j0, j1 = sj * bc, (sj + 1) * bc

            q_i = q_full[i0:i1]
            k_j = k_full[j0:j1]
            bk_i = bk_full[i0:i1]
            gc_i = gc[i0:i1]
            gc_j = gc[j0:j1]

            dM_qk = dAqk[i0:i1, j0:j1]
            dM_kk = dAkk[i0:i1, j0:j1]
            if si == sj:
                idx = jnp.arange(bc)
                causal = (idx[:, None] >= idx[None, :]).astype(jnp.float32)
                strict = (idx[:, None] > idx[None, :]).astype(jnp.float32)
                dM_qk = dM_qk * causal
                dM_kk = dM_kk * strict

            L_qk = scale * q_i
            R_qk = k_j
            L_kk = bk_i
            R_kk = k_j
            if si > sj:
                center = gc_j[-1:]
                left = jnp.exp(gc_i - center)
                right = jnp.exp(center - gc_j)
                dL_qk = jnp.dot(dM_qk, R_qk * right, precision=_HIGHEST) * left
                dR_qk = jnp.dot(dM_qk.T, L_qk * left, precision=_HIGHEST) * right
                dL_kk = jnp.dot(dM_kk, R_kk * right, precision=_HIGHEST) * left
                dR_kk = jnp.dot(dM_kk.T, L_kk * left, precision=_HIGHEST) * right
                dgc_i_qk, dgc_j_qk = dL_qk * L_qk, -dR_qk * R_qk
                dgc_i_kk, dgc_j_kk = dL_kk * L_kk, -dR_kk * R_kk
            elif config.score_layout == ScoreLayout.FEATURE_FIRST:
                dL_qk, dR_qk, dL_kk, dR_kk = _diagonal_intra_grads(dM_qk, dM_kk, L_qk, L_kk, R_qk, gc_i)
                dgc_i_qk, dgc_j_qk = dL_qk * L_qk, -dR_qk * R_qk
                dgc_i_kk, dgc_j_kk = dL_kk * L_kk, -dR_kk * R_kk
            else:
                decay_diff = gc_i[:, None, :] - gc_j[None, :, :]
                row = jax.lax.broadcasted_iota(jnp.int32, decay_diff.shape, 0)
                col = jax.lax.broadcasted_iota(jnp.int32, decay_diff.shape, 1)
                edecay = jnp.exp(jnp.where(row >= col, decay_diff, -jnp.inf))
                dL_qk = _dL_pair_sum(dM_qk, edecay, R_qk)
                dR_qk = _dR_pair_sum(dM_qk, edecay, L_qk)
                dgc_i_qk, dgc_j_qk = _dgc_pair_sum(dM_qk, edecay, L_qk, R_qk)
                dL_kk = _dL_pair_sum(dM_kk, edecay, R_kk)
                dR_kk = _dR_pair_sum(dM_kk, edecay, L_kk)
                dgc_i_kk, dgc_j_kk = _dgc_pair_sum(dM_kk, edecay, L_kk, R_kk)

            dq_ref[0, 0, 0, i0:i1] = dq_ref[0, 0, 0, i0:i1] + dL_qk * scale
            db_ref[0, 0, 0, i0:i1] = db_ref[0, 0, 0, i0:i1] + dL_kk
            dk_ref[0, 0, 0, j0:j1] = dk_ref[0, 0, 0, j0:j1] + dR_qk + dR_kk
            dgc_ref[0, 0, 0, i0:i1] = dgc_ref[0, 0, 0, i0:i1] + dgc_i_qk + dgc_i_kk
            dgc_ref[0, 0, 0, j0:j1] = dgc_ref[0, 0, 0, j0:j1] + dgc_j_qk + dgc_j_kk

    dbk_final = db_ref[0, 0, 0]
    dk_final = dk_ref[0, 0, 0] + dbk_final * b_full
    db_final = dbk_final * k_full
    dq_final = dq_ref[0, 0, 0]
    dgc_final = dgc_ref[0, 0, 0]

    dq_ref[0, 0, 0] = dq_final
    dk_ref[0, 0, 0] = dk_final
    db_ref[0, 0, 0] = db_final
    dgc_ref[0, 0, 0] = dgc_final


def intra_backward_pallas(
    dAqk, dAkk, q, k, b, g, scale, config: KernelConfig = DEFAULT_CONFIG, interpret: bool = False
):
    bsz, L, H, D = q.shape
    n_chunks = L // config.bt

    def reshape_in(t):
        return _r2c(t, bsz, n_chunks, H, D, config.bt)

    q_r, k_r, b_r, g_r = map(reshape_in, (q, k, b, g))

    grid = (bsz, H, n_chunks)
    io_spec = pl.BlockSpec((1, 1, 1, config.bt, D), lambda i, h, c: (i, h, c, 0, 0))
    score_spec = pl.BlockSpec((1, 1, 1, config.bt, config.bt), lambda i, h, c: (i, h, c, 0, 0))

    outputs = [jax.ShapeDtypeStruct((bsz, H, n_chunks, config.bt, D), jnp.float32)] * 4
    dq, dk, db, dgc = pl.pallas_call(
        lambda *refs: _kernel_b4_body(
            *refs,
            scale=scale,
            bt=config.bt,
            bc=config.bc,
            n_sub=config.n_sub,
            config=config,
        ),
        grid=grid,
        name="gdn2_backward_intra",
        in_specs=[io_spec, io_spec, io_spec, io_spec, score_spec, score_spec],
        out_specs=[io_spec, io_spec, io_spec, io_spec],
        out_shape=outputs,
        cost_estimate=_kernel_cost(
            _kernel_b4_body,
            (q_r, k_r, b_r, g_r, dAqk, dAkk),
            outputs,
            scale=scale,
            bt=config.bt,
            bc=config.bc,
            n_sub=config.n_sub,
            config=config,
        ),
        compiler_params=pltpu.CompilerParams(),
        interpret=interpret or config.interpret,
    )(q_r, k_r, b_r, g_r, dAqk, dAkk)

    return dq, dk, db, dgc
