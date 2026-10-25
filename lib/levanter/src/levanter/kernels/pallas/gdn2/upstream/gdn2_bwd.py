"""
Backward kernels: B1 (state) -> B2 (dAqk/dv) -> B3 (WY/dqkg) -> B4 (intra) -> B5 (reverse cumsum).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .configs import (
    KernelConfig, DEFAULT_CONFIG, sanitize, clip_acc,
    _reshape_to_chunks as _r2c, _reshape_from_chunks as _r2f,
)

_HIGHEST = jax.lax.Precision.HIGHEST


# ---------- B5 ----------
def reverse_cumsum_bwd(dgc, chunk_size: int, config: KernelConfig = DEFAULT_CONFIG):
    C = chunk_size
    idx = jnp.arange(C)
    triu_ones = (idx[:, None] <= idx[None, :]).astype(jnp.float32)
    dg_raw = jnp.einsum("ij,...jd->...id", triu_ones, dgc.astype(jnp.float32), precision=_HIGHEST)
    return sanitize(dg_raw, config)


# ---------- B1 ----------
def gdn2_dhu_backward(do, dv_partial, w_pseudo, qg, kg, gc_last, scale, dht=None,
                       config: KernelConfig = DEFAULT_CONFIG):
    bsz, H, n_chunks, BT, D = qg.shape
    if dht is None:
        dht = jnp.zeros((bsz, H, D, D), dtype=jnp.float32)
    dht = sanitize(dht, config)

    to_scan = tuple(jnp.moveaxis(x, 2, 0) for x in (do, dv_partial, w_pseudo, qg, kg, gc_last))

    def step(dh_carry, inputs):
        do_c, dvp_c, wp_c, qg_c, kg_c, gclast_c = inputs
        decay_c = jnp.exp(gclast_c)[..., None]

        dqh = scale * do_c
        contrib_from_output = jnp.einsum("bhid,bhiv->bhdv", qg_c, dqh, precision=_HIGHEST)
        contrib_from_state = dh_carry * decay_c

        dv_write = jnp.einsum("bhid,bhdv->bhiv", kg_c, dh_carry, precision=_HIGHEST)
        dv_new_c = dvp_c + dv_write
        dv_new_c = sanitize(dv_new_c, config)

        contrib_from_vnew = -jnp.einsum("bhjd,bhjv->bhdv", wp_c, dv_new_c, precision=_HIGHEST)

        dh_pre_c = contrib_from_output + contrib_from_state + contrib_from_vnew
        dh_pre_c = sanitize(dh_pre_c, config)
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

    daqk_ref[0, 0, 0] = sanitize(dAqk, config)
    dvnew_ref[0, 0, 0] = sanitize(dv_new, config)


def dav_backward_pallas(Aqk, v_new, do, config: KernelConfig = DEFAULT_CONFIG):
    bsz, H, n_chunks, _BT, D = v_new.shape
    grid = (bsz, H, n_chunks)
    aqk_spec = pl.BlockSpec((1, 1, 1, config.bt, config.bt), lambda i, h, c: (i, h, c, 0, 0))
    io_spec = pl.BlockSpec((1, 1, 1, config.bt, D), lambda i, h, c: (i, h, c, 0, 0))

    dAqk, dv_new = pl.pallas_call(
        lambda *refs: _kernel_b2_body(*refs, bt=config.bt, config=config),
        grid=grid,
        in_specs=[aqk_spec, io_spec, io_spec],
        out_specs=[aqk_spec, io_spec],
        out_shape=[
            jax.ShapeDtypeStruct(Aqk.shape, jnp.float32),
            jax.ShapeDtypeStruct(v_new.shape, jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=64 * 1024 * 1024),
    )(Aqk, v_new, do)
    return dAqk, dv_new


# ---------- B3 ----------
def _kernel_b3_body(q_ref, k_ref, b_ref, w_ref, v_ref, gc_ref, a_ref, akk_ref,
                     hpre_ref, vnew_ref, do_ref, dv_ref, dhnext_ref,
                     dq_ref, dk_ref, db_ref, dw_ref, dvraw_ref, dgc_ref, dakk_ref,
                     *, scale: float, bt: int, wy_eps: float, config: KernelConfig):
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
    dA_total = sanitize(dA_total, config)

    idx = jnp.arange(C)
    strict = (idx[:, None] > idx[None, :]).astype(jnp.float32)

    tmp = jnp.dot(dA_total, A.T, precision=_HIGHEST)
    tmp = sanitize(tmp, config)
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
    dgc_last_from_decay = decay_h_row * jnp.sum(dh_next * h_pre, axis=-1)
    dgc_last_total = dgc_last_contrib + dgc_last_from_decay

    row_mask = (idx == (C - 1)).astype(jnp.float32)[:, None]
    dgc = dgc + row_mask * dgc_last_total[None, :]

    dq_ref[0, 0, 0] = sanitize(dq, config)
    dk_ref[0, 0, 0] = sanitize(dk, config)
    db_ref[0, 0, 0] = sanitize(db, config)
    dw_ref[0, 0, 0] = sanitize(dw, config)
    dvraw_ref[0, 0, 0] = sanitize(dv_raw, config)
    dakk_ref[0, 0, 0] = sanitize(dAkk, config)
    dgc_ref[0, 0, 0] = sanitize(dgc, config)


def wy_dqkg_backward_pallas(q, k, b, w, v, gc, A, Akk, h_pre_all, v_new_all,
                             do, dv, dh_next_all, scale, config: KernelConfig = DEFAULT_CONFIG):
    bsz, H, n_chunks, _BT, D = q.shape
    grid = (bsz, H, n_chunks)

    io_spec = pl.BlockSpec((1, 1, 1, config.bt, D), lambda i, h, c: (i, h, c, 0, 0))
    score_spec = pl.BlockSpec((1, 1, 1, config.bt, config.bt), lambda i, h, c: (i, h, c, 0, 0))
    h_spec = pl.BlockSpec((1, 1, 1, D, D), lambda i, h, c: (i, h, c, 0, 0))

    dq, dk, db, dw, dv_raw, dgc, dAkk = pl.pallas_call(
        lambda *refs: _kernel_b3_body(*refs, scale=scale, bt=config.bt, wy_eps=config.wy_eps, config=config),
        grid=grid,
        in_specs=[io_spec, io_spec, io_spec, io_spec, io_spec, io_spec,
                   score_spec, score_spec, h_spec, io_spec, io_spec, io_spec, h_spec],
        out_specs=[io_spec, io_spec, io_spec, io_spec, io_spec, io_spec, score_spec],
        out_shape=[
            jax.ShapeDtypeStruct((bsz, H, n_chunks, config.bt, D), jnp.float32),
            jax.ShapeDtypeStruct((bsz, H, n_chunks, config.bt, D), jnp.float32),
            jax.ShapeDtypeStruct((bsz, H, n_chunks, config.bt, D), jnp.float32),
            jax.ShapeDtypeStruct((bsz, H, n_chunks, config.bt, D), jnp.float32),
            jax.ShapeDtypeStruct((bsz, H, n_chunks, config.bt, D), jnp.float32),
            jax.ShapeDtypeStruct((bsz, H, n_chunks, config.bt, D), jnp.float32),
            jax.ShapeDtypeStruct((bsz, H, n_chunks, config.bt, config.bt), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=100 * 1024 * 1024),
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


def _dgc_pair_sum(dM, edecay, L, R, clipmask):
    weight = dM[:, :, None] * L[:, None, :] * R[None, :, :] * edecay * clipmask
    dgc_i = jnp.sum(weight, axis=1)
    dgc_j = -jnp.sum(weight, axis=0)
    return dgc_i, dgc_j


def _kernel_b4_body(q_ref, k_ref, b_ref, g_ref, daqk_ref, dakk_ref,
                     dq_ref, dk_ref, db_ref, dgc_ref, *, scale: float, bt: int, bc: int, n_sub: int,
                     config: KernelConfig):
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
            k_i = k_full[i0:i1]
            k_j = k_full[j0:j1]
            b_i = b_full[i0:i1]
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

            decay_diff = gc_i[:, None, :] - gc_j[None, :, :]
            clipmask = ((decay_diff >= -20.0) & (decay_diff <= 20.0)).astype(jnp.float32)
            edecay = jnp.exp(jnp.clip(decay_diff, -20.0, 20.0))

            L_qk = scale * q_i
            R_qk = k_j
            dL_qk = _dL_pair_sum(dM_qk, edecay, R_qk)
            dR_qk = _dR_pair_sum(dM_qk, edecay, L_qk)
            dgc_i_qk, dgc_j_qk = _dgc_pair_sum(dM_qk, edecay, L_qk, R_qk, clipmask)

            L_kk = bk_i
            R_kk = k_j
            dL_kk = _dL_pair_sum(dM_kk, edecay, R_kk)
            dR_kk = _dR_pair_sum(dM_kk, edecay, L_kk)
            dgc_i_kk, dgc_j_kk = _dgc_pair_sum(dM_kk, edecay, L_kk, R_kk, clipmask)

            dq_ref[0, 0, 0, i0:i1] = clip_acc(dq_ref[0, 0, 0, i0:i1] + dL_qk * scale, config)
            db_ref[0, 0, 0, i0:i1] = clip_acc(db_ref[0, 0, 0, i0:i1] + dL_kk, config)
            dk_ref[0, 0, 0, j0:j1] = clip_acc(dk_ref[0, 0, 0, j0:j1] + dR_qk + dR_kk, config)
            dgc_ref[0, 0, 0, i0:i1] = clip_acc(dgc_ref[0, 0, 0, i0:i1] + dgc_i_qk + dgc_i_kk, config)
            dgc_ref[0, 0, 0, j0:j1] = clip_acc(dgc_ref[0, 0, 0, j0:j1] + dgc_j_qk + dgc_j_kk, config)

    dbk_final = db_ref[0, 0, 0]
    dk_final = dk_ref[0, 0, 0] + dbk_final * b_full
    db_final = dbk_final * k_full
    dq_final = dq_ref[0, 0, 0]
    dgc_final = dgc_ref[0, 0, 0]

    dq_ref[0, 0, 0] = sanitize(dq_final, config)
    dk_ref[0, 0, 0] = sanitize(dk_final, config)
    db_ref[0, 0, 0] = sanitize(db_final, config)
    dgc_ref[0, 0, 0] = sanitize(dgc_final, config)

def intra_backward_pallas(dAqk, dAkk, q, k, b, g, scale, config: KernelConfig = DEFAULT_CONFIG, interpret: bool = False):
    bsz, L, H, D = q.shape
    n_chunks = L // config.bt

    def reshape_in(t):
        return _r2c(t, bsz, n_chunks, H, D, config.bt)

    q_r, k_r, b_r, g_r = map(reshape_in, (q, k, b, g))

    grid = (bsz, H, n_chunks)
    io_spec = pl.BlockSpec((1, 1, 1, config.bt, D), lambda i, h, c: (i, h, c, 0, 0))
    score_spec = pl.BlockSpec((1, 1, 1, config.bt, config.bt), lambda i, h, c: (i, h, c, 0, 0))

    dq, dk, db, dgc = pl.pallas_call(
        lambda *refs: _kernel_b4_body(
            *refs, scale=scale, bt=config.bt, bc=config.bc, n_sub=config.n_sub,
            config=config,
        ),
        grid=grid,
        in_specs=[io_spec, io_spec, io_spec, io_spec, score_spec, score_spec],
        out_specs=[io_spec, io_spec, io_spec, io_spec],
        out_shape=[
            jax.ShapeDtypeStruct((bsz, H, n_chunks, config.bt, D), jnp.float32),
            jax.ShapeDtypeStruct((bsz, H, n_chunks, config.bt, D), jnp.float32),
            jax.ShapeDtypeStruct((bsz, H, n_chunks, config.bt, D), jnp.float32),
            jax.ShapeDtypeStruct((bsz, H, n_chunks, config.bt, D), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=150 * 1024 * 1024),
        interpret=interpret,
    )(q_r, k_r, b_r, g_r, dAqk, dAkk)

    return dq, dk, db, dgc
