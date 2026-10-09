# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Parallax attention (arXiv 2605.29157) as Pallas Triton GPU kernels, forward and backward.

A port of the official ``parallax/triton/parallax_train.py`` onto JAX's Pallas Triton backend, in the layout of
``jax.experimental.pallas.ops.gpu.attention`` (``[B, L, H, D]``, one program per query or key block per head).

Forward: the FlashAttention online softmax with two extra accumulators. Per query row, ``w = exp2(qk - m)`` (the
log2-domain softmax numerator), ``t = r . k`` (unscaled), and

    d1 = sum w,  d2 = sum w t,  O1 = sum w v,  O2 = sum w t v,

all four rescaled by the same ``alpha = exp2(m_old - m_new)`` when the running max moves (``t`` carries no
exponential). Then ``barv = O1 / d1``, ``bart = d2 / d1`` and ``o = barv (1 + bart) - O2 / d1``. Saved for the
backward: ``o``, ``barv`` (input dtype) and fp32 ``m`` (log2 units), ``d1``, ``bart``.

Backward (closed form, the paper's Appendix D): with ``p = w / d1``, ``a_ij = dO_i . v_j``, ``tau_i = dO_i . o_i``,
``beta_i = dO_i . barv_i`` and ``delta = a - beta``,

    g1 = p (a - tau + (bart - t) delta),   g2 = -p delta,
    dQ = scale sum_j g1 k_j,  dR = sum_j g2 k_j,
    dK = scale sum_i g1 q_i + sum_i g2 r_i,  dV = sum_i p (1 + bart - t) dO_i.

A preprocess computes ``tau`` and ``beta``; a column pass (one program per key block per query head) accumulates
dK and dV; a row pass (one program per query block) accumulates dQ and dR. Under GQA the column pass writes dK and
dV per query head and the wrapper sums each KV group. ``HEAD_DIM`` must be a power of two.
"""

import dataclasses
import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)
_LOG2E = math.log2(math.e)


@dataclasses.dataclass(frozen=True)
class BlockSizes:
    """Query/key tiles of the forward, column (dK, dV) and row (dQ, dR) passes, and their warp counts. The defaults
    were the fastest of 20 configurations on one H100 at B 4, L 4096, H 16/8, D 128 (forward 1.11 ms, forward plus
    backward 5.40 ms)."""

    block_q: int = 64
    block_k: int = 64
    block_q_dkv: int = 32
    block_kv_dkv: int = 64
    block_q_dq: int = 64
    block_kv_dq: int = 32
    num_warps_fwd: int = 4
    num_warps_bwd: int = 4
    num_stages: int = 2


def _mask(q_idx, k_idx, q_seg, k_seg):
    keep = q_idx[:, None] >= k_idx[None, :]
    if q_seg is not None:
        keep = jnp.logical_and(keep, q_seg[:, None] == k_seg[None, :])
    return keep


def _forward_kernel(q_ref, r_ref, k_ref, v_ref, seg_ref, o_ref, barv_ref, m_ref, d1_ref, bart_ref, *, scale, bq, bk):
    start_q = pl.program_id(0)
    head_dim = q_ref.shape[-1]
    q = q_ref[...]
    r = r_ref[...]
    q_idx = start_q * bq + jnp.arange(bq)
    q_seg = None if seg_ref is None else seg_ref[pl.ds(start_q * bq, bq)]
    qk_scale = scale * _LOG2E

    def body(start_k, carry):
        m, d1, d2, o1, o2 = carry
        k_slice = pl.ds(start_k * bk, bk)
        k = k_ref[k_slice, :]
        v = v_ref[k_slice, :]
        qk = plgpu.dot(q, k.T).astype(jnp.float32) * qk_scale
        k_seg = None if seg_ref is None else seg_ref[k_slice]
        qk = jnp.where(_mask(q_idx, start_k * bk + jnp.arange(bk), q_seg, k_seg), qk, _MASK_VALUE)
        m_new = jnp.maximum(m, jnp.max(qk, axis=-1))
        alpha = jnp.exp2(m - m_new)
        w = jnp.exp2(qk - m_new[:, None])
        wt = w * plgpu.dot(r, k.T).astype(jnp.float32)
        d1 = alpha * d1 + jnp.sum(w, axis=-1)
        d2 = alpha * d2 + jnp.sum(wt, axis=-1)
        o1 = alpha[:, None] * o1 + plgpu.dot(w.astype(v.dtype), v).astype(jnp.float32)
        o2 = alpha[:, None] * o2 + plgpu.dot(wt.astype(v.dtype), v).astype(jnp.float32)
        return m_new, d1, d2, o1, o2

    init = (
        jnp.full((bq,), -jnp.inf, jnp.float32),
        jnp.zeros((bq,), jnp.float32),
        jnp.zeros((bq,), jnp.float32),
        jnp.zeros((bq, head_dim), jnp.float32),
        jnp.zeros((bq, head_dim), jnp.float32),
    )
    upper = lax.div(bq * (start_q + 1) + bk - 1, bk)
    m, d1, d2, o1, o2 = lax.fori_loop(0, upper, body, init)
    inv_d1 = 1.0 / d1
    barv = o1 * inv_d1[:, None]
    bart = d2 * inv_d1
    o = barv * (1.0 + bart[:, None]) - o2 * inv_d1[:, None]
    o_ref[...] = o.astype(o_ref.dtype)
    barv_ref[...] = barv.astype(barv_ref.dtype)
    m_ref[...] = m
    d1_ref[...] = d1
    bart_ref[...] = bart


def _preprocess_kernel(o_ref, barv_ref, do_ref, tau_ref, beta_ref):
    do = do_ref[...].astype(jnp.float32)
    tau_ref[...] = jnp.sum(o_ref[...].astype(jnp.float32) * do, axis=-1)
    beta_ref[...] = jnp.sum(barv_ref[...].astype(jnp.float32) * do, axis=-1)


def _coefficients(q, r, k, v, do, m, d1, bart, tau, beta, keep, scale):
    """``(p, t, g1, g2)`` of one ``[rows, cols]`` tile (masked entries have ``p = 0``)."""
    qk = jnp.where(keep, plgpu.dot(q, k.T).astype(jnp.float32) * (scale * _LOG2E), _MASK_VALUE)
    p = jnp.exp2(qk - m[:, None]) / d1[:, None]
    t = plgpu.dot(r, k.T).astype(jnp.float32)
    a = plgpu.dot(do, v.T).astype(jnp.float32)
    delta = a - beta[:, None]
    g1 = p * (a - tau[:, None] + (bart[:, None] - t) * delta)
    g2 = -p * delta
    return p, t, g1, g2


def _dkv_kernel(
    q_ref,
    r_ref,
    k_ref,
    v_ref,
    seg_ref,
    do_ref,
    m_ref,
    d1_ref,
    bart_ref,
    tau_ref,
    beta_ref,
    dk_ref,
    dv_ref,
    *,
    scale,
    bq,
    bk,
):
    start_k = pl.program_id(0)
    head_dim = k_ref.shape[-1]
    seq = q_ref.shape[0]
    k = k_ref[...]
    v = v_ref[...]
    k_idx = start_k * bk + jnp.arange(bk)
    k_seg = None if seg_ref is None else seg_ref[pl.ds(start_k * bk, bk)]

    def body(start_q, carry):
        dk, dv = carry
        q_slice = pl.ds(start_q * bq, bq)
        q = q_ref[q_slice, :]
        r = r_ref[q_slice, :]
        do = do_ref[q_slice, :]
        q_seg = None if seg_ref is None else seg_ref[q_slice]
        keep = _mask(start_q * bq + jnp.arange(bq), k_idx, q_seg, k_seg)
        p, t, g1, g2 = _coefficients(
            q,
            r,
            k,
            v,
            do,
            m_ref[q_slice],
            d1_ref[q_slice],
            bart_ref[q_slice],
            tau_ref[q_slice],
            beta_ref[q_slice],
            keep,
            scale,
        )
        dk = dk + plgpu.dot((g1 * scale).astype(q.dtype).T, q).astype(jnp.float32)
        dk = dk + plgpu.dot(g2.astype(r.dtype).T, r).astype(jnp.float32)
        weights = p * (1.0 + bart_ref[q_slice][:, None] - t)
        dv = dv + plgpu.dot(weights.astype(do.dtype).T, do).astype(jnp.float32)
        return dk, dv

    init = (jnp.zeros((bk, head_dim), jnp.float32), jnp.zeros((bk, head_dim), jnp.float32))
    dk, dv = lax.fori_loop(lax.div(start_k * bk, bq), pl.cdiv(seq, bq), body, init)
    dk_ref[...] = dk.astype(dk_ref.dtype)
    dv_ref[...] = dv.astype(dv_ref.dtype)


def _dq_kernel(
    q_ref,
    r_ref,
    k_ref,
    v_ref,
    seg_ref,
    do_ref,
    m_ref,
    d1_ref,
    bart_ref,
    tau_ref,
    beta_ref,
    dq_ref,
    dr_ref,
    *,
    scale,
    bq,
    bk,
):
    start_q = pl.program_id(0)
    head_dim = q_ref.shape[-1]
    q = q_ref[...]
    r = r_ref[...]
    do = do_ref[...]
    m, d1, bart, tau, beta = m_ref[...], d1_ref[...], bart_ref[...], tau_ref[...], beta_ref[...]
    q_idx = start_q * bq + jnp.arange(bq)
    q_seg = None if seg_ref is None else seg_ref[pl.ds(start_q * bq, bq)]

    def body(start_k, carry):
        dq, dr = carry
        k_slice = pl.ds(start_k * bk, bk)
        k = k_ref[k_slice, :]
        v = v_ref[k_slice, :]
        k_seg = None if seg_ref is None else seg_ref[k_slice]
        keep = _mask(q_idx, start_k * bk + jnp.arange(bk), q_seg, k_seg)
        _, _, g1, g2 = _coefficients(q, r, k, v, do, m, d1, bart, tau, beta, keep, scale)
        dq = dq + plgpu.dot(g1.astype(k.dtype), k).astype(jnp.float32)
        dr = dr + plgpu.dot(g2.astype(k.dtype), k).astype(jnp.float32)
        return dq, dr

    init = (jnp.zeros((bq, head_dim), jnp.float32), jnp.zeros((bq, head_dim), jnp.float32))
    dq, dr = lax.fori_loop(0, pl.cdiv((start_q + 1) * bq, bk), body, init)
    dq_ref[...] = (dq * scale).astype(dq_ref.dtype)
    dr_ref[...] = dr.astype(dr_ref.dtype)


def _check(q, k, block_sizes: BlockSizes):
    batch, seq, heads, head_dim = q.shape
    if k.shape[0] != batch or k.shape[1] != seq or k.shape[3] != head_dim or heads % k.shape[2]:
        raise ValueError(f"incompatible q {q.shape} and k {k.shape} (causal self-attention, GQA)")
    if head_dim & (head_dim - 1):
        raise ValueError(f"head_dim must be a power of two, got {head_dim}")
    for name in ("block_q", "block_k", "block_q_dkv", "block_kv_dkv", "block_q_dq", "block_kv_dq"):
        if seq % getattr(block_sizes, name):
            raise ValueError(f"sequence length {seq} must be a multiple of {name}={getattr(block_sizes, name)}")


def parallax_forward(q, r, k, v, segment_ids, *, scale: float, block_sizes: BlockSizes, interpret: bool):
    """``(o, barv, m, d1, bart)``: the output and the residuals of the backward."""
    _check(q, k, block_sizes)
    batch, seq, heads, head_dim = q.shape
    reps = heads // k.shape[2]
    bq, bk = block_sizes.block_q, block_sizes.block_k
    row_block = pl.BlockSpec((None, bq, None, head_dim), lambda i, b, h: (b, i, h, 0))
    kv_full = pl.BlockSpec((None, seq, None, head_dim), lambda i, b, h: (b, 0, h // reps, 0))
    stat_block = pl.BlockSpec((None, None, bq), lambda i, b, h: (b, h, i))
    seg_spec = None if segment_ids is None else pl.BlockSpec((None, seq), lambda i, b, h: (b, 0))
    stat_shape = jax.ShapeDtypeStruct((batch, heads, seq), jnp.float32)
    return pl.pallas_call(
        functools.partial(_forward_kernel, scale=scale, bq=bq, bk=bk),
        grid=(seq // bq, batch, heads),
        in_specs=[row_block, row_block, kv_full, kv_full, seg_spec],
        out_specs=[row_block, row_block, stat_block, stat_block, stat_block],
        out_shape=[
            jax.ShapeDtypeStruct(q.shape, q.dtype),
            jax.ShapeDtypeStruct(q.shape, q.dtype),
            stat_shape,
            stat_shape,
            stat_shape,
        ],
        compiler_params=plgpu.CompilerParams(num_warps=block_sizes.num_warps_fwd, num_stages=block_sizes.num_stages),
        interpret=interpret,
        name="parallax_forward",
    )(q, r, k, v, segment_ids)


def parallax_backward(
    q, r, k, v, segment_ids, o, barv, m, d1, bart, do, *, scale: float, block_sizes: BlockSizes, interpret: bool
):
    """``(dq, dr, dk, dv)`` of :func:`parallax_forward`."""
    batch, seq, heads, head_dim = q.shape
    kv_heads = k.shape[2]
    reps = heads // kv_heads
    full_q = pl.BlockSpec((None, seq, None, head_dim), lambda i, b, h: (b, 0, h, 0))
    full_stat = pl.BlockSpec((None, None, seq), lambda i, b, h: (b, h, 0))
    seg_spec = None if segment_ids is None else pl.BlockSpec((None, seq), lambda i, b, h: (b, 0))
    stat_shape = jax.ShapeDtypeStruct((batch, heads, seq), jnp.float32)
    params = plgpu.CompilerParams(num_warps=block_sizes.num_warps_bwd, num_stages=block_sizes.num_stages)

    pre_rows = block_sizes.block_q
    pre_block = pl.BlockSpec((None, pre_rows, None, head_dim), lambda i, b, h: (b, i, h, 0))
    pre_stat = pl.BlockSpec((None, None, pre_rows), lambda i, b, h: (b, h, i))
    tau, beta = pl.pallas_call(
        _preprocess_kernel,
        grid=(seq // pre_rows, batch, heads),
        in_specs=[pre_block, pre_block, pre_block],
        out_specs=[pre_stat, pre_stat],
        out_shape=[stat_shape, stat_shape],
        compiler_params=plgpu.CompilerParams(num_warps=4, num_stages=2),
        interpret=interpret,
        name="parallax_bwd_preprocess",
    )(o, barv, do)

    bq, bk = block_sizes.block_q_dkv, block_sizes.block_kv_dkv
    kv_block = pl.BlockSpec((None, bk, None, head_dim), lambda j, b, h: (b, j, h // reps, 0))
    per_head_block = pl.BlockSpec((None, bk, None, head_dim), lambda j, b, h: (b, j, h, 0))
    per_head_shape = jax.ShapeDtypeStruct((batch, seq, heads, head_dim), jnp.float32)
    dk_heads, dv_heads = pl.pallas_call(
        functools.partial(_dkv_kernel, scale=scale, bq=bq, bk=bk),
        grid=(seq // bk, batch, heads),
        in_specs=[
            full_q,
            full_q,
            kv_block,
            kv_block,
            seg_spec,
            full_q,
            full_stat,
            full_stat,
            full_stat,
            full_stat,
            full_stat,
        ],
        out_specs=[per_head_block, per_head_block],
        out_shape=[per_head_shape, per_head_shape],
        compiler_params=params,
        interpret=interpret,
        name="parallax_bwd_dkv",
    )(q, r, k, v, segment_ids, do, m, d1, bart, tau, beta)

    bq, bk = block_sizes.block_q_dq, block_sizes.block_kv_dq
    row_block = pl.BlockSpec((None, bq, None, head_dim), lambda i, b, h: (b, i, h, 0))
    stat_block = pl.BlockSpec((None, None, bq), lambda i, b, h: (b, h, i))
    kv_full = pl.BlockSpec((None, seq, None, head_dim), lambda i, b, h: (b, 0, h // reps, 0))
    dq, dr = pl.pallas_call(
        functools.partial(_dq_kernel, scale=scale, bq=bq, bk=bk),
        grid=(seq // bq, batch, heads),
        in_specs=[
            row_block,
            row_block,
            kv_full,
            kv_full,
            seg_spec,
            row_block,
            stat_block,
            stat_block,
            stat_block,
            stat_block,
            stat_block,
        ],
        out_specs=[row_block, row_block],
        out_shape=[jax.ShapeDtypeStruct(q.shape, q.dtype), jax.ShapeDtypeStruct(r.shape, r.dtype)],
        compiler_params=params,
        interpret=interpret,
        name="parallax_bwd_dq",
    )(q, r, k, v, segment_ids, do, m, d1, bart, tau, beta)

    def per_group(x):
        return jnp.sum(x.reshape(batch, seq, kv_heads, reps, head_dim), axis=3).astype(k.dtype)

    return dq, dr, per_group(dk_heads), per_group(dv_heads)
