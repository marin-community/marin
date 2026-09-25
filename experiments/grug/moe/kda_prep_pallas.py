# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fused intra-chunk *prep* for chunked KDA as one Pallas (Triton) kernel per chunk.

``chunk_kda``'s parallel path spends most of its forward in a chain of ~15 small
per-chunk ops (gate cumsum, inflate/deflate, the delta-correction matrix ``A``, its
unit-lower-triangular inverse, the pseudo-values, the intra-chunk attention, and the
per-chunk affine transition ``(M, C)`` fed to the associative scan), each round-
tripping HBM. This module computes that whole chain for one ``(batch, chunk)`` tile
in a single kernel with every intermediate on-chip, on a fully parallel grid
``(G, n_chunks)``. The inter-chunk recurrence consumes ``(Kw, decay, K_cumdecay,
V_pseudo)`` (``kda_state_pallas`` or the associative scan in ``kda.py``).

Per chunk (rows ``r`` = tokens, all decay math fp32; ``mm`` = operands cast to
``mm_dtype`` with fp32 accumulation, exactly mirroring the XLA path in ``kda.py``),
after the optional q/k L2-norm and the ``d_k**-0.5`` query scale:

    G = cumsum(g),  Eg = exp(G),  Eng = exp(min(-G, cap)),  gt = sum(g)
    Kbe = beta*k*Eg,  Kd = k*Eng
    A = -strict_tril(mm(Kbe, Kd^T)),   T = (I - A)^-1   (log-depth block doubling)
    Vp = mm(T, beta*v),  Kcd = mm(T, Kbe),  Qi = q*Eg,  attn = tril(mm(Qi, Kd^T))
    Kw = k*exp(gt - G),  decay = exp(gt)

The backward is the hand-derived VJP of the chain above in two kernels (every GEMM,
then a purely elementwise tail), split so neither spills registers; ``T`` is saved
from the forward so the inverse is not recomputed.
"""

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt

f32 = jnp.float32

# Cap on the deflation exponent exp(-G) (shared with kda.py's XLA path, which imports it).
# The intra-chunk products are exact only while the chunk's cumulative log-decay stays above
# -DEFLATE_EXP_CAP: past it, exp(G_r) * exp(-G_i) under-counts exp(G_r - G_i) (it does not just
# drop decayed-away terms). 80 keeps exp(80) ~ 5.5e34 inside bf16/fp32 range, so a per-token
# log-decay floor of -5 (Kimi K3's -5*sigmoid gate) is exact with 16-token chunks.
DEFLATE_EXP_CAP = 80.0
L2NORM_EPS = 1e-6


class ChunkPrep(NamedTuple):
    """Per-chunk prep outputs, each shaped ``(G, n_chunks, ...)`` (``G`` = batch x heads)."""

    q_inflate: jax.Array  # (C, d_k) mm_dtype
    k_cumdecay: jax.Array  # (C, d_k) mm_dtype
    v_pseudo: jax.Array  # (C, d_v) fp32
    attn: jax.Array  # (C, C) mm_dtype, lower-triangular incl. diagonal
    kw: jax.Array  # (C, d_k) mm_dtype, keys decayed to the chunk end
    decay: jax.Array  # (d_k,) fp32 whole-chunk decay exp(g_tail)


class PrepConfig(NamedTuple):
    chunk_size: int
    mm_dtype: str
    l2norm: bool
    num_warps: int
    bwd_num_warps: int
    tail_block_d: int
    tail_num_warps: int
    interpret: bool


def _iota2_rows(n: int, m: int) -> jax.Array:
    return lax.broadcasted_iota(jnp.int32, (n, m), 0)


def _iota2(n: int, dim: int) -> jax.Array:
    return lax.broadcasted_iota(jnp.int32, (n, n), dim)


def _dot(a, b, trans_a=False, trans_b=False):
    """fp32-accumulated 2-D matmul with optional operand transposes."""
    ca = 0 if trans_a else 1
    cb = 1 if trans_b else 0
    return lax.dot_general(a, b, (((ca,), (cb,)), ((), ())), preferred_element_type=f32)


def _mm(a, b, mm_dtype, trans_a=False, trans_b=False):
    return _dot(a.astype(mm_dtype), b.astype(mm_dtype), trans_a, trans_b)


def _block_inverse(a, mm_dtype):
    """(I - A)^-1 for strictly-lower A by recursive block doubling, same op order as kda.py:
    ``T_2s = T_s + T_s (A * M_s) T_s`` with ``M_s`` the lower-left ``s x s`` block of every
    ``2s`` block. Every intermediate is a block of the true inverse (bounded even for nearly
    parallel keys, where the Neumann product overflowed). ``T`` is carried in fp32; the
    products run in ``mm_dtype``; the result is returned in ``mm_dtype``."""
    c = a.shape[0]
    rows, cols = _iota2(c, 0), _iota2(c, 1)
    inv = (rows == cols).astype(f32)
    s = 1
    while s < c:
        lower_left = (rows // (2 * s) == cols // (2 * s)) & ((rows // s) % 2 == 1) & ((cols // s) % 2 == 0)
        inv = inv + _mm(_mm(inv, jnp.where(lower_left, a, 0.0), mm_dtype), inv, mm_dtype)
        s *= 2
    return inv.astype(mm_dtype)


def _softplus(x):
    return jnp.maximum(x, 0.0) + jnp.log1p(jnp.exp(-jnp.abs(x)))


def _log_decay(g_raw, gate_refs, cols=slice(None)):
    """The per-channel log-decay of one chunk: ``g_raw`` itself, or with a fused gate
    (``gate_refs = (rate_ref, bias_ref)``, this head's rows) ``rate * softplus(g_raw + bias)``,
    the Kimi Linear / Mamba2 parameterization with ``rate = -exp(A_log)``."""
    if not gate_refs:
        return g_raw
    rate_ref, bias_ref = gate_refs
    return rate_ref[cols].astype(f32)[None, :] * _softplus(g_raw + bias_ref[cols].astype(f32)[None, :])


class _DocMasks(NamedTuple):
    same: jax.Array  # (C, C) bool: tokens in the same document
    carry_in: jax.Array  # (C,) fp32: token still in the document the chunk started in
    last_doc: jax.Array  # (C,) fp32: token in the chunk's last document
    carry_through: jax.Array  # () fp32: no document starts inside the chunk


def _doc_masks(starts_ref) -> _DocMasks | None:
    """Document masks of one chunk from its document-start flags (see kda._segment_masks).
    ``None`` when the sequence is unsegmented."""
    if starts_ref is None:
        return None
    idx = jnp.cumsum(starts_ref[...].astype(f32), axis=0)
    total = jnp.sum(starts_ref[...].astype(f32))
    return _DocMasks(
        same=idx[:, None] == idx[None, :],
        carry_in=(idx == 0.0).astype(f32),
        last_doc=(idx == total).astype(f32),
        carry_through=(total == 0.0).astype(f32),
    )


def _split_refs(refs, gated: bool, segmented: bool):
    """(gate_refs, starts_ref, rest) from a kernel's optional-then-positional refs."""
    gate_refs, refs = (refs[:2], refs[2:]) if gated else ((), refs)
    starts_ref, refs = (refs[0], refs[1:]) if segmented else (None, refs)
    return gate_refs, starts_ref, refs


def _exact_mask_matmul(mask, x, trans_mask: bool = False):
    """``mask @ x`` for a 0/1 bf16 ``mask`` at fp32 accuracy: x is split into three bf16
    parts (8+8+8 mantissa bits), each product with a 0/1 entry is exact and accumulates in
    fp32. Cheaper on registers than 3xTF32, which also splits the (exact) mask."""
    hi = x.astype(jnp.bfloat16)
    rest = x - hi.astype(f32)
    mid = rest.astype(jnp.bfloat16)
    lo = (rest - mid.astype(f32)).astype(jnp.bfloat16)
    return sum(_dot(mask, part, trans_a=trans_mask) for part in (hi, mid, lo))


def _doc_lower_mask(starts_ref):
    """(C, C) bf16 0/1: ``j <= t`` within one document (the segmented-cumsum operator)."""
    idx = jnp.cumsum(starts_ref[...].astype(f32), axis=0)
    c = idx.shape[0]
    return ((_iota2(c, 1) <= _iota2(c, 0)) & (idx[:, None] == idx[None, :])).astype(jnp.bfloat16)


def _cum_log_decay(g_ref, gate_refs, docs: _DocMasks | None, cols=slice(None)):
    """The chunk's cumulative log-decay G (C x cols). Unsegmented: cumsum of the (gated)
    log-decay. With documents, ``g_ref`` already holds G restarted at every document start
    (``_seg_decay_fwd_kernel``), which keeps the heavy kernels free of that work."""
    if docs is not None:
        return g_ref[:, cols].astype(f32)
    return jnp.cumsum(_log_decay(g_ref[:, cols].astype(f32), gate_refs, cols), axis=0)


def _gates(gcum, docs: _DocMasks | None = None):
    """(G, gt, exp(G), exp(-G) capped, exp(gt - G)) from the cumulative log-decay G; gt is
    its last row (with documents: the last document's total, the only one that feeds the
    outgoing state)."""
    c = gcum.shape[0]
    gt = jnp.sum(jnp.where(_iota2_rows(c, gcum.shape[1]) == c - 1, gcum, 0.0), axis=0)
    eg = jnp.exp(gcum)
    eng = jnp.exp(jnp.minimum(-gcum, DEFLATE_EXP_CAP))
    # <= 0 for every row that feeds the state (the last document); clamped so the masked-off
    # rows of earlier documents (whose own G can sit far above gt) cannot overflow to inf
    # and turn their 0 mask into NaN.
    w = jnp.exp(jnp.minimum(gt[None, :] - gcum, 0.0))
    return gcum, gt, eg, eng, w


def _norm_qk(x, l2norm: bool, scale: float):
    """Optional row L2-normalization then scaling; returns (y, per-row rsqrt-norm or None).

    Row statistics stay 1-D and broadcast with ``[:, None]`` at use: (C, 1) keepdims
    values hit an illegal-address miscompile in a Triton backward kernel."""
    if not l2norm:
        return x * scale, None
    r = lax.rsqrt(jnp.sum(x * x, axis=1) + L2NORM_EPS)
    return x * (r * scale)[:, None], r


def _prep_fwd_kernel(q_ref, k_ref, v_ref, g_ref, b_ref, *refs, mm_dtype, l2norm, save_t, gated, segmented):
    gate_refs, starts_ref, out_refs = _split_refs(refs, gated, segmented)
    qi_ref, kcd_ref, vp_ref, attn_ref, kw_ref, decay_ref = out_refs[:6]
    c, dk = q_ref.shape
    docs = _doc_masks(starts_ref)
    k, _ = _norm_qk(k_ref[...].astype(f32), l2norm, 1.0)
    beta = b_ref[...].astype(f32)[:, None]

    _, gt, eg, eng, w = _gates(_cum_log_decay(g_ref, gate_refs, docs), docs)
    kw = k * w
    decay = jnp.exp(gt)
    if docs is not None:  # only the last document's keys, and the old state only if no start
        kw = kw * docs.last_doc[:, None]
        decay = decay * docs.carry_through
    kw_ref[...] = kw.astype(kw_ref.dtype)
    decay_ref[...] = decay
    kbe = k * beta * eg
    kd = k * eng
    row, col = _iota2(c, 0), _iota2(c, 1)
    strict, lower = col < row, col <= row
    if docs is not None:  # intra-chunk interactions only within a document
        strict, lower = strict & docs.same, lower & docs.same

    a = jnp.where(strict, -_mm(kbe, kd, mm_dtype, trans_b=True), 0.0)
    t = _block_inverse(a, mm_dtype)
    if save_t:
        out_refs[6][...] = t.astype(out_refs[6].dtype)
    kbe_state = kbe if docs is None else kbe * docs.carry_in[:, None]  # state reaches the first doc only
    kcd_ref[...] = _mm(t, kbe_state, mm_dtype).astype(kcd_ref.dtype)
    vp_ref[...] = _mm(t, v_ref[...].astype(f32) * beta, mm_dtype)
    q, _ = _norm_qk(q_ref[...].astype(f32), l2norm, dk**-0.5)
    qi = q * eg
    attn = jnp.where(lower, _mm(qi, kd, mm_dtype, trans_b=True), 0.0)
    attn_ref[...] = attn.astype(attn_ref.dtype)
    if docs is not None:  # q_inflate only multiplies the incoming state
        qi = qi * docs.carry_in[:, None]
    qi_ref[...] = qi.astype(qi_ref.dtype)


def _wy_bwd_kernel(q_ref, k_ref, v_ref, g_ref, b_ref, *refs, mm_dtype, l2norm, gated, segmented):
    """Backward through every GEMM of the prep: Vp = T Vb, Kcd = T Kbe, T = (I - A)^-1,
    A = -tril(Kbe Kd^T), and attn = tril(Qi Kd^T).

    Emits dv, dKbe, dKd, dQi and the v-side part of dbeta, leaving only elementwise work
    for ``_tail_bwd_kernel``. Split from that tail so neither kernel spills registers (a
    single fused backward did, ~3KB/thread at C=64, d=128). Values are ordered so each
    C x d_k tile dies as early as possible."""
    gate_refs, starts_ref, refs = _split_refs(refs, gated, segmented)
    t_ref, dkcd_ref, dvp_ref, dqi_in_ref, dattn_ref, dv_ref, dkbe_ref, dkd_ref, dqi_ref, dbv_ref = refs
    c, dk = k_ref.shape
    docs = _doc_masks(starts_ref)
    beta = b_ref[...].astype(f32)[:, None]
    t = t_ref[...]
    row, col = _iota2(c, 0), _iota2(c, 1)

    dvp = dvp_ref[...].astype(f32)
    v = v_ref[...].astype(f32)
    dvb = _mm(t, dvp, mm_dtype, trans_a=True)
    dv_ref[...] = (dvb * beta).astype(dv_ref.dtype)
    dbv_ref[...] = jnp.sum(dvb * v, axis=1)
    dt = _mm(dvp, v * beta, mm_dtype, trans_b=True)

    k, _ = _norm_qk(k_ref[...].astype(f32), l2norm, 1.0)
    gcum = _cum_log_decay(g_ref, gate_refs, docs)
    eg = jnp.exp(gcum)
    kbe = k * beta * eg
    kd = k * jnp.exp(jnp.minimum(-gcum, DEFLATE_EXP_CAP))
    dkcd = dkcd_ref[...].astype(f32)
    kbe_state = kbe if docs is None else kbe * docs.carry_in[:, None]
    dt = dt + _mm(dkcd, kbe_state, mm_dtype, trans_b=True)
    dkbe = _mm(t, dkcd, mm_dtype, trans_a=True)
    strict, lower = col < row, col <= row
    if docs is not None:
        dkbe = dkbe * docs.carry_in[:, None]
        strict, lower = strict & docs.same, lower & docs.same
    # T = (I - A)^-1  =>  dA = T^T dT T^T
    da = _mm(_mm(t, dt, mm_dtype, trans_a=True), t, mm_dtype, trans_b=True)
    dp = jnp.where(strict, -da, 0.0)
    dkbe_ref[...] = dkbe + _mm(dp, kd, mm_dtype)
    dkd = _mm(dp, kbe, mm_dtype, trans_a=True)

    # attn = tril(Qi Kd^T) ;  Qi = q Eg
    q, _ = _norm_qk(q_ref[...].astype(f32), l2norm, dk**-0.5)
    ds = jnp.where(lower, dattn_ref[...].astype(f32), 0.0)
    dkd_ref[...] = dkd + _mm(ds, q * eg, mm_dtype, trans_a=True)
    dqi_in = dqi_in_ref[...].astype(f32)
    if docs is not None:
        dqi_in = dqi_in * docs.carry_in[:, None]
    dqi_ref[...] = dqi_in + _mm(ds, kd, mm_dtype)


def _tail_bwd_kernel(q_ref, k_ref, g_ref, b_ref, *refs, l2norm, block_d, gated, segmented):
    """Elementwise backward through the gating chain, the gate cumsum and the q/k L2-norm.

    Everything here is separable over d_k columns except the L2-norm row statistics, so it
    walks d_k in ``block_d`` column blocks (keeping only the pre-norm q/k cotangents live
    across blocks) instead of materializing ~15 full C x d_k tiles, which spilled.

    With a fused gate it recomputes ``g = rate * softplus(g_raw + bias)`` but still emits
    d/d(g) (fp32); the caller back-propagates the cheap elementwise gate in XLA -- doing
    that in-kernel as well pushed this kernel over its register budget (2.6x slower)."""
    gate_refs, starts_ref, refs = _split_refs(refs, gated, segmented)
    dqi_ref, dkw_ref, ddecay_ref, dkbe_ref, dkd_ref, dbv_ref, dq_ref, dk_ref, dg_ref, db_ref = refs
    c, dk = q_ref.shape
    docs = _doc_masks(starts_ref)
    blocks = [pl.ds(j * block_d, block_d) for j in range(dk // block_d)]
    beta = b_ref[...].astype(f32)[:, None]
    q_scale = dk**-0.5

    def row_rsqrt(ref):
        if not l2norm:
            return jnp.ones((c,), f32)
        sq = sum(jnp.sum(jnp.square(ref[:, blk].astype(f32)), axis=1) for blk in blocks)
        return lax.rsqrt(sq + L2NORM_EPS)

    rq, rk = row_rsqrt(q_ref), row_rsqrt(k_ref)
    dbeta = dbv_ref[...]
    dqn_blocks, dkn_blocks = [], []
    q_dot = jnp.zeros((c,), f32)
    k_dot = jnp.zeros((c,), f32)
    for blk in blocks:
        gcum, gt, eg, eng, w = _gates(_cum_log_decay(g_ref, gate_refs, docs, blk), docs)
        k_raw = k_ref[:, blk].astype(f32)
        q_raw = q_ref[:, blk].astype(f32)
        k = k_raw * rk[:, None]

        # Qi = q Eg
        dqi = dqi_ref[:, blk]
        dqn = dqi * eg * q_scale  # cotangent of the L2-normalized, unscaled q
        q_dot = q_dot + jnp.sum(q_raw * dqn, axis=1)
        dqn_blocks.append(dqn)

        # Kbe = beta k Eg ;  Kd = k Eng (no gradient where the deflation cap binds) ;
        # Kw = k exp(gt - G) ;  decay = exp(gt)
        dkbe = dkbe_ref[:, blk]
        dkd = dkd_ref[:, blk]
        dkw = dkw_ref[:, blk].astype(f32)
        if docs is not None:  # Kw = last_doc * k * exp(gt - G)
            dkw = dkw * docs.last_doc[:, None]
        kw_grad = dkw * k * w
        qi = q_raw * (rq * q_scale)[:, None] * eg
        dgc = dqi * qi + dkbe * (k * beta * eg) - dkd * k * eng * (-gcum < DEFLATE_EXP_CAP).astype(f32) - kw_grad
        dbeta = dbeta + jnp.sum(dkbe * k * eg, axis=1)
        dkn = dkbe * beta * eg + dkd * eng + dkw * w
        k_dot = k_dot + jnp.sum(k_raw * dkn, axis=1)
        dkn_blocks.append(dkn)

        ddecay = ddecay_ref[blk] if docs is None else ddecay_ref[blk] * docs.carry_through
        dgt = jnp.sum(kw_grad, axis=0) + ddecay * jnp.exp(gt)
        if docs is not None:  # emit dL/dG (gt is G's last row); _seg_decay_bwd_kernel takes it to g
            dg = jnp.where(_iota2_rows(c, dgc.shape[1]) == c - 1, dgc + dgt[None, :], dgc)
        else:  # G = cumsum(g), gt = sum(g): dg_t = sum_{r >= t} dG_r + dgt (reverse cumsum via a forward one)
            dg = jnp.sum(dgc, axis=0)[None, :] - jnp.cumsum(dgc, axis=0) + dgc + dgt[None, :]
        dg_ref[:, blk] = dg.astype(dg_ref.dtype)

    db_ref[...] = dbeta.astype(db_ref.dtype)
    # L2-norm backward: dx = r dy - x r^3 <x, dy>  (dy already includes the q scale).
    for blk, dqn, dkn in zip(blocks, dqn_blocks, dkn_blocks, strict=True):
        if l2norm:
            dqn = rq[:, None] * dqn - q_ref[:, blk].astype(f32) * (rq * rq * rq * q_dot)[:, None]
            dkn = rk[:, None] * dkn - k_ref[:, blk].astype(f32) * (rk * rk * rk * k_dot)[:, None]
        dq_ref[:, blk] = dqn.astype(dq_ref.dtype)
        dk_ref[:, blk] = dkn.astype(dk_ref.dtype)


def _seg_decay_fwd_kernel(g_ref, *refs, gated):
    """G = the (gated) log-decay's within-document cumsum for one chunk (restarted at each
    document start), so the heavy prep kernels read G instead of recomputing it."""
    gate_refs, (starts_ref, cum_ref) = (refs[:2], refs[2:]) if gated else ((), refs)
    decay = _log_decay(g_ref[...].astype(f32), gate_refs)
    cum_ref[...] = _exact_mask_matmul(_doc_lower_mask(starts_ref), decay)


def _seg_decay_bwd_kernel(g_ref, *refs, gated):
    """dL/dg from dL/dG: within-document suffix sums, then (fused gate) the chain through
    ``rate * softplus(g + bias)`` with this chunk's column sums of d/d(rate), d/d(bias)."""
    gate_refs, refs = (refs[:2], refs[2:]) if gated else ((), refs)
    starts_ref, dcum_ref, dg_ref, *dgate_refs = refs
    d_decay = _exact_mask_matmul(_doc_lower_mask(starts_ref), dcum_ref[...].astype(f32), trans_mask=True)
    if not gated:
        dg_ref[...] = d_decay.astype(dg_ref.dtype)
        return
    rate_ref, bias_ref = gate_refs
    rate = rate_ref[...].astype(f32)[None, :]
    u = g_ref[...].astype(f32) + bias_ref[...].astype(f32)[None, :]
    du = d_decay * rate * jax.nn.sigmoid(u)
    drate_ref, dbias_ref = dgate_refs
    drate_ref[...] = jnp.sum(d_decay * _softplus(u), axis=0)
    dbias_ref[...] = jnp.sum(du, axis=0)
    dg_ref[...] = du.astype(dg_ref.dtype)


def _specs(c: int, *dims: int):
    """Blocks of the internal per-chunk tensors, laid out ``(G, n, ...)``."""
    return [pl.BlockSpec((None, None, c, d), lambda gi, ni: (gi, ni, 0, 0)) for d in dims]


def _vec_spec(d: int):
    return pl.BlockSpec((None, None, d), lambda gi, ni: (gi, ni, 0))


def _seq_specs(c: int, heads: int, *dims: int, group: int = 1):
    """Chunk blocks of a model-layout ``(B, L, H', d)`` tensor for program ``(g = b*H + h, n)``,
    so the kernels read and write activations without any (B, H, L) transposes. With
    ``group > 1`` (grouped-query k/v, ``H' = H / group``) query head h reads kv head h // group."""
    return [pl.BlockSpec((None, c, None, d), lambda gi, ni: (gi // heads, ni, (gi % heads) // group, 0)) for d in dims]


def _seq_vec_spec(c: int, heads: int):
    return pl.BlockSpec((None, c, None), lambda gi, ni: (gi // heads, ni, gi % heads))


def _starts_spec(c: int, heads: int):
    """This program's chunk of the per-batch-row ``(B, L)`` document-start flags."""
    return pl.BlockSpec((None, c), lambda gi, ni: (gi // heads, ni))


def _head_row_specs(heads: int, dk: int):
    """This program's head row of the ``(H, d_k)`` gate parameters."""
    return [pl.BlockSpec((None, dk), lambda gi, ni: (gi % heads, 0)) for _ in range(2)]


def _seg_decay_fwd_call(g, gate, starts, cfg: PrepConfig):
    b, length, heads, dk = g.shape
    c = cfg.chunk_size
    gated = gate is not None
    return pl.pallas_call(
        functools.partial(_seg_decay_fwd_kernel, gated=gated),
        grid=(b * heads, length // c),
        in_specs=[*_seq_specs(c, heads, dk), *(_head_row_specs(heads, dk) if gated else []), _starts_spec(c, heads)],
        out_specs=_seq_specs(c, heads, dk)[0],
        out_shape=jax.ShapeDtypeStruct(g.shape, f32),
        compiler_params=plt.CompilerParams(num_warps=4, num_stages=1),  # Triton, not Mosaic GPU
        interpret=cfg.interpret,
        name="kda_seg_decay_fwd",
    )(g, *(gate or ()), starts)


def _seg_decay_bwd_call(g, gate, starts, dcum, cfg: PrepConfig):
    b, length, heads, dk = g.shape
    c = cfg.chunk_size
    gb, n = b * heads, length // c
    gated = gate is not None
    part = [pl.BlockSpec((None, None, dk), lambda gi, ni: (gi, ni, 0)) for _ in range(2)] if gated else []
    outs = pl.pallas_call(
        functools.partial(_seg_decay_bwd_kernel, gated=gated),
        grid=(gb, n),
        in_specs=[
            *_seq_specs(c, heads, dk),
            *(_head_row_specs(heads, dk) if gated else []),
            _starts_spec(c, heads),
            *_seq_specs(c, heads, dk),
        ],
        out_specs=[*_seq_specs(c, heads, dk), *part],
        out_shape=[
            jax.ShapeDtypeStruct(g.shape, g.dtype),
            *([jax.ShapeDtypeStruct((gb, n, dk), f32)] * 2 if gated else []),
        ],
        compiler_params=plt.CompilerParams(num_warps=4, num_stages=1),  # Triton, not Mosaic GPU
        interpret=cfg.interpret,
        name="kda_seg_decay_bwd",
    )(g, *(gate or ()), starts, dcum)
    if not gated:
        return outs[0], None
    # per-chunk column sums -> per-head (H, d_k)
    dgate = tuple(
        x.reshape(b, heads, n, dk).sum(axis=(0, 2)).astype(p.dtype) for x, p in zip(outs[1:], gate, strict=True)
    )
    return outs[0], dgate


def _prep_fwd_call(q, k, v, g, beta, gate, starts, cfg: PrepConfig, save_t: bool):
    if starts is not None:  # packed documents: the heavy kernel reads the restarted cumsum G
        g, gate = _seg_decay_fwd_call(g, gate, starts, cfg), None
    b, length, heads, dk = q.shape
    dv = v.shape[-1]
    group = heads // k.shape[2]
    c = cfg.chunk_size
    gb, n = b * heads, length // c
    mmd = jnp.dtype(cfg.mm_dtype)

    def sds(*shape, dtype=f32):
        return jax.ShapeDtypeStruct((gb, n, *shape), dtype)

    # Operands that are only ever consumed by GEMMs downstream are stored in mm_dtype.
    out_shape = [
        sds(c, dk, dtype=mmd),
        sds(c, dk, dtype=mmd),
        sds(c, dv),
        sds(c, c, dtype=mmd),
        sds(c, dk, dtype=mmd),
        sds(dk),
    ]
    out_specs = [*_specs(c, dk, dk, dv, c, dk), _vec_spec(dk)]
    if save_t:
        out_shape.append(sds(c, c, dtype=mmd))
        out_specs += _specs(c, c)
    gated, segmented = gate is not None, starts is not None
    return pl.pallas_call(
        functools.partial(
            _prep_fwd_kernel,
            mm_dtype=mmd,
            l2norm=cfg.l2norm,
            save_t=save_t,
            gated=gated,
            segmented=segmented,
        ),
        grid=(gb, n),
        in_specs=[
            *_seq_specs(c, heads, dk),
            *_seq_specs(c, heads, dk, dv, group=group),
            *_seq_specs(c, heads, dk),
            _seq_vec_spec(c, heads),
            *(_head_row_specs(heads, dk) if gated else []),
            *([_starts_spec(c, heads)] if segmented else []),
        ],
        out_specs=out_specs,
        out_shape=out_shape,
        compiler_params=plt.CompilerParams(num_warps=cfg.num_warps, num_stages=1),
        interpret=cfg.interpret,
        name="kda_prep_fwd",
    )(q, k, v, g, beta, *(gate or ()), *(() if starts is None else (starts,)))


def _group_sum(x: jax.Array, group: int, dtype) -> jax.Array:
    """Per-query-head cotangent ``(B, L, H, d)`` -> its grouped kv head ``(B, L, H/group, d)``."""
    if group == 1:
        return x.astype(dtype)
    b, length, heads, d = x.shape
    return x.reshape(b, length, heads // group, group, d).sum(axis=3).astype(dtype)


def _prep_bwd_call(q, k, v, g, beta, gate, starts, t, cts: ChunkPrep, cfg: PrepConfig):
    if starts is not None:  # heavy kernels on the restarted cumsum G; dL/dG -> dL/dg after
        g_in, gate_in = g, gate
        g, gate = _seg_decay_fwd_call(g, gate, starts, cfg), None
    b, length, heads, dk = q.shape
    dv_dim = v.shape[-1]
    group = heads // k.shape[2]
    c = cfg.chunk_size
    gb, n = b * heads, length // c
    mmd = jnp.dtype(cfg.mm_dtype)
    gated, segmented = gate is not None, starts is not None
    gate_specs = [*(_head_row_specs(heads, dk) if gated else []), *([_starts_spec(c, heads)] if segmented else [])]
    extra = (*(gate or ()), *(() if starts is None else (starts,)))
    # Grouped k/v get per-query-head cotangents (fp32), summed over each group afterwards.
    kv_ct_dtype = f32 if group > 1 else None

    def internal(*shape):
        return jax.ShapeDtypeStruct((gb, n, *shape), f32)

    dv, dkbe, dkd, dqi, dbeta_v = pl.pallas_call(
        functools.partial(
            _wy_bwd_kernel,
            mm_dtype=mmd,
            l2norm=cfg.l2norm,
            gated=gated,
            segmented=segmented,
        ),
        grid=(gb, n),
        in_specs=[
            *_seq_specs(c, heads, dk),
            *_seq_specs(c, heads, dk, dv_dim, group=group),
            *_seq_specs(c, heads, dk),
            _seq_vec_spec(c, heads),
            *gate_specs,
            *_specs(c, c, dk, dv_dim, dk, c),
        ],
        out_specs=[*_seq_specs(c, heads, dv_dim), *_specs(c, dk, dk, dk), _vec_spec(c)],
        out_shape=[
            jax.ShapeDtypeStruct((b, length, heads, dv_dim), kv_ct_dtype or v.dtype),
            internal(c, dk),
            internal(c, dk),
            internal(c, dk),
            internal(c),
        ],
        compiler_params=plt.CompilerParams(num_warps=cfg.bwd_num_warps, num_stages=1),
        interpret=cfg.interpret,
        name="kda_prep_bwd_wy",
    )(q, k, v, g, beta, *extra, t, cts.k_cumdecay, cts.v_pseudo, cts.q_inflate, cts.attn)
    dq, dk_, dg, dbeta = pl.pallas_call(
        functools.partial(
            _tail_bwd_kernel,
            l2norm=cfg.l2norm,
            block_d=min(cfg.tail_block_d, dk),
            gated=gated,
            segmented=segmented,
        ),
        grid=(gb, n),
        in_specs=[
            *_seq_specs(c, heads, dk),
            *_seq_specs(c, heads, dk, group=group),
            *_seq_specs(c, heads, dk),
            _seq_vec_spec(c, heads),
            *gate_specs,
            *_specs(c, dk, dk),
            _vec_spec(dk),
            *_specs(c, dk, dk),
            _vec_spec(c),
        ],
        out_specs=[*_seq_specs(c, heads, dk, dk, dk), _seq_vec_spec(c, heads)],
        out_shape=[
            jax.ShapeDtypeStruct(q.shape, q.dtype),
            jax.ShapeDtypeStruct((b, length, heads, dk), kv_ct_dtype or k.dtype),
            jax.ShapeDtypeStruct(g.shape, f32 if gated else g.dtype),
            jax.ShapeDtypeStruct(beta.shape, beta.dtype),
        ],
        compiler_params=plt.CompilerParams(num_warps=cfg.tail_num_warps, num_stages=1),
        interpret=cfg.interpret,
        name="kda_prep_bwd_tail",
    )(q, k, g, beta, *extra, dqi, cts.kw, cts.decay, dkbe, dkd, dbeta_v)
    dgate = None
    if segmented:
        dg, dgate = _seg_decay_bwd_call(g_in, gate_in, starts, dg, cfg)
    elif gated:  # dg is d/d(log-decay); chain through g = rate * softplus(g_raw + bias) in XLA
        rate, bias = gate
        u = g.astype(f32) + bias.astype(f32)
        dgate = (
            jnp.sum(dg * jax.nn.softplus(u), axis=(0, 1)).astype(rate.dtype),
            jnp.sum(dg * rate.astype(f32) * jax.nn.sigmoid(u), axis=(0, 1)).astype(bias.dtype),
        )
        dg = (dg * rate.astype(f32) * jax.nn.sigmoid(u)).astype(g.dtype)
    dstarts = None if starts is None else jnp.zeros_like(starts)  # document starts are data
    return dq, _group_sum(dk_, group, k.dtype), _group_sum(dv, group, v.dtype), dg, dbeta, dgate, dstarts


@functools.partial(jax.custom_vjp, nondiff_argnums=(7,))
def _prep(q, k, v, g, beta, gate, starts, cfg: PrepConfig) -> ChunkPrep:
    return ChunkPrep(*_prep_fwd_call(q, k, v, g, beta, gate, starts, cfg, save_t=False))


def _prep_fwd(q, k, v, g, beta, gate, starts, cfg):
    *outs, t = _prep_fwd_call(q, k, v, g, beta, gate, starts, cfg, save_t=True)
    prep = ChunkPrep(*outs)
    return prep, (q, k, v, g, beta, gate, starts, t)


def _prep_bwd(cfg, res, cts: ChunkPrep):
    q, k, v, g, beta, gate, starts, t = res
    # Cotangents are consumed in their own dtypes (bf16 ones are upcast on-chip).
    return _prep_bwd_call(q, k, v, g, beta, gate, starts, t, cts, cfg)


_prep.defvjp(_prep_fwd, _prep_bwd)


def fused_chunk_prep(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    *,
    chunk_size: int,
    mm_dtype: jnp.dtype,
    gate: tuple[jax.Array, jax.Array] | None = None,
    doc_starts: jax.Array | None = None,
    use_qk_l2norm: bool,
    num_warps: int = 8,
    bwd_num_warps: int = 8,
    tail_block_d: int = 32,
    tail_num_warps: int = 8,
    interpret: bool = False,
) -> ChunkPrep:
    """Fused KDA intra-chunk prep (differentiable).

    Args:
        q: ``(B, L, H, d_k)``; k: ``(B, L, H_kv, d_k)`` raw queries/keys (model layout;
            ``H_kv`` divides ``H``, grouped-query k/v are read in place). The kernel
            optionally L2-normalizes both (``use_qk_l2norm``) and scales q by ``d_k**-0.5``.
        v: ``(B, L, H_kv, d_v)``.  beta: ``(B, L, H)``.
        g: ``(B, L, H, d_k)`` log-decay, or with ``gate`` its pre-activation.
        gate: optional ``(rate, bias)``, each ``(H, d_k)``: the kernels compute the
            log-decay as ``rate * softplus(g + bias)`` on-chip (``rate = -exp(A_log)``
            broadcast over d_k) and return cotangents for both.
        doc_starts: optional ``(B, L)`` fp32 flags (1 = token starts a new document, see
            ``kda.doc_starts``): per chunk, intra-chunk interactions are masked to one
            document, the incoming state reaches only the chunk's first document, and
            only the last document's keys (with no decay across a start) feed the
            outgoing state -- a hard state reset at every document start.
        Inputs may be bf16 or fp32 (all math is fp32 on-chip; cotangents come back
        in the input dtypes). ``L`` must be a multiple of ``chunk_size``.
        mm_dtype: operand dtype of the intra-chunk GEMMs (fp32 accumulate).
        num_warps, bwd_num_warps: Triton warps per program (forward / backward kernels).
        tail_block_d, tail_num_warps: d_k column block / warps of the backward's
            elementwise tail kernel.
        interpret: run in the Pallas interpreter (CPU tests).

    Returns:
        Per-chunk tensors laid out ``(G = B*H, n = L/C, ...)`` (b-major). ``v_pseudo``
        and ``decay`` are fp32; the pure GEMM operands (``q_inflate``, ``k_cumdecay``,
        ``attn``, ``kw``) are stored in ``mm_dtype``.

    ``C``, ``d_k`` and ``d_v`` must be powers of two >= 16 (Triton tile constraint).
    """
    for name, dim in (("C", chunk_size), ("d_k", q.shape[3]), ("d_v", v.shape[3])):
        if dim < 16 or dim & (dim - 1):
            raise ValueError(f"fused_chunk_prep needs {name} a power of two >= 16, got {dim}")
    if q.shape[1] % chunk_size:
        raise ValueError(f"sequence length {q.shape[1]} must be a multiple of chunk_size={chunk_size}")
    cfg = PrepConfig(
        chunk_size,
        jnp.dtype(mm_dtype).name,
        use_qk_l2norm,
        num_warps,
        bwd_num_warps,
        tail_block_d,
        tail_num_warps,
        interpret,
    )
    return _prep(q, k, v, g, beta, gate, doc_starts, cfg)
