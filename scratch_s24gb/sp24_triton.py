# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Triton kernels: MXFP8 quantization with and without 2:4 compression, one pass over bf16 activations.

All kernels read ``x[M, K]`` bf16 once. Formats (bandwidth-faithful; the hardware's interleaved metadata order would be
a permutation of the same bytes):
- plain rows:   e4m3 ``[M, K]`` + uint8 exponents ``[M, K/32]``           (== levanter mx_quantize_rows)
- plain cols:   e4m3 ``[M, K]`` + uint8 exponents ``[M/32, K]``           (== levanter mx_quantize_cols)
- 2:4 rows:     e4m3 ``[M, K/2]`` (kept pairs along K) + metadata uint8 ``[M, K/8]`` (2-bit indices, 2 groups / byte)
                + exponents ``[M, K/64]``: one UE8M0 per 64 LOGICAL elements (32 kept values), the only scale
                granularity of sparse ``tcgen05.mma.sp.kind::mxf8f6f4`` (PTX scale_vec::1X at sparse K=64; cuSPARSELt
                VEC64_UE8M0; CUTLASS SM100_MMA_MXF8F6F4_SS_SPARSE SFVecSize=64).
- 2:4 cols:     e4m3 ``[M/2, K]`` (kept pairs along M) + metadata ``[M/8, K]`` + exponents ``[M/64, K]``.
The block amax over kept values equals the dense block amax (the max of each 4-group is always kept).

Masks: ``rows`` = per-row top-2 of 4 by |x| (plain 2:4). ``t`` = transposable 4x4 (2 per row and 2 per column), the best
of the 90 such patterns by kept |x| mass, chosen with two tensor-core dots (scores and the one-hot pattern gather).
"""

import itertools

import torch
import triton
import triton.language as tl

E4M3_MAX = 448.0


@triton.jit
def _exponent(amax):
    bits = (amax / 448.0).to(tl.int32, bitcast=True)
    e = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    e = tl.minimum(tl.maximum(e, 1), 254)
    return tl.where(amax > 0, e, 127)


@triton.jit
def _scale_of(e):
    return (e << 23).to(tl.float32, bitcast=True)


@triton.jit
def _split4(x):
    """[A, B, 4] -> four [A, B] tensors."""
    a, b = tl.split(tl.reshape(x, (x.shape[0], x.shape[1], 2, 2)))
    a0, a1 = tl.split(a)
    a2, a3 = tl.split(b)
    return a0, a2, a1, a3


@triton.jit
def _top2_keep(a0, a1, a2, a3):
    """Keep bits of the top-2 |a| of 4 (ties to the lower index)."""
    v0, v1, v2, v3 = tl.abs(a0), tl.abs(a1), tl.abs(a2), tl.abs(a3)
    # number of elements that beat i
    b0 = (v1 > v0).to(tl.int32) + (v2 > v0).to(tl.int32) + (v3 > v0).to(tl.int32)
    b1 = (v0 >= v1).to(tl.int32) + (v2 > v1).to(tl.int32) + (v3 > v1).to(tl.int32)
    b2 = (v0 >= v2).to(tl.int32) + (v1 >= v2).to(tl.int32) + (v3 > v2).to(tl.int32)
    b3 = (v0 >= v3).to(tl.int32) + (v1 >= v3).to(tl.int32) + (v2 >= v3).to(tl.int32)
    return b0 < 2, b1 < 2, b2 < 2, b3 < 2


@triton.jit
def _compress4(a0, a1, a2, a3, k0, k1, k2, k3):
    """Exactly two of k0..k3 set -> (first value, second value, nibble idx0 | idx1 << 2)."""
    first = tl.where(k0, a0, tl.where(k1, a1, a2))
    i0 = tl.where(k0, 0, tl.where(k1, 1, 2))
    second = tl.where(k3, a3, tl.where(k2, a2, a1))
    i1 = tl.where(k3, 3, tl.where(k2, 2, 1))
    return first, second, (i0 | (i1 << 2)).to(tl.int32)


@triton.jit
def _quant_rows_kernel(x_ptr, v_ptr, e_ptr, M, K, BM: tl.constexpr, BK: tl.constexpr):
    pm, pk = tl.program_id(0), tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM).to(tl.int64)
    rk = pk * BK + tl.arange(0, BK)
    x = tl.load(x_ptr + rm[:, None] * K + rk[None, :]).to(tl.float32)
    blk = tl.reshape(x, (BM, BK // 32, 32))
    e = _exponent(tl.max(tl.abs(blk), axis=2))
    q = (blk / _scale_of(e)[:, :, None]).to(tl.float8e4nv)
    tl.store(v_ptr + rm[:, None] * K + rk[None, :], tl.reshape(q, (BM, BK)))
    re = pk * (BK // 32) + tl.arange(0, BK // 32)
    tl.store(e_ptr + rm[:, None] * (K // 32) + re[None, :], e.to(tl.uint8))


@triton.jit
def _quant_rows_cols_kernel(x_ptr, rv_ptr, re_ptr, cv_ptr, ce_ptr, M, K, BK: tl.constexpr):
    """Like levanter mx_quantize_rows_cols_pallas: a [32, BK] tile, both orientations."""
    pm, pk = tl.program_id(0), tl.program_id(1)
    rm = pm * 32 + tl.arange(0, 32).to(tl.int64)
    rk = pk * BK + tl.arange(0, BK)
    x = tl.load(x_ptr + rm[:, None] * K + rk[None, :]).to(tl.float32)
    ce = _exponent(tl.max(tl.abs(x), axis=0))
    tl.store(cv_ptr + rm[:, None] * K + rk[None, :], (x / _scale_of(ce)[None, :]).to(tl.float8e4nv))
    tl.store(ce_ptr + pm * K + rk, ce.to(tl.uint8))
    blk = tl.reshape(x, (32, BK // 32, 32))
    e = _exponent(tl.max(tl.abs(blk), axis=2))
    q = (blk / _scale_of(e)[:, :, None]).to(tl.float8e4nv)
    tl.store(rv_ptr + rm[:, None] * K + rk[None, :], tl.reshape(q, (32, BK)))
    rre = pk * (BK // 32) + tl.arange(0, BK // 32)
    tl.store(re_ptr + rm[:, None] * (K // 32) + rre[None, :], e.to(tl.uint8))


@triton.jit
def _store_rows_sparse(x, k0, k1, k2, k3, v_ptr, m_ptr, e_ptr, pm, pk, K, BM: tl.constexpr, BK: tl.constexpr):
    """x [BM, BK] (fp32), keep bits [BM, BK/4] -> 2:4-along-K compressed MXFP8 values, metadata, scales."""
    a0, a1, a2, a3 = _split4(tl.reshape(x, (BM, BK // 4, 4)))
    f, s, nib = _compress4(a0, a1, a2, a3, k0, k1, k2, k3)
    comp = tl.reshape(tl.join(f, s), (BM, BK // 64, 32))
    e = _exponent(tl.max(tl.abs(comp), axis=2))
    q = tl.reshape((comp / _scale_of(e)[:, :, None]).to(tl.float8e4nv), (BM, BK // 2))
    rm = pm * BM + tl.arange(0, BM).to(tl.int64)
    rc = pk * (BK // 2) + tl.arange(0, BK // 2)
    tl.store(v_ptr + rm[:, None] * (K // 2) + rc[None, :], q)
    lo, hi = tl.split(tl.reshape(nib, (BM, BK // 8, 2)))
    rmeta = pk * (BK // 8) + tl.arange(0, BK // 8)
    tl.store(m_ptr + rm[:, None] * (K // 8) + rmeta[None, :], (lo | (hi << 4)).to(tl.uint8))
    re = pk * (BK // 64) + tl.arange(0, BK // 64)
    tl.store(e_ptr + rm[:, None] * (K // 64) + re[None, :], e.to(tl.uint8))


@triton.jit
def _sp24_rows_kernel(x_ptr, v_ptr, m_ptr, e_ptr, M, K, BM: tl.constexpr, BK: tl.constexpr):
    pm, pk = tl.program_id(0), tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM).to(tl.int64)
    rk = pk * BK + tl.arange(0, BK)
    x = tl.load(x_ptr + rm[:, None] * K + rk[None, :]).to(tl.float32)
    a0, a1, a2, a3 = _split4(tl.reshape(x, (BM, BK // 4, 4)))
    k0, k1, k2, k3 = _top2_keep(a0, a1, a2, a3)
    _store_rows_sparse(x, k0, k1, k2, k3, v_ptr, m_ptr, e_ptr, pm, pk, K, BM, BK)


@triton.jit
def _sp24t_kernel(
    x_ptr, pat_ptr, rv_ptr, rm_ptr, re_ptr, cv_ptr, cm_ptr, ce_ptr, M, K, BM: tl.constexpr, BK: tl.constexpr
):
    """Transposable 2:4 (optimal of the 90 4x4 patterns), emitting both the 2:4-along-K and 2:4-along-M operands."""
    pm, pk = tl.program_id(0), tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM).to(tl.int64)
    rk = pk * BK + tl.arange(0, BK)
    xb = tl.load(x_ptr + rm[:, None] * K + rk[None, :])
    x = xb.to(tl.float32)
    NT: tl.constexpr = (BM // 4) * (BK // 4)
    # [BM, BK] -> [BM/4, 4, BK/4, 4] -> [BM/4, BK/4, 4, 4] -> [NT, 16]
    tiles = tl.reshape(tl.permute(tl.reshape(tl.abs(xb), (BM // 4, 4, BK // 4, 4)), (0, 2, 1, 3)), (NT, 16))
    pr = tl.arange(0, 16)
    pc = tl.arange(0, 128)
    pats = tl.load(pat_ptr + pr[:, None] * 128 + pc[None, :])  # [16, 128] bf16, cols >= 90 zero
    scores = tl.dot(tiles, pats)  # [NT, 128] fp32
    scores = tl.where(pc[None, :] < 90, scores, -1.0)
    best = tl.argmax(scores, axis=1)
    onehot = (pc[None, :] == best[:, None]).to(tl.bfloat16)
    mask16 = tl.dot(onehot, tl.trans(pats))  # [NT, 16] 0/1
    keep = tl.reshape(tl.permute(tl.reshape(mask16, (BM // 4, BK // 4, 4, 4)), (0, 2, 1, 3)), (BM, BK)) > 0.5
    # rows (2:4 along K)
    k0, k1, k2, k3 = _split4(tl.reshape(keep, (BM, BK // 4, 4)))
    _store_rows_sparse(x, k0, k1, k2, k3, rv_ptr, rm_ptr, re_ptr, pm, pk, K, BM, BK)
    # cols (2:4 along M): [BM, BK] -> [BM/4, BK, 4]
    xt = tl.permute(tl.reshape(x, (BM // 4, 4, BK)), (0, 2, 1))
    kt = tl.permute(tl.reshape(keep, (BM // 4, 4, BK)), (0, 2, 1))
    c0, c1, c2, c3 = _split4(xt)
    j0, j1, j2, j3 = _split4(kt)
    f, s, nib = _compress4(c0, c1, c2, c3, j0, j1, j2, j3)  # [BM/4, BK]
    comp = tl.reshape(tl.permute(tl.join(f, s), (0, 2, 1)), (BM // 64, 32, BK))  # pairs along M
    ce = _exponent(tl.max(tl.abs(comp), axis=1))  # [BM/64, BK]
    q = tl.reshape((comp / _scale_of(ce)[:, None, :]).to(tl.float8e4nv), (BM // 2, BK))
    rcm = pm * (BM // 2) + tl.arange(0, BM // 2).to(tl.int64)
    tl.store(cv_ptr + rcm[:, None] * K + rk[None, :], q)
    lo, hi = tl.split(tl.permute(tl.reshape(nib, (BM // 8, 2, BK)), (0, 2, 1)))
    rmm = pm * (BM // 8) + tl.arange(0, BM // 8).to(tl.int64)
    tl.store(cm_ptr + rmm[:, None] * K + rk[None, :], (lo | (hi << 4)).to(tl.uint8))
    rce = pm * (BM // 64) + tl.arange(0, BM // 64).to(tl.int64)
    tl.store(ce_ptr + rce[:, None] * K + rk[None, :], ce.to(tl.uint8))


def patterns_tensor(device) -> torch.Tensor:
    rows = [r for r in itertools.product((0, 1), repeat=4) if sum(r) == 2]
    pats = [m for m in itertools.product(rows, repeat=4) if all(sum(c) == 2 for c in zip(*m, strict=True))]
    assert len(pats) == 90
    p = torch.zeros(16, 128, dtype=torch.bfloat16)
    for j, m in enumerate(pats):
        p[:, j] = torch.tensor([v for row in m for v in row], dtype=torch.bfloat16)
    return p.to(device).contiguous()


def quant_rows(x, BM=16, BK=512):
    M, K = x.shape
    v = torch.empty(M, K, device=x.device, dtype=torch.float8_e4m3fn)
    e = torch.empty(M, K // 32, device=x.device, dtype=torch.uint8)
    _quant_rows_kernel[(M // BM, K // BK)](x, v, e, M, K, BM=BM, BK=BK, num_warps=4)
    return v, e


def quant_rows_cols(x, BK=128):
    M, K = x.shape
    rv = torch.empty(M, K, device=x.device, dtype=torch.float8_e4m3fn)
    re = torch.empty(M, K // 32, device=x.device, dtype=torch.uint8)
    cv = torch.empty(M, K, device=x.device, dtype=torch.float8_e4m3fn)
    ce = torch.empty(M // 32, K, device=x.device, dtype=torch.uint8)
    _quant_rows_cols_kernel[(M // 32, K // BK)](x, rv, re, cv, ce, M, K, BK=BK, num_warps=4)
    return (rv, re), (cv, ce)


def sp24_rows(x, BM=16, BK=512, num_warps=4):
    M, K = x.shape
    v = torch.empty(M, K // 2, device=x.device, dtype=torch.float8_e4m3fn)
    m = torch.empty(M, K // 8, device=x.device, dtype=torch.uint8)
    e = torch.empty(M, K // 64, device=x.device, dtype=torch.uint8)
    _sp24_rows_kernel[(M // BM, K // BK)](x, v, m, e, M, K, BM=BM, BK=BK, num_warps=num_warps)
    return v, m, e


def sp24t(x, pats, BM=64, BK=128, num_warps=4):
    M, K = x.shape
    rv = torch.empty(M, K // 2, device=x.device, dtype=torch.float8_e4m3fn)
    rmeta = torch.empty(M, K // 8, device=x.device, dtype=torch.uint8)
    re = torch.empty(M, K // 64, device=x.device, dtype=torch.uint8)
    cv = torch.empty(M // 2, K, device=x.device, dtype=torch.float8_e4m3fn)
    cmeta = torch.empty(M // 8, K, device=x.device, dtype=torch.uint8)
    ce = torch.empty(M // 64, K, device=x.device, dtype=torch.uint8)
    _sp24t_kernel[(M // BM, K // BK)](x, pats, rv, rmeta, re, cv, cmeta, ce, M, K, BM=BM, BK=BK, num_warps=num_warps)
    return (rv, rmeta, re), (cv, cmeta, ce)


# ---------------------------------------------------------------- references (torch) for correctness


def ref_exponent(amax):
    bits = (amax / E4M3_MAX).view(torch.int32)
    e = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    return torch.where(amax > 0, e.clamp(1, 254), torch.full_like(e, 127))


def ref_mask_rows(a):
    g = a.abs().view(a.shape[0], -1, 4)
    gi, gj = g.unsqueeze(-1), g.unsqueeze(-2)
    idx = torch.arange(4, device=a.device)
    beats = (gj > gi) | ((gj == gi) & (idx.view(1, 4) < idx.view(4, 1)))
    return (beats.sum(-1) < 2).view_as(a)


def ref_mask_t(a, pats16x128):
    m, f = a.shape
    pats = pats16x128[:, :90].t().float()  # [90, 16]
    tiles = a.abs().float().view(m // 4, 4, f // 4, 4).permute(0, 2, 1, 3).reshape(-1, 16)
    best = (tiles @ pats.t()).argmax(-1)
    return pats[best].view(m // 4, f // 4, 4, 4).permute(0, 2, 1, 3).reshape(m, f) > 0.5


def decompress_rows(v, meta, e, K):
    """2:4-along-K compressed -> dense fp32 [M, K]."""
    M = v.shape[0]
    vals = v.float().view(M, K // 64, 32) * torch.exp2(e.float() - 127)[:, :, None]
    vals = vals.view(M, K // 4, 2)
    nib = torch.stack([meta & 0xF, meta >> 4], dim=-1).view(M, K // 4).long()
    i0, i1 = nib & 3, nib >> 2
    out = torch.zeros(M, K // 4, 4, device=v.device)
    out.scatter_(2, i0[..., None], vals[..., :1])
    out.scatter_(2, i1[..., None], vals[..., 1:])
    return out.view(M, K)
