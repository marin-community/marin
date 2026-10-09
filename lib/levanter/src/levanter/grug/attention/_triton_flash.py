# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0
"""Grug's ``gpu_triton_flash`` attention: a Triton flash attention for causal self-attention.

Written for AMD Instinct GPUs (ROCm), where Grug has no other fused kernel. Forward, dQ and dK/dV are separate
Triton kernels in the FlashAttention-2 layout, launched through ``jax_triton``:

- Every mask is reduced to one inclusive key lower bound per query, ``lb[b, q]``. A query ``q`` attends to keys
  ``lb[q] <= k <= q``. A sliding window ``W`` gives ``lb[q] >= q - W + 1`` and packed segment IDs give
  ``lb[q] >= start of q's segment``. Key blocks entirely outside ``[lb, q]`` are never visited, so a window-2048
  layer at 4096 tokens does about 37.5% of the dense work and a causal layer about 50%.
- Blocks entirely inside the allowed band skip the mask computation.
- Grouped-query attention indexes the shared K/V head per query head instead of materializing repeated K/V. The
  dK/dV program owns one KV head and accumulates over its query-head group.
- bf16 inputs, f32 softmax statistics and accumulators, and the log-sum-exp kept in base 2.

Segment semantics: each maximal run of equal segment IDs is one document, the packed-data convention. A segment
ID that reappears after a different ID starts a new document here, whereas ``reference_attention`` would let the
two runs see each other. Query and key segment IDs must be equal (self-attention).

The kernels are plain Triton rather than Pallas because JAX deprecates the Pallas Triton backend, and Mosaic GPU,
its replacement, does not target AMD GPUs. On ROCm, ``jax_triton`` needs jaxlib 0.11.1 or newer: 0.11.0 never
loads ``jax_rocm10_plugin``'s Triton launcher.
"""

import dataclasses
import functools
import math

import equinox as eqx
import jax
import jax_triton as jt
import triton
import triton.language as tl
from jax import lax
from jax import numpy as jnp
from jax import shard_map
from jax.sharding import PartitionSpec as P
from jax.sharding import get_abstract_mesh, reshard
from jaxtyping import Array, Float, Int

from levanter.grug.attention._core import AttentionMask
from levanter.sharding import partition_spec_of, partitioned_dims

# Finite stand-in for -inf: a fully masked tile keeps the running max finite, and the next unmasked tile's
# rescale factor exp2(m_old - m_new) then zeroes whatever the masked tile accumulated.
_MASK_VALUE = tl.constexpr(-0.7 * float(jnp.finfo(jnp.float32).max))
_LOG2_E = math.log2(math.e)
# Full-precision dots for float32 inputs, which Triton otherwise runs as xf32 on MI300X; bf16 dots are unaffected.
_DOT_PRECISION = tl.constexpr("ieee")


@dataclasses.dataclass(frozen=True)
class TritonFlashBlockSizes:
    """Tile sizes and Triton launch parameters for the three kernels.

    ``block_q``/``block_k`` tile the forward pass, ``block_q_dq``/``block_k_dq`` the dQ pass, and
    ``block_q_dkv``/``block_k_dkv`` the dK/dV pass (each dK/dV program owns ``block_k_dkv`` keys and walks
    queries ``block_q_dkv`` at a time). All sizes must be powers of two that divide the sequence length.
    """

    block_q: int = 256
    block_k: int = 64
    num_warps: int = 8
    num_stages: int = 1
    block_q_dq: int = 128
    block_k_dq: int = 32
    num_warps_dq: int = 4
    num_stages_dq: int = 1
    block_q_dkv: int = 32
    block_k_dkv: int = 128
    num_warps_dkv: int = 4
    num_stages_dkv: int = 2


# Tile sweeps at the June per-GPU shape (8 x 4096 tokens, 20 query heads, 5 KV heads, head_dim 128, bf16) set the
# defaults, which are the fastest found on MI300X (gfx942). On MI350X (gfx950), a second pipeline stage runs the
# forward 14% (window 2048) to 19% (causal) faster; on MI300X it nearly doubles the forward's time.
_TUNED_BLOCK_SIZES = {"gfx950": TritonFlashBlockSizes(num_stages=2)}
# Untuned tiles small enough for float32 inputs in MI300X's 64 KiB of shared memory, which the bf16 tiles overflow.
_FLOAT32_BLOCK_SIZES = TritonFlashBlockSizes(block_q=128, block_k=32, block_k_dkv=64)


def _is_power_of_two(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


def _validate(q: jax.Array, k: jax.Array, v: jax.Array, block_sizes: TritonFlashBlockSizes) -> None:
    batch, seq_len, num_q_heads, head_dim = q.shape
    if k.shape != v.shape or k.shape[0] != batch or k.shape[1] != seq_len or k.shape[3] != head_dim:
        raise ValueError(f"gpu_triton_flash needs self-attention shapes, got q={q.shape} k={k.shape} v={v.shape}")
    num_kv_heads = k.shape[2]
    if num_q_heads % num_kv_heads != 0 or not _is_power_of_two(num_q_heads // num_kv_heads):
        raise ValueError(f"gpu_triton_flash needs a power-of-two GQA ratio, got Hq={num_q_heads} Hkv={num_kv_heads}")
    if not _is_power_of_two(head_dim) or head_dim < 16:
        raise ValueError(f"gpu_triton_flash needs a power-of-two head_dim >= 16, got {head_dim}")
    for name, size in dataclasses.asdict(block_sizes).items():
        if name.startswith("block_") and (not _is_power_of_two(size) or seq_len % size != 0):
            raise ValueError(f"{name}={size} must be a power of two dividing the sequence length {seq_len}")


def _key_lower_bounds(mask: AttentionMask, *, batch: int, seq_len: int) -> Int[Array, "B S"]:
    """Inclusive lowest key position each query may attend to, for a causal ``AttentionMask``."""
    if not mask.is_causal:
        raise NotImplementedError("gpu_triton_flash supports causal self-attention only")
    if mask.fa4_bounds is not None:
        raise NotImplementedError("gpu_triton_flash does not read FA4/CuTe precomputed bounds")
    positions = jnp.arange(seq_len, dtype=jnp.int32)[None, :]
    lower = jnp.zeros((batch, seq_len), dtype=jnp.int32)
    if mask.segment_ids is not None:
        q_seg, kv_seg = mask.segment_ids
        same = q_seg is kv_seg
        q_seg = jnp.broadcast_to(q_seg, (batch, seq_len))
        if not same:
            kv_seg = jnp.broadcast_to(kv_seg, (batch, seq_len))
            q_seg = eqx.error_if(q_seg, jnp.any(q_seg != kv_seg), "gpu_triton_flash needs equal q/kv segment ids")
        starts = jnp.concatenate([jnp.ones((batch, 1), dtype=jnp.bool_), q_seg[:, 1:] != q_seg[:, :-1]], axis=1)
        lower = lax.cummax(jnp.where(starts, positions, 0), axis=1)
    if mask.sliding_window is not None:
        if mask.sliding_window <= 0:
            raise ValueError(f"sliding_window must be positive, got {mask.sliding_window}")
        lower = jnp.maximum(lower, positions - (mask.sliding_window - 1))
    return lower


def _dkv_query_ranges(
    lower: Int[Array, "B S"], *, block_k: int, block_q: int
) -> tuple[Int[Array, "B Nk"], Int[Array, "B Nk"]]:
    """Per key block, the ends of the query ranges its lower bounds allow; the dK/dV kernel applies causality.

    ``reach`` counts the queries whose lower bound admits some key of the block, and ``full`` counts the query tiles
    whose lower bounds admit all of its keys. ``lower`` is nondecreasing along the sequence, so both are prefixes.
    """
    seq_len = lower.shape[1]
    starts = jnp.arange(0, seq_len, block_k, dtype=jnp.int32)
    reach = jnp.sum(lower[:, None, :] <= (starts + block_k - 1)[None, :, None], axis=-1, dtype=jnp.int32)
    full = jnp.sum(lower[:, None, :] <= starts[None, :, None], axis=-1, dtype=jnp.int32) // block_q
    return reach, full


# Each kernel walks a band of blocks ``[lo, hi)`` in three loops: masked on ``[lo, full_lo)``, unmasked on
# ``[full_lo, full_hi)`` and masked on ``[full_hi, hi)``. The bounds are clamped at zero, which they already satisfy,
# so Triton can prove the load offsets non-negative; on MI350X that made the backward 1-2% faster.
# Row tensors are ``[rows, head_dim]`` slices of a ``[B, S, H, D]`` array, ``row_stride = H * D`` apart.


@triton.jit
def _band_mask(q_pos, q_lower, k_pos):
    return (k_pos[None, :] <= q_pos[:, None]) & (k_pos[None, :] >= q_lower[:, None])


@triton.jit
def _clip(x, lo, hi):
    return tl.minimum(tl.maximum(x, lo), hi)


@triton.jit
def _forward_key_blocks(
    acc,
    m_i,
    l_i,
    q,
    q_pos,
    q_lower,
    k_ptrs,
    v_ptrs,
    start,
    stop,
    qk_scale: tl.constexpr,
    block_k: tl.constexpr,
    kv_row_stride: tl.constexpr,
    masked: tl.constexpr,
):
    for kb in range(start, stop):
        offset = kb * (block_k * kv_row_stride)
        k = tl.load(k_ptrs + offset)
        s = tl.dot(q, tl.trans(k), input_precision=_DOT_PRECISION) * qk_scale
        if masked:
            k_pos = kb * block_k + tl.arange(0, block_k)
            s = tl.where(_band_mask(q_pos, q_lower, k_pos), s, _MASK_VALUE)
        m_next = tl.maximum(m_i, tl.max(s, axis=1))
        correction = tl.exp2(m_i - m_next)
        p = tl.exp2(s - m_next[:, None])
        l_i = l_i * correction + tl.sum(p, axis=1)
        v = tl.load(v_ptrs + offset)
        acc = acc * correction[:, None] + tl.dot(p.to(v.dtype), v, input_precision=_DOT_PRECISION)
        m_i = m_next
    return acc, m_i, l_i


@triton.jit
def _forward_kernel(
    q_ptr,  # [B, S, Hq, D]
    k_ptr,  # [B, S, Hkv, D]
    v_ptr,  # [B, S, Hkv, D]
    lower_ptr,  # [B, S] int32
    o_ptr,  # [B, S, Hq, D]
    lse_ptr,  # [B, Hq, S] float32, base 2
    seq_len: tl.constexpr,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    qk_scale: tl.constexpr,
    block_q: tl.constexpr,
    block_k: tl.constexpr,
):
    q_block = tl.num_programs(0) - 1 - tl.program_id(0)  # longest rows first
    batch = tl.program_id(1).to(tl.int64)
    head = tl.program_id(2)
    kv_head = head // (num_q_heads // num_kv_heads)
    q_row_stride: tl.constexpr = num_q_heads * head_dim
    kv_row_stride: tl.constexpr = num_kv_heads * head_dim
    q_ptr += batch * seq_len * q_row_stride + head * head_dim
    o_ptr += batch * seq_len * q_row_stride + head * head_dim
    k_ptr += batch * seq_len * kv_row_stride + kv_head * head_dim
    v_ptr += batch * seq_len * kv_row_stride + kv_head * head_dim
    lower_ptr += batch * seq_len
    lse_ptr += (batch * num_q_heads + head) * seq_len

    q_start = q_block * block_q
    q_pos = q_start + tl.arange(0, block_q)
    k_rows = tl.arange(0, block_k)
    cols = tl.arange(0, head_dim)
    q_offsets = q_pos[:, None] * q_row_stride + cols[None, :]
    k_ptrs = k_ptr + k_rows[:, None] * kv_row_stride + cols[None, :]
    v_ptrs = v_ptr + k_rows[:, None] * kv_row_stride + cols[None, :]

    q = tl.load(q_ptr + q_offsets)
    q_lower = tl.load(lower_ptr + q_pos)
    lo = tl.maximum(tl.min(q_lower, axis=0), 0) // block_k
    hi = (q_start + block_q + block_k - 1) // block_k
    full_lo = _clip((tl.max(q_lower, axis=0) + block_k - 1) // block_k, lo, hi)
    full_hi = _clip((q_start + 1) // block_k, full_lo, hi)

    acc = tl.zeros((block_q, head_dim), tl.float32)
    m_i = tl.full((block_q,), _MASK_VALUE, tl.float32)
    l_i = tl.zeros((block_q,), tl.float32)
    acc, m_i, l_i = _forward_key_blocks(
        acc, m_i, l_i, q, q_pos, q_lower, k_ptrs, v_ptrs, lo, full_lo, qk_scale, block_k, kv_row_stride, True
    )
    acc, m_i, l_i = _forward_key_blocks(
        acc, m_i, l_i, q, q_pos, q_lower, k_ptrs, v_ptrs, full_lo, full_hi, qk_scale, block_k, kv_row_stride, False
    )
    acc, m_i, l_i = _forward_key_blocks(
        acc, m_i, l_i, q, q_pos, q_lower, k_ptrs, v_ptrs, full_hi, hi, qk_scale, block_k, kv_row_stride, True
    )
    tl.store(o_ptr + q_offsets, (acc / l_i[:, None]).to(o_ptr.dtype.element_ty))
    tl.store(lse_ptr + q_pos, m_i + tl.log2(l_i))


@triton.jit
def _dq_key_blocks(
    dq,
    q,
    do,
    lse,
    delta,
    q_pos,
    q_lower,
    k_ptrs,
    v_ptrs,
    start,
    stop,
    qk_scale: tl.constexpr,
    block_k: tl.constexpr,
    kv_row_stride: tl.constexpr,
    masked: tl.constexpr,
):
    for kb in range(start, stop):
        offset = kb * (block_k * kv_row_stride)
        k = tl.load(k_ptrs + offset)
        v = tl.load(v_ptrs + offset)
        s = tl.dot(q, tl.trans(k), input_precision=_DOT_PRECISION) * qk_scale
        if masked:
            k_pos = kb * block_k + tl.arange(0, block_k)
            s = tl.where(_band_mask(q_pos, q_lower, k_pos), s, _MASK_VALUE)
        p = tl.exp2(s - lse[:, None])
        dp = tl.dot(do, tl.trans(v), input_precision=_DOT_PRECISION)
        ds = p * (dp - delta[:, None])
        dq += tl.dot(ds.to(k.dtype), k, input_precision=_DOT_PRECISION)
    return dq


@triton.jit
def _dq_kernel(
    q_ptr,  # [B, S, Hq, D]
    k_ptr,  # [B, S, Hkv, D]
    v_ptr,  # [B, S, Hkv, D]
    lower_ptr,  # [B, S] int32
    do_ptr,  # [B, S, Hq, D]
    lse_ptr,  # [B, Hq, S] float32
    delta_ptr,  # [B, Hq, S] float32
    dq_ptr,  # [B, S, Hq, D]
    seq_len: tl.constexpr,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    scale: tl.constexpr,
    qk_scale: tl.constexpr,
    block_q: tl.constexpr,
    block_k: tl.constexpr,
):
    q_block = tl.num_programs(0) - 1 - tl.program_id(0)
    batch = tl.program_id(1).to(tl.int64)
    head = tl.program_id(2)
    kv_head = head // (num_q_heads // num_kv_heads)
    q_row_stride: tl.constexpr = num_q_heads * head_dim
    kv_row_stride: tl.constexpr = num_kv_heads * head_dim
    q_ptr += batch * seq_len * q_row_stride + head * head_dim
    do_ptr += batch * seq_len * q_row_stride + head * head_dim
    dq_ptr += batch * seq_len * q_row_stride + head * head_dim
    k_ptr += batch * seq_len * kv_row_stride + kv_head * head_dim
    v_ptr += batch * seq_len * kv_row_stride + kv_head * head_dim
    lower_ptr += batch * seq_len
    lse_ptr += (batch * num_q_heads + head) * seq_len
    delta_ptr += (batch * num_q_heads + head) * seq_len

    q_start = q_block * block_q
    q_pos = q_start + tl.arange(0, block_q)
    k_rows = tl.arange(0, block_k)
    cols = tl.arange(0, head_dim)
    q_offsets = q_pos[:, None] * q_row_stride + cols[None, :]
    k_ptrs = k_ptr + k_rows[:, None] * kv_row_stride + cols[None, :]
    v_ptrs = v_ptr + k_rows[:, None] * kv_row_stride + cols[None, :]

    q = tl.load(q_ptr + q_offsets)
    do = tl.load(do_ptr + q_offsets)
    lse = tl.load(lse_ptr + q_pos)
    delta = tl.load(delta_ptr + q_pos)
    q_lower = tl.load(lower_ptr + q_pos)
    lo = tl.maximum(tl.min(q_lower, axis=0), 0) // block_k
    hi = (q_start + block_q + block_k - 1) // block_k
    full_lo = _clip((tl.max(q_lower, axis=0) + block_k - 1) // block_k, lo, hi)
    full_hi = _clip((q_start + 1) // block_k, full_lo, hi)

    dq = tl.zeros((block_q, head_dim), tl.float32)
    dq = _dq_key_blocks(
        dq, q, do, lse, delta, q_pos, q_lower, k_ptrs, v_ptrs, lo, full_lo, qk_scale, block_k, kv_row_stride, True
    )
    dq = _dq_key_blocks(
        dq,
        q,
        do,
        lse,
        delta,
        q_pos,
        q_lower,
        k_ptrs,
        v_ptrs,
        full_lo,
        full_hi,
        qk_scale,
        block_k,
        kv_row_stride,
        False,
    )
    dq = _dq_key_blocks(
        dq, q, do, lse, delta, q_pos, q_lower, k_ptrs, v_ptrs, full_hi, hi, qk_scale, block_k, kv_row_stride, True
    )
    tl.store(dq_ptr + q_offsets, (dq * scale).to(dq_ptr.dtype.element_ty))


# The dK/dV loop works on transposed tiles, as Triton's fused-attention tutorial does: ``S^T = K Q^T`` leaves
# ``P^T`` and ``dS^T`` in the layout the dV and dK matmuls take, where ``S = Q K^T`` would transpose both in registers.
@triton.jit
def _dkv_query_blocks(
    dk,
    dv,
    k,
    v,
    k_pos,
    qT_ptrs,
    do_ptrs,
    lse_ptrs,
    delta_ptrs,
    lower_ptrs,
    start,
    stop,
    qk_scale: tl.constexpr,
    block_q: tl.constexpr,
    q_row_stride: tl.constexpr,
    masked: tl.constexpr,
):
    for qb in range(start, stop):
        row_offset = qb * (block_q * q_row_stride)
        qT = tl.load(qT_ptrs + row_offset)
        lse = tl.load(lse_ptrs + qb * block_q)
        sT = tl.dot(k, qT, input_precision=_DOT_PRECISION) * qk_scale
        if masked:
            q_pos = qb * block_q + tl.arange(0, block_q)
            q_lower = tl.load(lower_ptrs + qb * block_q)
            allowed = (k_pos[:, None] <= q_pos[None, :]) & (k_pos[:, None] >= q_lower[None, :])
            sT = tl.where(allowed, sT, _MASK_VALUE)
        pT = tl.exp2(sT - lse[None, :])
        do = tl.load(do_ptrs + row_offset)
        dv += tl.dot(pT.to(do.dtype), do, input_precision=_DOT_PRECISION)
        delta = tl.load(delta_ptrs + qb * block_q)
        dpT = tl.dot(v, tl.trans(do), input_precision=_DOT_PRECISION)
        dsT = pT * (dpT - delta[None, :])
        dk += tl.dot(dsT.to(qT.dtype), tl.trans(qT), input_precision=_DOT_PRECISION)
    return dk, dv


@triton.jit
def _dkv_kernel(
    q_ptr,  # [B, S, Hq, D]
    k_ptr,  # [B, S, Hkv, D]
    v_ptr,  # [B, S, Hkv, D]
    lower_ptr,  # [B, S] int32
    do_ptr,  # [B, S, Hq, D]
    lse_ptr,  # [B, Hq, S] float32
    delta_ptr,  # [B, Hq, S] float32
    reach_ptr,  # [B, S // block_k] int32
    full_ptr,  # [B, S // block_k] int32
    dk_ptr,  # [B, S, Hkv, D]
    dv_ptr,  # [B, S, Hkv, D]
    seq_len: tl.constexpr,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    scale: tl.constexpr,
    qk_scale: tl.constexpr,
    block_q: tl.constexpr,
    block_k: tl.constexpr,
):
    k_block = tl.program_id(0)
    batch = tl.program_id(1).to(tl.int64)
    kv_head = tl.program_id(2)
    group: tl.constexpr = num_q_heads // num_kv_heads
    q_row_stride: tl.constexpr = num_q_heads * head_dim
    kv_row_stride: tl.constexpr = num_kv_heads * head_dim
    k_ptr += batch * seq_len * kv_row_stride + kv_head * head_dim
    v_ptr += batch * seq_len * kv_row_stride + kv_head * head_dim
    dk_ptr += batch * seq_len * kv_row_stride + kv_head * head_dim
    dv_ptr += batch * seq_len * kv_row_stride + kv_head * head_dim
    lower_ptr += batch * seq_len
    reach_ptr += batch * (seq_len // block_k)
    full_ptr += batch * (seq_len // block_k)

    k_start = k_block * block_k
    k_pos = k_start + tl.arange(0, block_k)
    q_rows = tl.arange(0, block_q)
    cols = tl.arange(0, head_dim)
    kv_offsets = k_pos[:, None] * kv_row_stride + cols[None, :]
    k = tl.load(k_ptr + kv_offsets)
    v = tl.load(v_ptr + kv_offsets)

    lo = k_start // block_q
    hi = (tl.maximum(tl.load(reach_ptr + k_block), 0) + block_q - 1) // block_q
    full_lo = _clip((k_start + block_k - 1 + block_q - 1) // block_q, lo, hi)
    full_hi = _clip(tl.maximum(tl.load(full_ptr + k_block), 0), full_lo, hi)
    lower_ptrs = lower_ptr + q_rows
    row_offsets = q_rows[:, None] * q_row_stride + cols[None, :]
    rowT_offsets = q_rows[None, :] * q_row_stride + cols[:, None]

    dk = tl.zeros((block_k, head_dim), tl.float32)
    dv = tl.zeros((block_k, head_dim), tl.float32)
    for g in tl.static_range(group):
        head = kv_head * group + g
        rows_base = batch * seq_len * q_row_stride + head * head_dim
        qT_ptrs = q_ptr + rows_base + rowT_offsets
        do_ptrs = do_ptr + rows_base + row_offsets
        stats_base = (batch * num_q_heads + head) * seq_len
        lse_ptrs = lse_ptr + stats_base + q_rows
        delta_ptrs = delta_ptr + stats_base + q_rows
        dk, dv = _dkv_query_blocks(
            dk,
            dv,
            k,
            v,
            k_pos,
            qT_ptrs,
            do_ptrs,
            lse_ptrs,
            delta_ptrs,
            lower_ptrs,
            lo,
            full_lo,
            qk_scale,
            block_q,
            q_row_stride,
            True,
        )
        dk, dv = _dkv_query_blocks(
            dk,
            dv,
            k,
            v,
            k_pos,
            qT_ptrs,
            do_ptrs,
            lse_ptrs,
            delta_ptrs,
            lower_ptrs,
            full_lo,
            full_hi,
            qk_scale,
            block_q,
            q_row_stride,
            False,
        )
        dk, dv = _dkv_query_blocks(
            dk,
            dv,
            k,
            v,
            k_pos,
            qT_ptrs,
            do_ptrs,
            lse_ptrs,
            delta_ptrs,
            lower_ptrs,
            full_hi,
            hi,
            qk_scale,
            block_q,
            q_row_stride,
            True,
        )
    tl.store(dk_ptr + kv_offsets, (dk * scale).to(dk_ptr.dtype.element_ty))
    tl.store(dv_ptr + kv_offsets, dv.to(dv_ptr.dtype.element_ty))


def _shape_params(q: jax.Array, k: jax.Array) -> dict:
    """The constexpr shape and scale parameters every kernel takes."""
    _, seq_len, num_q_heads, head_dim = q.shape
    scale = 1.0 / math.sqrt(head_dim)
    return dict(
        seq_len=seq_len,
        num_q_heads=num_q_heads,
        num_kv_heads=k.shape[2],
        head_dim=head_dim,
        qk_scale=scale * _LOG2_E,
    )


def _forward(q, k, v, lower, *, block_sizes: TritonFlashBlockSizes):
    batch, seq_len, num_q_heads, _ = q.shape
    return jt.triton_call(
        q,
        k,
        v,
        lower,
        kernel=_forward_kernel,
        out_shape=(
            jax.ShapeDtypeStruct(q.shape, q.dtype),
            jax.ShapeDtypeStruct((batch, num_q_heads, seq_len), jnp.float32),
        ),
        grid=(seq_len // block_sizes.block_q, batch, num_q_heads),
        num_warps=block_sizes.num_warps,
        num_stages=block_sizes.num_stages,
        name="grug_triton_flash_fwd",
        block_q=block_sizes.block_q,
        block_k=block_sizes.block_k,
        **_shape_params(q, k),
    )


def _backward(q, k, v, lower, out, lse, do, *, block_sizes: TritonFlashBlockSizes):
    batch, seq_len, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[2]
    params = dict(scale=1.0 / math.sqrt(head_dim), **_shape_params(q, k))
    delta = jnp.einsum("bshd,bshd->bhs", out.astype(jnp.float32), do.astype(jnp.float32))

    dq = jt.triton_call(
        q,
        k,
        v,
        lower,
        do,
        lse,
        delta,
        kernel=_dq_kernel,
        out_shape=jax.ShapeDtypeStruct(q.shape, q.dtype),
        grid=(seq_len // block_sizes.block_q_dq, batch, num_q_heads),
        num_warps=block_sizes.num_warps_dq,
        num_stages=block_sizes.num_stages_dq,
        name="grug_triton_flash_dq",
        block_q=block_sizes.block_q_dq,
        block_k=block_sizes.block_k_dq,
        **params,
    )

    block_k = block_sizes.block_k_dkv
    block_q = block_sizes.block_q_dkv
    reach, full = _dkv_query_ranges(lower, block_k=block_k, block_q=block_q)
    dk, dv = jt.triton_call(
        q,
        k,
        v,
        lower,
        do,
        lse,
        delta,
        reach,
        full,
        kernel=_dkv_kernel,
        out_shape=(jax.ShapeDtypeStruct(k.shape, k.dtype), jax.ShapeDtypeStruct(v.shape, v.dtype)),
        grid=(seq_len // block_k, batch, num_kv_heads),
        num_warps=block_sizes.num_warps_dkv,
        num_stages=block_sizes.num_stages_dkv,
        name="grug_triton_flash_dkv",
        block_q=block_q,
        block_k=block_k,
        **params,
    )
    return dq, dk, dv


@functools.partial(jax.custom_vjp, nondiff_argnums=(4,))
def _flash(q, k, v, lower, block_sizes: TritonFlashBlockSizes):
    out, _ = _forward(q, k, v, lower, block_sizes=block_sizes)
    return out


def _flash_fwd(q, k, v, lower, block_sizes):
    out, lse = _forward(q, k, v, lower, block_sizes=block_sizes)
    return out, (q, k, v, lower, out, lse)


def _flash_bwd(block_sizes, residuals, do):
    q, k, v, lower, out, lse = residuals
    dq, dk, dv = _backward(q, k, v, lower, out, lse, do, block_sizes=block_sizes)
    return dq, dk, dv, None


_flash.defvjp(_flash_fwd, _flash_bwd)


def triton_flash_attention(
    q: Float[Array, "B S Hq D"],
    k: Float[Array, "B S Hkv D"],
    v: Float[Array, "B S Hkv D"],
    mask: AttentionMask,
    *,
    block_sizes: TritonFlashBlockSizes | None = None,
) -> Float[Array, "B S Hq D"]:
    """Causal (optionally windowed and segmented) self-attention through the Triton flash kernels.

    The softmax scale is ``1 / sqrt(head_dim)``, as in ``reference_attention``. ``block_sizes`` defaults to the
    sizes tuned for this GPU, or to smaller tiles for float32 inputs. Under a mesh, the kernels run per shard inside
    ``shard_map``; batch and heads may be sharded, sequence and head_dim may not.
    """
    if jax.default_backend() != "gpu":
        raise RuntimeError("gpu_triton_flash requires the JAX GPU backend")
    if block_sizes is None and q.dtype == jnp.float32:
        block_sizes = _FLOAT32_BLOCK_SIZES
    if block_sizes is None:
        block_sizes = _TUNED_BLOCK_SIZES.get(jax.devices()[0].compute_capability, TritonFlashBlockSizes())
    _validate(q, k, v, block_sizes)
    batch, seq_len = q.shape[:2]
    lower = _key_lower_bounds(mask, batch=batch, seq_len=seq_len)
    run = functools.partial(_flash, block_sizes=block_sizes)

    mesh = get_abstract_mesh()
    if mesh is None or mesh.empty:
        return run(q, k, v, lower)
    q_dims = partitioned_dims(q, mesh)
    if q_dims[1] or q_dims[3]:
        raise ValueError(f"gpu_triton_flash needs unsharded sequence and head_dim, got {partition_spec_of(q)}")
    for name, x in (("k", k), ("v", v)):
        if partitioned_dims(x, mesh) != q_dims:
            raise ValueError(
                f"gpu_triton_flash needs {name} sharded like q, got q={partition_spec_of(q)} "
                f"{name}={partition_spec_of(x)}"
            )
    if not any(q_dims):
        return run(q, k, v, lower)
    q_spec = tuple(partition_spec_of(q))
    q_spec = P(*q_spec, *((None,) * (q.ndim - len(q_spec))))
    lower = reshard(lower, P(q_spec[0], None))

    @shard_map(mesh=mesh, out_specs=q_spec, check_vma=False)
    def _local(q_local, k_local, v_local, lower_local):
        return run(q_local, k_local, v_local, lower_local)

    return _local(q, k, v, lower)
