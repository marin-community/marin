# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0
"""Grug's ``gpu_pallas_triton_flash`` attention: a Pallas-Triton flash attention for causal self-attention.

Written for AMD Instinct GPUs (ROCm), where Pallas lowers through Triton and Grug has no other fused kernel.
Forward, dQ and dK/dV are separate ``pallas_call``s in the FlashAttention-2 layout:

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

JAX has deprecated the Pallas Triton backend in favor of Mosaic GPU, which does not target AMD GPUs; this kernel
is a stopgap until an AMD fused attention (AITER/CK) is reachable from JAX.
"""

import dataclasses
import functools
import math
from collections.abc import Callable

import equinox as eqx
import jax
from jax import lax
from jax import numpy as jnp
from jax import shard_map
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu
from jax.sharding import PartitionSpec as P
from jax.sharding import get_abstract_mesh, reshard
from jaxtyping import Array, Float, Int

from levanter.grug.attention._core import AttentionMask
from levanter.sharding import partition_spec_of, partitioned_dims

# Finite stand-in for -inf: a fully masked tile keeps the running max finite, and the next unmasked tile's
# rescale factor exp2(m_old - m_new) then zeroes whatever the masked tile accumulated.
_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.float32).max)
_LOG2_E = math.log2(math.e)


@dataclasses.dataclass(frozen=True)
class TritonFlashBlockSizes:
    """Tile sizes and Triton launch parameters for the three kernels.

    ``block_q``/``block_k`` tile the forward pass, ``block_q_dq``/``block_k_dq`` the dQ pass, and
    ``block_q_dkv``/``block_k_dkv`` the dK/dV pass (each dK/dV program owns ``block_k_dkv`` keys and walks
    queries ``block_q_dkv`` at a time). All sizes must be powers of two that divide the sequence length.
    """

    block_q: int = 128
    block_k: int = 32
    num_warps: int = 4
    num_stages: int = 2
    block_q_dq: int = 128
    block_k_dq: int = 32
    num_warps_dq: int = 4
    num_stages_dq: int = 1
    block_q_dkv: int = 128
    block_k_dkv: int = 64
    num_warps_dkv: int = 4
    num_stages_dkv: int = 1


def _is_power_of_two(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


def _validate(q: jax.Array, k: jax.Array, v: jax.Array, block_sizes: TritonFlashBlockSizes) -> None:
    batch, seq_len, num_q_heads, head_dim = q.shape
    if k.shape != v.shape or k.shape[0] != batch or k.shape[1] != seq_len or k.shape[3] != head_dim:
        raise ValueError(
            f"gpu_pallas_triton_flash needs self-attention shapes, got q={q.shape} k={k.shape} v={v.shape}"
        )
    num_kv_heads = k.shape[2]
    if num_q_heads % num_kv_heads != 0 or not _is_power_of_two(num_q_heads // num_kv_heads):
        raise ValueError(
            f"gpu_pallas_triton_flash needs a power-of-two GQA ratio, got Hq={num_q_heads} Hkv={num_kv_heads}"
        )
    if not _is_power_of_two(head_dim) or head_dim < 16:
        raise ValueError(f"gpu_pallas_triton_flash needs a power-of-two head_dim >= 16, got {head_dim}")
    for name, size in dataclasses.asdict(block_sizes).items():
        if name.startswith("block_") and (not _is_power_of_two(size) or seq_len % size != 0):
            raise ValueError(f"{name}={size} must be a power of two dividing the sequence length {seq_len}")


def _key_lower_bounds(mask: AttentionMask, *, batch: int, seq_len: int) -> Int[Array, "B S"]:
    """Inclusive lowest key position each query may attend to, for a causal ``AttentionMask``."""
    if not mask.is_causal:
        raise NotImplementedError("gpu_pallas_triton_flash supports causal self-attention only")
    if mask.fa4_bounds is not None:
        raise NotImplementedError("gpu_pallas_triton_flash does not read FA4/CuTe precomputed bounds")
    positions = jnp.arange(seq_len, dtype=jnp.int32)[None, :]
    lower = jnp.zeros((batch, seq_len), dtype=jnp.int32)
    if mask.segment_ids is not None:
        q_seg, kv_seg = mask.segment_ids
        same = q_seg is kv_seg
        q_seg = jnp.broadcast_to(q_seg, (batch, seq_len))
        if not same:
            kv_seg = jnp.broadcast_to(kv_seg, (batch, seq_len))
            q_seg = eqx.error_if(
                q_seg, jnp.any(q_seg != kv_seg), "gpu_pallas_triton_flash needs equal q/kv segment ids"
            )
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
    """Per key block: the query count that may see it, and the count of query tiles that see all of it.

    ``lower`` is nondecreasing along the sequence, so "queries with ``lower <= x``" is a prefix.
    """
    seq_len = lower.shape[1]
    starts = jnp.arange(0, seq_len, block_k, dtype=jnp.int32)
    reach = jnp.sum(lower[:, None, :] <= (starts + block_k - 1)[None, :, None], axis=-1, dtype=jnp.int32)
    full = jnp.sum(lower[:, None, :] <= starts[None, :, None], axis=-1, dtype=jnp.int32) // block_q
    return reach, full


def _banded_loop(lo, full_lo, full_hi, hi, body: Callable, carry):
    """Run ``body(i, carry, masked)`` over ``[lo, hi)``, unmasked on ``[full_lo, full_hi)`` and masked elsewhere."""
    a = jnp.clip(full_lo, lo, hi)
    b = jnp.clip(full_hi, a, hi)
    carry = lax.fori_loop(lo, a, functools.partial(body, masked=True), carry)
    carry = lax.fori_loop(a, b, functools.partial(body, masked=False), carry)
    return lax.fori_loop(b, hi, functools.partial(body, masked=True), carry)


def _key_block_band(q_lower, q_start: int, block_q: int, block_k: int):
    """Key-block bounds ``(lo, full_lo, full_hi, hi)`` for one query tile, as ``_banded_loop`` takes them.

    Blocks ``[lo, hi)`` hold at least one allowed key for some query in the tile; blocks ``[full_lo, full_hi)``
    are allowed for every query in the tile and need no mask.
    """
    lo = jnp.min(q_lower) // block_k
    full_lo = (jnp.max(q_lower) + block_k - 1) // block_k
    full_hi = (q_start + 1) // block_k
    hi = (q_start + block_q + block_k - 1) // block_k
    return lo, full_lo, full_hi, hi


def _band_mask(q_pos, q_lower, k_pos):
    return (k_pos[None, :] <= q_pos[:, None]) & (k_pos[None, :] >= q_lower[:, None])


def _forward_kernel(q_ref, k_ref, v_ref, lower_ref, o_ref, lse_ref, *, scale: float, block_k: int, num_q_blocks: int):
    block_q = q_ref.shape[0]
    q_block = num_q_blocks - 1 - pl.program_id(0)  # longest rows first
    q_start = q_block * block_q
    q = q_ref[...]
    q_lower = lower_ref[...]
    q_pos = q_start + jnp.arange(block_q, dtype=jnp.int32)
    qk_scale = scale * _LOG2_E

    def body(kb, carry, *, masked: bool):
        acc, m_prev, l_prev = carry
        k_slice = pl.ds(kb * block_k, block_k)
        k = k_ref[k_slice, :]
        s = plgpu.dot(q, k, trans_b=True) * qk_scale
        if masked:
            k_pos = kb * block_k + jnp.arange(block_k, dtype=jnp.int32)
            s = jnp.where(_band_mask(q_pos, q_lower, k_pos), s, _MASK_VALUE)
        m_next = jnp.maximum(m_prev, jnp.max(s, axis=1))
        correction = jnp.exp2(m_prev - m_next)
        p = jnp.exp2(s - m_next[:, None])
        l_next = l_prev * correction + jnp.sum(p, axis=1)
        v = v_ref[k_slice, :]
        acc = acc * correction[:, None] + plgpu.dot(p.astype(v.dtype), v)
        return acc, m_next, l_next

    head_dim = q_ref.shape[-1]
    carry = (
        jnp.zeros((block_q, head_dim), jnp.float32),
        jnp.full((block_q,), _MASK_VALUE, jnp.float32),
        jnp.zeros((block_q,), jnp.float32),
    )
    acc, m, l = _banded_loop(*_key_block_band(q_lower, q_start, block_q, block_k), body, carry)
    o_ref[...] = (acc / l[:, None]).astype(o_ref.dtype)
    lse_ref[...] = m + jnp.log2(l)


def _dq_kernel(
    q_ref,
    k_ref,
    v_ref,
    lower_ref,
    do_ref,
    lse_ref,
    delta_ref,
    dq_ref,
    *,
    scale: float,
    block_k: int,
    num_q_blocks: int,
):
    block_q = q_ref.shape[0]
    q_block = num_q_blocks - 1 - pl.program_id(0)
    q_start = q_block * block_q
    q = q_ref[...]
    do = do_ref[...]
    lse = lse_ref[...]
    delta = delta_ref[...]
    q_lower = lower_ref[...]
    q_pos = q_start + jnp.arange(block_q, dtype=jnp.int32)
    qk_scale = scale * _LOG2_E

    def body(kb, dq, *, masked: bool):
        k_slice = pl.ds(kb * block_k, block_k)
        k = k_ref[k_slice, :]
        v = v_ref[k_slice, :]
        s = plgpu.dot(q, k, trans_b=True) * qk_scale
        if masked:
            k_pos = kb * block_k + jnp.arange(block_k, dtype=jnp.int32)
            s = jnp.where(_band_mask(q_pos, q_lower, k_pos), s, _MASK_VALUE)
        p = jnp.exp2(s - lse[:, None])
        dp = plgpu.dot(do, v, trans_b=True)
        ds = p * (dp - delta[:, None])
        return dq + plgpu.dot(ds.astype(k.dtype), k)

    band = _key_block_band(q_lower, q_start, block_q, block_k)
    dq = _banded_loop(*band, body, jnp.zeros(q_ref.shape, jnp.float32))
    dq_ref[...] = (dq * scale).astype(dq_ref.dtype)


def _dkv_kernel(
    q_ref,
    k_ref,
    v_ref,
    lower_ref,
    do_ref,
    lse_ref,
    delta_ref,
    reach_ref,
    full_ref,
    dk_ref,
    dv_ref,
    *,
    scale: float,
    block_q: int,
    group: int,
):
    block_k = k_ref.shape[0]
    k_start = pl.program_id(0) * block_k
    k = k_ref[...]
    v = v_ref[...]
    k_pos = k_start + jnp.arange(block_k, dtype=jnp.int32)
    qk_scale = scale * _LOG2_E

    lo = k_start // block_q
    full_lo = (k_start + block_k - 1 + block_q - 1) // block_q
    full_hi = full_ref[()]
    hi = (reach_ref[()] + block_q - 1) // block_q

    dk = jnp.zeros(k_ref.shape, jnp.float32)
    dv = jnp.zeros(v_ref.shape, jnp.float32)
    for g in range(group):

        def body(qb, carry, *, masked: bool, g=g):
            dk, dv = carry
            q_slice = pl.ds(qb * block_q, block_q)
            q = q_ref[q_slice, g, :]
            do = do_ref[q_slice, g, :]
            lse = lse_ref[g, q_slice]
            delta = delta_ref[g, q_slice]
            s = plgpu.dot(q, k, trans_b=True) * qk_scale
            if masked:
                q_pos = qb * block_q + jnp.arange(block_q, dtype=jnp.int32)
                s = jnp.where(_band_mask(q_pos, lower_ref[q_slice], k_pos), s, _MASK_VALUE)
            p = jnp.exp2(s - lse[:, None])
            dv = dv + plgpu.dot(p.astype(do.dtype), do, trans_a=True)
            dp = plgpu.dot(do, v, trans_b=True)
            ds = p * (dp - delta[:, None])
            dk = dk + plgpu.dot(ds.astype(q.dtype), q, trans_a=True)
            return dk, dv

        dk, dv = _banded_loop(lo, full_lo, full_hi, hi, body, (dk, dv))
    dk_ref[...] = (dk * scale).astype(dk_ref.dtype)
    dv_ref[...] = dv.astype(dv_ref.dtype)


def _forward(q, k, v, lower, *, block_sizes: TritonFlashBlockSizes, interpret: bool):
    batch, seq_len, num_q_heads, head_dim = q.shape
    group = num_q_heads // k.shape[2]
    scale = 1.0 / math.sqrt(head_dim)
    bq = block_sizes.block_q
    nq = seq_len // bq
    out_shape = (
        jax.ShapeDtypeStruct(q.shape, q.dtype),
        jax.ShapeDtypeStruct((batch, num_q_heads, seq_len), jnp.float32),
    )
    return pl.pallas_call(
        functools.partial(_forward_kernel, scale=scale, block_k=block_sizes.block_k, num_q_blocks=nq),
        grid=(nq, batch, num_q_heads),
        in_specs=[
            pl.BlockSpec((None, bq, None, head_dim), lambda i, b, h: (b, nq - 1 - i, h, 0)),
            pl.BlockSpec((None, seq_len, None, head_dim), lambda i, b, h: (b, 0, h // group, 0)),
            pl.BlockSpec((None, seq_len, None, head_dim), lambda i, b, h: (b, 0, h // group, 0)),
            pl.BlockSpec((None, bq), lambda i, b, h: (b, nq - 1 - i)),
        ],
        out_specs=[
            pl.BlockSpec((None, bq, None, head_dim), lambda i, b, h: (b, nq - 1 - i, h, 0)),
            pl.BlockSpec((None, None, bq), lambda i, b, h: (b, h, nq - 1 - i)),
        ],
        out_shape=out_shape,
        compiler_params=plgpu.CompilerParams(num_warps=block_sizes.num_warps, num_stages=block_sizes.num_stages),
        interpret=interpret,
        name="grug_triton_flash_fwd",
    )(q, k, v, lower)


def _backward(q, k, v, lower, out, lse, do, *, block_sizes: TritonFlashBlockSizes, interpret: bool):
    batch, seq_len, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[2]
    group = num_q_heads // num_kv_heads
    scale = 1.0 / math.sqrt(head_dim)
    delta = jnp.einsum("bshd,bshd->bhs", out.astype(jnp.float32), do.astype(jnp.float32))

    bq = block_sizes.block_q_dq
    nq = seq_len // bq
    dq = pl.pallas_call(
        functools.partial(_dq_kernel, scale=scale, block_k=block_sizes.block_k_dq, num_q_blocks=nq),
        grid=(nq, batch, num_q_heads),
        in_specs=[
            pl.BlockSpec((None, bq, None, head_dim), lambda i, b, h: (b, nq - 1 - i, h, 0)),
            pl.BlockSpec((None, seq_len, None, head_dim), lambda i, b, h: (b, 0, h // group, 0)),
            pl.BlockSpec((None, seq_len, None, head_dim), lambda i, b, h: (b, 0, h // group, 0)),
            pl.BlockSpec((None, bq), lambda i, b, h: (b, nq - 1 - i)),
            pl.BlockSpec((None, bq, None, head_dim), lambda i, b, h: (b, nq - 1 - i, h, 0)),
            pl.BlockSpec((None, None, bq), lambda i, b, h: (b, h, nq - 1 - i)),
            pl.BlockSpec((None, None, bq), lambda i, b, h: (b, h, nq - 1 - i)),
        ],
        out_specs=pl.BlockSpec((None, bq, None, head_dim), lambda i, b, h: (b, nq - 1 - i, h, 0)),
        out_shape=jax.ShapeDtypeStruct(q.shape, q.dtype),
        compiler_params=plgpu.CompilerParams(num_warps=block_sizes.num_warps_dq, num_stages=block_sizes.num_stages_dq),
        interpret=interpret,
        name="grug_triton_flash_dq",
    )(q, k, v, lower, do, lse, delta)

    bk = block_sizes.block_k_dkv
    bq_dkv = block_sizes.block_q_dkv
    reach, full = _dkv_query_ranges(lower, block_k=bk, block_q=bq_dkv)
    kv_block = pl.BlockSpec((None, bk, None, head_dim), lambda j, b, h: (b, j, h, 0))
    group_rows = pl.BlockSpec((None, seq_len, group, head_dim), lambda j, b, h: (b, 0, h, 0))
    group_stats = pl.BlockSpec((None, group, seq_len), lambda j, b, h: (b, h, 0))
    block_scalar = pl.BlockSpec((None, None), lambda j, b, h: (b, j))
    dk, dv = pl.pallas_call(
        functools.partial(_dkv_kernel, scale=scale, block_q=bq_dkv, group=group),
        grid=(seq_len // bk, batch, num_kv_heads),
        in_specs=[
            group_rows,
            kv_block,
            kv_block,
            pl.BlockSpec((None, seq_len), lambda j, b, h: (b, 0)),
            group_rows,
            group_stats,
            group_stats,
            block_scalar,
            block_scalar,
        ],
        out_specs=[kv_block, kv_block],
        out_shape=[jax.ShapeDtypeStruct(k.shape, k.dtype), jax.ShapeDtypeStruct(v.shape, v.dtype)],
        compiler_params=plgpu.CompilerParams(
            num_warps=block_sizes.num_warps_dkv, num_stages=block_sizes.num_stages_dkv
        ),
        interpret=interpret,
        name="grug_triton_flash_dkv",
    )(q, k, v, lower, do, lse, delta, reach, full)
    return dq, dk, dv


@functools.partial(jax.custom_vjp, nondiff_argnums=(4, 5))
def _flash(q, k, v, lower, block_sizes: TritonFlashBlockSizes, interpret: bool):
    out, _ = _forward(q, k, v, lower, block_sizes=block_sizes, interpret=interpret)
    return out


def _flash_fwd(q, k, v, lower, block_sizes, interpret):
    out, lse = _forward(q, k, v, lower, block_sizes=block_sizes, interpret=interpret)
    return out, (q, k, v, lower, out, lse)


def _flash_bwd(block_sizes, interpret, residuals, do):
    q, k, v, lower, out, lse = residuals
    dq, dk, dv = _backward(q, k, v, lower, out, lse, do, block_sizes=block_sizes, interpret=interpret)
    return dq, dk, dv, None


_flash.defvjp(_flash_fwd, _flash_bwd)


def pallas_triton_flash_attention(
    q: Float[Array, "B S Hq D"],
    k: Float[Array, "B S Hkv D"],
    v: Float[Array, "B S Hkv D"],
    mask: AttentionMask,
    *,
    block_sizes: TritonFlashBlockSizes = TritonFlashBlockSizes(),
    interpret: bool = False,
) -> Float[Array, "B S Hq D"]:
    """Causal (optionally windowed and segmented) self-attention through the Pallas-Triton flash kernels.

    The softmax scale is ``1 / sqrt(head_dim)``, as in ``reference_attention``. Under a mesh, the kernels run per
    shard inside ``shard_map``; batch and heads may be sharded, sequence and head_dim may not. ``interpret`` runs
    the kernels in the Pallas interpreter, which works on CPU.
    """
    if not interpret and jax.default_backend() != "gpu":
        raise RuntimeError("gpu_pallas_triton_flash requires the JAX GPU backend")
    _validate(q, k, v, block_sizes)
    batch, seq_len = q.shape[:2]
    lower = _key_lower_bounds(mask, batch=batch, seq_len=seq_len)
    run = functools.partial(_flash, block_sizes=block_sizes, interpret=interpret)

    mesh = get_abstract_mesh()
    if mesh is None or mesh.empty:
        return run(q, k, v, lower)
    q_dims = partitioned_dims(q, mesh)
    if q_dims[1] or q_dims[3]:
        raise ValueError(f"gpu_pallas_triton_flash needs unsharded sequence and head_dim, got {partition_spec_of(q)}")
    for name, x in (("k", k), ("v", v)):
        if partitioned_dims(x, mesh) != q_dims:
            raise ValueError(
                f"gpu_pallas_triton_flash needs {name} sharded like q, got q={partition_spec_of(q)} "
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
