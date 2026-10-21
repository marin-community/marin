# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Simulated MXFP8 training matmuls (OCP microscaling: e4m3 elements, one power-of-two e8m0 scale per 32 elements
along the contraction axis).

Every GEMM of a linear (forward, dgrad, wgrad) quantizes both operands along *its own* contraction axis, as
block-scaled tensor cores require, and gradients use e4m3 like the activations (the MXFP8 recipe needs no e5m2).
Scales are computed just in time from each block's amax, so nothing depends on width, step or a calibration
history. The rounding is exact e4m3fn; the GEMMs themselves run in the input dtype, so this measures the recipe's
numerics on any accelerator, not its speed.
"""

import functools
from enum import StrEnum

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P
from jax.sharding import reshard

MX_BLOCK = 32
_E4M3 = jnp.float8_e4m3fn
_E4M3_MAX = 448.0


class ExpertMxfp8(StrEnum):
    """Which routed-expert GEMMs run with MXFP8 operands on QuACK's block-scaled SM100 kernels."""

    NONE = "none"
    ACTIVATIONS = "activations"
    """The activation-path GEMMs (forward and input gradients); the weight gradients stay bf16."""
    ALL = "all"
    """The activation-path GEMMs and the weight gradients, quantized along the token axis per expert."""


class MxWeightQuant(StrEnum):
    """How the block-scaled expert GEMMs quantize their weights."""

    BLOCKS = "blocks"
    """Per use, one scale per 32 elements along that GEMM's contraction axis (OCP MX): the forward and the input
    gradient quantize the weight separately."""
    TILES = "tiles"
    """One scale per 32 x 32 tile (``mx_quantize_tiles``), valid along both axes: the forward and the input gradient
    share one quantization. Coarser scales than ``BLOCKS`` (a numerics change)."""


def mx_quantize_dequantize(x: jax.Array, axis: int) -> jax.Array:
    """Round ``x`` to MXFP8 with 32-element blocks along ``axis`` and return it in ``x.dtype``."""
    axis = axis % x.ndim
    moved = jnp.moveaxis(x, axis, -1).astype(jnp.float32)
    length = moved.shape[-1]
    pad = -length % MX_BLOCK
    if pad:
        moved = jnp.pad(moved, [(0, 0)] * (moved.ndim - 1) + [(0, pad)])
    blocks = moved.reshape(*moved.shape[:-1], -1, MX_BLOCK)
    scale = _e8m0_scale(jnp.max(jnp.abs(blocks), axis=-1, keepdims=True))
    # The barrier keeps XLA from folding the f32 -> f8 -> f32 round trip away under excess-precision rules.
    q = jax.lax.optimization_barrier((blocks / scale).astype(_E4M3))
    out = (q.astype(jnp.float32) * scale).reshape(moved.shape)[..., :length]
    return jnp.moveaxis(out, -1, axis).astype(x.dtype)


def _e8m0_exponent(amax: jax.Array) -> jax.Array:
    """The biased e8m0 exponent of ``2^ceil(log2(amax / 448))``, read from the f32 bits so the scale is an exact
    power of two on every backend (GPU ``exp2``/``log2`` are approximate); 127 (a scale of 1) for all-zero blocks."""
    bits = jax.lax.bitcast_convert_type(amax / _E4M3_MAX, jnp.int32)
    exponent = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).astype(jnp.int32)
    return jnp.where(amax > 0, jnp.clip(exponent, 1, 254), 127)


def _e8m0_scale(amax: jax.Array) -> jax.Array:
    return jax.lax.bitcast_convert_type(_e8m0_exponent(amax) << 23, jnp.float32)


def mx_quantize_rows(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """MXFP8 ``x[..., K]`` along its last axis: e4m3 values ``[..., K]`` and biased e8m0 exponents as uint8
    ``[..., K / 32]``, the operand format of block-scaled tensor cores."""
    *lead, k = x.shape
    if k % MX_BLOCK:
        raise ValueError(f"last dim {k} must divide {MX_BLOCK}")
    blocks = x.astype(jnp.float32).reshape(*lead, k // MX_BLOCK, MX_BLOCK)
    exponent = _e8m0_exponent(jnp.max(jnp.abs(blocks), axis=-1))
    scale = jax.lax.bitcast_convert_type(exponent << 23, jnp.float32)
    values = (blocks / scale[..., None]).astype(_E4M3).reshape(*lead, k)
    return values, exponent.astype(jnp.uint8)


def mx_quantize_cols(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """MXFP8 ``x[..., T, K]`` along ``T`` (32-row blocks), in place: e4m3 ``[..., T, K]`` and biased e8m0 exponents
    ``[..., T / 32, K]``."""
    *lead, t, k = x.shape
    if t % MX_BLOCK:
        raise ValueError(f"row count {t} must divide {MX_BLOCK}")
    blocks = x.astype(jnp.float32).reshape(*lead, t // MX_BLOCK, MX_BLOCK, k)
    exponent = _e8m0_exponent(jnp.max(jnp.abs(blocks), axis=-2))
    scale = jax.lax.bitcast_convert_type(exponent << 23, jnp.float32)
    values = (blocks / scale[..., None, :]).astype(_E4M3).reshape(*lead, t, k)
    return values, exponent.astype(jnp.uint8)


def mx_quantize_tiles(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """MXFP8 ``x[..., R, C]`` with one scale per 32 x 32 tile: e4m3 values ``[..., R, C]`` and biased e8m0
    exponents ``[..., R / 32, C / 32]``.

    A tile's scale is valid for 32-element blocks along either axis, so the same values feed a block-scaled GEMM
    that contracts over ``R`` and one that contracts over ``C`` (a weight's forward and input-gradient GEMMs). The
    scale is the tile's amax, at least every 1 x 32 block's, so values round as coarsely or more."""
    *lead, r, c = x.shape
    if r % MX_BLOCK or c % MX_BLOCK:
        raise ValueError(f"both trailing dims must divide {MX_BLOCK}, got {x.shape}")
    blocks = x.astype(jnp.float32).reshape(*lead, r // MX_BLOCK, MX_BLOCK, c // MX_BLOCK, MX_BLOCK)
    exponent = _e8m0_exponent(jnp.max(jnp.abs(blocks), axis=(-3, -1)))
    scale = jax.lax.bitcast_convert_type(exponent << 23, jnp.float32)
    values = (blocks / scale[..., :, None, :, None]).astype(_E4M3).reshape(x.shape)
    return values, exponent.astype(jnp.uint8)


def mx_quantize_dequantize_groups(x: jax.Array, group_sizes: jax.Array) -> jax.Array:
    """Round ``x[T, N]`` to MXFP8 along the token axis with 32-row blocks that restart at every row group (rows
    ``cu[g] + 32 k`` onward), so no block mixes groups; the numerics of the MXFP8 expert weight gradients. Rows past
    ``sum(group_sizes)`` are zeroed."""
    rows = x.shape[0]
    num_groups = group_sizes.shape[0]
    cu = jnp.concatenate([jnp.zeros((1,), jnp.int32), jnp.cumsum(group_sizes.astype(jnp.int32))])
    row = jnp.arange(rows, dtype=jnp.int32)
    group = jnp.searchsorted(cu[1:], row, side="right").astype(jnp.int32)
    clamped = jnp.minimum(group, num_groups - 1)
    blocks_per_group = rows // MX_BLOCK + 1
    num_blocks = num_groups * blocks_per_group
    local_block = (row - cu[clamped]) // MX_BLOCK
    block = jnp.where(group < num_groups, clamped * blocks_per_group + local_block, num_blocks)
    magnitude = jnp.abs(x.astype(jnp.float32))
    amax = jax.ops.segment_max(magnitude, block, num_segments=num_blocks + 1)
    scale = _e8m0_scale(jnp.maximum(amax, 0.0))[block]
    q = jax.lax.optimization_barrier((x.astype(jnp.float32) / scale).astype(_E4M3))
    out = q.astype(jnp.float32) * scale
    return jnp.where((group < num_groups)[:, None], out, 0.0).astype(x.dtype)


def mx_quantize_row_blocks(
    x: jax.Array, block_starts: jax.Array, block_rows: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """MXFP8 ``x[T, N]`` along ``T`` over arbitrary disjoint row blocks: block ``j`` is rows
    ``[block_starts[j], block_starts[j] + block_rows[j])`` (at most 32). Returns e4m3 ``[T, N]`` in place, where rows
    in no block are unspecified, and uint8 exponents ``[J, N]`` (127 for an empty block)."""
    rows = x.shape[0]
    offset = jnp.arange(MX_BLOCK, dtype=jnp.int32)
    valid = offset[None, :] < block_rows[:, None]  # [J, 32]
    index = jnp.where(valid, block_starts[:, None] + offset[None, :], rows)
    blocks = jnp.where(valid[..., None], x.astype(jnp.float32).at[index].get(mode="fill", fill_value=0), 0.0)
    exponent = _e8m0_exponent(jnp.max(jnp.abs(blocks), axis=1))  # [J, N]
    scale = jax.lax.bitcast_convert_type(exponent << 23, jnp.float32)
    values = (blocks / scale[:, None, :]).astype(_E4M3)
    out = jnp.zeros(x.shape, _E4M3).at[index.reshape(-1)].set(values.reshape(-1, x.shape[1]), mode="drop")
    return out, exponent.astype(jnp.uint8)


def mx_dequantize_rows(values: jax.Array, exponents: jax.Array, dtype: jnp.dtype) -> jax.Array:
    *lead, k = values.shape
    scale = jax.lax.bitcast_convert_type(exponents.astype(jnp.int32) << 23, jnp.float32)
    blocks = values.astype(jnp.float32).reshape(*lead, k // MX_BLOCK, MX_BLOCK) * scale[..., None]
    return blocks.reshape(*lead, k).astype(dtype)


def mx_packed_row_bytes(k: int) -> int:
    return k + k // MX_BLOCK


def mx_pack_rows(x: jax.Array) -> jax.Array:
    """``mx_quantize_rows`` as one uint8 row ``[..., K + K / 32]`` (values, then exponents), for byte transports."""
    values, exponents = mx_quantize_rows(x)
    return jnp.concatenate([jax.lax.bitcast_convert_type(values, jnp.uint8), exponents], axis=-1)


def mx_unpack_rows(packed: jax.Array, k: int) -> tuple[jax.Array, jax.Array]:
    if packed.shape[-1] != mx_packed_row_bytes(k):
        raise ValueError(f"packed rows of width {packed.shape[-1]} do not hold {k} MXFP8 values")
    return jax.lax.bitcast_convert_type(packed[..., :k], _E4M3), packed[..., k:]


def _spec(x: jax.Array) -> P:
    return jax.typeof(x).sharding.spec


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def mx_dense(x: jax.Array, w: jax.Array, out_sharding: P | None = None) -> jax.Array:
    """``x[..., k] @ w[k, n]`` with MXFP8 operands in all three GEMMs."""
    return _mx_dense_fwd(x, w, out_sharding)[0]


def _mx_dense_fwd(x, w, out_sharding):
    w_full = reshard(w, P(None, None))
    y = jnp.einsum(
        "...k,kn->...n",
        mx_quantize_dequantize(x, -1),
        mx_quantize_dequantize(w_full, 0),
        out_sharding=out_sharding,
    )
    return y, (x, w)


def _mx_dense_bwd(out_sharding, res, g):
    x, w = res
    w_full = reshard(w, P(None, None))
    dx = jnp.einsum(
        "...n,kn->...k", mx_quantize_dequantize(g, -1), mx_quantize_dequantize(w_full, 1), out_sharding=_spec(x)
    )
    # wgrad contracts over tokens: blocks run along the sequence (last leading) axis, which is never sharded.
    token_axis = x.ndim - 2
    dw = jnp.einsum(
        "...k,...n->kn",
        mx_quantize_dequantize(x, token_axis),
        mx_quantize_dequantize(g, token_axis),
        out_sharding=_spec(w),
    )
    return dx, dw.astype(w.dtype)


mx_dense.defvjp(_mx_dense_fwd, _mx_dense_bwd)


@jax.custom_vjp
def mx_batched(x: jax.Array, w: jax.Array) -> jax.Array:
    """Per-expert ``x[e, r, k] @ w[e, k, n]`` with MXFP8 operands in all three GEMMs (local arrays)."""
    return jnp.einsum("erk,ekn->ern", mx_quantize_dequantize(x, 2), mx_quantize_dequantize(w, 1))


def _mx_batched_fwd(x, w):
    return mx_batched(x, w), (x, w)


def _mx_batched_bwd(res, g):
    x, w = res
    dx = jnp.einsum("ern,ekn->erk", mx_quantize_dequantize(g, 2), mx_quantize_dequantize(w, 2))
    dw = jnp.einsum("erk,ern->ekn", mx_quantize_dequantize(x, 1), mx_quantize_dequantize(g, 1))
    return dx, dw.astype(w.dtype)


mx_batched.defvjp(_mx_batched_fwd, _mx_batched_bwd)


def _ragged(x: jax.Array, w: jax.Array, group_sizes: jax.Array) -> jax.Array:
    return jax.lax.ragged_dot(x, w, group_sizes, preferred_element_type=jnp.float32)


def _swiglu_pairs(gu: jax.Array) -> jax.Array:
    gu = gu.astype(jnp.float32)
    return jax.nn.silu(gu[:, 0::2]) * gu[:, 1::2]


@jax.custom_vjp
def mx_swiglu_experts(x: jax.Array, w13_il: jax.Array, w_down: jax.Array, group_sizes: jax.Array) -> jax.Array:
    """Simulated MXFP8 gated SwiGLU experts, ``(silu(x W_g) * x W_u) W_down`` per row group, with the numerics of
    the QuACK MXFP8 gated expert MLP (``quack_mxfp8.swiglu_expert_mlp_mxfp8_fwd``), its reference.

    ``w13_il`` is the interleaved ``[E, H, 2I]`` gate/up bank (``[g0, u0, g1, u1, ...]``), which decides the
    32-blocks of the input-gradient GEMM's contraction. The four activation-path GEMMs quantize both operands along
    their contraction axis; ``gu`` is kept in ``x.dtype`` and ``h`` is quantized from the f32 accumulator, as the
    kernel's epilogue does. The weight gradients are unquantized, as in the kernel path."""
    return _mx_swiglu_experts_fwd(x, w13_il, w_down, group_sizes)[0]


def _mx_swiglu_experts_fwd(x, w13_il, w_down, group_sizes):
    gu = _ragged(mx_quantize_dequantize(x, -1), mx_quantize_dequantize(w13_il, 1), group_sizes).astype(x.dtype)
    h = mx_quantize_dequantize(_swiglu_pairs(gu), -1)
    y = _ragged(h, mx_quantize_dequantize(w_down, 1), group_sizes).astype(x.dtype)
    return y, (x, w13_il, w_down, gu, group_sizes)


def _mx_swiglu_experts_bwd(res, dy):
    x, w13_il, w_down, gu, group_sizes = res
    dtype = x.dtype
    dh = _ragged(
        mx_quantize_dequantize(dy, -1), jnp.swapaxes(mx_quantize_dequantize(w_down, 2), 1, 2), group_sizes
    ).astype(dtype)
    gate, up = gu[:, 0::2], gu[:, 1::2]
    sg = jax.nn.sigmoid(gate)
    silu = gate * sg
    d_gu = jnp.stack([dh * up * (sg + silu * (1.0 - sg)), dh * silu], axis=-1).reshape(gu.shape).astype(dtype)
    dx = _ragged(
        mx_quantize_dequantize(d_gu, -1), jnp.swapaxes(mx_quantize_dequantize(w13_il, 2), 1, 2), group_sizes
    ).astype(dtype)
    h = _swiglu_pairs(gu).astype(dtype)
    dw13 = _ragged_wgrad(x, d_gu, w13_il, group_sizes)
    dw_down = _ragged_wgrad(h, dy, w_down, group_sizes)
    return dx, dw13, dw_down, np.zeros(group_sizes.shape, dtype=jax.dtypes.float0)


def _ragged_wgrad(a: jax.Array, g: jax.Array, w: jax.Array, group_sizes: jax.Array) -> jax.Array:
    """Per-group ``a^T g`` (the weight gradient of ``ragged_dot(a, w)``), f32-accumulated, in ``w.dtype``."""
    _, pullback = jax.vjp(lambda b: _ragged(a.astype(jnp.float32), b, group_sizes), w.astype(jnp.float32))
    return pullback(g.astype(jnp.float32))[0].astype(w.dtype)


mx_swiglu_experts.defvjp(_mx_swiglu_experts_fwd, _mx_swiglu_experts_bwd)
