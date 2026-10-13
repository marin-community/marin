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

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax.sharding import reshard

MX_BLOCK = 32
_E4M3 = jnp.float8_e4m3fn
_E4M3_MAX = 448.0


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
