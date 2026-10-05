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


def _e8m0_scale(amax: jax.Array) -> jax.Array:
    """``2^ceil(log2(amax / 448))`` from the f32 bits, so it is an exact power of two on every backend (GPU
    ``exp2``/``log2`` are approximate); 1 for all-zero blocks."""
    bits = jax.lax.bitcast_convert_type(amax / _E4M3_MAX, jnp.int32)
    exponent = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).astype(jnp.int32)
    exponent = jnp.where(amax > 0, jnp.clip(exponent, 1, 254), 127)
    return jax.lax.bitcast_convert_type(exponent << 23, jnp.float32)


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
