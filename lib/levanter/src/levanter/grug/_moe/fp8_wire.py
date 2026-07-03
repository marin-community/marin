# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""FP8 over-the-wire collectives for expert-parallel MoE dispatch/combine.

Forward activations cross the wire as E4M3 and backward gradients as E5M2,
with *current* per-sender scaling: each sender quantizes with its own
amax-derived per-tensor scale and receivers dequantize with the sender's
scale, which travels alongside as one scalar per shard (a tiny all_gather /
all_to_all). Reductions never happen in FP8: the ring backend's
``psum_scatter`` combine is decomposed into an FP8 ``all_to_all`` permutation
plus a local sum in float32, so accumulation precision is unchanged.

Payloads cross the collectives bitcast to uint8 — permutation collectives
move bytes, and this keeps the wire format independent of backend FP8 dtype
support.
"""

from functools import partial

import jax
import jax.numpy as jnp

_FP8_MAX = {
    jnp.float8_e4m3fn: 448.0,
    jnp.float8_e5m2: 57344.0,
}
_WIRE_EPS = 1e-12


def _quantize(x, fp8_dtype, amax_axes=None):
    """Quantize with a current per-tensor (or per-leading-axis) scale."""
    xf = x.astype(jnp.float32)
    amax = jnp.max(jnp.abs(xf), axis=amax_axes, keepdims=False)
    scale = jnp.maximum(amax, _WIRE_EPS) / _FP8_MAX[jnp.dtype(fp8_dtype).type]
    if amax_axes is not None:
        expand = (slice(None),) + (None,) * (xf.ndim - 1)
        q = (xf / scale[expand]).astype(fp8_dtype)
    else:
        q = (xf / scale).astype(fp8_dtype)
    return q, scale


def _as_wire(q):
    return jax.lax.bitcast_convert_type(q, jnp.uint8)


def _from_wire(bits, fp8_dtype):
    return jax.lax.bitcast_convert_type(bits, fp8_dtype)


def _axis_size(axis_name):
    return jax.lax.psum(1, axis_name)


def _fp8_all_gather_impl(x, axis_name, fp8_dtype, out_dtype):
    """all_gather in FP8; returns the tiled bf16 gather [S*T, ...]."""
    q, scale = _quantize(x, fp8_dtype)
    bits = jax.lax.all_gather(_as_wire(q), axis_name)  # [S, T, ...]
    scales = jax.lax.all_gather(scale, axis_name)  # [S]
    qg = _from_wire(bits, fp8_dtype).astype(jnp.float32)
    expand = (slice(None),) + (None,) * (qg.ndim - 1)
    out = (qg * scales[expand]).astype(out_dtype)
    return out.reshape((-1,) + out.shape[2:])


def _fp8_all_to_all_sum_impl(x, axis_name, fp8_dtype, out_dtype):
    """Decomposed psum_scatter: per-destination-chunk FP8 quantize -> all_to_all -> local f32 sum.

    ``x`` is the tiled psum_scatter operand ``[S*T, ...]`` laid out in S
    destination-major chunks; returns this shard's reduced ``[T, ...]`` slice.
    """
    s = _axis_size(axis_name)
    chunks = x.reshape((s, -1) + x.shape[1:])  # [S, T, ...]
    q, scales = _quantize(chunks, fp8_dtype, amax_axes=tuple(range(1, chunks.ndim)))  # scales [S]
    bits = jax.lax.all_to_all(_as_wire(q), axis_name, split_axis=0, concat_axis=0)
    recv_scales = jax.lax.all_to_all(scales, axis_name, split_axis=0, concat_axis=0)  # [S] sender scales
    qr = _from_wire(bits, fp8_dtype).astype(jnp.float32)
    expand = (slice(None),) + (None,) * (qr.ndim - 1)
    return jnp.sum(qr * recv_scales[expand], axis=0).astype(out_dtype)


@partial(jax.custom_vjp, nondiff_argnums=(1,))
def fp8_all_gather(x, axis_name):
    """Tiled ``all_gather`` carrying E4M3 forward / E5M2 backward over the wire.

    Forward matches ``jax.lax.all_gather(x, axis_name, tiled=True)`` up to FP8
    quantization of the payload; the backward (a ``psum_scatter`` in the exact
    arithmetic) runs as an E5M2 all_to_all permutation + local f32 sum.
    """
    return _fp8_all_gather_impl(x, axis_name, jnp.float8_e4m3fn, x.dtype)


def _fp8_all_gather_fwd(x, axis_name):
    return fp8_all_gather(x, axis_name), None


def _fp8_all_gather_bwd(axis_name, _res, ct):
    # The wire preserves dtype end-to-end, so ct.dtype is the input dtype.
    return (_fp8_all_to_all_sum_impl(ct, axis_name, jnp.float8_e5m2, ct.dtype),)


fp8_all_gather.defvjp(_fp8_all_gather_fwd, _fp8_all_gather_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(1,))
def fp8_psum_scatter(y, axis_name):
    """Tiled ``psum_scatter`` with the reduction pulled out of the wire.

    Forward matches ``jax.lax.psum_scatter(y, axis_name, scatter_dimension=0,
    tiled=True)`` up to FP8: the payload crosses as an E4M3 all_to_all and the
    sum happens locally in f32. Backward (an all_gather in the exact
    arithmetic) carries E5M2.
    """
    return _fp8_all_to_all_sum_impl(y, axis_name, jnp.float8_e4m3fn, y.dtype)


def _fp8_psum_scatter_fwd(y, axis_name):
    return fp8_psum_scatter(y, axis_name), None


def _fp8_psum_scatter_bwd(axis_name, _res, ct):
    return (_fp8_all_gather_impl(ct, axis_name, jnp.float8_e5m2, ct.dtype),)


fp8_psum_scatter.defvjp(_fp8_psum_scatter_fwd, _fp8_psum_scatter_bwd)
