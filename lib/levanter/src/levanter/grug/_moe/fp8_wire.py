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
import numpy as np

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


def _shard_a2a_params_from_counts(shard_counts, shard_id):
    """Sender-side ragged_all_to_all params for ``shard_counts[sender, receiver]``."""
    row = shard_counts[shard_id]
    input_offsets = jnp.cumsum(jnp.concatenate((jnp.array([0], dtype=row.dtype), row[:-1])))
    send_sizes = row
    recv_sizes = shard_counts[:, shard_id]
    sender_output_offsets = jnp.cumsum(shard_counts, axis=0, dtype=shard_counts.dtype) - shard_counts
    output_offsets = sender_output_offsets[shard_id]
    return input_offsets, send_sizes, output_offsets, recv_sizes


def _dequant_sender_segments(bits, recv_sizes, scales, fp8_dtype, out_dtype):
    """Dequantize a received ragged buffer whose rows group by sender shard."""
    qr = _from_wire(bits, fp8_dtype).astype(jnp.float32)
    ends = jnp.cumsum(recv_sizes.astype(jnp.int32))
    pos = jnp.arange(qr.shape[0], dtype=jnp.int32)
    sender = jnp.clip(jnp.searchsorted(ends, pos, side="right"), 0, scales.shape[0] - 1)
    return (qr * scales[sender][:, None]).astype(out_dtype)


def _fp8_ragged_a2a_impl(x, shard_counts, shard_id, out_rows, axis_name, fp8_dtype):
    params = _shard_a2a_params_from_counts(shard_counts, shard_id)
    input_offsets, send_sizes, output_offsets, recv_sizes = params
    q, scale = _quantize(x, fp8_dtype)
    scales = jax.lax.all_gather(scale, axis_name)  # [S] sender scales
    out_buf = jnp.zeros((out_rows,) + x.shape[1:], dtype=jnp.uint8)
    bits = jax.lax.ragged_all_to_all(
        _as_wire(q),
        out_buf,
        input_offsets,
        send_sizes,
        output_offsets,
        recv_sizes,
        axis_name=axis_name,
    )
    return _dequant_sender_segments(bits, recv_sizes, scales, fp8_dtype, x.dtype)


@partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def fp8_ragged_a2a(x, shard_counts, shard_id, out_rows, axis_name):
    """``ragged_all_to_all`` carrying E4M3 forward / E5M2 backward over the wire.

    ``shard_counts[sender, receiver]`` gives the ragged send matrix (rows this
    shard sends to each peer live contiguously in ``x``, receiver-major); the
    received buffer groups rows by sender. The backward runs the reverse
    ragged_all_to_all (transposed counts) with the cotangent quantized to E5M2.
    Per-sender scales travel as an ``[S]`` scalar all_gather.
    """
    return _fp8_ragged_a2a_impl(x, shard_counts, shard_id, out_rows, axis_name, jnp.float8_e4m3fn)


def _fp8_ragged_a2a_fwd(x, shard_counts, shard_id, out_rows, axis_name):
    return fp8_ragged_a2a(x, shard_counts, shard_id, out_rows, axis_name), (shard_counts, shard_id, x.shape[0])


def _fp8_ragged_a2a_bwd(out_rows, axis_name, res, ct):
    shard_counts, shard_id, in_rows = res
    dx = _fp8_ragged_a2a_impl(ct, shard_counts.T, shard_id, in_rows, axis_name, jnp.float8_e5m2)
    zero_counts = np.zeros(shard_counts.shape, dtype=jax.dtypes.float0)
    zero_id = np.zeros(shard_id.shape, dtype=jax.dtypes.float0)
    return dx, zero_counts, zero_id


fp8_ragged_a2a.defvjp(_fp8_ragged_a2a_fwd, _fp8_ragged_a2a_bwd)
