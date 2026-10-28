# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Packed Inkling attention with saved probabilities and bounded workspaces.

The local XLA kernel is the reference for the TPU implementation. Forward saves
FP32 softmax probabilities. Backward computes each derivative once and never
reconstructs logits, probabilities, or the attention branch. Aligned band slices
replace the source's globally broadcast dense gather indices.
"""

import functools
import math

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, Int
from levanter.grug.attention import AttentionMask, align_kv_heads

from experiments.grug.belay_tpu.causal_attention import causal_attention_local
from experiments.grug.belay_tpu.inkling_pallas import pallas_attention_local

BLOCK = 128


def _band_tile(band, start, length):
    extent = band.shape[-1] - BLOCK
    rows = jax.lax.dynamic_slice_in_dim(band, start, BLOCK, axis=1)
    padded = jnp.pad(rows, ((0, 0), (0, 0), (length, length)))
    return jax.lax.dynamic_slice_in_dim(padded, length + extent - start, length, axis=2)


def _band_adjoint(ds, start, width):
    # Every valid band position maps to exactly one dense key. This slice is the
    # transpose of _band_tile, without a scatter or a broadcast index tensor.
    padded = jnp.pad(ds, ((0, 0), (0, 0), (width, width)))
    return jax.lax.dynamic_slice_in_dim(padded, start + BLOCK, width, axis=2)


def _forward(q, k, v, band, q_segments, k_segments):
    groups, length, dim = q.shape
    scaled_q = q * (1.0 / math.sqrt(dim))

    def block(index):
        start = index * BLOCK
        qt = jax.lax.dynamic_slice_in_dim(scaled_q, start, BLOCK, axis=1)
        logits = jnp.einsum("gqd,gkd->gqk", qt, k)
        logits = logits + _band_tile(band, start, length).astype(logits.dtype)
        ids = jax.lax.dynamic_slice_in_dim(q_segments, start, BLOCK, axis=1)
        valid = (start + jnp.arange(BLOCK)[:, None] >= jnp.arange(length)[None, :])[None]
        valid = valid & (ids[:, :, None] == k_segments[:, None, :])
        logits = jnp.where(valid, logits, jnp.asarray(-1e9, logits.dtype))
        probabilities = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
        output = jnp.einsum("gqk,gkd->gqd", probabilities.astype(v.dtype), v)
        return output, probabilities

    outputs, probabilities = jax.lax.map(block, jnp.arange(length // BLOCK))
    output = outputs.transpose(1, 0, 2, 3).reshape(groups, length, dim)
    shape_token = jnp.zeros((0, band.shape[-1]), band.dtype)
    return output, (scaled_q, k, v, probabilities, q_segments, k_segments, shape_token)


@jax.custom_vjp
def saved_attention_local(
    q: Float[Array, "G S D"],
    k: Float[Array, "G S D"],
    v: Float[Array, "G S D"],
    band: Float[Array, "G S W"],
    q_segments: Int[Array, "G S"],
    k_segments: Int[Array, "G S"],
) -> Float[Array, "G S D"]:
    """Attention on fully device-local arrays, with no forward recomputation."""
    return _forward(q, k, v, band, q_segments, k_segments)[0]


def _backward(residual, dout):
    q, k, v, probabilities, q_segments, k_segments, shape_token = residual
    groups, length, dim = q.shape
    width = shape_token.shape[-1]

    def block(carry, index):
        dk, dv = carry
        start = index * BLOCK
        qt = jax.lax.dynamic_slice_in_dim(q, start, BLOCK, axis=1)
        dy = jax.lax.dynamic_slice_in_dim(dout, start, BLOCK, axis=1)
        prob = probabilities[index]
        # Keep the source's BF16 dP boundary. Without the barrier, XLA can
        # eliminate this round-trip cast and retain the unrounded FP32 dot.
        dp = jax.lax.optimization_barrier(jnp.einsum("gqd,gkd->gqk", dy, v)).astype(jnp.float32)
        weighted = prob * dp
        # Preserve the source softmax custom-JVP transpose's operation order.
        # Factoring out prob changes cancellation in saturated BF16 attention.
        ds = (weighted - prob * jnp.sum(weighted, axis=-1, keepdims=True)).astype(q.dtype)
        ids = jax.lax.dynamic_slice_in_dim(q_segments, start, BLOCK, axis=1)
        valid = (start + jnp.arange(BLOCK)[:, None] >= jnp.arange(length)[None, :])[None]
        valid = valid & (ids[:, :, None] == k_segments[:, None, :])
        ds = jnp.where(valid, ds, 0)
        dq = jnp.einsum("gqk,gkd->gqd", ds, k) * (1.0 / math.sqrt(dim))
        dk = dk + jnp.einsum("gqk,gqd->gkd", ds, qt, preferred_element_type=jnp.float32)
        dv = dv + jnp.einsum("gqk,gqd->gkd", prob.astype(v.dtype), dy, preferred_element_type=jnp.float32)
        return (dk, dv), (dq, _band_adjoint(ds, start, width).astype(shape_token.dtype))

    (dk, dv), (dq, dband) = jax.lax.scan(
        block, (jnp.zeros(k.shape, jnp.float32), jnp.zeros(v.shape, jnp.float32)), jnp.arange(length // BLOCK)
    )
    dq = dq.transpose(1, 0, 2, 3).reshape(groups, length, dim)
    dband = dband.transpose(1, 0, 2, 3).reshape(groups, length, width)
    return dq, dk.astype(k.dtype), dv.astype(v.dtype), dband, None, None


saved_attention_local.defvjp(_forward, _backward)


def inkling_attention(q, k, v, mask: AttentionMask, band, *, implementation: str):
    """Shard batch/heads before calling the Ckpt 1 causal attention kernel."""
    if implementation == "pallas":
        kernel = pallas_attention_local
    elif implementation == "causal":
        kernel = functools.partial(causal_attention_local, bucket_size=512)
    elif implementation == "xla":
        kernel = saved_attention_local
    else:
        raise ValueError(f"Unknown saved-probability attention implementation: {implementation}")
    if not isinstance(mask, AttentionMask) or not mask.is_causal or mask.sliding_window is not None:
        raise ValueError("Ckpt 1 kernel requires causal self-attention without a sliding window")
    if q.shape[1] % BLOCK or q.shape[1] != k.shape[1] or q.shape[-1] != v.shape[-1]:
        raise ValueError("Ckpt 1 attention requires aligned equal query/key lengths and head widths")
    k = align_kv_heads(k, num_q_heads=q.shape[2])
    v = align_kv_heads(v, num_q_heads=q.shape[2])
    batch_axes = ("replica_dcn", "data", "expert")
    spec = P(batch_axes, None, "model", None)
    band_spec = P(batch_axes, "model", None, None)
    segment_spec = P(batch_axes, None)
    q, k, v = (jax.sharding.reshard(x, spec) for x in (q, k, v))
    band = jax.sharding.reshard(band, band_spec)
    if mask.segment_ids is None:
        ids = jnp.zeros(q.shape[:2], jnp.int32, out_sharding=segment_spec)
        qids, kids = ids, ids
    else:
        qids, kids = (jax.sharding.reshard(x, segment_spec) for x in mask.segment_ids)

    @functools.partial(
        jax.shard_map,
        in_specs=(spec, spec, spec, band_spec, segment_spec, segment_spec),
        out_specs=spec,
        check_vma=False,
    )
    def local(ql, kl, vl, bl, qi, ki):
        b, s, h, d = ql.shape
        arrays = (x.transpose(0, 2, 1, 3).reshape(b * h, s, d) for x in (ql, kl, vl))
        qi = jnp.broadcast_to(qi[:, None, :], (b, h, s)).reshape(b * h, s)
        ki = jnp.broadcast_to(ki[:, None, :], (b, h, s)).reshape(b * h, s)
        result = kernel(*arrays, bl.reshape(b * h, s, -1), qi, ki)
        return result.reshape(b, h, s, d).transpose(0, 2, 1, 3)

    return local(q, k, v, band, qids, kids)
