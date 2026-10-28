# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``trace_capture``: forwards traced inside it record every AttnRes layer's attention and MLP outputs, the final
gate's sources and weights, and each MLA query's top attended keys (read by ``trace_probe.trace_stats``)."""

import math
from collections.abc import Iterator
from contextlib import contextmanager

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax.sharding import reshard
from jaxtyping import Array, Float, Int
from levanter.grug.attention import align_kv_heads
from levanter.grug.attention._inkling_relpos import REL_BIAS_BLOCK, rel_extent_of_band

TRACE = False
"""Set by ``trace_capture``; read by the model at trace time."""
TRACE_TOP_KEYS = 6
_QUERY_BLOCK = 256
TOPK_FIELDS = ("p", "logit_shifted", "logit_current", "logit_bias")
"""Per top key: softmax weight and the q.k logit over the ``mla_key_offset`` shifted channels (the previous token's
key), over the rest (the key's own token), and the Inkling bias."""


@contextmanager
def trace_capture() -> Iterator[None]:
    global TRACE
    previous, TRACE = TRACE, True
    try:
        yield
    finally:
        TRACE = previous


def attention_topk(
    q: Float[Array, "B S H D"],
    k: Float[Array, "B S Hk D"],
    rel_bias: Float[Array, "B H S W"] | None,
    segment_ids: Int[Array, "B S"],
    shifted: jax.Array,
    batch_axes,
) -> tuple[Int[Array, "B S H K"], Float[Array, "B S H K F"]]:
    """Each query's ``TRACE_TOP_KEYS`` most-attended keys (causal, same document) and their ``TOPK_FIELDS``, from the
    kernel's inputs (scale ``1/sqrt(head_dim)``). ``shifted`` ``[H, D]`` marks the key-offset channels."""
    b, s, h, d = q.shape
    spec = P(batch_axes, None, None, None)
    q = reshard(q, spec)
    k = reshard(align_kv_heads(k, num_q_heads=h), spec)
    shifted = jnp.broadcast_to(shifted, (h, d)).astype(jnp.float32)
    keys = jnp.arange(s)
    scale = 1.0 / math.sqrt(d)

    def one_block(i):
        rows = i * _QUERY_BLOCK + jnp.arange(_QUERY_BLOCK)
        qb = jax.lax.dynamic_slice_in_dim(q, i * _QUERY_BLOCK, _QUERY_BLOCK, axis=1).astype(jnp.float32)
        kf = k.astype(jnp.float32)
        part_s = jnp.einsum("bqhd,bkhd->bhqk", qb * shifted[None, None], kf) * scale
        part_c = jnp.einsum("bqhd,bkhd->bhqk", qb * (1.0 - shifted)[None, None], kf) * scale
        bias = jnp.zeros_like(part_s)
        if rel_bias is not None:
            band = jax.lax.dynamic_slice_in_dim(rel_bias, i * _QUERY_BLOCK, _QUERY_BLOCK, axis=2)  # [B, H, Q, W]
            column = keys[None, :] - (rows[:, None] // REL_BIAS_BLOCK) * REL_BIAS_BLOCK + rel_extent_of_band(rel_bias)
            in_band = (column >= 0) & (column < band.shape[-1])
            idx = jnp.broadcast_to(jnp.clip(column, 0, band.shape[-1] - 1), (b, h, _QUERY_BLOCK, s))
            bias = jnp.where(in_band, jnp.take_along_axis(band.astype(jnp.float32), idx, axis=-1), 0.0)
        seg_q = jax.lax.dynamic_slice_in_dim(segment_ids, i * _QUERY_BLOCK, _QUERY_BLOCK, axis=1)
        ok = (keys[None, None, :] <= rows[None, :, None]) & (segment_ids[:, None, :] == seg_q[:, :, None])
        logits = jnp.where(ok[:, None], part_s + part_c + bias, -jnp.inf)
        p = jax.nn.softmax(logits, axis=-1)
        top_p, top_i = jax.lax.top_k(p, TRACE_TOP_KEYS)  # [B, H, Q, K]
        pick = lambda x: jnp.take_along_axis(x, top_i, axis=-1)  # noqa: E731
        fields = jnp.stack([top_p, pick(part_s), pick(part_c), pick(bias)], axis=-1)
        return top_i, fields

    top_i, fields = jax.lax.map(one_block, jnp.arange(s // _QUERY_BLOCK))  # [N, B, H, Q, K(, F)]
    top_i = jnp.moveaxis(top_i, 0, 2).reshape(b, h, s, TRACE_TOP_KEYS).transpose(0, 2, 1, 3)
    fields = jnp.moveaxis(fields, 0, 2).reshape(b, h, s, TRACE_TOP_KEYS, len(TOPK_FIELDS)).transpose(0, 2, 1, 3, 4)
    return jax.lax.stop_gradient(top_i), jax.lax.stop_gradient(fields)
