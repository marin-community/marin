# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``attention_rows_probe``: the MLA layers' full softmax rows (and Inkling bias by distance) for traced queries."""

import math
from collections.abc import Iterator
from contextlib import contextmanager

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, Int
from levanter.grug.attention import AttentionMask, align_kv_heads
from levanter.grug.attention._inkling_relpos import REL_BIAS_BLOCK, rel_extent_of_band

ROWS_PROBE: tuple[jax.Array, jax.Array] | None = None
"""Traced ``(rows, positions)`` ``[Q]`` set by ``attention_rows_probe``; read by the MLA layers at trace time."""


@contextmanager
def attention_rows_probe(rows: jax.Array, positions: jax.Array) -> Iterator[None]:
    """MLA layers traced inside this context record each query ``(rows[i], positions[i])``'s full softmax row over the
    sequence (``model.ATTN_ROWS_STAT``, ``[Q, H, S]``). The arrays may be traced, so one compile serves any queries."""
    global ROWS_PROBE
    previous, ROWS_PROBE = ROWS_PROBE, (rows, positions)
    try:
        yield
    finally:
        ROWS_PROBE = previous


def probe_attention_rows(
    q: Float[Array, "B S H D"],
    k: Float[Array, "B S Hk D"],
    mask: AttentionMask | jax.Array,
    rel_bias: Float[Array, "B H S W"] | None,
    rows: Int[Array, " Q"],
    positions: Int[Array, " Q"],
) -> Float[Array, "Q H S"]:
    """``_probe_attention`` over every key and for traced queries: scale ``1/sqrt(head_dim)``, the banded Inkling
    bias, causal, document-masked."""
    if not isinstance(mask, AttentionMask):
        raise ValueError("attention_rows_probe needs an AttentionMask")
    k = align_kv_heads(k, num_q_heads=q.shape[2])
    seq = q.shape[1]
    rep = P(None, None, None)
    q_rows = q.at[rows, positions].get(out_sharding=rep).astype(jnp.float32)  # [Q, H, D]
    k_rows = k.at[rows].get(out_sharding=P(None, None, None, None)).astype(jnp.float32)  # [Q, S, H, D]
    logits = jnp.einsum("qhd,qkhd->qhk", q_rows, k_rows) / math.sqrt(q.shape[-1])
    keys = jnp.arange(seq)
    valid = keys[None, :] <= positions[:, None]
    if rel_bias is not None:
        band = rel_bias.at[rows, :, positions].get(out_sharding=rep)  # [Q, H, W]
        column = keys[None, :] - (positions[:, None] // REL_BIAS_BLOCK) * REL_BIAS_BLOCK + rel_extent_of_band(rel_bias)
        in_band = (column >= 0) & (column < band.shape[-1])
        gathered = jnp.take_along_axis(band, jnp.clip(column, 0, band.shape[-1] - 1)[:, None, :], axis=-1)
        logits = logits + jnp.where(in_band[:, None, :], gathered, 0.0)
    if mask.segment_ids is not None:
        segments = mask.segment_ids[1].at[rows].get(out_sharding=P(None, None))  # [Q, S]
        own = jnp.take_along_axis(segments, positions[:, None], axis=1)
        valid &= segments == own
    return jax.lax.stop_gradient(jax.nn.softmax(jnp.where(valid[:, None, :], logits, -jnp.inf), axis=-1))


def relpos_rows(rel_bias: Float[Array, "B H S W"], rows: Int[Array, " Q"], positions: Int[Array, " Q"]) -> jax.Array:
    """Each query's banded Inkling bias by distance back, ``[Q, H, rel_extent]``."""
    band = rel_bias.at[rows, :, positions].get(out_sharding=P(None, None, None))  # [Q, H, W]
    extent = rel_extent_of_band(rel_bias)
    column = (positions % REL_BIAS_BLOCK)[:, None] + extent - jnp.arange(extent)[None, :]
    return jax.lax.stop_gradient(jnp.take_along_axis(band, column[:, None, :], axis=-1).astype(jnp.float32))
