# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0
import functools
import math
from dataclasses import dataclass

import jax
from jax import numpy as jnp
from jax.tree_util import register_dataclass

from .config import RotaryConfig


@functools.partial(
    register_dataclass, data_fields=["segment_ids"], meta_fields=["is_causal", "causal_offset", "sliding_window"]
)
@dataclass(frozen=True)
class AttentionMask:
    """Grug attention mask spec.

    This is deliberately simpler than `levanter.layers.attention.AttentionMask`:
    - Stores raw JAX arrays (no NamedArray fields).
    - Does not support explicit masks (for now).
    - Supports causal masking, sliding windows, and segment IDs.
    """

    is_causal: bool = False
    causal_offset: int = 0
    segment_ids: tuple[jax.Array, jax.Array] | None = None
    sliding_window: int | None = None

    @classmethod
    def causal(cls, *, offset: int = 0, sliding_window: int | None = None) -> "AttentionMask":
        return cls(is_causal=True, causal_offset=offset, sliding_window=sliding_window)

    def with_segment_ids(self, q_segment_ids: jax.Array, kv_segment_ids: jax.Array | None = None) -> "AttentionMask":
        kv_ids = q_segment_ids if kv_segment_ids is None else kv_segment_ids
        return AttentionMask(
            is_causal=self.is_causal,
            causal_offset=self.causal_offset,
            segment_ids=(q_segment_ids, kv_ids),
            sliding_window=self.sliding_window,
        )

    def with_sliding_window(self, sliding_window: int | None) -> "AttentionMask":
        return AttentionMask(
            is_causal=self.is_causal,
            causal_offset=self.causal_offset,
            segment_ids=self.segment_ids,
            sliding_window=sliding_window,
        )

    def materialize_mask(self, q_len: int, k_len: int) -> jax.Array | None:
        """Return a (q_len, k_len) boolean mask (True = allowed) or None."""
        mask = None

        if self.is_causal:
            q_idx = jnp.arange(q_len)[:, None]
            k_idx = jnp.arange(k_len)[None, :]
            allowed = k_idx <= q_idx + self.causal_offset
            mask = allowed

        if self.sliding_window is not None:
            q_idx = jnp.arange(q_len)[:, None]
            k_idx = jnp.arange(k_len)[None, :]
            allowed = k_idx >= q_idx - self.sliding_window
            mask = allowed if mask is None else jnp.logical_and(mask, allowed)

        if self.segment_ids is not None:
            q_seg, k_seg = self.segment_ids
            allowed = q_seg[:, None] == k_seg[None, :]
            mask = allowed if mask is None else jnp.logical_and(mask, allowed)

        return mask


def _rotary_cache(seq_len: int, head_dim: int, rope: RotaryConfig) -> tuple[jax.Array, jax.Array]:
    half_dim = head_dim // 2
    inv_freq = 1.0 / (rope.theta ** (jnp.arange(0, half_dim, dtype=jnp.float32) / half_dim))
    positions = jnp.arange(seq_len, dtype=jnp.float32)
    angles = positions[:, None] * inv_freq[None, :]
    cos = jnp.cos(angles)
    sin = jnp.sin(angles)
    return cos, sin


def apply_rotary_embedding(
    q: jax.Array,
    k: jax.Array,
    *,
    seq_len: int,
    head_dim: int,
    rope: RotaryConfig,
) -> tuple[jax.Array, jax.Array]:
    cos, sin = _rotary_cache(seq_len, head_dim, rope)
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]

    def _apply(x: jax.Array) -> jax.Array:
        x1, x2 = jnp.split(x, 2, axis=-1)
        return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)

    return _apply(q), _apply(k)


def reference_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    mask: AttentionMask | jax.Array | None,
    *,
    logits_dtype: jnp.dtype | None,
) -> jax.Array:
    head_dim = q.shape[-1]
    num_q_heads = q.shape[2]
    num_kv_heads = k.shape[2]

    if num_q_heads != num_kv_heads:
        if num_q_heads % num_kv_heads != 0:
            raise ValueError(f"num_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})")
        repeat = num_q_heads // num_kv_heads
        k = jnp.repeat(k, repeat, axis=2)
        v = jnp.repeat(v, repeat, axis=2)

    scale = 1.0 / math.sqrt(head_dim)
    scores = jnp.einsum("bqhd,bkhd->bhqk", q * scale, k)
    if isinstance(mask, AttentionMask):
        mask = mask.materialize_mask(scores.shape[-2], scores.shape[-1])

    if mask is not None:
        if mask.dtype == jnp.bool_:
            scores = jnp.where(mask[None, None, :, :], scores, jnp.array(-1e9, dtype=scores.dtype))
        else:
            scores = scores + mask[None, None, :, :]
    if logits_dtype is not None:
        scores = scores.astype(logits_dtype)
    weights = jax.nn.softmax(scores, axis=-1).astype(v.dtype)
    ctx = jnp.einsum("bhqk,bkhd->bqhd", weights, v)
    return ctx.astype(v.dtype)


def _blocksparse_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    mask: AttentionMask | jax.Array | None,
) -> jax.Array:
    from ejkernel.modules.operations import blocksparse_attention as ej_blocksparse_attention

    q_segment_ids = None
    kv_segment_ids = None
    sliding_window = None
    causal = False

    if isinstance(mask, AttentionMask):
        if mask.causal_offset != 0:
            raise NotImplementedError("Grug AttentionMask.causal_offset is not supported by ejkernel blocksparse.")

        if mask.segment_ids is not None:
            q_segment_ids, kv_segment_ids = mask.segment_ids

        sliding_window = mask.sliding_window
        causal = mask.is_causal

    if isinstance(mask, jax.Array):
        # Grug sometimes passes an explicit (q,k) boolean mask. ejkernel blocksparse_attention
        # does not accept an arbitrary dense mask; it only supports structured masks
        # (causal/window/segments) and block-sparse patterns.
        raise NotImplementedError("Dense boolean masks are not supported by ejkernel blocksparse attention.")

    out = ej_blocksparse_attention.blocksparse_attention(
        q.transpose(0, 2, 1, 3),
        k.transpose(0, 2, 1, 3),
        v.transpose(0, 2, 1, 3),
        softmax_aux=None,
        bias=None,
        q_segment_ids=q_segment_ids,
        kv_segment_ids=kv_segment_ids,
        sliding_window=sliding_window,
        causal=causal,
    )
    return out.transpose(0, 2, 1, 3)


def attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    mask: AttentionMask | jax.Array | None,
) -> jax.Array:
    return _blocksparse_attention(q, k, v, mask)


__all__ = [
    "AttentionMask",
    "apply_rotary_embedding",
    "attention",
    "reference_attention",
]
