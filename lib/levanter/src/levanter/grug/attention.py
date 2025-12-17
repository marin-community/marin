# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0
import dataclasses
import functools as ft
import math
from dataclasses import dataclass
from typing import Protocol

import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass

from .config import AttentionRuntimeConfig, RotaryConfig


class AttentionBackend(Protocol):
    def __call__(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        mask: jax.Array | None,
        causal: bool,
    ) -> jax.Array: ...


def default_attention_mask(seq_len: int) -> jax.Array:
    """Boolean causal mask with shape (seq, seq). True == keep, False == block."""
    return jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))


@ft.partial(register_dataclass, meta_fields=["is_causal"])
@dataclass(frozen=True)
class AttentionMask:
    """Structured mask that can emit a boolean mask."""

    is_causal: bool = dataclasses.field(metadata=dict(static=True), default=False)
    causal_offset: int = 0
    explicit_mask: jax.Array | None = None
    segment_ids: tuple[jax.Array, jax.Array] | None = None
    sliding_window: int | None = None

    @classmethod
    def causal(cls, *, offset: int = 0, sliding_window: int | None = None) -> "AttentionMask":
        return cls(is_causal=True, causal_offset=offset, sliding_window=sliding_window)

    @classmethod
    def explicit(cls, mask: jax.Array) -> "AttentionMask":
        return cls(explicit_mask=mask)

    def with_segment_ids(
        self, query_segment_ids: jax.Array, kv_segment_ids: jax.Array | None = None
    ) -> "AttentionMask":
        kv_ids = query_segment_ids if kv_segment_ids is None else kv_segment_ids
        return AttentionMask(
            is_causal=self.is_causal,
            causal_offset=self.causal_offset,
            explicit_mask=self.explicit_mask,
            segment_ids=(query_segment_ids, kv_ids),
            sliding_window=self.sliding_window,
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

        if self.explicit_mask is not None:
            mask = self.explicit_mask if mask is None else jnp.logical_and(mask, self.explicit_mask)

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
    causal: bool,
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

    if mask is None and causal:
        mask = default_attention_mask(scores.shape[-2])

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


def _maybe_splash_attention() -> AttentionBackend | None:
    try:
        from jax.experimental.pallas.ops.tpu import splash_attention as splash
        from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as kernel
        from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as mask_lib
    except Exception:
        return None

    def _backend(q, k, v, mask, causal, *, runtime: AttentionRuntimeConfig):
        if runtime.head_shards is None or runtime.q_seq_shards is None:
            raise ValueError("Splash attention requires head_shards and q_seq_shards to be set in AttentionRuntimeConfig.")
        Sq = q.shape[1]
        Sk = k.shape[1]
        Hq = q.shape[2]

        # Build base mask
        base_mask: mask_lib.Mask
        if causal:
            base_mask = mask_lib.CausalMask((Sq, Sk), offset=0, shard_count=runtime.q_seq_shards)
        else:
            base_mask = mask_lib.FullMask(_shape=(Sq, Sk))

        if isinstance(mask, AttentionMask):
            if mask.segment_ids is not None:
                raise NotImplementedError("segment_ids not supported by splash attention wrapper yet.")
            if mask.sliding_window is not None:
                local = mask_lib.LocalMask(
                    (Sq, Sk),
                    window_size=(mask.sliding_window, mask.sliding_window),
                    offset=mask.causal_offset,
                    shard_count=runtime.q_seq_shards,
                )
                base_mask = mask_lib.LogicalAnd(base_mask, local)
            if mask.explicit_mask is not None:
                explicit = mask_lib.NumpyMask(jnp.array(mask.explicit_mask, dtype=bool))
                base_mask = mask_lib.LogicalAnd(base_mask, explicit)
        elif isinstance(mask, jax.Array) and mask.dtype == jnp.bool_:
            explicit = mask_lib.NumpyMask(jnp.array(mask, dtype=bool))
            base_mask = mask_lib.LogicalAnd(base_mask, explicit)

        multi_mask = mask_lib.MultiHeadMask(masks=[base_mask for _ in range(Hq)])

        # Splash currently assumes Hq is divisible by Kv heads; if not, fall back elsewhere.
        mha = kernel.make_splash_mha(
            multi_mask,
            head_shards=runtime.head_shards,
            q_seq_shards=runtime.q_seq_shards,
            is_mqa=False,
        )
        out = mha(q, k, v)
        return out

    return _backend


def resolve_attention_backend(runtime: AttentionRuntimeConfig) -> AttentionBackend:
    def _reference_backend(q, k, v, mask, causal):
        return reference_attention(q, k, v, mask, causal=causal, logits_dtype=runtime.logits_dtype)

    if runtime.backend == "reference":
        return _reference_backend
    if runtime.backend == "splash":
        backend = _maybe_splash_attention()
        if backend is None:
            raise RuntimeError("Splash attention requested but unavailable")
        # bind runtime so shard counts flow through
        return ft.partial(backend, runtime=runtime)
    backend = _maybe_splash_attention() if runtime.backend == "auto" else None
    if backend is not None:
        return ft.partial(backend, runtime=runtime)
    return _reference_backend


__all__ = [
    "AttentionBackend",
    "AttentionMask",
    "apply_rotary_embedding",
    "default_attention_mask",
    "reference_attention",
    "resolve_attention_backend",
]
