from __future__ import annotations

import math
from typing import Protocol

import jax
import jax.numpy as jnp

from .config import AttentionRuntimeConfig, RotaryConfig


class AttentionBackend(Protocol):
    def __call__(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        mask: jax.Array | None,
        causal: bool,
    ) -> jax.Array:
        ...


def default_attention_mask(seq_len: int, *, dtype: jnp.dtype = jnp.float32) -> jax.Array:
    tril = jnp.tril(jnp.ones((seq_len, seq_len), dtype=dtype))
    return jnp.where(tril == 1, 0.0, jnp.array(-1e9, dtype=dtype))


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
    mask: jax.Array | None,
    *,
    causal: bool,
    logits_dtype: jnp.dtype | None,
) -> jax.Array:
    head_dim = q.shape[-1]
    num_q_heads = q.shape[2]
    num_kv_heads = k.shape[2]

    if num_q_heads != num_kv_heads:
        if num_q_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
            )
        repeat = num_q_heads // num_kv_heads
        k = jnp.repeat(k, repeat, axis=2)
        v = jnp.repeat(v, repeat, axis=2)

    scale = 1.0 / math.sqrt(head_dim)
    scores = jnp.einsum("bqhd,bkhd->bhqk", q * scale, k)
    if mask is None and causal:
        mask = default_attention_mask(scores.shape[-2], dtype=scores.dtype)
    if mask is not None:
        scores = scores + mask[None, None, :, :]
    if logits_dtype is not None:
        scores = scores.astype(logits_dtype)
    weights = jax.nn.softmax(scores, axis=-1).astype(v.dtype)
    ctx = jnp.einsum("bhqk,bkhd->bqhd", weights, v)
    return ctx.astype(v.dtype)


def _maybe_splash_attention() -> AttentionBackend | None:
    return None


def resolve_attention_backend(runtime: AttentionRuntimeConfig) -> AttentionBackend:
    def _reference_backend(q, k, v, mask, causal):
        return reference_attention(q, k, v, mask, causal=causal, logits_dtype=runtime.logits_dtype)

    if runtime.backend == "reference":
        return _reference_backend
    if runtime.backend == "splash":
        backend = _maybe_splash_attention()
        if backend is None:
            raise RuntimeError("Splash attention requested but unavailable")
        return backend
    backend = _maybe_splash_attention() if runtime.backend == "auto" else None
    if backend is not None:
        return backend
    return _reference_backend


__all__ = [
    "AttentionBackend",
    "apply_rotary_embedding",
    "default_attention_mask",
    "reference_attention",
    "resolve_attention_backend",
]
