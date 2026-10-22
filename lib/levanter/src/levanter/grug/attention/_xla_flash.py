# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0
"""Grug's ``xla_flash`` attention: Levanter's pure-JAX flash attention over Grug's raw arrays and mask.

No Pallas, CUDA, or TPU kernel dependency, so it runs on any XLA backend (including AMD GPUs and CPUs).
"""
from collections.abc import Callable

import haliax as hax
import jax
from jax import numpy as jnp
from jax.sharding import auto_axes
from jaxtyping import Array, Bool, Float

from levanter.grug.attention._core import AttentionMask, align_kv_heads, reference_attention
from levanter.kernels.pallas.autotune_utils import named_sharding_of
from levanter.layers.attention_mask import AttentionMask as LevanterAttentionMask
from levanter.layers.flash_attention import flash_attention

# Largest key/query block the flash attention may use, and the smallest worth blocking over.
XLA_FLASH_MAX_BLOCK_SIZE = 1024
XLA_FLASH_MIN_BLOCK_SIZE = 16


def _flash_block_size(q_len: int, k_len: int, *, requested: int | None) -> int | None:
    """A power-of-two block that divides both lengths, or None when the reference path should run instead.

    Levanter's flash attention requires its block size to divide both sequence lengths. A requested size
    is used as given; otherwise the largest power of two dividing both lengths, capped at the default.
    """
    if requested is not None:
        return requested
    block = XLA_FLASH_MAX_BLOCK_SIZE
    while block >= XLA_FLASH_MIN_BLOCK_SIZE and (q_len % block or k_len % block):
        block //= 2
    if block < XLA_FLASH_MIN_BLOCK_SIZE or q_len <= block or k_len <= block:
        return None
    return block


def _levanter_mask(
    mask: AttentionMask | Bool[Array, "B Q K"] | Float[Array, "B Q K"] | None,
    Batch: hax.Axis,
    QPos: hax.Axis,
    KPos: hax.Axis,
) -> LevanterAttentionMask | hax.NamedArray | None:
    """Translate a Grug mask into Levanter's mask type for ``levanter.layers.flash_attention``."""
    if mask is None:
        return None
    if isinstance(mask, jax.Array):
        if mask.dtype != jnp.bool_:
            raise NotImplementedError("xla_flash attention supports boolean dense masks only, not additive biases")
        if mask.ndim == 2 or (mask.ndim == 3 and mask.shape[0] == 1):
            return hax.named(mask.reshape(mask.shape[-2:]), (QPos, KPos))
        return hax.named(mask, (Batch, QPos, KPos))
    if mask.sliding_window is not None and not mask.is_causal:
        raise NotImplementedError("xla_flash attention supports a sliding window only together with causal masking")
    out = (
        LevanterAttentionMask.causal(sliding_window=mask.sliding_window) if mask.is_causal else LevanterAttentionMask()
    )
    if mask.segment_ids is not None:
        q_seg, k_seg = mask.segment_ids
        if q_seg.ndim == 2:
            # A size-1 batch of segment ids is shared across the real batch, as the reference path broadcasts it.
            q_seg = jnp.broadcast_to(q_seg, (Batch.size, q_seg.shape[1]))
            k_seg = jnp.broadcast_to(k_seg, (Batch.size, k_seg.shape[1]))
        q_axes = (QPos,) if q_seg.ndim == 1 else (Batch, QPos)
        k_axes = (KPos,) if k_seg.ndim == 1 else (Batch, KPos)
        out = out.with_segment_ids(hax.named(q_seg, q_axes), hax.named(k_seg, k_axes))
    return out


def _xla_flash_attention_math(
    q: Float[Array, "B Q Hq D"],
    k: Float[Array, "B K Hkv D"],
    v: Float[Array, "B K Hkv D"],
    mask: AttentionMask | Bool[Array, "B Q K"] | Float[Array, "B Q K"] | None,
    *,
    block_size: int | None,
) -> Float[Array, "B Q Hq D"]:
    batch, q_len, num_q_heads, head_dim = q.shape
    k = align_kv_heads(k, num_q_heads=num_q_heads)
    v = align_kv_heads(v, num_q_heads=num_q_heads)
    Batch = hax.Axis("batch", batch)
    QPos = hax.Axis("position", q_len)
    KPos = hax.Axis("key_position", k.shape[1])
    Heads = hax.Axis("heads", num_q_heads)
    Key = hax.Axis("head_dim", head_dim)
    out = flash_attention(
        QPos,
        KPos,
        Key,
        hax.named(q, (Batch, QPos, Heads, Key)),
        hax.named(k, (Batch, KPos, Heads, Key)),
        hax.named(v, (Batch, KPos, Heads, Key)),
        _levanter_mask(mask, Batch, QPos, KPos),
        inference=True,
        block_size=block_size,
    )
    return out.rearrange((Batch, QPos, Heads, Key)).array.astype(v.dtype)


def xla_flash_attention(
    q: Float[Array, "B Q Hq D"],
    k: Float[Array, "B K Hkv D"],
    v: Float[Array, "B K Hkv D"],
    mask: AttentionMask | Bool[Array, "B Q K"] | Float[Array, "B Q K"] | None,
    *,
    block_size: int | None = None,
) -> Float[Array, "B Q Hq D"]:
    """Levanter's pure-JAX flash attention (blockwise, custom backward) on Grug's raw arrays and mask.

    Runs on any backend. The output sharding follows ``q``, as for ``reference_attention``.
    """
    block_size = _flash_block_size(q.shape[1], k.shape[1], requested=block_size)
    if block_size is None:
        return reference_attention(q, k, v, mask, logits_dtype=jnp.float32)
    out_sharding = named_sharding_of(q)
    if out_sharding is None:
        return _xla_flash_attention_math(q, k, v, mask, block_size=block_size)
    # pyrefly: ignore[bad-assignment]  # auto_axes's decorator overload erases the wrapped signature
    wrapped: Callable[..., Float[Array, "B Q Hq D"]] = auto_axes(_xla_flash_attention_math, out_sharding=out_sharding)
    return wrapped(q, k, v, mask, block_size=block_size)
