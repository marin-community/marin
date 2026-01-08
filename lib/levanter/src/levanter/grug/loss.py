# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Blockwise linear softmax cross-entropy for grug.

This is the "large vocab friendly" alternative to materializing full logits
`hidden @ lm_head` with shape (batch, seq, vocab).

Design notes:
  - Works on plain `jax.Array` inputs (grug core doesn't use NamedArray).
  - Computes `logsumexp` over vocab in blocks to reduce peak memory.
  - Computes the correct-class logit via gather+dot (O(N*H)), avoiding a full
    logits materialization.
"""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P, reshard


def linear_softmax_cross_entropy_loss_and_logz(
    hidden: jax.Array,
    lm_head: jax.Array,
    labels: jax.Array,
    *,
    block_size: int,
    dtype: jnp.dtype = jnp.float32,
    logit_soft_cap: float | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Compute per-position cross entropy without materializing full logits.

    Args:
        hidden: Array with shape (..., hidden_dim).
        lm_head: Array with shape (hidden_dim, vocab_size).
        labels: Integer array with shape (...,).
        block_size: Vocab block size for logsumexp.
        dtype: Accumulator dtype for logsumexp.
        logit_soft_cap: Optional tanh soft cap for logits (applied before exp).

    Returns:
        (loss, logz) each with shape labels.shape.
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    hidden_dim = hidden.shape[-1]
    if lm_head.ndim != 2:
        raise ValueError(f"lm_head must be 2D (hidden_dim, vocab), got shape={lm_head.shape}")
    if lm_head.shape[0] != hidden_dim:
        raise ValueError(f"hidden_dim mismatch: hidden={hidden_dim}, lm_head={lm_head.shape[0]}")

    vocab_size = lm_head.shape[1]
    nblocks = (vocab_size + block_size - 1) // block_size
    vocab_padded = nblocks * block_size
    if vocab_padded != vocab_size:
        # TODO: avoid materializing a padded `lm_head` by switching to a tokamax/pallas kernel (TPU)
        # or by clamping the final block slice + masking. For now, the overhead is bounded by
        # `hidden_dim * (block_size - 1)` elements.
        pad = jnp.zeros((hidden_dim, vocab_padded - vocab_size), dtype=lm_head.dtype)
        # If lm_head is sharded, concatenate requires the pad to have the same sharding.
        lm_head_sharding = getattr(lm_head, "sharding", None)
        if isinstance(lm_head_sharding, NamedSharding) and isinstance(lm_head_sharding.spec, P):
            pad = reshard(pad, lm_head_sharding.spec)
        lm_head = jnp.concatenate([lm_head, pad], axis=1)
    flat_hidden = hidden.reshape((-1, hidden_dim)).astype(dtype)
    flat_labels = labels.reshape((-1,)).astype(jnp.int32)

    # correct logits: dot(hidden, lm_head[:, label])
    # take along vocab axis (axis=1) -> (hidden_dim, N) then transpose to (N, hidden_dim)
    w_y = jnp.take(lm_head, flat_labels, axis=1).T.astype(dtype)
    hidden_sharding = getattr(flat_hidden, "sharding", None)
    if isinstance(hidden_sharding, NamedSharding) and isinstance(hidden_sharding.spec, P):
        w_y = reshard(w_y, hidden_sharding.spec)
    logit_y = jnp.sum(flat_hidden * w_y, axis=-1)

    neg_inf = jnp.array(-jnp.inf, dtype=dtype)
    # Match the sharding of the computed per-example logits so the loop carry types are stable.
    m0 = jnp.full_like(logit_y, neg_inf)
    s0 = jnp.zeros_like(logit_y, dtype=dtype)

    neg_inf_logits = jnp.array(-jnp.inf, dtype=dtype)

    def _body(i: int, state: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        m, s = state
        start = i * block_size
        w_block = jax.lax.dynamic_slice_in_dim(lm_head, start_index=start, slice_size=block_size, axis=1).astype(dtype)
        logits = flat_hidden @ w_block
        valid = jnp.arange(block_size) < (vocab_size - start)
        logits = jnp.where(valid[None, :], logits, neg_inf_logits)
        if logit_soft_cap is not None:
            logits = jnp.tanh(logits / logit_soft_cap) * logit_soft_cap
        block_max = jnp.max(logits, axis=-1)
        new_m = jnp.maximum(m, block_max)
        s = s * jnp.exp(m - new_m) + jnp.sum(jnp.exp(logits - new_m[:, None]), axis=-1)
        return new_m, s

    m, s = jax.lax.fori_loop(0, nblocks, _body, (m0, s0))
    logz = m + jnp.log(s)
    loss = logz - logit_y

    return loss.reshape(labels.shape), logz.reshape(labels.shape)


def next_token_linear_softmax_cross_entropy(
    token_ids: jax.Array,
    hidden: jax.Array,
    lm_head: jax.Array,
    *,
    block_size: int,
    reduction: Literal["mean", "sum", "none"] = "mean",
    dtype: jnp.dtype = jnp.float32,
    logsumexp_weight: float | None = None,
    logit_soft_cap: float | None = None,
) -> jax.Array:
    """Next-token loss using blockwise logits.

    This matches the common "predict token t+1 from hidden at t" loss.
    The last position is ignored.
    """
    if token_ids.ndim < 1:
        raise ValueError("token_ids must have at least 1 dimension")

    # Shift tokens left: label[t] = token_ids[t+1]
    labels = jnp.concatenate([token_ids[..., 1:], token_ids[..., :1] * 0], axis=-1)
    weight = jnp.concatenate(
        [
            jnp.ones(token_ids.shape[:-1] + (token_ids.shape[-1] - 1,), dtype=dtype),
            jnp.zeros(token_ids.shape[:-1] + (1,), dtype=dtype),
        ],
        axis=-1,
    )

    loss, logz = linear_softmax_cross_entropy_loss_and_logz(
        hidden,
        lm_head,
        labels,
        block_size=block_size,
        dtype=dtype,
        logit_soft_cap=logit_soft_cap,
    )
    loss = loss * weight

    if logsumexp_weight is not None and logsumexp_weight != 0.0:
        loss = loss + logsumexp_weight * (logz**2) * weight

    if reduction == "none":
        return loss
    if reduction == "sum":
        return jnp.sum(loss)
    if reduction == "mean":
        denom = jnp.sum(weight)
        return jnp.sum(loss) / jnp.maximum(denom, jnp.array(1.0, dtype=dtype))

    raise ValueError(f"Unknown reduction: {reduction}")


__all__ = [
    "linear_softmax_cross_entropy_loss_and_logz",
    "next_token_linear_softmax_cross_entropy",
]
