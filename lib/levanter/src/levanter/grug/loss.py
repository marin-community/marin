# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Fused linear softmax cross-entropy for grug.

This wraps the shared fused kernel API for TPU and falls back to a full-logits
reference implementation on non-TPU backends.
"""

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from levanter.kernels.pallas.fused_cross_entropy_loss import (
    BlockSizes,
    fused_cross_entropy_loss_and_logsumexp_penalty,
)


def fused_linear_softmax_cross_entropy_loss(
    hidden: jax.Array,
    lm_head: jax.Array,
    labels: jax.Array,
    *,
    weight: jax.Array | None = None,
    reduction: str = "mean",
    logsumexp_weight: float | None = None,
    block_size: int | None = None,
    dtype: jnp.dtype = jnp.float32,
    precision: jax.lax.PrecisionLike = None,
) -> jax.Array:
    """Compute cross-entropy loss via the fused kernel path.

    Args:
        hidden: Array with shape (..., hidden_dim).
        lm_head: Array with shape (hidden_dim, vocab_size).
        labels: Integer array with shape (...,).
        weight: Optional per-example weights with shape matching labels.
        reduction: One of {"mean", "sum", "none"}.
        logsumexp_weight: Optional z-loss weight (logsumexp^2 term).
        block_size: Optional vocab block size (used as v_block_size).
        dtype: Accumulator dtype for logits/logsumexp.
        precision: Optional matmul precision override for XLA/reference paths.

    Returns:
        If reduction=="none": array with shape labels.shape.
        Else: scalar array.
    """
    if lm_head.ndim != 2:
        raise ValueError(f"lm_head must be 2D (hidden_dim, vocab), got shape={lm_head.shape}")
    hidden_dim = hidden.shape[-1]
    if lm_head.shape[0] != hidden_dim:
        raise ValueError(f"hidden_dim mismatch: hidden={hidden_dim}, lm_head={lm_head.shape[0]}")

    reduction_mode: str | None
    if reduction == "none":
        reduction_mode = None
    elif reduction in ("sum", "mean"):
        reduction_mode = reduction
    else:
        raise ValueError(f"Unknown reduction: {reduction}")

    block_sizes = BlockSizes(v_block_size=block_size) if block_size is not None else None
    weight_array = weight if weight is not None else jnp.ones_like(labels, dtype=dtype)

    def _loss_shard(
        shard_hidden: jax.Array,
        shard_lm_head: jax.Array,
        shard_labels: jax.Array,
        shard_weight: jax.Array,
    ) -> jax.Array:
        block_sizes_local = block_sizes if jax.default_backend() == "tpu" else None
        flat_hidden = shard_hidden.reshape((-1, hidden_dim))
        flat_labels = shard_labels.reshape((-1,)).astype(jnp.int32)
        flat_weight = shard_weight.reshape((-1,))

        loss = fused_cross_entropy_loss_and_logsumexp_penalty(
            flat_hidden,
            flat_labels,
            shard_lm_head,
            reduction=None,
            weight=flat_weight,
            logsumexp_weight=logsumexp_weight,
            block_sizes=block_sizes_local,
            dtype=dtype,
            logit_soft_cap=None,
            precision=precision,
        )

        if reduction_mode is None:
            return loss.reshape(shard_labels.shape)

        local_sum = jnp.sum(loss)
        local_denom = jnp.sum(flat_weight)
        total_sum = jax.lax.psum(local_sum, "data")
        if reduction_mode == "sum":
            return total_sum
        total_denom = jax.lax.psum(local_denom, "data")
        return jnp.where(total_denom != 0, total_sum / total_denom, jnp.zeros_like(total_denom))

    out_specs = P(("data",)) if reduction_mode is None else P()
    return jax.shard_map(
        _loss_shard,
        in_specs=(P(("data",)), P(None, None), P(("data",)), P(("data",))),
        out_specs=out_specs,
    )(hidden, lm_head, labels, weight_array)


__all__ = [
    "fused_linear_softmax_cross_entropy_loss",
]
