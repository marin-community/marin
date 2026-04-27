# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Fused linear softmax cross-entropy for grug.

This wraps the shared fused kernel API for TPU and falls back to a full-logits
reference implementation on non-TPU backends.
"""

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from haliax.jax_utils import named_call
from levanter.kernels.pallas.fused_cross_entropy_loss import (
    fused_cross_entropy_loss_and_logsumexp_penalty,
)


def _batch_axis_spec(x: jax.Array):
    x_type = jax.typeof(x)
    sharding = getattr(x_type, "sharding", None)
    spec = getattr(sharding, "spec", None)
    if spec is not None and len(spec) > 0:
        return spec[0]
    sharding = getattr(x, "sharding", None)
    spec = getattr(sharding, "spec", None)
    if spec is not None and len(spec) > 0:
        return spec[0]
    return ("data",)


def _axis_names_from_spec(axis_spec) -> tuple[str, ...]:
    if axis_spec is None:
        return ()
    if isinstance(axis_spec, tuple):
        return tuple(str(name) for name in axis_spec)
    return (str(axis_spec),)


def _psum_over_axes(x: jax.Array, axis_names: tuple[str, ...]) -> jax.Array:
    if len(axis_names) == 0:
        return x
    if len(axis_names) == 1:
        return jax.lax.psum(x, axis_names[0])
    return jax.lax.psum(x, axis_names)


@named_call
def fused_linear_softmax_cross_entropy_loss(
    hidden: jax.Array,
    lm_head: jax.Array,
    labels: jax.Array,
    *,
    weight: jax.Array | None = None,
    reduction: str = "mean",
    logsumexp_weight: float | None = None,
    logit_soft_cap: float | None = None,
    dtype: jnp.dtype = jnp.float32,
    precision: jax.lax.PrecisionLike = None,
    implementation: str | tuple[str, ...] | None = None,
) -> jax.Array:
    """Compute cross-entropy loss via the fused kernel path.

    Args:
        hidden: Array with shape (..., hidden_dim).
        lm_head: Array with shape (hidden_dim, vocab_size).
        labels: Integer array with shape (...,).
        weight: Optional per-example weights with shape matching labels.
        reduction: One of {"mean", "sum", "none"}.
        logsumexp_weight: Optional z-loss weight (logsumexp^2 term).
        logit_soft_cap: Optional Gemma2-style tanh soft cap on logits:
            ``logits := c * tanh(logits / c)``. Applied inside the fused kernel.
        dtype: Accumulator dtype for logits/logsumexp.
        precision: Optional matmul precision override for XLA/reference paths.
        implementation: Optional fused CE backend selection override.

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

    weight_array = weight if weight is not None else jnp.ones_like(labels, dtype=dtype)
    batch_axis_spec = _batch_axis_spec(hidden)
    batch_axis_names = _axis_names_from_spec(batch_axis_spec)

    def _loss_shard(
        shard_hidden: jax.Array,
        shard_lm_head: jax.Array,
        shard_labels: jax.Array,
        shard_weight: jax.Array,
    ) -> jax.Array:
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
            dtype=dtype,
            logit_soft_cap=logit_soft_cap,
            precision=precision,
            implementation=implementation,
        )

        if reduction_mode is None:
            return loss.reshape(shard_labels.shape)

        local_sum = jnp.sum(loss)
        local_denom = jnp.sum(flat_weight)
        total_sum = _psum_over_axes(local_sum, batch_axis_names)
        if reduction_mode == "sum":
            return total_sum
        total_denom = _psum_over_axes(local_denom, batch_axis_names)
        return jnp.where(total_denom != 0, total_sum / total_denom, jnp.zeros_like(total_denom))

    out_specs = P(batch_axis_spec) if reduction_mode is None else P()
    return jax.shard_map(
        _loss_shard,
        in_specs=(P(batch_axis_spec), P(None, None), P(batch_axis_spec), P(batch_axis_spec)),
        out_specs=out_specs,
        check_vma=False,
    )(hidden, lm_head, labels, weight_array)


__all__ = [
    "fused_linear_softmax_cross_entropy_loss",
]
