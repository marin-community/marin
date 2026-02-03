# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, cast

import jax
import jax.numpy as jnp

import haliax as hax
from haliax import NamedArray
from haliax.core import flatten_all_axes_but
from haliax.nn import cross_entropy_loss_and_log_normalizers
from haliax.partitioning import pspec_for_axis, shard_map
from levanter.kernels.pallas.fused_cross_entropy_loss import (
    fused_cross_entropy_loss_and_logsumexp_penalty as fused_cross_entropy_loss_and_logsumexp_penalty_kernel,
)

DEFAULT_REDUCTION = cast(hax.ReductionFunction, hax.mean)


def maybe_fused_next_token_loss(
    Pos: hax.AxisSelector,
    Embed: hax.AxisSelector,
    Vocab: hax.AxisSelector,
    pred_embeddings: NamedArray,
    pred_lm_head: NamedArray,
    true_ids: NamedArray,
    loss_weight: Optional[NamedArray] = None,
    reduction: Optional[hax.ReductionFunction] = DEFAULT_REDUCTION,
    reduction_axis: Optional[hax.AxisSelection] = None,
    logsumexp_weight: Optional[float] = None,
    block_size: Optional[int] = None,
    dtype: Optional[jnp.dtype] = jnp.float32,
    logit_soft_cap: Optional[float] = None,
    precision: jax.lax.PrecisionLike = None,
    implementation: str | None = None,
) -> NamedArray:
    """
    Compute the next token loss with optional block-wise processing.

    Args:
        Pos (hax.AxisSelector): Position axis selector.
        Vocab (hax.AxisSelector): Vocabulary axis selector.
        pred_embeddings (NamedArray): Predicted embeddings.
        pred_lm_head (NamedArray): Language model head weights.
        true_ids (NamedArray): True token IDs.
        loss_weight (Optional[NamedArray]): Mask to apply to the loss.
        reduction (Optional[hax.ReductionFunction]): Reduction function.
        reduction_axis (Optional[hax.AxisSelection]): Axis to apply reduction.
        logsumexp_weight (Optional[float]): Weight for logsumexp penalty.
        block_size (Optional[int]): Size of each block for processing.
        dtype (Optional[jnp.dtype]): Data type for the loss.
        logit_soft_cap (Optional[float]): Optional soft cap for logits
        precision (Optional[jax.lax.PrecisionLike]): Optional matmul precision for full-logits path.
    Returns:
        NamedArray: Computed loss.
    """
    # Resolve axes
    Pos = pred_embeddings.resolve_axis(Pos.name)
    Vocab = pred_lm_head.resolve_axis(Vocab)

    if block_size is None:
        # Full softmax computation
        logits = hax.dot(pred_embeddings, pred_lm_head, axis=Embed, precision=precision)
        if dtype is not None:
            logits = logits.astype(dtype)

        if logit_soft_cap is not None:
            logits = hax.tanh(logits / logit_soft_cap) * logit_soft_cap

        # Shift target tokens to predict the next token
        return next_token_loss(Pos, Vocab, logits, true_ids, loss_weight, reduction, reduction_axis, logsumexp_weight)

    # Shift target tokens to predict the next token
    target_y = hax.roll(true_ids, -1, Pos)

    # Create a mask that excludes the last token
    not_last_mask = hax.logical_not(hax.nn.one_hot(-1, Pos, dtype=jnp.bool_))  # type: ignore
    if loss_weight is not None:
        weight_dtype = dtype if dtype is not None else pred_embeddings.dtype
        loss_weight = loss_weight.astype(weight_dtype) * not_last_mask.astype(weight_dtype)
    else:
        weight_dtype = dtype if dtype is not None else pred_embeddings.dtype
        loss_weight = not_last_mask.astype(weight_dtype)

    # Compute the loss with optional block-wise processing
    return fused_cross_entropy_loss_and_logsumexp_penalty(
        pred_embeddings,
        pred_lm_head,
        Contract=Embed,
        Label=Vocab,
        target_y=target_y,
        reduction=reduction,
        reduction_axis=reduction_axis,
        weight=loss_weight,
        logsumexp_weight=logsumexp_weight,
        block_size=block_size,
        dtype=dtype,
        logit_soft_cap=logit_soft_cap,
        precision=precision,
        implementation=implementation,
    )


def next_token_loss(
    Pos: hax.AxisSelector,
    Vocab: hax.AxisSelector,
    logits: NamedArray,
    true_ids: NamedArray,
    loss_weight: Optional[NamedArray] = None,
    reduction: Optional[hax.ReductionFunction] = DEFAULT_REDUCTION,
    reduction_axis: Optional[hax.AxisSelection] = None,
    logsumexp_weight: Optional[float] = None,
):
    """
    Compute the next token loss with optional logsumexp penalty.

    Args:
        Pos: axis selector for the position axis
        Vocab: axis selector for the vocabulary axis
        logits: predicted logits
        true_ids: true token IDs (not shifted)
        loss_weight: mask to apply to the loss
        reduction: reduction function or None to disable reduction
        reduction_axis: axis to apply reduction. None means all axes
        logsumexp_weight: weight for the logsumexp penalty
        logit_soft_cap: optional soft cap for logits
    Returns:
        NamedArray: computed loss
    """
    Pos = logits.resolve_axis(hax.axis_name(Pos))

    target_y = hax.roll(true_ids, -1, Pos)
    target_y_full = hax.nn.one_hot(target_y, Vocab, dtype=logits.dtype)

    # Create a mask that excludes the last token
    not_last_mask = hax.logical_not(hax.nn.one_hot(-1, Pos, dtype=jnp.bool_))
    if loss_weight is not None:
        weight_dtype = logits.dtype
        loss_weight = loss_weight.astype(weight_dtype) * not_last_mask.astype(weight_dtype)
    else:
        loss_weight = not_last_mask.astype(logits.dtype)

    return cross_entropy_and_logsumexp_penalty(
        Vocab=Vocab,
        pred_y=logits,
        target_y=target_y_full,
        reduction=reduction,
        reduction_axis=reduction_axis,
        weight=loss_weight,
        logsumexp_weight=logsumexp_weight,
    )


def cross_entropy_and_logsumexp_penalty(
    Vocab: hax.Axis,
    pred_y: NamedArray,
    target_y: NamedArray,
    *,
    reduction: Optional[hax.ReductionFunction] = DEFAULT_REDUCTION,
    reduction_axis: Optional[hax.AxisSelection] = None,
    weight: Optional[NamedArray] = None,
    logsumexp_weight=0.0,
) -> NamedArray:
    """A loss function that combines cross entropy loss with a logsumexp penalty."""

    loss, log_normalizers = cross_entropy_loss_and_log_normalizers(pred_y, Vocab, target_y)

    if logsumexp_weight is not None and logsumexp_weight != 0.0:
        loss = loss + logsumexp_weight * (log_normalizers**2)

    return hax.nn.loss.reduce_loss(loss, reduction, reduction_axis, weight=weight)


def fused_cross_entropy_loss_and_logsumexp_penalty(
    pred_embeddings: NamedArray,
    pred_lm_head: NamedArray,
    Contract: hax.AxisSelector,
    Label: hax.AxisSelector,
    target_y: NamedArray,
    *,
    reduction: Optional[hax.ReductionFunction] = DEFAULT_REDUCTION,
    reduction_axis: Optional[hax.AxisSelection] = None,
    weight: Optional[NamedArray] = None,
    logsumexp_weight: float | None = 0.0,
    block_size: int,
    dtype: Optional[jnp.dtype] = jnp.float32,
    logit_soft_cap: Optional[float] = None,
    precision: jax.lax.PrecisionLike = None,
    implementation: str | None = None,
) -> NamedArray:
    """
    Compute the cross-entropy loss and logsumexp penalty using embeddings and lm_head,
    with optional block-wise processing.

    Args:
        pred_embeddings (NamedArray): Predicted embeddings.
        pred_lm_head (NamedArray): Language model head weights.
        Contract (hax.AxisSelector): Axis to contract over.
        Label (hax.AxisSelector): Label (Vocab) axis.
        target_y (NamedArray): Target token ids.
        reduction (Optional[hax.ReductionFunction]): Reduction function.
        reduction_axis (Optional[hax.AxisSelection]): Axis to apply reduction.
        weight (Optional[NamedArray]): Sample weights to apply to the loss.
        logsumexp_weight (float): Weight for logsumexp penalty.
        block_size (int): Size of each block for processing.
        dtype (Optional[jnp.dtype]): Data type for the loss.
        precision (Optional[jax.lax.PrecisionLike]): Optional matmul precision override for the fused kernel.

    Returns:
        NamedArray: Computed loss.
    """

    Contract = pred_embeddings.resolve_axis(Contract)
    Label = pred_lm_head.resolve_axis(Label)
    batch_axes = hax.axis.without_axes(pred_embeddings.axes, Contract)
    # IMPORTANT: keep the flattened batch axis named `token` so it picks up the standard axis mapping
    # (e.g. `token -> (replica_dcn, replica, data)`) and stays sharded in `shard_map` rather than being
    # replicated. Replicating `B = batch*seq` can lead to massive intermediates and TPU compile failures.
    flat_embeddings, _ = flatten_all_axes_but(pred_embeddings, "token", batch_axes, reorder_to_front=True)
    batch_axis = flat_embeddings.resolve_axis("token")
    flat_embeddings = flat_embeddings.rearrange((batch_axis, Contract))

    flat_labels = hax.flatten_axes(target_y, target_y.axes, batch_axis)

    lm_head = pred_lm_head.rearrange((Contract, Label))

    def fused_impl(shard_embeddings: NamedArray, shard_labels: NamedArray, shard_lm_head: NamedArray) -> jax.Array:
        return fused_cross_entropy_loss_and_logsumexp_penalty_kernel(
            shard_embeddings.array,
            shard_labels.array.astype(jnp.int32),
            shard_lm_head.array,
            reduction=None,
            weight=None,
            logsumexp_weight=logsumexp_weight,
            block_size=block_size,
            dtype=dtype,
            logit_soft_cap=logit_soft_cap,
            precision=precision,
            implementation=implementation,
        )

    in_specs = (
        pspec_for_axis(flat_embeddings.axes),
        pspec_for_axis(flat_labels.axes),
        pspec_for_axis(lm_head.axes),
    )
    loss_flat = shard_map(
        fused_impl,
        in_specs=in_specs,
        out_specs=pspec_for_axis((batch_axis,)),
        check_rep=False,
    )(flat_embeddings, flat_labels, lm_head)

    loss_named = hax.named(loss_flat, batch_axis).unflatten_axis(batch_axis, target_y.axes)

    return hax.nn.loss.maybe_reduce_loss(loss_named, reduction, reduction_axis, where=None, weight=weight)
