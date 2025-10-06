# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import functools
from typing import Optional

import equinox
import jax
import jax.numpy as jnp

import haliax as hax
from haliax import NamedArray
from haliax.nn import cross_entropy_loss_and_log_normalizers


def maybe_fused_next_token_loss(
    Pos: hax.AxisSelector,
    Embed: hax.AxisSelector,
    Vocab: hax.AxisSelector,
    pred_embeddings: NamedArray,
    pred_lm_head: NamedArray,
    true_ids: NamedArray,
    loss_mask: Optional[NamedArray] = None,
    reduction: Optional[hax.ReductionFunction] = hax.mean,
    reduction_axis: Optional[hax.AxisSelection] = None,
    logsumexp_weight: Optional[float] = None,
    block_size: Optional[int] = None,
    dtype: Optional[jnp.dtype] = jnp.float32,
    logit_soft_cap: Optional[float] = None,
) -> NamedArray:
    """
    Compute the next token loss with optional block-wise processing.

    Args:
        Pos (hax.AxisSelector): Position axis selector.
        Vocab (hax.AxisSelector): Vocabulary axis selector.
        pred_embeddings (NamedArray): Predicted embeddings.
        pred_lm_head (NamedArray): Language model head weights.
        true_ids (NamedArray): True token IDs.
        loss_mask (Optional[NamedArray]): Mask to apply to the loss.
        reduction (Optional[hax.ReductionFunction]): Reduction function.
        reduction_axis (Optional[hax.AxisSelection]): Axis to apply reduction.
        logsumexp_weight (Optional[float]): Weight for logsumexp penalty.
        block_size (Optional[int]): Size of each block for processing.
        dtype (Optional[jnp.dtype]): Data type for the loss.
        logit_soft_cap (Optional[float]): Optional soft cap for logits
    Returns:
        NamedArray: Computed loss.
    """
    # Resolve axes
    Pos = pred_embeddings.resolve_axis(Pos.name)
    Vocab = pred_lm_head.resolve_axis(Vocab)

    if block_size is None:
        # Full softmax computation
        logits = hax.dot(pred_embeddings, pred_lm_head, axis=Embed)
        if dtype is not None:
            logits = logits.astype(dtype)

        if logit_soft_cap is not None:
            logits = hax.tanh(logits / logit_soft_cap) * logit_soft_cap

        # Shift target tokens to predict the next token
        return next_token_loss(Pos, Vocab, logits, true_ids, loss_mask, reduction, reduction_axis, logsumexp_weight)

    # Shift target tokens to predict the next token
    target_y = hax.roll(true_ids, -1, Pos)

    # Create a mask that excludes the last token
    not_last_loss_mask = 1 - hax.nn.one_hot(-1, Pos, dtype=jnp.float32)  # type: ignore
    if loss_mask is not None:
        loss_mask = loss_mask * not_last_loss_mask
    else:
        loss_mask = not_last_loss_mask

    # Compute the loss with optional block-wise processing
    return fused_cross_entropy_loss_and_logsumexp_penalty(
        pred_embeddings,
        pred_lm_head,
        Contract=Embed,
        Label=Vocab,
        target_y=target_y,
        reduction=reduction,
        reduction_axis=reduction_axis,
        where=loss_mask,
        logsumexp_weight=logsumexp_weight,
        block_size=block_size,
        dtype=dtype,
        logit_soft_cap=logit_soft_cap,
    )


def next_token_loss(
    Pos: hax.AxisSelector,
    Vocab: hax.AxisSelector,
    logits: NamedArray,
    true_ids: NamedArray,
    loss_mask: Optional[NamedArray] = None,
    reduction: Optional[hax.ReductionFunction] = hax.mean,
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
        loss_mask: mask to apply to the loss
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
    not_last_loss_mask = 1 - hax.nn.one_hot(-1, Pos, dtype=jnp.float32)  # type: ignore
    if loss_mask is not None:
        loss_mask = loss_mask * not_last_loss_mask
    else:
        loss_mask = not_last_loss_mask

    return cross_entropy_and_logsumexp_penalty(
        Vocab=Vocab,
        pred_y=logits,
        target_y=target_y_full,
        reduction=reduction,
        reduction_axis=reduction_axis,
        where=loss_mask,
        logsumexp_weight=logsumexp_weight,
    )


def cross_entropy_and_logsumexp_penalty(
    Vocab: hax.Axis,
    pred_y: NamedArray,
    target_y: NamedArray,
    *,
    reduction: Optional[hax.ReductionFunction] = hax.mean,
    reduction_axis: Optional[hax.AxisSelection] = None,
    where: Optional[NamedArray] = None,
    logsumexp_weight=0.0,
) -> NamedArray:
    """A loss function that combines cross entropy loss with a logsumexp penalty."""

    loss, log_normalizers = cross_entropy_loss_and_log_normalizers(pred_y, Vocab, target_y)

    if logsumexp_weight is not None and logsumexp_weight != 0.0:
        loss = loss + logsumexp_weight * (log_normalizers**2)

    return hax.nn.loss.maybe_reduce_loss(loss, reduction, reduction_axis, where)


def fused_cross_entropy_loss_and_logsumexp_penalty(
    pred_embeddings: NamedArray,
    pred_lm_head: NamedArray,
    Contract: hax.AxisSelector,
    Label: hax.AxisSelector,
    target_y: NamedArray,
    *,
    reduction: Optional[hax.ReductionFunction] = hax.mean,
    reduction_axis: Optional[hax.AxisSelection] = None,
    where: Optional[NamedArray] = None,
    logsumexp_weight: float | None = 0.0,
    block_size: int,
    dtype: Optional[jnp.dtype] = jnp.float32,
    logit_soft_cap: Optional[float] = None,
) -> NamedArray:
    """
    Compute the cross-entropy loss and logsumexp penalty using embeddings and lm_head,
    with optional block-wise processing.

    Args:
        pred_embeddings (NamedArray): Predicted embeddings.
        pred_lm_head (NamedArray): Language model head weights.
        Contract (hax.AxisSelector): Axis to contract over.
        Label (hax.AxisSelector): Label (Vocab) axis.
        target_y (NamedArray): One-hot encoded target tokens.
        reduction (Optional[hax.ReductionFunction]): Reduction function.
        reduction_axis (Optional[hax.AxisSelection]): Axis to apply reduction.
        where (Optional[NamedArray]): Mask to apply to the loss.
        logsumexp_weight (float): Weight for logsumexp penalty.
        block_size (int): Size of each block for processing.
        dtype (Optional[jnp.dtype]): Data type for the loss.

    Returns:
        NamedArray: Computed loss.
    """

    # Block-wise softmax computation
    loss, log_normalizers = _blockwise_cross_entropy_loss(
        (pred_embeddings, pred_lm_head),
        Contract,
        Label,
        target_y,
        block_size,
        dtype=dtype,
        logit_soft_cap=logit_soft_cap,
    )

    if logsumexp_weight is not None and (not isinstance(logsumexp_weight, (int, float)) or logsumexp_weight != 0.0):
        loss = loss + logsumexp_weight * (log_normalizers**2)

    return hax.nn.loss.maybe_reduce_loss(loss, reduction, reduction_axis, where)


@equinox.filter_custom_vjp
def _blockwise_cross_entropy_loss(
    # pred_embeddings: NamedArray,
    # pred_lm_head: NamedArray,
    pred: tuple[NamedArray, NamedArray],
    Contract: hax.Axis,
    Label: hax.Axis,
    labels_y: NamedArray,
    block_size: int,
    dtype: Optional[jnp.dtype],
    logit_soft_cap: Optional[float] = None,
) -> tuple[NamedArray, NamedArray]:
    """
    Compute cross-entropy loss and log normalizers in a block-wise manner without materializing the full logits.

    Args:
        pred_embeddings (NamedArray): Predicted embeddings.
        pred_lm_head (NamedArray): Language model head weights.
        Contract (hax.Axis): Axis to contract over.
        Label (hax.AxisSelector): Label (Vocab) axis.
        labels_y (NamedArray): label tensor.
        block_size (int): Size of each block for processing.
        dtype (Optional[jnp.dtype]): Data type for the loss.

    Notes:
        labels_y being anything other than the label tensor would remove any benefits

        TODO: but if XLA smart enough to optimize it out?

    Returns:
        tuple[NamedArray, NamedArray]: tuple of loss and log_normalizers.
    """

    return _block_cross_entropy_forward(None, pred, Contract, Label, labels_y, block_size, dtype, logit_soft_cap)[0]


def _block_cross_entropy_forward(
    ignore,
    pred: tuple[NamedArray, NamedArray],
    Contract: hax.Axis,
    Label: hax.Axis,
    labels_y: NamedArray,
    block_size: int,
    dtype: Optional[jnp.dtype],
    logit_soft_cap: Optional[float] = None,
) -> tuple[tuple[NamedArray, NamedArray], tuple[NamedArray]]:
    """
    Forward pass for block-wise cross-entropy loss.

    This function computes the cross-entropy loss and log-sum-exp (`log_z`) in a block-wise manner
    to maintain memory efficiency by processing subsets of the vocabulary at a time.

    Args:
        ignore: Placeholder argument (unused).
        pred (Tuple[NamedArray, NamedArray]): Tuple containing predicted embeddings and language model head weights.
        Contract (hax.Axis): Axis to contract over (e.g., embedding axis).
        Label (hax.Axis): Label axis (e.g., vocabulary axis).
        labels_y (NamedArray): True target labels [Batch, Seq].
        block_size (int): Number of vocabulary tokens per block.
        dtype (Optional[jnp.dtype]): Data type for the computations.

    Returns:
        Tuple:
            - Tuple[NamedArray, NamedArray]: Computed loss and logsumexp.
            - Tuple[NamedArray]: Residuals needed for the backward pass.
    """
    vocab_size = Label.size

    pred_embeddings, pred_lm_head = pred

    #
    # if num_blocks == 1:
    #     # No need for block-wise processing
    #     logits = hax.dot(pred_embeddings, pred_lm_head, axis=Contract)
    #     labels_y = hax.nn.one_hot(labels_y, Label, dtype=pred_embeddings.dtype)
    #     return cross_entropy_loss_and_log_normalizers(logits, Label, labels_y)
    #
    # ensure block size divides vocab size
    if vocab_size % block_size != 0:
        has_stragglers = True
    else:
        has_stragglers = False

    num_blocks = vocab_size // block_size

    # Initialize accumulators: loss, logsumexp, max_logits
    initial_O = hax.zeros(labels_y.axes)
    initial_logsumexp = hax.full(labels_y.axes, -jnp.inf)
    initial_max = hax.full(labels_y.axes, -jnp.inf)
    # We don't need this b/c we're using one-hot targets
    # initial_sumV = hax.full(labels_y.axes, 0.0)

    def process_block(block_idx, acc, current_block_size):
        """
        Process a single block of the Vocab dimension.

        Args:
            block_idx (int): Index of the current block.
            acc (tuple[NamedArray, NamedArray, jnp.ndarray]): Accumulators for loss, logsumexp, and max logits.
            current_block_size (int): Size of the current block (used for stragglers).

        Returns:
            tuple[NamedArray, NamedArray, jnp.ndarray]: Updated accumulators
        """
        loss, logsumexp_prev, max_logit_prev = acc

        start = block_idx * block_size
        Block = Label.resize(current_block_size)

        # Materialize the logits for the current block
        lm_head_b = pred_lm_head[Label, hax.dslice(start, Block)]  # [Contract, Block]
        logits_b = hax.dot(pred_embeddings, lm_head_b, axis=Contract)  # [Batch, Seq, Block]

        if dtype is not None:
            logits_b = logits_b.astype(dtype)

        if logit_soft_cap is not None:
            logits_b = hax.tanh(logits_b / logit_soft_cap) * logit_soft_cap

        # Update max and logsumexp
        max_logit = hax.maximum(max_logit_prev, hax.max(logits_b, axis=Block))  # [Batch, Seq]
        # reweight the previous logsumexp by the new max, fold in the new logits' contribution
        logsumexp = max_logit + hax.log(
            hax.exp(logsumexp_prev - max_logit) + hax.sum(hax.exp(logits_b - max_logit), axis=Block)
        )  # [Batch, Seq]

        # Materialize the target for the current block (one-hot)
        target_y_b = _block_one_hot(Block, start, labels_y, logits_b.dtype)  # [Batch, Seq, Block]

        # Update sumV. This is actually unnecessary if we're using one-hot targets
        # sV = sV_prev + hax.sum(target_y_b, axis=Label.name)

        loss += hax.dot(logits_b, target_y_b, axis=Block)  # [Batch, Seq]

        return loss, logsumexp, max_logit  # , sV

    if num_blocks == 0:
        o = initial_O
        log_z = initial_logsumexp
        max_logits = initial_max
    elif num_blocks == 1:
        o, log_z, max_logits = process_block(0, (initial_O, initial_logsumexp, initial_max), vocab_size)
    else:
        (o, log_z, max_logits) = jax.lax.fori_loop(
            lower=0,
            upper=num_blocks,
            body_fun=functools.partial(process_block, current_block_size=block_size),
            init_val=(initial_O, initial_logsumexp, initial_max),  # , initial_sumV
        )

    if has_stragglers:
        # Handle the stragglers
        remainder_size = vocab_size - num_blocks * block_size
        o, log_z, _ = process_block(num_blocks, (o, log_z, max_logits), remainder_size)

    # unnecessary if we're using one-hot targets
    # logz_outer = hax.einsum("->...", log_z, sum_v)
    o = log_z - o

    return (o, log_z), (log_z,)


def _block_cross_entropy_backward(
    residuals: tuple[NamedArray,],
    grad_in: tuple[NamedArray, NamedArray],
    ignore,
    pred: tuple[NamedArray, NamedArray],
    Contract: hax.Axis,
    Label: hax.Axis,
    labels_y: NamedArray,
    block_size: int,
    dtype: Optional[jnp.dtype],
    logit_soft_cap: Optional[float] = None,
) -> tuple[NamedArray, NamedArray]:
    """
    Compute the gradients of the block-wise cross-entropy loss.

    Args:
        residuals (tuple[NamedArray, NamedArray]): Residuals from the forward pass.
        grad_in (tuple[NamedArray, NamedArray]): Incoming gradients.
        pred (tuple[NamedArray, NamedArray]): Predictions.
        Contract (hax.Axis): Axis to contract over.
        Label (hax.Axis): Label axis.
        labels_y (NamedArray): Target labels.
        block_size (int): Size of each block.
        dtype (Optional[jnp.dtype]): Data type for the loss.
        logit_soft_cap (Optional[float]): Optional soft cap for logits.

    Returns:
        tuple[NamedArray, NamedArray]: Gradients.
    """

    (log_z,) = residuals
    grad_loss, grad_log_z = grad_in

    vocab_size = Label.size

    pred_embeddings, pred_lm_head = pred

    if vocab_size % block_size != 0:
        has_stragglers = True
    else:
        has_stragglers = False

    num_blocks = vocab_size // block_size

    grad_embeddings = hax.zeros(pred_embeddings.axes, dtype=pred_embeddings.dtype)
    grad_lm_head = hax.zeros(pred_lm_head.axes, dtype=pred_lm_head.dtype)

    def process_block(block_idx, acc, current_block_size):
        """
        Process a single block of the Vocab dimension.

        Args:
            block_idx (int): Index of the current block.
            acc (tuple[NamedArray, NamedArray]): Accumulators for gradients.
            current_block_size (int): Size of the current block (used for stragglers).

        Returns:
            tuple[NamedArray, NamedArray]: Updated accumulators.
        """
        grad_embeddings_prev, grad_lm_head_prev = acc

        start = block_idx * block_size
        Block = Label.resize(current_block_size)

        # Materialize the logits for the current block
        lm_head_b = pred_lm_head[Label, hax.dslice(start, Block)]  # [Contract, Block]
        logits_b = hax.dot(pred_embeddings, lm_head_b, axis=Contract)  # [Batch, Seq, Block]

        # Materialize the target for the current block (one-hot)
        target_y_block = _block_one_hot(Block, start, labels_y, logits_b.dtype)  # [Batch, Seq, Block]

        # materialize the softmax for the current block
        if dtype is not None:
            logits_b = logits_b.astype(dtype)

        if logit_soft_cap is not None:
            logits_b = hax.tanh(logits_b / logit_soft_cap) * logit_soft_cap

        p_b = hax.exp(logits_b - log_z)  # [Batch, Seq, Block]

        delta_b = p_b - target_y_block

        #  # dLoss/dL = g_loss * delta_b + g_log_z * probs_b
        #         # = g_loss * (probs_b - Y) + g_log_z * probs_b
        #         # = (g_loss + g_log_z) * probs_b - g_loss * Y

        # Compute gradients. We get None if the gradient is not provided.
        if grad_loss.array is not None:
            dLoss = grad_loss * delta_b  # [Batch, Seq, Block]
        else:
            dLoss = 0.0

        # Add the gradient of the logsumexp term (should be None if not provided)
        if grad_log_z.array is not None:
            dLoss += grad_log_z * p_b  # [Batch, Seq, Block]

        # Compute gradients for the current block
        # embeddings has shape [Batch, Seq, Embed], so we need to eliminate Block
        g_embeddings_b = hax.dot(
            dLoss, lm_head_b, axis=Block, preferred_element_type=grad_embeddings.dtype
        )  # [Batch, Seq, Embed]

        # lm_head has shape [Block, Embed], so we need to eliminate Batch, Seq, etc.
        eliminated_axes_W = hax.axis.without_axes(pred_embeddings.axes, lm_head_b.axes)
        g_lm_head_b = hax.dot(
            dLoss, pred_embeddings, axis=eliminated_axes_W, preferred_element_type=grad_lm_head_prev.dtype
        )  # [Block, Embed]

        g_lm_head = grad_lm_head_prev.at[Label, hax.dslice(start, Block)].set(g_lm_head_b)
        g_embeddings = grad_embeddings_prev + g_embeddings_b

        return g_embeddings, g_lm_head

    if num_blocks == 0:
        pass
    elif num_blocks == 1:
        grad_embeddings, grad_lm_head = process_block(0, (grad_embeddings, grad_lm_head), vocab_size)
    else:
        grad_embeddings, grad_lm_head = jax.lax.fori_loop(
            lower=0,
            upper=num_blocks,
            body_fun=functools.partial(process_block, current_block_size=block_size),
            init_val=(grad_embeddings, grad_lm_head),
        )

    if has_stragglers:
        # Handle the stragglers
        remainder_size = vocab_size - num_blocks * block_size
        grad_embeddings, grad_lm_head = process_block(num_blocks, (grad_embeddings, grad_lm_head), remainder_size)

    return grad_embeddings.astype(pred_embeddings.dtype), grad_lm_head.astype(pred_lm_head.dtype)


_blockwise_cross_entropy_loss.def_fwd(_block_cross_entropy_forward)
_blockwise_cross_entropy_loss.def_bwd(_block_cross_entropy_backward)


def _block_one_hot(LBlock, block_start, labels, dtype):
    end = block_start + LBlock.size
    target_is_in_this_block = hax.logical_and(labels >= block_start, labels < end)
    target_y_block = hax.nn.one_hot(labels - block_start, LBlock, dtype=dtype)
    # 0 out the logits that are not in this block
    target_y_block *= target_is_in_this_block
    return target_y_block
