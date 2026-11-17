"""Reduce token rows against data-weight counts for a complete optimizer step."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from skyrl_train.config.objective_spec import LossReduction

SPAN_THINK_TAG = 1


@dataclass(frozen=True)
class WeightCounts:
    tokens: float
    rows: float


@dataclass(frozen=True)
class StepCounts:
    policy: WeightCounts
    mask: WeightCounts
    teacher: WeightCounts
    nonzero_advantage_rows: float
    max_seq_len: int


def policy_data_weights(
    loss_mask: torch.Tensor, response_span_tags: torch.Tensor | None, think_token_weight: float
) -> torch.Tensor:
    """Return loss eligibility weighted by the configured THINK-token contribution."""
    if response_span_tags is None or think_token_weight == 1.0:
        return loss_mask
    # Tags and loss masks share response positions; span_tagger uses 1 for THINK.
    tags = torch.zeros_like(loss_mask)
    width = min(response_span_tags.shape[-1], loss_mask.shape[-1])
    tags[..., :width] = response_span_tags[..., :width]
    return loss_mask * torch.where(tags == SPAN_THINK_TAG, think_token_weight, 1.0)


@torch.no_grad()
def step_counts(
    policy_weights: Sequence[torch.Tensor],
    loss_masks: Sequence[torch.Tensor],
    teacher_weights: Sequence[torch.Tensor],
    advantages: Sequence[torch.Tensor],
    max_seq_len: int,
    all_reduce_sum: Callable[[torch.Tensor], torch.Tensor],
) -> StepCounts:
    """Sum data counts across the accumulation window and the data-parallel group."""
    device = policy_weights[0].device
    counts = torch.zeros(7, device=device, dtype=torch.float64)
    for offset, weights in ((0, policy_weights), (2, loss_masks), (4, teacher_weights)):
        for weight in weights:
            counts[offset] += weight.sum(dtype=torch.float64)
            counts[offset + 1] += (weight.sum(-1) > 0).sum()
    for weight, advantage in zip(policy_weights, advantages, strict=True):
        valid_advantage = torch.where(weight > 0, advantage, 0)
        counts[6] += ((valid_advantage.abs() * weight).sum(-1) > 0).sum()
    policy_tokens, policy_rows, mask_tokens, mask_rows, teacher_tokens, teacher_rows, nonzero_rows = all_reduce_sum(
        counts
    ).tolist()
    return StepCounts(
        policy=WeightCounts(policy_tokens, policy_rows),
        mask=WeightCounts(mask_tokens, mask_rows),
        teacher=WeightCounts(teacher_tokens, teacher_rows),
        nonzero_advantage_rows=nonzero_rows,
        max_seq_len=max_seq_len,
    )


def reduce_to_step(
    values: torch.Tensor,
    data_weights: torch.Tensor,
    counts: WeightCounts,
    mode: LossReduction,
    *,
    max_seq_len: int,
    nonzero_advantage_rows: float,
    numerator_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return this microbatch's contribution to the globally normalized objective row."""
    assert values.shape == data_weights.shape
    valid = data_weights > 0
    weighted = torch.where(valid, values, 0) * data_weights
    if numerator_weights is not None:
        assert numerator_weights.shape == values.shape
        weighted = weighted * torch.where(valid, numerator_weights, 0)
    if mode == LossReduction.TOKEN_MEAN:
        return weighted.sum() / (counts.tokens or 1.0)
    if mode == LossReduction.SEQUENCE_MEAN:
        row_weights = data_weights.sum(-1)
        row_denominator = torch.where(row_weights > 0, row_weights, 1)
        return (weighted.sum(-1) / row_denominator).sum() / max(counts.rows, 1.0)
    if mode == LossReduction.PAIR_MEAN:
        # Both rows of a valid pair are nonempty, so half the global row count is the pair count.
        return weighted.sum() / max(counts.rows / 2.0, 1.0)
    if mode == LossReduction.SEQ_MEAN_TOKEN_SUM_NORM:
        return weighted.sum() / (max(counts.rows, 1.0) * max_seq_len)
    if mode == LossReduction.SEQ_MEAN_TOKEN_SUM_NORM_GLOBAL:
        return weighted.sum() / (max(nonzero_advantage_rows, 1.0) * max_seq_len)
    raise ValueError(f"Invalid loss reduction: {mode}")
