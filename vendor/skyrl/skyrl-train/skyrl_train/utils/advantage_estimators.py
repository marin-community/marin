"""Advantage estimator implementations and dispatch.

Adapted from VERL's ``trainer/ppo/core_algos.py`` (ByteDance and Hugging Face),
licensed under Apache 2.0.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from jaxtyping import Float
from omegaconf import DictConfig

from marinskyrl.runtime_options import AdvantageEstimator
from skyrl_train.utils.algorithm_registry import (
    AdvantageEstimatorRegistry,
    ExactPhysicalGroup,
    MinimumBaselineEligibleGroup,
    NoGroupAdvantage,
    register_advantage_estimator,
)
from skyrl_train.utils.policy_math import masked_whiten
from skyrl_train.group_admission import (
    MIN_BASELINE_GROUP_SIZE,
    GroupAdvantageInvariant,
    GroupAdvantageKind,
    rewards_are_flat,
)

# Added to a group's reward standard deviation before dividing by it.
GROUP_STD_EPSILON = 1e-6


def _baseline_groups(index: Sequence[object], exclude_from_baseline: np.ndarray | None) -> list[list[int]]:
    excluded = np.zeros(len(index), dtype=bool) if exclude_from_baseline is None else exclude_from_baseline
    groups: dict[object, list[int]] = defaultdict(list)
    for row, (group_id, row_excluded) in enumerate(zip(index, excluded, strict=True)):
        if not row_excluded:
            groups[group_id].append(row)
    return list(groups.values())


@torch.no_grad()
def _group_advantages(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: Sequence[object],
    exclude_from_baseline: np.ndarray | None,
    *,
    leave_one_out: bool,
    divide_by_std: bool,
    min_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Centre eligible rewards on their group's baseline and broadcast over response tokens."""
    scores = token_level_rewards.sum(dim=-1)
    advantages = torch.zeros_like(scores)
    for rows in _baseline_groups(index, exclude_from_baseline):
        group = scores[rows]
        if len(rows) < min_size or rewards_are_flat(group.tolist()):
            continue
        centred = group - group.mean()
        if leave_one_out:
            centred = centred * (len(rows) / (len(rows) - 1))
        if divide_by_std:
            centred = centred / (group.std() + GROUP_STD_EPSILON)
        advantages[rows] = centred
    advantages = advantages.unsqueeze(-1) * response_mask
    return advantages, advantages


def flat_group_fraction(
    token_level_rewards: torch.Tensor,
    index: Sequence[object],
    exclude_from_baseline: np.ndarray | None,
) -> float:
    """Fraction of all prompt groups with at least two eligible rewards that are flat."""
    scores = token_level_rewards.sum(dim=-1)
    group_count = len(set(index))
    if group_count == 0:
        return 0.0
    flat = sum(
        len(rows) >= MIN_BASELINE_GROUP_SIZE and rewards_are_flat(scores[rows].tolist())
        for rows in _baseline_groups(index, exclude_from_baseline)
    )
    return flat / group_count


@register_advantage_estimator(AdvantageEstimator.UNIFORM, group_contract=NoGroupAdvantage())
def compute_uniform_advantage(
    token_level_rewards: torch.Tensor,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Give every selected response token unit SFT weight."""
    ones = torch.ones_like(token_level_rewards)
    return ones, ones


@register_advantage_estimator(AdvantageEstimator.REWARD, group_contract=NoGroupAdvantage())
@torch.no_grad()
def compute_reward_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Broadcast each response's eligible reward sum to its eligible tokens."""
    valid = response_mask > 0
    rewards = torch.where(valid, token_level_rewards, 0)
    advantages = torch.where(valid, rewards.sum(dim=-1, keepdim=True), 0)
    return advantages, advantages


@register_advantage_estimator(AdvantageEstimator.REINFORCE_PP, group_contract=NoGroupAdvantage())
def compute_reinforce_plus_plus_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: float,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262

    Args:
        - token_level_rewards: Float[torch.Tensor, "batch_size seqlen"]
        - response_mask: Float[torch.Tensor, "batch_size seqlen"]

    Returns:
        - advantages: Float[torch.Tensor, "batch_size seqlen"]
        - returns: Float[torch.Tensor, "batch_size seqlen"]
    """
    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * response_mask[:, t]

        advantages = masked_whiten(returns, response_mask)
        advantages = advantages * response_mask

    return advantages, returns


@register_advantage_estimator(AdvantageEstimator.RLOO, group_contract=ExactPhysicalGroup())
def compute_rloo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    exclude_from_baseline: np.ndarray | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RLOO: each eligible response minus the mean of its group's other eligible responses.

    See https://arxiv.org/abs/2402.14740 and https://openreview.net/pdf?id=r1lgTGL5DE.
    """
    return _group_advantages(
        token_level_rewards,
        response_mask,
        index,
        exclude_from_baseline,
        leave_one_out=True,
        divide_by_std=False,
        min_size=MIN_BASELINE_GROUP_SIZE,
    )


@register_advantage_estimator(AdvantageEstimator.RLOO_N, group_contract=MinimumBaselineEligibleGroup())
def compute_rloo_n_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    exclude_from_baseline: np.ndarray | None = None,
    group_advantage_invariant: GroupAdvantageInvariant | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RLOO with the run's configured minimum number of baseline-eligible responses per group."""
    if group_advantage_invariant is None:
        raise ValueError("RLOO-N requires a resolved group_advantage_invariant")
    if group_advantage_invariant.kind is not GroupAdvantageKind.MINIMUM_BASELINE_ELIGIBLE:
        raise ValueError(f"RLOO-N requires a minimum baseline-eligible contract, got {group_advantage_invariant.kind}")
    assert group_advantage_invariant.minimum_group_size is not None
    return _group_advantages(
        token_level_rewards,
        response_mask,
        index,
        exclude_from_baseline,
        leave_one_out=True,
        divide_by_std=False,
        min_size=group_advantage_invariant.minimum_group_size,
    )


@register_advantage_estimator(AdvantageEstimator.GAE, group_contract=NoGroupAdvantage())
def compute_gae_advantage_return(
    token_level_rewards: Float[torch.Tensor, "batch_size seqlen"],
    values: Float[torch.Tensor, "batch_size seqlen"],
    response_mask: Float[torch.Tensor, "batch_size seqlen"],
    gamma: float,
    lambd: float,
    **kwargs,
) -> Tuple[Float[torch.Tensor, "batch_size seqlen"], Float[torch.Tensor, "batch_size seqlen"]]:
    """
    Compute advantage and return for GAE.

    Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py
    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = masked_whiten(advantages, response_mask)
    return advantages, returns


@register_advantage_estimator(AdvantageEstimator.GRPO, group_contract=ExactPhysicalGroup())
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    grpo_norm_by_std: bool = True,
    exclude_from_baseline: np.ndarray | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GRPO: each eligible response minus its group's eligible mean, optionally divided by its standard deviation."""
    return _group_advantages(
        token_level_rewards,
        response_mask,
        index,
        exclude_from_baseline,
        leave_one_out=False,
        divide_by_std=grpo_norm_by_std,
        min_size=MIN_BASELINE_GROUP_SIZE,
    )


def compute_advantages_and_returns(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    adv_estimator: AdvantageEstimator,
    config: DictConfig,
    values: Optional[torch.Tensor] = None,
    grpo_norm_by_std: bool = True,
    gamma=1.0,
    lambd=1.0,
    exclude_from_baseline: Optional[np.ndarray] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    estimator_func = AdvantageEstimatorRegistry.get(adv_estimator)

    return estimator_func(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        values=values,
        grpo_norm_by_std=grpo_norm_by_std,
        gamma=gamma,
        lambd=lambd,
        config=config,
        exclude_from_baseline=exclude_from_baseline,
        **kwargs,
    )
