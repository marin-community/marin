"""Advantage estimator implementations and dispatch.

Adapted from VERL's ``trainer/ppo/core_algos.py`` (ByteDance and Hugging Face),
licensed under Apache 2.0.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from omegaconf import DictConfig

from marinskyrl.runtime_options import AdvantageEstimator
from skyrl_train.utils.algorithm_registry import (
    AdvantageEstimatorRegistry,
    register_advantage_estimator,
)
from skyrl_train.utils.policy_math import masked_whiten
from skyrl_train.group_admission import (
    BUILTIN_GROUP_ADVANTAGE_CONTRACTS,
    FLAT_GROUP_REWARD_TOLERANCE,
    MIN_BASELINE_GROUP_SIZE,
    GroupAdvantageInvariant,
    GroupAdvantageKind,
    baseline_group_rows,
)
from skyrl_train.training_batch import TrainingInputBatch

GROUP_STD_EPSILON = 1e-6


def apply_loop_advantages(data: TrainingInputBatch) -> TrainingInputBatch:
    """Add masked loop credit to policy advantages."""
    loop_advantages = data.get("loop_advantages")
    if loop_advantages is None:
        return data
    advantages = data["advantages"]
    loop_advantages = loop_advantages.to(device=advantages.device, dtype=advantages.dtype)
    data["advantages"] = advantages + loop_advantages * data["response_mask"]
    return data


def drop_advantage_inputs(data: TrainingInputBatch) -> TrainingInputBatch:
    """Retain optimizer inputs after advantage estimation and loop credit."""
    data.pop("rewards")
    data.pop("loop_advantages", None)
    data.metadata.pop("uids")
    return data


def batch_advantages(
    data: TrainingInputBatch, algorithm: DictConfig, *, invariant: GroupAdvantageInvariant
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute batch credit through the configured full-width token estimator."""
    return compute_advantages_and_returns(
        token_level_rewards=data["rewards"],
        response_mask=data["response_mask"],
        index=data.metadata["uids"],
        adv_estimator=algorithm.advantage_estimator,
        config=algorithm,
        gamma=algorithm.gamma,
        grpo_norm_by_std=algorithm.grpo_norm_by_std,
        exclude_from_baseline=data.metadata.get("exclude_from_baseline"),
        group_advantage_invariant=invariant,
    )


def _register_builtin_advantage_estimator(name: str):
    return register_advantage_estimator(name, group_contract=BUILTIN_GROUP_ADVANTAGE_CONTRACTS[name])


@_register_builtin_advantage_estimator(AdvantageEstimator.UNIFORM)
def compute_uniform_advantage(
    token_level_rewards: torch.Tensor,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Give every selected response token unit SFT weight."""
    ones = torch.ones_like(token_level_rewards)
    return ones, ones


@_register_builtin_advantage_estimator(AdvantageEstimator.REWARD)
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


@_register_builtin_advantage_estimator(AdvantageEstimator.REINFORCE_PP)
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


@torch.no_grad()
def _group_advantages(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: Sequence[object],
    exclude_from_baseline: Sequence[bool] | None,
    *,
    estimator: AdvantageEstimator,
    min_size: int = MIN_BASELINE_GROUP_SIZE,
    grpo_norm_by_std: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Center eligible rewards and broadcast their advantages over each response."""
    scores = token_level_rewards.sum(dim=-1)
    advantages = torch.zeros_like(scores)
    for rows in baseline_group_rows(index, exclude_from_baseline):
        if len(rows) < min_size:
            continue
        group = scores[rows]
        centered = group - group.mean()
        if estimator is not AdvantageEstimator.GRPO:
            centered = centered * (len(rows) / (len(rows) - 1))
        elif grpo_norm_by_std:
            centered = centered / (group.std() + GROUP_STD_EPSILON)
        reward_range = group.max() - group.min()
        advantages[rows] = torch.where(reward_range <= FLAT_GROUP_REWARD_TOLERANCE, 0, centered)
    advantages = advantages.unsqueeze(-1) * response_mask
    return advantages, advantages


@_register_builtin_advantage_estimator(AdvantageEstimator.RLOO)
def compute_rloo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: Sequence[object],
    exclude_from_baseline: Sequence[bool] | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Subtract other eligible responses' mean for non-flat groups with at least two eligible responses."""
    return _group_advantages(
        token_level_rewards, response_mask, index, exclude_from_baseline, estimator=AdvantageEstimator.RLOO
    )


@_register_builtin_advantage_estimator(AdvantageEstimator.RLOO_N)
def compute_rloo_n_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: Sequence[object],
    exclude_from_baseline: Sequence[bool] | None = None,
    group_advantage_invariant: GroupAdvantageInvariant | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply leave-one-out credit subject to the configured minimum eligible group size."""
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
        estimator=AdvantageEstimator.RLOO_N,
        min_size=group_advantage_invariant.minimum_group_size,
    )


@_register_builtin_advantage_estimator(AdvantageEstimator.GRPO)
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: Sequence[object],
    grpo_norm_by_std: bool = True,
    exclude_from_baseline: Sequence[bool] | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Center non-flat eligible rewards on their prompt's mean, with optional standard-deviation scaling."""
    return _group_advantages(
        token_level_rewards,
        response_mask,
        index,
        exclude_from_baseline,
        estimator=AdvantageEstimator.GRPO,
        grpo_norm_by_std=grpo_norm_by_std,
    )


def compute_advantages_and_returns(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: Sequence[object],
    adv_estimator: AdvantageEstimator,
    config: DictConfig,
    grpo_norm_by_std: bool = True,
    gamma=1.0,
    exclude_from_baseline: Sequence[bool] | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    estimator_func = AdvantageEstimatorRegistry.get(adv_estimator)

    return estimator_func(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        grpo_norm_by_std=grpo_norm_by_std,
        gamma=gamma,
        config=config,
        exclude_from_baseline=exclude_from_baseline,
        **kwargs,
    )
