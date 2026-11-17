"""Advantage estimator implementations and dispatch.

Adapted from VERL's ``trainer/ppo/core_algos.py`` (ByteDance and Hugging Face),
licensed under Apache 2.0.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional, Tuple

import loguru
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
from skyrl_train.utils.policy_math import masked_whiten, right_pad_to_match
from skyrl_train.group_admission import GroupAdvantageInvariant, GroupAdvantageKind

GRPO_FLAT_REWARD_STD_TOLERANCE = 1e-6


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
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    This advantage estimator is also used in LOOP (https://arxiv.org/pdf/2502.01600),
    and was originally introduced in "Buy 4 REINFORCE Samples, Get a Baseline for Free!"
    (https://openreview.net/pdf?id=r1lgTGL5DE).

    Args:
        - token_level_rewards: Float[torch.Tensor, "batch_size seqlen"]
        - response_mask: Float[torch.Tensor, "batch_size seqlen"]
        - index: np.ndarray (batch_size)

    Returns:
        - advantages: Float[torch.Tensor, "batch_size seqlen"]
        - returns: Float[torch.Tensor, "batch_size seqlen"]
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0, device=scores.device)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                factor = response_num / (response_num - 1)
                scores[i] = (scores[i] - id2mean[index[i]]) * factor
            else:
                # if there's only one response, set the advantage to 0
                loguru.logger.warning(f"Only one response for prompt index {index[i]}, setting advantage to 0")
                scores[i] = 0.0
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_advantage_estimator(AdvantageEstimator.RLOO_N, group_contract=MinimumBaselineEligibleGroup())
def compute_rloo_n_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    exclude_from_baseline: Optional[np.ndarray] = None,
    group_advantage_invariant: GroupAdvantageInvariant | None = None,
    config=None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    RLOO-N (RLOO-Neutral): RLOO variant that excludes masked samples from baseline computation.

    This addresses a key limitation in standard RLOO when handling failed samples:
    - Infrastructure failures (DaytonaError, NetworkError) should be treated as "neutral" -
      they don't reflect agent quality and shouldn't affect the baseline.
    - Agent failures (timeout, context overflow) should be included with zero reward.

    When exclude_from_baseline[i] is True:
    1. The sample is excluded from the group baseline calculation
    2. The sample receives advantage=0 (no gradient contribution)
    3. Other samples in the group have their baselines computed WITHOUT this sample

    This is different from just setting reward=0, which would still pollute the baseline
    by dragging down the mean for the entire group.

    Args:
        - token_level_rewards: Float[torch.Tensor, "batch_size seqlen"]
        - response_mask: Float[torch.Tensor, "batch_size seqlen"]
        - index: np.ndarray (batch_size) - group IDs for each sample
        - exclude_from_baseline: Optional[np.ndarray] (batch_size) - bool array, True = exclude

    Returns:
        - advantages: Float[torch.Tensor, "batch_size seqlen"]
        - returns: Float[torch.Tensor, "batch_size seqlen"]
    """
    scores = token_level_rewards.sum(dim=-1)
    bsz = scores.shape[0]

    # Minimum included samples per group for a reliable leave-one-out baseline.
    # Groups below this threshold get advantage=0 for all samples.
    if group_advantage_invariant is None:
        raise ValueError("RLOO-N requires a resolved group_advantage_invariant")
    if group_advantage_invariant.kind is not GroupAdvantageKind.MINIMUM_BASELINE_ELIGIBLE:
        raise ValueError(f"RLOO-N requires a minimum baseline-eligible contract, got {group_advantage_invariant.kind}")
    assert group_advantage_invariant.minimum_group_size is not None
    min_group_size = group_advantage_invariant.minimum_group_size
    filter_zero_reward_groups = True
    if config is not None:
        filter_zero_reward_groups = getattr(config, "rloo_n_filter_zero_reward_groups", True)

    # Default: include all samples in baseline
    if exclude_from_baseline is None:
        exclude_from_baseline = np.zeros(bsz, dtype=bool)

    # Build per-group score lists, separating included vs excluded
    id2included_scores = defaultdict(list)  # scores to include in baseline
    id2included_indices = defaultdict(list)  # indices of included samples
    id2excluded_indices = defaultdict(list)  # indices of excluded samples

    with torch.no_grad():
        # First pass: categorize samples
        for i in range(bsz):
            group_id = index[i]
            if exclude_from_baseline[i]:
                id2excluded_indices[group_id].append(i)
            else:
                id2included_scores[group_id].append(scores[i])
                id2included_indices[group_id].append(i)

        # Constant-reward groups have zero RLOO advantage. Filter them instead of
        # introducing noise that can push the policy toward entropy collapse.
        id2no_variance = {}
        n_no_variance_groups = 0
        n_no_variance_samples = 0
        for group_id in set(index):
            included = id2included_scores[group_id]
            if filter_zero_reward_groups and len(included) > 1:
                stacked = torch.stack(included)
                has_no_variance = (stacked.max() - stacked.min()).item() == 0.0
                id2no_variance[group_id] = has_no_variance
                if has_no_variance:
                    n_no_variance_groups += 1
                    n_no_variance_samples += len(included) + len(id2excluded_indices[group_id])
            else:
                id2no_variance[group_id] = False

        # Second pass: compute baselines using only included samples
        id2mean = {}
        for group_id in set(index):
            included = id2included_scores[group_id]
            if id2no_variance.get(group_id, False):
                # Zero-variance group — skip entirely
                id2mean[group_id] = torch.tensor(0.0, device=scores.device)
            elif len(included) < min_group_size:
                # Below minimum group size — can't compute reliable baseline
                id2mean[group_id] = torch.tensor(0.0, device=scores.device)
            else:
                id2mean[group_id] = torch.mean(torch.stack(included))

        # Third pass: compute advantages
        for i in range(bsz):
            group_id = index[i]

            if exclude_from_baseline[i]:
                # Excluded samples get zero advantage (no gradient contribution)
                scores[i] = 0.0
                continue

            if id2no_variance.get(group_id, False):
                # Zero-variance reward group — zero advantage, no gradient
                scores[i] = 0.0
                continue

            # For included samples: use leave-one-out baseline from OTHER included samples
            included_scores = id2included_scores[group_id]
            n_included = len(included_scores)

            if n_included < min_group_size:
                # Below minimum group size — zero advantage for all included samples
                loguru.logger.warning(
                    f"RLOO-N: Group {group_id} has {n_included} included sample(s) "
                    f"(min_group_size={min_group_size}), setting advantage to 0"
                )
                scores[i] = 0.0
            else:
                # Standard RLOO leave-one-out: baseline = mean of OTHER samples
                # With correction factor: (n / (n-1)) * (score - group_mean)
                factor = n_included / (n_included - 1)
                scores[i] = (scores[i] - id2mean[group_id]) * factor

        # Log summary statistics
        n_excluded = sum(len(v) for v in id2excluded_indices.values())
        n_groups_all_excluded = sum(1 for group_id in set(index) if len(id2included_scores[group_id]) == 0)
        n_groups_below_min = sum(1 for group_id in set(index) if 0 < len(id2included_scores[group_id]) < min_group_size)
        n_total_groups = len(set(index))
        if n_excluded > 0 or n_groups_below_min > 0 or n_no_variance_groups > 0:
            loguru.logger.info(
                f"RLOO-N: {n_excluded}/{bsz} samples excluded from baseline, "
                f"{n_groups_all_excluded} groups had all samples excluded, "
                f"{n_groups_below_min} groups below min_group_size={min_group_size}"
                + (
                    f", {n_no_variance_groups}/{n_total_groups} groups filtered "
                    f"(zero reward variance, {n_no_variance_samples} samples)"
                    if n_no_variance_groups > 0
                    else ""
                )
            )

        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_advantage_estimator("rloo_n_pbs", group_contract=MinimumBaselineEligibleGroup())
def compute_rloo_n_pbs_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    exclude_from_baseline: Optional[np.ndarray] = None,
    group_advantage_invariant: GroupAdvantageInvariant | None = None,
    config=None,
    token_level_shaping: Optional[torch.Tensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine RLOO-N outcome advantage with potential-based token shaping.

    Combines RLOO-N's per-trajectory outcome advantage (computed exactly as
    ``compute_rloo_n_outcome_advantage`` — the outcome term reads ``rewards``
    ONLY and is left bit-for-bit intact) with the per-token potential-based
    shaping channel ``token_level_shaping`` (the PBS test-delta credit scattered
    onto the EDIT-token span by ``pbs_shaping.compute_pbs_token_shaping``).

    The two signals remain additive and separate:

        advantage = rloo_n_outcome_advantage + token_level_shaping * response_mask

    Properties:
      * ``token_level_shaping is None`` or all-zeros ⇒ this returns EXACTLY the
        RLOO-N advantage (pure RLOO-N; the flag-off / no-signal path).
      * PBS is policy-invariant (Ng 1999): ``token_level_shaping`` is a true
        potential difference ``γ·Φ(s') − Φ(s)`` built upstream, so adding it
        cannot change the optimal policy.
      * The shaping is masked by ``response_mask`` and only applied to response
        tokens (the same support as the outcome advantage), so the
        advantage/loss denominator (``response_mask.sum()``) is unchanged — no
        seqnorm-style denominator break.

    Returns ``(advantages, returns)`` with the same shape/semantics as RLOO-N
    (advantages == returns; critic-free).
    """
    # Outcome term: unchanged RLOO-N (reads `rewards` only).
    outcome_adv, _ = compute_rloo_n_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        exclude_from_baseline=exclude_from_baseline,
        group_advantage_invariant=group_advantage_invariant,
        config=config,
        **kwargs,
    )

    if token_level_shaping is None:
        return outcome_adv, outcome_adv

    with torch.no_grad():
        shaping = token_level_shaping.to(device=outcome_adv.device, dtype=outcome_adv.dtype)
        shaping = right_pad_to_match(shaping, response_mask, dtype=outcome_adv.dtype)
        combined = outcome_adv + shaping * response_mask

    return combined, combined


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
    epsilon: float = 1e-6,
    grpo_norm_by_std: bool = True,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO, operating only on Outcome reward (with only one scalar reward for each response).

    Expects:
        - token_level_rewards: Float[torch.Tensor, "batch_size seqlen"]
        - response_mask: Float[torch.Tensor, "batch_size seqlen"]
        - index: np.ndarray (batch_size)
        - epsilon: float
        - grpo_norm_by_std: bool

    Returns:
        - advantages: Float[torch.Tensor, "batch_size seqlen"]
        - returns: Float[torch.Tensor, "batch_size seqlen"]
    """
    # this assumes response-level rewards
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if grpo_norm_by_std:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


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
