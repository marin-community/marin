"""Shared group-selection policy for dynamic sampling."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import math
import statistics
from typing import Mapping, Protocol, Sequence


class DynamicSamplingType(StrEnum):
    FILTER = "filter"


class DynamicSamplingRewardSource(StrEnum):
    SHAPED = "shaped"
    UNSHAPED = "unshaped"


DEFAULT_DYNAMIC_SAMPLING_REWARD_SOURCE = DynamicSamplingRewardSource.SHAPED
DEFAULT_DYNAMIC_SAMPLING_MIN_REWARD_STD = 0.0


@dataclass(frozen=True)
class DynamicSamplingCriteria:
    reward_source: DynamicSamplingRewardSource
    min_reward_std: float
    max_mean_reward: float | None = None


DEFAULT_DYNAMIC_SAMPLING_CRITERIA = DynamicSamplingCriteria(
    reward_source=DEFAULT_DYNAMIC_SAMPLING_REWARD_SOURCE,
    min_reward_std=DEFAULT_DYNAMIC_SAMPLING_MIN_REWARD_STD,
)


def resolve_dynamic_sampling_criteria(
    informative_on: str = DEFAULT_DYNAMIC_SAMPLING_REWARD_SOURCE,
    min_reward_std: float = DEFAULT_DYNAMIC_SAMPLING_MIN_REWARD_STD,
    max_mean_reward: float | None = None,
) -> DynamicSamplingCriteria:
    reward_source = DynamicSamplingRewardSource(informative_on)
    if not math.isfinite(min_reward_std) or min_reward_std < 0:
        raise ValueError("dynamic_sampling.min_reward_std must be finite and non-negative")
    if max_mean_reward is not None and not math.isfinite(max_mean_reward):
        raise ValueError("dynamic_sampling.max_mean_reward must be finite or null")
    return DynamicSamplingCriteria(reward_source, min_reward_std, max_mean_reward)


class GroupSelectionResult(StrEnum):
    KEEP = "keep"
    INSUFFICIENT_REWARD_SPREAD = "insufficient_reward_spread"
    REWARD_MEAN_TOO_HIGH = "reward_mean_too_high"


class GeneratedGroup(Protocol):
    trajectory_batch: Mapping[str, object]


def _aligned_sequence(batch: Mapping[str, object], key: str, row_count: int) -> Sequence[object] | None:
    value = batch.get(key)
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{key} must be a sequence when present")
    if len(value) != row_count:
        raise ValueError(f"{key} must have one entry per response row, got {len(value)} and {row_count}")
    return value


def _reward_total(reward: object) -> float:
    if isinstance(reward, Sequence) and not isinstance(reward, (str, bytes)):
        return sum(float(value) for value in reward)
    return float(reward)


def group_selection_result(
    trajectory_batch: Mapping[str, object],
    row_indices: Sequence[int] | None = None,
    *,
    criteria: DynamicSamplingCriteria,
) -> GroupSelectionResult:
    """Select groups whose final outcome mean and spread satisfy the configured limits."""
    response_ids = trajectory_batch.get("response_ids")
    if not isinstance(response_ids, Sequence) or isinstance(response_ids, (str, bytes)):
        raise ValueError("response_ids must be a sequence")
    row_count = len(response_ids)
    if row_indices is None:
        row_indices = range(row_count)
    reward_key = "rewards" if criteria.reward_source is DynamicSamplingRewardSource.SHAPED else "unshaped_rewards"
    outcomes = _aligned_sequence(trajectory_batch, reward_key, row_count)
    if outcomes is None:
        raise ValueError(f"dynamic sampling filter requires {reward_key} for every generated group")
    if criteria.reward_source is DynamicSamplingRewardSource.UNSHAPED:
        availability = _aligned_sequence(trajectory_batch, "unshaped_reward_available", row_count)
        if availability is not None and any(not availability[index] for index in row_indices):
            return GroupSelectionResult.INSUFFICIENT_REWARD_SPREAD
    is_last_step = _aligned_sequence(trajectory_batch, "is_last_step", row_count)

    final_outcomes = []
    for index in row_indices:
        if is_last_step is not None and not bool(is_last_step[index]):
            continue
        final_outcomes.append(_reward_total(outcomes[index]))
    if not final_outcomes:
        raise ValueError("dynamic sampling group must contain at least one final trial row")
    if criteria.max_mean_reward is not None and statistics.mean(final_outcomes) >= criteria.max_mean_reward:
        return GroupSelectionResult.REWARD_MEAN_TOO_HIGH
    if len(final_outcomes) == 1 or statistics.pstdev(final_outcomes) > criteria.min_reward_std:
        return GroupSelectionResult.KEEP
    return GroupSelectionResult.INSUFFICIENT_REWARD_SPREAD


class GroupSelectionPolicy:
    """Apply prompt-consuming selection rules after training eligibility checks."""

    def __init__(
        self,
        sampling_type: DynamicSamplingType | None,
        *,
        criteria: DynamicSamplingCriteria = DEFAULT_DYNAMIC_SAMPLING_CRITERIA,
    ) -> None:
        self.sampling_type = sampling_type
        self.criteria = criteria

    def evaluate(self, group: GeneratedGroup) -> GroupSelectionResult:
        if self.sampling_type is None:
            return GroupSelectionResult.KEEP
        return group_selection_result(group.trajectory_batch, criteria=self.criteria)
