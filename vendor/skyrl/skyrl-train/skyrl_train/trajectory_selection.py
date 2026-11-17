"""Reusable selection of completed trajectories before teacher scoring and training."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from statistics import fmean

from omegaconf import DictConfig, OmegaConf

from marinskyrl.runtime_options import TRAJECTORY_SELECTOR_TYPE_PATH
from skyrl_train.batch_sampling import filter_trajectory_batch
from skyrl_train.trajectory_runners.trajectory_reward_shaping import NormalizedReward
from skyrl_train.trajectory_runners.types import TrajectoryBatch


class TrajectorySelectorKind(StrEnum):
    BEST_OF_N = "best_of_n"


@dataclass(frozen=True)
class TrajectorySelection:
    trajectory_batch: TrajectoryBatch
    uids: list[str]
    metrics: dict[str, float]


def best_of_n_indices(rewards: list[float], n_samples_per_prompt: int) -> list[int]:
    """Return the first maximum-reward sample in each contiguous prompt group."""
    if n_samples_per_prompt <= 0:
        raise ValueError("n_samples_per_prompt must be positive")
    if not rewards:
        raise ValueError("best-of-N selection requires at least one reward")
    if len(rewards) % n_samples_per_prompt:
        raise ValueError(
            f"number of rewards ({len(rewards)}) must be divisible by n_samples_per_prompt ({n_samples_per_prompt})"
        )
    selected = []
    for start in range(0, len(rewards), n_samples_per_prompt):
        group = rewards[start : start + n_samples_per_prompt]
        selected.append(start + max(range(len(group)), key=group.__getitem__))
    return selected


class BestOfNTrajectorySelector:
    """Retain one reward-maximizing completion from each physical rollout group."""

    def __init__(self, n_samples_per_prompt: int) -> None:
        if n_samples_per_prompt <= 1:
            raise ValueError("best-of-N selection requires n_samples_per_prompt greater than one")
        self._n_samples_per_prompt = n_samples_per_prompt

    @property
    def optimization_samples_per_prompt(self) -> int:
        return 1

    def select(self, trajectory_batch: TrajectoryBatch, uids: list[str]) -> TrajectorySelection:
        row_count = len(trajectory_batch["response_ids"])
        if row_count == 0:
            raise ValueError("best-of-N selection requires at least one trajectory")
        if len(uids) != row_count:
            raise ValueError(f"trajectory selection received {row_count} rows but {len(uids)} uids")
        rewards = [NormalizedReward.from_output(reward).total for reward in trajectory_batch["rewards"]]
        selected_indices = best_of_n_indices(rewards, self._n_samples_per_prompt)
        selected_rewards = [rewards[index] for index in selected_indices]
        group_means = [
            fmean(rewards[start : start + self._n_samples_per_prompt])
            for start in range(0, row_count, self._n_samples_per_prompt)
        ]
        return TrajectorySelection(
            trajectory_batch=filter_trajectory_batch(trajectory_batch, selected_indices),
            uids=[uids[index] for index in selected_indices],
            metrics={
                "best_of_n/best_reward_mean": fmean(selected_rewards),
                "best_of_n/group_reward_mean": fmean(group_means),
                "best_of_n/reward_improvement": fmean(selected_rewards) - fmean(group_means),
                "best_of_n/n_samples_per_prompt": float(self._n_samples_per_prompt),
            },
        )


def trajectory_selector_from_config(cfg: DictConfig) -> BestOfNTrajectorySelector | None:
    selector_type = OmegaConf.select(cfg, TRAJECTORY_SELECTOR_TYPE_PATH, default=None)
    if selector_type is None:
        return None
    try:
        selector_kind = TrajectorySelectorKind(selector_type)
    except ValueError as error:
        raise ValueError(f"unsupported trajectory selector {selector_type!r}") from error
    if selector_kind is TrajectorySelectorKind.BEST_OF_N:
        return BestOfNTrajectorySelector(int(cfg.generator.n_samples_per_prompt))
    raise AssertionError(f"unhandled trajectory selector {selector_kind!r}")


def optimization_samples_per_prompt(cfg: DictConfig) -> int:
    """Return the number of rows per prompt that reach learner workers."""
    selector = trajectory_selector_from_config(cfg)
    if selector is None:
        return int(cfg.generator.n_samples_per_prompt)
    return selector.optimization_samples_per_prompt
