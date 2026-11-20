"""Reusable selection of completed trajectories before teacher scoring and training."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean

from omegaconf import DictConfig

from skyrl_train.batch_sampling import filter_trajectory_batch
from skyrl_train.config.trajectory_selection import BEST_OF_N_LEARNER_ROWS, best_of_n_sample_count, require_best_of_n_samples
from skyrl_train.trajectory_runners.trajectory_reward_shaping import NormalizedReward
from skyrl_train.trajectory_runners.types import TrajectoryBatch


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
        require_best_of_n_samples(n_samples_per_prompt)
        self._n_samples_per_prompt = n_samples_per_prompt

    @property
    def optimization_samples_per_prompt(self) -> int:
        return BEST_OF_N_LEARNER_ROWS

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
    samples = best_of_n_sample_count(cfg)
    return None if samples is None else BestOfNTrajectorySelector(samples)
