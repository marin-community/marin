# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Replay buffer for RL training data.

This module provides a replay buffer that manages rollout data for training,
with balanced sampling across environments and configurable prioritization.
"""

import dataclasses
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from marin.rl.rl_losses import RLLossModule

from .rollout_storage import RolloutReader
from .types import Rollout, RolloutBatch, RolloutWithAdvantage

logger = logging.getLogger(__name__)

# TODO(power) - move advantage calculation and separate it from count


@dataclass
class SoftOverlongPenaltyConfig:
    """Linearly penalize responses that enter the final decoding-length buffer."""

    buffer_length: int
    """Number of tokens at the end of the response budget to penalize."""

    penalty_factor: float
    """Maximum reward penalty applied at or beyond the response budget."""

    def __post_init__(self):
        if self.buffer_length <= 0:
            raise ValueError(f"buffer_length must be positive, got {self.buffer_length}")
        if self.penalty_factor < 0.0:
            raise ValueError(f"penalty_factor must be non-negative, got {self.penalty_factor}")


def soft_overlong_penalty(response_length: int, max_output_tokens: int, config: SoftOverlongPenaltyConfig) -> float:
    """Return the DAPO soft overlong reward penalty for a response length."""
    if config.buffer_length > max_output_tokens:
        raise ValueError(
            "SoftOverlongPenaltyConfig.buffer_length must be less than or equal to rollout "
            f"max_output_tokens, got {config.buffer_length} and {max_output_tokens}"
        )

    penalty_start = max_output_tokens - config.buffer_length
    if response_length <= penalty_start:
        return 0.0

    penalty_fraction = min((response_length - penalty_start) / config.buffer_length, 1.0)
    return penalty_fraction * config.penalty_factor


def apply_soft_overlong_penalty(rollout: Rollout, config: SoftOverlongPenaltyConfig | None) -> Rollout:
    """Return a rollout with DAPO soft overlong reward shaping applied."""
    if config is None:
        return rollout

    penalty = soft_overlong_penalty(
        response_length=int(rollout.response_tokens.size),
        max_output_tokens=rollout.decoding.max_output_tokens,
        config=config,
    )
    if penalty == 0.0:
        return rollout

    return dataclasses.replace(rollout, episode_reward=rollout.episode_reward - penalty)


@dataclass
class ReplayBufferConfig:
    """Configuration for the replay buffer."""

    capacity: int
    """Maximum number of examples per environment in the buffer."""

    alpha: float
    """Recency bias for sampling, higher values favor newer examples."""

    max_samples: int
    """Maximum number of times to use an example before retiring."""

    max_rollout_step_delay: int
    """Maximum age of rollouts in training steps, rollouts earlier than this will be dropped."""

    max_rollout_timestamp_delay: float = 3600.0
    """Maximum age of rollouts in seconds."""

    filter_out_groups_with_no_variance: bool = False
    """Filter out groups with no variance in rewards."""

    soft_overlong_penalty: SoftOverlongPenaltyConfig | None = None
    """Optional DAPO soft overlong reward shaping applied before advantage computation."""


@dataclass
class RolloutWithCount(RolloutWithAdvantage):
    """Single rollout with precomputed RLOO advantage & usage count tracking."""

    usage_count: int = 0
    weight_step: int = 0


@dataclass
class ReplayBuffer:
    """The replay buffer manages incoming rollout data and produces training batches.

    It attempts to prioritize recent data while balancing the number of samples from
    each environment. As rollout examples appear, they are folded into a shared
    rollout array which is sampled from to produce training batches.
    """

    capacity: int
    local_batch_size: int
    alpha: float
    total_processes: int
    max_samples: int
    max_rollout_step_delay: int
    max_rollout_timestamp_delay: float
    filter_out_groups_with_no_variance: bool
    loss_module: RLLossModule
    seed: int
    soft_overlong_penalty: SoftOverlongPenaltyConfig | None = None

    _total_batches_added: int = 0
    _total_batches_sampled: int = 0
    _total_reward_groups_seen: int = 0
    _total_reward_groups_dropped_no_variance: int = 0
    _total_soft_overlong_penalty_count: int = 0
    _total_soft_overlong_penalty_sum: float = 0.0
    _lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)
    _current_step: int = 0
    _rng: np.random.Generator = dataclasses.field(init=False)
    rollout_storage: dict[str, list[RolloutWithCount]] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        self._rng = np.random.default_rng(seed=self.seed)

    @classmethod
    def from_config(
        cls,
        config: ReplayBufferConfig,
        local_batch_size: int,
        total_processes: int,
        loss_module: RLLossModule,
        seed: int,
    ) -> "ReplayBuffer":
        """Create ReplayBuffer from configuration.

        Args:
            config: Replay buffer configuration.
            local_batch_size: Batch size for sampling.
            total_processes: Total number of processes.
            loss_module: Loss module for computing advantages.
            seed: Random seed for sampling.

        Returns:
            Configured ReplayBuffer instance.
        """
        return cls(
            capacity=config.capacity,
            local_batch_size=local_batch_size,
            alpha=config.alpha,
            total_processes=total_processes,
            max_samples=config.max_samples,
            max_rollout_step_delay=config.max_rollout_step_delay,
            max_rollout_timestamp_delay=config.max_rollout_timestamp_delay,
            filter_out_groups_with_no_variance=config.filter_out_groups_with_no_variance,
            soft_overlong_penalty=config.soft_overlong_penalty,
            loss_module=loss_module,
            seed=seed,
        )

    def _is_rollout_fresh(
        self,
        rollout_step: int,
        rollout_timestamp: float,
        current_step: int,
        current_time: float,
    ) -> bool:
        # We can receive "future" rollouts if the training worker crashed and restarted.
        # These can introduce unexpected non-determinism, so we explicitly disallow them.
        min_step = current_step - self.max_rollout_step_delay
        max_step = current_step
        if rollout_step < min_step or rollout_step > max_step:
            return False

        # Check timestamp
        min_time = current_time - self.max_rollout_timestamp_delay
        if rollout_timestamp <= min_time:
            return False

        return True

    def set_current_step(self, step: int) -> None:
        """Set current training step and filter stale rollouts."""
        self._current_step = step
        current_time = time.time()

        with self._lock:
            total_removed = 0
            for env_name in self.rollout_storage:
                rollouts = self.rollout_storage[env_name]
                before = len(rollouts)
                self.rollout_storage[env_name] = [
                    r
                    for r in rollouts
                    if self._is_rollout_fresh(
                        r.rollout.metadata.weight_step, r.rollout.metadata.timestamp, step, current_time
                    )
                ]
                total_removed += before - len(self.rollout_storage[env_name])

            total_remaining = sum(len(rollouts) for rollouts in self.rollout_storage.values())

            if total_removed > 0:
                logger.info(f"Filtered {total_removed} stale rollouts {total_remaining} remaining")

    def _retire_overused_rollouts(self):
        """Remove rollouts that exceeded max_samples usage."""
        if self.max_samples < 0:
            return

        for env_name in self.rollout_storage:
            rollouts = self.rollout_storage[env_name]
            # Keep only rollouts under usage limit
            self.rollout_storage[env_name] = [r for r in rollouts if r.usage_count < self.max_samples]

    def add_batches(self, new_batches: list[RolloutBatch]) -> None:
        """Add new rollout batches into the replay buffer.

        Computes RLOO advantages at ingestion and stores individual rollouts
        with their precomputed advantages and usage tracking.
        """
        env_examples: dict[str, list[RolloutWithCount]] = defaultdict(list)
        current_time = time.time()
        reward_groups_seen = 0
        reward_groups_dropped_no_variance = 0
        soft_overlong_penalty_count = 0
        soft_overlong_penalty_sum = 0.0

        for batch in new_batches:
            if not batch.groups or not batch.groups[0].rollouts:
                continue

            # R[num_groups, num_rollouts_per_group]
            batch_rewards = np.zeros((len(batch.groups), len(batch.groups[0].rollouts)), dtype=np.float32)

            # Read weight_step from first rollout's metadata
            first_rollout = batch.groups[0].rollouts[0]
            rollout_step = first_rollout.metadata.weight_step
            rollout_timestamp = first_rollout.metadata.timestamp

            if not self._is_rollout_fresh(rollout_step, rollout_timestamp, self._current_step, current_time):
                logger.info(
                    f"Skipping stale rollout batch (rollout_step={rollout_step}, current_step={self._current_step})"
                )
                continue

            self._total_batches_added += 1

            for group_idx, group in enumerate(batch.groups):
                if not group.rollouts:
                    continue

                reward_groups_seen += 1
                shaped_rollouts = []
                for rollout in group.rollouts:
                    shaped_rollout = apply_soft_overlong_penalty(rollout, self.soft_overlong_penalty)
                    penalty = rollout.episode_reward - shaped_rollout.episode_reward
                    if penalty > 0.0:
                        soft_overlong_penalty_count += 1
                        soft_overlong_penalty_sum += penalty
                    shaped_rollouts.append(shaped_rollout)

                advantages = self.loss_module.compute_advantages(shaped_rollouts)
                maybe_used_rollouts = []
                for rollout_idx, (rollout, advantage) in enumerate(zip(shaped_rollouts, advantages, strict=True)):
                    individual = RolloutWithCount(
                        rollout=rollout, advantage=advantage, usage_count=0, weight_step=rollout_step
                    )
                    maybe_used_rollouts.append(individual)
                    batch_rewards[group_idx, rollout_idx] = rollout.episode_reward

                env_name = shaped_rollouts[0].env_name
                if np.std(batch_rewards[group_idx]) > 0.0:
                    env_examples[env_name].extend(maybe_used_rollouts)
                    continue

                if self.filter_out_groups_with_no_variance:
                    reward_groups_dropped_no_variance += 1
                else:
                    env_examples[env_name].extend(maybe_used_rollouts)

                logger.info(f"Group {group_idx} has no variance in rewards")

            logger.info(f"Reward mean across all groups: {batch_rewards.mean()}")
            logger.info(f"Reward std across all groups: {batch_rewards.std(axis=1).mean()}")

        with self._lock:
            self._total_reward_groups_seen += reward_groups_seen
            self._total_reward_groups_dropped_no_variance += reward_groups_dropped_no_variance
            self._total_soft_overlong_penalty_count += soft_overlong_penalty_count
            self._total_soft_overlong_penalty_sum += soft_overlong_penalty_sum

            for env_name, examples in env_examples.items():
                if env_name in self.rollout_storage:
                    self.rollout_storage[env_name].extend(examples)
                else:
                    self.rollout_storage[env_name] = examples

                if len(self.rollout_storage[env_name]) > self.capacity:
                    self.rollout_storage[env_name] = self.rollout_storage[env_name][-self.capacity :]

    def sample_rollouts(self) -> list[RolloutWithCount] | None:
        """Sample individual rollouts with balanced environment sampling.

        If no samples are available, returns None.

        Returns:
            List of IndividualRollout objects up to local_batch_size.
        """
        with self._lock:
            # Get all environments with rollouts
            env_names = [name for name, rollouts in self.rollout_storage.items() if rollouts]
            if not env_names:
                return None

            # sample environments N times without replacement
            # we fill the array with the number of rollouts in each env
            total_count = 0
            env_choices = []
            for env_name in env_names:
                env_choices.extend([env_name] * len(self.rollout_storage[env_name]))
                total_count += len(self.rollout_storage[env_name])

            # not enough samples to fill a batch
            if total_count < self.local_batch_size:
                return None

            env_choices = np.array(env_choices)
            env_indices = self._rng.choice(
                env_choices,
                size=self.local_batch_size,
                replace=False,
            )

            # count the number of times each env is chosen
            env_count = defaultdict(int)
            for env_name in env_indices:
                env_count[env_name] += 1

            # now sample from each env according to recency weights & number of times chosen
            sampled: list[RolloutWithCount] = []
            for env_name, count in env_count.items():
                rollouts = self.rollout_storage[env_name]
                weights = np.arange(len(rollouts)) + 1
                weights = weights**self.alpha
                weights = weights / weights.sum()

                # Sample rollout index
                idx = self._rng.choice(len(rollouts), p=weights, size=count, replace=False)
                for i in idx:
                    sampled.append(rollouts[i])
                    rollouts[i].usage_count += 1

            # Retire overused rollouts
            self._retire_overused_rollouts()

            # Update stats
            self._total_batches_sampled += 1
            return sampled

    def size(self) -> int:
        """Get total number of rollouts across all environments."""
        with self._lock:
            return sum(len(rollouts) for rollouts in self.rollout_storage.values())

    def get_stats(self) -> dict:
        """Get buffer statistics for monitoring."""
        with self._lock:
            env_sizes = {env: len(rollouts) for env, rollouts in self.rollout_storage.items()}
            reward_groups_accepted = self._total_reward_groups_seen - self._total_reward_groups_dropped_no_variance
            reward_group_acceptance_rate = (
                reward_groups_accepted / self._total_reward_groups_seen if self._total_reward_groups_seen > 0 else 0.0
            )
            soft_overlong_penalty_mean = (
                self._total_soft_overlong_penalty_sum / self._total_soft_overlong_penalty_count
                if self._total_soft_overlong_penalty_count > 0
                else 0.0
            )
            return {
                "total_size": sum(env_sizes.values()),
                "env_sizes": env_sizes,
                "num_environments": len(self.rollout_storage),
                "total_batches_added": self._total_batches_added,
                "total_batches_sampled": self._total_batches_sampled,
                "reward_groups_seen": self._total_reward_groups_seen,
                "reward_groups_dropped_no_variance": self._total_reward_groups_dropped_no_variance,
                "reward_group_acceptance_rate": reward_group_acceptance_rate,
                "soft_overlong_penalty_count": self._total_soft_overlong_penalty_count,
                "soft_overlong_penalty_mean": soft_overlong_penalty_mean,
            }


class ReplayDataLoader:
    """Loads data from rollout reader into replay buffer and provides training batches."""

    def __init__(
        self,
        rollout_reader: RolloutReader,
        replay_buffer: ReplayBuffer,
        rollout_fetch_interval: float = 1.0,
    ):
        """Initialize replay data loader.

        Args:
            rollout_reader: Reader to get rollout data from.
            replay_buffer: Buffer to store rollout data.
            batch_interval: Interval in seconds between data collection iterations.
        """
        self.rollout_reader = rollout_reader
        self.replay_buffer = replay_buffer
        self.rollout_fetch_interval = rollout_fetch_interval

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Start background thread for loading data."""
        if self._thread is not None:
            raise RuntimeError("ReplayDataLoader already running")

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()
        logger.info("Started ReplayDataLoader background thread")

    def stop(self) -> None:
        """Stop background thread."""
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join(timeout=5.0)
            self._thread = None
            logger.info("Stopped ReplayDataLoader background thread")

    def get_rollouts(self, timeout: float = 5.0) -> list[RolloutWithCount] | None:
        """Get next batch of rollouts from replay buffer.

        Args:
            timeout: Maximum time to wait for rollouts in seconds.

        Returns:
            List of IndividualRollout if available, None if timeout or no data.
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            rollouts = self.replay_buffer.sample_rollouts()
            if rollouts is not None:
                return rollouts
            time.sleep(0.1)
        return None

    def _worker_loop(self) -> None:
        """Main worker loop for background data loading."""
        while not self._stop_event.is_set():
            try:
                self._collect_rollouts()
            except Exception as e:
                logger.error(f"Error in ReplayDataLoader worker loop: {e}", exc_info=True)

            # Sleep via the stop event so shutdown is responsive instead of blocking the full interval.
            self._stop_event.wait(self.rollout_fetch_interval)

    def _collect_rollouts(self) -> None:
        """Collect available rollouts from reader and add to buffer."""
        batches = self.rollout_reader.read_all_available()

        if not batches:
            return

        start_time = time.time()
        self.replay_buffer.add_batches(batches)
        elapsed = time.time() - start_time
        logger.info(f"Collected {len(batches)} rollout batches, updated replay buffer in {elapsed:.3f}s")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
