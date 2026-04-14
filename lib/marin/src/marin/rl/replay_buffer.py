# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Replay buffer for RL training data.

Replay is responsible for storing, filtering, and sampling neutral trajectory
records. It intentionally does not compute objective-specific signals such as
advantages.
"""

import dataclasses
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .rollout_storage import RolloutReader
from .train_batch import rollout_group_to_trajectory_group_record
from .types import RolloutBatch, TrajectoryGroupRecord, TrajectoryRecord

logger = logging.getLogger(__name__)


def _trajectory_key(trajectory: TrajectoryRecord) -> tuple[str, str, str]:
    return (trajectory.group_id, trajectory.env_example_id, trajectory.trace_id)


@dataclass
class ReplayBufferConfig:
    """Configuration for the replay buffer."""

    capacity: int
    """Maximum number of active trajectories per environment in the buffer."""

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


@dataclass
class StoredTrajectory:
    """Neutral replay entry for one trajectory with retained group context."""

    trajectory: TrajectoryRecord
    group: TrajectoryGroupRecord
    usage_count: int = 0

    @property
    def env_name(self) -> str:
        return self.trajectory.env_name

    @property
    def group_id(self) -> str:
        return self.trajectory.group_id

    @property
    def weight_step(self) -> int:
        return self.trajectory.rollout_metadata.weight_step

    @property
    def timestamp(self) -> float:
        return self.trajectory.rollout_metadata.timestamp


@dataclass
class ReplayBuffer:
    """Store and sample neutral trajectories for trainer-side objective runtimes."""

    capacity: int
    local_batch_size: int
    alpha: float
    total_processes: int
    max_samples: int
    max_rollout_step_delay: int
    max_rollout_timestamp_delay: float
    filter_out_groups_with_no_variance: bool
    seed: int

    _total_batches_added: int = 0
    _total_batches_sampled: int = 0
    _lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)
    _current_step: int = 0
    _rng: np.random.Generator = dataclasses.field(init=False)
    rollout_storage: dict[str, list[StoredTrajectory]] = dataclasses.field(default_factory=dict)
    _group_records: dict[str, TrajectoryGroupRecord] = dataclasses.field(default_factory=dict)
    _group_active_counts: dict[str, int] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        self._rng = np.random.default_rng(seed=self.seed)

    @classmethod
    def from_config(
        cls,
        config: ReplayBufferConfig,
        local_batch_size: int,
        total_processes: int,
        seed: int,
    ) -> "ReplayBuffer":
        """Create ReplayBuffer from configuration."""
        return cls(
            capacity=config.capacity,
            local_batch_size=local_batch_size,
            alpha=config.alpha,
            total_processes=total_processes,
            max_samples=config.max_samples,
            max_rollout_step_delay=config.max_rollout_step_delay,
            max_rollout_timestamp_delay=config.max_rollout_timestamp_delay,
            filter_out_groups_with_no_variance=config.filter_out_groups_with_no_variance,
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

        min_time = current_time - self.max_rollout_timestamp_delay
        if rollout_timestamp <= min_time:
            return False

        return True

    def _refresh_group_records(self, group_ids: set[str]) -> None:
        for group_id in group_ids:
            group_record = self._group_records.get(group_id)
            if group_record is None:
                continue

            active_members: list[StoredTrajectory] = []
            for rollouts in self.rollout_storage.values():
                for stored_trajectory in rollouts:
                    if stored_trajectory.group_id == group_id:
                        active_members.append(stored_trajectory)

            if not active_members:
                self._group_active_counts.pop(group_id, None)
                self._group_records.pop(group_id, None)
                continue

            active_members_by_key = {
                _trajectory_key(stored_trajectory.trajectory): stored_trajectory for stored_trajectory in active_members
            }
            active_trajectories = tuple(
                stored_trajectory.trajectory
                for trajectory in group_record.trajectories
                if (stored_trajectory := active_members_by_key.get(_trajectory_key(trajectory))) is not None
            )
            refreshed_group = dataclasses.replace(group_record, trajectories=active_trajectories)
            self._group_records[group_id] = refreshed_group
            self._group_active_counts[group_id] = len(active_trajectories)
            for stored_trajectory in active_members:
                stored_trajectory.group = refreshed_group

    def _evict_over_capacity(self, env_name: str) -> None:
        rollouts = self.rollout_storage[env_name]
        overflow = len(rollouts) - self.capacity
        if overflow <= 0:
            return

        removed = rollouts[:overflow]
        self.rollout_storage[env_name] = rollouts[overflow:]
        self._refresh_group_records({stored_trajectory.group_id for stored_trajectory in removed})

    def _retire_overused_trajectories(self) -> None:
        """Remove trajectories that exceeded max_samples usage."""
        if self.max_samples < 0:
            return

        affected_group_ids: set[str] = set()
        for env_name, rollouts in self.rollout_storage.items():
            kept_rollouts = []
            for stored_trajectory in rollouts:
                if stored_trajectory.usage_count < self.max_samples:
                    kept_rollouts.append(stored_trajectory)
                    continue
                affected_group_ids.add(stored_trajectory.group_id)
            self.rollout_storage[env_name] = kept_rollouts
        self._refresh_group_records(affected_group_ids)

    def _ordered_full_group_ids_for_env(self, env_name: str) -> list[str]:
        latest_index: dict[str, int] = {}
        for index, stored_trajectory in enumerate(self.rollout_storage.get(env_name, [])):
            group_id = stored_trajectory.group_id
            group_record = self._group_records.get(group_id)
            if group_record is None:
                continue
            if self._group_active_counts.get(group_id) != len(group_record.trajectories):
                continue
            latest_index[group_id] = index

        return [group_id for group_id, _ in sorted(latest_index.items(), key=lambda item: item[1])]

    def _increment_group_usage(self, group_id: str) -> None:
        for rollouts in self.rollout_storage.values():
            for stored_trajectory in rollouts:
                if stored_trajectory.group_id == group_id:
                    stored_trajectory.usage_count += 1

    def set_current_step(self, step: int) -> None:
        """Set current training step and filter stale rollouts."""
        self._current_step = step
        current_time = time.time()

        with self._lock:
            total_removed = 0
            affected_group_ids: set[str] = set()
            for env_name, rollouts in self.rollout_storage.items():
                kept_rollouts = []
                for stored_trajectory in rollouts:
                    if self._is_rollout_fresh(
                        stored_trajectory.weight_step,
                        stored_trajectory.timestamp,
                        step,
                        current_time,
                    ):
                        kept_rollouts.append(stored_trajectory)
                        continue

                    total_removed += 1
                    affected_group_ids.add(stored_trajectory.group_id)

                self.rollout_storage[env_name] = kept_rollouts

            self._refresh_group_records(affected_group_ids)

            total_remaining = sum(len(rollouts) for rollouts in self.rollout_storage.values())
            if total_removed > 0:
                logger.info("Filtered %d stale rollouts %d remaining", total_removed, total_remaining)

    def add_batches(self, new_batches: list[RolloutBatch]) -> None:
        """Add new rollout batches into the replay buffer without computing objectives."""
        env_examples: dict[str, list[StoredTrajectory]] = defaultdict(list)
        group_records_to_store: dict[str, TrajectoryGroupRecord] = {}
        group_active_increments: dict[str, int] = defaultdict(int)
        reward_arrays: list[np.ndarray] = []
        current_time = time.time()

        for batch in new_batches:
            accepted_any_group = False
            for group_idx, group in enumerate(batch.groups):
                if not group.rollouts:
                    continue

                group_record = rollout_group_to_trajectory_group_record(group)
                first_trajectory = group_record.trajectories[0]
                rollout_step = first_trajectory.rollout_metadata.weight_step
                rollout_timestamp = first_trajectory.rollout_metadata.timestamp

                if not self._is_rollout_fresh(rollout_step, rollout_timestamp, self._current_step, current_time):
                    logger.info(
                        "Skipping stale rollout group (rollout_step=%d, current_step=%d)",
                        rollout_step,
                        self._current_step,
                    )
                    continue

                rewards = np.asarray(
                    [trajectory.episode_reward for trajectory in group_record.trajectories],
                    dtype=np.float32,
                )
                if rewards.size == 0:
                    continue

                if np.std(rewards) == 0.0:
                    logger.info("Group %d has no variance in rewards", group_idx)
                    if self.filter_out_groups_with_no_variance:
                        continue

                env_name = first_trajectory.env_name
                env_examples[env_name].extend(
                    StoredTrajectory(trajectory=trajectory, group=group_record)
                    for trajectory in group_record.trajectories
                )
                group_records_to_store[group_record.group_id] = group_record
                group_active_increments[group_record.group_id] += len(group_record.trajectories)
                reward_arrays.append(rewards)
                accepted_any_group = True

            if accepted_any_group:
                self._total_batches_added += 1

        with self._lock:
            for group_id, group_record in group_records_to_store.items():
                self._group_records[group_id] = group_record
                self._group_active_counts[group_id] = (
                    self._group_active_counts.get(group_id, 0) + group_active_increments[group_id]
                )

            for env_name, examples in env_examples.items():
                if env_name in self.rollout_storage:
                    self.rollout_storage[env_name].extend(examples)
                else:
                    self.rollout_storage[env_name] = list(examples)
                self._evict_over_capacity(env_name)

        if reward_arrays:
            all_rewards = np.concatenate(reward_arrays)
            per_group_std = np.asarray([rewards.std() for rewards in reward_arrays], dtype=np.float32)
            logger.info("Reward mean across all groups: %s", float(all_rewards.mean()))
            logger.info("Reward std across all groups: %s", float(per_group_std.mean()))

    def sample_sequences(self) -> list[StoredTrajectory] | None:
        """Sample neutral trajectories with balanced environment sampling."""
        with self._lock:
            env_names = [name for name, rollouts in self.rollout_storage.items() if rollouts]
            if not env_names:
                return None

            total_count = 0
            env_choices = []
            for env_name in env_names:
                env_choices.extend([env_name] * len(self.rollout_storage[env_name]))
                total_count += len(self.rollout_storage[env_name])

            if total_count < self.local_batch_size:
                return None

            env_choices = np.asarray(env_choices)
            env_indices = self._rng.choice(env_choices, size=self.local_batch_size, replace=False)

            env_count = defaultdict(int)
            for env_name in env_indices:
                env_count[str(env_name)] += 1

            sampled: list[StoredTrajectory] = []
            for env_name, count in env_count.items():
                rollouts = self.rollout_storage[env_name]
                weights = np.arange(len(rollouts)) + 1
                weights = weights**self.alpha
                weights = weights / weights.sum()
                indices = self._rng.choice(len(rollouts), p=weights, size=count, replace=False)
                for index in indices:
                    sampled.append(rollouts[index])
                    rollouts[index].usage_count += 1

            self._retire_overused_trajectories()
            self._total_batches_sampled += 1
            return sampled

    def sample_groups(self, min_trajectories: int | None = None) -> list[TrajectoryGroupRecord] | None:
        """Sample full trajectory groups, preserving prompt grouping."""
        target_trajectories = self.local_batch_size if min_trajectories is None else min_trajectories
        with self._lock:
            ordered_full_group_ids_for_env = self._ordered_full_group_ids_for_env
            available_groups_by_env = {
                env_name: ordered_full_group_ids_for_env(env_name) for env_name in self.rollout_storage
            }
            total_available = sum(
                len(self._group_records[group_id].trajectories)
                for group_ids in available_groups_by_env.values()
                for group_id in group_ids
            )
            if total_available < target_trajectories:
                return None

            sampled_groups: list[TrajectoryGroupRecord] = []
            sampled_trajectories = 0

            while sampled_trajectories < target_trajectories:
                available_envs = [env_name for env_name, group_ids in available_groups_by_env.items() if group_ids]
                if not available_envs:
                    return None

                env_choices = []
                for env_name in available_envs:
                    for group_id in available_groups_by_env[env_name]:
                        env_choices.extend([env_name] * len(self._group_records[group_id].trajectories))

                chosen_env = str(self._rng.choice(np.asarray(env_choices)))
                group_ids = available_groups_by_env[chosen_env]
                weights = np.arange(len(group_ids)) + 1
                weights = weights**self.alpha
                weights = weights / weights.sum()
                group_index = int(self._rng.choice(len(group_ids), p=weights))
                group_id = group_ids.pop(group_index)
                group_record = self._group_records[group_id]
                sampled_groups.append(group_record)
                sampled_trajectories += len(group_record.trajectories)
                self._increment_group_usage(group_id)

            self._retire_overused_trajectories()
            self._total_batches_sampled += 1
            return sampled_groups

    def sample_rollouts(self) -> list[StoredTrajectory] | None:
        """Compatibility alias for the old replay API name."""
        return self.sample_sequences()

    def size(self) -> int:
        """Get total number of active trajectories across all environments."""
        with self._lock:
            return sum(len(rollouts) for rollouts in self.rollout_storage.values())

    def get_stats(self) -> dict:
        """Get buffer statistics for monitoring."""
        with self._lock:
            env_sizes = {env: len(rollouts) for env, rollouts in self.rollout_storage.items()}
            return {
                "total_size": sum(env_sizes.values()),
                "env_sizes": env_sizes,
                "num_environments": len(self.rollout_storage),
                "num_active_groups": len(self._group_records),
                "total_batches_added": self._total_batches_added,
                "total_batches_sampled": self._total_batches_sampled,
            }


class ReplayDataLoader:
    """Loads data from rollout reader into replay buffer and provides neutral samples."""

    def __init__(
        self,
        rollout_reader: RolloutReader,
        replay_buffer: ReplayBuffer,
        rollout_fetch_interval: float = 1.0,
    ):
        """Initialize replay data loader."""
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

    def get_trajectories(self, timeout: float = 5.0) -> list[StoredTrajectory] | None:
        """Get the next sampled neutral trajectories from replay."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            trajectories = self.replay_buffer.sample_sequences()
            if trajectories is not None:
                return trajectories
            time.sleep(0.1)
        return None

    def get_groups(self, timeout: float = 5.0) -> list[TrajectoryGroupRecord] | None:
        """Get the next sampled full trajectory groups from replay."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            groups = self.replay_buffer.sample_groups()
            if groups is not None:
                return groups
            time.sleep(0.1)
        return None

    def get_rollouts(self, timeout: float = 5.0) -> list[StoredTrajectory] | None:
        """Compatibility alias for the old loader API name."""
        return self.get_trajectories(timeout=timeout)

    def _worker_loop(self) -> None:
        """Main worker loop for background data loading."""
        while not self._stop_event.is_set():
            try:
                self._collect_rollouts()
            except Exception as exc:
                logger.error("Error in ReplayDataLoader worker loop: %s", exc, exc_info=True)

            self._stop_event.wait(self.rollout_fetch_interval)

    def _collect_rollouts(self) -> None:
        """Collect available rollouts from reader and add to buffer."""
        batches = self.rollout_reader.read_all_available()
        if not batches:
            return

        start_time = time.time()
        self.replay_buffer.add_batches(batches)
        elapsed = time.time() - start_time
        logger.info("Collected %d rollout batches, updated replay buffer in %s", len(batches), elapsed)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
