# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Replay buffer for RL training data.

This module provides a replay buffer that manages rollout data for training,
with balanced sampling across environments and configurable prioritization.
"""

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .rl_losses import compute_rloo_advantages
from .rollout_storage import RolloutReader
from .types import RolloutBatch, RolloutWithAdvantage

logger = logging.getLogger(__name__)

# TODO(power) - move advantage calculation and separate it from count


@dataclass
class ReplayBufferConfig:
    """Configuration for the replay buffer."""

    capacity: int = 10000
    """Maximum number of examples per environment in the buffer."""

    alpha: float = 3.0
    """Recency bias for sampling, higher values favor newer examples."""

    max_samples: int = 4
    """Maximum number of times to use an example before retiring."""

    max_rollout_delay: int = 1000
    """Maximum age of rollouts in training steps."""


@dataclass
class RolloutWithCount(RolloutWithAdvantage):
    """Single rollout with precomputed RLOO advantage & usage count tracking."""

    usage_count: int = 0
    weight_step: int = 0


class ReplayBuffer:
    """The replay buffer manages incoming rollout data and produces training batches.

    It attempts to prioritize recent data while balancing the number of samples from
    each environment. As rollout examples appear, they are folded into a shared
    rollout array which is sampled from to produce training batches.
    """

    def __init__(
        self,
        config: ReplayBufferConfig,
        local_batch_size: int,
        process_id: int,
        total_processes: int,
    ):
        """Initialize replay buffer.

        Args:
            config: Configuration for buffer capacity, recency bias, and sampling limits.
            local_batch_size: Target size for training batches.
            process_id: Identifier for this process. Used to sample across processes.
            total_processes: Total number of training processes.
        """
        self.capacity = config.capacity
        self.local_batch_size = local_batch_size
        self.recency_alpha = config.alpha
        self.total_processes = total_processes
        self.process_id = process_id
        self.max_samples = config.max_samples
        self.max_rollout_delay = config.max_rollout_delay

        self.rollout_storage: dict[str, list[RolloutWithCount]] = {}
        self._lock = threading.Lock()

        self._total_batches_added = 0
        self._total_batches_sampled = 0
        self._current_step: int = 0

        self._rng = np.random.default_rng(seed=process_id + 42)

    def set_current_step(self, step: int) -> None:
        """Set current training step and filter stale rollouts."""
        self._current_step = step
        min_step = step - self.max_rollout_delay

        with self._lock:
            total_removed = 0
            for env_name in self.rollout_storage:
                rollouts = self.rollout_storage[env_name]
                before = len(rollouts)
                self.rollout_storage[env_name] = [r for r in rollouts if r.weight_step >= min_step]
                total_removed += before - len(self.rollout_storage[env_name])

            total_remaining = sum(len(rollouts) for rollouts in self.rollout_storage.values())

            if total_removed > 0:
                logger.info(
                    f"Filtered {total_removed} stale rollouts (min_step={min_step}), " f"{total_remaining} remaining"
                )

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
        for batch in new_batches:
            self._total_batches_added += 1
            weight_step = batch.metadata.weight_step
            for group in batch.groups:
                # Compute RLOO advantages for the group
                advantages = compute_rloo_advantages(group.rollouts)
                for rollout, advantage in zip(group.rollouts, advantages, strict=True):
                    individual = RolloutWithCount(
                        rollout=rollout, advantage=advantage, usage_count=0, weight_step=weight_step
                    )
                    env_examples[rollout.env_name].append(individual)

        with self._lock:
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

            # Sample individual rollouts
            sampled = []
            for _ in range(self.local_batch_size):
                # Select environment (balanced sampling)
                env_name = self._rng.choice(env_names)
                rollouts = self.rollout_storage[env_name]

                # Compute recency weights
                weights = np.arange(len(rollouts)) + 1
                weights = weights**self.recency_alpha
                weights = weights / weights.sum()

                # Sample rollout index
                idx = self._rng.choice(len(rollouts), p=weights)
                individual = rollouts[idx]

                sampled.append(individual)

                # Update usage count
                individual.usage_count += 1

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
            return {
                "total_size": sum(env_sizes.values()),
                "env_sizes": env_sizes,
                "num_environments": len(self.rollout_storage),
                "total_batches_added": self._total_batches_added,
                "total_batches_sampled": self._total_batches_sampled,
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
                time.sleep(self.rollout_fetch_interval)
            except Exception as e:
                logger.error(f"Error in ReplayDataLoader worker loop: {e}", exc_info=True)

            self._stop_event.wait(self.rollout_fetch_interval)

    def _collect_rollouts(self) -> None:
        """Collect available rollouts from reader and add to buffer."""
        batches = self.rollout_reader.read_all_available()

        if not batches:
            return

        start_time = time.time()

        self.replay_buffer.add_batches(batches)

        elapsed = time.time() - start_time
        if batches:
            logger.info(f"Collected {len(batches)} rollout batches, updated replay buffer in {elapsed}")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
