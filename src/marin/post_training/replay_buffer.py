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

import numpy as np

from .rollout_storage import RolloutBatch, RolloutReader, TaggedRolloutBatch

logger = logging.getLogger(__name__)


class ReplayBuffer:
    """The replay buffer manages incoming rollout data and produces training batches.

    It attempts to prioritize recent data while balancing the number of samples from
    each environment. As rollout examples appear, they are folded into a shared
    rollout array which is sampled from to produce training batches.

    TODO:

    * Add back-pressure to reduce over-training when the number of new samples is low.
      This may be handled by either sampling without replacement or aging buffers out.
    """

    def __init__(
        self,
        local_batch_size: int,
        process_id: int,
        total_processes: int,
        recency_alpha: float,
        capacity: int,
    ):
        """Initialize replay buffer.

        Args:
            capacity: Maximum number of examples to store per environment.
            local_batch_size: Target size for training batches.
            recency_alpha: Power law exponent for recency weighting (higher = more recent bias).
            train_process_id: Identifier for this process. Used to sample across processes.
            num_train_processes: Total number of training processes.
        """
        self.capacity = capacity
        self.local_batch_size = local_batch_size
        self.recency_alpha = recency_alpha
        self.total_processes = total_processes
        self.process_id = process_id

        # Per-environment buffers for balanced sampling - store pure RolloutBatch
        self.env_buffers: dict[str, RolloutBatch] = {}
        self._lock = threading.Lock()

        # Statistics
        self._total_batches_added = 0
        self._total_batches_sampled = 0

        self._rng = np.random.default_rng(seed=process_id + 42)

    def add_batches(self, tagged_batches: list[TaggedRolloutBatch]) -> None:
        """Add tagged batchs to the replay buffer.

        The batch is added to the buffer corresponding to its environment name.
        If this environment has seen more than `capacity` batches, the oldest batch is discarded.

        Args:
            tagged_batch: TaggedRolloutBatch to add to the buffer.
        """

        # group batches by environment type
        batches_by_env: dict[str, list[RolloutBatch]] = {}
        for tagged_batch in tagged_batches:
            env_name = tagged_batch.env_name
            if env_name not in batches_by_env:
                batches_by_env[env_name] = []
            batches_by_env[env_name].append(tagged_batch.batch)

        with self._lock:
            for env_name, batches in batches_by_env.items():
                if env_name not in self.env_buffers:
                    buffer = RolloutBatch.concat(batches)
                    self.env_buffers[env_name] = buffer
                else:
                    buffer = self.env_buffers[env_name]
                    buffer = RolloutBatch.concat([buffer, *batches])

                env_size = len(buffer)
                if env_size > self.capacity:
                    buffer = buffer.slice(env_size - self.capacity, env_size)
                self.env_buffers[env_name] = buffer

                logger.info(
                    "Added batches to env '%s', new size: %d",
                    env_name,
                    len(self.env_buffers[env_name]),
                )
                self._total_batches_added += len(batches)

    def sample_training_batch(self) -> RolloutBatch | None:
        """Sample a training batch with balanced environment sampling.

        Returns:
            RolloutBatch containing as many examples as possible up to local_batch_size.
        """
        with self._lock:
            if len(self.env_buffers) == 0:
                return None

            # We sample first an environment, then an example from that environment.
            # We use a power-law distribution to prioritize recent examples.
            # We don't currently guarantee processes will fetch unique examples
            # but processes do sample independently.

            env_buffers = [b for b in self.env_buffers.values() if len(b) > 0]
            env_count = len(env_buffers)
            env_ids = np.arange(env_count)
            env_sample_weights = []
            for b in env_buffers:
                sample_weights = np.arange(len(b)) + 1
                sample_weights = sample_weights**self.recency_alpha
                sample_weights = sample_weights / sample_weights.sum()
                env_sample_weights.append(sample_weights)

            samples = []

            for _ in range(self.local_batch_size):
                env_idx = self._rng.choice(env_ids)
                env_buffer = env_buffers[env_idx]
                buffer_size = len(env_buffer)
                sample_idx = self._rng.choice(buffer_size, p=env_sample_weights[env_idx])
                samples.append(env_buffer.slice(sample_idx, sample_idx + 1))

            self._total_batches_sampled += 1
            return RolloutBatch.concat(samples)

    def size(self) -> int:
        """Get total number of batches across all environments."""
        with self._lock:
            return sum(len(buffer) for buffer in self.env_buffers.values())

    def get_stats(self) -> dict:
        """Get buffer statistics for monitoring."""
        with self._lock:
            env_sizes = {env: len(buffer) for env, buffer in self.env_buffers.items()}
            return {
                "total_size": sum(env_sizes.values()),
                "env_sizes": env_sizes,
                "num_environments": len(self.env_buffers),
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

    def get_training_batch(self, timeout: float = 5.0) -> RolloutBatch | None:
        """Get next training batch from replay buffer.

        Args:
            timeout: Maximum time to wait for a batch in seconds.

        Returns:
            RolloutBatch if available, None if timeout or no data.
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            batch = self.replay_buffer.sample_training_batch()
            if batch is not None:
                return batch
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
