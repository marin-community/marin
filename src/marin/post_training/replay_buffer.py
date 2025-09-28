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
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from .rollout_storage import RolloutReader
from .rollout_types import JaxRolloutBatch, Rollout, RolloutBatch

logger = logging.getLogger(__name__)


def _trim_and_pad(ary, max_input_length: int, max_output_length: int, pad_token_id: int):
    """Trim and pad array to max sequence length."""
    max_seq_len = max_input_length + max_output_length
    ary = ary[:max_seq_len]
    ary = np.pad(
        ary,
        (0, max_seq_len - len(ary)),
        mode="constant",
        constant_values=pad_token_id if ary.dtype == np.int32 else 0,
    )
    return ary


def _convert_to_training_format(
    rollout: Rollout, advantage: float, max_input_length: int, max_output_length: int, pad_token_id: int
) -> dict:
    """Convert a single rollout to training format with advantage."""
    full_tokens = np.concatenate([rollout.prompt_tokens, rollout.response_tokens])
    full_mask = np.ones(len(full_tokens))
    full_position_ids = np.maximum(np.cumsum(full_mask) - 1, 0)

    # Shifted for next-token prediction
    input_tokens = full_tokens[:-1]
    input_attention_mask = full_mask[:-1]
    target_tokens = full_tokens[1:]
    position_ids = full_position_ids[:-1]

    # Loss mask (only on response tokens)
    loss_mask = np.concatenate(
        [
            np.zeros(len(rollout.prompt_tokens) - 1, dtype=np.float32),
            np.ones(len(rollout.response_tokens), dtype=np.float32),
        ]
    )

    # Loss weights (advantage for all response tokens)
    loss_weight = np.concatenate(
        [
            np.zeros(len(rollout.prompt_tokens) - 1, dtype=np.float32),
            np.full(len(rollout.response_tokens), advantage, dtype=np.float32),
        ]
    )

    # Policy logprobs
    policy_logprob = np.concatenate(
        [np.zeros(len(rollout.prompt_tokens) - 1, dtype=np.float32), rollout.response_logprobs.astype(np.float32)]
    )

    return {
        "input_ids": _trim_and_pad(input_tokens, max_input_length, max_output_length, pad_token_id),
        "attention_mask": _trim_and_pad(input_attention_mask, max_input_length, max_output_length, pad_token_id),
        "position_ids": _trim_and_pad(position_ids, max_input_length, max_output_length, pad_token_id),
        "target_ids": _trim_and_pad(target_tokens, max_input_length, max_output_length, pad_token_id),
        "loss_weights": _trim_and_pad(loss_weight, max_input_length, max_output_length, pad_token_id),
        "loss_masks": _trim_and_pad(loss_mask, max_input_length, max_output_length, pad_token_id),
        "policy_logprobs": _trim_and_pad(policy_logprob, max_input_length, max_output_length, pad_token_id),
    }


def _stack_training_examples(examples: list[dict]) -> JaxRolloutBatch:
    """Stack training examples into a JAX batch."""
    stacked = {}
    for key in examples[0].keys():
        stacked[key] = jnp.stack([ex[key] for ex in examples], axis=0)

    return JaxRolloutBatch(
        input_ids=stacked["input_ids"],
        attention_mask=stacked["attention_mask"],
        position_ids=stacked["position_ids"],
        target_ids=stacked["target_ids"],
        loss_weights=stacked["loss_weights"],
        loss_masks=stacked["loss_masks"],
        policy_logprobs=stacked["policy_logprobs"],
    )


@dataclass
class IndividualRollout:
    """Single rollout with precomputed RLOO advantage."""

    rollout: Rollout
    advantage: float
    usage_count: int = 0


class ReplayBuffer:
    """The replay buffer manages incoming rollout data and produces training batches.

    It attempts to prioritize recent data while balancing the number of samples from
    each environment. As rollout examples appear, they are folded into a shared
    rollout array which is sampled from to produce training batches.
    """

    def __init__(
        self,
        local_batch_size: int,
        process_id: int,
        total_processes: int,
        recency_alpha: float,
        capacity: int,
        max_samples: int = -1,
        max_input_length: int = 512,
        max_output_length: int = 512,
        pad_token_id: int = 0,
    ):
        """Initialize replay buffer.

        Args:
            capacity: Maximum number of examples to store per environment.
            local_batch_size: Target size for training batches.
            recency_alpha: Power law exponent for recency weighting (higher = more recent bias).
            train_process_id: Identifier for this process. Used to sample across processes.
            num_train_processes: Total number of training processes.
            max_samples: Maximum number of times an example can be used before being retired.
            max_input_length: Maximum input sequence length for padding.
            max_output_length: Maximum output sequence length for padding.
            pad_token_id: Token ID to use for padding.

        A `max_samples` of -1 indicates no limit, 0 or 1 means each example is used at most once.
        """
        self.capacity = capacity
        self.local_batch_size = local_batch_size
        self.recency_alpha = recency_alpha
        self.total_processes = total_processes
        self.process_id = process_id
        self.max_samples = max_samples
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.pad_token_id = pad_token_id

        self.rollout_storage: dict[str, list[IndividualRollout]] = {}
        self._lock = threading.Lock()

        self._total_batches_added = 0
        self._total_batches_sampled = 0

        self._rng = np.random.default_rng(seed=process_id + 42)

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
        with self._lock:
            for batch in new_batches:
                for group in batch.groups:
                    # Compute RLOO advantages for the group
                    advantages = group.compute_rloo_advantages()

                    # Store each rollout individually with its advantage
                    for rollout, advantage in zip(group.rollouts, advantages, strict=False):
                        individual = IndividualRollout(rollout=rollout, advantage=advantage, usage_count=0)

                        env_name = rollout.env_name
                        if env_name not in self.rollout_storage:
                            self.rollout_storage[env_name] = []

                        self.rollout_storage[env_name].append(individual)

                        # Apply capacity limit per environment (keep most recent)
                        if len(self.rollout_storage[env_name]) > self.capacity:
                            self.rollout_storage[env_name] = self.rollout_storage[env_name][-self.capacity :]

                logger.info("Added batch with %d groups to replay buffer", len(batch.groups))
                self._total_batches_added += 1

    def sample_training_batch(self) -> JaxRolloutBatch | None:
        """Sample a training batch from individual rollouts with balanced environment sampling.

        If no samples are available, returns None.

        Returns:
            JaxRolloutBatch containing as many examples as possible up to local_batch_size.
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

                # Convert to training format with precomputed advantage
                training_example = _convert_to_training_format(
                    individual.rollout,
                    individual.advantage,
                    self.max_input_length,
                    self.max_output_length,
                    self.pad_token_id,
                )
                sampled.append(training_example)

                # Update usage count
                individual.usage_count += 1

            # Retire overused rollouts
            self._retire_overused_rollouts()

            # Stack into batch
            self._total_batches_sampled += 1
            return _stack_training_examples(sampled)

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

    def get_training_batch(self, timeout: float = 5.0) -> JaxRolloutBatch | None:
        """Get next training batch from replay buffer.

        Args:
            timeout: Maximum time to wait for a batch in seconds.

        Returns:
            JaxRolloutBatch if available, None if timeout or no data.
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
