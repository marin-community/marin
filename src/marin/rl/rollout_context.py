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
Rollout context for managing environments and rollout generation.

This module provides a clean separation between rollout generation logic
and infrastructure concerns (JAX setup, weight transfer, etc.).
"""

import logging
import os
import socket
import time
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax.random as jrandom
from transformers import PreTrainedTokenizer

from marin.rl.curriculum import CurriculumConfig
from marin.rl.environments import MarinEnv
from marin.rl.environments.base import EnvConfig, load_environment_from_spec
from marin.rl.inference_ctx import InferenceContext
from marin.rl.types import (
    RolloutBatch,
    RolloutGroup,
    RolloutMetadata,
    RolloutStats,
)

logger = logging.getLogger(__name__)


# ===== Stateless utility functions =====


def compute_rollout_stats(rollout_batch: RolloutBatch, lesson_id: str) -> list[RolloutStats]:
    """Extract rollout statistics from a batch.

    Args:
        rollout_batch: Batch of rollouts
        lesson_id: ID of the lesson/environment

    Returns:
        List of RolloutStats for each rollout
    """
    stats = []
    for group in rollout_batch.groups:
        for rollout in group.rollouts:
            stats.append(
                RolloutStats(
                    lesson_id=lesson_id,
                    episode_reward=rollout.episode_reward,
                    env_example_id=rollout.env_example_id,
                )
            )
    return stats


@dataclass
class BatchMetrics:
    """Metrics computed from a rollout batch."""

    total_count: int
    success_count: int
    avg_reward: float
    rollout_stats: list[RolloutStats]


def compute_batch_metrics(rollout_batch: RolloutBatch, lesson_id: str) -> BatchMetrics:
    """Compute metrics from a rollout batch.

    Args:
        rollout_batch: Batch of rollouts
        lesson_id: ID of the lesson/environment

    Returns:
        BatchMetrics containing aggregated statistics
    """
    rollout_stats = compute_rollout_stats(rollout_batch, lesson_id)

    total_count = len(rollout_stats)
    success_count = sum(1 for stat in rollout_stats if stat.episode_reward > 0)
    reward_sum = sum(stat.episode_reward for stat in rollout_stats)

    return BatchMetrics(
        total_count=total_count,
        success_count=success_count,
        avg_reward=(reward_sum / total_count) if total_count > 0 else 0.0,
        rollout_stats=rollout_stats,
    )


def build_eval_metrics(prefix: str, lesson_id: str, batch_metrics: BatchMetrics) -> dict[str, Any]:
    """Build evaluation metrics dictionary for logging.

    Args:
        prefix: Metric prefix (e.g., "rollout", "inference.eval")
        lesson_id: ID of the lesson/environment
        batch_metrics: Computed batch metrics

    Returns:
        Dictionary of metrics ready for logging
    """
    if batch_metrics.total_count == 0:
        return {}

    success_rate = batch_metrics.success_count / batch_metrics.total_count
    return {
        f"{prefix}/{lesson_id}/success_rate": success_rate,
        f"{prefix}/{lesson_id}/avg_reward": batch_metrics.avg_reward,
        f"{prefix}/{lesson_id}/total_count": batch_metrics.total_count,
    }


def format_sample_for_logging(
    rollout_batch: RolloutBatch,
    tokenizer: PreTrainedTokenizer,
) -> dict[str, str]:
    """Format a representative sample from a batch for logging.

    Args:
        rollout_batch: Batch containing rollouts
        tokenizer: Tokenizer for decoding

    Returns:
        Dictionary with decoded prompt/response for logging
    """
    if not rollout_batch or not rollout_batch.groups:
        return {}

    # Take first rollout as representative
    sample = rollout_batch.groups[0].rollouts[0]

    prompt_text = tokenizer.decode(sample.prompt_tokens, skip_special_tokens=True)
    response_text = tokenizer.decode(sample.response_tokens, skip_special_tokens=True)

    return {
        "sample_prompt": prompt_text,
        "sample_response": response_text,
        "sample_example_id": sample.env_example_id,
    }


# ===== RolloutContext class =====


class RolloutContext:
    """Context for managing environments and generating rollouts.

    This class encapsulates the core rollout generation logic without
    infrastructure dependencies like JAX device mesh, weight transfer,
    or rollout storage.

    Attributes:
        tokenizer: Tokenizer instance
        curriculum_config: Configuration for curriculum (optional)
        seed: Random seed for reproducibility
        worker_id: Unique identifier for this context
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        curriculum_config: CurriculumConfig | None = None,
        seed: int = 42,
        worker_id: str | None = None,
    ):
        """Initialize the rollout context.

        Args:
            tokenizer: Tokenizer for the model
            curriculum_config: Optional curriculum configuration
            seed: Random seed
            worker_id: Worker identifier (auto-generated if not provided)
        """
        self.tokenizer = tokenizer
        self.curriculum_config = curriculum_config
        self.seed = seed
        self.worker_id = worker_id or f"{socket.gethostname()}_{os.getpid()}"

        self._environments: dict[str, MarinEnv] = {}
        self._rng = jrandom.PRNGKey(seed)
        self._current_weight_step = 0

    def load_environment(self, env_config: EnvConfig | str) -> MarinEnv:
        """Load an environment from configuration or lesson ID.

        Args:
            env_config: Either an EnvConfig or a lesson ID string

        Returns:
            Loaded environment instance

        Raises:
            ValueError: If lesson ID is provided but no curriculum config exists
        """
        if isinstance(env_config, str):
            # It's a lesson ID
            lesson_id = env_config
            if lesson_id in self._environments:
                return self._environments[lesson_id]

            if not self.curriculum_config:
                raise ValueError(f"Cannot load lesson '{lesson_id}' without curriculum config")

            lesson_config = self.curriculum_config.lessons.get(lesson_id)
            if not lesson_config:
                raise ValueError(f"Unknown lesson: {lesson_id}")

            env = load_environment_from_spec(lesson_config.env_config)
            self._environments[lesson_id] = env
            return env
        else:
            # Direct env config
            env_key = f"{env_config.env_class}_{id(env_config)}"
            if env_key in self._environments:
                return self._environments[env_key]

            env = load_environment_from_spec(env_config)
            self._environments[env_key] = env
            return env

    def get_loaded_environments(self) -> dict[str, MarinEnv]:
        """Get all currently loaded environments.

        Returns:
            Dictionary mapping environment IDs to instances
        """
        return dict(self._environments)

    def sample_rollouts(
        self,
        inference_ctx: InferenceContext,
        env_or_lesson_id: MarinEnv | EnvConfig | str,
        n_examples: int,
        n_generations: int,
        temperature: float,
        mode: str = "train",
        weight_step: int | None = None,
        stop_tokens: list[int] | None = None,
        max_tokens: int | None = None,
    ) -> tuple[RolloutBatch | None, dict[str, Any] | None]:
        """Sample rollouts from an environment.

        Args:
            inference_ctx: Inference context for generation
            env_or_lesson_id: Environment instance, config, or lesson ID
            n_examples: Number of examples to sample
            n_generations: Number of generations per example
            temperature: Sampling temperature
            mode: "train" or "eval"
            weight_step: Optional weight step for metadata
            stop_tokens: Optional stop tokens (from lesson config if using curriculum)
            max_tokens: Optional max tokens (from lesson config if using curriculum)

        Returns:
            Tuple of (rollout_batch, metrics) or (None, None) if no rollouts
        """
        # Get environment
        if isinstance(env_or_lesson_id, MarinEnv):
            env = env_or_lesson_id
            env_name = env.__class__.__name__
            # Store direct environment instances so they can be accessed later
            env_key = f"{env_name}_{id(env)}"
            self._environments[env_key] = env
        else:
            env = self.load_environment(env_or_lesson_id)
            env_name = env_or_lesson_id if isinstance(env_or_lesson_id, str) else env.__class__.__name__

        # Get sampling params from lesson config if available
        if isinstance(env_or_lesson_id, str) and self.curriculum_config:
            lesson_config = self.curriculum_config.lessons[env_or_lesson_id]
            if stop_tokens is None:
                stop_tokens = lesson_config.sampling_params.stop_tokens
            if max_tokens is None:
                max_tokens = lesson_config.sampling_params.max_tokens

        # Update inference context if needed
        if stop_tokens is not None and hasattr(inference_ctx, "_stop_tokens"):
            inference_ctx._stop_tokens = stop_tokens
        if max_tokens is not None:
            inference_ctx.max_tokens = max_tokens

        # Split RNG
        self._rng, sample_rng = jrandom.split(self._rng)

        # Sample from environment
        rollout_groups, metrics = env.sample(
            inference_ctx=inference_ctx,
            n_examples=n_examples,
            n_generations=n_generations,
            temperature=temperature,
            prng_key=sample_rng,
            mode=mode,
        )

        if not rollout_groups:
            logger.warning(f"No valid rollouts generated from {env_name}")
            return None, None

        # Use provided weight_step or internal counter
        if weight_step is None:
            weight_step = self._current_weight_step

        # Create metadata
        batch_metadata = RolloutMetadata(
            worker_id=self.worker_id,
            timestamp=time.time(),
            weight_step=weight_step,
        )

        # Attach metadata to rollouts
        rollout_groups_with_metadata = []
        for group in rollout_groups:
            rollouts_with_metadata = []
            for rollout in group.rollouts:
                rollout_with_meta = eqx.tree_at(lambda r: r.metadata, rollout, batch_metadata)
                rollouts_with_metadata.append(rollout_with_meta)
            rollout_groups_with_metadata.append(RolloutGroup(rollouts=rollouts_with_metadata))

        rollout_batch = RolloutBatch(
            groups=rollout_groups_with_metadata,
            metadata=batch_metadata,
        )

        logger.info(f"Generated {len(rollout_groups)} rollout groups from {env_name} " f"at weight step {weight_step}")

        return rollout_batch, metrics

    def evaluate_lesson(
        self,
        inference_ctx: InferenceContext,
        lesson_id: str,
        n_examples: int,
    ) -> tuple[BatchMetrics, dict[str, Any]]:
        """Evaluate a single lesson and compute metrics.

        Args:
            inference_ctx: Inference context for generation
            lesson_id: ID of the lesson to evaluate
            n_examples: Number of examples to evaluate

        Returns:
            Tuple of (batch_metrics, env_metrics)
        """
        batch, env_metrics = self.sample_rollouts(
            inference_ctx=inference_ctx,
            env_or_lesson_id=lesson_id,
            n_examples=n_examples,
            n_generations=1,
            temperature=1.0,  # Standard eval temperature
            mode="eval",
        )

        if batch is None:
            return BatchMetrics(total_count=0, success_count=0, avg_reward=0.0, rollout_stats=[]), {}

        return compute_batch_metrics(batch, lesson_id), env_metrics or {}

    def evaluate_all_lessons(
        self,
        inference_ctx: InferenceContext,
        n_examples_per_lesson: int,
    ) -> dict[str, tuple[BatchMetrics, dict[str, Any]]]:
        """Evaluate all lessons in the curriculum.

        Args:
            inference_ctx: Inference context for generation
            n_examples_per_lesson: Number of examples per lesson

        Returns:
            Dictionary mapping lesson IDs to (batch_metrics, env_metrics) tuples

        Raises:
            ValueError: If no curriculum config is available
        """
        if not self.curriculum_config:
            raise ValueError("Cannot evaluate curriculum without curriculum config")

        results = {}
        for lesson_id in self.curriculum_config.lessons:
            logger.info(f"Evaluating lesson: {lesson_id}")
            batch_metrics, env_metrics = self.evaluate_lesson(inference_ctx, lesson_id, n_examples_per_lesson)
            results[lesson_id] = (batch_metrics, env_metrics)

        return results

    def set_weight_step(self, step: int):
        """Update the current weight step.

        Args:
            step: New weight step value
        """
        self._current_weight_step = step
