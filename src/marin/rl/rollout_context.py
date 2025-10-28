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
import threading
import time
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import haliax as hax
import jax.random as jrandom
import ray
from levanter.inference.openai import InferenceServer, InferenceServerConfig
from levanter.models.lm_model import LmConfig
from levanter.trainer import TrainerConfig
from transformers import PreTrainedTokenizer

from marin.rl.curriculum import CurriculumConfig, get_or_create_curriculum_actor
from marin.rl.environments import MarinEnv
from marin.rl.environments.base import load_environment_from_spec
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
    """Manages inference server, rollout generation, and curriculum.

    Handles all curriculum-related activities:
    - Inference server lifecycle
    - Model state management
    - Environment loading/caching
    - Rollout generation
    - Curriculum evaluation
    - Curriculum stats updates

    Returns data structures for the worker to log.
    Worker has NO curriculum responsibilities.
    """

    def __init__(
        self,
        inference_config: InferenceServerConfig,
        model_config: LmConfig,
        trainer_config: TrainerConfig,
        curriculum_config: CurriculumConfig,
        tokenizer: PreTrainedTokenizer,
        initial_model: Any,
        worker_id: str | None = None,
    ):
        """Initialize the rollout context.

        Args:
            inference_config: Configuration for inference server
            model_config: Model configuration
            trainer_config: Trainer configuration
            curriculum_config: Curriculum configuration
            tokenizer: Tokenizer for the model
            initial_model: Initial policy model
            worker_id: Worker identifier (auto-generated if not provided)
        """
        self.inference_config = inference_config
        self.model_config = model_config
        self.trainer_config = trainer_config
        self.curriculum_config = curriculum_config
        self.tokenizer = tokenizer
        self.worker_id = worker_id or f"{socket.gethostname()}_{os.getpid()}"

        self._policy_model = initial_model
        self._environments: dict[str, MarinEnv] = {}
        self._rng = jrandom.PRNGKey(42)  # Will be set properly by worker
        self._current_weight_step = 0

        # Start inference server
        with trainer_config.device_mesh, hax.axis_mapping(trainer_config.compute_axis_mapping):
            self._inference_server = InferenceServer.create(
                inference_config,
                model=self._policy_model,
                tokenizer=self.tokenizer,
            )
        self._inference_thread = threading.Thread(
            target=lambda: self._inference_server.serve(),
            daemon=True,
        )
        self._inference_thread.start()
        time.sleep(1.0)  # TODO: replace with wait_until_ready()

        # Create curriculum actor
        self._curriculum_actor = get_or_create_curriculum_actor(curriculum_config)

    def _load_environment(self, lesson_id: str) -> MarinEnv:
        """Load and cache environment for lesson.

        Args:
            lesson_id: ID of the lesson

        Returns:
            Loaded environment instance
        """
        if lesson_id in self._environments:
            return self._environments[lesson_id]

        lesson_config = self.curriculum_config.lessons[lesson_id]
        env = load_environment_from_spec(lesson_config.env_config)
        self._environments[lesson_id] = env
        return env

    def get_loaded_environments(self) -> dict[str, MarinEnv]:
        """Get all currently loaded environments.

        Returns:
            Dictionary mapping environment IDs to instances
        """
        return dict(self._environments)

    def sample_batch(
        self,
        lesson_id: str,
        n_examples: int,
        n_generations: int,
        mode: str,
        rng,
        weight_step: int,
        worker_id: str,
    ) -> tuple[RolloutBatch | None, dict[str, Any] | None]:
        """Generate a batch of rollouts.

        Args:
            lesson_id: Lesson to sample from
            n_examples: Number of examples to generate
            n_generations: Generations per example
            mode: 'train' or 'eval'
            rng: JAX PRNG key
            weight_step: Current weight step for metadata
            worker_id: Worker identifier for metadata

        Returns:
            (RolloutBatch with metadata attached, env metrics dict)
        """
        env = self._load_environment(lesson_id)
        lesson_config = self.curriculum_config.lessons[lesson_id]

        # Get sampling params
        temperature = lesson_config.sampling_params.temperature
        stop_tokens = lesson_config.sampling_params.stop_tokens
        max_tokens = lesson_config.sampling_params.max_tokens

        # Create inference context
        policy_ctx = InferenceContext(
            tokenizer=self.tokenizer,
            inference_server=self._inference_server,
            max_tokens=max_tokens,
            stop_tokens=stop_tokens,
        )

        # Sample from environment
        with (
            self.trainer_config.device_mesh,
            hax.axis_mapping(self.trainer_config.compute_axis_mapping),
        ):
            rollout_groups, metrics = env.sample(
                inference_ctx=policy_ctx,
                n_examples=n_examples,
                n_generations=n_generations,
                temperature=temperature,
                prng_key=rng,
                mode=mode,
            )

        if len(rollout_groups) == 0:
            logger.warning("No valid rollouts generated")
            return None, None

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

        logger.info(
            "Generated %d rollout groups from lesson %s at step %d",
            len(rollout_groups),
            lesson_id,
            weight_step,
        )

        return rollout_batch, metrics

    def evaluate_lesson(
        self,
        lesson_id: str,
        n_examples: int,
        eval_type: str,
        rng,
        step: int,
        weight_step: int,
        worker_id: str,
    ) -> tuple[BatchMetrics, RolloutBatch]:
        """Evaluate a single lesson and update curriculum.

        Args:
            lesson_id: ID of the lesson to evaluate
            n_examples: Number of examples to evaluate
            eval_type: 'eval' or 'micro_eval' - only 'eval' updates curriculum
            rng: JAX PRNG key
            step: Current training step
            weight_step: Current weight step
            worker_id: Worker identifier

        Returns:
            (batch_metrics, batch) for worker to log
        """
        batch, _ = self.sample_batch(
            lesson_id=lesson_id,
            n_examples=n_examples,
            n_generations=1,
            mode="eval",
            rng=rng,
            weight_step=weight_step,
            worker_id=worker_id,
        )

        if batch is None:
            empty_metrics = BatchMetrics(total_count=0, success_count=0, avg_reward=0.0, rollout_stats=[])
            return empty_metrics, RolloutBatch(
                groups=[], metadata=RolloutMetadata(worker_id=worker_id, timestamp=time.time(), weight_step=weight_step)
            )

        metrics = compute_batch_metrics(batch, lesson_id)

        # Manager handles curriculum updates
        if eval_type == "eval":
            ray.get(
                self._curriculum_actor.update_lesson_stats.options(enable_task_events=False).remote(
                    metrics.rollout_stats, mode="eval", current_step=step
                )
            )

        return metrics, batch

    def evaluate_curriculum(
        self,
        eval_n_examples: int,
        rng,
        step: int,
        weight_step: int,
        worker_id: str,
    ) -> dict[str, tuple[BatchMetrics, RolloutBatch]]:
        """Evaluate all lessons and update curriculum.

        Args:
            eval_n_examples: Number of examples per lesson
            rng: JAX PRNG key
            step: Current training step
            weight_step: Current weight step
            worker_id: Worker identifier

        Returns:
            {lesson_id: (batch_metrics, batch)} for worker to log
        """
        lesson_names = list(self.curriculum_config.lessons.keys())
        if not lesson_names:
            logger.info("No lessons to evaluate")
            return {}

        logger.info(f"Evaluating {len(lesson_names)} lessons")

        results = {}
        for lesson_id in lesson_names:
            metrics, batch = self.evaluate_lesson(
                lesson_id=lesson_id,
                n_examples=eval_n_examples,
                eval_type="eval",  # Full eval updates curriculum
                rng=rng,
                step=step,
                weight_step=weight_step,
                worker_id=worker_id,
            )
            results[lesson_id] = (metrics, batch)

        return results

    def update_training_stats(
        self,
        lesson_id: str,
        batch: RolloutBatch,
        step: int,
    ) -> None:
        """Update curriculum with training rollout stats.

        Args:
            lesson_id: Lesson that generated the batch
            batch: Rollout batch from training
            step: Current training step
        """
        metrics = compute_batch_metrics(batch, lesson_id)
        ray.get(
            self._curriculum_actor.update_lesson_stats.options(enable_task_events=False).remote(
                metrics.rollout_stats, mode="training", current_step=step
            )
        )

    def set_weight_step(self, step: int):
        """Update the current weight step.

        Args:
            step: New weight step value
        """
        self._current_weight_step = step

    def update_model(self, model: Any) -> None:
        """Update the policy model and reload inference server.

        Args:
            model: New policy model
        """
        self._policy_model = model
        self._inference_server.reload(lambda m: self._policy_model)
        logger.info("Model updated in inference server")

    def shutdown(self) -> None:
        """Shutdown inference server and cleanup resources."""
        if self._inference_server:
            self._inference_server.shutdown()
            logger.info("Inference server shut down")
