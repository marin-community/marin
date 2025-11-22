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

"""Hook system for RolloutWorker to enable flexible callbacks and monitoring.

This module provides a flexible hook system that allows arbitrary callbacks to be
registered with the RolloutWorker, replacing the previous hardcoded evaluation logic.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import jax.random as jrandom
from jax import numpy as jnp
from levanter.utils.jax_utils import barrier_sync

if TYPE_CHECKING:
    from marin.rl.rollout_worker import RolloutWorker

from marin.rl.types import RolloutBatch, RolloutStats

logger = logging.getLogger("ray")


@dataclass
class RolloutBatchStats:
    total_count: int
    success_count: int
    rollout_stats: list[RolloutStats]
    avg_reward: float


def compute_batch_stats(batch: RolloutBatch, lesson_id: str) -> RolloutBatchStats:
    """Compute statistics from a rollout batch."""
    rollout_stats_list = []
    total_count = 0
    success_count = 0
    reward_sum = 0.0

    for group in batch.groups:
        for rollout in group.rollouts:
            rollout_stats_list.append(
                RolloutStats(
                    lesson_id=lesson_id,
                    episode_reward=rollout.episode_reward,
                    env_example_id=rollout.env_example_id,
                )
            )

            total_count += 1
            if rollout.episode_reward > 0:
                success_count += 1
            reward_sum += rollout.episode_reward

    return RolloutBatchStats(
        total_count=total_count,
        success_count=success_count,
        rollout_stats=rollout_stats_list,
        avg_reward=(reward_sum / total_count) if total_count > 0 else 0.0,
    )


def build_eval_metrics(prefix: str, lesson_id: str, batch: RolloutBatch) -> dict[str, Any]:
    """Build evaluation metrics from a rollout batch."""
    metrics = {}
    stats = compute_batch_stats(batch, lesson_id)
    if stats.total_count == 0:
        return metrics
    success_rate = stats.success_count / stats.total_count
    metrics[f"{prefix}/{lesson_id}/success_rate"] = success_rate
    metrics[f"{prefix}/{lesson_id}/avg_reward"] = stats.avg_reward
    metrics[f"{prefix}/{lesson_id}/total_count"] = stats.total_count
    return metrics


@dataclass
class HookContext:
    """Context passed to hooks containing relevant state and utilities."""

    worker: "RolloutWorker"
    step: int
    rng: jnp.ndarray
    curriculum_actor: Any
    lesson_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def split_rng(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Split the RNG key for use in the hook."""
        keys = jrandom.split(self.rng)
        return keys[0], keys[1]

    def sample_batch(
        self, lesson_id: str, n_examples: int, n_generations: int, mode: str, rng
    ) -> tuple[RolloutBatch | None, dict | None]:
        """Sample a batch of rollouts from the environment for the given lesson ID."""
        return self.worker._sample_batch(lesson_id, n_examples, n_generations, mode, rng)

    def log_prompt_example(self, lesson_id: str, batch: RolloutBatch, eval_type: str = "eval") -> None:
        """Log a single representative sample from an evaluation batch."""
        if not batch or not batch.groups:
            return

        # Take first rollout from first group as representative
        sample = batch.groups[0].rollouts[0]

        # Decode tokens to human-readable text
        prompt_text = self.worker._tokenizer.decode(sample.prompt_tokens, skip_special_tokens=True)
        response_text = self.worker._tokenizer.decode(sample.response_tokens, skip_special_tokens=True)

        # Log with structured keys
        prefix = f"inference.{eval_type}/{lesson_id}"
        metrics = {
            f"{prefix}/sample_prompt": prompt_text,
            f"{prefix}/sample_response": response_text,
            f"{prefix}/sample_example_id": sample.env_example_id,
        }
        self.worker.tracker.log(metrics, step=self.step)
        logger.info(f"Eval sample for lesson {lesson_id} at step {self.step}: {metrics}")


class Hook(ABC):
    """Base class for all hooks that can be registered with RolloutWorker."""

    @abstractmethod
    def should_run(self, context: HookContext) -> bool:
        """Determine if this hook should run at the current step."""
        pass

    @abstractmethod
    def run(self, context: HookContext) -> dict[str, Any] | None:
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


@dataclass
class PeriodicHook(Hook):
    """Base class for hooks that run periodically based on step count."""

    frequency: int
    """Run hook every N steps."""

    name: str = "periodic_hook"

    start_step: int = 0
    """First step to start running the hook."""

    def should_run(self, context: HookContext) -> bool:
        return context.step > self.start_step and context.step % self.frequency == 0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(frequency={self.frequency}, start_step={self.start_step})"


@dataclass
class EvaluateLessonHook(PeriodicHook):
    """Evaluate the current lesson periodically."""

    n_examples: int = field(kw_only=True)
    """Number of examples to evaluate."""

    name: str = "evaluate_lesson"

    update_curriculum: bool = False
    """Whether to update curriculum stats (only for full evals)."""

    def run(self, context: HookContext) -> dict[str, Any] | None:
        assert context.lesson_id is not None, "lesson_id is required for lesson evaluation"

        _rng, eval_rng = context.split_rng()
        logger.info(f"Running micro-eval for lesson {context.lesson_id} at step {context.step}")

        batch, _ = context.sample_batch(
            lesson_id=context.lesson_id,
            n_examples=self.n_examples,
            n_generations=1,
            mode="eval",
            rng=eval_rng,
        )

        if batch is None:
            return None

        stats = compute_batch_stats(batch, context.lesson_id)
        context.log_prompt_example(context.lesson_id, batch, eval_type="micro_eval")
        metrics = build_eval_metrics(prefix="inference.micro_eval", lesson_id=context.lesson_id, batch=batch)
        context.worker.tracker.log(metrics, step=context.step)
        logger.info("Eval metrics for lesson %s at step %d: %s", context.lesson_id, context.step, metrics)

        if self.update_curriculum:
            context.curriculum_actor.update_lesson_stats.options(enable_task_events=False).call(
                stats.rollout_stats, mode="eval", current_step=context.step
            )

        return metrics


@dataclass
class EvaluateCurriculumHook(PeriodicHook):
    """Evaluate all lessons in the curriculum periodically."""

    n_examples: int = field(kw_only=True)
    """Number of examples to evaluate."""

    name: str = "evaluate_curriculum"

    def run(self, context: HookContext) -> dict[str, Any] | None:
        lesson_names = list(context.worker.config.curriculum_config.lessons.keys())
        if not lesson_names:
            logger.info("No lessons to evaluate")
            return {}

        logger.info(f"Evaluating {len(lesson_names)} lessons")

        rng = context.rng
        for lesson_id in lesson_names:
            rng, lesson_rng = jrandom.split(rng)
            batch, _ = context.sample_batch(
                lesson_id=lesson_id,
                n_examples=self.n_examples,
                n_generations=1,
                mode="eval",
                rng=lesson_rng,
            )

            if batch is None:
                continue

            stats = compute_batch_stats(batch, lesson_id)
            context.log_prompt_example(lesson_id, batch, eval_type="eval")
            metrics = build_eval_metrics(prefix="inference.eval", lesson_id=lesson_id, batch=batch)
            context.worker.tracker.log(metrics, step=context.step)
            logger.info("Eval metrics for lesson %s at step %d: %s", lesson_id, context.step, metrics)

            context.curriculum_actor.update_lesson_stats.options(enable_task_events=False).call(
                stats.rollout_stats, mode="eval", current_step=context.step
            )

        barrier_sync()
        return {}


class HookManager:
    """Manages the execution of hooks in the RolloutWorker."""

    def __init__(self):
        """Initialize the hook manager."""
        self.hooks: list[Hook] = []

    def register_hook(self, hook: Hook) -> None:
        self.hooks.append(hook)
        logger.info(f"Registered hook: {hook}")

    def unregister_hook(self, hook: Hook) -> bool:
        if hook in self.hooks:
            self.hooks.remove(hook)
            logger.info(f"Unregistered hook: {hook}")
            return True
        return False

    def clear_hooks(self) -> None:
        self.hooks.clear()
        logger.info("Cleared all hooks")

    def run_hooks(self, context: HookContext) -> dict[str, Any]:
        """Run all hooks that should execute at the current step.

        Args:
            context: The hook context

        Returns:
            Aggregated results from all hooks
        """
        results = {}
        for hook in self.hooks:
            try:
                if hook.should_run(context):
                    logger.debug(f"Running hook: {hook}")
                    hook_results = hook.run(context)
                    if hook_results:
                        results.update({f"{hook.name}/{k}": v for k, v in hook_results.items()})
            except Exception as e:
                logger.error(f"Error running hook {hook}: {e}", exc_info=True)
        return results

    def __len__(self) -> int:
        return len(self.hooks)

    def __repr__(self) -> str:
        return f"HookManager(hooks={self.hooks})"


def create_default_evaluation_hooks(curriculum_config) -> list[Hook]:
    hooks = []

    if curriculum_config.micro_eval_frequency > 0:
        hooks.append(
            EvaluateLessonHook(
                frequency=curriculum_config.micro_eval_frequency,
                n_examples=curriculum_config.micro_eval_n_examples,
            )
        )

    if curriculum_config.eval_frequency > 0:
        hooks.append(
            EvaluateCurriculumHook(
                frequency=curriculum_config.eval_frequency,
                n_examples=curriculum_config.eval_n_examples,
            )
        )

    return hooks
