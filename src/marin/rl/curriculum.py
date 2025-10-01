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
Adaptive curriculum learning system for RL training.

This module implements an adaptive curriculum that automatically adjusts
environment sampling based on performance, managing dependencies between
lessons and tracking progress to maximize learning efficiency.
"""

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from marin.rl.environments.base import EnvConfig, MarinEnv, load_environment_from_spec
from marin.rl.types import Rollout


@dataclass
class LessonStats:
    """Statistics tracking performance for a single lesson."""

    smoothed_success: float = 0.0
    """Exponentially smoothed success rate from training rollouts."""

    smoothed_reward: float = 0.0
    """Exponentially smoothed reward from training rollouts."""

    eval_success: float = 0.0
    """Last evaluation success rate."""

    eval_reward: float = 0.0
    """Last evaluation reward."""

    eval_step: int = -1
    """Step at which last evaluation was performed."""

    total_samples: int = 0
    """Total number of rollouts seen for this lesson."""

    reward_history: list[float] = field(default_factory=list)
    """Recent reward history for plateau detection."""

    def to_dict(self) -> dict:
        """Serialize to dictionary for checkpointing."""
        return {
            "smoothed_success": self.smoothed_success,
            "smoothed_reward": self.smoothed_reward,
            "eval_success": self.eval_success,
            "eval_reward": self.eval_reward,
            "eval_step": self.eval_step,
            "total_samples": self.total_samples,
            "reward_history": self.reward_history,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LessonStats":
        """Deserialize from dictionary."""
        return cls(**data)


@dataclass
class LessonDependency:
    """Dependency specification for a lesson."""

    dependency_name: str
    """Name of the lesson this depends on."""

    reward_threshold: float = 0.0
    """Reward threshold that dependency must reach before this lesson activates.
    By default (0.0), only wait for dependency to plateau."""


@dataclass
class LessonConfig:
    """Configuration for a single lesson in the curriculum."""

    lesson_name: str
    """Unique identifier for this lesson."""

    env_config: EnvConfig
    """Environment configuration for this lesson."""

    dependencies: list[LessonDependency] = field(default_factory=list)
    """Prerequisites that must be satisfied before this lesson unlocks."""

    initial_weight: float = 1.0
    """Initial sampling weight before performance data is available."""

    start_threshold: float = 0.0
    """Minimum eval performance required to begin training on this lesson once unlocked."""

    stop_threshold: float = 1.0
    """Performance threshold for graduation consideration."""

    plateau_window: int = 50
    """Number of recent samples to consider for plateau detection."""

    plateau_threshold: float = 0.01
    """Relative slope threshold for detecting plateaus."""


@dataclass
class CurriculumConfig:
    """Configuration for the adaptive curriculum system."""

    lessons: list[LessonConfig]
    """List of lesson configurations in the curriculum."""

    eval_frequency: int = 1000
    """How often to run evaluation (in rollout worker steps)."""

    temperature: float = 1.0
    """Temperature for sampling weight distribution."""


def _validate_dependencies(lesson_configs: dict[str, LessonConfig]):
    """Validate that lesson dependencies form a valid DAG (no cycles).

    Args:
        lesson_configs: Mapping from lesson name to configuration.

    Raises:
        ValueError: If circular dependencies are detected.
    """
    visited = set()
    rec_stack = set()

    def has_cycle(node: str) -> bool:
        """Check if there's a cycle starting from node using DFS."""
        if node not in lesson_configs:
            # Dependency doesn't exist - we'll catch this elsewhere
            return False

        visited.add(node)
        rec_stack.add(node)

        for dep in lesson_configs[node].dependencies:
            dep_name = dep.dependency_name
            if dep_name not in visited:
                if has_cycle(dep_name):
                    return True
            elif dep_name in rec_stack:
                return True

        rec_stack.remove(node)
        return False

    for name in lesson_configs:
        if name not in visited:
            if has_cycle(name):
                raise ValueError(f"Circular dependency detected involving lesson '{name}'")

    # Also validate that all dependencies reference existing lessons
    for name, config in lesson_configs.items():
        for dep in config.dependencies:
            if dep.dependency_name not in lesson_configs:
                raise ValueError(f"Lesson '{name}' depends on unknown lesson '{dep.dependency_name}'")


class Curriculum:
    """Manages adaptive curriculum learning with lesson progression and sampling.

    The curriculum tracks performance across multiple lessons, manages dependencies
    between them, and dynamically adjusts sampling weights to focus on the most
    productive learning tasks.
    """

    def __init__(self, config: CurriculumConfig):
        """Initialize curriculum from configuration.

        Args:
            config: Curriculum configuration with lesson specs.
        """
        self.config = config

        # Build lesson mapping and validate
        self.lesson_configs = {lesson.lesson_name: lesson for lesson in config.lessons}
        _validate_dependencies(self.lesson_configs)

        # Initialize statistics for each lesson
        self.stats: dict[str, LessonStats] = {lesson.lesson_name: LessonStats() for lesson in config.lessons}

        # Load environments for each lesson
        self.environments: dict[str, MarinEnv] = {
            lesson.lesson_name: load_environment_from_spec(lesson.env_config) for lesson in config.lessons
        }

        # Lesson state tracking
        self.unlocked: set[str] = set()
        self.graduated: set[str] = set()

        # Unlock lessons without dependencies
        for lesson in config.lessons:
            if not lesson.dependencies:
                self.unlocked.add(lesson.lesson_name)

        # Step counter for internal tracking
        self.current_step = 0

    def step(self):
        """Increment the curriculum step counter."""
        self.current_step += 1

    def compute_sampling_weights(self) -> dict[str, float]:
        """Compute sampling weights for all active lessons.

        Uses quadratic weighting that peaks at intermediate success rates,
        with sigmoid smoothing at boundaries and minimum probability guarantees.

        Returns:
            Dictionary mapping lesson names to sampling probabilities.
        """
        active_lessons = self.unlocked - self.graduated
        if not active_lessons:
            return {}

        weights = {}
        min_prob = 0.01

        for name in active_lessons:
            stats = self.stats[name]
            config = self.lesson_configs[name]

            # Get combined success rate
            success_rate = get_combined_success_rate(stats, self.current_step)

            # Quadratic weight peaking at 50% success
            # w = 1 - 4(s - 0.5)^2 = -4s^2 + 4s
            base_weight = max(0.0, -4 * success_rate**2 + 4 * success_rate)

            # Apply sigmoid smoothing at low and high ends
            low_sigmoid = sigmoid(success_rate, center=0.1, steepness=20.0)
            high_sigmoid = 1.0 - sigmoid(success_rate, center=0.9, steepness=20.0)
            base_weight = base_weight * low_sigmoid * high_sigmoid

            # Use initial weight if we have no data yet
            if stats.total_samples == 0:
                weights[name] = config.initial_weight
            else:
                weights[name] = max(base_weight, min_prob)

        # Renormalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights

    def sample_lesson(self, prng_key) -> tuple[str, MarinEnv]:
        """Sample a lesson for training based on current weights.

        Args:
            prng_key: JAX random key for sampling.

        Returns:
            Tuple of (lesson_name, environment).
        """
        weights = self.compute_sampling_weights()
        if not weights:
            raise RuntimeError("No active lessons available for sampling")

        names = list(weights.keys())
        probs = jnp.array([weights[name] for name in names])

        # Sample lesson
        idx = jax.random.choice(prng_key, len(names), p=probs)
        lesson_name = names[int(idx)]

        return lesson_name, self.environments[lesson_name]

    def check_dependencies(self, lesson_name: str) -> bool:
        """Check if all dependencies for a lesson are satisfied.

        Args:
            lesson_name: Name of the lesson to check.

        Returns:
            True if all dependencies are satisfied, False otherwise.
        """
        lesson_config = self.lesson_configs[lesson_name]

        for dep in lesson_config.dependencies:
            dep_name = dep.dependency_name
            dep_stats = self.stats[dep_name]

            # Check if dependency has reached required threshold
            success_rate = get_combined_success_rate(dep_stats, self.current_step)
            if success_rate < dep.reward_threshold:
                return False

            # Check if dependency has plateaued (if threshold is met or is 0.0)
            if success_rate >= dep.reward_threshold:
                dep_config = self.lesson_configs[dep_name]
                if not is_plateaued(dep_stats, window=dep_config.plateau_window, threshold=dep_config.plateau_threshold):
                    return False

        return True

    def update_unlocked_lessons(self):
        """Update which lessons are currently unlocked based on dependencies."""
        for lesson_name in self.lesson_configs:
            # Skip if already unlocked or graduated
            if lesson_name in self.unlocked or lesson_name in self.graduated:
                continue

            # Check if dependencies are satisfied
            if self.check_dependencies(lesson_name):
                self.unlocked.add(lesson_name)


def update_from_rollout(stats: LessonStats, rollout: Rollout, alpha: float = 0.1) -> LessonStats:
    """Update lesson statistics from a training rollout.

    Args:
        stats: Current lesson statistics.
        rollout: Rollout to incorporate.
        alpha: Exponential smoothing parameter (higher = more weight on new data).

    Returns:
        Updated lesson statistics.
    """
    # Compute success (binary: reward > 0)
    success = 1.0 if rollout.episode_reward > 0 else 0.0

    # Update exponential moving averages
    if stats.total_samples == 0:
        # First sample - initialize
        smoothed_success = success
        smoothed_reward = rollout.episode_reward
    else:
        smoothed_success = (1 - alpha) * stats.smoothed_success + alpha * success
        smoothed_reward = (1 - alpha) * stats.smoothed_reward + alpha * rollout.episode_reward

    # Update reward history (keep last 100 samples)
    reward_history = stats.reward_history.copy()
    reward_history.append(float(rollout.episode_reward))
    if len(reward_history) > 100:
        reward_history = reward_history[-100:]

    return LessonStats(
        smoothed_success=smoothed_success,
        smoothed_reward=smoothed_reward,
        eval_success=stats.eval_success,
        eval_reward=stats.eval_reward,
        eval_step=stats.eval_step,
        total_samples=stats.total_samples + 1,
        reward_history=reward_history,
    )


def get_combined_success_rate(stats: LessonStats, current_step: int) -> float:
    """Blend training and evaluation success rates based on recency.

    Weights evaluation more heavily when recent, training more when eval is stale.

    Args:
        stats: Lesson statistics.
        current_step: Current training step.

    Returns:
        Combined success rate in [0, 1].
    """
    if stats.eval_step < 0:
        # No evaluation data yet
        return stats.smoothed_success

    # Compute staleness and decay eval weight
    staleness = current_step - stats.eval_step
    eval_weight = 0.7 * np.exp(-0.001 * staleness)

    # Blend eval and training metrics
    return eval_weight * stats.eval_success + (1 - eval_weight) * stats.smoothed_success


def sigmoid(x: float, center: float, steepness: float) -> float:
    """Smooth sigmoid transition.

    Args:
        x: Input value.
        center: Center point of the sigmoid.
        steepness: Steepness of the transition.

    Returns:
        Sigmoid output in [0, 1].
    """
    return 1 / (1 + np.exp(-steepness * (x - center)))


def is_plateaued(stats: LessonStats, window: int = 50, threshold: float = 0.01) -> bool:
    """Detect if reward has plateaued using linear regression on recent history.

    Args:
        stats: Lesson statistics with reward history.
        window: Number of recent samples to analyze.
        threshold: Relative slope threshold for plateau detection.

    Returns:
        True if the reward has plateaued, False otherwise.
    """
    if len(stats.reward_history) < window:
        return False

    # Get recent rewards
    recent = np.array(stats.reward_history[-window:])

    # Fit linear trend
    x = np.arange(len(recent))
    coeffs = np.polyfit(x, recent, 1)
    slope = coeffs[0]

    # Check if relative trend is flat
    mean_reward = np.mean(recent)
    if abs(mean_reward) > 1e-6:
        relative_trend = abs(slope) / abs(mean_reward)
        return relative_trend < threshold

    return True
