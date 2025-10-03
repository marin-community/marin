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

import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field

import fsspec
import numpy as np

from marin.rl.environments.base import EnvConfig
from marin.rl.types import RolloutStats

logger = logging.getLogger(__name__)

MAX_REWARD_HISTORY = 1000


@dataclass
class PerformanceStats:
    """Statistics for a particular mode (training or eval)."""

    total_samples: int = 0
    """Total number of rollouts seen."""

    reward_history: np.ndarray = field(default_factory=lambda: np.array([]))
    """Recent reward history for plateau detection and computing success rate."""

    last_update_step: int = -1
    """Step at which this was last updated."""


@dataclass
class LessonStats:
    """Statistics tracking performance for a single lesson."""

    training_stats: PerformanceStats = field(default_factory=PerformanceStats)
    """Performance metrics from training rollouts."""

    eval_stats: PerformanceStats = field(default_factory=PerformanceStats)
    """Performance metrics from evaluation rollouts."""


@dataclass
class LessonDependency:
    """Dependency specification for a lesson."""

    dependency_id: str
    """ID of the lesson this depends on."""

    reward_threshold: float = 0.0
    """Reward threshold that dependency must reach before this lesson activates.
    By default (0.0), only wait for dependency to plateau."""


@dataclass
class SamplingParams:
    """Parameters for sampling rollouts from an environment."""

    temperature: float = 1.0
    n_prompts: int = 8
    n_generations_per_prompt: int = 4
    stop_tokens: list[int] | None = None


@dataclass
class LessonConfig:
    """Configuration for a single lesson in the curriculum."""

    lesson_id: str
    """Unique identifier for this lesson."""

    env_config: EnvConfig
    """Environment configuration for this lesson."""

    dependencies: list[LessonDependency] = field(default_factory=list)
    """Prerequisites that must be satisfied before this lesson unlocks."""

    start_threshold: float = 0.0
    """Minimum eval performance required to begin training on this lesson once unlocked."""

    stop_threshold: float = 1.0
    """Performance threshold for graduation consideration."""

    plateau_window: int = 50
    """Number of recent samples to consider for plateau detection."""

    plateau_threshold: float = 0.01
    """Relative slope threshold for detecting plateaus."""

    sampling_params: SamplingParams
    """Per-lesson sampling configuration (overrides global defaults)."""


@dataclass
class CurriculumConfig:
    """Configuration for the adaptive curriculum system."""

    lessons: dict[str, LessonConfig]
    """Dictionary mapping lesson names to lesson configurations."""

    eval_frequency: int = 1000
    """How often to run evaluation (in rollout worker steps)."""

    eval_n_examples: int = 32
    """Number of examples to use for each lesson during evaluation."""

    eval_n_generations: int = 1
    """Number of generations per example during evaluation."""

    temperature: float = 1.0
    """Temperature for sampling weight distribution."""

    actor_name: str = "curriculum"
    """Name for the Ray actor (shared between rollout and train workers)."""

    minimum_sample_probability: float = 0.1
    """Minimum probability for sampling any active lesson."""

    checkpoint_steps: int = 10
    """How often to checkpoint curriculum state (in training steps)."""


def _validate_dependencies(lesson_configs: dict[str, LessonConfig]):
    """Validate that lesson dependencies form a valid DAG (no cycles)."""
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
            dep_id = dep.dependency_id
            if dep_id not in visited:
                if has_cycle(dep_id):
                    return True
            elif dep_id in rec_stack:
                return True

        rec_stack.remove(node)
        return False

    for lesson_id in lesson_configs:
        if lesson_id not in visited:
            if has_cycle(lesson_id):
                raise ValueError(f"Circular dependency detected involving lesson '{lesson_id}'")

    # Also validate that all dependencies reference existing lessons
    for lesson_id, config in lesson_configs.items():
        for dep in config.dependencies:
            if dep.dependency_id not in lesson_configs:
                raise ValueError(f"Lesson '{lesson_id}' depends on unknown lesson '{dep.dependency_id}'")


def compute_smoothed_success(reward_history: np.ndarray, alpha: float = 0.1, prior_mean: float = 0.5) -> float:
    """Compute exponentially smoothed success rate with Bayesian prior initialization.

    The prior naturally decays as data accumulates due to the exponential smoothing:
    - After ~10 samples: ~35% prior influence (with alpha=0.1)
    - After ~20 samples: ~12% prior influence
    - After ~50 samples: <1% prior influence

    Args:
        reward_history: Array of episode rewards.
        alpha: Exponential smoothing parameter (higher = more weight on recent data).
        prior_mean: Prior belief about success rate (default 0.5 = uncertain).

    Returns:
        Exponentially smoothed success rate.
    """
    if len(reward_history) == 0:
        return prior_mean

    # Convert rewards to binary success (reward > 0)
    successes = (reward_history > 0).astype(float)

    # Initialize EMA with prior
    ema = prior_mean
    for success in successes:
        ema = (1 - alpha) * ema + alpha * success

    return ema


def update_performance_stats(
    stats: PerformanceStats, rollout_stats: list[RolloutStats], current_step: int
) -> PerformanceStats:
    """Update performance statistics from rollout stats.

    Args:
        stats: Current performance statistics.
        rollout_stats: Lightweight rollout statistics.
        current_step: Current training step for tracking staleness.

    Returns:
        Updated performance statistics.
    """
    # Append new rewards to history
    new_rewards = np.array([rs.episode_reward for rs in rollout_stats])
    reward_history = np.concatenate([stats.reward_history, new_rewards])

    if len(reward_history) > MAX_REWARD_HISTORY:
        reward_history = reward_history[-MAX_REWARD_HISTORY:]

    return PerformanceStats(
        total_samples=stats.total_samples + len(rollout_stats),
        reward_history=reward_history,
        last_update_step=current_step,
    )


def compute_success_ratio(stats: LessonStats, current_step: int, max_staleness: int = 1000) -> float:
    """Get success rate for a lesson."""
    return compute_smoothed_success(stats.training_stats.reward_history)


def is_plateaued(stats: LessonStats, window: int = 100, threshold: float = 0.01) -> bool:
    """Detect if reward has plateaued using conservative statistical tests.

    Args:
        stats: Lesson statistics containing reward history.
        window: Number of recent samples to analyze.
        threshold: Relative slope threshold (slope/mean) for plateau detection.

    Returns:
        True if performance has plateaued (no significant improvement).
    """
    perf_stats = stats.training_stats

    if len(perf_stats.reward_history) < window:
        return False

    recent = np.array(perf_stats.reward_history[-window:])

    # Linear regression to measure trend
    x = np.arange(len(recent))
    from scipy import stats as scipy_stats

    result = scipy_stats.linregress(x, recent)
    slope = result.slope
    p_value = result.pvalue

    mean_reward = np.mean(np.abs(recent))
    std_reward = np.std(recent)

    # Condition 1: Slope must be small or negative
    if abs(mean_reward) > 1e-6:
        relative_slope = abs(slope) / abs(mean_reward)
        slope_is_flat = relative_slope < threshold
    else:
        slope_is_flat = True

    # Condition 2: Coefficient of variation must be low (stable performance)
    # CV < 0.1 indicates very stable values (conservative threshold)
    if abs(mean_reward) > 1e-6:
        cv = std_reward / abs(mean_reward)
        cv_is_stable = cv < 0.1
    else:
        cv_is_stable = True

    # Condition 3: Slope must be statistically insignificant (p > 0.1)
    # Conservative: require strong evidence of NO trend
    # Note: p_value can be NaN for perfectly flat data (zero variance)
    slope_not_significant = np.isnan(p_value) or p_value > 0.1

    # Conservative: Require ALL three conditions
    return slope_is_flat and cv_is_stable and slope_not_significant


@dataclass
class Curriculum:
    """Manages adaptive curriculum learning with lesson progression and sampling.

    The curriculum tracks performance across multiple lessons, manages dependencies
    between them, and dynamically adjusts sampling weights to focus on the most
    productive learning tasks.
    """

    config: CurriculumConfig

    def __post_init__(self):
        _validate_dependencies(self.config.lessons)

        # Validate lesson_id matches dict key
        for lesson_id, lesson_config in self.config.lessons.items():
            if lesson_config.lesson_id != lesson_id:
                raise ValueError(f"Lesson dict key '{lesson_id}' must match lesson_id '{lesson_config.lesson_id}'")

        # Initialize statistics for each lesson
        self.stats: dict[str, LessonStats] = {lesson_id: LessonStats() for lesson_id in self.config.lessons}

        # Lesson state tracking
        self.unlocked: set[str] = set()
        self.graduated: set[str] = set()

        # Unlock lessons without dependencies
        for lesson_id, lesson in self.config.lessons.items():
            if not lesson.dependencies:
                self.unlocked.add(lesson_id)

        # Step counter for internal tracking
        self.current_step = 0

    def compute_sampling_weights(self) -> dict[str, float]:
        """Compute sampling weights for all active lessons.

        Uses quadratic weighting that peaks at intermediate success rates
        and minimum probability guarantees.

        Returns:
            Dictionary mapping lesson names to sampling probabilities.
        """
        active_lessons = self.unlocked - self.graduated
        if not active_lessons:
            return {}

        weights = {}

        for name in active_lessons:
            stats = self.stats[name]

            # Get success rate for decisions
            success_rate = compute_success_ratio(stats, self.current_step)

            # Quadratic weight peaking at 50% success
            base_weight = max(0.0, -4 * success_rate**2 + 4 * success_rate)
            total_samples = stats.training_stats.total_samples
            exploration_bonus = 1.0 + np.exp(-0.01 * total_samples)
            weights[name] = base_weight * exploration_bonus

        total = 1 + sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}

        # bump up to minimum probability & renormalize
        for k in weights:
            weights[k] = max(weights[k], self.config.minimum_sample_probability)

        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}
        return weights

    def sample_lesson(self, seed: int) -> str:
        """Sample a lesson for training based on current weights.

        Args:
            seed: Integer seed for random sampling.

        Returns:
            Lesson ID string.
        """
        weights = self.compute_sampling_weights()
        if not weights:
            raise RuntimeError("No active lessons available for sampling")

        lesson_ids = list(weights.keys())
        probs = np.array([weights[lesson_id] for lesson_id in lesson_ids])

        rng = np.random.default_rng(seed)
        idx = rng.choice(len(lesson_ids), p=probs)
        lesson_id = lesson_ids[idx]

        return lesson_id

    def update_lesson_stats(self, rollout_stats: list[RolloutStats], mode: str, current_step: int) -> None:
        """Update lesson statistics from rollout stats and trigger lesson state updates.

        Args:
            rollout_stats_list: List of statistics from completed rollouts.
            mode: "training" or "eval" to determine which stats to update.
            current_step: Current training step for tracking and triggering lesson updates.
        """
        assert mode in ("training", "eval"), f"Invalid mode: {mode}"

        # Update current step (use max to handle potential concurrent updates from multiple workers)
        self.current_step = max(self.current_step, current_step)
        stats_by_lesson = defaultdict(list)
        for rs in rollout_stats:
            stats_by_lesson[rs.lesson_id].append(rs)

        for lesson_id, stats in stats_by_lesson.items():
            if mode == "training":
                self.stats[lesson_id].training_stats = update_performance_stats(
                    self.stats[lesson_id].training_stats, stats, self.current_step
                )
            else:  # eval
                self.stats[lesson_id].eval_stats = update_performance_stats(
                    self.stats[lesson_id].eval_stats, stats, self.current_step
                )

        # Automatically update lesson states (unlocking/graduation) after updating stats
        self._unlock_and_graduate_lessons()

    def get_metrics(self) -> dict:
        """Get curriculum metrics for monitoring.

        Returns:
            Dictionary of metrics including sampling weights, active lessons, etc.
        """
        weights = self.compute_sampling_weights()
        active = self.unlocked - self.graduated

        # Sampling entropy
        entropy = -sum(w * np.log(w + 1e-10) for w in weights.values() if w > 0)

        # Effective lessons (inverse Simpson index)
        effective = 1 / sum(w**2 for w in weights.values()) if weights else 0

        return {
            "step": self.current_step,
            "total_lessons": len(self.config.lessons),
            "unlocked_lessons": len(self.unlocked),
            "active_lessons": len(active),
            "graduated_lessons": len(self.graduated),
            "sampling_entropy": entropy,
            "effective_lessons": effective,
            "mean_success": (
                np.mean([compute_success_ratio(self.stats[n], self.current_step) for n in active]) if active else 0
            ),
            "sampling_weights": weights,
        }

    def _check_dependencies_for_lesson(self, lesson_id: str) -> bool:
        """Return true if all dependencies for a lesson are satisfied."""
        lesson_config = self.config.lessons[lesson_id]

        for dep in lesson_config.dependencies:
            dep_id = dep.dependency_id
            dep_stats = self.stats[dep_id]
            dep_config = self.config.lessons[dep_id]

            # Check if dependency has reached required threshold
            dep_success_rate = compute_success_ratio(dep_stats, self.current_step)
            if dep_success_rate < dep.reward_threshold:
                return False

            # Check if dependency has plateaued (if threshold is met or is 0.0)
            if dep_success_rate >= dep.reward_threshold:
                if not is_plateaued(dep_stats, window=dep_config.plateau_window, threshold=dep_config.plateau_threshold):
                    return False

        logger.info("All dependencies satisfied for lesson '%s'", lesson_id)
        for dep in lesson_config.dependencies:
            success_rate = compute_success_ratio(self.stats[dep.dependency_id], self.current_step)
            logger.info(
                f"  Dependency '{dep.dependency_id}' met. Success ratio: {success_rate}, recent rewards: %s",
                self.stats[dep.dependency_id].training_stats.reward_history[-10:],
            )
        return True

    def check_graduation(self, lesson_id: str) -> bool:
        """Return true if a lesson should graduate and be removed from active sampling."""
        lesson_config = self.config.lessons[lesson_id]
        stats = self.stats[lesson_id]
        logger.info("Checking graduation for lesson '%s' with stats %s", lesson_id, stats)

        # Must have evaluation data to graduate
        if stats.eval_stats.last_update_step < 0:
            logger.info("Lesson '%s' cannot graduate: no eval data", lesson_id)
            return False

        # Check if performance meets graduation threshold
        lesson_success_rate = compute_success_ratio(stats, self.current_step)
        if lesson_success_rate < lesson_config.stop_threshold:
            logger.info(
                "Lesson '%s' cannot graduate: success rate %f < threshold %f",
                lesson_id,
                lesson_success_rate,
                lesson_config.stop_threshold,
            )
            return False

        # Check if performance has plateaued
        if not is_plateaued(stats, window=lesson_config.plateau_window, threshold=lesson_config.plateau_threshold):
            logger.info("Lesson '%s' cannot graduate: performance not plateaued", lesson_id)
            return False

        return True

    def _unlock_and_graduate_lessons(self):
        """Update which lessons are currently available based on dependencies or graduation."""
        for lesson_id in self.config.lessons:
            if lesson_id not in self.unlocked and self._check_dependencies_for_lesson(lesson_id):
                logger.info("Unlocking lesson '%s' with stats %s", lesson_id, self.stats[lesson_id])
                self.unlocked.add(lesson_id)

            if lesson_id in self.unlocked and lesson_id not in self.graduated and self.check_graduation(lesson_id):
                logger.info("Graduating lesson '%s' with stats %s", lesson_id, self.stats[lesson_id])
                self.graduated.add(lesson_id)

    def save_checkpoint(self, checkpoint_dir: str, filename: str = "curriculum_state.json"):
        """Save curriculum state to disk as JSON.

        Args:
            checkpoint_dir: Directory to save checkpoint in.
            filename: Name of the checkpoint file.
        """

        logger.info("Saving curriculum checkpoint to %s/%s at step %d", checkpoint_dir, filename, self.current_step)

        fs, _ = fsspec.core.url_to_fs(checkpoint_dir)
        fs.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, filename)

        # Convert stats to dict with numpy arrays converted to lists
        stats_dict = {}
        for name, stats in self.stats.items():
            # Handle both np.array and list for reward_history
            train_history = stats.training_stats.reward_history.tolist()
            eval_history = stats.eval_stats.reward_history.tolist()
            stats_dict[name] = {
                "training_stats": {
                    "total_samples": stats.training_stats.total_samples,
                    "reward_history": train_history,
                    "last_update_step": stats.training_stats.last_update_step,
                },
                "eval_stats": {
                    "total_samples": stats.eval_stats.total_samples,
                    "reward_history": eval_history,
                    "last_update_step": stats.eval_stats.last_update_step,
                },
            }

        checkpoint_data = {
            "stats": stats_dict,
            "unlocked": list(self.unlocked),
            "graduated": list(self.graduated),
            "current_step": self.current_step,
        }

        with fs.open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2)

    def restore_checkpoint(self, checkpoint_dir: str, filename: str = "curriculum_state.json"):
        """Restore curriculum state from checkpoint in-place.

        Args:
            checkpoint_dir: Directory containing the checkpoint.
            filename: Name of the checkpoint file to load.
        """
        import os

        import fsspec

        fs, _ = fsspec.core.url_to_fs(checkpoint_dir)
        checkpoint_path = os.path.join(checkpoint_dir, filename)

        if not fs.exists(checkpoint_path):
            logger.info("No curriculum checkpoint found at %s, starting fresh", checkpoint_path)
            return

        with fs.open(checkpoint_path) as f:
            checkpoint_data = json.load(f)

        # Restore state in-place, converting lists back to numpy arrays
        self.stats = {}
        for name, stats_dict in checkpoint_data["stats"].items():
            self.stats[name] = LessonStats(
                training_stats=PerformanceStats(
                    total_samples=stats_dict["training_stats"]["total_samples"],
                    reward_history=np.array(stats_dict["training_stats"]["reward_history"]),
                    last_update_step=stats_dict["training_stats"]["last_update_step"],
                ),
                eval_stats=PerformanceStats(
                    total_samples=stats_dict["eval_stats"]["total_samples"],
                    reward_history=np.array(stats_dict["eval_stats"]["reward_history"]),
                    last_update_step=stats_dict["eval_stats"]["last_update_step"],
                ),
            )
        self.unlocked = set(checkpoint_data["unlocked"])
        self.graduated = set(checkpoint_data["graduated"])
        self.current_step = checkpoint_data["current_step"]

        logger.info("Restored curriculum checkpoint from %s at step %d", checkpoint_path, self.current_step)


def get_or_create_curriculum_actor(config: CurriculumConfig):
    import ray

    return ray.remote(Curriculum).options(name=config.actor_name, get_if_exists=True, max_restarts=-1).remote(config)
