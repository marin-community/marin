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
Alert system for curriculum learning.

Alerts are configured per lesson and evaluated using existing curriculum statistics,
providing a natural extension of environment testing without duplicating logic.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from enum import Enum

from marin.rl.curriculum import LessonStats, compute_success_ratio, is_plateaued

if TYPE_CHECKING:
    from marin.rl.curriculum import LessonConfig

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels for alerts."""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class AlertResult:
    """Result of evaluating an alert condition."""

    triggered: bool
    """Whether the alert condition was met."""

    message: str
    """Human-readable message describing the alert."""

    health_status: HealthStatus
    """Health status indicating the severity of the alert."""

    metrics: dict[str, Any] = field(default_factory=dict)
    """Additional metrics associated with the alert."""

    timestamp: float = field(default_factory=time.time)
    """When the alert was triggered."""


class Alert(ABC):
    """Base class for lesson alerts.

    Alerts are evaluated using existing curriculum statistics and provide
    a way to monitor lesson health without duplicating statistical logic.
    """

    @abstractmethod
    def evaluate(
        self,
        lesson_id: str,
        stats: LessonStats,
        lesson_config: "LessonConfig",
        current_step: int,
        lesson_state: str,  # "locked", "active", "graduated"
        graduation_performance: float | None = None,
    ) -> AlertResult | None:
        """Evaluate whether this alert condition is met.

        Args:
            lesson_id: ID of the lesson being evaluated.
            stats: Current lesson statistics.
            lesson_config: Configuration for this lesson.
            current_step: Current training step.
            lesson_state: Current state of the lesson.
            graduation_performance: Performance at graduation (if graduated).

        Returns:
            AlertResult if condition is met, None otherwise.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this alert type."""
        pass


@dataclass
class GraduationRegressionAlert(Alert):
    """Alert when a graduated lesson's performance regresses below a threshold.

    This alert monitors graduated lessons to detect if performance has dropped
    significantly from the graduation baseline.
    """

    regression_threshold: float = 0.85
    """Warn if performance drops below this fraction of graduation level."""

    critical_threshold: float = 0.70
    """Critical warning if performance drops below this fraction."""

    cooldown_steps: int = 200
    """Steps between repeated alerts of this type."""

    @property
    def name(self) -> str:
        return "graduation_regression"

    def evaluate(
        self,
        lesson_id: str,
        stats: LessonStats,
        lesson_config: "LessonConfig",
        current_step: int,
        lesson_state: str,
        graduation_performance: float | None = None,
    ) -> AlertResult | None:
        if lesson_state != "graduated":
            return None

        # Compute graduation performance from eval stats if not provided
        if graduation_performance is None:
            eval_rewards = stats.eval_stats.reward_history
            if len(eval_rewards) == 0:
                return None
            # Use mean reward value at graduation time (before any regression)
            # Strategy: Use mean of eval rewards excluding recent window (which might contain regression)
            # This approximates graduation-time performance

            # Use mean of eval rewards excluding the most recent window
            # This assumes recent samples might contain regression, older samples represent graduation performance
            recent_window = min(20, len(eval_rewards))
            if len(eval_rewards) > recent_window:
                # Use mean of older samples (before potential regression)
                # Need at least a few samples for reliable baseline
                graduation_rewards = eval_rewards[:-recent_window]
                if len(graduation_rewards) > 0:
                    graduation_performance = float(np.mean(graduation_rewards))
                else:
                    # Fallback: use all samples if split would be empty
                    graduation_performance = float(np.mean(eval_rewards))
            else:
                # Not enough samples - use mean of all (fallback)
                # With < 20 samples, can't reliably detect regression anyway
                graduation_performance = float(np.mean(eval_rewards))

        # For graduated lessons, compare recent mean eval reward (not all-time mean)
        # Use recent window to detect current regression
        if len(stats.eval_stats.reward_history) > 0:
            eval_rewards = stats.eval_stats.reward_history
            # Use recent eval rewards (last 20 samples) for current performance
            recent_window = min(20, len(eval_rewards))
            current_perf = float(np.mean(eval_rewards[-recent_window:]))
        else:
            # Fallback to training stats if no eval data
            if len(stats.training_stats.reward_history) > 0:
                train_rewards = stats.training_stats.reward_history
                recent_window = min(20, len(train_rewards))
                current_perf = float(np.mean(train_rewards[-recent_window:]))
            else:
                return None

        if current_perf < graduation_performance * self.regression_threshold:
            health_status = (
                HealthStatus.CRITICAL
                if current_perf < graduation_performance * self.critical_threshold
                else HealthStatus.WARNING
            )

            # Calculate performance ratio safely (avoid division by zero)
            if graduation_performance > 1e-6:
                performance_ratio = current_perf / graduation_performance
                ratio_pct = performance_ratio * 100
            else:
                performance_ratio = float("inf") if current_perf > 0 else 0.0
                ratio_pct = float("inf")

            return AlertResult(
                triggered=True,
                message=(
                    f"Graduated lesson '{lesson_id}' performance dropped from "
                    f"{graduation_performance:.3f} to {current_perf:.3f} "
                    f"({ratio_pct:.1f}% of graduation level)"
                ),
                health_status=health_status,
                metrics={
                    "graduation_performance": graduation_performance,
                    "current_performance": current_perf,
                    "performance_ratio": performance_ratio,
                },
            )

        return None


@dataclass
class TrainingStalledAlert(Alert):
    """Alert when training shows no progress for an extended period.

    Detects stagnation by checking if performance has plateaued and hasn't
    improved beyond a threshold for a specified number of steps.
    """

    stagnation_window: int = 100
    """Number of steps without progress before alerting."""

    plateau_window: int | None = None
    """Window size for plateau detection. If None, uses lesson_config.plateau_window."""

    plateau_threshold: float | None = None
    """Threshold for plateau detection. If None, uses lesson_config.plateau_threshold."""

    cooldown_steps: int = 200
    """Steps between repeated alerts of this type."""

    @property
    def name(self) -> str:
        return "training_stalled"

    def evaluate(
        self,
        lesson_id: str,
        stats: LessonStats,
        lesson_config: "LessonConfig",
        current_step: int,
        lesson_state: str,
        graduation_performance: float | None = None,
    ) -> AlertResult | None:
        if lesson_state != "active":
            return None

        # Check if we have enough samples
        if len(stats.training_stats.reward_history) < self.stagnation_window:
            return None

        # Use lesson config defaults if not specified
        plateau_win = self.plateau_window if self.plateau_window is not None else lesson_config.plateau_window
        plateau_thresh = (
            self.plateau_threshold if self.plateau_threshold is not None else lesson_config.plateau_threshold
        )

        # Check if performance has plateaued
        if not is_plateaued(stats, window=plateau_win, threshold=plateau_thresh):
            return None

        # Check if we've been stuck for the stagnation window
        # If overall stats have plateaued and we have enough samples in the stagnation window,
        # we've been stuck for at least that long
        recent_rewards = stats.training_stats.reward_history[-self.stagnation_window :]

        # If we've plateaued and have enough samples, alert
        # The plateau check already ensures performance isn't improving
        current_perf = compute_success_ratio(stats, current_step)

        return AlertResult(
            triggered=True,
            message=(
                f"Lesson '{lesson_id}' showing no progress for "
                f"{self.stagnation_window} steps "
                f"(current performance: {current_perf:.3f})"
            ),
            health_status=HealthStatus.WARNING,
            metrics={
                "steps_without_progress": self.stagnation_window,
                "current_performance": current_perf,
                "mean_recent_performance": float(np.mean(recent_rewards)),
            },
        )


@dataclass
class PerformanceVolatilityAlert(Alert):
    """Alert when performance shows high volatility/variance.

    Detects unstable performance by computing coefficient of variation
    over recent samples.
    """

    volatility_threshold: float = 0.15
    """Coefficient of variation threshold for alerting."""

    window: int = 30
    """Number of recent samples to analyze."""

    cooldown_steps: int = 200
    """Steps between repeated alerts of this type."""

    @property
    def name(self) -> str:
        return "performance_volatility"

    def evaluate(
        self,
        lesson_id: str,
        stats: LessonStats,
        lesson_config: "LessonConfig",
        current_step: int,
        lesson_state: str,
        graduation_performance: float | None = None,
    ) -> AlertResult | None:
        if lesson_state not in ["active", "graduated"]:
            return None

        # Use training stats for active, eval stats for graduated
        reward_history = (
            stats.eval_stats.reward_history if lesson_state == "graduated" else stats.training_stats.reward_history
        )

        if len(reward_history) < self.window:
            return None

        recent = np.array(reward_history[-self.window :])
        mean_perf = np.mean(recent)
        std_perf = np.std(recent)

        if abs(mean_perf) > 1e-6:
            cv = std_perf / abs(mean_perf)
            if cv > self.volatility_threshold:
                mode = "eval" if lesson_state == "graduated" else "training"
                return AlertResult(
                    triggered=True,
                    message=(
                        f"Lesson '{lesson_id}' showing high performance volatility " f"(CV={cv:.3f}) in {mode} mode"
                    ),
                    health_status=HealthStatus.WARNING,
                    metrics={
                        "coefficient_of_variation": float(cv),
                        "mean_performance": float(mean_perf),
                        "std_performance": float(std_perf),
                        "state": lesson_state,
                    },
                )

        return None
