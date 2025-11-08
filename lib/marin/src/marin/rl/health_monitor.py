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
Health monitoring system for RL curriculum learning.

This module provides comprehensive monitoring of environment performance,
automated warning generation, and regression detection for graduated environments.
"""

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class WarningType(Enum):
    """Types of warnings that can be generated."""

    GRADUATED_REGRESSION = "graduated_regression"
    TRAINING_STAGNATION = "training_stagnation"
    PERFORMANCE_VOLATILITY = "performance_volatility"
    UNEXPECTED_TRANSITION = "unexpected_transition"
    PROLONGED_TRAINING = "prolonged_training"


class HealthStatus(Enum):
    """Health status levels for environments."""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class HealthWarning:
    """A warning about environment health."""

    type: WarningType
    env_id: str
    message: str
    severity: str  # "low", "medium", "high", "critical"
    timestamp: float = field(default_factory=time.time)
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvironmentHealthMetrics:
    """Health metrics for a single environment."""

    env_id: str
    current_state: str  # "locked", "active", "graduated"

    # Performance tracking
    current_train_performance: float = 0.0
    current_eval_performance: float = 0.0
    peak_train_performance: float = 0.0
    peak_eval_performance: float = 0.0
    performance_trend: str = "stable"  # "improving", "stable", "regressing"

    # State duration
    time_in_current_state: int = 0
    total_training_steps: int = 0
    steps_since_last_progress: int = 0

    # Graduation tracking
    graduation_performance: float | None = None
    graduation_step: int | None = None

    # Health indicators
    health_status: HealthStatus = HealthStatus.HEALTHY
    warnings: list[HealthWarning] = field(default_factory=list)

    # Performance volatility
    performance_variance: float = 0.0
    recent_performance_changes: list[float] = field(default_factory=list)


@dataclass
class HealthMonitorConfig:
    """Configuration for health monitoring system."""

    enabled: bool = True
    """Whether health monitoring is enabled."""

    regression_threshold: float = 0.85
    """Warn if graduated env performance drops below this fraction of graduation level."""

    critical_regression_threshold: float = 0.70
    """Critical warning if performance drops below this fraction."""

    stagnation_window: int = 100
    """Number of steps without progress before warning about stagnation."""

    evaluation_frequency: int = 50
    """How often to evaluate all environments (in training steps)."""

    warning_cooldown: int = 200
    """Steps before re-issuing the same warning."""

    volatility_threshold: float = 0.15
    """Coefficient of variation threshold for volatility warnings."""

    performance_history_size: int = 100
    """Number of recent performance samples to track."""

    prolonged_training_threshold: int = 1000
    """Warn if environment stays in training for this many steps."""

    enable_wandb_alerts: bool = True
    """Whether to send alerts to wandb."""

    wandb_alert_rate_limit: int = 300
    """Minimum seconds between wandb alerts of the same type."""


class CurriculumHealthMonitor:
    """Monitors health of curriculum learning and generates warnings."""

    def __init__(self, config: HealthMonitorConfig):
        self.config = config

        # Performance tracking
        self.historical_performance: dict[str, dict[str, deque]] = defaultdict(
            lambda: {
                "training": deque(maxlen=config.performance_history_size),
                "eval": deque(maxlen=config.performance_history_size),
            }
        )

        # Graduation baselines
        self.graduation_baselines: dict[str, dict[str, Any]] = {}

        # State tracking
        self.environment_states: dict[str, str] = {}
        self.state_transition_times: dict[str, dict[str, int]] = defaultdict(dict)
        self.last_progress_step: dict[str, int] = defaultdict(int)

        # Warning management
        self.active_warnings: dict[str, list[HealthWarning]] = defaultdict(list)
        self.warning_cooldowns: dict[str, dict[WarningType, int]] = defaultdict(dict)
        self.last_wandb_alert: dict[WarningType, float] = {}

        # Metrics for each environment
        self.env_metrics: dict[str, EnvironmentHealthMetrics] = {}

        # Step counter
        self.current_step = 0

    def update(
        self,
        env_id: str,
        performance: float,
        mode: str,  # "training" or "eval"
        state: str,  # "locked", "active", "graduated"
        current_step: int,
    ) -> None:
        """Update health metrics for an environment.

        Args:
            env_id: Environment identifier.
            performance: Current performance metric (e.g., success rate).
            mode: Whether this is training or evaluation performance.
            state: Current state of the environment.
            current_step: Current training step.
        """
        self.current_step = current_step

        # Initialize metrics if needed
        if env_id not in self.env_metrics:
            self.env_metrics[env_id] = EnvironmentHealthMetrics(env_id=env_id, current_state=state)

        metrics = self.env_metrics[env_id]

        # Update performance history first (before state transition)
        self.historical_performance[env_id][mode].append((current_step, performance))

        # Update current performance (before state transition so graduation has correct performance)
        if mode == "training":
            metrics.current_train_performance = performance
            metrics.peak_train_performance = max(metrics.peak_train_performance, performance)
            if state == "active":
                metrics.total_training_steps += 1
        else:  # eval
            metrics.current_eval_performance = performance
            metrics.peak_eval_performance = max(metrics.peak_eval_performance, performance)

        # Update state tracking (after performance update)
        if state != metrics.current_state:
            self._handle_state_transition(env_id, metrics.current_state, state)
            metrics.current_state = state

        # Update state duration
        if env_id in self.state_transition_times and state in self.state_transition_times[env_id]:
            metrics.time_in_current_state = current_step - self.state_transition_times[env_id][state]

        # Check for progress
        if mode == "training" and state == "active":
            if self._is_making_progress(env_id, performance):
                self.last_progress_step[env_id] = current_step
            metrics.steps_since_last_progress = current_step - self.last_progress_step.get(env_id, current_step)

        # Update performance trend
        metrics.performance_trend = self._compute_performance_trend(env_id, mode)

        # Update volatility metrics
        metrics.performance_variance = self._compute_performance_variance(env_id, mode)

    def _handle_state_transition(self, env_id: str, old_state: str, new_state: str) -> None:
        """Handle environment state transitions."""
        logger.info(f"Environment {env_id} transitioning from {old_state} to {new_state}")

        # Record transition time
        self.state_transition_times[env_id][new_state] = self.current_step

        # Handle graduation
        if new_state == "graduated":
            metrics = self.env_metrics[env_id]
            # Use the current eval performance as graduation baseline
            # Note: This will be the eval performance at the time of graduation
            graduation_perf = (
                metrics.current_eval_performance
                if metrics.current_eval_performance > 0
                else metrics.current_train_performance
            )
            metrics.graduation_performance = graduation_perf
            metrics.graduation_step = self.current_step

            # Store graduation baseline for regression detection
            self.graduation_baselines[env_id] = {
                "performance": graduation_perf,
                "step": self.current_step,
                "train_performance": metrics.current_train_performance,
            }

            logger.info(
                f"Environment {env_id} graduated with performance"
                f"{metrics.graduation_performance:.3f} at step {self.current_step}"
            )

    def check_warnings(self) -> list[HealthWarning]:
        """Check for health warnings across all environments.

        Returns:
            List of new warnings generated.
        """
        if not self.config.enabled:
            return []

        new_warnings = []

        for env_id, metrics in self.env_metrics.items():
            # Check graduated regression
            if metrics.current_state == "graduated":
                warning = self._check_graduated_regression(env_id, metrics)
                if warning and self._should_issue_warning(env_id, warning.type):
                    new_warnings.append(warning)
                    self._record_warning(env_id, warning)

            # Check training stagnation
            elif metrics.current_state == "active":
                # Stagnation check
                warning = self._check_training_stagnation(env_id, metrics)
                if warning and self._should_issue_warning(env_id, warning.type):
                    new_warnings.append(warning)
                    self._record_warning(env_id, warning)

                # Prolonged training check
                warning = self._check_prolonged_training(env_id, metrics)
                if warning and self._should_issue_warning(env_id, warning.type):
                    new_warnings.append(warning)
                    self._record_warning(env_id, warning)

            # Check performance volatility (for all active/graduated envs)
            if metrics.current_state in ["active", "graduated"]:
                warning = self._check_performance_volatility(env_id, metrics)
                if warning and self._should_issue_warning(env_id, warning.type):
                    new_warnings.append(warning)
                    self._record_warning(env_id, warning)

        # Update health status for all environments
        self._update_health_status()

        return new_warnings

    def _check_graduated_regression(self, env_id: str, metrics: EnvironmentHealthMetrics) -> HealthWarning | None:
        """Check if a graduated environment has regressed."""
        if env_id not in self.graduation_baselines:
            return None

        baseline = self.graduation_baselines[env_id]["performance"]
        current_perf = metrics.current_eval_performance

        # Check for regression
        if current_perf < baseline * self.config.regression_threshold:
            severity = "critical" if current_perf < baseline * self.config.critical_regression_threshold else "high"

            return HealthWarning(
                type=WarningType.GRADUATED_REGRESSION,
                env_id=env_id,
                message=(
                    f"Graduated environment '{env_id}' performance dropped from "
                    f"{baseline:.3f} to {current_perf:.3f} "
                    f"({(current_perf/baseline)*100:.1f}% of graduation level)"
                ),
                severity=severity,
                metrics={
                    "graduation_performance": baseline,
                    "current_performance": current_perf,
                    "performance_ratio": current_perf / baseline,
                    "steps_since_graduation": self.current_step - self.graduation_baselines[env_id]["step"],
                },
            )

        return None

    def _check_training_stagnation(self, env_id: str, metrics: EnvironmentHealthMetrics) -> HealthWarning | None:
        """Check if training has stagnated."""
        if metrics.steps_since_last_progress > self.config.stagnation_window:
            return HealthWarning(
                type=WarningType.TRAINING_STAGNATION,
                env_id=env_id,
                message=(
                    f"Environment '{env_id}' showing no progress for "
                    f"{metrics.steps_since_last_progress} steps "
                    f"(current performance: {metrics.current_train_performance:.3f})"
                ),
                severity="medium",
                metrics={
                    "steps_without_progress": metrics.steps_since_last_progress,
                    "current_performance": metrics.current_train_performance,
                    "peak_performance": metrics.peak_train_performance,
                },
            )

        return None

    def _check_prolonged_training(self, env_id: str, metrics: EnvironmentHealthMetrics) -> HealthWarning | None:
        """Check if environment has been training for too long."""
        if metrics.total_training_steps > self.config.prolonged_training_threshold:
            return HealthWarning(
                type=WarningType.PROLONGED_TRAINING,
                env_id=env_id,
                message=(
                    f"Environment '{env_id}' has been training for "
                    f"{metrics.total_training_steps} steps without graduating"
                ),
                severity="low",
                metrics={
                    "total_training_steps": metrics.total_training_steps,
                    "current_performance": metrics.current_train_performance,
                    "performance_trend": metrics.performance_trend,
                },
            )

        return None

    def _check_performance_volatility(self, env_id: str, metrics: EnvironmentHealthMetrics) -> HealthWarning | None:
        """Check for high performance volatility."""
        if metrics.performance_variance > self.config.volatility_threshold:
            mode = "eval" if metrics.current_state == "graduated" else "training"
            return HealthWarning(
                type=WarningType.PERFORMANCE_VOLATILITY,
                env_id=env_id,
                message=(
                    f"Environment '{env_id}' showing high performance volatility "
                    f"(CV={metrics.performance_variance:.3f}) in {mode} mode"
                ),
                severity="low",
                metrics={
                    "coefficient_of_variation": metrics.performance_variance,
                    "current_performance": (
                        metrics.current_eval_performance if mode == "eval" else metrics.current_train_performance
                    ),
                    "state": metrics.current_state,
                },
            )

        return None

    def _is_making_progress(self, env_id: str, current_performance: float, window: int = 20) -> bool:
        """Check if environment is making progress."""
        history = self.historical_performance[env_id]["training"]
        if len(history) < window:
            return True  # Assume progress early on

        # Need at least 2*window samples to compare
        if len(history) < 2 * window:
            return True

        recent_perfs = [p for _, p in list(history)[-window:]]
        older_perfs = [p for _, p in list(history)[-2 * window : -window]]

        if not older_perfs:
            return True

        # Progress if recent average is better than older average
        return np.mean(recent_perfs) > np.mean(older_perfs) * 1.01  # 1% improvement threshold

    def _compute_performance_trend(self, env_id: str, mode: str, window: int = 30) -> str:
        """Compute performance trend (improving/stable/regressing)."""
        history = self.historical_performance[env_id][mode]
        if len(history) < window:
            return "stable"

        recent = [p for _, p in list(history)[-window:]]

        # Linear regression for trend
        x = np.arange(len(recent))
        from scipy import stats

        result = stats.linregress(x, recent)
        slope = result.slope

        mean_perf = np.mean(recent)
        if abs(mean_perf) > 1e-6:
            relative_slope = slope / abs(mean_perf)
            if relative_slope > 0.01:
                return "improving"
            elif relative_slope < -0.01:
                return "regressing"

        return "stable"

    def _compute_performance_variance(self, env_id: str, mode: str) -> float:
        """Compute coefficient of variation for performance."""
        history = self.historical_performance[env_id][mode]
        if len(history) < 10:
            return 0.0

        recent = [p for _, p in list(history)[-30:]]
        mean_perf = np.mean(recent)
        if abs(mean_perf) > 1e-6:
            return np.std(recent) / abs(mean_perf)
        return 0.0

    def _should_issue_warning(self, env_id: str, warning_type: WarningType) -> bool:
        """Check if we should issue a warning (respecting cooldowns)."""
        if env_id in self.warning_cooldowns and warning_type in self.warning_cooldowns[env_id]:
            last_step = self.warning_cooldowns[env_id][warning_type]
            if self.current_step - last_step < self.config.warning_cooldown:
                return False
        return True

    def _record_warning(self, env_id: str, warning: HealthWarning) -> None:
        """Record a warning and update cooldowns."""
        self.active_warnings[env_id].append(warning)
        self.warning_cooldowns[env_id][warning.type] = self.current_step

        # Keep only recent warnings
        self.active_warnings[env_id] = self.active_warnings[env_id][-10:]

    def _update_health_status(self) -> None:
        """Update health status for all environments based on active warnings."""
        for env_id, metrics in self.env_metrics.items():
            active = [
                w
                for w in self.active_warnings[env_id]
                if self.current_step - self.warning_cooldowns[env_id].get(w.type, 0) < 500
            ]

            if any(w.severity == "critical" for w in active):
                metrics.health_status = HealthStatus.CRITICAL
            elif any(w.severity in ["high", "medium"] for w in active):
                metrics.health_status = HealthStatus.WARNING
            else:
                metrics.health_status = HealthStatus.HEALTHY

            metrics.warnings = active

    def get_health_summary(self) -> dict[str, Any]:
        """Get a summary of curriculum health.

        Returns:
            Dictionary with health metrics and statistics.
        """
        total_envs = len(self.env_metrics)
        healthy = sum(1 for m in self.env_metrics.values() if m.health_status == HealthStatus.HEALTHY)
        warning = sum(1 for m in self.env_metrics.values() if m.health_status == HealthStatus.WARNING)
        critical = sum(1 for m in self.env_metrics.values() if m.health_status == HealthStatus.CRITICAL)

        # Count by state
        locked = sum(1 for m in self.env_metrics.values() if m.current_state == "locked")
        active = sum(1 for m in self.env_metrics.values() if m.current_state == "active")
        graduated = sum(1 for m in self.env_metrics.values() if m.current_state == "graduated")

        # Regression statistics
        regressions = sum(
            1
            for env_id, m in self.env_metrics.items()
            if m.current_state == "graduated"
            and env_id in self.graduation_baselines
            and m.current_eval_performance
            < self.graduation_baselines[env_id]["performance"] * self.config.regression_threshold
        )

        return {
            "total_environments": total_envs,
            "health_distribution": {"healthy": healthy, "warning": warning, "critical": critical},
            "state_distribution": {"locked": locked, "active": active, "graduated": graduated},
            "graduated_regressions": regressions,
            "active_warnings": sum(len(w) for w in self.active_warnings.values()),
            "average_training_steps": (
                np.mean([m.total_training_steps for m in self.env_metrics.values() if m.total_training_steps > 0])
                if any(m.total_training_steps > 0 for m in self.env_metrics.values())
                else 0
            ),
        }

    def get_environment_report(self, env_id: str) -> dict[str, Any] | None:
        """Get detailed health report for a specific environment.

        Args:
            env_id: Environment identifier.

        Returns:
            Detailed health report or None if environment not found.
        """
        if env_id not in self.env_metrics:
            return None

        metrics = self.env_metrics[env_id]
        report = {
            "env_id": env_id,
            "current_state": metrics.current_state,
            "health_status": metrics.health_status.value,
            "performance": {
                "current_train": metrics.current_train_performance,
                "current_eval": metrics.current_eval_performance,
                "peak_train": metrics.peak_train_performance,
                "peak_eval": metrics.peak_eval_performance,
                "trend": metrics.performance_trend,
                "variance": metrics.performance_variance,
            },
            "training": {
                "total_steps": metrics.total_training_steps,
                "time_in_state": metrics.time_in_current_state,
                "steps_since_progress": metrics.steps_since_last_progress,
            },
            "warnings": [{"type": w.type.value, "message": w.message, "severity": w.severity} for w in metrics.warnings],
        }

        if metrics.graduation_performance is not None:
            report["graduation"] = {
                "performance": metrics.graduation_performance,
                "step": metrics.graduation_step,
                "current_ratio": (
                    metrics.current_eval_performance / metrics.graduation_performance
                    if metrics.graduation_performance > 0
                    else 0
                ),
            }

        return report

    def should_send_wandb_alert(self, warning_type: WarningType) -> bool:
        """Check if we should send a wandb alert (rate limiting).

        Args:
            warning_type: Type of warning to check.

        Returns:
            True if alert should be sent.
        """
        if not self.config.enable_wandb_alerts:
            return False

        current_time = time.time()
        if warning_type in self.last_wandb_alert:
            if current_time - self.last_wandb_alert[warning_type] < self.config.wandb_alert_rate_limit:
                return False

        self.last_wandb_alert[warning_type] = current_time
        return True
