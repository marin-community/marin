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

"""Tests for the curriculum health monitoring system."""

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from marin.rl.health_monitor import (
    CurriculumHealthMonitor,
    EnvironmentHealthMetrics,
    HealthMonitorConfig,
    HealthStatus,
    Warning,
    WarningType,
)


class TestHealthMonitorConfig:
    """Tests for HealthMonitorConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = HealthMonitorConfig()
        assert config.enabled is True
        assert config.regression_threshold == 0.85
        assert config.critical_regression_threshold == 0.70
        assert config.stagnation_window == 100
        assert config.evaluation_frequency == 50
        assert config.warning_cooldown == 200
        assert config.volatility_threshold == 0.15
        assert config.performance_history_size == 100
        assert config.prolonged_training_threshold == 1000
        assert config.enable_wandb_alerts is True
        assert config.wandb_alert_rate_limit == 300

    def test_custom_config(self):
        """Test custom configuration values."""
        config = HealthMonitorConfig(
            enabled=False,
            regression_threshold=0.90,
            stagnation_window=50,
        )
        assert config.enabled is False
        assert config.regression_threshold == 0.90
        assert config.stagnation_window == 50


class TestCurriculumHealthMonitor:
    """Tests for CurriculumHealthMonitor."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return HealthMonitorConfig(
            regression_threshold=0.85,
            critical_regression_threshold=0.70,
            stagnation_window=50,
            warning_cooldown=100,
            volatility_threshold=0.15,
            prolonged_training_threshold=500,
            wandb_alert_rate_limit=10,  # Short for testing
        )

    @pytest.fixture
    def monitor(self, config):
        """Create a health monitor instance."""
        return CurriculumHealthMonitor(config)

    def test_initialization(self, monitor):
        """Test monitor initialization."""
        assert monitor.config is not None
        assert monitor.current_step == 0
        assert len(monitor.env_metrics) == 0
        assert len(monitor.active_warnings) == 0

    def test_update_creates_metrics(self, monitor):
        """Test that update creates environment metrics."""
        monitor.update(
            env_id="env1",
            performance=0.5,
            mode="training",
            state="active",
            current_step=10,
        )

        assert "env1" in monitor.env_metrics
        metrics = monitor.env_metrics["env1"]
        assert metrics.env_id == "env1"
        assert metrics.current_state == "active"
        assert metrics.current_train_performance == 0.5
        assert metrics.peak_train_performance == 0.5

    def test_state_transition_tracking(self, monitor):
        """Test that state transitions are tracked correctly."""
        # Start with active state
        monitor.update("env1", 0.3, "training", "active", 10)

        # Transition to graduated
        monitor.update("env1", 0.9, "eval", "graduated", 20)

        metrics = monitor.env_metrics["env1"]
        assert metrics.current_state == "graduated"
        assert metrics.graduation_performance == 0.9
        assert metrics.graduation_step == 20
        assert "env1" in monitor.graduation_baselines

    def test_graduated_regression_warning(self, monitor):
        """Test detection of graduated environment regression."""
        # First update eval performance, then transition to graduated
        monitor.update("env1", 0.9, "eval", "active", 100)
        monitor.update("env1", 0.9, "eval", "graduated", 100)

        # Performance drops below threshold
        monitor.update("env1", 0.7, "eval", "graduated", 200)

        warnings = monitor.check_warnings()
        assert len(warnings) == 1
        warning = warnings[0]
        assert warning.type == WarningType.GRADUATED_REGRESSION
        assert warning.env_id == "env1"
        assert warning.severity == "high"
        assert "dropped from" in warning.message

    def test_critical_regression_warning(self, monitor):
        """Test detection of critical regression."""
        # First update eval performance, then transition to graduated
        monitor.update("env1", 0.9, "eval", "active", 100)
        monitor.update("env1", 0.9, "eval", "graduated", 100)

        # Performance drops below critical threshold
        monitor.update("env1", 0.6, "eval", "graduated", 200)

        warnings = monitor.check_warnings()
        assert len(warnings) == 1
        assert warnings[0].severity == "critical"

    def test_training_stagnation_warning(self, monitor):
        """Test detection of training stagnation."""
        # Simulate no progress for many steps
        for step in range(100):
            monitor.update("env1", 0.3, "training", "active", step)

        # Manually set last progress step to simulate stagnation
        monitor.last_progress_step["env1"] = 0
        monitor.env_metrics["env1"].steps_since_last_progress = 100

        warnings = monitor.check_warnings()
        stagnation_warnings = [w for w in warnings if w.type == WarningType.TRAINING_STAGNATION]
        assert len(stagnation_warnings) == 1
        assert "no progress" in stagnation_warnings[0].message

    def test_prolonged_training_warning(self, monitor):
        """Test detection of prolonged training."""
        # Simulate long training
        monitor.update("env1", 0.5, "training", "active", 0)
        monitor.env_metrics["env1"].total_training_steps = 600

        warnings = monitor.check_warnings()
        prolonged_warnings = [w for w in warnings if w.type == WarningType.PROLONGED_TRAINING]
        assert len(prolonged_warnings) == 1
        assert "600 steps" in prolonged_warnings[0].message

    def test_performance_volatility_warning(self, monitor):
        """Test detection of performance volatility."""
        # Create volatile performance history
        env_id = "env1"
        performances = [0.2, 0.8, 0.3, 0.9, 0.1, 0.7, 0.4, 0.8]  # High variance

        for i, perf in enumerate(performances * 5):  # Repeat to fill history
            monitor.update(env_id, perf, "training", "active", i)

        # Manually set high variance
        monitor.env_metrics[env_id].performance_variance = 0.5

        warnings = monitor.check_warnings()
        volatility_warnings = [w for w in warnings if w.type == WarningType.PERFORMANCE_VOLATILITY]
        assert len(volatility_warnings) == 1
        assert "high performance volatility" in volatility_warnings[0].message

    def test_warning_cooldown(self, monitor):
        """Test that warning cooldown prevents spam."""
        # Graduate and cause regression
        monitor.update("env1", 0.9, "eval", "active", 100)
        monitor.update("env1", 0.9, "eval", "graduated", 100)
        monitor.update("env1", 0.7, "eval", "graduated", 110)

        # First check should generate warning
        warnings1 = monitor.check_warnings()
        assert len(warnings1) == 1

        # Second check within cooldown should not
        monitor.current_step = 120  # Still within cooldown
        warnings2 = monitor.check_warnings()
        assert len(warnings2) == 0

        # After cooldown, warning should be issued again
        monitor.current_step = 250  # Past cooldown
        monitor.update("env1", 0.7, "eval", "graduated", 250)
        warnings3 = monitor.check_warnings()
        assert len(warnings3) == 1

    def test_performance_trend_computation(self, monitor):
        """Test performance trend computation."""
        # Create improving trend
        for i in range(50):
            perf = 0.3 + i * 0.01  # Steadily improving
            monitor.update("env1", perf, "training", "active", i)

        metrics = monitor.env_metrics["env1"]
        assert metrics.performance_trend == "improving"

        # Create regressing trend
        for i in range(50, 100):
            perf = 0.8 - (i - 50) * 0.01  # Steadily declining
            monitor.update("env2", perf, "training", "active", i)

        metrics2 = monitor.env_metrics["env2"]
        assert metrics2.performance_trend == "regressing"

    def test_health_status_updates(self, monitor):
        """Test that health status is updated based on warnings."""
        # Start healthy
        monitor.update("env1", 0.5, "training", "active", 0)
        monitor.check_warnings()
        assert monitor.env_metrics["env1"].health_status == HealthStatus.HEALTHY

        # Create critical warning
        monitor.update("env2", 0.9, "eval", "active", 100)
        monitor.update("env2", 0.9, "eval", "graduated", 100)
        monitor.update("env2", 0.5, "eval", "graduated", 200)  # Critical regression
        monitor.check_warnings()
        assert monitor.env_metrics["env2"].health_status == HealthStatus.CRITICAL

    def test_health_summary(self, monitor):
        """Test health summary generation."""
        # Create various environments in different states
        monitor.update("env1", 0.5, "training", "active", 10)
        monitor.update("env2", 0.9, "eval", "active", 20)
        monitor.update("env2", 0.9, "eval", "graduated", 20)
        monitor.update("env3", 0.0, "training", "locked", 30)

        # Create a regression
        monitor.update("env2", 0.7, "eval", "graduated", 40)
        monitor.check_warnings()

        summary = monitor.get_health_summary()
        assert summary["total_environments"] == 3
        assert summary["state_distribution"]["active"] == 1
        assert summary["state_distribution"]["graduated"] == 1
        assert summary["state_distribution"]["locked"] == 1
        assert summary["graduated_regressions"] == 1

    def test_environment_report(self, monitor):
        """Test detailed environment report."""
        monitor.update("env1", 0.5, "training", "active", 10)
        monitor.update("env1", 0.6, "eval", "active", 20)
        monitor.env_metrics["env1"].total_training_steps = 100

        report = monitor.get_environment_report("env1")
        assert report is not None
        assert report["env_id"] == "env1"
        assert report["current_state"] == "active"
        assert report["performance"]["current_train"] == 0.5
        assert report["performance"]["current_eval"] == 0.6
        assert report["training"]["total_steps"] == 100

        # Test non-existent environment
        assert monitor.get_environment_report("nonexistent") is None

    def test_wandb_alert_rate_limiting(self, monitor):
        """Test wandb alert rate limiting."""
        # First alert should be allowed
        assert monitor.should_send_wandb_alert(WarningType.GRADUATED_REGRESSION) is True

        # Immediate second alert should be blocked
        assert monitor.should_send_wandb_alert(WarningType.GRADUATED_REGRESSION) is False

        # Different warning type should be allowed
        assert monitor.should_send_wandb_alert(WarningType.TRAINING_STAGNATION) is True

        # After rate limit period, should be allowed again
        time.sleep(monitor.config.wandb_alert_rate_limit + 0.1)
        assert monitor.should_send_wandb_alert(WarningType.GRADUATED_REGRESSION) is True

    def test_disabled_monitoring(self):
        """Test that disabled monitoring doesn't generate warnings."""
        config = HealthMonitorConfig(enabled=False)
        monitor = CurriculumHealthMonitor(config)

        # Create conditions that would normally generate warnings
        monitor.update("env1", 0.9, "eval", "graduated", 100)
        monitor.update("env1", 0.5, "eval", "graduated", 200)

        warnings = monitor.check_warnings()
        assert len(warnings) == 0

    def test_progress_detection(self, monitor):
        """Test progress detection logic."""
        # Simulate improving performance
        for i in range(40):
            if i < 20:
                perf = 0.3  # Flat performance
            else:
                perf = 0.3 + (i - 20) * 0.02  # Improving

            monitor.update("env1", perf, "training", "active", i)

        # Should detect progress after improvement starts
        assert monitor._is_making_progress("env1", 0.7)

        # Simulate flat performance with enough samples for comparison
        # Need at least 40 samples (2 * window of 20) to properly test
        for i in range(50):
            monitor.update("env2", 0.5, "training", "active", i)

        # Should not detect progress for flat performance
        assert not monitor._is_making_progress("env2", 0.5)

    def test_multiple_environments(self, monitor):
        """Test monitoring multiple environments simultaneously."""
        envs = ["env1", "env2", "env3"]
        states = ["active", "graduated", "locked"]

        for env_id, state in zip(envs, states):
            for step in range(10):
                monitor.update(env_id, 0.5 + step * 0.05, "training", state, step * 10)

        assert len(monitor.env_metrics) == 3
        for env_id in envs:
            assert env_id in monitor.env_metrics

        # Check that each environment has independent metrics
        assert monitor.env_metrics["env1"].current_state == "active"
        assert monitor.env_metrics["env2"].current_state == "graduated"
        assert monitor.env_metrics["env3"].current_state == "locked"


class TestIntegrationWithCurriculum:
    """Test integration with the Curriculum class."""

    @pytest.fixture
    def mock_wandb(self):
        """Mock wandb for testing."""
        # Mock wandb at the module level
        import sys
        mock = MagicMock()
        mock.run = MagicMock()
        mock.alert = MagicMock()
        mock.log = MagicMock()
        sys.modules['wandb'] = mock
        yield mock
        # Clean up
        if 'wandb' in sys.modules:
            del sys.modules['wandb']

    def test_curriculum_integration(self, mock_wandb):
        """Test that health monitoring integrates with curriculum."""
        from marin.rl.curriculum import Curriculum, CurriculumConfig, LessonConfig
        from marin.rl.environments.base import EnvConfig
        from marin.rl.types import RolloutStats

        # Create a simple curriculum config with health monitoring
        health_config = HealthMonitorConfig(
            enabled=True,
            regression_threshold=0.85,
            stagnation_window=10,
            enable_wandb_alerts=True,
        )

        env_config = EnvConfig(env_name="test_env")
        lesson_config = LessonConfig(
            lesson_id="lesson1",
            env_config=env_config,
            stop_threshold=0.9,
        )

        curriculum_config = CurriculumConfig(
            lessons={"lesson1": lesson_config},
            health_monitor_config=health_config,
        )

        curriculum = Curriculum(curriculum_config)

        # Verify health monitor is initialized
        assert curriculum.health_monitor is not None
        assert curriculum.health_monitor.config.enabled is True

        # Simulate some rollouts
        rollout_stats = [
            RolloutStats(episode_reward=0.5, env_example_id="ex1", lesson_id="lesson1"),
            RolloutStats(episode_reward=0.6, env_example_id="ex2", lesson_id="lesson1"),
        ]

        curriculum.update_lesson_stats(rollout_stats, "training", 10)

        # Check that health metrics are included in curriculum metrics
        metrics = curriculum.get_metrics()
        assert "health" in metrics
        assert "environment_health" in metrics

    def test_wandb_alert_sending(self, mock_wandb):
        """Test that wandb alerts are sent correctly."""
        from marin.rl.curriculum import Curriculum, CurriculumConfig, LessonConfig
        from marin.rl.environments.base import EnvConfig

        health_config = HealthMonitorConfig(
            enabled=True,
            regression_threshold=0.85,
            enable_wandb_alerts=True,
        )

        env_config = EnvConfig(env_name="test_env")
        lesson_config = LessonConfig(
            lesson_id="lesson1",
            env_config=env_config,
        )

        curriculum_config = CurriculumConfig(
            lessons={"lesson1": lesson_config},
            health_monitor_config=health_config,
        )

        curriculum = Curriculum(curriculum_config)

        # Create a warning
        warning = Warning(
            type=WarningType.GRADUATED_REGRESSION,
            env_id="lesson1",
            message="Test regression",
            severity="high",
            metrics={"test": 1.0},
        )

        # Send the alert
        curriculum._send_wandb_alert(warning)

        # Verify wandb.alert was called
        mock_wandb.alert.assert_called_once()
        call_args = mock_wandb.alert.call_args
        assert "Curriculum Health" in call_args[1]["title"]
        assert "Test regression" in call_args[1]["text"]
