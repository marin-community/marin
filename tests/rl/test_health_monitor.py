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

import logging
from unittest.mock import MagicMock, patch


from marin.rl.alerts import (
    GraduationRegressionAlert,
    PerformanceVolatilityAlert,
    TrainingStalledAlert,
)
from marin.rl.curriculum import Curriculum, CurriculumConfig, LessonConfig
from marin.rl.environments.base import EnvConfig
from marin.rl.types import RolloutStats

logger = logging.getLogger(__name__)


def create_test_rollout_stats(episode_reward: float, lesson_id: str = "test") -> RolloutStats:
    """Helper to create test rollout stats."""
    return RolloutStats(lesson_id=lesson_id, episode_reward=episode_reward, env_example_id="test_example")


def create_test_curriculum_with_alerts(alerts=None):
    """Create a test curriculum with optional alerts."""
    if alerts is None:
        alerts = []

    env_config = EnvConfig(
        env_class="marin.rl.environments.mock_env.MockEnv",
        env_args={"env_id": "test_env", "env_args": {}},
    )
    lesson_config = LessonConfig(
        lesson_id="lesson1",
        env_config=env_config,
        stop_threshold=0.9,
        plateau_window=20,
        alerts=alerts,
    )

    curriculum_config = CurriculumConfig(
        lessons={"lesson1": lesson_config},
        enable_wandb_alerts=False,  # Disable for testing unless explicitly testing wandb
    )

    return Curriculum(curriculum_config)


class TestGraduationRegressionAlert:
    """Test graduation regression detection behavior."""

    def test_alert_triggers_on_regression(self):
        """Test that alert triggers when graduated lesson performance drops."""
        curriculum = create_test_curriculum_with_alerts(
            [GraduationRegressionAlert(regression_threshold=0.85, critical_threshold=0.70)]
        )

        # Train lesson to graduation
        for step in range(100):
            rewards = [0.9] * 10  # High performance
            rollout_stats = [create_test_rollout_stats(r, "lesson1") for r in rewards]
            curriculum.update_lesson_stats(rollout_stats, "training", step)

        # Graduate the lesson (meets threshold and plateaus)
        for step in range(100, 120):
            rewards = [0.95] * 10  # Peak performance
            rollout_stats = [create_test_rollout_stats(r, "lesson1") for r in rewards]
            curriculum.update_lesson_stats(rollout_stats, "eval", step)

        assert "lesson1" in curriculum.graduated

        # Now cause regression
        with patch("marin.rl.curriculum.logger") as mock_logger:
            for step in range(120, 130):
                rewards = [0.7] * 10  # Performance drops below threshold
                rollout_stats = [create_test_rollout_stats(r, "lesson1") for r in rewards]
                curriculum.update_lesson_stats(rollout_stats, "eval", step)

            # Check that warning was logged
            warning_calls = [call for call in mock_logger.warning.call_args_list if "performance dropped" in str(call)]
            assert len(warning_calls) > 0

    def test_graduation_performance_stored(self):
        """Test that graduation performance is stored when lesson graduates."""
        curriculum = create_test_curriculum_with_alerts([GraduationRegressionAlert(regression_threshold=0.85)])

        # Train lesson to graduation
        for step in range(100):
            rewards = [0.9] * 10
            rollout_stats = [create_test_rollout_stats(r, "lesson1") for r in rewards]
            curriculum.update_lesson_stats(rollout_stats, "training", step)

        # Graduate the lesson
        for step in range(100, 120):
            rewards = [0.95] * 10
            rollout_stats = [create_test_rollout_stats(r, "lesson1") for r in rewards]
            curriculum.update_lesson_stats(rollout_stats, "eval", step)

        assert "lesson1" in curriculum.graduated
        # Verify graduation performance was stored
        assert "lesson1" in curriculum.graduation_performances
        assert curriculum.graduation_performances["lesson1"] > 0.0
        # Should be close to the success ratio at graduation time
        from marin.rl.curriculum import compute_success_ratio

        expected_perf = compute_success_ratio(curriculum.stats["lesson1"], curriculum.current_step)
        assert abs(curriculum.graduation_performances["lesson1"] - expected_perf) < 0.1


class TestTrainingStalledAlert:
    """Test training stagnation detection behavior."""

    def test_alert_triggers_on_stagnation(self):
        """Test that alert triggers when training plateaus."""
        curriculum = create_test_curriculum_with_alerts([TrainingStalledAlert(stagnation_window=30, plateau_window=20)])

        # Train with improving performance
        for step in range(50):
            perf = 0.3 + step * 0.01
            rewards = [perf] * 5
            rollout_stats = [create_test_rollout_stats(r, "lesson1") for r in rewards]
            curriculum.update_lesson_stats(rollout_stats, "training", step)

        # Now plateau for many steps
        with patch("marin.rl.curriculum.logger") as mock_logger:
            for step in range(50, 100):
                rewards = [0.8] * 5  # Flat performance
                rollout_stats = [create_test_rollout_stats(r, "lesson1") for r in rewards]
                curriculum.update_lesson_stats(rollout_stats, "training", step)

            # Should detect stagnation
            warning_calls = [call for call in mock_logger.warning.call_args_list if "no progress" in str(call)]
            assert len(warning_calls) > 0

    def test_alert_does_not_trigger_during_improvement(self):
        """Test that alert doesn't trigger when performance is improving."""
        curriculum = create_test_curriculum_with_alerts([TrainingStalledAlert(stagnation_window=30, plateau_window=20)])

        with patch("marin.rl.curriculum.logger") as mock_logger:
            # Continuously improving performance
            for step in range(100):
                perf = 0.3 + step * 0.005
                rewards = [perf] * 5
                rollout_stats = [create_test_rollout_stats(r, "lesson1") for r in rewards]
                curriculum.update_lesson_stats(rollout_stats, "training", step)

            # Should not trigger stagnation alert
            warning_calls = [call for call in mock_logger.warning.call_args_list if "no progress" in str(call)]
            assert len(warning_calls) == 0


class TestPerformanceVolatilityAlert:
    """Test performance volatility detection behavior."""

    def test_alert_triggers_on_high_volatility(self):
        """Test that alert triggers when performance is highly variable."""
        curriculum = create_test_curriculum_with_alerts(
            [PerformanceVolatilityAlert(volatility_threshold=0.2, window=30)]
        )

        with patch("marin.rl.curriculum.logger") as mock_logger:
            # Create highly volatile performance
            for step in range(50):
                # Alternate between high and low performance
                perf = 0.9 if step % 2 == 0 else 0.1
                rewards = [perf] * 5
                rollout_stats = [create_test_rollout_stats(r, "lesson1") for r in rewards]
                curriculum.update_lesson_stats(rollout_stats, "training", step)

            # Should detect volatility
            warning_calls = [call for call in mock_logger.warning.call_args_list if "volatility" in str(call)]
            assert len(warning_calls) > 0

    def test_alert_does_not_trigger_on_stable_performance(self):
        """Test that alert doesn't trigger for stable performance."""
        curriculum = create_test_curriculum_with_alerts(
            [PerformanceVolatilityAlert(volatility_threshold=0.2, window=30)]
        )

        with patch("marin.rl.curriculum.logger") as mock_logger:
            # Stable performance
            for step in range(50):
                rewards = [0.5] * 5  # Consistent performance
                rollout_stats = [create_test_rollout_stats(r, "lesson1") for r in rewards]
                curriculum.update_lesson_stats(rollout_stats, "training", step)

            # Should not trigger volatility alert
            warning_calls = [call for call in mock_logger.warning.call_args_list if "volatility" in str(call)]
            assert len(warning_calls) == 0


class TestAlertIntegration:
    """Test alert system integration with curriculum."""

    def test_multiple_alerts_on_same_lesson(self):
        """Test that multiple alerts can be configured for one lesson."""
        curriculum = create_test_curriculum_with_alerts(
            [
                TrainingStalledAlert(stagnation_window=30),
                PerformanceVolatilityAlert(volatility_threshold=0.2),
            ]
        )

        # Test volatility alert triggers
        with patch("marin.rl.curriculum.logger") as mock_logger:
            curriculum._last_alert_time.clear()
            # Volatile performance
            for step in range(50):
                perf = 0.5 if step % 2 == 0 else 0.3  # Volatile
                rewards = [perf] * 5
                rollout_stats = [create_test_rollout_stats(r, "lesson1") for r in rewards]
                curriculum.update_lesson_stats(rollout_stats, "training", step)

            # Volatility alert should trigger
            volatility_calls = [call for call in mock_logger.warning.call_args_list if "volatility" in str(call)]
            assert len(volatility_calls) > 0

        # Test stagnation alert triggers (separate scenario)
        curriculum2 = create_test_curriculum_with_alerts([TrainingStalledAlert(stagnation_window=30, plateau_window=20)])
        with patch("marin.rl.curriculum.logger") as mock_logger:
            curriculum2._last_alert_time.clear()
            # Flat performance (plateaued)
            for step in range(50):
                rewards = [0.5] * 5  # Flat
                rollout_stats = [create_test_rollout_stats(r, "lesson1") for r in rewards]
                curriculum2.update_lesson_stats(rollout_stats, "training", step)

            # Stagnation alert should trigger
            stagnation_calls = [call for call in mock_logger.warning.call_args_list if "no progress" in str(call)]
            assert len(stagnation_calls) > 0

    def test_alerts_only_evaluate_when_conditions_met(self):
        """Test that alerts don't trigger unnecessarily."""
        curriculum = create_test_curriculum_with_alerts(
            [
                GraduationRegressionAlert(),
                TrainingStalledAlert(stagnation_window=30),
            ]
        )

        with patch("marin.rl.curriculum.logger") as mock_logger:
            # Normal training - shouldn't trigger alerts
            for step in range(50):
                perf = 0.3 + step * 0.01  # Improving
                rewards = [perf] * 5
                rollout_stats = [create_test_rollout_stats(r, "lesson1") for r in rewards]
                curriculum.update_lesson_stats(rollout_stats, "training", step)

            # Regression alert shouldn't trigger (not graduated)
            # Stagnation alert shouldn't trigger (improving)
            warning_calls = mock_logger.warning.call_args_list
            critical_calls = mock_logger.critical.call_args_list
            assert len(warning_calls) == 0
            assert len(critical_calls) == 0

    def test_wandb_alert_integration(self):
        """Test that wandb alerts are sent when enabled."""
        curriculum = create_test_curriculum_with_alerts([GraduationRegressionAlert(regression_threshold=0.85)])
        curriculum.config.enable_wandb_alerts = True

        # Train lesson first, then graduate
        for step in range(100):
            rewards = [0.9] * 10
            rollout_stats = [create_test_rollout_stats(r, "lesson1") for r in rewards]
            curriculum.update_lesson_stats(rollout_stats, "training", step)

        # Graduate the lesson
        for step in range(100, 120):
            rewards = [0.95] * 10
            rollout_stats = [create_test_rollout_stats(r, "lesson1") for r in rewards]
            curriculum.update_lesson_stats(rollout_stats, "eval", step)

        assert "lesson1" in curriculum.graduated

        # Cause regression - need enough samples to affect recent window
        with (
            patch("wandb.run", new_callable=MagicMock) as mock_wandb_run,
            patch("wandb.alert") as mock_wandb_alert,
            patch("wandb.log") as mock_wandb_log,
        ):
            # Make wandb.run truthy - need to set it on the module
            import wandb

            mock_wandb_run.__bool__ = lambda self: True
            wandb.run = mock_wandb_run

            # Add enough regression samples to affect the recent window (20 samples)
            for step in range(200, 210):
                rewards = [0.7] * 10  # Performance drops below threshold
                rollout_stats = [create_test_rollout_stats(r, "lesson1") for r in rewards]
                curriculum.update_lesson_stats(rollout_stats, "eval", step)

            # Should send wandb alert
            assert mock_wandb_alert.called
            call_args = mock_wandb_alert.call_args
            assert "Curriculum Alert" in call_args[1]["title"]
            assert "lesson1" in call_args[1]["text"]
            assert call_args[1]["level"] == "WARN"  # Should be WARNING level
            assert call_args[1]["wait_duration"] == curriculum.config.wandb_alert_rate_limit

            # Should also log metrics
            assert mock_wandb_log.called
            log_call_args = mock_wandb_log.call_args
            logged_metrics = log_call_args[0][0]  # First positional arg is the metrics dict
            assert "alerts/lesson1/graduation_regression" in logged_metrics
            assert logged_metrics["alerts/lesson1/graduation_regression"] == 1
            assert "alerts/lesson1/health_status" in logged_metrics
            assert logged_metrics["alerts/lesson1/health_status"] == 1  # WARNING = 1
            # Check that metrics from alert result are logged
            assert any(k.startswith("alerts/lesson1/metrics/") for k in logged_metrics.keys())

    def test_alert_evaluation_uses_existing_stats(self):
        """Test that alerts use curriculum's existing statistics, not duplicated logic."""
        curriculum = create_test_curriculum_with_alerts([TrainingStalledAlert(stagnation_window=30, plateau_window=20)])

        # Train lesson
        for step in range(50):
            perf = 0.5
            rewards = [perf] * 5
            rollout_stats = [create_test_rollout_stats(r, "lesson1") for r in rewards]
            curriculum.update_lesson_stats(rollout_stats, "training", step)

        # Verify alert uses same stats that curriculum uses for decisions
        stats = curriculum.stats["lesson1"]
        assert len(stats.training_stats.reward_history) > 0

        # Alert evaluation should use these same stats
        with patch("marin.rl.curriculum.logger") as mock_logger:
            # Continue plateau - need enough steps to trigger alert after cooldown
            # Alert has cooldown_steps=200, so we need to ensure it triggers
            # Reset cooldown by ensuring we're past any previous alert
            curriculum._last_alert_time.clear()

            for step in range(50, 80):
                rewards = [0.5] * 5
                rollout_stats = [create_test_rollout_stats(r, "lesson1") for r in rewards]
                curriculum.update_lesson_stats(rollout_stats, "training", step)

            # Alert should trigger using the same stats
            # Check that warning was called (alert is logged)
            warning_calls = [
                call
                for call in mock_logger.warning.call_args_list
                if len(call[0]) > 0 and "no progress" in str(call[0][0])
            ]
            assert len(warning_calls) > 0
