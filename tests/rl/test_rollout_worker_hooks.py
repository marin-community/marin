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

"""Tests demonstrating RolloutWorker hook system usage.

This file shows how to use the hook system to customize RolloutWorker behavior
without monkey-patching, addressing the issues mentioned in the GitHub issue.
"""

import unittest
from unittest.mock import MagicMock
from dataclasses import dataclass
from typing import Any

import jax.random as jrandom

from marin.rl.hooks import Hook, HookContext, PeriodicHook


class TestRolloutWorkerHookIntegration(unittest.TestCase):
    """Test integration of hooks with RolloutWorker."""

    def setUp(self):
        """Set up test fixtures."""
        # We'll mock the RolloutWorker since we're testing the hook integration pattern
        self.mock_worker = MagicMock()
        self.mock_worker._hook_manager = MagicMock()
        self.mock_worker._hook_manager.hooks = []

        # Mock the register/unregister methods
        def register_hook(hook):
            self.mock_worker._hook_manager.hooks.append(hook)

        def unregister_hook(hook):
            if hook in self.mock_worker._hook_manager.hooks:
                self.mock_worker._hook_manager.hooks.remove(hook)
                return True
            return False

        def clear_hooks():
            self.mock_worker._hook_manager.hooks.clear()

        def get_hooks():
            return list(self.mock_worker._hook_manager.hooks)

        self.mock_worker.register_hook = register_hook
        self.mock_worker.unregister_hook = unregister_hook
        self.mock_worker.clear_hooks = clear_hooks
        self.mock_worker.get_hooks = get_hooks

    def test_custom_evaluation_hook(self):
        """Test adding a custom evaluation hook instead of monkey-patching."""

        # Define a custom evaluation hook
        class CustomEvaluationHook(PeriodicHook):
            """Custom evaluation that runs every 5 steps."""

            def __init__(self):
                super().__init__(frequency=5, start_step=0)
                self.evaluation_count = 0

            def run(self, context: HookContext) -> dict[str, Any] | None:
                self.evaluation_count += 1
                # Custom evaluation logic here
                return {"custom_eval_count": self.evaluation_count, "custom_metric": context.step * 2}

        # Register the custom hook
        custom_hook = CustomEvaluationHook()
        self.mock_worker.register_hook(custom_hook)

        # Verify hook was registered
        self.assertIn(custom_hook, self.mock_worker.get_hooks())

        # Simulate running at different steps
        rng = jrandom.PRNGKey(42)

        # Step 5: should run
        context = HookContext(worker=self.mock_worker, step=5, rng=rng)
        self.assertTrue(custom_hook.should_run(context))
        result = custom_hook.run(context)
        self.assertEqual(result["custom_eval_count"], 1)
        self.assertEqual(result["custom_metric"], 10)

        # Step 7: should not run
        context = HookContext(worker=self.mock_worker, step=7, rng=rng)
        self.assertFalse(custom_hook.should_run(context))

    def test_monitoring_hook(self):
        """Test adding a monitoring hook for health checks."""

        @dataclass
        class HealthMetrics:
            """Track health metrics across steps."""

            total_steps: int = 0
            failed_lessons: int = 0
            successful_lessons: int = 0
            last_checkpoint_step: int = 0

        class HealthMonitoringHook(Hook):
            """Monitor curriculum health and alert on issues."""

            def __init__(self, alert_threshold: int = 5):
                self.alert_threshold = alert_threshold
                self.metrics = HealthMetrics()
                self.consecutive_failures = 0

            def should_run(self, context: HookContext) -> bool:
                # Always run to track metrics
                return True

            def run(self, context: HookContext) -> dict[str, Any] | None:
                self.metrics.total_steps += 1

                # Check for failures in context metadata
                if context.metadata.get("lesson_failed", False):
                    self.metrics.failed_lessons += 1
                    self.consecutive_failures += 1
                else:
                    self.metrics.successful_lessons += 1
                    self.consecutive_failures = 0

                # Alert if too many consecutive failures
                alert = self.consecutive_failures >= self.alert_threshold

                return {
                    "health.total_steps": self.metrics.total_steps,
                    "health.failed_lessons": self.metrics.failed_lessons,
                    "health.successful_lessons": self.metrics.successful_lessons,
                    "health.failure_rate": (
                        self.metrics.failed_lessons / self.metrics.total_steps if self.metrics.total_steps > 0 else 0
                    ),
                    "health.alert": alert,
                }

        # Register the monitoring hook
        monitor_hook = HealthMonitoringHook(alert_threshold=3)
        self.mock_worker.register_hook(monitor_hook)

        # Simulate some steps with failures
        rng = jrandom.PRNGKey(42)

        # Successful step
        context = HookContext(worker=self.mock_worker, step=1, rng=rng, metadata={"lesson_failed": False})
        result = monitor_hook.run(context)
        self.assertEqual(result["health.successful_lessons"], 1)
        self.assertFalse(result["health.alert"])

        # Failed steps
        for step in range(2, 5):
            context = HookContext(worker=self.mock_worker, step=step, rng=rng, metadata={"lesson_failed": True})
            result = monitor_hook.run(context)

        # Should trigger alert after 3 consecutive failures
        self.assertTrue(result["health.alert"])
        self.assertEqual(result["health.failed_lessons"], 3)

    def test_data_collection_hook(self):
        """Test adding a hook for collecting additional data during rollouts."""

        class DataCollectionHook(PeriodicHook):
            """Collect additional statistics about rollouts."""

            def __init__(self, frequency: int = 50):
                super().__init__(frequency=frequency)
                self.collected_data = []

            def run(self, context: HookContext) -> dict[str, Any] | None:
                # Collect data about the current state
                data_point = {
                    "step": context.step,
                    "lesson_id": context.lesson_id,
                    "timestamp": context.metadata.get("timestamp"),
                    # Could collect more complex statistics here
                }
                self.collected_data.append(data_point)

                # Return summary statistics
                return {
                    "data_collection.total_points": len(self.collected_data),
                    "data_collection.unique_lessons": len(
                        set(d["lesson_id"] for d in self.collected_data if d["lesson_id"])
                    ),
                }

        # Register the data collection hook
        data_hook = DataCollectionHook(frequency=10)
        self.mock_worker.register_hook(data_hook)

        # Simulate some rollout steps
        rng = jrandom.PRNGKey(42)

        for step in [10, 20, 30]:
            context = HookContext(
                worker=self.mock_worker,
                step=step,
                rng=rng,
                lesson_id=f"lesson_{step % 3}",
                metadata={"timestamp": f"2024-01-{step:02d}"},
            )
            if data_hook.should_run(context):
                result = data_hook.run(context)

        # Check collected data
        self.assertEqual(len(data_hook.collected_data), 3)
        self.assertEqual(result["data_collection.total_points"], 3)
        self.assertEqual(result["data_collection.unique_lessons"], 3)

    def test_multiple_hooks_interaction(self):
        """Test that multiple hooks can work together without conflicts."""

        class Hook1(PeriodicHook):
            def __init__(self):
                super().__init__(frequency=10)

            def run(self, context):
                return {"hook1.value": context.step * 10}

        class Hook2(PeriodicHook):
            def __init__(self):
                super().__init__(frequency=20)

            def run(self, context):
                return {"hook2.value": context.step * 20}

        class Hook3(Hook):
            def should_run(self, context):
                # Run only on specific steps
                return context.step in [15, 25, 35]

            def run(self, context):
                return {"hook3.special": True}

        # Register all hooks
        hook1 = Hook1()
        hook2 = Hook2()
        hook3 = Hook3()

        self.mock_worker.register_hook(hook1)
        self.mock_worker.register_hook(hook2)
        self.mock_worker.register_hook(hook3)

        # Verify all hooks are registered
        self.assertEqual(len(self.mock_worker.get_hooks()), 3)

        # Test at different steps
        rng = jrandom.PRNGKey(42)

        # Step 10: only hook1 runs
        context = HookContext(worker=self.mock_worker, step=10, rng=rng)
        self.assertTrue(hook1.should_run(context))
        self.assertFalse(hook2.should_run(context))
        self.assertFalse(hook3.should_run(context))

        # Step 20: hook1 and hook2 run
        context = HookContext(worker=self.mock_worker, step=20, rng=rng)
        self.assertTrue(hook1.should_run(context))
        self.assertTrue(hook2.should_run(context))
        self.assertFalse(hook3.should_run(context))

        # Step 15: only hook3 runs
        context = HookContext(worker=self.mock_worker, step=15, rng=rng)
        self.assertFalse(hook1.should_run(context))
        self.assertFalse(hook2.should_run(context))
        self.assertTrue(hook3.should_run(context))

    def test_hook_removal(self):
        """Test dynamically removing hooks during execution."""

        class TemporaryHook(PeriodicHook):
            """Hook that should be removed after certain conditions."""

            def __init__(self):
                super().__init__(frequency=1)  # Run every step
                self.run_count = 0

            def run(self, context):
                self.run_count += 1
                return {"temp.count": self.run_count}

        temp_hook = TemporaryHook()
        self.mock_worker.register_hook(temp_hook)

        # Run for a few steps
        rng = jrandom.PRNGKey(42)
        for step in range(1, 4):
            context = HookContext(worker=self.mock_worker, step=step, rng=rng)
            temp_hook.run(context)

        self.assertEqual(temp_hook.run_count, 3)

        # Remove the hook
        success = self.mock_worker.unregister_hook(temp_hook)
        self.assertTrue(success)
        self.assertNotIn(temp_hook, self.mock_worker.get_hooks())

        # Try to remove again (should fail)
        success = self.mock_worker.unregister_hook(temp_hook)
        self.assertFalse(success)

    def test_clear_all_hooks(self):
        """Test clearing all hooks at once."""

        # Create concrete hook implementations
        class TestHook(PeriodicHook):
            def run(self, context):
                return None

        # Register multiple hooks
        hooks = [
            TestHook(frequency=10),
            TestHook(frequency=20),
            TestHook(frequency=30),
        ]

        for hook in hooks:
            self.mock_worker.register_hook(hook)

        self.assertEqual(len(self.mock_worker.get_hooks()), 3)

        # Clear all hooks
        self.mock_worker.clear_hooks()
        self.assertEqual(len(self.mock_worker.get_hooks()), 0)


class TestHookUsagePatterns(unittest.TestCase):
    """Test common usage patterns for the hook system."""

    def test_replacing_hardcoded_evaluation(self):
        """Demonstrate how to replace hardcoded evaluation with hooks.

        This addresses the main issue: replacing hardcoded evaluation logic
        with a flexible hook system.
        """

        # Before: Hardcoded evaluation (what we're replacing)
        # if step > 0 and step % config.micro_eval_frequency == 0:
        #     self._evaluate_lesson(...)
        # if step > 0 and step % config.eval_frequency == 0:
        #     self._evaluate_curriculum(...)

        # After: Using hooks
        from marin.rl.hooks import EvaluationHook

        # Create hooks that replicate the original behavior
        micro_eval_hook = EvaluationHook(
            frequency=10,  # config.micro_eval_frequency
            n_examples=4,  # config.micro_eval_n_examples
            eval_type="micro_eval",
            evaluate_all_lessons=False,
        )

        full_eval_hook = EvaluationHook(
            frequency=100,  # config.eval_frequency
            n_examples=64,  # config.eval_n_examples
            eval_type="eval",
            evaluate_all_lessons=True,
        )

        # These would be registered with the worker
        # worker.register_hook(micro_eval_hook)
        # worker.register_hook(full_eval_hook)

        # The hooks will automatically run at the appropriate steps
        # No need to hardcode the logic in the main loop

        # Verify the hooks are configured correctly
        self.assertEqual(micro_eval_hook.frequency, 10)
        self.assertEqual(full_eval_hook.frequency, 100)
        self.assertFalse(micro_eval_hook.evaluate_all_lessons)
        self.assertTrue(full_eval_hook.evaluate_all_lessons)

    def test_custom_test_hook_pattern(self):
        """Show how tests can use custom hooks instead of monkey-patching.

        This addresses the testing issue mentioned: tests need to monkey-patch
        to customize behavior. With hooks, they can just register custom hooks.
        """

        class TestEvaluationHook(PeriodicHook):
            """Custom evaluation hook for testing."""

            def __init__(self):
                # Run more frequently for testing
                super().__init__(frequency=2, start_step=0)
                self.evaluations = []

            def run(self, context):
                # Simplified evaluation for testing
                eval_result = {"step": context.step, "lesson": context.lesson_id, "test_metric": context.step * 0.1}
                self.evaluations.append(eval_result)
                return {"test_eval": eval_result}

        # In tests, instead of monkey-patching _evaluate_lesson:
        # worker._evaluate_lesson = mock_evaluate_lesson  # OLD WAY

        # Use a custom hook:
        test_hook = TestEvaluationHook()  # NEW WAY
        # worker.register_hook(test_hook)

        # Simulate test execution
        rng = jrandom.PRNGKey(42)
        mock_worker = MagicMock()

        for step in [2, 4, 6]:
            context = HookContext(worker=mock_worker, step=step, rng=rng, lesson_id=f"test_lesson_{step}")
            if test_hook.should_run(context):
                test_hook.run(context)

        # Verify test data was collected
        self.assertEqual(len(test_hook.evaluations), 3)
        self.assertEqual(test_hook.evaluations[0]["step"], 2)
        self.assertEqual(test_hook.evaluations[-1]["step"], 6)


if __name__ == "__main__":
    unittest.main()
