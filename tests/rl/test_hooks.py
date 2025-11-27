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

"""Tests for the RolloutWorker hook system."""

import unittest
from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import jax.random as jrandom

from marin.rl.hooks import (
    Hook,
    HookContext,
    HookManager,
    PeriodicHook,
    EvaluateLessonHook,
    EvaluateCurriculumHook,
    create_default_evaluation_hooks,
)


class TestHookContext(unittest.TestCase):
    """Test HookContext functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_worker = MagicMock()
        self.mock_curriculum_actor = MagicMock()
        self.rng = jrandom.PRNGKey(42)

    def _create_context(self, **kwargs):
        """Helper to create HookContext with required fields."""
        defaults = {
            "worker": self.mock_worker,
            "step": 100,
            "rng": self.rng,
            "curriculum_actor": self.mock_curriculum_actor,
        }
        defaults.update(kwargs)
        return HookContext(**defaults)

    def test_hook_context_creation(self):
        """Test creating a hook context."""
        context = self._create_context(lesson_id="test_lesson", metadata={"key": "value"})

        self.assertEqual(context.worker, self.mock_worker)
        self.assertEqual(context.step, 100)
        self.assertEqual(context.lesson_id, "test_lesson")
        self.assertEqual(context.metadata["key"], "value")

    def test_split_rng(self):
        """Test RNG splitting in context."""
        context = self._create_context()
        rng1, rng2 = context.split_rng()

        # Check that we get two different RNG keys
        self.assertFalse(jnp.array_equal(rng1, rng2))
        self.assertFalse(jnp.array_equal(self.rng, rng1))
        self.assertFalse(jnp.array_equal(self.rng, rng2))


class TestPeriodicHook(unittest.TestCase):
    """Test PeriodicHook functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_worker = MagicMock()
        self.mock_curriculum_actor = MagicMock()
        self.rng = jrandom.PRNGKey(42)

    def _create_context(self, **kwargs):
        """Helper to create HookContext with required fields."""
        defaults = {
            "worker": self.mock_worker,
            "step": 100,
            "rng": self.rng,
            "curriculum_actor": self.mock_curriculum_actor,
        }
        defaults.update(kwargs)
        return HookContext(**defaults)

    def test_periodic_hook_frequency(self):
        """Test that periodic hooks run at the correct frequency."""

        class TestHook(PeriodicHook):
            def run(self, context):
                return {"ran": True}

        hook = TestHook(frequency=10, start_step=0)

        # Should not run at step 5
        context = self._create_context(step=5)
        self.assertFalse(hook.should_run(context))

        # Should run at step 10
        context = self._create_context(step=10)
        self.assertTrue(hook.should_run(context))

        # Should not run at step 15
        context = self._create_context(step=15)
        self.assertFalse(hook.should_run(context))

        # Should run at step 20
        context = self._create_context(step=20)
        self.assertTrue(hook.should_run(context))

    def test_periodic_hook_start_step(self):
        """Test that periodic hooks respect the start_step parameter."""

        class TestHook(PeriodicHook):
            def run(self, context):
                return {"ran": True}

        hook = TestHook(frequency=10, start_step=50)

        # Should not run before start_step
        context = self._create_context(step=10)
        self.assertFalse(hook.should_run(context))

        context = self._create_context(step=50)
        self.assertFalse(hook.should_run(context))

        # Should run at step 60 (first multiple of 10 after start_step)
        context = self._create_context(step=60)
        self.assertTrue(hook.should_run(context))


class TestEvaluationHook(unittest.TestCase):
    """Test evaluation hook functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_worker = MagicMock()
        self.mock_curriculum_actor = MagicMock()
        self.rng = jrandom.PRNGKey(42)

    def _create_context(self, **kwargs):
        """Helper to create HookContext with required fields."""
        defaults = {
            "worker": self.mock_worker,
            "step": 100,
            "rng": self.rng,
            "curriculum_actor": self.mock_curriculum_actor,
        }
        defaults.update(kwargs)
        return HookContext(**defaults)

    def test_evaluate_lesson_hook(self):
        """Test lesson evaluation hook configuration."""
        from marin.rl.types import RolloutBatch, RolloutGroup, Rollout, RolloutMetadata
        import numpy as np

        hook = EvaluateLessonHook(frequency=10, start_step=0, n_examples=4)

        # Create a mock batch
        rollout = Rollout(
            env_name="test",
            env_example_id="test_example",
            prompt_tokens=np.array([1, 2, 3], dtype=np.int32),
            response_tokens=np.array([4, 5, 6], dtype=np.int32),
            response_logprobs=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            token_rewards=np.array([0.5, 0.6, 0.7], dtype=np.float32),
            episode_reward=1.0,
        )
        batch = RolloutBatch(
            groups=[RolloutGroup(rollouts=[rollout])],
            metadata=RolloutMetadata(worker_id="test", timestamp=0.0, weight_step=0),
        )

        self.mock_worker._sample_batch = MagicMock(return_value=(batch, None))
        self.mock_worker._tokenizer = MagicMock()
        self.mock_worker._tokenizer.decode = MagicMock(side_effect=lambda x, **kwargs: "decoded")
        self.mock_worker.tracker = MagicMock()
        self.mock_worker.config = MagicMock()
        self.mock_worker.config.curriculum_config = MagicMock()
        self.mock_worker.config.curriculum_config.lessons = {}

        context = self._create_context(step=10, lesson_id="test_lesson")

        # Should run at step 10
        self.assertTrue(hook.should_run(context))
        result = hook.run(context)

        # Check that _sample_batch was called correctly
        self.mock_worker._sample_batch.assert_called_once()
        args, _kwargs = self.mock_worker._sample_batch.call_args
        self.assertEqual(args[0], "test_lesson")
        self.assertEqual(args[1], 4)
        self.assertEqual(args[2], 1)
        self.assertEqual(args[3], "eval")
        self.assertIsNotNone(result)

    def test_evaluate_curriculum_hook(self):
        """Test curriculum evaluation hook configuration."""
        from marin.rl.types import RolloutBatch, RolloutGroup, Rollout, RolloutMetadata
        import numpy as np

        hook = EvaluateCurriculumHook(frequency=100, start_step=0, n_examples=64)

        # Create a mock batch
        rollout = Rollout(
            env_name="test",
            env_example_id="test_example",
            prompt_tokens=np.array([1, 2, 3], dtype=np.int32),
            response_tokens=np.array([4, 5, 6], dtype=np.int32),
            response_logprobs=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            token_rewards=np.array([0.5, 0.6, 0.7], dtype=np.float32),
            episode_reward=1.0,
        )
        batch = RolloutBatch(
            groups=[RolloutGroup(rollouts=[rollout])],
            metadata=RolloutMetadata(worker_id="test", timestamp=0.0, weight_step=0),
        )

        self.mock_worker._sample_batch = MagicMock(return_value=(batch, None))
        self.mock_worker._tokenizer = MagicMock()
        self.mock_worker._tokenizer.decode = MagicMock(side_effect=lambda x, **kwargs: "decoded")
        self.mock_worker.tracker = MagicMock()
        self.mock_worker.config = MagicMock()
        self.mock_worker.config.curriculum_config = MagicMock()
        self.mock_worker.config.curriculum_config.lessons = {"lesson1": MagicMock()}

        context = self._create_context(step=100, lesson_id="test_lesson")

        # Should run at step 100
        self.assertTrue(hook.should_run(context))
        result = hook.run(context)

        # Check that _sample_batch was called
        self.mock_worker._sample_batch.assert_called()
        self.assertIsNotNone(result)


class TestHookManager(unittest.TestCase):
    """Test HookManager functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_worker = MagicMock()
        self.mock_curriculum_actor = MagicMock()
        self.rng = jrandom.PRNGKey(42)

    def _create_context(self, **kwargs):
        """Helper to create HookContext with required fields."""
        defaults = {
            "worker": self.mock_worker,
            "step": 10,
            "rng": self.rng,
            "curriculum_actor": self.mock_curriculum_actor,
        }
        defaults.update(kwargs)
        return HookContext(**defaults)

    def test_register_and_unregister_hooks(self):
        """Test registering and unregistering hooks."""

        class TestHook(PeriodicHook):
            def run(self, context):
                return {"ran": True}

        manager = HookManager()

        hook1 = TestHook(frequency=10)
        hook2 = TestHook(frequency=100)

        # Register hooks
        manager.register_hook(hook1)
        manager.register_hook(hook2)
        self.assertEqual(len(manager), 2)

        # Unregister a hook
        success = manager.unregister_hook(hook1)
        self.assertTrue(success)
        self.assertEqual(len(manager), 1)

        # Try to unregister a hook that's not registered
        hook3 = TestHook(frequency=50)
        success = manager.unregister_hook(hook3)
        self.assertFalse(success)
        self.assertEqual(len(manager), 1)

        # Clear all hooks
        manager.clear_hooks()
        self.assertEqual(len(manager), 0)

    def test_run_hooks(self):
        """Test running hooks through the manager."""
        manager = HookManager()

        # Create mock hooks
        hook1 = MagicMock(spec=Hook)
        hook1.name = "hook1"
        hook1.should_run.return_value = True
        hook1.run.return_value = {"metric1": 1.0}

        hook2 = MagicMock(spec=Hook)
        hook2.name = "hook2"
        hook2.should_run.return_value = False
        hook2.run.return_value = {"metric2": 2.0}

        hook3 = MagicMock(spec=Hook)
        hook3.name = "hook3"
        hook3.should_run.return_value = True
        hook3.run.return_value = {"metric3": 3.0}

        manager.register_hook(hook1)
        manager.register_hook(hook2)
        manager.register_hook(hook3)

        context = self._create_context()

        results = manager.run_hooks(context)

        # Check that only hooks that should run were executed
        hook1.should_run.assert_called_once_with(context)
        hook1.run.assert_called_once_with(context)

        hook2.should_run.assert_called_once_with(context)
        hook2.run.assert_not_called()  # Should not run

        hook3.should_run.assert_called_once_with(context)
        hook3.run.assert_called_once_with(context)

        # Check aggregated results (with hook name prefixes)
        self.assertEqual(results, {"hook1/metric1": 1.0, "hook3/metric3": 3.0})

    def test_hook_error_handling(self):
        """Test that hook errors are handled gracefully."""
        manager = HookManager()

        # Create a hook that raises an exception
        bad_hook = MagicMock(spec=Hook)
        bad_hook.name = "bad_hook"
        bad_hook.should_run.return_value = True
        bad_hook.run.side_effect = RuntimeError("Test error")
        bad_hook.__repr__ = lambda self: "BadHook()"

        # Create a good hook
        good_hook = MagicMock(spec=Hook)
        good_hook.name = "good_hook"
        good_hook.should_run.return_value = True
        good_hook.run.return_value = {"good": "result"}
        good_hook.__repr__ = lambda self: "GoodHook()"

        manager.register_hook(bad_hook)
        manager.register_hook(good_hook)

        context = self._create_context()

        # Should continue running despite the error
        with patch("marin.rl.hooks.logger") as mock_logger:
            results = manager.run_hooks(context)

            # Check that error was logged
            mock_logger.error.assert_called()
            error_call_args = mock_logger.error.call_args[0]
            self.assertIn("Error running hook", error_call_args[0])

            # Good hook should still have run (with hook name prefix)
            self.assertEqual(results, {"good_hook/good": "result"})


class TestMetricsHook(unittest.TestCase):
    """Test MetricsHook functionality."""

    def test_metrics_hook(self):
        """Test custom metrics collection."""

        def custom_metric_fn(context):
            return {"step_squared": context.step**2, "has_lesson": context.lesson_id is not None}

        class CustomMetricsHook(PeriodicHook):
            def run(self, context):
                return custom_metric_fn(context)

        hook = CustomMetricsHook(frequency=10, name="custom_metrics")
        mock_worker = MagicMock()
        mock_curriculum_actor = MagicMock()
        mock_worker.tracker = MagicMock()
        rng = jrandom.PRNGKey(42)

        context = HookContext(
            worker=mock_worker,
            step=10,
            rng=rng,
            curriculum_actor=mock_curriculum_actor,
            lesson_id="test_lesson",
        )

        self.assertTrue(hook.should_run(context))
        results = hook.run(context)

        # Check returned metrics (hook returns raw metrics)
        self.assertEqual(results["step_squared"], 100)
        self.assertEqual(results["has_lesson"], True)

        # Test through manager to see prefixed results
        manager = HookManager()
        manager.register_hook(hook)
        manager_results = manager.run_hooks(context)
        self.assertEqual(manager_results["custom_metrics/step_squared"], 100)
        self.assertEqual(manager_results["custom_metrics/has_lesson"], True)


class TestDefaultEvaluationHooks(unittest.TestCase):
    """Test creation of default evaluation hooks."""

    def test_create_default_hooks_with_config(self):
        """Test creating default hooks from curriculum config."""
        mock_config = MagicMock()
        mock_config.micro_eval_frequency = 10
        mock_config.micro_eval_n_examples = 4
        mock_config.eval_frequency = 100
        mock_config.eval_n_examples = 64

        hooks = create_default_evaluation_hooks(mock_config)

        self.assertEqual(len(hooks), 2)

        # Check micro-eval hook
        micro_hook = hooks[0]
        self.assertIsInstance(micro_hook, EvaluateLessonHook)
        self.assertEqual(micro_hook.frequency, 10)
        self.assertEqual(micro_hook.n_examples, 4)

        # Check full eval hook
        full_hook = hooks[1]
        self.assertIsInstance(full_hook, EvaluateCurriculumHook)
        self.assertEqual(full_hook.frequency, 100)
        self.assertEqual(full_hook.n_examples, 64)

    def test_create_default_hooks_with_disabled_eval(self):
        """Test that no hooks are created when evaluation is disabled."""
        mock_config = MagicMock()
        mock_config.micro_eval_frequency = 0
        mock_config.eval_frequency = 0

        hooks = create_default_evaluation_hooks(mock_config)

        self.assertEqual(len(hooks), 0)

    def test_create_default_hooks_with_partial_config(self):
        """Test creating hooks when only some evaluations are enabled."""
        mock_config = MagicMock()
        mock_config.micro_eval_frequency = 10
        mock_config.micro_eval_n_examples = 4
        mock_config.eval_frequency = 0  # Disabled

        hooks = create_default_evaluation_hooks(mock_config)

        self.assertEqual(len(hooks), 1)

        # Only micro-eval hook should be created
        hook = hooks[0]
        self.assertIsInstance(hook, EvaluateLessonHook)
        self.assertEqual(hook.frequency, 10)


if __name__ == "__main__":
    unittest.main()
