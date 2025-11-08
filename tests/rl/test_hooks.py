"""Tests for the RolloutWorker hook system."""

import unittest
from unittest.mock import MagicMock, patch, call

import jax.numpy as jnp
import jax.random as jrandom

from marin.rl.hooks import (
    Hook,
    HookContext,
    HookManager,
    PeriodicHook,
    EvaluationHook,
    create_default_evaluation_hooks,
)


class TestHookContext(unittest.TestCase):
    """Test HookContext functionality."""
    
    def test_hook_context_creation(self):
        """Test creating a hook context."""
        mock_worker = MagicMock()
        rng = jrandom.PRNGKey(42)
        
        context = HookContext(
            worker=mock_worker,
            step=100,
            rng=rng,
            lesson_id="test_lesson",
            metadata={"key": "value"}
        )
        
        self.assertEqual(context.worker, mock_worker)
        self.assertEqual(context.step, 100)
        self.assertEqual(context.lesson_id, "test_lesson")
        self.assertEqual(context.metadata["key"], "value")
    
    def test_split_rng(self):
        """Test RNG splitting in context."""
        mock_worker = MagicMock()
        rng = jrandom.PRNGKey(42)
        
        context = HookContext(worker=mock_worker, step=100, rng=rng)
        rng1, rng2 = context.split_rng()
        
        # Check that we get two different RNG keys
        self.assertFalse(jnp.array_equal(rng1, rng2))
        self.assertFalse(jnp.array_equal(rng, rng1))
        self.assertFalse(jnp.array_equal(rng, rng2))


class TestPeriodicHook(unittest.TestCase):
    """Test PeriodicHook functionality."""
    
    def test_periodic_hook_frequency(self):
        """Test that periodic hooks run at the correct frequency."""
        
        class TestHook(PeriodicHook):
            def run(self, context):
                return {"ran": True}
        
        hook = TestHook(frequency=10, start_step=0)
        mock_worker = MagicMock()
        rng = jrandom.PRNGKey(42)
        
        # Should not run at step 5
        context = HookContext(worker=mock_worker, step=5, rng=rng)
        self.assertFalse(hook.should_run(context))
        
        # Should run at step 10
        context = HookContext(worker=mock_worker, step=10, rng=rng)
        self.assertTrue(hook.should_run(context))
        
        # Should not run at step 15
        context = HookContext(worker=mock_worker, step=15, rng=rng)
        self.assertFalse(hook.should_run(context))
        
        # Should run at step 20
        context = HookContext(worker=mock_worker, step=20, rng=rng)
        self.assertTrue(hook.should_run(context))
    
    def test_periodic_hook_start_step(self):
        """Test that periodic hooks respect the start_step parameter."""
        
        class TestHook(PeriodicHook):
            def run(self, context):
                return {"ran": True}
        
        hook = TestHook(frequency=10, start_step=50)
        mock_worker = MagicMock()
        rng = jrandom.PRNGKey(42)
        
        # Should not run before start_step
        context = HookContext(worker=mock_worker, step=10, rng=rng)
        self.assertFalse(hook.should_run(context))
        
        context = HookContext(worker=mock_worker, step=50, rng=rng)
        self.assertFalse(hook.should_run(context))
        
        # Should run at step 60 (first multiple of 10 after start_step)
        context = HookContext(worker=mock_worker, step=60, rng=rng)
        self.assertTrue(hook.should_run(context))


class TestEvaluationHook(unittest.TestCase):
    """Test EvaluationHook functionality."""
    
    def test_micro_eval_hook(self):
        """Test micro-evaluation hook configuration."""
        hook = EvaluationHook(
            frequency=10,
            n_examples=4,
            eval_type="micro_eval",
            evaluate_all_lessons=False
        )
        
        mock_worker = MagicMock()
        mock_worker._evaluate_lesson = MagicMock(return_value={"metric": 1.0})
        rng = jrandom.PRNGKey(42)
        
        context = HookContext(
            worker=mock_worker,
            step=10,
            rng=rng,
            lesson_id="test_lesson"
        )
        
        # Should run at step 10
        self.assertTrue(hook.should_run(context))
        result = hook.run(context)
        
        # Check that _evaluate_lesson was called correctly
        mock_worker._evaluate_lesson.assert_called_once()
        args, kwargs = mock_worker._evaluate_lesson.call_args
        self.assertEqual(args[0], "test_lesson")
        self.assertEqual(args[1], 4)
        self.assertEqual(kwargs["eval_type"], "micro_eval")
    
    def test_full_eval_hook(self):
        """Test full evaluation hook configuration."""
        hook = EvaluationHook(
            frequency=100,
            n_examples=64,
            eval_type="eval",
            evaluate_all_lessons=True
        )
        
        mock_worker = MagicMock()
        mock_worker._evaluate_curriculum = MagicMock(return_value={"metric": 2.0})
        rng = jrandom.PRNGKey(42)
        
        context = HookContext(
            worker=mock_worker,
            step=100,
            rng=rng,
            lesson_id="test_lesson"
        )
        
        # Should run at step 100
        self.assertTrue(hook.should_run(context))
        result = hook.run(context)
        
        # Check that _evaluate_curriculum was called
        mock_worker._evaluate_curriculum.assert_called_once()


class TestHookManager(unittest.TestCase):
    """Test HookManager functionality."""
    
    def test_register_and_unregister_hooks(self):
        """Test registering and unregistering hooks."""
        manager = HookManager()
        
        hook1 = PeriodicHook(frequency=10)
        hook2 = PeriodicHook(frequency=100)
        
        # Register hooks
        manager.register_hook(hook1)
        manager.register_hook(hook2)
        self.assertEqual(len(manager), 2)
        
        # Unregister a hook
        success = manager.unregister_hook(hook1)
        self.assertTrue(success)
        self.assertEqual(len(manager), 1)
        
        # Try to unregister a hook that's not registered
        hook3 = PeriodicHook(frequency=50)
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
        hook1.should_run.return_value = True
        hook1.run.return_value = {"metric1": 1.0}
        
        hook2 = MagicMock(spec=Hook)
        hook2.should_run.return_value = False
        hook2.run.return_value = {"metric2": 2.0}
        
        hook3 = MagicMock(spec=Hook)
        hook3.should_run.return_value = True
        hook3.run.return_value = {"metric3": 3.0}
        
        manager.register_hook(hook1)
        manager.register_hook(hook2)
        manager.register_hook(hook3)
        
        mock_worker = MagicMock()
        rng = jrandom.PRNGKey(42)
        context = HookContext(worker=mock_worker, step=10, rng=rng)
        
        results = manager.run_hooks(context)
        
        # Check that only hooks that should run were executed
        hook1.should_run.assert_called_once_with(context)
        hook1.run.assert_called_once_with(context)
        
        hook2.should_run.assert_called_once_with(context)
        hook2.run.assert_not_called()  # Should not run
        
        hook3.should_run.assert_called_once_with(context)
        hook3.run.assert_called_once_with(context)
        
        # Check aggregated results
        self.assertEqual(results, {"metric1": 1.0, "metric3": 3.0})
    
    def test_hook_error_handling(self):
        """Test that hook errors are handled gracefully."""
        manager = HookManager()
        
        # Create a hook that raises an exception
        bad_hook = MagicMock(spec=Hook)
        bad_hook.should_run.return_value = True
        bad_hook.run.side_effect = RuntimeError("Test error")
        bad_hook.__repr__ = lambda self: "BadHook()"
        
        # Create a good hook
        good_hook = MagicMock(spec=Hook)
        good_hook.should_run.return_value = True
        good_hook.run.return_value = {"good": "result"}
        good_hook.__repr__ = lambda self: "GoodHook()"
        
        manager.register_hook(bad_hook)
        manager.register_hook(good_hook)
        
        mock_worker = MagicMock()
        rng = jrandom.PRNGKey(42)
        context = HookContext(worker=mock_worker, step=10, rng=rng)
        
        # Should continue running despite the error
        with patch("marin.rl.hooks.logger") as mock_logger:
            results = manager.run_hooks(context)
            
            # Check that error was logged
            mock_logger.error.assert_called()
            error_call_args = mock_logger.error.call_args[0]
            self.assertIn("Error running hook", error_call_args[0])
            
            # Good hook should still have run
            self.assertEqual(results, {"good": "result"})


class TestMetricsHook(unittest.TestCase):
    """Test MetricsHook functionality."""
    
    def test_metrics_hook(self):
        """Test custom metrics collection."""
        
        def custom_metric_fn(context):
            return {
                "step_squared": context.step ** 2,
                "has_lesson": context.lesson_id is not None
            }
        
        hook = PeriodicHook(
            frequency=10,
        )
        hook.run = custom_metric_fn
        
        mock_worker = MagicMock()
        mock_worker.tracker = MagicMock()
        rng = jrandom.PRNGKey(42)
        
        context = HookContext(
            worker=mock_worker,
            step=10,
            rng=rng,
            lesson_id="test_lesson"
        )
        
        self.assertTrue(hook.should_run(context))
        results = hook.run(context)
        
        # Check returned metrics
        self.assertEqual(results["custom.step_squared"], 100)
        self.assertEqual(results["custom.has_lesson"], True)
        
        # Check that metrics were logged
        mock_worker.tracker.log.assert_called_once_with(
            {"custom.step_squared": 100, "custom.has_lesson": True},
            step=10
        )


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
        self.assertIsInstance(micro_hook, EvaluationHook)
        self.assertEqual(micro_hook.frequency, 10)
        self.assertEqual(micro_hook.n_examples, 4)
        self.assertEqual(micro_hook.eval_type, "micro_eval")
        self.assertFalse(micro_hook.evaluate_all_lessons)
        
        # Check full eval hook
        full_hook = hooks[1]
        self.assertIsInstance(full_hook, EvaluationHook)
        self.assertEqual(full_hook.frequency, 100)
        self.assertEqual(full_hook.n_examples, 64)
        self.assertEqual(full_hook.eval_type, "eval")
        self.assertTrue(full_hook.evaluate_all_lessons)
    
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
        self.assertIsInstance(hook, EvaluationHook)
        self.assertEqual(hook.frequency, 10)
        self.assertEqual(hook.eval_type, "micro_eval")


if __name__ == "__main__":
    unittest.main()
