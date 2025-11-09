"""Hook system for RolloutWorker to enable flexible callbacks and monitoring.

This module provides a flexible hook system that allows arbitrary callbacks to be
registered with the RolloutWorker, replacing the previous hardcoded evaluation logic.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import jax.random as jrandom
from jax import numpy as jnp

if TYPE_CHECKING:
    from marin.rl.rollout_worker import RolloutWorker

logger = logging.getLogger("ray")


@dataclass
class HookContext:
    """Context passed to hooks containing relevant state and utilities."""
    
    worker: "RolloutWorker"
    step: int
    rng: jnp.ndarray
    lesson_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def split_rng(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Split the RNG key for use in the hook."""
        keys = jrandom.split(self.rng)
        return keys[0], keys[1]


class Hook(ABC):
    """Base class for all hooks that can be registered with RolloutWorker."""
    
    @abstractmethod
    def should_run(self, context: HookContext) -> bool:
        """Determine if this hook should run at the current step.
        
        Args:
            context: The current hook context
            
        Returns:
            True if the hook should run, False otherwise
        """
        pass
    
    @abstractmethod
    def run(self, context: HookContext) -> Optional[dict[str, Any]]:
        """Execute the hook logic.
        
        Args:
            context: The current hook context
            
        Returns:
            Optional dictionary of metrics or results
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class PeriodicHook(Hook):
    """Base class for hooks that run periodically based on step count."""
    
    def __init__(self, frequency: int, start_step: int = 0):
        """Initialize a periodic hook.
        
        Args:
            frequency: Run hook every N steps
            start_step: First step to start running the hook (default: 0)
        """
        self.frequency = frequency
        self.start_step = start_step
    
    def should_run(self, context: HookContext) -> bool:
        """Check if hook should run based on frequency."""
        return (
            context.step > self.start_step 
            and context.step % self.frequency == 0
        )
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(frequency={self.frequency}, start_step={self.start_step})"


class EvaluationHook(PeriodicHook):
    """Hook that performs curriculum evaluation - preserves existing behavior."""
    
    def __init__(
        self,
        frequency: int,
        n_examples: int,
        eval_type: str = "eval",
        start_step: int = 0,
        evaluate_all_lessons: bool = False
    ):
        """Initialize evaluation hook.
        
        Args:
            frequency: How often to run evaluation
            n_examples: Number of examples to evaluate
            eval_type: Type of evaluation ("eval", "micro_eval")
            start_step: First step to start evaluation
            evaluate_all_lessons: If True, evaluate all lessons; if False, only current
        """
        super().__init__(frequency, start_step)
        self.n_examples = n_examples
        self.eval_type = eval_type
        self.evaluate_all_lessons = evaluate_all_lessons
    
    def run(self, context: HookContext) -> Optional[dict[str, Any]]:
        """Run evaluation on lessons."""
        worker = context.worker
        rng, eval_rng = context.split_rng()
        
        if self.evaluate_all_lessons:
            # Full curriculum evaluation
            logger.info(f"Running full curriculum evaluation at step {context.step}")
            return worker._evaluate_curriculum(eval_rng, context.step)
        else:
            # Single lesson evaluation (micro-eval)
            if context.lesson_id is None:
                logger.warning("No lesson_id in context for micro-evaluation")
                return None
                
            logger.info(f"Running {self.eval_type} for lesson {context.lesson_id} at step {context.step}")
            return worker._evaluate_lesson(
                context.lesson_id,
                self.n_examples,
                eval_type=self.eval_type,
                rng=eval_rng,
                step=context.step
            )
    
    def __repr__(self) -> str:
        return (
            f"EvaluationHook(frequency={self.frequency}, n_examples={self.n_examples}, "
            f"eval_type='{self.eval_type}', evaluate_all_lessons={self.evaluate_all_lessons})"
        )


class HookManager:
    """Manages the execution of hooks in the RolloutWorker."""
    
    def __init__(self):
        """Initialize the hook manager."""
        self.hooks: list[Hook] = []
    
    def register_hook(self, hook: Hook) -> None:
        """Register a new hook.
        
        Args:
            hook: The hook to register
        """
        self.hooks.append(hook)
        logger.info(f"Registered hook: {hook}")
    
    def unregister_hook(self, hook: Hook) -> bool:
        """Unregister a hook.
        
        Args:
            hook: The hook to unregister
            
        Returns:
            True if hook was found and removed, False otherwise
        """
        if hook in self.hooks:
            self.hooks.remove(hook)
            logger.info(f"Unregistered hook: {hook}")
            return True
        return False
    
    def clear_hooks(self) -> None:
        """Remove all registered hooks."""
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
                        results.update(hook_results)
            except Exception as e:
                logger.error(f"Error running hook {hook}: {e}", exc_info=True)
        return results
    
    def __len__(self) -> int:
        """Return the number of registered hooks."""
        return len(self.hooks)
    
    def __repr__(self) -> str:
        return f"HookManager(hooks={self.hooks})"


def create_default_evaluation_hooks(curriculum_config) -> list[EvaluationHook]:
    """Create the default evaluation hooks based on curriculum config.
    
    This preserves the existing evaluation behavior of RolloutWorker.
    
    Args:
        curriculum_config: The curriculum configuration
        
    Returns:
        List of default evaluation hooks
    """
    hooks = []
    
    # Micro-evaluation hook (evaluates current lesson)
    if hasattr(curriculum_config, 'micro_eval_frequency') and curriculum_config.micro_eval_frequency > 0:
        micro_eval_hook = EvaluationHook(
            frequency=curriculum_config.micro_eval_frequency,
            n_examples=curriculum_config.micro_eval_n_examples,
            eval_type="micro_eval",
            start_step=0,
            evaluate_all_lessons=False
        )
        hooks.append(micro_eval_hook)
    
    # Full evaluation hook (evaluates all lessons)
    if hasattr(curriculum_config, 'eval_frequency') and curriculum_config.eval_frequency > 0:
        full_eval_hook = EvaluationHook(
            frequency=curriculum_config.eval_frequency,
            n_examples=curriculum_config.eval_n_examples,
            eval_type="eval",
            start_step=0,
            evaluate_all_lessons=True
        )
        hooks.append(full_eval_hook)
    
    return hooks
