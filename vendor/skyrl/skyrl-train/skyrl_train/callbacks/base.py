"""
Base classes for the SkyRL callback system.

This module provides the core abstractions for trainer callbacks:
- TrainerState: Immutable state passed to callbacks
- TrainerControl: Mutable control object for influencing training flow
- TrainerCallback: Base class for implementing callbacks
- CallbackHandler: Manages callback execution with async support
- AtomicStepCounter: Thread-safe step counter for async workers
"""

import asyncio
import math
import threading
from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from loguru import logger

from .types import CallbackErrorBehavior


def checkpoint_save_allowed(state: "TrainerState", save_after_fraction: float) -> bool:
    """Allow checkpoint writes after the configured fraction of planned updates."""
    if not 0.0 <= save_after_fraction <= 1.0:
        raise ValueError("save_after_fraction must be between 0 and 1")
    return state.global_step >= math.ceil(save_after_fraction * state.total_steps)


class AtomicStepCounter:
    """
    Thread-safe atomic step counter for async callbacks.

    This provides a master step count that can be safely accessed and updated
    from multiple async workers. It uses a threading.Lock for thread safety.

    Example:
        ```python
        counter = AtomicStepCounter()
        counter.set(10)

        # From any async worker:
        current_step = counter.get()
        counter.increment()
        ```
    """

    def __init__(self, initial_value: int = 0):
        self._value = initial_value
        self._lock = threading.Lock()

    def get(self) -> int:
        """Get the current step count (thread-safe)."""
        with self._lock:
            return self._value

    def set(self, value: int) -> None:
        """Set the step count (thread-safe)."""
        with self._lock:
            self._value = value

    def increment(self, delta: int = 1) -> int:
        """Increment and return the new value (thread-safe)."""
        with self._lock:
            self._value += delta
            return self._value

    def compare_and_swap(self, expected: int, new_value: int) -> bool:
        """
        Atomically set the value if it equals the expected value.

        Args:
            expected: The expected current value
            new_value: The new value to set

        Returns:
            True if the swap was successful, False otherwise
        """
        with self._lock:
            if self._value == expected:
                self._value = new_value
                return True
            return False


@dataclass
class TrainerState:
    """
    Immutable state passed to callbacks at each event.

    This provides callbacks with read-only access to the current training state
    without exposing the full trainer object.

    Attributes:
        global_step: Current global training step (1-indexed during training)
        epoch: Current epoch number (0-indexed)
        total_steps: Total number of training steps
        num_steps_per_epoch: Number of steps per epoch
        is_last_step: Whether this is the last step of training
        is_epoch_end: Whether this is the last step of the current epoch
        metrics: Dictionary of metrics from the current step
        timings: Dictionary of timing measurements from the current step
    """

    global_step: int
    epoch: int
    total_steps: int
    num_steps_per_epoch: int
    is_last_step: bool = False
    is_epoch_end: bool = False
    metrics: Dict[str, Any] = field(default_factory=dict)
    timings: Dict[str, float] = field(default_factory=dict)

    @property
    def step_in_epoch(self) -> int:
        """Returns the current step within the epoch (1-indexed)."""
        return self.global_step - (self.epoch * self.num_steps_per_epoch)


@dataclass
class TrainerControl:
    """
    Mutable control object for influencing training flow from callbacks.

    Callbacks can modify this object to request specific actions from the trainer.
    The trainer checks these flags after each event and takes appropriate action.

    Attributes:
        step_limit: Absolute final step requested at training start
        should_training_stop: Set to True to request early stopping
        should_save: Set to True to request a checkpoint save
        should_evaluate: Set to True to request an evaluation run
        should_log: Set to True to request logging (default True)
        should_save_hf_model: Set to True to request an HF-format export
    """

    step_limit: int | None = None
    should_training_stop: bool = False
    should_save: bool = False
    should_evaluate: bool = False
    should_log: bool = True
    should_save_hf_model: bool = False

    def reset(self) -> None:
        """Reset all control flags to their defaults."""
        self.step_limit = None
        self.should_training_stop = False
        self.should_save = False
        self.should_evaluate = False
        self.should_log = True
        self.should_save_hf_model = False


class TrainerCallback(ABC):
    """
    Base class for trainer callbacks.

    Callbacks allow extending trainer behavior without modifying the core training loop.
    Override the methods you need - all methods have empty default implementations.

    Supports both sync and async variants. Override sync methods for simple callbacks,
    or async variants for I/O-bound operations. Framework prefers async if both defined.

    Attributes:
        error_behavior: How to handle errors - "raise" stops training, "warn" logs
            and continues, "ignore" silently continues. Default is "warn".

    Example:
        ```python
        from loguru import logger

        class MyCallback(TrainerCallback):
            def __init__(self, log_every: int = 10):
                self.log_every = log_every

            def on_step_end(self, state: TrainerState, control: TrainerControl, **kwargs):
                if state.global_step % self.log_every == 0:
                    logger.info(f"Step {state.global_step}: {state.metrics}")
        ```
    """

    error_behavior: CallbackErrorBehavior = CallbackErrorBehavior.WARN

    # Sync variants (default implementations)
    def on_train_begin(
        self,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        """
        Called at the beginning of training, before the first step.

        Args:
            state: Current trainer state
            control: Control object for influencing training
            **kwargs: Additional arguments (trainer reference available as 'trainer')

        Returns:
            Optionally return a modified control object
        """
        pass

    def on_train_end(
        self,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        """
        Called at the end of training, after the last step.

        Args:
            state: Current trainer state
            control: Control object for influencing training
            **kwargs: Additional arguments

        Returns:
            Optionally return a modified control object
        """
        pass

    def on_epoch_begin(
        self,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        """
        Called at the beginning of each epoch.

        Args:
            state: Current trainer state
            control: Control object for influencing training
            **kwargs: Additional arguments

        Returns:
            Optionally return a modified control object
        """
        pass

    def on_epoch_end(
        self,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        """
        Called at the end of each epoch.

        Args:
            state: Current trainer state
            control: Control object for influencing training
            **kwargs: Additional arguments

        Returns:
            Optionally return a modified control object
        """
        pass

    def on_step_begin(
        self,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        """
        Called at the beginning of each training step.

        Args:
            state: Current trainer state
            control: Control object for influencing training
            **kwargs: Additional arguments

        Returns:
            Optionally return a modified control object
        """
        pass

    def on_step_end(
        self,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        """
        Called at the end of each training step.

        This is the most commonly used hook for periodic actions like
        checkpointing, evaluation, or custom logging.

        Args:
            state: Current trainer state
            control: Control object for influencing training
            **kwargs: Additional arguments

        Returns:
            Optionally return a modified control object
        """
        pass

    def on_evaluate(
        self,
        state: TrainerState,
        control: TrainerControl,
        metrics: Dict[str, Any],
        **kwargs,
    ) -> Optional[TrainerControl]:
        """
        Called after evaluation completes.

        Args:
            state: Current trainer state
            control: Control object for influencing training
            metrics: Evaluation metrics
            **kwargs: Additional arguments

        Returns:
            Optionally return a modified control object
        """
        pass

    def on_save(
        self,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        """
        Called after a checkpoint is saved.

        Args:
            state: Current trainer state
            control: Control object for influencing training
            **kwargs: Additional arguments

        Returns:
            Optionally return a modified control object
        """
        pass

    def on_log(
        self,
        state: TrainerState,
        control: TrainerControl,
        logs: Dict[str, Any],
        **kwargs,
    ) -> Optional[TrainerControl]:
        """
        Called when logs are recorded.

        Args:
            state: Current trainer state
            control: Control object for influencing training
            logs: Dictionary of logged values
            **kwargs: Additional arguments

        Returns:
            Optionally return a modified control object
        """
        pass

    # Async variants (override for I/O-bound operations)
    async def on_train_begin_async(
        self,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        """Async version of on_train_begin. Defaults to calling sync version."""
        return self.on_train_begin(state, control, **kwargs)

    async def on_train_end_async(
        self,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        """Async version of on_train_end. Defaults to calling sync version."""
        return self.on_train_end(state, control, **kwargs)

    async def on_epoch_begin_async(
        self,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        """Async version of on_epoch_begin. Defaults to calling sync version."""
        return self.on_epoch_begin(state, control, **kwargs)

    async def on_epoch_end_async(
        self,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        """Async version of on_epoch_end. Defaults to calling sync version."""
        return self.on_epoch_end(state, control, **kwargs)

    async def on_step_begin_async(
        self,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        """Async version of on_step_begin. Defaults to calling sync version."""
        return self.on_step_begin(state, control, **kwargs)

    async def on_step_end_async(
        self,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        """Async version of on_step_end. Defaults to calling sync version."""
        return self.on_step_end(state, control, **kwargs)

    async def collect_evaluation(self, state, *, trainer, pool) -> Dict[str, float]:
        """Collect additional sources within the trainer's frozen evaluation session."""
        return {}

    async def on_evaluate_async(
        self,
        state: TrainerState,
        control: TrainerControl,
        metrics: Dict[str, Any],
        **kwargs,
    ) -> Optional[TrainerControl]:
        """Async version of on_evaluate. Defaults to calling sync version."""
        return self.on_evaluate(state, control, metrics, **kwargs)

    async def on_save_async(
        self,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        """Async version of on_save. Defaults to calling sync version."""
        return self.on_save(state, control, **kwargs)

    async def on_log_async(
        self,
        state: TrainerState,
        control: TrainerControl,
        logs: Dict[str, Any],
        **kwargs,
    ) -> Optional[TrainerControl]:
        """Async version of on_log. Defaults to calling sync version."""
        return self.on_log(state, control, logs, **kwargs)


class CallbackHandler:
    """
    Manages callback execution with async support and per-callback error handling.

    This handler dispatches events to all registered callbacks, handling both
    sync and async callbacks appropriately. It respects each callback's
    error_behavior setting.

    Example:
        ```python
        handler = CallbackHandler([
            CheckpointCallback(save_steps=10),
            EvaluationCallback(eval_steps=20),
            MyCustomCallback(),
        ])

        # In training loop:
        await handler.call_event_async("on_step_end", state, control)
        ```
    """

    def __init__(self, callbacks: Optional[List[TrainerCallback]] = None):
        """
        Initialize the callback handler.

        Args:
            callbacks: List of callback instances to manage
        """
        self.callbacks = callbacks or []

    def add_callback(self, callback: TrainerCallback) -> None:
        """Add a callback to the handler."""
        self.callbacks.append(callback)

    def remove_callback(self, callback_type: type) -> None:
        """Remove all callbacks of a given type."""
        self.callbacks = [cb for cb in self.callbacks if not isinstance(cb, callback_type)]

    def pop_callback(self, callback_type: type) -> Optional[TrainerCallback]:
        """Remove and return the first callback of a given type."""
        for i, cb in enumerate(self.callbacks):
            if isinstance(cb, callback_type):
                return self.callbacks.pop(i)
        return None

    def call_event(
        self,
        event: str,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        """
        Call an event on all callbacks synchronously.

        Args:
            event: Event name (e.g., "on_step_end")
            state: Current trainer state
            control: Control object for influencing training
            **kwargs: Additional arguments to pass to callbacks

        Returns:
            The (potentially modified) control object
        """
        for callback in self.callbacks:
            try:
                step_limit = control.step_limit
                method = getattr(callback, event, None)
                if method is not None:
                    result = method(state, control, **kwargs)
                    if result is not None:
                        control = result
                    if step_limit is not None:
                        control.step_limit = (
                            min(step_limit, control.step_limit) if control.step_limit is not None else step_limit
                        )
            except Exception as e:
                self._handle_error(callback, event, e)

        return control

    async def call_event_async(
        self,
        event: str,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        """
        Call an event on all callbacks, preferring async variants.

        For each callback, this method:
        1. Checks if an async variant is overridden (not just delegating to sync)
        2. If overridden, calls the async variant directly
        3. Otherwise, runs the sync method in a thread pool

        Args:
            event: Event name (e.g., "on_step_end")
            state: Current trainer state
            control: Control object for influencing training
            **kwargs: Additional arguments to pass to callbacks

        Returns:
            The (potentially modified) control object
        """
        async_event = f"{event}_async"

        for callback in self.callbacks:
            try:
                step_limit = control.step_limit
                async_method = getattr(callback, async_event, None)
                sync_method = getattr(callback, event, None)

                if async_method is not None and self._is_overridden(callback, async_event):
                    # Use the overridden async variant
                    result = await async_method(state, control, **kwargs)
                elif sync_method is not None:
                    # Run sync method in thread pool to not block event loop
                    result = await asyncio.to_thread(sync_method, state, control, **kwargs)
                else:
                    result = None

                if result is not None:
                    control = result
                if step_limit is not None:
                    control.step_limit = (
                        min(step_limit, control.step_limit) if control.step_limit is not None else step_limit
                    )

            except Exception as e:
                self._handle_error(callback, event, e)

        return control

    def _handle_error(self, callback: TrainerCallback, event: str, error: Exception) -> None:
        """Handle an error from a callback based on its error_behavior setting."""
        callback_name = callback.__class__.__name__

        if callback.error_behavior == CallbackErrorBehavior.RAISE:
            logger.error(f"Callback {callback_name}.{event} raised an error")
            raise error
        elif callback.error_behavior == CallbackErrorBehavior.WARN:
            logger.warning(f"Callback {callback_name}.{event} failed: {error}")
        # "ignore" does nothing

    def _is_overridden(self, callback: TrainerCallback, method_name: str) -> bool:
        """
        Check if a method is overridden from the base TrainerCallback class.

        This is used to determine whether to use the async variant or fall back
        to the sync version.
        """
        callback_method = getattr(type(callback), method_name, None)
        base_method = getattr(TrainerCallback, method_name, None)

        # If the method exists and is different from the base class method,
        # it has been overridden
        return callback_method is not None and callback_method is not base_method
