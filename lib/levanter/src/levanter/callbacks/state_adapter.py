# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar

import jax
from jaxtyping import PyTree

from levanter.callbacks._core import Callback, LambdaCallback, StepInfo


S = TypeVar("S")


@dataclass(frozen=True)
class CallbackStateView:
    """Minimal state surface expected by generic callback hooks."""

    step: jax.Array
    model: PyTree
    eval_model: PyTree
    opt_state: PyTree


@dataclass(frozen=True)
class _Hook:
    fn: Callback
    every: int


class StateCallbackRunner(Generic[S]):
    """Run callback hooks against arbitrary state types via simple accessors."""

    def __init__(
        self,
        *,
        step_getter: Callable[[S], jax.Array],
        model_getter: Callable[[S], PyTree],
        eval_model_getter: Callable[[S], PyTree],
        opt_state_getter: Callable[[S], PyTree],
    ):
        self._step_getter = step_getter
        self._model_getter = model_getter
        self._eval_model_getter = eval_model_getter
        self._opt_state_getter = opt_state_getter
        self._hooks: list[_Hook] = []

    def add_hook(self, fn: Callable[[StepInfo], Any] | Callback, *, every: int = 1) -> None:
        if every <= 0:
            raise ValueError(f"Hook interval must be positive, got {every}")
        callback = fn if isinstance(fn, Callback) else LambdaCallback(fn)
        self._hooks.append(_Hook(callback, every))

    def run(self, state: S, *, loss: float | jax.Array, step_duration: float, force: bool = False) -> None:
        info = StepInfo(
            state=CallbackStateView(
                step=self._step_getter(state),
                model=self._model_getter(state),
                eval_model=self._eval_model_getter(state),
                opt_state=self._opt_state_getter(state),
            ),
            loss=loss,  # type: ignore[arg-type]
            step_duration=step_duration,
        )
        for hook in self._hooks:
            if force or info.step % hook.every == 0:
                hook.fn.on_step(info, force=force)
