# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import jax.numpy as jnp

from levanter.callbacks import Callback, ProgressEvent, StepInfo
from levanter.callbacks.state_adapter import StateCallbackRunner


@dataclass(frozen=True)
class _TrainerLikeState:
    step: jnp.ndarray
    model: str
    eval_model: str
    opt_state: str


@dataclass(frozen=True)
class _GrugLikeState:
    step: jnp.ndarray
    params: str
    ema_params: str
    opt_state: str


def test_state_callback_runner_fires_after_first_completed_step():
    events: list[tuple[int, int, str, str, str]] = []
    runner = StateCallbackRunner[_TrainerLikeState](
        step_getter=lambda s: s.step,
        model_getter=lambda s: s.model,
        eval_model_getter=lambda s: s.eval_model,
        opt_state_getter=lambda s: s.opt_state,
    )
    runner.add_hook(
        lambda info: events.append((info.step, info.next_step, info.model, info.eval_model, info.opt_state))
    )

    state = _TrainerLikeState(
        step=jnp.array(1, dtype=jnp.int32),
        model="model",
        eval_model="eval_model",
        opt_state="opt_state",
    )
    runner.run(state, loss=1.0, step_duration=0.1)

    assert events == [(0, 1, "model", "eval_model", "opt_state")]


def test_state_callback_runner_supports_grug_style_state_and_force_flag():
    calls: list[int] = []
    runner = StateCallbackRunner[_GrugLikeState](
        step_getter=lambda s: s.step,
        model_getter=lambda s: s.params,
        eval_model_getter=lambda s: s.ema_params,
        opt_state_getter=lambda s: s.opt_state,
    )
    runner.add_hook(lambda info: calls.append(info.step), every=2)

    state = _GrugLikeState(
        step=jnp.array(1, dtype=jnp.int32),
        params="params",
        ema_params="ema_params",
        opt_state="opt_state",
    )

    runner.run(state, loss=1.0, step_duration=0.1)
    assert calls == []

    runner.run(state, loss=1.0, step_duration=0.1, force=True)
    assert calls == [0]


def test_state_callback_runner_broadcasts_progress_events_from_callbacks():
    events: list[ProgressEvent] = []

    class EvaluationCallback(Callback):
        def on_event(self, event: ProgressEvent) -> None:
            events.append(event)

        def on_step(self, info: StepInfo, force: bool = False) -> None:
            del force
            info.emit_event(ProgressEvent.EVALUATION_STARTED)
            info.emit_event(ProgressEvent.EVALUATION_FINISHED)

    runner = StateCallbackRunner[_TrainerLikeState](
        step_getter=lambda s: s.step,
        model_getter=lambda s: s.model,
        eval_model_getter=lambda s: s.eval_model,
        opt_state_getter=lambda s: s.opt_state,
    )
    runner.add_hook(EvaluationCallback())
    state = _TrainerLikeState(
        step=jnp.array(1, dtype=jnp.int32),
        model="model",
        eval_model="eval_model",
        opt_state="opt_state",
    )

    runner.emit_event(ProgressEvent.TRAIN_STEP_STARTED)
    runner.run(state, loss=1.0, step_duration=0.1)

    assert events == [
        ProgressEvent.TRAIN_STEP_STARTED,
        ProgressEvent.EVALUATION_STARTED,
        ProgressEvent.EVALUATION_FINISHED,
    ]
