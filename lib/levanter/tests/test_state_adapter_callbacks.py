# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import jax.numpy as jnp

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


def test_state_callback_runner_supports_trainer_style_state():
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
        step=jnp.array(3, dtype=jnp.int32),
        model="model",
        eval_model="eval_model",
        opt_state="opt_state",
    )
    runner.run(state, loss=1.0, step_duration=0.1)

    assert events == [(2, 3, "model", "eval_model", "opt_state")]


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
        step=jnp.array(2, dtype=jnp.int32),  # completed step is 1, so not divisible by 2
        params="params",
        ema_params="ema_params",
        opt_state="opt_state",
    )

    runner.run(state, loss=1.0, step_duration=0.1)
    assert calls == []

    runner.run(state, loss=1.0, step_duration=0.1, force=True)
    assert calls == [1]
