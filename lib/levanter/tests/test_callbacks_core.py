# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import jax
import jax.numpy as jnp
import optax
import pytest

import haliax as hax

from levanter.callbacks import JitCallback, LambdaCallback
from levanter.checkpoint import CheckpointerConfig
from levanter.tracker import NoopConfig
from levanter.trainer import Trainer, TrainerConfig


def test_lambda_callback_passes_force_when_supported():
    calls: list[bool] = []

    def fn(_info: Any, *, force: bool = False):
        calls.append(force)

    cb = LambdaCallback(fn)
    cb.on_step(None, force=True)

    assert calls == [True]


def test_lambda_callback_does_not_pass_force_when_not_supported():
    calls: list[Any] = []

    def fn(info: Any):
        calls.append(info)

    cb = LambdaCallback(fn)
    marker = object()
    cb.on_step(marker, force=True)

    assert calls == [marker]


def test_lambda_callback_passes_force_via_kwargs():
    calls: list[bool] = []

    def fn(_info: Any, **kwargs: Any):
        calls.append(bool(kwargs["force"]))

    cb = LambdaCallback(fn)
    cb.on_step(None, force=True)

    assert calls == [True]


@pytest.mark.parametrize("every", [1, 2])
def test_trainer_hooks_follow_completed_updates(tmp_path, every):
    steps = []
    gradients = []

    class GradientCallback(JitCallback):
        def inside_step(self, state, inside_info):
            return inside_info.grads

        def on_step(self, info, gradient):
            gradients.append((info.next_step, float(gradient)))

    def loss_fn(model, batch, *, key):
        return jnp.mean((model - batch.array) ** 2)

    Batch = hax.Axis("batch", jax.device_count())
    config = TrainerConfig(
        id="hook-cadence",
        tracker=NoopConfig(),
        train_batch_size=Batch.size,
        per_device_parallelism=1,
        num_train_steps=4,
        load_checkpoint=False,
        require_accelerator=False,
        checkpointer=CheckpointerConfig(base_path=str(tmp_path)),
    )
    trainer = Trainer(config, optax.sgd(0.25), loss_fn, add_default_hooks=False)

    @trainer.add_hook(every=every)
    def record_step(info):
        steps.append(info.next_step)

    trainer.add_hook(GradientCallback(), every=every)
    with trainer:
        state = trainer.initial_state(jax.random.PRNGKey(0), model=jnp.array(0.0))
        trainer.train(state, [hax.ones(Batch)] * 4)

    if every == 1:
        assert steps == [1, 2, 3, 4, 4]
        assert gradients == [(1, -2.0), (2, -1.0), (3, -0.5), (4, -0.25)]
    else:
        assert steps == [2, 4, 4]
        assert gradients == [(2, -1.0), (4, -0.25)]
