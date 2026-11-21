# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import logging
import threading

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import jmp

from levanter.tracker.json_file import JsonFileTrackerConfig
from levanter.tracker.wandb import MAX_WANDB_ARTIFACT_BYTES, WandbTracker
from levanter.trainer import Trainer, TrainerConfig
from levanter.trainer_state import TrainerState
from levanter.utils.mesh import MeshConfig


def test_fp32_master_parameters_accumulate_updates_below_bfloat16_resolution():
    optimizer = optax.sgd(1e-4)
    policy = jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16")
    state = TrainerState.init(optimizer, jnp.asarray(1.0, jnp.bfloat16), key=jax.random.key(0), mp=policy)

    @jax.jit
    def update(state):
        return state.take_step(jnp.asarray(1.0, jnp.float32), key=state.training_key)[0]

    for _ in range(100):
        state = update(state)
    assert state.model.dtype == jnp.float32
    np.testing.assert_allclose(state.model, 0.99, atol=2e-6)
    assert policy.cast_to_compute(state.model).dtype == jnp.bfloat16


class WeightedBatch(eqx.Module):
    values: hax.NamedArray
    weights: hax.NamedArray


def test_debug_artifacts_do_not_block_updates_or_propagate_oversize_errors(tmp_path, monkeypatch, caplog):
    artifact = tmp_path / "oversized.bin"
    with artifact.open("wb") as stream:
        stream.truncate(MAX_WANDB_ARTIFACT_BYTES + 1)
    uploaded = []

    class Run:
        def log_artifact(self, *args, **kwargs):
            uploaded.append(args)

    wandb = WandbTracker(Run())
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    written = []

    def write_artifact(name, content, type):
        entered.set()
        assert release.wait(timeout=10)
        wandb.log_artifact(artifact, name=name, type=type)
        written.append(type)
        if type == "hlo":
            finished.set()

    config = TrainerConfig(
        id="nonblocking-debug-artifacts",
        train_batch_size=1,
        mesh=MeshConfig(axes={"data": 1, "model": 1, "replica": 1, "context": 1}),
        tracker=JsonFileTrackerConfig(output_path=str(tmp_path)),
        require_accelerator=False,
        log_jaxprs=True,
        log_xla_hlo=True,
    )
    optimizer = optax.sgd(0.1)

    def loss(model, batch, *, key=None):
        return hax.mean((model - batch) ** 2).scalar()

    with caplog.at_level(logging.WARNING), Trainer(config, optimizer, loss, add_default_hooks=False) as trainer:
        monkeypatch.setattr(trainer, "write_artifact", write_artifact)
        state = TrainerState.init(optimizer, jnp.asarray(2.0), key=jax.random.key(0), mp=config.mp)
        batch = hax.named(jnp.ones((1,)), hax.Axis("batch", 1))
        try:
            first = trainer.train_step(state, batch)
            assert entered.wait(timeout=10)
            second = trainer.train_step(first.state, batch)
            assert int(second.state.step) == 2
            np.testing.assert_allclose(second.state.model, 1.64, rtol=1e-6)
        finally:
            release.set()
        assert finished.wait(timeout=10)
    assert written == ["jaxpr", "hlo"]
    assert not uploaded
    assert "Skipping W&B artifact" in caplog.text


def test_token_weighted_microbatches_match_full_weighted_update(tmp_path):
    Batch = hax.Axis("batch", 4)
    values = jnp.asarray([1.0, 3.0, 5.0, 7.0])
    weights = jnp.asarray([1.0, 0.0, 2.0, 8.0])
    batch = WeightedBatch(hax.named(values, Batch), hax.named(weights, Batch))

    def loss(model, batch, *, key=None):
        del key
        return hax.sum((model - batch.values) ** 2 * batch.weights).scalar()

    def weight(batch, *, key=None):
        del key
        return hax.sum(batch.weights).scalar()

    config = TrainerConfig(
        id="weighted-microbatch-regression",
        train_batch_size=4,
        per_device_parallelism=1,
        mesh=MeshConfig(axes={"data": 1, "model": 1, "replica": 1, "context": 1}),
        tracker=JsonFileTrackerConfig(output_path=str(tmp_path)),
        require_accelerator=False,
        log_jaxprs=False,
        log_xla_hlo=False,
    )
    optimizer = optax.sgd(0.1)
    initial = jnp.asarray(2.0)
    reference_loss = jnp.sum((initial - values) ** 2 * weights) / weights.sum()
    reference_update = initial - 0.1 * (2 * (initial - values) * weights).sum() / weights.sum()
    with Trainer(config, optimizer, loss, add_default_hooks=False, loss_weight_fn=weight) as trainer:
        state = TrainerState.init(optimizer, initial, key=jax.random.key(0), mp=config.mp)
        actual = trainer.train_step(state, batch)
    np.testing.assert_allclose(actual.loss, reference_loss, rtol=1e-6)
    np.testing.assert_allclose(actual.state.model, reference_update, rtol=1e-6)
