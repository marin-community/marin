# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import json
from datetime import timedelta
from itertools import repeat

import haliax as hax
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import optax
import pytest

import levanter.training_budget as training_budget
from levanter.checkpoint import CheckpointRetention, CheckpointerConfig, load_checkpoint
from levanter.tracker.json_file import JsonFileTrackerConfig
from levanter.trainer import Trainer, TrainerConfig
from levanter.trainer_state import TrainerState
from levanter.training_budget import (
    TrainingBudgetConfig,
    TrainingStopReason,
    WallClockCooldown,
    budget_learning_rate_scale,
    read_training_completion,
)
from levanter.utils.mesh import MeshConfig


@pytest.mark.parametrize(
    "maximum_updates,advance,expected_reason",
    [
        (746, 40, TrainingStopReason.TIME_BOUND),
        (2, 10, TrainingStopReason.MAXIMUM_UPDATES),
        (746, 10, TrainingStopReason.DATA_EXHAUSTED),
    ],
)
def test_deadline_finalizes_and_restores_the_last_complete_update(
    tmp_path, monkeypatch, maximum_updates, advance, expected_reason
):
    clock = [0.0]
    monkeypatch.setattr(training_budget, "monotonic", lambda: clock[0])
    monkeypatch.setattr(training_budget, "time", lambda: 1000 + clock[0])
    policy = TrainingBudgetConfig(timedelta(seconds=100), timedelta(seconds=20), timedelta(seconds=20), 1100)
    checkpoint_root, telemetry_root = tmp_path / "checkpoints", tmp_path / "telemetry"
    config = TrainerConfig(
        id="time-limited-training",
        train_batch_size=1,
        per_device_parallelism=1,
        num_train_steps=maximum_updates,
        mp=jmp.get_policy("params=float32,compute=float32,output=float32"),
        mesh=MeshConfig(axes={"data": 1, "model": 1, "replica": 1, "context": 1}),
        tracker=JsonFileTrackerConfig(output_path=str(tmp_path)),
        native_telemetry_root=str(telemetry_root),
        require_accelerator=False,
        log_jaxprs=False,
        log_xla_hlo=False,
        checkpointer=CheckpointerConfig(
            base_path=str(checkpoint_root),
            temporary_base_path=str(tmp_path / "resume"),
            append_run_id_to_base_path=False,
            save_interval=timedelta(hours=2),
            keep=None,
        ),
        training_budget=policy,
    )
    optimizer = optax.sgd(0.1)
    batch = hax.named(jnp.zeros(1), hax.Axis("batch", 1))

    def loss(model, batch, *, key=None):
        return jnp.mean((model - batch.array) ** 2)

    def advance_clock(info):
        clock[0] = int(info.state.step) * advance
        if int(info.state.step) == 1:
            trainer.request_checkpoint(CheckpointRetention.TEMPORARY)

    with Trainer(config, optimizer, loss) as trainer:
        trainer.add_hook(advance_clock, every=1)
        state = TrainerState.init(optimizer, jnp.asarray(1.0), key=jax.random.PRNGKey(0), mp=config.mp)
        batches = [batch, batch] if expected_reason == TrainingStopReason.DATA_EXHAUSTED else repeat(batch)
        trainer.train(state, batches)
        assert not (telemetry_root / "completion.json").exists()
    if expected_reason == TrainingStopReason.DATA_EXHAUSTED:
        receipt = json.loads((telemetry_root / "completion.json").read_text())
        with pytest.raises(ValueError, match="exhausted its data"):
            read_training_completion(
                telemetry_root=str(telemetry_root),
                checkpoint_root=str(checkpoint_root),
                maximum_updates=maximum_updates,
                budget=policy,
            )
    else:
        receipt = read_training_completion(
            telemetry_root=str(telemetry_root),
            checkpoint_root=str(checkpoint_root),
            maximum_updates=maximum_updates,
            budget=policy,
        )
    assert receipt["completed_updates"] == 2
    assert receipt["stop_reason"] == expected_reason
    restored = load_checkpoint({"model": jnp.asarray(0.0)}, receipt["checkpoint"])
    np.testing.assert_allclose(restored["model"], 0.64, rtol=1e-6)


def test_wall_clock_cooldown_scales_both_adam_groups_after_step_warmup(tmp_path, monkeypatch):
    admission_times = [0.0] * 30 + [40.0, 50.0, 51.0, 55.5, 59.95, 61.0]
    clock = [admission_times[0]]
    monkeypatch.setattr(training_budget, "monotonic", lambda: clock[0])
    monkeypatch.setattr(training_budget, "time", lambda: 1000 + clock[0])
    policy = TrainingBudgetConfig(
        timedelta(seconds=100),
        timedelta(seconds=20),
        timedelta(seconds=20),
        1100,
        cooldown=WallClockCooldown(fraction=0.15, min_lr_ratio=0.1),
    )
    assert budget_learning_rate_scale(policy, 0) == pytest.approx(0.1)
    checkpoint_root, telemetry_root = tmp_path / "checkpoints", tmp_path / "telemetry"
    config = TrainerConfig(
        id="wall-clock-cooldown",
        train_batch_size=1,
        per_device_parallelism=1,
        num_train_steps=4750,
        mp=jmp.get_policy("f32"),
        mesh=MeshConfig(axes={"data": 1, "model": 1, "replica": 1, "context": 1}),
        tracker=JsonFileTrackerConfig(output_path=str(tmp_path)),
        native_telemetry_root=str(telemetry_root),
        require_accelerator=False,
        log_jaxprs=False,
        log_xla_hlo=False,
        checkpointer=CheckpointerConfig(
            base_path=str(checkpoint_root),
            temporary_base_path=str(tmp_path / "resume"),
            append_run_id_to_base_path=False,
            save_interval=timedelta(hours=2),
            keep=None,
        ),
        training_budget=policy,
    )

    def curve(peak):
        return optax.join_schedules([optax.linear_schedule(0, peak, 30), optax.constant_schedule(peak)], [30])

    optimizer = optax.multi_transform(
        {"muonh": optax.adam(curve(0.1)), "adam": optax.adam(curve(0.03))},
        {"muonh": "muonh", "adam": "adam"},
    )
    batch = hax.named(jnp.zeros(1), hax.Axis("batch", 1))

    def loss(model, batch, *, key=None):
        return model["muonh"] + model["adam"]

    def advance_clock(info):
        clock[0] = admission_times[int(info.state.step)]

    with Trainer(config, optimizer, loss) as trainer:
        trainer.add_hook(advance_clock, every=1)
        state = TrainerState.init(
            optimizer,
            {"muonh": jnp.asarray(1.0), "adam": jnp.asarray(2.0)},
            key=jax.random.PRNGKey(0),
            mp=config.mp,
        )
        trainer.train(state, repeat(batch))
    receipt = read_training_completion(
        telemetry_root=str(telemetry_root),
        checkpoint_root=str(checkpoint_root),
        maximum_updates=4750,
        budget=policy,
    )
    assert receipt["completed_updates"] == 35
    assert receipt["stop_reason"] == TrainingStopReason.TIME_BOUND
    assert receipt["final_learning_rate_scale"] == pytest.approx(0.105, abs=1e-6)
    restored = load_checkpoint({"model": {"muonh": jnp.asarray(0.0), "adam": jnp.asarray(0.0)}}, receipt["checkpoint"])
    expected = {"muonh": jnp.asarray(1.0), "adam": jnp.asarray(2.0)}
    opt_state = optimizer.init(expected)
    for scale in [1.0] * 33 + [0.55, 0.105]:
        updates, opt_state = optimizer.update(jax.tree.map(jnp.ones_like, expected), opt_state, expected)
        expected = optax.apply_updates(expected, jax.tree.map(lambda update: update * scale, updates))
    for group in expected:
        np.testing.assert_allclose(restored["model"][group], expected[group], rtol=1e-6)
    changed = json.loads((telemetry_root / "completion.json").read_text())
    changed["final_learning_rate_scale"] = 1.0
    (telemetry_root / "completion.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError):
        read_training_completion(
            telemetry_root=str(telemetry_root),
            checkpoint_root=str(checkpoint_root),
            maximum_updates=4750,
            budget=policy,
        )
