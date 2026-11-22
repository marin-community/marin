# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml
from draccus import ParsingError
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec
from levanter.checkpoint import Checkpointer, CheckpointInterval, CheckpointRetention

from experiments.post_training.long_loop.sft import SMOKE_CONFIG, fresh_sft_config
from experiments.post_training.long_loop.sft_runtime.train import checkpoint_samples, restore_first_checkpoint


class State(NamedTuple):
    step: jax.Array
    params: jax.Array
    optimizer: jax.Array


def test_first_step_restores_full_state_and_releases_old_buffers(tmp_path):
    mesh = Mesh(np.asarray(jax.devices("cpu")[:1]), ("data",))
    with jax.set_mesh(mesh):
        state = State(jnp.array(1), jnp.arange(32, dtype=jnp.float32), jnp.full(32, 0.125))
        expected = [np.asarray(x).copy() for x in state]
        checkpointer = Checkpointer(tmp_path, save_interval=None, step_policies=[])
        restored = restore_first_checkpoint(state, checkpointer, mesh)
        for original, actual, wanted in zip(state, restored, expected, strict=True):
            assert original.is_deleted()
            np.testing.assert_array_equal(actual, wanted)
        assert (tmp_path / "step-1" / "metadata.json").is_file()


@pytest.mark.multidevice
def test_sharded_checkpoint_gate_samples_locally_and_restores_full_state(tmp_path, monkeypatch):
    devices = jax.devices("cpu")
    if len(devices) < 2:
        pytest.skip("Run with JAX_NUM_CPU_DEVICES=4 set before importing JAX")
    mesh = Mesh(np.asarray(devices), ("data",), axis_types=(AxisType.Explicit,))
    values = np.arange(len(devices) * 64, dtype=np.float32).reshape(len(devices) * 8, 8)
    with jax.set_mesh(mesh):
        sharding = NamedSharding(mesh, PartitionSpec("data", None))
        state = State(
            jax.device_put(np.array(1, dtype=np.int32), NamedSharding(mesh, PartitionSpec())),
            jax.device_put(values, sharding),
            jax.device_put(values * 0.125, sharding),
        )
        # This is the operation that failed after the first distributed SFT update.
        with pytest.raises(ValueError, match="incompatible devices"):
            state.params.addressable_shards[0].data.reshape(-1)
        transferred = []
        device_get = jax.device_get

        def record_transfer(array):
            transferred.append(array.size)
            return device_get(array)

        with monkeypatch.context() as patch:
            patch.setattr(jax, "device_get", record_transfer)
            expected_samples = checkpoint_samples(state)
        assert max(transferred) == 16
        assert len(expected_samples) == 3 * len(devices)
        checkpointer = Checkpointer(tmp_path, save_interval=None, step_policies=[])
        restored = restore_first_checkpoint(state, checkpointer, mesh)
        assert checkpoint_samples(restored) == expected_samples
        np.testing.assert_array_equal(restored.params, values)
        np.testing.assert_array_equal(restored.optimizer, values * 0.125)
        assert int(restored.step) == 1
        assert all(leaf.is_deleted() for leaf in state)
        assert (tmp_path / "step-1" / "metadata.json").is_file()


def test_first_step_restore_gate_uses_temporary_root_and_prunes_on_later_commit(tmp_path):
    mesh = Mesh(np.asarray(jax.devices("cpu")[:1]), ("data",))
    perm = tmp_path / "checkpoints"
    temp = tmp_path / "temporary-checkpoints"
    with jax.set_mesh(mesh):
        state1 = State(jnp.array(1), jnp.arange(32, dtype=jnp.float32), jnp.full(32, 0.125))
        expected = [np.asarray(x).copy() for x in state1]
        checkpointer = Checkpointer(
            perm,
            save_interval=None,
            step_policies=[CheckpointInterval(every=3, until=3)],
            temporary_base_path=temp,
            keep_last_temporary_checkpoints=1,
        )
        restored = restore_first_checkpoint(state1, checkpointer, mesh)
        for actual, wanted in zip(restored, expected, strict=True):
            np.testing.assert_array_equal(actual, wanted)
        assert not (perm / "step-1").exists()
        assert json.loads((temp / "step-1" / "metadata.json").read_text())["is_temporary"] is True

        state2 = State(jnp.array(2), jnp.arange(32, dtype=jnp.float32) + 1.0, jnp.full(32, 0.25))
        checkpointer.request_checkpoint(CheckpointRetention.TEMPORARY)
        checkpointer.on_step(tree=state2, step=2)
        checkpointer.wait_until_finished()
        assert not (temp / "step-1").exists()
        assert (temp / "step-2" / "metadata.json").is_file()

        state3 = State(jnp.array(3), jnp.arange(32, dtype=jnp.float32) + 2.0, jnp.full(32, 0.5))
        checkpointer.on_step(tree=state3, step=3)
        checkpointer.wait_until_finished()
        assert not (temp / "step-2").exists()
        assert json.loads((perm / "step-3" / "metadata.json").read_text())["is_temporary"] is False


def test_first_step_restore_gate_preserves_named_permanent_step_one(tmp_path):
    mesh = Mesh(np.asarray(jax.devices("cpu")[:1]), ("data",))
    perm = tmp_path / "checkpoints"
    temp = tmp_path / "temporary-checkpoints"
    with jax.set_mesh(mesh):
        state1 = State(jnp.array(1), jnp.arange(16, dtype=jnp.float32), jnp.full(16, 0.25))
        checkpointer = Checkpointer(
            perm,
            save_interval=None,
            step_policies=[CheckpointInterval(every=1, until=1)],
            temporary_base_path=temp,
            keep_last_temporary_checkpoints=1,
        )
        restored = restore_first_checkpoint(state1, checkpointer, mesh)
        assert int(restored.step) == 1
        assert json.loads((perm / "step-1" / "metadata.json").read_text())["is_temporary"] is False
        assert not (temp / "step-1").exists()


def test_first_step_restore_gate_reloads_permanent_root_when_permanent_request_queued(tmp_path):
    mesh = Mesh(np.asarray(jax.devices("cpu")[:1]), ("data",))
    perm = tmp_path / "checkpoints"
    temp = tmp_path / "temporary-checkpoints"
    with jax.set_mesh(mesh):
        state1 = State(jnp.array(1), jnp.arange(16, dtype=jnp.float32), jnp.full(16, 0.25))
        expected = [np.asarray(x).copy() for x in state1]
        checkpointer = Checkpointer(
            perm,
            save_interval=None,
            step_policies=[],
            temporary_base_path=temp,
            keep_last_temporary_checkpoints=1,
        )
        checkpointer.request_checkpoint(CheckpointRetention.PERMANENT)
        restored = restore_first_checkpoint(state1, checkpointer, mesh)
        for actual, wanted in zip(restored, expected, strict=True):
            np.testing.assert_array_equal(actual, wanted)
        assert json.loads((perm / "step-1" / "metadata.json").read_text())["is_temporary"] is False
        assert not (temp / "step-1").exists()


@pytest.mark.integration
def test_smoke_recipe_decodes_final_only_checkpointing(tmp_path):
    recipe = yaml.safe_load(SMOKE_CONFIG.read_text())
    cfg = fresh_sft_config(
        output=str(tmp_path / "run"), initializer=str(tmp_path / "base"), recipe=recipe, run_id="final-only"
    )
    # keep=None keeps every permanent checkpoint, including the final one, and the run keeps one
    # rolling temporary checkpoint for restores.
    assert cfg.trainer.trainer.checkpointer.keep is None
    assert cfg.trainer.trainer.checkpointer.keep_last_temporary_checkpoints == 1


@pytest.mark.integration
def test_fresh_sft_config_rejects_restore_gate_without_checkpoint_saving(tmp_path):
    recipe = yaml.safe_load(SMOKE_CONFIG.read_text())
    recipe["first_checkpoint_restore_step"] = 3
    recipe["trainer"]["save_checkpoints"] = False
    with pytest.raises(ParsingError) as raised:
        fresh_sft_config(
            output=str(tmp_path / "run"),
            initializer=str(tmp_path / "base"),
            recipe=recipe,
            run_id="gate-without-saves",
        )
    # draccus wraps construction failures; the decode error names the misconfiguration.
    assert str(raised.value.__cause__) == "first_checkpoint_restore_step requires save_checkpoints"
