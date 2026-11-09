# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec
from levanter.checkpoint import Checkpointer

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
