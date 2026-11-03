# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.testing.devices import use_test_devices

from experiments.grug.moe_hero_ep import train

MESH_AXES = ("replica_dcn", "data", "context", "expert", "model")


def _device_groups(mesh, axis):
    moved = np.moveaxis(mesh.devices, mesh.axis_names.index(axis), -1)
    return {tuple(d.id for d in group) for group in moved.reshape(-1, moved.shape[-1])}


@pytest.mark.multidevice
def test_dropless_eval_mesh_reuses_training_device_groups_and_shards():
    # A collective over a device group training never used makes XLA create and cache a new NCCL
    # communicator, which at 65K CP4 exhausts device memory. The eval mesh must give every device the
    # same context group and, as its data group, its training (data, expert) group; moving the
    # weights onto it must leave every device holding the shard it already had.
    with use_test_devices(16) as devices:
        devices = np.asarray(devices)
        train_mesh = Mesh(devices.reshape(2, 1, 2, 4, 1), MESH_AXES, axis_types=(AxisType.Explicit,) * 5)
        eval_mesh = train.expert_collapsed_eval_mesh(train_mesh)
        assert dict(eval_mesh.shape) == {"replica_dcn": 2, "context": 2, "data": 4, "expert": 1, "model": 1}
        assert list(eval_mesh.devices.flat) == list(train_mesh.devices.flat)

        assert _device_groups(eval_mesh, "context") == _device_groups(train_mesh, "context")
        assert _device_groups(eval_mesh, "data") == _device_groups(train_mesh, "expert")

        values = np.arange(16 * 8 * 6, dtype=np.float32).reshape(16, 8, 6)
        tree = {
            "experts": jax.device_put(values, NamedSharding(train_mesh, P("expert", None, None))),
            "batch": jax.device_put(values, NamedSharding(train_mesh, P(("replica_dcn", "data", "expert"), "context"))),
            "dense": jax.device_put(values, NamedSharding(train_mesh, P(None, "context", None))),
            "step": 3,
        }
        moved = train._reshard_tree_to_mesh(tree, train_mesh, eval_mesh)
        assert moved["step"] == 3
        for key in ("experts", "batch", "dense"):
            assert moved[key].sharding.mesh == eval_mesh
            before = {s.device: np.asarray(s.data) for s in tree[key].addressable_shards}
            after = {s.device: np.asarray(s.data) for s in moved[key].addressable_shards}
            assert before.keys() == after.keys()
            for device, shard in before.items():
                np.testing.assert_array_equal(after[device], shard)

        # With training data > 1 no eval axis matches expert alone; the move must still keep every value.
        wide_mesh = Mesh(devices.reshape(1, 2, 2, 4, 1), MESH_AXES, axis_types=(AxisType.Explicit,) * 5)
        wide_eval_mesh = train.expert_collapsed_eval_mesh(wide_mesh)
        assert _device_groups(wide_eval_mesh, "context") == _device_groups(wide_mesh, "context")
        wide_tree = {
            "experts": jax.device_put(values, NamedSharding(wide_mesh, P(("expert", "context"), None, None))),
            "batch": jax.device_put(values, NamedSharding(wide_mesh, P(("data", "expert"), "context"))),
        }
        for leaf in train._reshard_tree_to_mesh(wide_tree, wide_mesh, wide_eval_mesh).values():
            np.testing.assert_array_equal(np.asarray(leaf), values)
