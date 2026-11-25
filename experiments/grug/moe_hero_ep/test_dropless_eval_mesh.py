# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import numpy as np
import pytest
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.grug.grug_moe import MoEExpertMlpPspecs
from levanter.grug.sharding import compact_grug_mesh
from levanter.testing.devices import use_test_devices

from experiments.grug.moe_hero_ep import model, train


def _device_groups(mesh, axes):
    """The device-id sets that vary along ``axes`` with every other mesh coordinate fixed."""
    moved = np.moveaxis(mesh.devices, [mesh.axis_names.index(axis) for axis in axes], range(-len(axes), 0))
    size = int(np.prod([mesh.shape[axis] for axis in axes]))
    return {frozenset(device.id for device in group) for group in moved.reshape(-1, size)}


@pytest.mark.multidevice
@pytest.mark.parametrize(("replica", "expert"), [(2, 2), (1, 2)])
def test_dropless_eval_mesh_reuses_training_device_groups_at_cp4(replica, expert):
    # A collective over a device group training never used makes XLA create and cache a new NCCL
    # communicator outside the allocator pool, which at 65K CP4 exhausted device memory. The eval
    # mesh must give every device its training context group and, as its data group, its training
    # (data, expert) group; moving the weights onto it must keep every batch- or context-sharded
    # shard on the device that already held it.
    with use_test_devices(16):
        if jax.process_count() != 1 or len(jax.devices()) != 16:
            pytest.skip("the layout comparison needs exactly sixteen devices in one process")
        train_mesh = compact_grug_mesh(replica_axis_size=replica, expert_axis_size=expert, context_axis_size=4)
        data = 16 // (replica * expert * 4)
        eval_mesh = train.expert_collapsed_eval_mesh(train_mesh)
        assert dict(eval_mesh.shape) == {
            "replica_dcn": replica,
            "data": data * expert,
            "expert": 1,
            "context": 4,
            "model": 1,
        }
        assert list(eval_mesh.devices.flat) == list(train_mesh.devices.flat)
        assert _device_groups(eval_mesh, ["context"]) == _device_groups(train_mesh, ["context"])
        assert _device_groups(eval_mesh, ["data"]) == _device_groups(train_mesh, ["data", "expert"])

        values = np.arange(16 * 8 * 8, dtype=np.float32).reshape(16, 8, 8)
        kept = {
            "batch": P(("replica_dcn", "data", "expert"), "context"),
            "fsdp": P(("data", "expert", "context")),
            "dense": P(None, "context", None),
        }
        tree = {key: jax.device_put(values, NamedSharding(train_mesh, spec)) for key, spec in kept.items()}
        # The routed-expert weights, as the hero shards them: experts over (expert, context), and the
        # hidden dimension over data. At training data 1 they must also stay put: moving them runs a
        # collective on a communicator training lacks, whose NCCL buffers outlive the eval. At 262K
        # CP4 x EP16 those buffers left the next train step 1.6 GiB short.
        expert_specs = MoEExpertMlpPspecs(expert=model._EXPERT_WEIGHT_AXES)
        expert_keys = {"w_gate_up": expert_specs.w_gate_up, "w_down": expert_specs.w_down}
        for key, spec in expert_keys.items():
            tree[key] = jax.device_put(values, NamedSharding(train_mesh, spec))
        moved = train._move_tree_to_eval_mesh(tree, train_mesh)
        for key in (*kept, *expert_keys):
            assert moved[key].sharding.mesh == eval_mesh
            np.testing.assert_array_equal(np.asarray(moved[key]), values)
        stationary = (*kept, *expert_keys) if data == 1 else kept
        for key in stationary:
            before = {shard.device: np.asarray(shard.data) for shard in tree[key].addressable_shards}
            after = {shard.device: np.asarray(shard.data) for shard in moved[key].addressable_shards}
            assert before.keys() == after.keys()
            for device, shard in before.items():
                np.testing.assert_array_equal(after[device], shard)
