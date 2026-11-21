# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import jax
import numpy as np
import pytest
from jax.sharding import NamedSharding, PartitionSpec

from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig


@dataclass(frozen=True)
class TopologyDevice:
    id: int
    process_index: int
    slice_index: int
    platform: str
    device_kind: str
    coords: tuple[int, int, int]
    core_on_chip: int = 0


@pytest.mark.parametrize("platform", ["gpu", "tpu"])
def test_trainer_mesh_preserves_gpu_global_axes_and_tpu_slice_axes(platform, monkeypatch):
    # GPU PJRT 0.11 exposes a slice_index per host. These are not TPU slices.
    devices = [
        TopologyDevice(
            id=host * 131072 + local,
            process_index=host,
            slice_index=host,
            platform=platform,
            device_kind="NVIDIA H100 80GB HBM3" if platform == "gpu" else "TPU v5 lite",
            coords=(local // 4, (local // 2) % 2, local % 2),
        )
        for host in range(4)
        for local in range(8)
    ]
    devices.reverse()
    monkeypatch.setattr(jax, "devices", lambda: devices)
    monkeypatch.setattr(jax, "device_count", lambda: len(devices))
    config = TrainerConfig(
        train_batch_size=512,
        per_device_parallelism=1,
        mesh=MeshConfig(
            axes={"data": 4 if platform == "gpu" else 1, "expert": 8},
            dcn_axes={"replica_dcn": 1 if platform == "gpu" else 4},
            compute_mapping={"batch": ["replica_dcn", "data", "expert"]},
        ),
    )
    mesh = config.device_mesh
    assert config.data_axis_size == 32
    assert config.microbatch_size == 32
    assert mesh.shape["expert"] == 8
    # Each expert collective stays on one eight-device host for both layouts.
    host_axis = "data" if platform == "gpu" else "replica_dcn"
    host_rows = np.moveaxis(mesh.devices, mesh.axis_names.index(host_axis), 0).reshape(4, 8)
    np.testing.assert_array_equal(
        [[device.process_index for device in row] for row in host_rows],
        np.repeat(np.arange(4)[:, None], 8, axis=1),
    )
    assert {device.id for device in mesh.devices.flat} == {
        host * 131072 + local for host in range(4) for local in range(8)
    }
    if platform == "gpu":
        batch_sharding = NamedSharding(mesh, PartitionSpec(("replica_dcn", "data", "expert"), None))
        expert_sharding = NamedSharding(mesh, PartitionSpec("expert", None))
        batch_indices = batch_sharding.devices_indices_map((512, 8))
        expert_indices = expert_sharding.devices_indices_map((64, 8))
        for device in devices:
            local = device.id % 131072
            rank = device.process_index * 8 + local
            assert batch_indices[device] == (slice(rank * 16, (rank + 1) * 16), slice(None))
            assert expert_indices[device] == (slice(local * 8, (local + 1) * 8), slice(None))


def test_axis_shapes_inherit_defaults_and_absorb():
    cfg = MeshConfig(axes={"model": 2})
    ici, dcn = cfg.axis_shapes(num_devices=8, num_slices=1)
    # data should absorb remaining ICI after replica=1, model=2 -> data = 4
    assert ici == {"data": 4, "replica": 1, "model": 2}
    # replica_dcn should absorb all slices by default
    assert dcn == {"replica_dcn": 1}


def test_axis_shapes_force_data_when_other_absorber():
    cfg = MeshConfig(axes={"model": -1})
    ici, _ = cfg.axis_shapes(num_devices=4, num_slices=1)
    # data forced to 1 to keep single absorber
    assert ici["data"] == 1
    # model absorbs the rest
    assert ici["model"] == 4


def test_axis_shapes_force_replica_dcn_when_other_absorber():
    cfg = MeshConfig(dcn_axes={"other_dcn": -1})
    _, dcn = cfg.axis_shapes(num_devices=8, num_slices=2)
    # replica_dcn forced to 1 to leave only one absorber
    assert dcn["replica_dcn"] == 1
    assert dcn["other_dcn"] == 2


def test_axis_shapes_overlap_error():
    cfg = MeshConfig(axes={"data": 1}, dcn_axes={"data": 1})
    with pytest.raises(ValueError):
        cfg.axis_shapes(num_devices=4, num_slices=1)


def test_axis_shapes_multiple_absorbers_error():
    cfg = MeshConfig(axes={"data": -1, "model": -1})
    with pytest.raises(ValueError):
        cfg.axis_shapes(num_devices=8, num_slices=1)


def test_axis_shapes_multiple_dcn_absorbers_error():
    cfg = MeshConfig(dcn_axes={"replica_dcn": -1, "other": -1})
    with pytest.raises(ValueError):
        cfg.axis_shapes(num_devices=8, num_slices=2)


def test_resolved_param_mapping_inherits_shared():
    cfg = MeshConfig()
    # shared mapping defaults map mlp/heads to model
    mapping = cfg.resolved_param_mapping
    assert mapping["mlp"] == "model"
    assert mapping["heads"] == "model"
    # embed should come from the default param_mapping override
    assert mapping["embed"] == "data"
