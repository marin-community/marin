# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from dataclasses import dataclass
from jax.sharding import AxisType

from levanter.utils.mesh import MeshConfig, create_mesh_from_axis_specs


@dataclass(frozen=True)
class GpuDevice:
    id: int
    slice_index: int
    process_index: int
    platform: str = "gpu"
    device_kind: str = "NVIDIA H100"


def test_hybrid_gpu_mesh_preserves_iota_assignment_and_named_groups():
    devices = [GpuDevice(i, i // 8, i // 8) for i in range(256)]
    mesh = create_mesh_from_axis_specs(
        ici_axes={"expert": 8},
        dcn_axes={"context": 32},
        devices=list(reversed(devices)),
        axis_types=(AxisType.Explicit, AxisType.Auto),
    )
    assert [device.id for device in mesh.devices.flat] == list(range(256))
    assert mesh.axis_types == (AxisType.Auto, AxisType.Explicit)
    for node in range(32):
        assert [device.id for device in mesh.devices[node, :]] == list(range(node * 8, (node + 1) * 8))
    for expert in range(8):
        assert [device.id for device in mesh.devices[:, expert]] == list(range(expert, 256, 8))


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
