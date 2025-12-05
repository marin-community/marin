# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from levanter.utils.mesh import MeshConfig


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
