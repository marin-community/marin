# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import jax

import levanter.tracker.tracker_fns as tracker_fns
from levanter.distributed import DistributedConfig
from levanter.tracker.json_file import JsonFileTrackerConfig
from levanter.trainer import TrainerConfig
from levanter.utils.hardware_topology import nvidia_topology_matrix_summary, tpu_topology_shape


def test_trainer_initialize_logs_hardware_topology_to_tracker(tmp_path, monkeypatch):
    monkeypatch.setattr(tracker_fns, "_global_tracker", None)
    config = TrainerConfig(
        id="startup-topology",
        log_dir=tmp_path,
        train_batch_size=len(jax.devices()),
        tracker=JsonFileTrackerConfig(output_path=str(tmp_path)),
        require_accelerator=False,
        distributed=DistributedConfig(initialize_jax_distributed=False),
    )

    try:
        config.initialize()
        tracker_fns.current_tracker().finish()
    finally:
        monkeypatch.setattr(tracker_fns, "_global_tracker", None)

    with open(tmp_path / "eval_results.json") as f:
        summary = json.load(f)

    assert summary["hardware_topology/devices"]
    assert summary["hardware_topology/local_devices"]
    assert "hardware_topology/backend" not in summary
    assert "hardware_topology/device_count" not in summary
    assert "hardware_topology/local_device_count" not in summary
    assert "hardware_topology/process_count" not in summary
    assert "hardware_topology/process_index" not in summary
    assert "hardware_topology/mesh_axis_shapes" not in summary
    assert "hardware_topology/compute_axis_mapping" not in summary


def test_tpu_topology_shape_uses_device_coordinate_extents():
    devices = [SimpleNamespace(platform="tpu", coords=(x, y, z)) for x in range(4) for y in range(8) for z in range(8)]

    assert tpu_topology_shape(devices) == "4x8x8"


def test_tpu_topology_shape_includes_multiple_slices():
    devices = [
        SimpleNamespace(platform="tpu", slice_index=s, coords=(x, y, z))
        for s in range(2)
        for x in range(4)
        for y in range(8)
        for z in range(8)
    ]

    assert tpu_topology_shape(devices) == "2x4x8x8"


def test_nvidia_topology_matrix_summary_counts_gpu_and_nic_links():
    topology = """
        GPU0    GPU1    GPU2    NIC0    mlx5_0    CPU Affinity    NUMA Affinity
GPU0     X      NV18    SYS     PIX     SYS       0-95            0
GPU1    NV18     X      SYS     PXB     PIX       0-95            0
GPU2    SYS     SYS      X      SYS     SYS       0-95            0
NIC0    PIX     PXB     SYS      X      SYS
mlx5_0  SYS     PIX     SYS     SYS      X
    """

    assert nvidia_topology_matrix_summary(topology) == {
        "gpu_gpu_link_counts": {"NV18": 1, "SYS": 2},
        "gpu_nic_link_counts": {"PIX": 2, "PXB": 1, "SYS": 3},
    }
