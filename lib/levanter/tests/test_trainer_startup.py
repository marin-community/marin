# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import jax

import levanter.tracker.tracker_fns as tracker_fns
from levanter.distributed import DistributedConfig
from levanter.tracker.json_file import JsonFileTrackerConfig
from levanter.trainer import TrainerConfig
from levanter.utils.hardware_topology import nvidia_topology_matrix_summary, tpu_topology_shape


_DISTRIBUTED_STARTUP_PROCESS = textwrap.dedent(
    """
    import importlib
    import socket
    import tempfile
    from types import SimpleNamespace

    import jax

    import levanter.cutlass_kernel_cache as cache_module
    import levanter.tracker.tracker_fns as tracker_fns
    import levanter.trainer as trainer_module
    from finestore.cache import PersistentKvCache
    from levanter.distributed import DistributedConfig
    from levanter.tracker import NoopConfig
    from levanter.trainer import TrainerConfig

    primitive = SimpleNamespace(get_or_compile_kernel=lambda fn, spec: None)
    compile_module = SimpleNamespace(_CUTLASS_COMPILE_CACHE={}, CompileResult=SimpleNamespace)
    real_import_module = importlib.import_module

    def import_module(name):
        if name == "cutlass.jax.primitive":
            jax.devices()
            return primitive
        if name == "cutlass.jax.compile":
            return compile_module
        return real_import_module(name)

    cache_module.importlib.import_module = import_module
    trainer_module.cutlass_kernel_cache = PersistentKvCache.in_memory

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]

    config = TrainerConfig(
        id="distributed-startup",
        log_dir=tempfile.mkdtemp(),
        tracker=NoopConfig(),
        train_batch_size=1,
        require_accelerator=False,
        distributed=DistributedConfig(
            coordinator_address=f"127.0.0.1:{port}",
            num_processes=1,
            process_id=0,
        ),
    )
    config.initialize()
    tracker_fns.current_tracker().finish()
    jax.distributed.shutdown()
    """
)


def test_trainer_initializes_distributed_before_importing_cutlass_jax():
    subprocess.run([sys.executable, "-c", _DISTRIBUTED_STARTUP_PROCESS], check=True, timeout=30)


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
