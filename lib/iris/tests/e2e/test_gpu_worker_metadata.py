# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""E2E test: GPU worker metadata propagation.

Validates that when a worker with GPUs registers with the controller, the
controller's ListWorkers API returns the correct GPU count, name, memory,
and DeviceConfig. This exercises the real DefaultEnvironmentProvider code
path with a mocked nvidia-smi subprocess.
"""

import subprocess
import time
from unittest.mock import patch

import pytest
from iris.cluster.config import load_config, make_local_config
from iris.cluster.manager import connect_cluster
from iris.cluster.runtime.process import ProcessRuntime
from iris.cluster.worker.env_probe import DefaultEnvironmentProvider
from iris.cluster.worker.worker import Worker, WorkerConfig
from iris.managed_thread import ThreadContainer
from iris.rpc import cluster_pb2, config_pb2
from iris.rpc.cluster_connect import ControllerServiceClientSync
from iris.time_utils import Duration

from .conftest import DEFAULT_CONFIG

pytestmark = pytest.mark.e2e

# nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
# output for 8x H100 80GB
_NVIDIA_SMI_H100_8X = "\n".join(["NVIDIA H100 80GB HBM3, 81559"] * 8)


def _make_controller_only_config() -> config_pb2.IrisClusterConfig:
    """Build a local config with no auto-scaled workers (min_slices=0, max_slices=0)."""
    config = load_config(DEFAULT_CONFIG)
    config.scale_groups.clear()

    sg = config.scale_groups["placeholder"]
    sg.name = "placeholder"
    sg.accelerator_type = config_pb2.ACCELERATOR_TYPE_CPU
    sg.num_vms = 1
    sg.min_slices = 0
    sg.max_slices = 0
    sg.resources.cpu = 1
    sg.resources.memory_bytes = 1 * 1024**3
    sg.resources.disk_bytes = 10 * 1024**3
    sg.slice_template.local.SetInParent()

    return make_local_config(config)


def test_gpu_worker_metadata_visible_to_controller(tmp_path):
    """A worker using DefaultEnvironmentProvider with mocked nvidia-smi registers
    with the controller, and ListWorkers returns correct GPU metadata."""
    config = _make_controller_only_config()

    with connect_cluster(config) as url:
        # Patch subprocess.run in env_probe to intercept nvidia-smi
        original_run = subprocess.run
        with patch(
            "iris.cluster.worker.env_probe.subprocess.run",
            side_effect=lambda cmd, *a, **kw: (
                subprocess.CompletedProcess(args=cmd, returncode=0, stdout=_NVIDIA_SMI_H100_8X, stderr="")
                if isinstance(cmd, list) and cmd and cmd[0] == "nvidia-smi"
                else original_run(cmd, *a, **kw)
            ),
        ):
            env_provider = DefaultEnvironmentProvider()
            threads = ThreadContainer(name="test-gpu-worker")
            cache_dir = tmp_path / "cache"
            cache_dir.mkdir()

            worker_config = WorkerConfig(
                host="127.0.0.1",
                port=0,
                cache_dir=cache_dir,
                controller_address=url,
                poll_interval=Duration.from_seconds(0.1),
            )
            worker = Worker(
                worker_config,
                container_runtime=ProcessRuntime(),
                environment_provider=env_provider,
                threads=threads,
            )
            worker.start()

            try:
                # Wait for worker to register
                controller_client = ControllerServiceClientSync(address=url, timeout_ms=10000)
                deadline = time.monotonic() + 15.0
                workers = []
                while time.monotonic() < deadline:
                    request = cluster_pb2.Controller.ListWorkersRequest()
                    response = controller_client.list_workers(request)
                    workers = [w for w in response.workers if w.healthy]
                    if workers:
                        break
                    time.sleep(0.5)

                assert workers, "Worker did not register within timeout"

                w = workers[0]
                meta = w.metadata

                # Scalar GPU fields (populated by DefaultEnvironmentProvider from nvidia-smi)
                assert meta.gpu_count == 8, f"Expected gpu_count=8, got {meta.gpu_count}"
                assert "H100" in meta.gpu_name, f"Expected 'H100' in gpu_name, got {meta.gpu_name!r}"
                assert meta.gpu_memory_mb == 81559, f"Expected gpu_memory_mb=81559, got {meta.gpu_memory_mb}"

                # DeviceConfig GPU oneof
                assert meta.device.HasField("gpu"), f"Expected device.gpu, got {meta.device}"
                assert meta.device.gpu.count == 8, f"Expected device.gpu.count=8, got {meta.device.gpu.count}"
                assert (
                    "H100" in meta.device.gpu.variant
                ), f"Expected 'H100' in device.gpu.variant, got {meta.device.gpu.variant!r}"

                # Worker attributes map (used by dashboard and constraint scheduler)
                attrs = meta.attributes
                assert "gpu-variant" in attrs, f"Expected 'gpu-variant' in attributes, got keys: {sorted(attrs.keys())}"
                assert "H100" in attrs["gpu-variant"].string_value
                assert "gpu-count" in attrs, f"Expected 'gpu-count' in attributes, got keys: {sorted(attrs.keys())}"
                assert attrs["gpu-count"].int_value == 8

                controller_client.close()
            finally:
                worker.stop()
                threads.stop(timeout=Duration.from_seconds(5.0))
