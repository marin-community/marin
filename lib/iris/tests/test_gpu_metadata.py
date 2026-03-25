# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""GPU worker metadata test with mocked nvidia-smi."""

import subprocess
import time
import uuid
from unittest.mock import patch

from iris.cluster.config import connect_cluster
from iris.cluster.runtime.process import ProcessRuntime
from iris.cluster.worker.env_probe import DefaultEnvironmentProvider
from iris.cluster.worker.worker import Worker, WorkerConfig
from iris.managed_thread import ThreadContainer
from iris.rpc import cluster_pb2
from iris.rpc.cluster_connect import ControllerServiceClientSync
from iris.time_utils import Duration

from .conftest import _make_controller_only_config

_NVIDIA_SMI_H100_8X = "\n".join(["NVIDIA H100 80GB HBM3, 81559"] * 8)


def test_gpu_worker_metadata(tmp_path):
    """Mocked nvidia-smi registers GPU metadata on worker."""
    config = _make_controller_only_config()

    with connect_cluster(config) as url:
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
                worker_id=f"test-gpu-worker-{uuid.uuid4().hex[:8]}",
                poll_interval=Duration.from_seconds(0.1),
            )
            worker = Worker(
                worker_config,
                container_runtime=ProcessRuntime(cache_dir=cache_dir),
                environment_provider=env_provider,
                threads=threads,
            )
            worker.start()

            try:
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
                assert meta.gpu_count == 8
                assert "H100" in meta.gpu_name
                assert meta.gpu_memory_mb == 81559
                assert meta.device.gpu.count == 8
                assert "H100" in meta.device.gpu.variant

                from iris.cluster.constraints import WellKnownAttribute

                attrs = meta.attributes
                assert WellKnownAttribute.GPU_VARIANT in attrs
                assert "H100" in attrs[WellKnownAttribute.GPU_VARIANT].string_value
                assert WellKnownAttribute.GPU_COUNT in attrs
                assert attrs[WellKnownAttribute.GPU_COUNT].int_value == 8

                controller_client.close()
            finally:
                worker.stop()
                threads.stop(timeout=Duration.from_seconds(5.0))
