# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""E2E test for local cluster mode via the CLI.

Uses ``iris cluster start --local`` through Click's test runner, then submits
a job through the IrisClient to verify the full stack works.
"""

import re
import time
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from click.testing import CliRunner
from iris.cli import iris
from iris.client import IrisClient
from iris.cluster.types import Entrypoint, ResourceSpec
from iris.cluster.platform.cluster_manager import ClusterManager
from iris.rpc import cluster_pb2
from iris.rpc.cluster_connect import ControllerServiceClientSync


@pytest.fixture
def cluster_config_file(tmp_path: Path) -> Path:
    config_path = tmp_path / "cluster.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "platform": {
                    "gcp": {
                        "project_id": "test-project",
                        "region": "us-central1",
                        "zone": "us-central1-a",
                    }
                },
                "defaults": {
                    "bootstrap": {
                        "docker_image": "test-image:latest",
                        "worker_port": 10001,
                        "controller_address": "127.0.0.1:10000",
                    },
                },
                "controller": {
                    "gcp": {
                        "port": 10000,
                    }
                },
                "scale_groups": {
                    "local-cpu": {
                        "vm_type": "gce_vm",
                        "accelerator_type": "ACCELERATOR_TYPE_CPU",
                        "min_slices": 1,
                        "max_slices": 1,
                        "zones": ["us-central1-a"],
                        "slice_size": 1,
                        "resources": {
                            "cpu": 1,
                            "ram": "1GB",
                            "disk": 0,
                            "gpu_count": 0,
                            "tpu_count": 0,
                        },
                    },
                },
            }
        )
    )
    return config_path


def _wait_for_workers(address: str, timeout: float = 30.0) -> None:
    """Poll until at least one worker registers (healthy or not)."""
    client = ControllerServiceClientSync(address=address, timeout_ms=5000)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            resp = client.list_workers(cluster_pb2.Controller.ListWorkersRequest())
            if resp.workers:
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise TimeoutError(f"No workers registered within {timeout}s")


def test_cli_local_cluster_e2e(cluster_config_file: Path):
    """Start a local cluster via CLI, submit a job via IrisClient, verify completion."""
    runner = CliRunner()

    # Capture the ClusterManager instance so we can get the address and stop it
    captured_manager: list[ClusterManager] = []
    original_start = ClusterManager.start

    def patched_start(self):
        captured_manager.append(self)
        return original_start(self)

    with patch.object(ClusterManager, "start", patched_start):
        result = runner.invoke(
            iris,
            ["--config", str(cluster_config_file), "cluster", "start", "--local"],
        )

    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert "Controller started at" in result.output
    assert len(captured_manager) == 1

    manager = captured_manager[0]
    try:
        # Extract the address from the CLI output
        address = None
        for line in result.output.splitlines():
            m = re.search(r"Controller started at (http://\S+)", line)
            if m:
                address = m.group(1)
                break
        if not address:
            pytest.fail(f"Could not find controller address in CLI output:\n{result.output}")

        _wait_for_workers(address)

        # Submit a job through IrisClient
        client = IrisClient.remote(address, workspace=Path.cwd())

        def hello():
            return 42

        job = client.submit(
            entrypoint=Entrypoint.from_callable(hello),
            name="cli-e2e-hello",
            resources=ResourceSpec(cpu=1),
        )

        status = job.wait(timeout=30, raise_on_failure=True)
        assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED
    finally:
        manager.stop()
