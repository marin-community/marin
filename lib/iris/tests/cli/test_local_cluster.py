# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""E2E test for local cluster mode via the CLI.

Uses ``iris cluster start --local`` through Click's test runner, then submits
a job through the IrisClient to verify the full stack works.
"""

import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from click.testing import CliRunner
from iris.cli import iris
from iris.client import IrisClient
from iris.cluster.types import Entrypoint, ResourceSpec
from iris.cluster.controller.local import LocalController
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
                    }
                },
                "defaults": {
                    "worker": {
                        "docker_image": "test-image:latest",
                        "port": 10001,
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
                        "accelerator_type": "ACCELERATOR_TYPE_CPU",
                        "min_slices": 1,
                        "max_slices": 1,
                        "num_vms": 1,
                        "resources": {
                            "cpu": 1,
                            "ram": "1GB",
                            "disk": 0,
                            "gpu_count": 0,
                            "tpu_count": 0,
                        },
                        "slice_template": {
                            "accelerator_type": "ACCELERATOR_TYPE_CPU",
                            "num_vms": 1,
                            "local": {},
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

    # Capture the LocalController instance so we can get the address and stop it
    captured_controller: list[LocalController] = []
    controller_ready = threading.Event()
    original_start = LocalController.start

    def patched_start(self):
        captured_controller.append(self)
        result = original_start(self)
        controller_ready.set()
        return result

    # Run CLI in a background thread because `cluster start --local` blocks
    # until the controller is stopped.
    invoke_result: list = []

    def run_cli():
        with patch.object(LocalController, "start", patched_start):
            invoke_result.append(
                runner.invoke(
                    iris,
                    ["--config", str(cluster_config_file), "cluster", "start", "--local"],
                )
            )

    cli_thread = threading.Thread(target=run_cli, daemon=True)
    cli_thread.start()

    assert controller_ready.wait(timeout=10), "Controller didn't start in time"
    assert len(captured_controller) == 1

    controller = captured_controller[0]
    try:
        address = controller.discover()
        assert address is not None

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
        controller.stop()
        cli_thread.join(timeout=5)

    assert invoke_result, "CLI did not return"
    result = invoke_result[0]
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert "Controller started at" in result.output
