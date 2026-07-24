# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""E2E test for local cluster mode via the CLI.

Uses ``iris cluster start --local`` through Click's test runner with the
canonical ``config/examples/local.yaml``, then submits a job through the
IrisClient to verify the full stack works.
"""

import threading
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner
from iris.cli import iris
from iris.client import IrisClient
from iris.cluster.local_cluster import LocalCluster
from iris.cluster.types import Entrypoint, ResourceSpec
from iris.rpc import job_pb2

pytestmark = pytest.mark.requires_cluster

LOCAL_CONFIG = Path(__file__).resolve().parents[2] / "config" / "examples" / "local.yaml"


def test_cli_local_cluster_e2e():
    """Start a local cluster via CLI, submit a job via IrisClient, verify completion."""
    runner = CliRunner()

    # Capture the LocalCluster instance so we can get the address and stop it
    captured_controller: list[LocalCluster] = []
    controller_ready = threading.Event()
    original_start = LocalCluster.start

    def patched_start(self):
        captured_controller.append(self)
        result = original_start(self)
        controller_ready.set()
        return result

    # Run CLI in a background thread because `cluster start --local` blocks
    # until the controller is stopped.
    invoke_result: list = []

    def run_cli():
        with patch.object(LocalCluster, "start", patched_start):
            invoke_result.append(
                runner.invoke(
                    iris,
                    ["--config", str(LOCAL_CONFIG), "cluster", "start", "--local"],
                )
            )

    cli_thread = threading.Thread(target=run_cli, daemon=True)
    cli_thread.start()

    if not controller_ready.wait(timeout=30):
        cli_thread.join(timeout=1)
        if invoke_result and invoke_result[0].exception:
            raise AssertionError(
                f"CLI exited before the controller started: {invoke_result[0].output}"
            ) from invoke_result[0].exception
        raise AssertionError("Controller didn't start in time")
    assert len(captured_controller) == 1

    controller = captured_controller[0]
    try:
        address = controller.discover()
        assert address is not None

        # Submit a job through IrisClient; the autoscaler provisions a local
        # worker on demand, so the wait covers provisioning too.
        client = IrisClient.remote(address, workspace=Path.cwd())

        def hello():
            return 42

        job = client.submit(
            entrypoint=Entrypoint.from_callable(hello),
            name="cli-e2e-hello",
            resources=ResourceSpec(cpu=1),
        )

        status = job.wait(timeout=90, raise_on_failure=True)
        assert status.state == job_pb2.JOB_STATE_SUCCEEDED
    finally:
        controller.close()
        cli_thread.join(timeout=5)

    assert invoke_result, "CLI did not return"
    result = invoke_result[0]
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert "Controller started at" in result.output
