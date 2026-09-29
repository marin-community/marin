# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""E2E test for local cluster mode via the CLI.

Uses ``iris cluster start --local`` through Click's test runner with the
canonical ``config/examples/local.yaml``, then submits a job through the
IrisClient to verify the full stack works.
"""

import re
import signal
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from iris.client.client import IrisClient
from iris.cluster.types import Entrypoint, ResourceSpec
from iris.rpc import job_pb2

pytestmark = pytest.mark.requires_cluster

LOCAL_CONFIG = Path(__file__).resolve().parents[2] / "config" / "examples" / "local.yaml"


def test_cli_local_cluster_e2e():
    """Start a local cluster via CLI, submit a job via IrisClient, verify completion."""
    output: list[str] = []
    controller_urls: list[str] = []
    controller_ready = threading.Event()
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "from iris.cli import main; main()",
            "--config",
            str(LOCAL_CONFIG),
            "cluster",
            "start",
            "--local",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    def read_output() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            output.append(line)
            match = re.search(r"Controller started at (http://\S+)", line)
            if match is not None:
                controller_urls.append(match.group(1))
                controller_ready.set()

    output_thread = threading.Thread(target=read_output, daemon=True)
    output_thread.start()

    try:
        assert controller_ready.wait(timeout=30), "Controller did not start:\n" + "".join(output)
        assert controller_urls

        # Submit a job through IrisClient; the autoscaler provisions a local
        # worker on demand, so the wait covers provisioning too.
        client = IrisClient.remote(controller_urls[0], workspace=Path.cwd())

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
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
        try:
            return_code = process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            return_code = process.wait(timeout=5)
        output_thread.join(timeout=5)

    assert return_code == 0, "".join(output)
