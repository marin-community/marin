# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""CLI dispatch integration tests — submit jobs via 'iris job run' subprocess."""

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
from iris.cluster.types import is_job_finished
from iris.rpc import cluster_pb2

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _find_iris_executable() -> str:
    """Locate the ``iris`` entry-point script in the current venv."""
    venv_bin = Path(sys.executable).parent / "iris"
    if venv_bin.exists():
        return str(venv_bin)
    found = shutil.which("iris")
    if found:
        return found
    raise FileNotFoundError("Cannot find 'iris' CLI entry point")


def _run_iris_cli(controller_url: str, *args: str, timeout: float = 60) -> subprocess.CompletedProcess:
    """Run an iris CLI command as a subprocess."""
    cmd = [_find_iris_executable(), "--controller-url", controller_url, *args]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _wait_for_job_state(
    controller_client, job_id: str, expected_state: int, timeout: float = 120.0
) -> cluster_pb2.JobStatus:
    """Poll job status until expected state or a terminal state is reached."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        request = cluster_pb2.Controller.GetJobStatusRequest(job_id=job_id)
        response = controller_client.get_job_status(request)
        state = response.job.state
        if state == expected_state:
            return response.job
        if is_job_finished(state):
            raise AssertionError(
                f"Job reached {cluster_pb2.JobState.Name(state)}, "
                f"expected {cluster_pb2.JobState.Name(expected_state)}"
            )
        time.sleep(1)
    raise TimeoutError(f"Job {job_id} did not reach {cluster_pb2.JobState.Name(expected_state)} in {timeout}s")


def test_cli_submit_and_succeed(integration_cluster):
    """Submit a simple command via 'iris job run' CLI and verify it succeeds."""
    # Use --cpu 2 to avoid the executor heuristic (≤1 CPU → non-preemptible),
    # since test cluster workers are all preemptible.
    result = _run_iris_cli(
        integration_cluster.url,
        "job",
        "run",
        "--no-wait",
        "--job-name",
        "itest-cli-ok",
        "--cpu",
        "2",
        "--memory",
        "1g",
        "--",
        "python",
        "-c",
        "print('hello from cli')",
    )
    assert result.returncode == 0, f"CLI failed: {result.stderr}"

    job_id = result.stdout.strip().splitlines()[-1].strip()
    assert job_id, f"No job ID in output: {result.stdout}"

    _wait_for_job_state(integration_cluster.controller_client, job_id, cluster_pb2.JOB_STATE_SUCCEEDED)


def test_cli_submit_failing_command(integration_cluster):
    """CLI-submitted job that exits non-zero is reported as FAILED."""
    result = _run_iris_cli(
        integration_cluster.url,
        "job",
        "run",
        "--no-wait",
        "--job-name",
        "itest-cli-fail",
        "--cpu",
        "2",
        "--memory",
        "1g",
        "--",
        "python",
        "-c",
        "raise SystemExit(1)",
    )
    assert result.returncode == 0, f"CLI submission failed: {result.stderr}"

    job_id = result.stdout.strip().splitlines()[-1].strip()
    _wait_for_job_state(integration_cluster.controller_client, job_id, cluster_pb2.JOB_STATE_FAILED)


def test_cli_job_list(integration_cluster):
    """Verify 'iris job list --json' returns valid JSON list."""
    result = _run_iris_cli(
        integration_cluster.url,
        "job",
        "list",
        "--json",
    )
    assert result.returncode == 0, f"job list failed: {result.stderr}"

    jobs = json.loads(result.stdout)
    assert isinstance(jobs, list)
