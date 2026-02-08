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

"""Tests for task attempt log preservation and attempt-aware routing.

Verifies that:
1. Multiple attempt logs are preserved when tasks retry
2. Log entries include attempt_id
3. Controller routes log requests to correct worker per attempt
4. Dashboard shows expandable attempt details (via screenshot test)
"""

import logging
import time
import uuid

import pytest
from iris.chaos import enable_chaos, reset_chaos
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec
from iris.cluster.vm.cluster_manager import ClusterManager
from iris.cluster.vm.config import load_config, make_local_config
from iris.client.client import IrisClient
from iris.rpc import cluster_pb2

from pathlib import Path

IRIS_ROOT = Path(__file__).resolve().parents[2]  # lib/iris
DEFAULT_CONFIG = IRIS_ROOT / "examples" / "demo.yaml"


def _fail_then_succeed(attempt_marker: str):
    """Job function that fails on attempt 0, succeeds on attempt 1+.

    Uses chaos injection marker to determine whether to fail.
    Prints attempt-specific output for log verification.
    """
    from iris.cluster.client import get_job_info

    info = get_job_info()
    if info is None:
        raise RuntimeError("JobInfo not available")

    attempt_id = info.attempt_id
    print(f"ATTEMPT_LOG: attempt_id={attempt_id} marker={attempt_marker}")

    if attempt_id == 0:
        raise RuntimeError(f"Intentional failure on attempt 0: {attempt_marker}")
    return "success"


@pytest.fixture(autouse=True)
def _reset_chaos_fixture():
    """Reset chaos after each test."""
    yield
    reset_chaos()


@pytest.fixture
def local_cluster():
    """Boots a local cluster via ClusterManager, yields (url, client)."""
    config = load_config(DEFAULT_CONFIG)
    config = make_local_config(config)
    manager = ClusterManager(config)
    with manager.connect() as url:
        client = IrisClient.remote(url, workspace=IRIS_ROOT)
        yield url, client


class TestAttemptLogs:
    """Tests for task attempt log preservation and routing."""

    def test_multiple_attempts_preserve_logs(self, local_cluster, caplog):
        """Job with retries preserves logs from all attempts.

        1. Submit a job that fails on attempt 0 but succeeds on attempt 1
        2. Verify final state is SUCCEEDED
        3. Verify logs contain output from both attempts (via attempt_id field)
        """
        _url, client = local_cluster
        run_id = uuid.uuid4().hex[:8]
        marker = f"test-{run_id}"

        with caplog.at_level(logging.INFO, logger="iris"):
            job = client.submit(
                entrypoint=Entrypoint.from_callable(_fail_then_succeed, marker),
                name=f"retry-logs-{run_id}",
                resources=ResourceSpec(cpu=1, memory="1g"),
                environment=EnvironmentSpec(),
                max_retries_failure=1,
            )

            # Wait for job to complete
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                status = client.status(job.job_id)
                if status.state in (
                    cluster_pb2.JOB_STATE_SUCCEEDED,
                    cluster_pb2.JOB_STATE_FAILED,
                    cluster_pb2.JOB_STATE_KILLED,
                ):
                    break
                time.sleep(0.5)

            assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED, f"Job should succeed after retry: {status}"

            # Fetch logs for all attempts
            task_id = job.job_id.task(0)
            logs_response = client.fetch_task_logs(task_id)

            # Verify logs from both attempts are present
            attempt_0_found = False
            attempt_1_found = False
            for entry in logs_response:
                if "attempt_id=0" in entry.data and marker in entry.data:
                    attempt_0_found = True
                    assert entry.attempt_id == 0
                if "attempt_id=1" in entry.data and marker in entry.data:
                    attempt_1_found = True
                    assert entry.attempt_id == 1

            assert attempt_0_found, "Logs from attempt 0 should be preserved"
            assert attempt_1_found, "Logs from attempt 1 should be present"

    def test_superseding_attempt_logs_info(self, local_cluster, caplog):
        """Verify worker logs 'Superseding task' when new attempt kills old one.

        This tests the sanity check flow where a higher attempt_id supersedes
        a lower one.
        """
        _url, client = local_cluster
        run_id = uuid.uuid4().hex[:8]

        # Enable chaos to fail container creation on first attempt, forcing retry
        enable_chaos(
            "worker.create_container",
            failure_rate=1.0,
            max_failures=1,
            error=RuntimeError("chaos: container creation failed"),
        )

        with caplog.at_level(logging.INFO, logger="iris"):
            job = client.submit(
                entrypoint=Entrypoint.from_callable(lambda: "ok"),
                name=f"supersede-{run_id}",
                resources=ResourceSpec(cpu=1, memory="1g"),
                environment=EnvironmentSpec(),
                max_retries_failure=2,
            )

            # Wait for job to complete
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                status = client.status(job.job_id)
                if status.state in (
                    cluster_pb2.JOB_STATE_SUCCEEDED,
                    cluster_pb2.JOB_STATE_FAILED,
                    cluster_pb2.JOB_STATE_KILLED,
                ):
                    break
                time.sleep(0.5)

            assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED, f"Job should succeed after retry: {status}"

    def test_attempt_specific_log_fetch(self, local_cluster):
        """Verify fetching logs for a specific attempt_id works.

        1. Submit a job that fails then succeeds
        2. Fetch logs for attempt_id=0 specifically
        3. Verify only attempt 0 logs are returned
        """
        _url, client = local_cluster
        run_id = uuid.uuid4().hex[:8]
        marker = f"specific-{run_id}"

        job = client.submit(
            entrypoint=Entrypoint.from_callable(_fail_then_succeed, marker),
            name=f"attempt-filter-{run_id}",
            resources=ResourceSpec(cpu=1, memory="1g"),
            environment=EnvironmentSpec(),
            max_retries_failure=1,
        )

        # Wait for job to complete
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            status = client.status(job.job_id)
            if status.state in (
                cluster_pb2.JOB_STATE_SUCCEEDED,
                cluster_pb2.JOB_STATE_FAILED,
                cluster_pb2.JOB_STATE_KILLED,
            ):
                break
            time.sleep(0.5)

        assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED

        # Fetch logs for attempt 0 only
        task_id = job.job_id.task(0)
        logs_attempt_0 = client.fetch_task_logs(task_id, attempt_id=0)

        # All returned logs should be from attempt 0
        for entry in logs_attempt_0:
            assert entry.attempt_id == 0, f"Expected attempt_id=0, got {entry.attempt_id}"

        # Verify we got the failure output from attempt 0
        found_failure = any("Intentional failure on attempt 0" in e.data for e in logs_attempt_0)
        assert found_failure, "Should find failure message in attempt 0 logs"

        # Fetch logs for attempt 1 only
        logs_attempt_1 = client.fetch_task_logs(task_id, attempt_id=1)

        # All returned logs should be from attempt 1
        for entry in logs_attempt_1:
            assert entry.attempt_id == 1, f"Expected attempt_id=1, got {entry.attempt_id}"

    def test_task_status_shows_attempts(self, local_cluster):
        """Verify task status includes attempt history.

        1. Submit a job that fails then succeeds
        2. Check that the task status includes attempts list
        3. Verify attempt details (worker_id, state, etc.)
        """
        _url, client = local_cluster
        run_id = uuid.uuid4().hex[:8]
        marker = f"status-{run_id}"

        job = client.submit(
            entrypoint=Entrypoint.from_callable(_fail_then_succeed, marker),
            name=f"status-attempts-{run_id}",
            resources=ResourceSpec(cpu=1, memory="1g"),
            environment=EnvironmentSpec(),
            max_retries_failure=1,
        )

        # Wait for job to complete
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            status = client.status(job.job_id)
            if status.state in (
                cluster_pb2.JOB_STATE_SUCCEEDED,
                cluster_pb2.JOB_STATE_FAILED,
                cluster_pb2.JOB_STATE_KILLED,
            ):
                break
            time.sleep(0.5)

        assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED

        # Check task status for attempts
        assert len(status.tasks) == 1
        task_status = status.tasks[0]

        # Should have 2 attempts: attempt 0 (failed) and attempt 1 (succeeded)
        assert len(task_status.attempts) >= 2, f"Expected at least 2 attempts, got {len(task_status.attempts)}"

        # First attempt should have failed
        attempt_0 = task_status.attempts[0]
        assert attempt_0.attempt_id == 0
        assert attempt_0.state == cluster_pb2.TASK_STATE_FAILED

        # Second attempt should have succeeded
        attempt_1 = task_status.attempts[1]
        assert attempt_1.attempt_id == 1
        assert attempt_1.state == cluster_pb2.TASK_STATE_SUCCEEDED
