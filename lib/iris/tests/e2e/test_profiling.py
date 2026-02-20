# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Profiling E2E tests: py-spy CPU profiling via ProfileTask RPC.

Migrated from tests/cluster/test_e2e.py::TestProfiling.
"""

import time

import pytest
from connectrpc.errors import ConnectError
from iris.rpc import cluster_pb2
from iris.time_utils import Duration, ExponentialBackoff

pytestmark = pytest.mark.e2e


def _wait_for_task_running(cluster, job, timeout: float = 30.0) -> str:
    """Wait for task 0 of a job to reach TASK_STATE_RUNNING and return its task_id.

    The worker's profile_task RPC requires the task to be in RUNNING state
    (not BUILDING), so we poll the task status rather than the job status.
    """
    last_state = "unknown"

    def _is_running() -> bool:
        nonlocal last_state
        task = cluster.task_status(job, task_index=0)
        last_state = task.state
        return last_state == cluster_pb2.TASK_STATE_RUNNING

    ExponentialBackoff(initial=0.1, maximum=2.0).wait_until_or_raise(
        _is_running,
        timeout=Duration.from_seconds(timeout),
        error_message=f"Task did not reach RUNNING within {timeout}s, last state: {last_state}",
    )
    return cluster.task_status(job, task_index=0).task_id


def test_profile_running_task(cluster):
    """Profile a running task and verify we get profile data back."""

    def slow_task():
        # Busy-loop with real Python work so py-spy can capture stack frames
        # (time.sleep blocks in C, producing zero Python samples)
        end = time.monotonic() + 3
        while time.monotonic() < end:
            sum(range(1000))

    job = cluster.submit(slow_task, name="profile-test")
    task_id = _wait_for_task_running(cluster, job)

    request = cluster_pb2.ProfileTaskRequest(
        task_id=task_id,
        duration_seconds=1,
        profile_type=cluster_pb2.ProfileType(cpu=cluster_pb2.CpuProfile(format=cluster_pb2.CpuProfile.FLAMEGRAPH)),
    )
    response = cluster.controller_client.profile_task(request, timeout_ms=3000)

    assert len(response.profile_data) > 0
    assert not response.error

    cluster.wait(job, timeout=30)


def test_profile_formats(cluster):
    """All three CPU profile formats return data."""

    def slow_task():
        end = time.monotonic() + 4
        while time.monotonic() < end:
            sum(range(1000))

    job = cluster.submit(slow_task, name="profile-fmts")
    task_id = _wait_for_task_running(cluster, job)

    cpu_formats = [
        ("flamegraph", cluster_pb2.CpuProfile.FLAMEGRAPH),
        ("speedscope", cluster_pb2.CpuProfile.SPEEDSCOPE),
        ("raw", cluster_pb2.CpuProfile.RAW),
    ]
    for name, fmt in cpu_formats:
        request = cluster_pb2.ProfileTaskRequest(
            task_id=task_id,
            duration_seconds=1,
            profile_type=cluster_pb2.ProfileType(cpu=cluster_pb2.CpuProfile(format=fmt)),
        )
        response = cluster.controller_client.profile_task(request, timeout_ms=3000)
        assert len(response.profile_data) > 0, f"Format {name} returned empty data"
        assert not response.error, f"Format {name} returned error: {response.error}"

    cluster.wait(job, timeout=30)


def test_profile_nonexistent_task(cluster):
    """Profiling a non-existent task returns an error."""
    request = cluster_pb2.ProfileTaskRequest(
        task_id="/nonexistent/task/0",
        duration_seconds=1,
        profile_type=cluster_pb2.ProfileType(cpu=cluster_pb2.CpuProfile(format=cluster_pb2.CpuProfile.FLAMEGRAPH)),
    )
    with pytest.raises(ConnectError):
        cluster.controller_client.profile_task(request, timeout_ms=5000)
