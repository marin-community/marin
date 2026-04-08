# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for task attempt log preservation and routing.

These tests need real job execution to verify log content from actual
callable output and chaos injection behavior.
"""

import uuid

import pytest
from iris.chaos import enable_chaos
from iris.rpc import job_pb2

pytestmark = [pytest.mark.e2e, pytest.mark.timeout(60)]


def _fail_then_succeed(attempt_marker: str):
    """Job function that fails on attempt 0, succeeds on attempt 1+.

    Uses the attempt_id from JobInfo to determine whether to fail.
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


def test_multiple_attempts_preserve_logs(cluster, caplog):
    """Job with retries preserves logs from all attempts.

    1. Submit a job that fails on attempt 0 but succeeds on attempt 1
    2. Verify final state is SUCCEEDED
    3. Verify logs contain output from both attempts (via attempt_id field)
    """
    import logging

    run_id = uuid.uuid4().hex[:8]
    marker = f"test-{run_id}"

    with caplog.at_level(logging.INFO, logger="iris"):
        job = cluster.submit(
            _fail_then_succeed,
            f"retry-logs-{run_id}",
            marker,
            max_retries_failure=1,
        )

        status = cluster.wait(job, timeout=60)
        assert status.state == job_pb2.JOB_STATE_SUCCEEDED, f"Job should succeed after retry: {status}"

        # Fetch logs for all attempts
        task_id = job.job_id.task(0)
        logs_response = cluster.client.fetch_task_logs(task_id)

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


def test_superseding_attempt_logs_info(cluster):
    """Verify job succeeds after chaos injection fails container creation on first attempt.

    Uses chaos injection to fail container creation once, forcing retry.
    The second attempt should succeed normally.
    """
    run_id = uuid.uuid4().hex[:8]

    enable_chaos(
        "worker.create_container",
        failure_rate=1.0,
        max_failures=1,
        error=RuntimeError("chaos: container creation failed"),
    )

    job = cluster.submit(
        lambda: "ok",
        f"supersede-{run_id}",
        max_retries_failure=2,
    )

    status = cluster.wait(job, timeout=60)
    assert status.state == job_pb2.JOB_STATE_SUCCEEDED, f"Job should succeed after retry: {status}"


def test_attempt_specific_log_fetch(cluster):
    """Verify fetching logs filtered by specific attempt_id works.

    1. Submit a job that fails then succeeds
    2. Fetch logs for attempt_id=0 specifically
    3. Verify only attempt 0 logs are returned with the failure message
    4. Fetch logs for attempt_id=1 and verify they're from attempt 1
    """
    run_id = uuid.uuid4().hex[:8]
    marker = f"specific-{run_id}"

    job = cluster.submit(
        _fail_then_succeed,
        f"attempt-filter-{run_id}",
        marker,
        max_retries_failure=1,
    )

    status = cluster.wait(job, timeout=60)
    assert status.state == job_pb2.JOB_STATE_SUCCEEDED

    # Fetch logs for attempt 0 only
    task_id = job.job_id.task(0)
    logs_attempt_0 = cluster.client.fetch_task_logs(task_id, attempt_id=0)

    for entry in logs_attempt_0:
        assert entry.attempt_id == 0, f"Expected attempt_id=0, got {entry.attempt_id}"

    found_failure = any("Intentional failure on attempt 0" in e.data for e in logs_attempt_0)
    assert found_failure, "Should find failure message in attempt 0 logs"

    # Fetch logs for attempt 1 only
    logs_attempt_1 = cluster.client.fetch_task_logs(task_id, attempt_id=1)

    for entry in logs_attempt_1:
        assert entry.attempt_id == 1, f"Expected attempt_id=1, got {entry.attempt_id}"
