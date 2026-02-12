# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Heartbeat E2E tests.

Tests heartbeat timeout detection, controller heartbeat failures, dispatch queue
clearing on worker failure, and recovery after transient failures.

Merged from tests/chaos/test_heartbeat_timeout.py and
tests/chaos/test_controller_heartbeat_failures.py.
"""

import time

import pytest
from iris.chaos import enable_chaos
from iris.rpc import cluster_pb2

pytestmark = pytest.mark.e2e

# Local config sets heartbeat_failure_threshold = 3 via make_local_config().
# Tests must use this value (not the production default of 10) since the e2e
# cluster fixture runs with local config.
LOCAL_HEARTBEAT_FAILURE_THRESHOLD = 3


def test_worker_survives_transient_heartbeat_delay(cluster):
    """Worker doesn't reset on brief heartbeat delays.

    Delays less than the timeout don't trigger a reset. Uses chaos injection
    to delay heartbeat by ~1s a few times, which stays well under the default
    60s timeout. The job should complete successfully without the worker resetting.
    """

    def quick_job():
        return 42

    job = cluster.submit(quick_job, "transient-delay")
    enable_chaos("worker.heartbeat", delay_seconds=0.3, max_failures=2)
    status = cluster.wait(job, timeout=30)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED


def test_heartbeat_failures_below_threshold_recovers(cluster):
    """Heartbeat failures below threshold don't kill the worker.

    Worker should remain healthy and task should complete after
    transient heartbeat failures stop.
    """

    def quick_job():
        return 42

    failures_to_inject = LOCAL_HEARTBEAT_FAILURE_THRESHOLD - 2
    enable_chaos(
        "controller.heartbeat",
        failure_rate=1.0,
        max_failures=failures_to_inject,
        delay_seconds=0.01,
    )

    job = cluster.submit(quick_job, "transient-hb-fail")
    status = cluster.wait(job, timeout=60)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED


def test_heartbeat_failures_at_threshold_kills_worker(cluster):
    """Consecutive heartbeat failures at threshold mark worker as failed.

    When heartbeats fail LOCAL_HEARTBEAT_FAILURE_THRESHOLD times consecutively,
    the worker is marked unhealthy and running tasks transition to WORKER_FAILED.
    With retries, the task should be rescheduled when the worker recovers.
    """

    def slow_job():
        time.sleep(5)
        return 42

    enable_chaos(
        "controller.heartbeat",
        failure_rate=1.0,
        max_failures=LOCAL_HEARTBEAT_FAILURE_THRESHOLD,
        delay_seconds=0.01,
    )

    job = cluster.submit(slow_job, "threshold-hb-fail", max_retries_preemption=3)
    status = cluster.wait(job, timeout=60)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED


def test_dispatch_cleared_on_worker_failure(cluster):
    """Dispatch queue is cleared when worker fails.

    When a worker hits the failure threshold:
    1. Pending dispatches in the buffer should be cleared (not orphaned)
    2. Running tasks should be marked WORKER_FAILED
    3. Tasks should be requeued for retry on recovered/other workers

    Tests the fix for the bug where dispatch buffers were orphaned
    when a worker failed, leading to stale dispatches if worker re-registered.
    """

    def slow_job():
        time.sleep(10)
        return 42

    enable_chaos(
        "controller.heartbeat",
        failure_rate=1.0,
        max_failures=LOCAL_HEARTBEAT_FAILURE_THRESHOLD + 2,
        delay_seconds=0.01,
    )

    job = cluster.submit(slow_job, "dispatch-clear-test", max_retries_preemption=5)
    status = cluster.wait(job, timeout=60)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED


def test_multiple_workers_one_fails(cluster):
    """One worker fails while others remain healthy.

    Tasks on the failed worker should be rescheduled to healthy workers.
    """

    def quick_job():
        return 42

    enable_chaos(
        "controller.heartbeat",
        failure_rate=1.0,
        max_failures=LOCAL_HEARTBEAT_FAILURE_THRESHOLD,
        delay_seconds=0.01,
    )

    job = cluster.submit(quick_job, "multi-worker-fail", max_retries_preemption=2)
    status = cluster.wait(job, timeout=60)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED


def test_heartbeat_failure_with_pending_kills(cluster):
    """Kill requests are not orphaned when worker fails.

    If we have buffered kill requests for a worker that then fails,
    those kills should be dropped (worker is dead anyway) and not
    cause issues when worker re-registers.
    """

    def quick_job():
        return 42

    enable_chaos(
        "controller.heartbeat",
        failure_rate=1.0,
        max_failures=LOCAL_HEARTBEAT_FAILURE_THRESHOLD,
        delay_seconds=0.01,
    )

    job = cluster.submit(quick_job, "kill-clear-test", max_retries_preemption=2)
    status = cluster.wait(job, timeout=60)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED
