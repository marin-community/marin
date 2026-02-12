# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""RPC failure E2E tests.

Tests that Iris handles RPC failures gracefully:
- Dispatch retries (4x with exponential backoff)
- Heartbeat timeout (60s)
- Heartbeat reconciliation (running_tasks)
"""

import pytest
from iris.chaos import enable_chaos
from iris.rpc import cluster_pb2
from iris.time_utils import Duration

from .helpers import _quick, _slow

pytestmark = pytest.mark.e2e


def test_dispatch_intermittent_failure(cluster):
    """Test intermittent heartbeat failure during dispatch (30%). Task assignments are
    buffered and retried on next heartbeat cycle. Task should eventually succeed.
    """
    cluster.wait_for_workers(1, timeout=15)
    enable_chaos("controller.heartbeat", failure_rate=0.3)
    job = cluster.submit(_quick, "intermittent-dispatch")
    status = cluster.wait(job, timeout=30)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED


def test_dispatch_permanent_failure(cluster):
    """Test permanent heartbeat failure. After consecutive failures exceed the threshold ->
    WorkerFailedEvent -> task marked WORKER_FAILED -> rescheduled -> all workers fail -> job FAILED or UNSCHEDULABLE.
    """
    cluster.wait_for_workers(1, timeout=15)
    enable_chaos("controller.heartbeat", failure_rate=1.0)
    job = cluster.submit(_quick, "permanent-dispatch", scheduling_timeout=Duration.from_seconds(5))
    status = cluster.wait(job, timeout=20)
    assert status.state in (cluster_pb2.JOB_STATE_FAILED, cluster_pb2.JOB_STATE_UNSCHEDULABLE)


def test_heartbeat_temporary_failure(cluster):
    """Test heartbeat fails 3 times, but stays under the heartbeat_failure_threshold.
    Worker should NOT be marked failed. Task should still succeed.
    """
    cluster.wait_for_workers(1, timeout=15)
    enable_chaos("worker.heartbeat", failure_rate=1.0, max_failures=2)
    job = cluster.submit(_quick, "temp-hb-fail")
    status = cluster.wait(job, timeout=30)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED


def test_heartbeat_permanent_failure(cluster):
    """Test heartbeat permanently fails. After exceeding the heartbeat failure threshold,
    worker marked failed, tasks become WORKER_FAILED. With scheduling timeout, job eventually fails.
    """
    cluster.wait_for_workers(1, timeout=15)
    enable_chaos("worker.heartbeat", failure_rate=1.0)
    job = cluster.submit(_slow, "perm-hb-fail", scheduling_timeout=Duration.from_seconds(5))
    status = cluster.wait(job, timeout=20)
    assert status.state in (
        cluster_pb2.JOB_STATE_FAILED,
        cluster_pb2.JOB_STATE_WORKER_FAILED,
        cluster_pb2.JOB_STATE_UNSCHEDULABLE,
    )


def test_notify_task_update_failure(cluster):
    """Test notify_task_update always fails. Worker buffers completion and delivers
    it via the next heartbeat response. Controller processes it -> job SUCCEEDED.
    The notify_task_update RPC is just a hint to trigger priority heartbeat.
    """
    cluster.wait_for_workers(1, timeout=15)
    enable_chaos("worker.notify_task_update", failure_rate=1.0)
    job = cluster.submit(_quick, "notify-fail")
    status = cluster.wait(job, timeout=30)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED
