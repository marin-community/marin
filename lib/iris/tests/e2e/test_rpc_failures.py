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
    enable_chaos("controller.heartbeat", failure_rate=0.3)
    job = cluster.submit(_quick, "intermittent-dispatch")
    status = cluster.wait(job, timeout=60)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED


def test_dispatch_permanent_failure(cluster):
    """Test permanent heartbeat failure. After 3 consecutive failures -> WorkerFailedEvent ->
    task marked WORKER_FAILED -> rescheduled -> all workers fail -> job FAILED or UNSCHEDULABLE.
    """
    enable_chaos("controller.heartbeat", failure_rate=1.0)
    job = cluster.submit(_quick, "permanent-dispatch", scheduling_timeout=Duration.from_seconds(5))
    status = cluster.wait(job, timeout=20)
    assert status.state in (cluster_pb2.JOB_STATE_FAILED, cluster_pb2.JOB_STATE_UNSCHEDULABLE)


def test_heartbeat_temporary_failure(cluster):
    """Test heartbeat fails 3 times (30s gap), but worker timeout is 60s. Worker should
    NOT be marked failed. Task should still succeed.
    """
    enable_chaos("worker.heartbeat", failure_rate=1.0, max_failures=3)
    job = cluster.submit(_quick, "temp-hb-fail")
    status = cluster.wait(job, timeout=60)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED


def test_heartbeat_permanent_failure(cluster):
    """Test heartbeat permanently fails. After 60s, worker marked failed, tasks
    become WORKER_FAILED. With scheduling timeout, job eventually fails.
    """
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
    enable_chaos("worker.notify_task_update", failure_rate=1.0)
    job = cluster.submit(_quick, "notify-fail")
    status = cluster.wait(job, timeout=60)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED
