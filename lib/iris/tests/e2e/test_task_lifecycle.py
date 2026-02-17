# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Task lifecycle and scheduling E2E tests.

Tests task-level failure modes, timeouts, capacity, and scheduling behavior.
"""

import time

import pytest
from iris.chaos import enable_chaos
from iris.cluster.types import CoschedulingConfig
from iris.rpc import cluster_pb2
from iris.test_util import SentinelFile
from iris.time_utils import Duration

from .helpers import _block, _quick

pytestmark = pytest.mark.e2e


def test_bundle_download_intermittent(cluster):
    """Bundle download fails intermittently, task retries handle it."""
    enable_chaos(
        "worker.bundle_download", failure_rate=0.5, max_failures=2, error=RuntimeError("chaos: download failed")
    )
    job = cluster.submit(_quick, "bundle-fail", max_retries_failure=3)
    status = cluster.wait(job, timeout=60)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED


def test_task_timeout(cluster, sentinel):
    """Task times out, marked FAILED."""
    job = cluster.submit(_block, "timeout-test", sentinel, timeout=Duration.from_seconds(5))
    status = cluster.wait(job, timeout=30)
    assert status.state == cluster_pb2.JOB_STATE_FAILED


def test_coscheduled_sibling_failure(cluster):
    """Coscheduled job: one replica fails -> all siblings killed.

    Uses group_by="tpu-name" which is the production pattern. All local workers
    from the same slice share the same tpu-name attribute value.
    """
    enable_chaos("worker.create_container", failure_rate=1.0, max_failures=1, error=RuntimeError("chaos: replica fail"))
    job = cluster.submit(
        _quick,
        "cosched-fail",
        coscheduling=CoschedulingConfig(group_by="tpu-name"),
        replicas=2,
        scheduling_timeout=Duration.from_seconds(30),
    )
    status = cluster.wait(job, timeout=60)
    assert status.state in (cluster_pb2.JOB_STATE_FAILED, cluster_pb2.JOB_STATE_UNSCHEDULABLE)


def test_retry_budget_exact(cluster):
    """Task fails exactly N-1 times, succeeds on last attempt."""
    enable_chaos("worker.create_container", failure_rate=1.0, max_failures=2, error=RuntimeError("chaos: transient"))
    job = cluster.submit(_quick, "exact-retry", max_retries_failure=2)
    status = cluster.wait(job, timeout=60)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED


def test_capacity_wait(cluster, tmp_path):
    """Workers at capacity, task pends, schedules when capacity frees."""
    blocker_sentinels = []
    blockers = []
    for i in range(2):
        s = SentinelFile(str(tmp_path / f"blocker-{i}"))
        blocker_sentinels.append(s)
        job = cluster.submit(_block, f"blocker-{i}", s, cpu=4)
        blockers.append(job)

    time.sleep(1)

    pending = cluster.submit(_quick, "pending")
    status = cluster.status(pending)
    assert status.state in (
        cluster_pb2.JOB_STATE_PENDING,
        cluster_pb2.JOB_STATE_RUNNING,
        cluster_pb2.JOB_STATE_SUCCEEDED,
    )

    for s in blocker_sentinels:
        s.signal()
    for b in blockers:
        cluster.wait(b, timeout=30)
    status = cluster.wait(pending, timeout=30)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED


def test_scheduling_timeout(cluster):
    """Scheduling timeout exceeded -> UNSCHEDULABLE."""
    job = cluster.submit(
        _quick,
        "unsched",
        cpu=9999,
        scheduling_timeout=Duration.from_seconds(2),
    )
    status = cluster.wait(job, timeout=10)
    assert status.state in (cluster_pb2.JOB_STATE_FAILED, cluster_pb2.JOB_STATE_UNSCHEDULABLE)


def test_dispatch_delayed(cluster):
    """Dispatch delayed by chaos (via heartbeat), but eventually goes through."""
    enable_chaos("controller.heartbeat", delay_seconds=3.0, failure_rate=1.0, max_failures=2)
    job = cluster.submit(_quick, "delayed-dispatch")
    status = cluster.wait(job, timeout=60)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED
