# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Consolidated chaos / failure-injection E2E tests.

Every test uses the function-scoped ``cluster`` or ``multi_worker_cluster``
fixture (from conftest.py) so each test gets a fresh cluster with no chaos
bleed.  The autouse ``_reset_chaos`` fixture in conftest resets chaos state
after every test.

Merged from:
  - test_task_lifecycle.py
  - test_worker_failures.py
  - test_rpc_failures.py
  - test_heartbeat.py
  - test_snapshot.py
  - test_high_concurrency.py
"""

import time

import pytest
from iris.chaos import enable_chaos, reset_chaos
from iris.cluster.constraints import WellKnownAttribute
from iris.cluster.types import CoschedulingConfig
from iris.rpc import cluster_pb2
from iris.test_util import SentinelFile
from iris.time_utils import Duration

from .helpers import _block, _quick, _slow

pytestmark = [pytest.mark.e2e, pytest.mark.timeout(60)]


# ---------------------------------------------------------------------------
# Module-level job callables for Entrypoint.from_callable serialization.
# These must be top-level functions, not closures or lambdas.
# ---------------------------------------------------------------------------


def _quick_42():
    return 42


def _slow_2s():
    time.sleep(2)
    return 42


def _slow_3s():
    time.sleep(3)
    return 42


def _medium_5s():
    time.sleep(5)
    return 42


def _waiting_task(s):
    from iris.time_utils import Duration

    s.wait(timeout=Duration.from_seconds(2))
    return "done"


# ---------------------------------------------------------------------------
# Task lifecycle & scheduling (from test_task_lifecycle.py)
# ---------------------------------------------------------------------------


def test_bundle_download_intermittent(cluster):
    """Bundle download fails intermittently, task retries handle it."""
    enable_chaos(
        "worker.bundle_download", failure_rate=0.5, max_failures=2, error=RuntimeError("chaos: download failed")
    )
    job = cluster.submit(_quick, "bundle-fail", max_retries_failure=3)
    status = cluster.wait(job, timeout=30)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED


def test_task_timeout(cluster, sentinel):
    """Task times out, marked FAILED."""
    job = cluster.submit(_block, "timeout-test", sentinel, timeout=Duration.from_seconds(2))
    status = cluster.wait(job, timeout=15)
    assert status.state == cluster_pb2.JOB_STATE_FAILED


def test_coscheduled_sibling_failure(cluster):
    """Coscheduled job: one replica fails -> all siblings killed."""
    enable_chaos("worker.create_container", failure_rate=1.0, max_failures=1, error=RuntimeError("chaos: replica fail"))
    job = cluster.submit(
        _quick,
        "cosched-fail",
        coscheduling=CoschedulingConfig(group_by=WellKnownAttribute.TPU_NAME),
        replicas=2,
        scheduling_timeout=Duration.from_seconds(5),
    )
    status = cluster.wait(job, timeout=30)
    assert status.state in (cluster_pb2.JOB_STATE_FAILED, cluster_pb2.JOB_STATE_UNSCHEDULABLE)


def test_retry_budget_exact(cluster):
    """Task fails exactly N-1 times, succeeds on last attempt."""
    enable_chaos("worker.create_container", failure_rate=1.0, max_failures=2, error=RuntimeError("chaos: transient"))
    job = cluster.submit(_quick, "exact-retry", max_retries_failure=2)
    status = cluster.wait(job, timeout=30)
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
    enable_chaos("controller.heartbeat", delay_seconds=1.0, failure_rate=1.0, max_failures=2)
    job = cluster.submit(_quick, "delayed-dispatch")
    status = cluster.wait(job, timeout=30)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED


# ---------------------------------------------------------------------------
# Worker failures (from test_worker_failures.py)
# ---------------------------------------------------------------------------


def test_worker_crash_mid_task(cluster):
    """Worker task monitor crashes mid-task."""
    enable_chaos("worker.task_monitor", failure_rate=1.0)
    job = cluster.submit(_quick, "crash-mid-task")
    status = cluster.wait(job, timeout=30)
    assert status.state == cluster_pb2.JOB_STATE_FAILED


def test_worker_delayed_registration(cluster):
    """Worker registration delayed by 2s."""
    enable_chaos("worker.register", delay_seconds=2.0, max_failures=1)
    job = cluster.submit(_quick, "delayed-reg")
    status = cluster.wait(job, timeout=30)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED


def test_task_fails_once_then_succeeds(cluster):
    """Container creation fails once, succeeds on retry."""
    enable_chaos(
        "worker.create_container",
        failure_rate=1.0,
        max_failures=1,
        error=RuntimeError("chaos: transient container failure"),
    )
    job = cluster.submit(_quick, "retry-once", max_retries_failure=2)
    status = cluster.wait(job, timeout=30)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED


def test_worker_sequential_jobs(cluster):
    """Sequential jobs verify reconciliation works across job boundaries."""
    for i in range(3):
        job = cluster.submit(_quick, f"seq-{i}")
        status = cluster.wait(job, timeout=30)
        assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED


@pytest.mark.timeout(15)
def test_all_workers_fail(cluster):
    """All workers' registration fails permanently."""
    enable_chaos("worker.register", failure_rate=1.0, error=RuntimeError("chaos: registration failed"))
    job = cluster.submit(_slow, "all-workers-fail", cpu=9999, scheduling_timeout=Duration.from_seconds(3))
    status = cluster.wait(job, timeout=10)
    assert status.state in (cluster_pb2.JOB_STATE_FAILED, cluster_pb2.JOB_STATE_UNSCHEDULABLE)


# ---------------------------------------------------------------------------
# RPC failures (from test_rpc_failures.py)
# ---------------------------------------------------------------------------


def test_dispatch_intermittent_failure(cluster):
    """Intermittent heartbeat failure during dispatch (30%)."""
    cluster.wait_for_workers(1, timeout=15)
    enable_chaos("controller.heartbeat", failure_rate=0.3)
    job = cluster.submit(_quick, "intermittent-dispatch")
    status = cluster.wait(job, timeout=30)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED


def test_dispatch_permanent_failure(cluster):
    """Permanent heartbeat failure leads to worker failure."""
    cluster.wait_for_workers(1, timeout=15)
    enable_chaos("controller.heartbeat", failure_rate=1.0)
    job = cluster.submit(_quick, "permanent-dispatch", scheduling_timeout=Duration.from_seconds(2))
    status = cluster.wait(job, timeout=10)
    assert status.state in (cluster_pb2.JOB_STATE_FAILED, cluster_pb2.JOB_STATE_UNSCHEDULABLE)


def test_heartbeat_temporary_failure(cluster):
    """Heartbeat fails 3 times, stays under threshold."""
    cluster.wait_for_workers(1, timeout=15)
    enable_chaos("worker.heartbeat", failure_rate=1.0, max_failures=2)
    job = cluster.submit(_quick, "temp-hb-fail")
    status = cluster.wait(job, timeout=30)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED


def test_heartbeat_permanent_failure(cluster):
    """Heartbeat permanently fails -> worker marked failed."""
    cluster.wait_for_workers(1, timeout=15)
    enable_chaos("worker.heartbeat", failure_rate=1.0)
    job = cluster.submit(_slow, "perm-hb-fail", scheduling_timeout=Duration.from_seconds(2))
    status = cluster.wait(job, timeout=10)
    assert status.state in (
        cluster_pb2.JOB_STATE_FAILED,
        cluster_pb2.JOB_STATE_WORKER_FAILED,
        cluster_pb2.JOB_STATE_UNSCHEDULABLE,
    )


def test_notify_task_update_failure(cluster):
    """notify_task_update always fails; completion buffered via heartbeat."""
    cluster.wait_for_workers(1, timeout=15)
    enable_chaos("worker.notify_task_update", failure_rate=1.0)
    job = cluster.submit(_quick, "notify-fail")
    status = cluster.wait(job, timeout=30)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED


# ---------------------------------------------------------------------------
# Heartbeat threshold (from test_heartbeat.py)
# ---------------------------------------------------------------------------

# Local config sets heartbeat_failure_threshold = 3 via make_local_config().
LOCAL_HEARTBEAT_FAILURE_THRESHOLD = 3


def test_heartbeat_survives_transient_delay(cluster):
    """Brief heartbeat delays don't trigger reset."""
    job = cluster.submit(_quick_42, "transient-delay")
    enable_chaos("worker.heartbeat", delay_seconds=0.3, max_failures=2)
    status = cluster.wait(job, timeout=30)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED


def test_heartbeat_below_threshold_recovers(cluster):
    """Heartbeat failures below threshold don't kill the worker."""
    failures_to_inject = LOCAL_HEARTBEAT_FAILURE_THRESHOLD - 2
    enable_chaos(
        "controller.heartbeat",
        failure_rate=1.0,
        max_failures=failures_to_inject,
        delay_seconds=0.01,
    )
    job = cluster.submit(_quick_42, "transient-hb-fail")
    status = cluster.wait(job, timeout=30)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED


def test_heartbeat_at_threshold_kills_worker(cluster):
    """Consecutive failures at threshold mark worker failed, task retried."""
    enable_chaos(
        "controller.heartbeat",
        failure_rate=1.0,
        max_failures=LOCAL_HEARTBEAT_FAILURE_THRESHOLD,
        delay_seconds=0.01,
    )
    job = cluster.submit(_slow_2s, "threshold-hb-fail", max_retries_preemption=10)
    status = cluster.wait(job, timeout=30)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED


def test_dispatch_cleared_on_worker_failure(cluster):
    """Dispatch queue cleared when worker hits failure threshold."""
    enable_chaos(
        "controller.heartbeat",
        failure_rate=1.0,
        max_failures=LOCAL_HEARTBEAT_FAILURE_THRESHOLD + 2,
        delay_seconds=0.01,
    )
    job = cluster.submit(_slow_3s, "dispatch-clear-test", max_retries_preemption=10)
    status = cluster.wait(job, timeout=30)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED


def test_multiple_workers_one_fails(cluster):
    """One worker fails while others remain healthy; task rescheduled."""
    enable_chaos(
        "controller.heartbeat",
        failure_rate=1.0,
        max_failures=LOCAL_HEARTBEAT_FAILURE_THRESHOLD,
        delay_seconds=0.01,
    )
    job = cluster.submit(_quick_42, "multi-worker-fail", max_retries_preemption=10)
    status = cluster.wait(job, timeout=30)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED


def test_heartbeat_failure_with_pending_kills(cluster):
    """Kill requests not orphaned when worker fails."""
    enable_chaos(
        "controller.heartbeat",
        failure_rate=1.0,
        max_failures=LOCAL_HEARTBEAT_FAILURE_THRESHOLD,
        delay_seconds=0.01,
    )
    job = cluster.submit(_quick_42, "kill-clear-test", max_retries_preemption=10)
    status = cluster.wait(job, timeout=30)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED


# ---------------------------------------------------------------------------
# Snapshot / checkpoint (from test_snapshot.py)
# ---------------------------------------------------------------------------


def test_checkpoint_returns_metadata(cluster):
    """BeginCheckpoint RPC returns valid snapshot path and counts."""
    job = cluster.submit(_quick, "pre-checkpoint")
    cluster.wait(job, timeout=30)
    resp = cluster.controller_client.begin_checkpoint(cluster_pb2.Controller.BeginCheckpointRequest())
    assert resp.snapshot_path
    assert resp.created_at.epoch_ms > 0
    assert resp.job_count >= 1


def test_checkpoint_with_worker_death(cluster):
    """Worker dies after checkpoint; task retried via heartbeat failure."""
    job = cluster.submit(_medium_5s, "worker-death-retry", max_retries_preemption=10)
    cluster.wait_for_state(job, cluster_pb2.JOB_STATE_RUNNING, timeout=15)

    ckpt_resp = cluster.controller_client.begin_checkpoint(cluster_pb2.Controller.BeginCheckpointRequest())
    assert ckpt_resp.job_count >= 1

    enable_chaos("controller.heartbeat", failure_rate=1.0, max_failures=4, delay_seconds=0.01)
    status = cluster.wait(job, timeout=45)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED


# ---------------------------------------------------------------------------
# High concurrency (from test_high_concurrency.py)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_128_tasks_concurrent_scheduling(multi_worker_cluster, sentinel):
    """128 simultaneous tasks expose heartbeat iteration race conditions."""
    enable_chaos("controller.heartbeat.iteration", delay_seconds=0.01)

    try:
        job = multi_worker_cluster.submit(
            _waiting_task,
            "race-test",
            sentinel,
            cpu=0,
            replicas=128,
        )
        time.sleep(1.0)
        sentinel.signal()
        status = multi_worker_cluster.wait(job, timeout=60)
        assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED, f"Job failed: {status}"
    finally:
        reset_chaos()
