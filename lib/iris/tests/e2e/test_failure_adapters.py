# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Live adapters for failure-injection and concurrency boundaries."""

import pytest
from iris.chaos import enable_chaos, reset_chaos
from iris.rpc import controller_pb2, job_pb2
from rigging.filesystem import StoragePath
from rigging.timing import Duration, ExponentialBackoff

from .helpers import TestJobs

pytestmark = [pytest.mark.requires_cluster, pytest.mark.timeout(60)]


def test_bundle_download_adapter_retries_two_failures(cluster):
    enable_chaos(
        "worker.bundle_download",
        failure_rate=1.0,
        max_failures=2,
        error=RuntimeError("chaos: download failed"),
    )
    job = cluster.submit(TestJobs.quick, "bundle-fail", max_retries_failure=3, max_task_failures=3)

    status = cluster.wait(job, timeout=30)
    task = cluster.task_status(job)

    assert status.state == job_pb2.JOB_STATE_SUCCEEDED
    assert [attempt.state for attempt in task.attempts] == [
        job_pb2.TASK_STATE_FAILED,
        job_pb2.TASK_STATE_FAILED,
        job_pb2.TASK_STATE_SUCCEEDED,
    ]


@pytest.mark.timeout(120)
def test_task_process_timeout_is_reported_at_the_controller(cluster, sentinel):
    job = cluster.submit(TestJobs.block, "timeout-test", sentinel, timeout=Duration.from_seconds(2))

    # Execution timeouts are finalized by a controller sweep that runs at most
    # once a minute, so the 2s deadline is enforced with up to ~60s of lag.
    status = cluster.wait(job, timeout=90)
    task = cluster.task_status(job)

    assert status.state == job_pb2.JOB_STATE_FAILED
    assert task.state == job_pb2.TASK_STATE_FAILED
    assert task.error == "Execution timeout exceeded"


def test_worker_task_monitor_crash_is_reported_as_task_failure(cluster):
    enable_chaos("worker.task_monitor", failure_rate=1.0)
    job = cluster.submit(TestJobs.quick, "crash-mid-task")

    status = cluster.wait(job, timeout=30)
    task = cluster.task_status(job)

    assert status.state == job_pb2.JOB_STATE_FAILED
    assert task.state == job_pb2.TASK_STATE_FAILED
    assert task.error == "chaos: monitor crashed"


def test_checkpoint_rpc_publishes_filesystem_snapshot_and_metadata(cluster):
    job = cluster.submit(TestJobs.quick, "pre-checkpoint")
    cluster.wait(job, timeout=30)

    response = cluster.controller_client.begin_checkpoint(controller_pb2.Controller.BeginCheckpointRequest())
    checkpoint_path = StoragePath(response.checkpoint_path)

    assert response.created_at.epoch_ms > 0
    assert response.job_count >= 1
    assert response.task_count >= 1
    assert checkpoint_path.name == str(response.created_at.epoch_ms)
    assert (checkpoint_path / "controller.sqlite3.zst").isfile()


@pytest.mark.slow
def test_threaded_scheduler_completes_128_live_tasks(multi_worker_cluster, sentinel):
    enable_chaos("controller.reconcile", delay_seconds=0.01)

    try:
        job = multi_worker_cluster.submit(
            TestJobs.wait_for_sentinel,
            "race-test",
            sentinel,
            cpu=0,
            replicas=128,
        )
        ExponentialBackoff(initial=0.05, maximum=0.5).wait_until_or_raise(
            lambda: multi_worker_cluster.task_status(job, 127).state == job_pb2.TASK_STATE_RUNNING,
            timeout=Duration.from_seconds(15),
            error_message="task 127 did not reach the live worker boundary",
        )
        sentinel.signal()

        status = multi_worker_cluster.wait(job, timeout=45)
        task_states = [multi_worker_cluster.task_status(job, index).state for index in range(128)]

        assert status.state == job_pb2.JOB_STATE_SUCCEEDED, f"Job failed: {status}"
        assert task_states == [job_pb2.TASK_STATE_SUCCEEDED] * 128
    finally:
        reset_chaos()
