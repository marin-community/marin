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

"""Worker failure chaos tests.

Tests worker crashes, delayed registration, stale state, and task-level retries.
All chaos is injected inline in worker.py.
"""

import pytest
from iris.chaos import enable_chaos
from iris.rpc import cluster_pb2
from iris.time_utils import Duration
from .conftest import submit, wait, _quick, _slow


@pytest.mark.chaos
def test_worker_crash_mid_task(cluster):
    """Worker task monitor crashes mid-task. Task fails, controller detects
    via heartbeat reconciliation or report_task_state."""
    _url, client = cluster
    # task_monitor chaos kills the monitoring loop — task fails with error
    enable_chaos("worker.task_monitor", failure_rate=1.0)
    job = submit(client, _quick, "crash-mid-task")
    status = wait(client, job, timeout=60)
    assert status.state == cluster_pb2.JOB_STATE_FAILED


@pytest.mark.chaos
def test_worker_delayed_registration(cluster):
    """Worker registration delayed by 5s on first attempt. Task pends, then
    schedules once registration completes."""
    _url, client = cluster
    enable_chaos("worker.register", delay_seconds=5.0, max_failures=1)
    job = submit(client, _quick, "delayed-reg")
    status = wait(client, job, timeout=60)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED


@pytest.mark.chaos
def test_worker_sequential_jobs(cluster):
    """Sequential jobs verify reconciliation works across job boundaries.
    Worker state is consistent between tasks."""
    _url, client = cluster
    for i in range(3):
        job = submit(client, _quick, f"seq-{i}")
        status = wait(client, job, timeout=30)
        assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED


@pytest.mark.chaos
@pytest.mark.timeout(60)
def test_all_workers_fail(cluster):
    """All workers' registration fails permanently. With scheduling timeout,
    job transitions to FAILED/UNSCHEDULABLE when no workers register.
    """
    _url, client = cluster
    enable_chaos("worker.register", failure_rate=1.0, error=RuntimeError("chaos: registration failed"))
    job = submit(client, _slow, "all-workers-fail", scheduling_timeout=Duration.from_seconds(15))
    status = wait(client, job, timeout=30)
    assert status.state in (cluster_pb2.JOB_STATE_FAILED, cluster_pb2.JOB_STATE_UNSCHEDULABLE)


@pytest.mark.chaos
def test_task_fails_once_then_succeeds(cluster):
    """Container creation fails once, succeeds on retry."""
    _url, client = cluster
    enable_chaos(
        "worker.create_container",
        failure_rate=1.0,
        max_failures=1,
        error=RuntimeError("chaos: transient container failure"),
    )
    job = submit(client, _quick, "retry-once", max_retries_failure=2)
    status = wait(client, job, timeout=60)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED
