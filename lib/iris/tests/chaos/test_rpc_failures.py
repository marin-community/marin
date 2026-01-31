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

"""RPC failure chaos tests.

Tests that Iris handles RPC failures gracefully:
- Dispatch retries (4x with exponential backoff)
- Heartbeat timeout (60s)
- Heartbeat reconciliation (running_tasks)
"""
import pytest
from iris.chaos import enable_chaos
from iris.rpc import cluster_pb2
from .conftest import submit, wait, _quick, _slow


@pytest.mark.chaos
def test_dispatch_intermittent_failure(cluster):
    """Test intermittent heartbeat failure during dispatch (30%). Task assignments are
    buffered and retried on next heartbeat cycle. Task should eventually succeed.
    """
    _url, client = cluster
    enable_chaos("controller.heartbeat", failure_rate=0.3)
    job = submit(client, _quick, "intermittent-dispatch")
    status = wait(client, job, timeout=60)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED


@pytest.mark.chaos
def test_dispatch_permanent_failure(cluster):
    """Test permanent heartbeat failure. After 3 consecutive failures → WorkerFailedEvent →
    task marked WORKER_FAILED → rescheduled → all workers fail → job FAILED or UNSCHEDULABLE.
    """
    _url, client = cluster
    enable_chaos("controller.heartbeat", failure_rate=1.0)
    job = submit(client, _quick, "permanent-dispatch", scheduling_timeout_seconds=5)
    status = wait(client, job, timeout=20)
    assert status.state in (cluster_pb2.JOB_STATE_FAILED, cluster_pb2.JOB_STATE_UNSCHEDULABLE)


@pytest.mark.chaos
def test_heartbeat_temporary_failure(cluster):
    """Test heartbeat fails 3 times (30s gap), but worker timeout is 60s. Worker should
    NOT be marked failed. Task should still succeed.
    """
    _url, client = cluster
    enable_chaos("worker.heartbeat", failure_rate=1.0, max_failures=3)
    job = submit(client, _quick, "temp-hb-fail")
    status = wait(client, job, timeout=60)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED


@pytest.mark.chaos
def test_heartbeat_permanent_failure(cluster):
    """Test heartbeat permanently fails. After 60s, worker marked failed, tasks
    become WORKER_FAILED. With scheduling timeout, job eventually fails.
    """
    _url, client = cluster
    enable_chaos("worker.heartbeat", failure_rate=1.0)
    job = submit(client, _slow, "perm-hb-fail", scheduling_timeout_seconds=5)
    status = wait(client, job, timeout=20)
    assert status.state in (
        cluster_pb2.JOB_STATE_FAILED,
        cluster_pb2.JOB_STATE_WORKER_FAILED,
        cluster_pb2.JOB_STATE_UNSCHEDULABLE,
    )


@pytest.mark.chaos
def test_notify_task_update_failure(cluster):
    """Test notify_task_update always fails. Worker buffers completion and delivers
    it via the next heartbeat response. Controller processes it → job SUCCEEDED.
    The notify_task_update RPC is just a hint to trigger priority heartbeat.
    """
    _url, client = cluster
    enable_chaos("worker.notify_task_update", failure_rate=1.0)
    job = submit(client, _quick, "notify-fail")
    status = wait(client, job, timeout=60)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED
