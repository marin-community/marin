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
- Heartbeat reconciliation (running_task_ids)
"""
import pytest
from iris.chaos import enable_chaos
from iris.rpc import cluster_pb2
from .conftest import submit, wait, _quick, _slow


def test_dispatch_intermittent_failure(cluster):
    """Test intermittent dispatch failure (30%). Controller retries dispatch 4x with
    backoff. Task should eventually succeed.
    """
    _url, client = cluster
    enable_chaos("controller.dispatch", failure_rate=0.3)
    job = submit(client, _quick, "intermittent-dispatch")
    status = wait(client, job, timeout=60)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED


@pytest.mark.skip(reason="Iris bug: infinite retry loop, jobs stuck PENDING forever (gh#2533)")
def test_dispatch_permanent_failure(cluster):
    """Test permanent dispatch failure. All 4 retries fail → WorkerFailedEvent → task
    rescheduled to other workers → all fail → job FAILED.
    """
    _url, client = cluster
    enable_chaos("controller.dispatch", failure_rate=1.0)
    job = submit(client, _quick, "permanent-dispatch")
    status = wait(client, job, timeout=120)
    assert status.state == cluster_pb2.JOB_STATE_FAILED


def test_heartbeat_temporary_failure(cluster):
    """Test heartbeat fails 3 times (30s gap), but worker timeout is 60s. Worker should
    NOT be marked failed. Task should still succeed.
    """
    _url, client = cluster
    enable_chaos("worker.heartbeat", failure_rate=1.0, max_failures=3)
    job = submit(client, _quick, "temp-hb-fail")
    status = wait(client, job, timeout=60)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED


@pytest.mark.skip(reason="Iris bug: infinite retry loop, jobs stuck PENDING forever (gh#2533)")
def test_heartbeat_permanent_failure(cluster):
    """Test heartbeat permanently fails. After 60s, worker marked failed, tasks
    become WORKER_FAILED.
    """
    _url, client = cluster
    enable_chaos("worker.heartbeat", failure_rate=1.0)
    job = submit(client, _slow, "perm-hb-fail")
    status = wait(client, job, timeout=90)
    assert status.state in (cluster_pb2.JOB_STATE_FAILED, cluster_pb2.JOB_STATE_WORKER_FAILED)


@pytest.mark.skip(reason="Iris bug: reconciliation misidentifies completed tasks as worker failures (gh#2534)")
def test_report_task_state_failure(cluster):
    """Test report_task_state always fails. Controller should detect task completion
    via heartbeat reconciliation (running_task_ids goes empty when task finishes).
    """
    _url, client = cluster
    enable_chaos("worker.report_task_state", failure_rate=1.0)
    job = submit(client, _quick, "report-fail")
    status = wait(client, job, timeout=60)
    assert status.state in (cluster_pb2.JOB_STATE_SUCCEEDED, cluster_pb2.JOB_STATE_FAILED)
