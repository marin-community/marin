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

"""Task lifecycle and scheduling chaos tests.

Tests task-level failure modes, timeouts, capacity, and scheduling behavior.
"""
import time
import pytest

from iris.chaos import enable_chaos
from iris.rpc import cluster_pb2
from iris.cluster.types import ResourceSpec, CoschedulingConfig
from iris.test_util import SentinelFile

from .conftest import submit, wait, _quick, _block


@pytest.mark.chaos
def test_bundle_download_intermittent(cluster):
    """Bundle download fails intermittently, task retries handle it."""
    _url, client = cluster
    enable_chaos(
        "worker.bundle_download", failure_rate=0.5, max_failures=2, error=RuntimeError("chaos: download failed")
    )
    job = submit(client, _quick, "bundle-fail", max_retries_failure=3)
    status = wait(client, job, timeout=60)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED


@pytest.mark.chaos
def test_task_timeout(cluster, sentinel):
    """Task times out, marked FAILED."""
    _url, client = cluster
    job = submit(client, _block, "timeout-test", sentinel, timeout_seconds=5)
    status = wait(client, job, timeout=30)
    assert status.state == cluster_pb2.JOB_STATE_FAILED


@pytest.mark.chaos
def test_coscheduled_sibling_failure(cluster):
    """Coscheduled job: one replica fails → all siblings killed.

    Uses group_by="tpu-name" which is the production pattern. All local workers
    from the same slice share the same tpu-name attribute value.
    """
    _url, client = cluster
    # Fail container creation once — hits one replica, sibling cascade kills all.
    # failure_rate=1.0 ensures the first container creation deterministically fails.
    enable_chaos("worker.create_container", failure_rate=1.0, max_failures=1, error=RuntimeError("chaos: replica fail"))
    from iris.cluster.types import Entrypoint, EnvironmentSpec

    job = client.submit(
        entrypoint=Entrypoint.from_callable(_quick),
        name="cosched-fail",
        resources=ResourceSpec(cpu=1, memory="1g"),
        environment=EnvironmentSpec(),
        coscheduling=CoschedulingConfig(group_by="tpu-name"),
        replicas=2,
        scheduling_timeout_seconds=30,
    )
    status = wait(client, job, timeout=60)
    assert status.state in (cluster_pb2.JOB_STATE_FAILED, cluster_pb2.JOB_STATE_UNSCHEDULABLE)


@pytest.mark.chaos
def test_retry_budget_exact(cluster):
    """Task fails exactly N-1 times, succeeds on last attempt."""
    _url, client = cluster
    enable_chaos("worker.create_container", failure_rate=1.0, max_failures=2, error=RuntimeError("chaos: transient"))
    job = submit(client, _quick, "exact-retry", max_retries_failure=2)
    status = wait(client, job, timeout=60)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED


@pytest.mark.chaos
def test_capacity_wait(cluster, tmp_path):
    """Workers at capacity, task pends, schedules when capacity frees."""
    _url, client = cluster
    from iris.cluster.types import Entrypoint, EnvironmentSpec

    # Submit blockers with high CPU directly using client.submit
    blocker_sentinels = []
    blockers = []
    for i in range(2):
        s = SentinelFile(str(tmp_path / f"blocker-{i}"))
        blocker_sentinels.append(s)
        job = client.submit(
            entrypoint=Entrypoint.from_callable(_block, s),
            name=f"blocker-{i}",
            resources=ResourceSpec(cpu=4, memory="1g"),
            environment=EnvironmentSpec(),
        )
        blockers.append(job)
    time.sleep(1)
    # Submit pending task
    pending = submit(client, _quick, "pending")
    status = client.status(str(pending.job_id))
    assert status.state == cluster_pb2.JOB_STATE_PENDING
    # Release blockers so capacity frees up
    for s in blocker_sentinels:
        s.signal()
    for b in blockers:
        wait(client, b, timeout=30)
    status = wait(client, pending, timeout=30)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED


@pytest.mark.chaos
def test_scheduling_timeout(cluster):
    """Scheduling timeout exceeded → UNSCHEDULABLE."""
    _url, client = cluster
    from iris.cluster.types import Entrypoint, EnvironmentSpec

    # Use client.submit directly to pass scheduling_timeout_seconds
    job = client.submit(
        entrypoint=Entrypoint.from_callable(_quick),
        name="unsched",
        resources=ResourceSpec(cpu=9999, memory="1g"),
        environment=EnvironmentSpec(),
        scheduling_timeout_seconds=2,
    )
    status = wait(client, job, timeout=10)
    assert status.state in (cluster_pb2.JOB_STATE_FAILED, cluster_pb2.JOB_STATE_UNSCHEDULABLE)


@pytest.mark.chaos
def test_dispatch_delayed(cluster):
    """Dispatch delayed by chaos (via heartbeat), but eventually goes through."""
    _url, client = cluster
    enable_chaos("controller.heartbeat", delay_seconds=3.0, failure_rate=1.0, max_failures=2)
    job = submit(client, _quick, "delayed-dispatch")
    status = wait(client, job, timeout=60)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED
