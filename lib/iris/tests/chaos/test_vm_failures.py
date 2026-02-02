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

"""Autoscaler/VM failure chaos tests.

Tests that validate VM lifecycle failure modes through externally-observable behavior:
- Quota exceeded: System surfaces quota errors in autoscaler actions and recovers when capacity returns
- VM init stuck: Jobs become unschedulable when workers never initialize
- VM preempted: Worker failures trigger autoscaler cleanup with backoff

These tests use the full cluster fixture and verify user-facing outcomes through
job submission and autoscaler status APIs.
"""

import pytest
from iris.chaos import enable_chaos
from iris.cluster.vm.managed_vm import QuotaExceededError
from iris.rpc import cluster_pb2
from iris.time_utils import Duration
from .conftest import submit, wait, _quick


@pytest.mark.chaos
def test_quota_exceeded_creates_error_action(cluster):
    """Quota exceeded errors are surfaced as 'quota_exceeded' actions in autoscaler status.

    When VM creation hits quota limits, the autoscaler logs a user-visible
    quota_exceeded action. Jobs may eventually succeed if capacity becomes
    available, but the quota error must be visible in autoscaler status.
    """
    _url, client = cluster

    # Inject quota failure in VM creation - fail first few attempts
    # The autoscaler will try multiple scale groups, so we need enough failures
    # to ensure at least one quota action is logged
    enable_chaos(
        "vm.create",
        failure_rate=1.0,
        max_failures=3,
        error=QuotaExceededError("Quota 'TPU_V4_CORES' exceeded in zone us-central1-a"),
    )

    # Submit a job - autoscaler will attempt to create VMs
    job = submit(client, _quick, "quota-test", scheduling_timeout=Duration.from_seconds(15))

    # Wait for job to complete (may succeed or fail depending on timing)
    wait(client, job, timeout=25)

    # Verify autoscaler logged a quota_exceeded action (this is the key observable behavior)
    response = client._cluster.get_autoscaler_status()
    quota_actions = [action for action in response.status.recent_actions if action.action_type == "quota_exceeded"]
    assert len(quota_actions) > 0, "Expected autoscaler to log quota_exceeded action"
    assert "Quota" in quota_actions[0].reason or "quota" in quota_actions[0].reason.lower()


@pytest.mark.chaos
def test_vm_init_timeout_causes_unschedulable_jobs(cluster):
    """When VMs never initialize, jobs fail to schedule within timeout.

    VMs stuck in initialization phase never become healthy workers,
    causing jobs to remain unschedulable and eventually fail.
    """
    _url, client = cluster

    # Inject permanent failure in worker registration - workers will never become ready
    enable_chaos("worker.register", failure_rate=1.0, error=RuntimeError("chaos: worker init stuck"))

    # Submit job with short scheduling timeout
    job = submit(client, _quick, "init-stuck", scheduling_timeout=Duration.from_seconds(10))

    # Job should fail to schedule since workers never become ready
    status = wait(client, job, timeout=30)
    assert status.state in (
        cluster_pb2.JOB_STATE_UNSCHEDULABLE,
        cluster_pb2.JOB_STATE_FAILED,
    ), f"Expected job to fail scheduling, got {status.state}"


@pytest.mark.chaos
def test_autoscaler_logs_scale_up_actions(cluster):
    """Autoscaler logs scale_up actions when creating VMs.

    When demand exceeds capacity, the autoscaler creates new VM groups
    and logs scale_up actions that are visible in autoscaler status.
    This validates that autoscaler actions are properly logged and
    accessible via the status API.
    """
    _url, client = cluster

    # Submit a job - autoscaler will scale up to meet demand
    job = submit(client, _quick, "scale-test")
    status = wait(client, job, timeout=60)

    # Job should complete successfully
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED

    # Verify autoscaler logged scale_up actions
    response = client._cluster.get_autoscaler_status()
    scale_up_actions = [action for action in response.status.recent_actions if action.action_type == "scale_up"]
    assert len(scale_up_actions) > 0, "Expected autoscaler to log scale_up actions"
    # Verify the actions have expected fields
    assert scale_up_actions[0].scale_group, "scale_up action should have scale_group"
    assert scale_up_actions[0].status in ("pending", "completed"), "scale_up action should have status"
