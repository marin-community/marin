# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for build log visibility during BUILDING state."""

import time

import pytest
from iris.chaos import enable_chaos
from iris.rpc import cluster_pb2

pytestmark = pytest.mark.e2e


def _quick_task():
    """A simple task that returns immediately."""
    return 42


@pytest.mark.skip(reason="Flaky: chaos delay doesn't reliably trigger BUILDING state")
def test_build_logs_visible_during_building_state(cluster):
    """Verify that build logs are visible while task is in BUILDING state.

    This test:
    1. Injects a chaos delay during the building phase
    2. Submits a job
    3. Polls the job status and task logs while the task is BUILDING
    4. Verifies that:
       - Job status shows RUNNING when task is BUILDING
       - Build logs are available during the BUILDING state
    """
    enable_chaos("worker.building_delay", delay_seconds=5.0, max_failures=1)

    job = cluster.submit(_quick_task, "test-building-logs")

    building_state_observed = False
    build_logs_found = False
    job_running_during_building = False

    start_time = time.time()
    timeout = 15.0

    while time.time() - start_time < timeout:
        status = cluster.status(job)

        task_states = [task.state for task in status.tasks]
        if cluster_pb2.TASK_STATE_BUILDING in task_states:
            building_state_observed = True

            if status.state == cluster_pb2.JOB_STATE_RUNNING:
                job_running_during_building = True

            try:
                logs = cluster.get_task_logs(job, task_index=0)
                for line in logs:
                    if "build" in line:
                        build_logs_found = True
                        break
            except Exception:
                pass

        if building_state_observed and build_logs_found and job_running_during_building:
            break

        time.sleep(0.2)

    # Wait for job to complete
    status = cluster.wait(job, timeout=timeout)

    assert building_state_observed, "Task never entered BUILDING state (chaos delay may not have triggered)"
    assert job_running_during_building, "Job status was not RUNNING when task was BUILDING"
    assert build_logs_found, "Build logs were not available during BUILDING state"
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED, f"Job failed with state: {status.state}"
