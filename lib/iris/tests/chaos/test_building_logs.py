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

"""Tests for build log visibility during BUILDING state."""

import time

from iris.chaos import enable_chaos
from iris.rpc import cluster_pb2

from .conftest import submit


def _quick_task():
    """A simple task that returns immediately."""
    return 42


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
    _url, client = cluster

    # Enable chaos to inject a 5-second delay during the BUILDING state.
    # This gives us time to observe the BUILDING state and fetch logs.
    enable_chaos("worker.building_delay", delay_seconds=5.0, max_failures=1)

    job = submit(client, _quick_task, "test-building-logs")

    # Poll until we see the task in BUILDING state
    building_state_observed = False
    build_logs_found = False
    job_running_during_building = False

    start_time = time.time()
    timeout = 15.0  # Should be enough for 5s delay + some overhead

    while time.time() - start_time < timeout:
        status = client.status(str(job.job_id))

        # Check if any task is in BUILDING state
        task_states = [task.state for task in status.tasks]
        if cluster_pb2.TASK_STATE_BUILDING in task_states:
            building_state_observed = True

            # Verify job status shows RUNNING when task is BUILDING
            if status.state == cluster_pb2.JOB_STATE_RUNNING:
                job_running_during_building = True

            # Try to fetch logs for the building task
            try:
                logs = client.fetch_task_logs(str(job.job_id), 0)
                # Look for build logs
                for entry in logs:
                    if entry.source == "build":
                        build_logs_found = True
                        break
            except Exception:
                # If logs aren't available yet, keep polling
                pass

        # If we've observed everything we need, we can stop early
        if building_state_observed and build_logs_found and job_running_during_building:
            break

        time.sleep(0.2)  # Poll interval

    # Wait for job to complete
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = client.status(str(job.job_id))
        if status.state in (
            cluster_pb2.JOB_STATE_SUCCEEDED,
            cluster_pb2.JOB_STATE_FAILED,
            cluster_pb2.JOB_STATE_KILLED,
        ):
            break
        time.sleep(0.2)

    # Verify our observations
    assert building_state_observed, "Task never entered BUILDING state (chaos delay may not have triggered)"
    assert job_running_during_building, "Job status was not RUNNING when task was BUILDING"
    assert build_logs_found, "Build logs were not available during BUILDING state"

    # Verify job completed successfully
    final_status = client.status(str(job.job_id))
    assert final_status.state == cluster_pb2.JOB_STATE_SUCCEEDED, f"Job failed with state: {final_status.state}"
