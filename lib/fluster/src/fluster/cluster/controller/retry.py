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

"""Job failure and retry logic.

This module provides functions to handle job failures, distinguishing between:
- Worker failures (external): worker died or became unhealthy
- Job failures (internal): job exited with non-zero exit code

Each failure type has separate retry limits (max_retries_preemption for worker
failures, max_retries_failure for job failures). Failed jobs are reset to
PENDING state and re-queued for another scheduling attempt.

Gang scheduling requires all-or-nothing retry: the entire gang is only retried
if ALL jobs in the gang have retries remaining. This maintains gang consistency.
"""

import logging

from fluster.rpc import cluster_pb2
from fluster.cluster.controller.state import ControllerState
from fluster.cluster.types import JobId

logger = logging.getLogger(__name__)


def handle_job_failure(
    state: ControllerState,
    job_id: JobId,
    is_worker_failure: bool,
) -> bool:
    """Handle a job failure, potentially retrying.

    Args:
        state: Controller state
        job_id: ID of the failed job
        is_worker_failure: True if external failure (worker died),
                          False if internal (job exit code != 0)

    Returns:
        True if job was re-queued for retry, False otherwise
    """
    job = state.get_job(job_id)
    if not job:
        return False

    # Increment counter to track this failure
    if is_worker_failure:
        job.preemption_count += 1
        can_retry = job.preemption_count <= job.max_retries_preemption
    else:
        job.failure_count += 1
        can_retry = job.failure_count <= job.max_retries_failure

    if not can_retry:
        logger.warning(f"Job {job_id} exceeded retry limit, not retrying")
        return False

    logger.info(f"Retrying job {job_id} (failures={job.failure_count}, preemptions={job.preemption_count})")

    # Reset job state for retry
    job.state = cluster_pb2.JOB_STATE_PENDING
    job.worker_id = None
    job.started_at_ms = None
    job.finished_at_ms = None
    job.error = None

    # Re-queue the job
    state.add_job(job)
    return True


def handle_gang_failure(
    state: ControllerState,
    gang_id: str,
    is_worker_failure: bool,
) -> list[JobId]:
    """Handle gang failure - terminate all jobs, optionally retry.

    All-or-nothing retry: only retries if ALL jobs in gang have retries left.

    Args:
        state: Controller state
        gang_id: Gang identifier
        is_worker_failure: True if external failure (worker died),
                          False if internal (job failure)

    Returns:
        List of job IDs that were re-queued (empty if gang couldn't retry)
    """
    jobs = state.get_gang_jobs(gang_id)
    if not jobs:
        return []

    # Check if ALL jobs in gang have retries remaining (all-or-nothing)
    # Check before modifying any state to avoid partial updates on failure
    if is_worker_failure:
        can_retry = all(job.preemption_count < job.max_retries_preemption for job in jobs)
    else:
        can_retry = all(job.failure_count < job.max_retries_failure for job in jobs)

    if not can_retry:
        # Mark all running jobs as KILLED (no retry possible)
        for job in jobs:
            if job.state == cluster_pb2.JOB_STATE_RUNNING:
                job.state = cluster_pb2.JOB_STATE_KILLED
                job.error = f"Gang {gang_id} failed"
        logger.warning(f"Gang {gang_id} exceeded retry limit, not retrying")
        return []

    # Retry all jobs in gang
    retried = []
    for job in jobs:
        # Increment appropriate counter
        if is_worker_failure:
            job.preemption_count += 1
        else:
            job.failure_count += 1

        # Reset job state for retry
        job.state = cluster_pb2.JOB_STATE_PENDING
        job.worker_id = None
        job.started_at_ms = None
        job.finished_at_ms = None
        job.error = None

        # Re-queue the job
        state.add_job(job)
        retried.append(job.job_id)

    logger.info(f"Retrying gang {gang_id} with {len(retried)} jobs")
    return retried
