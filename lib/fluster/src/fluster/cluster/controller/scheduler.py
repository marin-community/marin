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

"""Pure job-to-worker matching without threading, dispatch, or state mutation."""

import logging
from dataclasses import dataclass, field

from fluster.cluster.controller.resources import (
    get_device_type,
    get_device_variant,
    get_gpu_count,
    parse_memory_string,
)
from fluster.cluster.controller.state import ControllerJob, ControllerState, ControllerWorker

logger = logging.getLogger(__name__)


def worker_can_fit_job(
    state: ControllerState,
    worker: ControllerWorker,
    job: ControllerJob,
    additional_jobs: list[ControllerJob] | None = None,
) -> bool:
    """Check if worker has sufficient available capacity for job.

    Checks CPU, memory, device type, device variant, and GPU count.

    Args:
        state: Controller state for looking up committed resources
        worker: Worker to check capacity for
        job: Job to check if it fits
        additional_jobs: Jobs assigned this scheduling round but not yet dispatched

    Returns:
        True if worker can fit the job
    """
    job_resources = job.request.resources
    worker_resources = worker.resources

    committed_cpu, committed_memory, committed_gpu = state.get_committed_resources(worker)

    if additional_jobs:
        for additional_job in additional_jobs:
            add_resources = additional_job.request.resources
            committed_cpu += add_resources.cpu
            committed_memory += parse_memory_string(add_resources.memory)
            committed_gpu += get_gpu_count(add_resources.device)

    # CPU check
    available_cpu = worker_resources.cpu - committed_cpu
    if job_resources.cpu > available_cpu:
        return False

    # Memory check
    worker_memory = parse_memory_string(worker_resources.memory)
    job_memory = parse_memory_string(job_resources.memory)
    available_memory = worker_memory - committed_memory
    if job_memory > available_memory:
        return False

    # Device type check
    job_device_type = get_device_type(job_resources.device)
    worker_device_type = get_device_type(worker_resources.device)
    if job_device_type != worker_device_type:
        return False

    # Device variant check (only if job specifies one that's not "auto")
    job_variant = get_device_variant(job_resources.device)
    if job_variant and job_variant != "auto":
        worker_variant = get_device_variant(worker_resources.device)
        if worker_variant != job_variant:
            return False

    # GPU count check
    if job_device_type == "gpu":
        job_gpu_count = get_gpu_count(job_resources.device)
        worker_gpu_count = get_gpu_count(worker_resources.device)
        available_gpus = worker_gpu_count - committed_gpu
        if job_gpu_count > available_gpus:
            return False

    return True


@dataclass
class ScheduleResult:
    """Result of a scheduling attempt.

    Args:
        assignments: List of (job, worker) pairs to dispatch
        timed_out_jobs: Jobs that exceeded their scheduling_timeout_seconds
    """

    assignments: list[tuple[ControllerJob, ControllerWorker]] = field(default_factory=list)
    timed_out_jobs: list[ControllerJob] = field(default_factory=list)


class Scheduler:
    """Pure job-to-worker matching logic. Does not dispatch jobs, modify state, or run threads."""

    def __init__(self, state: ControllerState):
        self._state = state

    def find_assignments(
        self,
        pending_jobs: list[ControllerJob],
        workers: list[ControllerWorker],
        now_ms: int,
    ) -> ScheduleResult:
        """Match pending jobs to available workers.

        Uses first-fit algorithm, skipping jobs that don't fit any worker.
        Also identifies jobs that have exceeded their scheduling timeout.

        The algorithm prevents head-of-line blocking: if a large job at the
        front of the queue doesn't fit, smaller jobs behind it can still be
        scheduled.

        Args:
            pending_jobs: Jobs waiting to be scheduled (in FIFO order)
            workers: Available workers (only healthy ones should be passed)
            now_ms: Current timestamp in milliseconds

        Returns:
            ScheduleResult with assignments and timed-out jobs
        """
        result = ScheduleResult()

        # Track which workers have been assigned jobs in this scheduling round
        # so we account for their capacity correctly
        assigned_jobs_by_worker: dict[str, list[ControllerJob]] = {}

        for job in pending_jobs:
            if self._is_job_timed_out(job, now_ms):
                result.timed_out_jobs.append(job)
                continue

            worker = self._find_worker_for_job(job, workers, assigned_jobs_by_worker)
            if worker:
                result.assignments.append((job, worker))
                assigned_jobs_by_worker.setdefault(worker.worker_id, []).append(job)
            else:
                logger.debug(
                    "No suitable worker for job %s (cpu=%d, memory=%s)",
                    job.job_id,
                    job.request.resources.cpu,
                    job.request.resources.memory,
                )

        if result.assignments or result.timed_out_jobs:
            logger.debug(
                "Scheduling cycle: %d pending, %d assigned, %d timed_out",
                len(pending_jobs),
                len(result.assignments),
                len(result.timed_out_jobs),
            )
        return result

    def _is_job_timed_out(self, job: ControllerJob, now_ms: int) -> bool:
        timeout_seconds = job.request.scheduling_timeout_seconds
        if timeout_seconds <= 0:
            return False

        pending_duration_ms = now_ms - job.submitted_at_ms
        timeout_ms = timeout_seconds * 1000
        return pending_duration_ms > timeout_ms

    def _find_worker_for_job(
        self,
        job: ControllerJob,
        workers: list[ControllerWorker],
        assigned_jobs_by_worker: dict[str, list[ControllerJob]],
    ) -> ControllerWorker | None:
        """Find first worker that can fit the job.

        Accounts for both jobs already running on each worker AND jobs assigned earlier
        in this scheduling round (tracked in assigned_jobs_by_worker).
        """
        for worker in workers:
            if not worker.healthy:
                continue
            # Check if worker can fit this job, considering jobs assigned this round
            jobs_assigned_this_round = assigned_jobs_by_worker.get(worker.worker_id, [])
            if worker_can_fit_job(self._state, worker, job, jobs_assigned_this_round):
                return worker
        return None
