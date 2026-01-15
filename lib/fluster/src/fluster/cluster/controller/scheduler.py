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

"""Pure job scheduler with shallow interface.

This module provides the Scheduler class, which implements job-to-worker matching
logic without any threading, dispatch, or state mutation. The scheduler takes
inputs (pending jobs, workers, current time) and returns outputs (assignments,
timed-out jobs). All side effects are the caller's responsibility.

This design follows the "shallow interface" pattern: the scheduler is a pure
function-like object that can be easily tested and composed without mocking
threads or callbacks.
"""

import logging
from dataclasses import dataclass, field

from fluster.cluster.controller.state import ControllerJob, ControllerState, ControllerWorker
from fluster.cluster.controller.workers import worker_can_fit_job

logger = logging.getLogger(__name__)


@dataclass
class ScheduleResult:
    """Result of a scheduling attempt.

    Contains the job-to-worker assignments that the caller should dispatch,
    and jobs that have exceeded their scheduling timeout.

    Args:
        assignments: List of (job, worker) pairs to dispatch
        timed_out_jobs: Jobs that exceeded their scheduling_timeout_seconds
    """

    assignments: list[tuple[ControllerJob, ControllerWorker]] = field(default_factory=list)
    timed_out_jobs: list[ControllerJob] = field(default_factory=list)


class Scheduler:
    """Pure job-to-worker matching logic.

    The scheduler matches pending jobs to available workers based on resource
    requirements and availability. It does NOT:
    - Dispatch jobs (caller does this)
    - Modify state (caller does this)
    - Run any threads (Controller owns threading)

    This is a stateless utility class - all inputs are passed to find_assignments()
    and all outputs are returned in ScheduleResult.
    """

    def __init__(self, state: ControllerState):
        """Initialize scheduler with controller state for resource lookups.

        Args:
            state: Controller state used for looking up job resources when
                   computing committed resources on workers.
        """
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
                    "No suitable worker for job %s (cpu=%d, memory=%d)",
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
        """Check if job has exceeded its scheduling timeout."""
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

        Takes into account both jobs already running on each worker AND jobs
        assigned earlier in this scheduling round (tracked in assigned_jobs_by_worker).

        Args:
            job: Job to find a worker for
            workers: Available workers to consider
            assigned_jobs_by_worker: Jobs assigned in this round, by worker_id

        Returns:
            First matching worker, or None if no worker can fit the job
        """
        for worker in workers:
            if not worker.healthy:
                continue
            # Check if worker can fit this job, considering jobs assigned this round
            jobs_assigned_this_round = assigned_jobs_by_worker.get(worker.worker_id, [])
            if worker_can_fit_job(self._state, worker, job, jobs_assigned_this_round):
                return worker
        return None
