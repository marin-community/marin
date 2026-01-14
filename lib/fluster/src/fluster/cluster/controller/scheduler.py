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

"""Job scheduler background thread.

This module provides the Scheduler class, which runs a background thread that
continuously matches pending jobs to available workers. The scheduler wakes on:
- Periodic timer (default 1 second)
- Explicit wake() calls (e.g., when new workers register or jobs finish)

The scheduler delegates the actual dispatch RPC to a provided dispatch_fn,
which should return True on success and False on failure. On success, the
scheduler updates job state to RUNNING and assigns the worker. On failure,
it marks the worker unhealthy and re-queues the job.
"""

import logging
import threading
import time
from collections.abc import Callable

from fluster import cluster_pb2
from fluster.cluster.controller.state import ControllerJob, ControllerState, ControllerWorker
from fluster.cluster.controller.workers import find_worker_for_job

logger = logging.getLogger(__name__)


class Scheduler:
    """Background scheduler that dispatches jobs to workers.

    The scheduler runs a daemon thread that continuously tries to match pending
    jobs from the controller state queue to available workers. It uses an event
    for wake signaling, allowing external code to trigger immediate scheduling
    attempts without waiting for the next timer tick.

    Args:
        state: Controller state containing jobs and workers
        dispatch_fn: Callable that dispatches a job to a worker. Should return
            True if dispatch succeeded, False otherwise.
        interval_seconds: How often to wake and check for pending jobs (default: 1.0)
    """

    def __init__(
        self,
        state: ControllerState,
        dispatch_fn: Callable[[ControllerJob, ControllerWorker], bool],
        interval_seconds: float = 1.0,
    ):
        self._state = state
        self._dispatch_fn = dispatch_fn
        self._interval = interval_seconds
        self._wake_event = threading.Event()
        self._stop = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the scheduler background thread.

        The thread is created as a daemon thread so it won't prevent the process
        from exiting.
        """
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the scheduler and wait for thread to finish.

        Signals the thread to stop, wakes it to process the stop signal, and
        waits up to 5 seconds for the thread to terminate.
        """
        self._stop = True
        self._wake_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)

    def wake(self) -> None:
        """Signal scheduler to run immediately (non-blocking).

        This is called when events occur that may make new jobs schedulable,
        such as:
        - New worker registered
        - Job finished (freeing worker capacity)
        - New job submitted
        """
        self._wake_event.set()

    def _run(self) -> None:
        """Main scheduler loop.

        Waits on the wake event with a timeout, then attempts to schedule
        pending jobs. Repeats until stop() is called.
        """
        while not self._stop:
            # Wait for wake signal or timeout
            self._wake_event.wait(timeout=self._interval)
            self._wake_event.clear()

            if self._stop:
                break

            self._schedule_pending_jobs()

    def _schedule_pending_jobs(self) -> None:
        """Try to schedule all pending jobs with resource-aware matching.

        New algorithm that prevents head-of-line blocking:
        1. Peek all pending jobs (don't pop) to iterate in FIFO order
        2. For each job:
           a. Check scheduling timeout - if expired, mark UNSCHEDULABLE
           b. Find a worker that can fit the job (capacity checked dynamically)
           c. If found: dispatch, remove from queue, add to worker.running_jobs
           d. If not found: skip to next job (DON'T block queue)

        This allows smaller jobs to be scheduled even when large jobs at the
        front of the queue are waiting for capacity.
        """
        now_ms = int(time.time() * 1000)
        pending_jobs = self._state.peek_pending_jobs()

        for job in pending_jobs:
            # Check scheduling timeout
            if self._is_job_timed_out(job, now_ms):
                self._mark_unschedulable(job, now_ms)
                continue

            # Try to find a worker that can fit this job
            worker = find_worker_for_job(self._state, job)
            if not worker:
                # No worker can fit this job right now - skip, don't block
                logger.debug(f"No worker available for job {job.job_id}, skipping")
                continue

            # Attempt to dispatch
            success = self._dispatch_fn(job, worker)
            if success:
                self._handle_successful_dispatch(job, worker, now_ms)
            else:
                self._handle_failed_dispatch(job, worker)

    def _is_job_timed_out(self, job: ControllerJob, now_ms: int) -> bool:
        """Check if job has exceeded its scheduling timeout."""
        timeout_seconds = job.request.scheduling_timeout_seconds
        if timeout_seconds <= 0:
            return False  # No timeout configured

        pending_duration_ms = now_ms - job.submitted_at_ms
        timeout_ms = timeout_seconds * 1000
        return pending_duration_ms > timeout_ms

    def _mark_unschedulable(self, job: ControllerJob, now_ms: int) -> None:
        """Mark job as unschedulable and remove from queue."""
        logger.warning(
            f"Job {job.job_id} exceeded scheduling timeout "
            f"({job.request.scheduling_timeout_seconds}s), marking as UNSCHEDULABLE"
        )
        job.state = cluster_pb2.JOB_STATE_UNSCHEDULABLE
        job.finished_at_ms = now_ms
        job.error = f"Scheduling timeout exceeded ({job.request.scheduling_timeout_seconds}s)"
        self._state.remove_from_queue(job.job_id)
        self._state.log_action(
            "job_unschedulable",
            job_id=job.job_id,
            details=f"timeout={job.request.scheduling_timeout_seconds}s",
        )

    def _handle_successful_dispatch(self, job: ControllerJob, worker: ControllerWorker, now_ms: int) -> None:
        """Update state after successful dispatch."""
        # Update job state
        job.state = cluster_pb2.JOB_STATE_RUNNING
        job.worker_id = worker.worker_id
        job.started_at_ms = now_ms

        # Update worker state - this is what get_committed_resources uses
        worker.running_jobs.add(job.job_id)

        # Remove from queue
        self._state.remove_from_queue(job.job_id)

        logger.info(f"Dispatched job {job.job_id} to worker {worker.worker_id}")
        self._state.log_action(
            "job_dispatched",
            job_id=job.job_id,
            worker_id=worker.worker_id,
        )

    def _handle_failed_dispatch(self, job: ControllerJob, worker: ControllerWorker) -> None:
        """Handle dispatch failure - mark worker unhealthy, keep job in queue."""
        worker.healthy = False
        logger.warning(f"Failed to dispatch job {job.job_id} to {worker.worker_id}, " "marking worker unhealthy")
        self._state.log_action(
            "dispatch_failed",
            job_id=job.job_id,
            worker_id=worker.worker_id,
        )
