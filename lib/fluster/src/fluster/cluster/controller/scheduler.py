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
        """Try to schedule all pending jobs.

        For each pending job in the queue:
        1. Find an available worker
        2. If no worker available, re-queue the job and stop (don't busy-loop)
        3. Call dispatch_fn to dispatch the job
        4. On success: update job state to RUNNING and assign to worker
        5. On failure: mark worker unhealthy and re-queue the job

        Jobs are processed one at a time in FIFO order. If we can't find a
        worker for a job, we stop trying to avoid spinning on the queue when
        no capacity is available.
        """
        while True:
            job = self._state.pop_next_pending()
            if not job:
                # No more pending jobs
                break

            worker = find_worker_for_job(self._state, job)
            if not worker:
                # No worker available - re-queue and stop trying
                # Re-queuing puts it at the back of the queue, which is correct
                # since we couldn't schedule it this round
                self._state.add_job(job)
                break

            # Attempt to dispatch
            success = self._dispatch_fn(job, worker)
            if success:
                # Update job state to running
                job.state = cluster_pb2.JOB_STATE_RUNNING
                job.worker_id = worker.worker_id
                job.started_at_ms = int(time.time() * 1000)
                worker.running_jobs.add(job.job_id)
                logger.info(f"Dispatched job {job.job_id} to worker {worker.worker_id}")
            else:
                # Dispatch failed - mark worker unhealthy and re-queue job
                worker.healthy = False
                self._state.add_job(job)
                logger.warning(f"Failed to dispatch to {worker.worker_id}, re-queuing job {job.job_id}")
