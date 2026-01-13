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

"""Worker heartbeat monitor.

This module provides the HeartbeatMonitor class, which runs a background thread
that periodically polls workers to check their health. The monitor:
- Sends heartbeats to all healthy workers on a periodic interval
- Tracks consecutive failures and marks workers unhealthy after N failures
- Syncs job states from worker responses (completed jobs, exit codes, etc.)
- Triggers failure callbacks when workers exceed the failure threshold

The monitor handles both detecting worker failures and keeping job state
synchronized with the worker's actual job status.
"""

import logging
import threading
import time
from collections.abc import Callable

from fluster import cluster_pb2
from fluster.cluster.controller.state import ControllerState
from fluster.cluster.types import JobId, WorkerId

logger = logging.getLogger(__name__)


class HeartbeatMonitor:
    """Monitors worker health via periodic heartbeats.

    The monitor runs a daemon thread that periodically checks all workers by
    calling the provided heartbeat_fn. On consecutive failures, it marks the
    worker unhealthy and triggers the on_worker_failed callback for retry logic.

    Unlike the Scheduler, the monitor doesn't need a wake event since it only
    runs on a periodic timer - there's no external event that should trigger
    an immediate heartbeat.

    Args:
        state: Controller state containing workers and jobs
        heartbeat_fn: Callable that performs heartbeat RPC to worker. Takes
            worker address as argument, returns HeartbeatResponse on success
            or None on failure (timeout, connection error, etc).
        on_worker_failed: Callback when worker exceeds failure threshold. Called
            with worker_id and list of failed job IDs.
        interval_seconds: How often to check all workers (default: 1.0)
    """

    MAX_CONSECUTIVE_FAILURES = 3

    def __init__(
        self,
        state: ControllerState,
        heartbeat_fn: Callable[[str], cluster_pb2.HeartbeatResponse | None],
        on_worker_failed: Callable[[WorkerId, list[JobId]], None],
        interval_seconds: float = 1.0,
    ):
        self._state = state
        self._heartbeat_fn = heartbeat_fn
        self._on_worker_failed = on_worker_failed
        self._interval = interval_seconds
        self._stop = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the heartbeat monitor background thread.

        The thread is created as a daemon thread so it won't prevent the process
        from exiting.
        """
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the monitor and wait for thread to finish.

        Signals the thread to stop and waits up to 5 seconds for the thread
        to terminate.
        """
        self._stop = True
        if self._thread:
            self._thread.join(timeout=5.0)

    def _run(self) -> None:
        """Main monitor loop.

        Sleeps for the interval, then checks all workers. Repeats until
        stop() is called.
        """
        while not self._stop:
            time.sleep(self._interval)

            if self._stop:
                break

            self._check_all_workers()

    def _check_all_workers(self) -> None:
        """Check health of all workers via heartbeat.

        For each healthy worker:
        1. Call heartbeat_fn to check health
        2. On failure: increment consecutive_failures, possibly mark dead
        3. On success: reset consecutive_failures, update last_heartbeat_ms,
           and sync job states from response

        Workers already marked unhealthy are skipped.
        """
        # Take snapshot of workers to avoid holding lock during RPC
        workers = list(self._state._workers.values())
        now_ms = int(time.time() * 1000)

        for worker in workers:
            if not worker.healthy:
                # Skip workers already marked dead
                continue

            response = self._heartbeat_fn(worker.address)

            if response is None:
                # Heartbeat failed
                worker.consecutive_failures += 1
                logger.warning(
                    f"Heartbeat failed for {worker.worker_id} "
                    f"({worker.consecutive_failures}/{self.MAX_CONSECUTIVE_FAILURES})"
                )

                if worker.consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
                    self._handle_worker_failure(worker)
            else:
                # Success - reset failure count and update state
                worker.consecutive_failures = 0
                worker.last_heartbeat_ms = now_ms
                self._sync_job_states(worker, response)

    def _handle_worker_failure(self, worker) -> None:
        """Mark worker dead and fail all its jobs.

        Called when a worker exceeds MAX_CONSECUTIVE_FAILURES. This:
        1. Marks worker as unhealthy
        2. Gets list of jobs currently running on worker
        3. Clears worker's running_jobs set
        4. Marks all jobs as JOB_STATE_WORKER_FAILED
        5. Calls on_worker_failed callback for retry logic

        Args:
            worker: Worker that has failed
        """
        logger.error(f"Worker {worker.worker_id} declared dead")
        worker.healthy = False

        # Get list of running jobs before clearing
        failed_jobs = list(worker.running_jobs)
        worker.running_jobs.clear()

        # Mark all jobs as failed
        now_ms = int(time.time() * 1000)
        for job_id in failed_jobs:
            job = self._state.get_job(job_id)
            if job:
                job.state = cluster_pb2.JOB_STATE_WORKER_FAILED
                job.finished_at_ms = now_ms
                job.error = f"Worker {worker.worker_id} failed"

        # Notify callback for retry handling
        self._on_worker_failed(worker.worker_id, failed_jobs)

    def _sync_job_states(self, worker, response: cluster_pb2.HeartbeatResponse) -> None:
        """Update controller state from worker's heartbeat response.

        The worker reports status of all jobs currently running or recently
        completed. This syncs the controller's view of job state with the
        worker's actual state.

        For jobs that have finished (SUCCEEDED, FAILED, KILLED), we update
        the job state and remove them from the worker's running_jobs set.

        Args:
            worker: Worker that sent the heartbeat
            response: HeartbeatResponse from worker
        """
        for status in response.jobs:
            job = self._state.get_job(JobId(status.job_id))
            if not job:
                continue

            # Check if job has reached terminal state
            if status.state in (
                cluster_pb2.JOB_STATE_SUCCEEDED,
                cluster_pb2.JOB_STATE_FAILED,
                cluster_pb2.JOB_STATE_KILLED,
            ):
                # Update job state from worker report
                job.state = status.state
                job.finished_at_ms = status.finished_at_ms
                job.error = status.error or None
                job.exit_code = status.exit_code

                # Remove from worker's running jobs
                worker.running_jobs.discard(job.job_id)
