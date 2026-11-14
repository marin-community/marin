# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Each step produces an `output_path`.
We associate each `output_path` with:
- A status file (`output_path/.executor_status`) containing simple text: SUCCESS, FAILED, DEP_FAILED, or RUNNING
- A LOCK file (`output_path/.executor_status.lock`) for distributed locking

The LOCK file contains JSON with {worker_id, timestamp} and is refreshed periodically.
On GCS, we use generation-based conditional writes for atomicity.
"""

import contextlib
import logging
from collections.abc import Generator
from threading import Event, Thread
from time import sleep

from iris.cluster.client.job_info import get_job_info
from rigging.filesystem.distributed_lock import (
    HEARTBEAT_INTERVAL,
    LeaseLostError,
    default_worker_id,
)
from rigging.filesystem.step_status import (
    STATUS_DEP_FAILED,
    STATUS_FAILED,
    STATUS_RUNNING,
    STATUS_SUCCESS,
    StatusFile,
)
from rigging.timing import RateLimiter

logger = logging.getLogger(__name__)

_LOCK_WAIT_LOG_INTERVAL = 5 * 60


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def worker_id() -> str:
    local_worker_id = default_worker_id()
    job_info = get_job_info()
    if job_info is None:
        return local_worker_id
    return f"{job_info.task_attempt.to_wire()} ({local_worker_id})"


class PreviousTaskFailedError(Exception):
    """Raised when a step failed previously and force_run_failed is False."""


def should_run(
    status_file: StatusFile, step_name: str, force_run_failed: bool = True, force_rerun: bool = False
) -> bool:
    """Check if the step should run based on lease-based distributed locking.

    Uses double-check locking: check status, attempt to acquire lock,
    re-check status after acquisition to avoid overwriting a concurrent
    completion. ``force_rerun`` rebuilds even over an existing ``STATUS_SUCCESS``
    (used for mutable ``dev`` artifacts), still serializing behind any active lock.
    """
    wid = status_file.worker_id
    log_once = True
    wait_log_limiter = RateLimiter(interval_seconds=_LOCK_WAIT_LOG_INTERVAL)

    while True:
        status = status_file.status
        active_lock_holder = status_file.active_lock_holder() if status == STATUS_RUNNING else None

        if log_once and active_lock_holder is None:
            logger.info(f"[{wid}] Status {step_name}: {status}")
            log_once = False

        if status == STATUS_SUCCESS and not force_rerun:
            logger.info(f"[{wid}] Step {step_name} has already succeeded.")
            return False

        if status in [STATUS_FAILED, STATUS_DEP_FAILED]:
            if force_run_failed:
                logger.info(f"[{wid}] Force running {step_name}, previous status: {status}")
            else:
                raise PreviousTaskFailedError(f"Step {step_name} failed previously. Status: {status}")
        elif status == STATUS_RUNNING and active_lock_holder is not None:
            if wait_log_limiter.should_run():
                logger.info(
                    "[%s] Status %s: %s. Another worker holds the active lock (owner=%s). "
                    "Waiting for that worker to finish.",
                    wid,
                    step_name,
                    status,
                    active_lock_holder,
                )
            log_once = False
            sleep(5)
            continue
        elif status == STATUS_RUNNING:
            logger.info(f"[{wid}] Step {step_name} has no active lock, taking over.")

        logger.info(f"[{wid}] Attempting to acquire lock for {step_name}")
        if status_file.try_acquire_lock():
            # Double-check: re-read status after acquiring lock to avoid
            # overwriting a concurrent SUCCESS.
            recheck = status_file.status
            if recheck == STATUS_SUCCESS and not force_rerun:
                logger.info(f"[{wid}] Step {step_name} completed by another worker after lock acquired.")
                status_file.release_lock()
                return False

            status_file.write_status(STATUS_RUNNING)
            logger.info(f"[{wid}] Acquired lock for {step_name}")
            return True

        logger.info(f"[{wid}] Lost lock race for {step_name}, retrying...")
        sleep(1)


# ---------------------------------------------------------------------------
# Step-level distributed lock
# ---------------------------------------------------------------------------


class StepAlreadyDone(Exception):
    """Raised by ``step_lock`` when the step has already succeeded."""


@contextlib.contextmanager
def step_lock(
    output_path: str, step_label: str, *, force_run_failed: bool = True, force_rerun: bool = False
) -> Generator[StatusFile, None, None]:
    """Context manager that acquires a distributed lock with heartbeat refresh.

    Acquires the lock, starts a daemon heartbeat thread, yields the
    ``StatusFile``, then tears down the heartbeat and releases the lock. Status
    writes inside the context do not release the lock; this context manager owns
    release ordering so the heartbeat is stopped first.

    Raises ``StepAlreadyDone`` if another worker completed the step
    while we waited for the lock. ``force_rerun`` rebuilds over an existing
    success (mutable ``dev`` artifacts).
    """
    status_file = StatusFile(output_path, worker_id())
    if not should_run(status_file, step_label, force_run_failed=force_run_failed, force_rerun=force_rerun):
        raise StepAlreadyDone(output_path)

    # Start heartbeat — LeaseLostError is fatal and signals the main thread.
    stop_event = Event()
    lease_lost_event = Event()

    def _heartbeat():
        while not stop_event.wait(HEARTBEAT_INTERVAL):
            try:
                status_file.refresh_lock()
            except LeaseLostError:
                logger.error("Lease lost for %s — step must terminate", output_path, exc_info=True)
                lease_lost_event.set()
                return

    heartbeat_thread = Thread(target=_heartbeat, daemon=True)
    heartbeat_thread.start()

    try:
        yield status_file
    finally:
        stop_event.set()
        heartbeat_thread.join(timeout=5)
        if lease_lost_event.is_set():
            raise LeaseLostError(f"Lease was lost during execution of {output_path}")
        status_file.release_lock()
