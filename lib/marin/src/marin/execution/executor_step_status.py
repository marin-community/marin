# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Each `ExecutorStep` produces an `output_path`.
We associate each `output_path` with:
- A status file (`output_path/.executor_status`) containing simple text: SUCCESS, FAILED, DEP_FAILED, or RUNNING
- A LOCK file (`output_path/.executor_status.lock`) for distributed locking

The LOCK file contains JSON with {worker_id, timestamp} and is refreshed periodically.
On GCS, we use generation-based conditional writes for atomicity.
"""

import contextlib
import functools
import json
import logging
import os
import time
from collections.abc import Callable, Generator
from threading import Event, Thread
from typing import TypeVar

from rigging.distributed_lock import (
    HEARTBEAT_INTERVAL,
    LeaseLostError,
    create_lock,
    default_worker_id,
)
from rigging.filesystem import url_to_fs

logger = logging.getLogger(__name__)

T = TypeVar("T")

STATUS_RUNNING = "RUNNING"
STATUS_FAILED = "FAILED"
STATUS_SUCCESS = "SUCCESS"
STATUS_DEP_FAILED = "DEP_FAILED"  # Dependency failed


def get_status_path(output_path: str) -> str:
    """Return the path of the status file associated with `output_path`."""
    return os.path.join(output_path, ".executor_status")


class StatusFile:
    """Manages executor step status with distributed locking.

    Two types of files:
    - LOCK file (JSON): Single file for distributed lock acquisition.
      Contains {worker_id, timestamp}. Must be refreshed periodically.
    - Status file (simple text): Step state - SUCCESS, FAILED, DEP_FAILED, or RUNNING.

    Lock acquisition and release delegate to ``rigging.distributed_lock``.
    """

    def __init__(self, output_path: str, worker_id: str):
        self.output_path = output_path
        self.path = get_status_path(output_path)
        self.worker_id = worker_id
        self._lock_path = self.path + ".lock"
        self.fs = url_to_fs(self.path, use_listings_cache=False)[0]
        self._lock = create_lock(self._lock_path, worker_id=worker_id)

    @property
    def status(self) -> str | None:
        """Read current status from status file.

        The modern representation stores a single status token, but older
        executors wrote JSON-line event logs. We support both until the legacy
        files are gone.
        """
        if not self.fs.exists(self.path):
            return None

        with self.fs.open(self.path, "r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        if not lines:
            return None

        # New format: a single status token such as SUCCESS/RUNNING/etc.
        if len(lines) == 1 and not lines[0].startswith("{"):
            return lines[0]

        # Legacy format: JSON event log, one object per line.
        legacy_status = self._parse_legacy_status(lines)
        if legacy_status:
            return legacy_status

        # Fall back to the last non-empty line if parsing fails.
        return lines[-1]

    @staticmethod
    def _parse_legacy_status(lines: list[str]) -> str | None:
        """Return the latest status from the deprecated JSON-lines format."""
        last_status: str | None = None
        for line in lines:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                status = event.get("status")
                if isinstance(status, str):
                    last_status = status
        return last_status

    def write_status(self, status: str) -> None:
        """Write the status file without changing lock ownership.

        ``step_lock`` owns the lock lifetime: it stops the heartbeat before
        releasing the lock. Keeping status writes separate from lock release
        prevents the heartbeat from observing our own terminal cleanup as lease
        loss.
        """
        parent = os.path.dirname(self.path)
        if not self.fs.exists(parent):
            self.fs.makedirs(parent, exist_ok=True)
        with self.fs.open(self.path, "w") as f:
            f.write(status)

        logger.debug("[%s] Wrote status %s to %s", self.worker_id, status, self.path)

    def refresh_lock(self) -> None:
        """Refresh a lock held by the current worker.

        Raises ``LeaseLostError`` if another worker holds the lock, or if the
        lock disappeared before the heartbeat stopped.
        """
        self._lock.refresh()

    def try_acquire_lock(self) -> bool:
        """Try to acquire the lock using atomic LOCK file."""
        return self._lock.try_acquire()

    def release_lock(self) -> None:
        """Release the lock if we hold it."""
        self._lock.release()

    def has_active_lock(self) -> bool:
        """Check if any worker has an active (non-stale) lock."""
        return self._lock.has_active_holder()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def worker_id() -> str:
    return default_worker_id()


class PreviousTaskFailedError(Exception):
    """Raised when a step failed previously and force_run_failed is False."""


def should_run(status_file: StatusFile, step_name: str, force_run_failed: bool = True) -> bool:
    """Check if the step should run based on lease-based distributed locking.

    Uses double-check locking: check status, attempt to acquire lock,
    re-check status after acquisition to avoid overwriting a concurrent
    completion.
    """
    wid = status_file.worker_id
    log_once = True

    while True:
        status = status_file.status

        if log_once:
            logger.info(f"[{wid}] Status {step_name}: {status}")
            log_once = False

        if status == STATUS_SUCCESS:
            logger.info(f"[{wid}] Step {step_name} has already succeeded.")
            return False

        if status in [STATUS_FAILED, STATUS_DEP_FAILED]:
            if force_run_failed:
                logger.info(f"[{wid}] Force running {step_name}, previous status: {status}")
            else:
                raise PreviousTaskFailedError(f"Step {step_name} failed previously. Status: {status}")
        elif status == STATUS_RUNNING and status_file.has_active_lock():
            logger.debug(f"[{wid}] Step {step_name} has active lock, waiting...")
            time.sleep(5)
            continue
        elif status == STATUS_RUNNING:
            logger.info(f"[{wid}] Step {step_name} has no active lock, taking over.")

        logger.info(f"[{wid}] Attempting to acquire lock for {step_name}")
        if status_file.try_acquire_lock():
            # Double-check: re-read status after acquiring lock to avoid
            # overwriting a concurrent SUCCESS.
            recheck = status_file.status
            if recheck == STATUS_SUCCESS:
                logger.info(f"[{wid}] Step {step_name} completed by another worker after lock acquired.")
                status_file.release_lock()
                return False

            status_file.write_status(STATUS_RUNNING)
            logger.info(f"[{wid}] Acquired lock for {step_name}")
            return True

        logger.info(f"[{wid}] Lost lock race for {step_name}, retrying...")
        time.sleep(1)


# ---------------------------------------------------------------------------
# Step-level distributed lock decorator
# ---------------------------------------------------------------------------


class StepAlreadyDone(Exception):
    """Raised by ``step_lock`` / ``distributed_lock`` when the step has already succeeded."""


@contextlib.contextmanager
def step_lock(output_path: str, step_label: str, *, force_run_failed: bool = True) -> Generator[StatusFile, None, None]:
    """Context manager that acquires a distributed lock with heartbeat refresh.

    Acquires the lock, starts a daemon heartbeat thread, yields the
    ``StatusFile``, then tears down the heartbeat and releases the lock. Status
    writes inside the context do not release the lock; this context manager owns
    release ordering so the heartbeat is stopped first.

    Raises ``StepAlreadyDone`` if another worker completed the step
    while we waited for the lock.
    """
    status_file = StatusFile(output_path, worker_id())
    if not should_run(status_file, step_label, force_run_failed=force_run_failed):
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


def distributed_lock(fn: Callable[[str], T], *, force_run_failed: bool = True) -> Callable[[str], T]:
    """Decorator: wrap *fn* with lease-based distributed locking.

    The lock is keyed on the *output_path* argument passed to *fn*.  If
    another worker already completed the step (``STATUS_SUCCESS``),
    ``StepAlreadyDone`` is raised so that the caller (typically
    ``disk_cached``) can load the cached artifact instead.

    While *fn* is executing a heartbeat thread refreshes the lock so that
    other workers see it as active.

    This decorator does **not** write status or save artifacts — that is the
    responsibility of the caller.
    """

    @functools.wraps(fn)
    def wrapper(output_path: str) -> T:
        step_label = output_path.rsplit("/", 1)[-1]
        with step_lock(output_path, step_label, force_run_failed=force_run_failed):
            return fn(output_path)

    return wrapper
