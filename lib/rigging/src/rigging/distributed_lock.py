# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Lease-based distributed locking decorator.

Wraps a ``fn(path) -> T`` so that only one worker executes the function for a
given *path* at a time.  If the work has already completed, ``AlreadyComplete``
is raised so the caller can load the cached result instead.
"""

from __future__ import annotations

import functools
import logging
import time
from collections.abc import Callable
from threading import Event, Thread
from typing import TypeVar

from rigging.status_file import (
    HEARTBEAT_INTERVAL,
    STATUS_FAILED,
    STATUS_RUNNING,
    STATUS_SUCCESS,
    StatusFile,
    worker_id,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class AlreadyComplete(Exception):
    """Raised by ``distributed_lock`` when the work has already succeeded."""


class PreviousRunFailedError(Exception):
    """Raised when a previous run failed and *force_rerun* is False."""


def try_claim(
    status_file: StatusFile,
    label: str,
    *,
    force_rerun: bool = True,
    failed_statuses: tuple[str, ...] = (STATUS_FAILED,),
) -> bool:
    """Attempt to claim the right to run work protected by *status_file*.

    Blocks until either this worker acquires the lock or another worker
    finishes successfully.

    Args:
        status_file: The status file to claim.
        label: Human-readable label for log messages.
        force_rerun: If True, re-run on previous failure.  If False,
            raise ``PreviousRunFailedError``.
        failed_statuses: Status tokens that indicate a previous failure.
    """
    wid = status_file.worker_id
    log_once = True

    while True:
        status = status_file.status

        if log_once:
            logger.info("[%s] Status %s: %s", wid, label, status)
            log_once = False

        if status == STATUS_SUCCESS:
            logger.info("[%s] %s has already succeeded.", wid, label)
            return False

        if status in failed_statuses:
            if force_rerun:
                logger.info("[%s] Force running %s, previous status: %s", wid, label, status)
            else:
                raise PreviousRunFailedError(f"{label} failed previously. Status: {status}")
        elif status == STATUS_RUNNING and status_file.has_active_lock():
            logger.debug("[%s] %s has active lock, waiting...", wid, label)
            time.sleep(5)
            continue
        elif status == STATUS_RUNNING:
            logger.info("[%s] %s has no active lock, taking over.", wid, label)

        logger.info("[%s] Attempting to acquire lock for %s", wid, label)
        if status_file.try_acquire_lock():
            status_file.write_status(STATUS_RUNNING)
            logger.info("[%s] Acquired lock for %s", wid, label)
            return True

        logger.info("[%s] Lost lock race for %s, retrying...", wid, label)
        time.sleep(1)


def distributed_lock(fn: Callable[[str], T], *, force_rerun: bool = True) -> Callable[[str], T]:
    """Decorator: wrap *fn* with lease-based distributed locking.

    The lock is keyed on the *path* argument passed to *fn*.  If another worker
    already completed the work (``STATUS_SUCCESS``), ``AlreadyComplete`` is
    raised so that the caller (typically ``disk_cache``) can load the cached
    result instead.

    While *fn* is executing a heartbeat thread refreshes the lock so that other
    workers see it as active.
    """

    @functools.wraps(fn)
    def wrapper(path: str) -> T:
        status_file = StatusFile(path, worker_id())
        label = path.rsplit("/", 1)[-1]

        if not try_claim(status_file, label, force_rerun=force_rerun):
            raise AlreadyComplete(path)

        stop_event = Event()

        def _heartbeat():
            while not stop_event.wait(HEARTBEAT_INTERVAL):
                status_file.refresh_lock()

        heartbeat_thread = Thread(target=_heartbeat, daemon=True)
        heartbeat_thread.start()

        try:
            return fn(path)
        finally:
            stop_event.set()
            heartbeat_thread.join(timeout=5)
            status_file.release_lock()

    return wrapper
