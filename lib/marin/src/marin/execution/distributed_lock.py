# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Lease-based distributed locking via ``StatusFile``.

Wraps a ``fn(output_path) -> T`` so that only one worker executes the
function for a given *output_path* at a time.  If the step has already
completed, ``StepAlreadyDone`` is raised so the caller can load the
cached result instead.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from threading import Event, Thread
from typing import TypeVar

from marin.execution.executor_step_status import (
    HEARTBEAT_INTERVAL,
    StatusFile,
    should_run,
    worker_id,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class StepAlreadyDone(Exception):
    """Raised by ``distributed_lock`` when the step has already succeeded."""


def distributed_lock(fn: Callable[[str], T], *, force_run_failed: bool = True) -> Callable[[str], T]:
    """Decorator: wrap *fn* with lease-based distributed locking.

    The lock is keyed on the *output_path* argument passed to *fn*.  If
    another worker already completed the step (``STATUS_SUCCESS``),
    ``StepAlreadyDone`` is raised so that the caller (typically
    ``disk_cached``) can load the cached artifact instead.

    While *fn* is executing a heartbeat thread refreshes the lock so that
    other workers see it as active.

    This decorator does **not** write status or save artifacts - that is the
    responsibility of the caller.
    """

    @functools.wraps(fn)
    def wrapper(output_path: str) -> T:
        status_file = StatusFile(output_path, worker_id())
        step_label = output_path.rsplit("/", 1)[-1]

        if not should_run(status_file, step_label, force_run_failed=force_run_failed):
            raise StepAlreadyDone(output_path)

        stop_event = Event()

        def _heartbeat():
            while not stop_event.wait(HEARTBEAT_INTERVAL):
                status_file.refresh_lock()

        heartbeat_thread = Thread(target=_heartbeat, daemon=True)
        heartbeat_thread.start()

        try:
            return fn(output_path)
        finally:
            stop_event.set()
            heartbeat_thread.join(timeout=5)
            status_file.release_lock()

    return wrapper
