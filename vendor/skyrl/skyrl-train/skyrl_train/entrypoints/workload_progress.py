"""Carry completed workload milestones from a Ray entrypoint to its Iris driver."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from contextvars import ContextVar
import os
from pathlib import Path
import time
from typing import Any

import ray
from ray.util.queue import Empty, Full, Queue

from marinskyrl.driver_status import DRIVER_PROGRESS_PATH_ENV

_REPORT_INTERVAL = 5
_reporter: ContextVar[Callable[[], None] | None] = ContextVar("workload_progress", default=None)


def report_workload_progress() -> None:
    """Record a completed phase or unit of work, independent of progress display."""
    if (report := _reporter.get()) is not None:
        report()


@contextmanager
def report_to_driver(progress: Queue | None) -> Iterator[None]:
    """Limit progress RPCs and coalesce updates if the driver has yet to read one."""
    if progress is None:
        yield
        return

    next_report = 0.0

    def report() -> None:
        nonlocal next_report
        now = time.monotonic()
        if now >= next_report:
            next_report = now + _REPORT_INTERVAL
            with suppress(Full):
                progress.put_nowait(None)

    token = _reporter.set(report)
    try:
        report()
        yield
    finally:
        _reporter.reset(token)


@contextmanager
def workload_progress_queue() -> Iterator[Queue | None]:
    """Make the progress channel only for drivers supervised by Iris."""
    if not os.environ.get(DRIVER_PROGRESS_PATH_ENV):
        yield None
        return
    queue = Queue(maxsize=1, actor_options={"num_cpus": 0})
    try:
        yield queue
    finally:
        queue.shutdown(force=True)


def wait_for_workload(result: ray.ObjectRef, progress: Queue | None, *, timeout: float = 1) -> bool:
    """Check for completion and relay a milestone to the Iris supervisor."""
    ready, _ = ray.wait([result], timeout=timeout)
    if ready:
        return True
    if progress is not None:
        try:
            progress.get_nowait()
        except Empty:
            pass
        else:
            Path(os.environ[DRIVER_PROGRESS_PATH_ENV]).touch()
    return False


def workload_result(result: ray.ObjectRef, progress: Queue | None) -> Any:
    """Wait for an evaluation or export while relaying workload progress."""
    if progress is not None:
        while not wait_for_workload(result, progress):
            pass
    return ray.get(result)
