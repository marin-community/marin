"""Carry completed workload milestones from a Ray entrypoint to its Iris driver."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
import time

import ray
from ray.util.queue import Empty, Full, Queue

_REPORT_INTERVAL = 5
# Tasks spawned by the entrypoint's asyncio loop inherit this binding. Workload
# code also runs standalone, where there is no supervising driver.
_reporter: ContextVar[Callable[[], None] | None] = ContextVar("workload_progress", default=None)


def report_workload_progress() -> None:
    """Record a completed phase or unit of work, independent of progress display."""
    if (report := _reporter.get()) is not None:
        report()


@contextmanager
def report_to_driver(progress: Queue | None) -> Iterator[None]:
    """Route milestones to the driver for this workload. None disables reporting."""
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


@dataclass(frozen=True)
class DriverProgress:
    queue: Queue
    path: Path


@contextmanager
def workload_progress_queue(path: Path | None) -> Iterator[DriverProgress | None]:
    """Open the Ray progress channel for a supervised workload."""
    if path is None:
        yield None
        return
    queue = Queue(maxsize=1, actor_options={"num_cpus": 0})
    try:
        yield DriverProgress(queue, path)
    finally:
        queue.shutdown(force=True)


def poll_workload(result: ray.ObjectRef, progress: DriverProgress | None) -> bool:
    """Check for completion and relay a milestone to the Iris supervisor."""
    ready, _ = ray.wait([result], timeout=1)
    if ready:
        return True
    if progress is not None:
        try:
            progress.queue.get_nowait()
        except Empty:
            pass
        else:
            progress.path.touch()
    return False


def workload_result[T](result: ray.ObjectRef[T], progress: DriverProgress | None) -> T:
    """Wait for an evaluation or export while relaying workload progress."""
    if progress is not None:
        while not poll_workload(result, progress):
            pass
    return ray.get(result)
