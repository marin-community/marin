# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Keep bounded inference admission separate from active trial deadlines."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


class InferenceAdmissionTimeout(TimeoutError):
    """The separate inference queue allowance expired before slot ownership."""


@dataclass
class _Timer:
    timeout: asyncio.Timeout
    first_schedule_budget: float | None
    remaining: float | None = None
    admitted: bool = False


_current: ContextVar[InferenceClock | None] = ContextVar("inference_clock", default=None)


class InferenceClock:
    """One trial's execution deadlines and cumulative queue allowance.

    The pre-inference attempt deadline still bounds setup. On first scheduling,
    registered execution budgets start afresh. Later queue waits pause their
    remaining budgets. Queue allowance is cumulative across requests and retries.
    """

    def __init__(self, queue_timeout: float | None):
        self.queue_remaining = queue_timeout
        self.queue_wait = 0.0
        self.queue_timeouts = 0
        self.generation = 0.0
        self.scheduled_requests = 0
        self.timers: list[_Timer] = []
        self._pending: tuple[asyncio.Timeout, float] | None = None

    @contextmanager
    def scope(self) -> Iterator[None]:
        token = _current.set(self)
        try:
            yield
        finally:
            _current.reset(token)

    @property
    def waiting(self) -> bool:
        return self._pending is not None

    @asynccontextmanager
    async def queue(self) -> AsyncIterator[None]:
        if self.waiting:
            raise RuntimeError("One trial cannot await two inference slots concurrently")
        loop = asyncio.get_running_loop()
        started = loop.time()
        for timer in self.timers:
            deadline = timer.timeout.when()
            timer.remaining = None if deadline is None else max(0.0, deadline - started)
            timer.timeout.reschedule(None)
        guard = asyncio.timeout(self.queue_remaining)
        pending = (guard, started)
        self._pending = pending
        try:
            async with guard:
                yield
        except TimeoutError as error:
            if not guard.expired():
                raise
            self.queue_timeouts += 1
            raise InferenceAdmissionTimeout("Inference queue allowance expired before slot ownership") from error
        finally:
            if self._pending is pending:
                self._resume(admitted=False)

    def _resume(self, *, admitted: bool) -> None:
        assert self._pending is not None
        guard, started = self._pending
        self._pending = None
        loop = asyncio.get_running_loop()
        elapsed = loop.time() - started
        self.queue_wait += elapsed
        if self.queue_remaining is not None:
            self.queue_remaining = max(0.0, self.queue_remaining - elapsed)
        if admitted and not guard.expired():
            guard.reschedule(None)
        for timer in self.timers:
            budget = timer.remaining
            if not timer.admitted and admitted:
                budget = timer.first_schedule_budget
                timer.admitted = True
            timer.timeout.reschedule(None if budget is None else loop.time() + budget)

    def scheduled(self) -> None:
        """Accept only a real engine SCHEDULED/first-token notification."""
        if not self.waiting:
            raise RuntimeError("Inference scheduling notification has no pending admission")
        self.scheduled_requests += 1
        self._resume(admitted=True)

    def execution_remaining(self) -> float | None:
        """Return the nearest active execution budget after admission."""
        deadlines = [deadline for timer in self.timers if (deadline := timer.timeout.when()) is not None]
        if not deadlines:
            return None
        return max(0.0, min(deadlines) - asyncio.get_running_loop().time())

    def metrics(self) -> dict[str, float]:
        return {
            "timing/inference_queue_wait": self.queue_wait,
            "timing/inference_generation": self.generation,
            "inference/queue_timeouts": float(self.queue_timeouts),
            "inference/scheduled_requests": float(self.scheduled_requests),
        }


def current_inference_clock() -> InferenceClock | None:
    return _current.get()


@asynccontextmanager
async def execution_timeout(
    seconds: float | None, *, first_schedule_budget: float | None = None
) -> AsyncIterator[asyncio.Timeout]:
    """A setup watchdog that becomes an active budget at first engine scheduling."""
    timeout = asyncio.timeout(seconds)
    clock = _current.get()
    timer = _Timer(timeout, seconds if first_schedule_budget is None else first_schedule_budget)
    async with timeout:
        if clock is not None:
            clock.timers.append(timer)
        try:
            yield timeout
        finally:
            if clock is not None:
                clock.timers.remove(timer)


async def wait_execution(task: asyncio.Task, seconds: float | None) -> tuple[set, set]:
    """Wait without cancelling the phase task when its active budget expires."""
    deadline = None
    try:
        async with execution_timeout(seconds) as deadline:
            return await asyncio.wait({task})
    except TimeoutError:
        if deadline is None or not deadline.expired():
            raise
        return set(), {task}


def remote_execution_watchdog(seconds: float | None) -> float | None:
    """Bound setup, cumulative queue wait and active remote execution."""
    clock = _current.get()
    if seconds is None or clock is None:
        return seconds
    if clock.queue_remaining is None:
        raise ValueError("A detached agent needs a bounded inference queue allowance")
    # Before admission the local phase timer bounds setup; admission grants its full active budget.
    setup_allowance = active_budget = seconds
    return setup_allowance + clock.queue_remaining + active_budget
