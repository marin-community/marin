# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Keep bounded inference admission separate from active trial deadlines."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

INFERENCE_QUEUE_WAIT_METRIC = "timing/inference_queue_wait"
INFERENCE_GENERATION_METRIC = "timing/inference_generation"
INFERENCE_QUEUE_TIMEOUTS_METRIC = "inference/queue_timeouts"
INFERENCE_SCHEDULED_REQUESTS_METRIC = "inference/scheduled_requests"
INFERENCE_METRIC_KEYS = (
    INFERENCE_QUEUE_WAIT_METRIC,
    INFERENCE_GENERATION_METRIC,
    INFERENCE_QUEUE_TIMEOUTS_METRIC,
    INFERENCE_SCHEDULED_REQUESTS_METRIC,
)
PARTIAL_DEADLINE_STOP_REASON = "partial_deadline"


class InferenceAdmissionTimeout(TimeoutError):
    """The separate inference queue allowance expired before slot ownership."""


@dataclass
class _Timer:
    timeout: asyncio.Timeout
    first_schedule_budget: float | None
    remaining: float | None = None
    admitted: bool = False


# Task-local scope; detached HTTP callback tasks restore the registered trial clock explicitly.
_current: ContextVar[InferenceClock | None] = ContextVar("inference_clock", default=None)


class InferenceClock:
    """One trial's execution deadlines and cumulative queue allowance.

    The pre-inference attempt deadline still bounds setup. On first scheduling,
    registered execution budgets start afresh. Later queue waits pause their
    remaining budgets. Queue allowance is cumulative across requests in this attempt.
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
        """Release pending admission after the caller observes engine scheduling."""
        if not self.waiting:
            raise RuntimeError("Inference scheduling notification has no pending admission")
        self.scheduled_requests += 1
        self._resume(admitted=True)

    def nearest_execution_timeout(self, completed_call: asyncio.Timeout | None = None) -> asyncio.Timeout | None:
        """Return the governing timer, including a call just removed from scope."""
        active = [timer.timeout for timer in self.timers if timer.timeout.when() is not None]
        if completed_call is not None and completed_call.when() is not None:
            active.append(completed_call)
        if not active:
            return None
        return min(active, key=lambda timeout: timeout.when())

    def execution_remaining(self) -> float | None:
        """Return the nearest active execution budget after admission."""
        nearest = self.nearest_execution_timeout()
        if nearest is None:
            return None
        deadline = nearest.when()
        assert deadline is not None
        return max(0.0, deadline - asyncio.get_running_loop().time())

    def metrics(self) -> dict[str, float]:
        return {
            INFERENCE_QUEUE_WAIT_METRIC: self.queue_wait,
            INFERENCE_GENERATION_METRIC: self.generation,
            INFERENCE_QUEUE_TIMEOUTS_METRIC: float(self.queue_timeouts),
            INFERENCE_SCHEDULED_REQUESTS_METRIC: float(self.scheduled_requests),
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


def remote_execution_watchdog(seconds: float | None) -> float | None:
    """Allow queue time in a detached command's hard safety watchdog."""
    clock = _current.get()
    if seconds is None or clock is None:
        return seconds
    if clock.queue_remaining is None:
        raise ValueError("A detached agent needs a bounded inference queue allowance")
    return 2 * seconds + clock.queue_remaining
