# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from dataclasses import dataclass
from unittest.mock import Mock

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from iris.rpc.errors import extract_error_details
from iris.rpc.interceptors import ConcurrencyLimitInterceptor, RequestTimingInterceptor


@dataclass(frozen=True)
class FakeMethodInfo:
    name: str


def _make_ctx(method_name: str, timeout_ms: int | None = None):
    ctx = Mock()
    ctx.method.return_value = FakeMethodInfo(name=method_name)
    ctx.timeout_ms.return_value = timeout_ms
    return ctx


def test_interceptor_passes_through_response():
    interceptor = RequestTimingInterceptor()
    ctx = _make_ctx("FetchLogs")
    result = interceptor.intercept_unary_sync(lambda req, ctx: "ok", "request", ctx)
    assert result == "ok"


def test_interceptor_wraps_exceptions_as_connect_error_with_traceback():
    interceptor = RequestTimingInterceptor(include_traceback=True)
    ctx = _make_ctx("LaunchJob")

    def failing_handler(req, ctx):
        raise ValueError("boom")

    with pytest.raises(ConnectError, match="boom") as exc_info:
        interceptor.intercept_unary_sync(failing_handler, "request", ctx)

    error = exc_info.value
    assert error.__cause__ is not None
    assert isinstance(error.__cause__, ValueError)
    # Verify traceback details are attached
    assert len(error.details) > 0
    details = extract_error_details(error)
    assert details is not None
    assert details.traceback != ""


def test_interceptor_sanitized_by_default():
    interceptor = RequestTimingInterceptor()
    ctx = _make_ctx("LaunchJob")

    def failing_handler(req, ctx):
        raise ValueError("boom")

    with pytest.raises(ConnectError, match="boom") as exc_info:
        interceptor.intercept_unary_sync(failing_handler, "request", ctx)

    error = exc_info.value
    assert error.__cause__ is not None
    assert isinstance(error.__cause__, ValueError)
    details = extract_error_details(error)
    assert details is not None
    assert details.traceback == ""
    assert details.exception_type.endswith("ValueError")
    assert details.message != ""


def test_interceptor_passes_through_connect_errors():
    interceptor = RequestTimingInterceptor()
    ctx = _make_ctx("LaunchJob")

    original = ConnectError(code=5, message="not found")

    def failing_handler(req, ctx):
        raise original

    with pytest.raises(ConnectError) as exc_info:
        interceptor.intercept_unary_sync(failing_handler, "request", ctx)

    assert exc_info.value is original


def test_interceptor_logs_traceback_regardless_of_flag(caplog):
    """Server-side logging always includes exc_info regardless of include_traceback."""
    interceptor = RequestTimingInterceptor(include_traceback=False)
    ctx = _make_ctx("Explode")

    def failing_handler(req, ctx):
        raise RuntimeError("kaboom")

    with caplog.at_level(logging.WARNING, logger="iris.rpc.interceptors"):
        with pytest.raises(ConnectError):
            interceptor.intercept_unary_sync(failing_handler, "request", ctx)

    assert any("kaboom" in r.message and r.exc_info is not None for r in caplog.records)


@pytest.mark.asyncio
async def test_concurrency_limit_passes_through_unlimited_methods():
    interceptor = ConcurrencyLimitInterceptor({"FetchLogs": 1})

    async def handler(req, ctx):
        return "ok"

    result = await interceptor.intercept_unary(handler, "request", _make_ctx("LaunchJob"))
    assert result == "ok"


@pytest.mark.asyncio
async def test_concurrency_limit_caps_in_flight_calls():
    limit = 3
    interceptor = ConcurrencyLimitInterceptor({"FetchLogs": limit})
    release = asyncio.Event()
    in_flight = 0
    peak = 0

    async def handler(req, ctx):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        try:
            await release.wait()
            return "ok"
        finally:
            in_flight -= 1

    num_callers = limit + 3
    callers = [
        asyncio.create_task(interceptor.intercept_unary(handler, "request", _make_ctx("FetchLogs")))
        for _ in range(num_callers)
    ]

    # Let the loop run until the first wave saturates the semaphore.
    for _ in range(20):
        await asyncio.sleep(0)
        if in_flight >= limit:
            break
    assert in_flight == limit

    release.set()
    results = await asyncio.gather(*callers)
    assert results == ["ok"] * num_callers
    assert peak == limit


@pytest.mark.asyncio
async def test_concurrency_limit_sheds_when_deadline_expired_before_acquire():
    interceptor = ConcurrencyLimitInterceptor({"FetchLogs": 1})
    called = False

    async def handler(req, ctx):
        nonlocal called
        called = True
        return "ok"

    with pytest.raises(ConnectError) as exc_info:
        await interceptor.intercept_unary(handler, "request", _make_ctx("FetchLogs", timeout_ms=0))
    assert exc_info.value.code == Code.DEADLINE_EXCEEDED
    assert not called


@pytest.mark.asyncio
async def test_concurrency_limit_sheds_when_deadline_expires_during_wait():
    """Deadline check after semaphore acquire: caller queues, then deadline lapses."""
    interceptor = ConcurrencyLimitInterceptor({"FetchLogs": 1})
    release = asyncio.Event()

    async def blocking_handler(req, ctx):
        await release.wait()
        return "ok"

    holder = asyncio.create_task(
        interceptor.intercept_unary(blocking_handler, "req", _make_ctx("FetchLogs", timeout_ms=5000))
    )
    await asyncio.sleep(0)  # let the holder grab the slot

    # A ctx whose timeout_ms flips to 0 after the semaphore finally releases.
    expired_ctx = Mock()
    expired_ctx.method.return_value = FakeMethodInfo(name="FetchLogs")
    expired_ctx.timeout_ms.side_effect = [5000, 0]

    late_called = False

    async def late_handler(req, ctx):
        nonlocal late_called
        late_called = True
        return "ok"

    late = asyncio.create_task(interceptor.intercept_unary(late_handler, "req", expired_ctx))
    await asyncio.sleep(0)  # let the late caller queue on the semaphore
    release.set()

    assert await holder == "ok"
    with pytest.raises(ConnectError) as exc_info:
        await late
    assert exc_info.value.code == Code.DEADLINE_EXCEEDED
    assert not late_called


@pytest.mark.asyncio
async def test_concurrency_limit_async_does_not_block_event_loop():
    """Regression: a saturated semaphore must not freeze the asyncio loop.

    The async interceptor is invoked directly on the event loop (connectrpc
    ASGI server). If the threading semaphore is acquired synchronously, a
    waiting RPC parks the loop and starves every other coroutine until a
    slot frees up. This test asserts the loop keeps making progress while
    one async caller is queued behind a held slot.
    """
    interceptor = ConcurrencyLimitInterceptor({"FetchLogs": 1})
    holder_release = asyncio.Event()

    async def held_handler(req, ctx):
        await holder_release.wait()
        return "ok"

    async def quick_handler(req, ctx):
        return "fast"

    holder = asyncio.create_task(interceptor.intercept_unary(held_handler, "req", _make_ctx("FetchLogs")))
    # Yield so the holder grabs the only slot before the waiter arrives.
    await asyncio.sleep(0)
    waiter = asyncio.create_task(interceptor.intercept_unary(quick_handler, "req", _make_ctx("FetchLogs")))

    # While the waiter is parked on the semaphore, an unrelated coroutine
    # must still get scheduled. With the buggy blocking acquire, this
    # ``asyncio.sleep`` never fires because the loop is wedged.
    ticked = False
    for _ in range(10):
        await asyncio.sleep(0.01)
        ticked = True
    assert ticked

    holder_release.set()
    assert await holder == "ok"
    assert await waiter == "fast"


@pytest.mark.asyncio
async def test_concurrency_limit_async_cancelled_waiter_does_not_leak_slot():
    """Regression: cancelling a queued async caller must not leak a permit.

    Prior implementation parked the wait on ``asyncio.to_thread(sem.acquire)``
    against a ``threading.Semaphore``. Cancellation raised ``CancelledError``
    before the ``try/finally`` ran, while the off-loop thread kept blocking
    and eventually acquired the permit — a permanent leak. ``asyncio.Semaphore``
    + ``async with`` is cancellation-safe.
    """
    limit = 2
    interceptor = ConcurrencyLimitInterceptor({"FetchLogs": limit})
    holder_release = asyncio.Event()

    async def held_handler(req, ctx):
        await holder_release.wait()
        return "ok"

    async def never_runs(req, ctx):  # pragma: no cover -- must not execute
        raise AssertionError("waiter handler ran despite cancellation")

    holders = [
        asyncio.create_task(interceptor.intercept_unary(held_handler, "req", _make_ctx("FetchLogs")))
        for _ in range(limit)
    ]
    await asyncio.sleep(0)  # let holders grab both slots

    # Queue several waiters, then cancel them all before any slot frees.
    waiters = [
        asyncio.create_task(interceptor.intercept_unary(never_runs, "req", _make_ctx("FetchLogs"))) for _ in range(5)
    ]
    await asyncio.sleep(0)
    for w in waiters:
        w.cancel()
    for w in waiters:
        with pytest.raises(asyncio.CancelledError):
            await w

    # Free the holders. A leaked permit would mean the next caller blocks
    # forever waiting on a slot that the cancelled waiter "stole".
    holder_release.set()
    for h in holders:
        assert await h == "ok"

    async def quick(req, ctx):
        return "fast"

    # All `limit` slots must be available again.
    results = await asyncio.gather(
        *(interceptor.intercept_unary(quick, "req", _make_ctx("FetchLogs")) for _ in range(limit))
    )
    assert results == ["fast"] * limit


@pytest.mark.asyncio
async def test_concurrency_limit_releases_slot_on_exception():
    interceptor = ConcurrencyLimitInterceptor({"FetchLogs": 1})

    async def boom(req, ctx):
        raise ValueError("nope")

    async def ok(req, ctx):
        return "ok"

    ctx = _make_ctx("FetchLogs")
    for _ in range(3):
        with pytest.raises(ValueError):
            await interceptor.intercept_unary(boom, "request", ctx)
    # If the slot leaked, the fourth call would deadlock — a successful
    # passthrough proves the semaphore was released on each exception.
    assert await asyncio.wait_for(interceptor.intercept_unary(ok, "request", ctx), timeout=1.0) == "ok"


# --- Stats collector integration -------------------------------------------


class _RecordingCollector:
    """Stub that captures every record() call for assertions."""

    def __init__(self):
        self.calls: list[dict] = []

    def record(self, *, method, duration_ms, request, ctx, error_code="", error_message=""):
        self.calls.append(
            {
                "method": method,
                "duration_ms": duration_ms,
                "error_code": error_code,
                "error_message": error_message,
            }
        )


def test_interceptor_records_successful_call():
    collector = _RecordingCollector()
    interceptor = RequestTimingInterceptor(collector=collector)
    ctx = _make_ctx("ListJobs")

    result = interceptor.intercept_unary_sync(lambda req, ctx: "ok", "request", ctx)

    assert result == "ok"
    assert len(collector.calls) == 1
    call = collector.calls[0]
    assert call["method"] == "ListJobs"
    assert call["error_code"] == ""
    assert call["duration_ms"] >= 0


def test_interceptor_records_connect_error_with_code_name():
    collector = _RecordingCollector()
    interceptor = RequestTimingInterceptor(collector=collector)
    ctx = _make_ctx("ListJobs")

    def boom(req, ctx):
        raise ConnectError(code=Code.NOT_FOUND, message="not found")

    with pytest.raises(ConnectError):
        interceptor.intercept_unary_sync(boom, "request", ctx)

    assert len(collector.calls) == 1
    assert collector.calls[0]["error_code"] == "NOT_FOUND"
    assert "not found" in collector.calls[0]["error_message"]


def test_interceptor_records_unhandled_exception_as_internal():
    collector = _RecordingCollector()
    interceptor = RequestTimingInterceptor(collector=collector)
    ctx = _make_ctx("LaunchJob")

    def boom(req, ctx):
        raise RuntimeError("kaboom")

    with pytest.raises(ConnectError):
        interceptor.intercept_unary_sync(boom, "request", ctx)

    assert len(collector.calls) == 1
    assert collector.calls[0]["error_code"] == "INTERNAL"
    assert collector.calls[0]["error_message"] == "kaboom"
