# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import threading
from dataclasses import dataclass
from unittest.mock import Mock

import pytest
import rigging.rpc
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from rigging.rpc import ConcurrencyLimitInterceptor, _release_if_acquired


@dataclass(frozen=True)
class FakeMethodInfo:
    name: str


def _make_ctx(method_name: str, timeout_ms: int | None = None):
    ctx = Mock()
    ctx.method.return_value = FakeMethodInfo(name=method_name)
    ctx.timeout_ms.return_value = timeout_ms
    return ctx


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
    await asyncio.sleep(0)

    expired_ctx = Mock()
    expired_ctx.method.return_value = FakeMethodInfo(name="FetchLogs")
    expired_ctx.timeout_ms.side_effect = [5000, 0]

    late_called = False

    async def late_handler(req, ctx):
        nonlocal late_called
        late_called = True
        return "ok"

    late = asyncio.create_task(interceptor.intercept_unary(late_handler, "req", expired_ctx))
    await asyncio.sleep(0)
    release.set()

    assert await holder == "ok"
    with pytest.raises(ConnectError) as exc_info:
        await late
    assert exc_info.value.code == Code.DEADLINE_EXCEEDED
    assert not late_called


@pytest.mark.asyncio
async def test_concurrency_limit_async_does_not_block_event_loop():
    """A saturated semaphore must not freeze the asyncio loop."""
    interceptor = ConcurrencyLimitInterceptor({"FetchLogs": 1})
    holder_release = asyncio.Event()

    async def held_handler(req, ctx):
        await holder_release.wait()
        return "ok"

    async def quick_handler(req, ctx):
        return "fast"

    holder = asyncio.create_task(interceptor.intercept_unary(held_handler, "req", _make_ctx("FetchLogs")))
    await asyncio.sleep(0)
    waiter = asyncio.create_task(interceptor.intercept_unary(quick_handler, "req", _make_ctx("FetchLogs")))

    ticked = False
    for _ in range(10):
        await asyncio.sleep(0.01)
        ticked = True
    assert ticked

    holder_release.set()
    assert await holder == "ok"
    assert await waiter == "fast"


@pytest.mark.asyncio
async def test_concurrency_limit_async_cancelled_waiter_does_not_leak_slot(monkeypatch):
    """Cancelling a queued async caller must not leak a permit.

    Worker threads keep blocking on ``threading.Semaphore.acquire`` past
    the awaiter's cancellation. The shield + late-release callback is what
    prevents the permit from leaking once the thread finally takes it.
    """
    limit = 2
    interceptor = ConcurrencyLimitInterceptor({"FetchLogs": limit})
    holder_release = threading.Event()
    holders_started = 0

    # Count late-releases so the test can wait for every cancelled waiter's worker
    # thread to drain before returning (see the wait at the end).
    drained = 0

    def counting_release(fut):
        nonlocal drained
        _release_if_acquired(fut)
        drained += 1

    monkeypatch.setattr(rigging.rpc, "_release_if_acquired", counting_release)

    async def held_handler(req, ctx):
        nonlocal holders_started
        holders_started += 1
        await asyncio.to_thread(holder_release.wait)
        return "ok"

    async def never_runs(req, ctx):  # pragma: no cover -- must not execute
        raise AssertionError("waiter handler ran despite cancellation")

    holders = [
        asyncio.create_task(interceptor.intercept_unary(held_handler, "req", _make_ctx("FetchLogs")))
        for _ in range(limit)
    ]
    # Wait until both holders are inside the handler (i.e. own a permit).
    for _ in range(50):
        await asyncio.sleep(0.01)
        if holders_started == limit:
            break
    assert holders_started == limit

    waiters = [
        asyncio.create_task(interceptor.intercept_unary(never_runs, "req", _make_ctx("FetchLogs"))) for _ in range(5)
    ]
    await asyncio.sleep(0.05)  # let the waiter threads enter sem.acquire()
    for w in waiters:
        w.cancel()
    for w in waiters:
        with pytest.raises(asyncio.CancelledError):
            await w

    holder_release.set()
    for h in holders:
        assert await h == "ok"

    async def quick(req, ctx):
        return "fast"

    # If a permit leaked, this gather would hang. wait_for gives us a clean
    # failure instead of the worker-level timeout.
    results = await asyncio.wait_for(
        asyncio.gather(*(interceptor.intercept_unary(quick, "req", _make_ctx("FetchLogs")) for _ in range(limit))),
        timeout=5.0,
    )
    assert results == ["fast"] * limit

    # Drain every cancelled waiter before the test returns. Each still has a worker
    # thread parked in sem.acquire(), and its late-release callback fires only while
    # the event loop runs: a thread that takes a permit after loop teardown can never
    # release it, so the remaining threads park forever and the loop's executor
    # shutdown joins them forever. On runners slow enough that the drain outlives the
    # test body this wedged teardown deterministically (10s pytest-timeout on CI).
    async with asyncio.timeout(5):
        while drained < len(waiters):
            await asyncio.sleep(0.01)


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
    assert await asyncio.wait_for(interceptor.intercept_unary(ok, "request", ctx), timeout=1.0) == "ok"
