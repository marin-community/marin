# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Connect/RPC interceptors shared across Marin RPC servers."""

from __future__ import annotations

import asyncio
import logging
import threading
from contextlib import asynccontextmanager

from connectrpc.code import Code
from connectrpc.errors import ConnectError

logger = logging.getLogger(__name__)


def _deadline_expired(ctx) -> bool:
    """True if the request's Connect deadline has already elapsed.

    Under overload we can sit behind other in-flight RPCs long enough that
    the client has given up. Running the handler anyway wastes CPU and
    compounds the pile-up — shed the request instead.
    """
    remaining_ms = ctx.timeout_ms()
    return remaining_ms is not None and remaining_ms <= 0


def _deadline_error(method: str) -> ConnectError:
    return ConnectError(Code.DEADLINE_EXCEEDED, f"RPC {method}: deadline exceeded before handler ran")


def _release_if_acquired(fut: asyncio.Future) -> None:
    """Release the semaphore returned by a late-completing ``_try_acquire``.

    Used when the awaiting coroutine was cancelled before the off-loop
    acquire finished. ``threading.Semaphore.acquire`` has no cancellation
    hook, so the worker thread keeps blocking and eventually takes the
    permit — release on the original waiter's behalf so it doesn't leak.
    """
    try:
        sem = fut.result()
    except BaseException:
        return
    if sem is not None:
        sem.release()


class ConcurrencyLimitInterceptor:
    """Caps the number of in-flight RPCs per method.

    Callers exceeding the limit park on a per-method semaphore until a slot
    frees up; Connect/gRPC deadlines already cover the case where a caller
    gave up waiting. Methods absent from ``limits`` are passed through
    unchanged.

    After the semaphore releases we re-check the Connect deadline: if the
    client's timeout has already elapsed (common under overload, where
    requests queue for seconds), we short-circuit with DEADLINE_EXCEEDED
    instead of running a handler whose result will be discarded. This
    stops stale RPCs from piling more load onto an already-overloaded
    server.

    Both ``intercept_unary_sync`` and ``intercept_unary`` are implemented
    against a single ``threading.Semaphore`` per method. An interceptor
    instance is expected to be wired to one transport (WSGI sync OR ASGI
    async) — the cap is not designed to coordinate across both.
    """

    def __init__(self, limits: dict[str, int]):
        self._semaphores: dict[str, threading.Semaphore] = {
            method: threading.Semaphore(n) for method, n in limits.items()
        }

    def _try_acquire(self, method: str) -> threading.Semaphore | None:
        sem = self._semaphores.get(method)
        if sem is None:
            return None
        if not sem.acquire(blocking=False):
            logger.debug("RPC %s blocked waiting for concurrency slot", method)
            sem.acquire()
        return sem

    def intercept_unary_sync(self, call_next, request, ctx):
        method = ctx.method().name
        if _deadline_expired(ctx):
            raise _deadline_error(method)
        sem = self._try_acquire(method)
        try:
            if _deadline_expired(ctx):
                raise _deadline_error(method)
            return call_next(request, ctx)
        finally:
            if sem is not None:
                sem.release()

    @asynccontextmanager
    async def _slot(self, method: str):
        """Acquire a method slot off the loop, cancellation-safe.

        ``asyncio.shield`` keeps the worker thread running past a cancelled
        awaiter; the done callback releases the permit it eventually takes
        so it can't leak. ``threading.Semaphore.acquire`` has no
        cancellation hook, so this dance is the price of sharing one
        semaphore type with the sync path.
        """
        sem: threading.Semaphore | None = None
        acquire_task = asyncio.ensure_future(asyncio.to_thread(self._try_acquire, method))
        try:
            sem = await asyncio.shield(acquire_task)
            yield
        except asyncio.CancelledError:
            if sem is None:
                acquire_task.add_done_callback(_release_if_acquired)
            raise
        finally:
            if sem is not None:
                sem.release()

    async def intercept_unary(self, call_next, request, ctx):
        method = ctx.method().name
        if _deadline_expired(ctx):
            raise _deadline_error(method)
        async with self._slot(method):
            if _deadline_expired(ctx):
                raise _deadline_error(method)
            return await call_next(request, ctx)
