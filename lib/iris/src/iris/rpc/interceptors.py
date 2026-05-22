# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import threading

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from google.protobuf.message import Message
from rigging.timing import Timer

from iris.rpc.errors import connect_error_sanitized, connect_error_with_traceback
from iris.rpc.stats import RpcStatsCollector

logger = logging.getLogger(__name__)

SLOW_RPC_THRESHOLD_MS = 1000


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


class ConcurrencyLimitInterceptor:
    """Caps the number of in-flight RPCs per method.

    Callers exceeding the limit block on a semaphore until a slot frees up;
    Connect/gRPC deadlines already cover the case where a caller gave up
    waiting. Methods absent from ``limits`` are passed through unchanged.

    After the semaphore releases we re-check the Connect deadline: if the
    client's timeout has already elapsed (common under overload, where
    requests queue for seconds), we short-circuit with DEADLINE_EXCEEDED
    instead of running a handler whose result will be discarded. This
    stops stale RPCs from piling more load onto an already-overloaded
    server.
    """

    def __init__(self, limits: dict[str, int]):
        self._semaphores: dict[str, threading.Semaphore] = {
            method: threading.Semaphore(n) for method, n in limits.items()
        }

    def _try_acquire(self, method: str) -> tuple[threading.Semaphore | None, bool]:
        """Return ``(sem, got_slot)`` for ``method``.

        ``got_slot`` is True iff a slot was already free. Callers that get
        ``got_slot=False`` must wait for the slot themselves — synchronously
        on a worker thread, or asynchronously off the event loop.
        """
        sem = self._semaphores.get(method)
        if sem is None:
            return None, True
        return sem, sem.acquire(blocking=False)

    def intercept_unary_sync(self, call_next, request, ctx):
        method = ctx.method().name
        if _deadline_expired(ctx):
            raise _deadline_error(method)
        sem, got_slot = self._try_acquire(method)
        if sem is not None and not got_slot:
            logger.debug("RPC %s blocked waiting for concurrency slot", method)
            sem.acquire()
        try:
            if _deadline_expired(ctx):
                raise _deadline_error(method)
            return call_next(request, ctx)
        finally:
            if sem is not None:
                sem.release()

    async def intercept_unary(self, call_next, request, ctx):
        method = ctx.method().name
        if _deadline_expired(ctx):
            raise _deadline_error(method)
        # Run the blocking ``threading.Semaphore.acquire`` off the asyncio
        # loop: connectrpc's ASGI server invokes this interceptor directly on
        # the event loop, so a synchronous ``sem.acquire()`` here would freeze
        # every other coroutine on the loop (dashboard, health, sibling RPCs)
        # until a slot freed up.
        sem, got_slot = self._try_acquire(method)
        if sem is not None and not got_slot:
            logger.debug("RPC %s blocked waiting for concurrency slot", method)
            await asyncio.to_thread(sem.acquire)
        try:
            if _deadline_expired(ctx):
                raise _deadline_error(method)
            return await call_next(request, ctx)
        finally:
            if sem is not None:
                sem.release()


class RequestTimingInterceptor:
    """Logs method name + duration for every unary RPC.

    Also converts unhandled (non-ConnectError) exceptions into ConnectErrors
    with server-side tracebacks attached as ErrorDetails when include_traceback
    is True. Tracebacks are always logged server-side regardless.

    When ``collector`` is provided, every call — success or failure — is
    recorded for inspection via ``iris.stats.StatsService``.
    """

    def __init__(
        self,
        include_traceback: bool = False,
        *,
        collector: RpcStatsCollector | None = None,
    ):
        self._include_traceback = include_traceback
        self._collector = collector

    def _make_connect_error(self, method: str, exc: Exception) -> ConnectError:
        message = f"RPC {method}: {exc}"
        if self._include_traceback:
            return connect_error_with_traceback(Code.INTERNAL, message, exc=exc)
        return connect_error_sanitized(Code.INTERNAL, message, exc=exc)

    def _record(
        self,
        method: str,
        elapsed_ms: float,
        request,
        ctx,
        error_code: str = "",
        error_message: str = "",
    ) -> None:
        if self._collector is None:
            return
        msg = request if isinstance(request, Message) else None
        self._collector.record(
            method=method,
            duration_ms=elapsed_ms,
            request=msg,
            ctx=ctx,
            error_code=error_code,
            error_message=error_message,
        )

    def intercept_unary_sync(self, call_next, request, ctx):
        method = ctx.method().name
        timer = Timer()
        try:
            response = call_next(request, ctx)
            elapsed = timer.elapsed_ms()
            if elapsed > SLOW_RPC_THRESHOLD_MS:
                logger.warning("RPC %s completed in %dms (slow)", method, elapsed)
            else:
                logger.debug("RPC %s completed in %dms", method, elapsed)
            self._record(method, elapsed, request, ctx)
            return response
        except ConnectError as e:
            elapsed = timer.elapsed_ms()
            logger.warning("RPC %s failed after %dms: %s", method, elapsed, e)
            self._record(method, elapsed, request, ctx, error_code=_code_name(e.code), error_message=str(e))
            raise
        except Exception as e:
            elapsed = timer.elapsed_ms()
            logger.warning("RPC %s failed after %dms: %s", method, elapsed, e, exc_info=True)
            self._record(method, elapsed, request, ctx, error_code="INTERNAL", error_message=str(e))
            raise self._make_connect_error(method, e) from e

    async def intercept_unary(self, call_next, request, ctx):
        method = ctx.method().name
        timer = Timer()
        try:
            response = await call_next(request, ctx)
            elapsed = timer.elapsed_ms()
            if elapsed > SLOW_RPC_THRESHOLD_MS:
                logger.warning("RPC %s completed in %dms (slow)", method, elapsed)
            else:
                logger.debug("RPC %s completed in %dms", method, elapsed)
            self._record(method, elapsed, request, ctx)
            return response
        except ConnectError as e:
            elapsed = timer.elapsed_ms()
            logger.warning("RPC %s failed after %dms: %s", method, elapsed, e)
            self._record(method, elapsed, request, ctx, error_code=_code_name(e.code), error_message=str(e))
            raise
        except Exception as e:
            elapsed = timer.elapsed_ms()
            logger.warning("RPC %s failed after %dms: %s", method, elapsed, e, exc_info=True)
            self._record(method, elapsed, request, ctx, error_code="INTERNAL", error_message=str(e))
            raise self._make_connect_error(method, e) from e


def _code_name(code) -> str:
    """Return a stable string for a Connect code, falling back to repr."""
    name = getattr(code, "name", None)
    if name:
        return name
    return str(code)
