# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import logging
import threading

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from google.protobuf.message import Message

from iris.rpc.errors import connect_error_sanitized, connect_error_with_traceback
from iris.rpc.stats import RpcStatsCollector
from rigging.timing import Timer

logger = logging.getLogger(__name__)

SLOW_RPC_THRESHOLD_MS = 1000


class ConcurrencyLimitInterceptor:
    """Caps the number of in-flight RPCs per method.

    Callers exceeding the limit block on a semaphore until a slot frees up;
    Connect/gRPC deadlines already cover the case where a caller gave up
    waiting. Methods absent from ``limits`` are passed through unchanged.
    """

    def __init__(self, limits: dict[str, int]):
        self._semaphores: dict[str, threading.Semaphore] = {
            method: threading.Semaphore(n) for method, n in limits.items()
        }

    def _acquire(self, method: str) -> threading.Semaphore | None:
        sem = self._semaphores.get(method)
        if sem is None:
            return None
        if not sem.acquire(blocking=False):
            logger.debug("RPC %s blocked waiting for concurrency slot", method)
            sem.acquire()
        return sem

    def intercept_unary_sync(self, call_next, request, ctx):
        method = ctx.method().name
        sem = self._acquire(method)
        try:
            return call_next(request, ctx)
        finally:
            if sem is not None:
                sem.release()

    async def intercept_unary(self, call_next, request, ctx):
        method = ctx.method().name
        sem = self._acquire(method)
        try:
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
