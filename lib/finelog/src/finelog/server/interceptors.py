# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generic Connect/RPC interceptors used by the finelog server."""

from __future__ import annotations

import asyncio
import logging
import threading

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from rigging.log_setup import slow_log

logger = logging.getLogger(__name__)

# Default threshold above which a query/write_rows handler is reported as
# slow. Picked to be well above the durable-write floor (one bg flush cycle
# ~= ``DEFAULT_FLUSH_INTERVAL_SEC``) so a normal WriteRows that waits for
# its L0 segment doesn't spam warnings.
DEFAULT_SLOW_RPC_THRESHOLD_MS = 7000


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
    instead of running a handler whose result will be discarded.
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
        if _deadline_expired(ctx):
            raise _deadline_error(method)
        sem = self._acquire(method)
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
        # ``_acquire`` blocks on a threading.Semaphore; hop off the loop so
        # other in-flight handlers can run while we wait for a slot.
        sem = await asyncio.to_thread(self._acquire, method)
        try:
            if _deadline_expired(ctx):
                raise _deadline_error(method)
            return await call_next(request, ctx)
        finally:
            if sem is not None:
                sem.release()


class SlowRpcInterceptor:
    """Log a WARNING when a unary RPC exceeds a per-method threshold.

    Delegates the actual timing + emit to :func:`rigging.log_setup.slow_log`
    so the threshold-and-log-once pattern lives in one place across the
    codebase. ``thresholds`` maps method name (``ctx.method().name``) to a
    ms cap; methods absent from the map use ``default_threshold_ms``.
    Setting a method's threshold to ``0`` suppresses the warning for that
    method entirely.
    """

    def __init__(
        self,
        thresholds: dict[str, int] | None = None,
        *,
        default_threshold_ms: int = DEFAULT_SLOW_RPC_THRESHOLD_MS,
    ) -> None:
        self._thresholds = dict(thresholds or {})
        self._default = default_threshold_ms

    def _threshold(self, method: str) -> int:
        return self._thresholds.get(method, self._default)

    def intercept_unary_sync(self, call_next, request, ctx):
        method = ctx.method().name
        threshold = self._threshold(method)
        if threshold <= 0:
            return call_next(request, ctx)
        with slow_log(logger, f"RPC {method}", threshold_ms=threshold):
            return call_next(request, ctx)

    async def intercept_unary(self, call_next, request, ctx):
        method = ctx.method().name
        threshold = self._threshold(method)
        if threshold <= 0:
            return await call_next(request, ctx)
        with slow_log(logger, f"RPC {method}", threshold_ms=threshold):
            return await call_next(request, ctx)
