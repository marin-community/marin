# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Connect/RPC interceptors shared across Marin RPC servers."""

from __future__ import annotations

import asyncio
import logging

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


class ConcurrencyLimitInterceptor:
    """Caps the number of in-flight RPCs per method on an ASGI connectrpc mount.

    Callers exceeding the limit await the semaphore until a slot frees up;
    Connect/gRPC deadlines already cover the case where a caller gave up
    waiting. Methods absent from ``limits`` are passed through unchanged.

    After the semaphore releases we re-check the Connect deadline: if the
    client's timeout has already elapsed (common under overload, where
    requests queue for seconds), we short-circuit with DEADLINE_EXCEEDED
    instead of running a handler whose result will be discarded. This
    stops stale RPCs from piling more load onto an already-overloaded
    server.

    Only the async path is implemented; ``asyncio.Semaphore`` cannot serve
    a WSGI mount. If a sync caller ever appears, add ``intercept_unary_sync``
    with a parallel ``threading.Semaphore`` — the two primitives do not
    share state.
    """

    def __init__(self, limits: dict[str, int]):
        self._semaphores: dict[str, asyncio.Semaphore] = {method: asyncio.Semaphore(n) for method, n in limits.items()}

    async def intercept_unary(self, call_next, request, ctx):
        method = ctx.method().name
        if _deadline_expired(ctx):
            raise _deadline_error(method)
        sem = self._semaphores.get(method)
        if sem is None:
            return await call_next(request, ctx)
        if sem.locked():
            logger.debug("RPC %s blocked waiting for concurrency slot", method)
        # ``async with`` yields to the loop on contention and releases on any
        # exit, including ``CancelledError`` raised mid-wait.
        async with sem:
            if _deadline_expired(ctx):
                raise _deadline_error(method)
            return await call_next(request, ctx)
