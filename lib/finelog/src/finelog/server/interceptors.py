# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generic Connect/RPC interceptors used by the finelog server."""

from __future__ import annotations

import logging

from rigging.log_setup import slow_log

logger = logging.getLogger(__name__)

# Default threshold above which a query/write_rows handler is reported as
# slow. Picked to be well above the durable-write floor (one bg flush cycle
# ~= ``DEFAULT_FLUSH_INTERVAL_SEC``) so a normal WriteRows that waits for
# its L0 segment doesn't spam warnings.
DEFAULT_SLOW_RPC_THRESHOLD_MS = 7000


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
