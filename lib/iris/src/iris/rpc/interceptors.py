# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import logging

from iris.time_utils import Timer

logger = logging.getLogger(__name__)

_SLOW_RPC_THRESHOLD_MS = 1000


class RequestTimingInterceptor:
    """Logs method name + duration for every unary RPC."""

    def intercept_unary_sync(self, call_next, request, ctx):
        method = ctx.method().name
        timer = Timer()
        try:
            response = call_next(request, ctx)
            elapsed = timer.elapsed_ms()
            if elapsed > _SLOW_RPC_THRESHOLD_MS:
                logger.warning("RPC %s completed in %dms (slow)", method, elapsed)
            else:
                logger.debug("RPC %s completed in %dms", method, elapsed)
            return response
        except Exception as e:
            logger.warning("RPC %s failed after %dms: %s", method, timer.elapsed_ms(), e)
            raise
