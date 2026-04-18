# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import logging

from connectrpc.code import Code
from connectrpc.errors import ConnectError

from iris.rpc.errors import connect_error_with_traceback
from iris.time_utils import Timer

logger = logging.getLogger(__name__)

_SLOW_RPC_THRESHOLD_MS = 1000


class RequestTimingInterceptor:
    """Logs method name + duration for every unary RPC.

    Also converts unhandled (non-ConnectError) exceptions into ConnectErrors
    with full server-side tracebacks attached as ErrorDetails, so clients can
    see exactly where the failure occurred.
    """

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
        except ConnectError as e:
            logger.warning("RPC %s failed after %dms: %s", method, timer.elapsed_ms(), e)
            raise
        except Exception as e:
            logger.warning("RPC %s failed after %dms: %s", method, timer.elapsed_ms(), e, exc_info=True)
            raise connect_error_with_traceback(Code.INTERNAL, f"RPC {method}: {e}", exc=e) from e

    async def intercept_unary(self, call_next, request, ctx):
        method = ctx.method().name
        timer = Timer()
        try:
            response = await call_next(request, ctx)
            elapsed = timer.elapsed_ms()
            if elapsed > _SLOW_RPC_THRESHOLD_MS:
                logger.warning("RPC %s completed in %dms (slow)", method, elapsed)
            else:
                logger.debug("RPC %s completed in %dms", method, elapsed)
            return response
        except ConnectError as e:
            logger.warning("RPC %s failed after %dms: %s", method, timer.elapsed_ms(), e)
            raise
        except Exception as e:
            logger.warning("RPC %s failed after %dms: %s", method, timer.elapsed_ms(), e, exc_info=True)
            raise connect_error_with_traceback(Code.INTERNAL, f"RPC {method}: {e}", exc=e) from e
