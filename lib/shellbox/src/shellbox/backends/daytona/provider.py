# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Retry provider reads and idempotent named lifecycle operations."""

import logging
from asyncio import get_running_loop, sleep
from collections.abc import Awaitable, Callable

from daytona import DaytonaError
from daytona.common.errors import DaytonaConnectionError, DaytonaTimeoutError
from rigging.timing import ExponentialBackoff

logger = logging.getLogger(__name__)
PROVIDER_RETRY_MAX_DELAY = 120


async def retry_provider_call[T](
    operation: str, request: Callable[[], Awaitable[T]], *, deadline: float | None = None
) -> T:
    """Retry transient failures until success or the caller cancels the operation.

    Use only for reads or creates whose stable provider name prevents duplicates.
    Command execution must not be replayed after an ambiguous response. An
    absolute loop-time deadline preserves the last SDK error when another
    retry would exceed the startup budget.
    """
    backoff = ExponentialBackoff(initial=1, maximum=PROVIDER_RETRY_MAX_DELAY, factor=2, jitter=0.5)
    attempt = 0
    while True:
        try:
            return await request()
        except DaytonaError as error:
            status = error.status_code
            transient = (
                isinstance(error, (DaytonaConnectionError, DaytonaTimeoutError))
                or status in (408, 429)
                or (status is not None and 500 <= status < 600)
            )
            if not transient:
                raise
            attempt += 1
            retry_after = max(
                (
                    int(value)
                    for name, value in error.headers.items()
                    if name.lower().startswith("retry-after") and str(value).isdecimal()
                ),
                default=0,
            )
            delay = min(PROVIDER_RETRY_MAX_DELAY, max(retry_after, backoff.next_interval()))
            if deadline is not None and get_running_loop().time() + delay >= deadline:
                raise
            logger.warning(
                "DAYTONA_PROVIDER_RETRY operation=%s attempt=%d status=%s delay=%.3f",
                operation,
                attempt,
                status,
                delay,
                extra={
                    "provider_operation": operation,
                    "provider_retry_attempt": attempt,
                    "provider_status_code": status,
                },
            )
            await sleep(delay)
