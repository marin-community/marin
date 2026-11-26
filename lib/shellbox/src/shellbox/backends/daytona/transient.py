# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Retry safe Daytona requests after temporary provider failures."""

import logging
from asyncio import sleep
from collections.abc import Awaitable, Callable

from daytona import DaytonaConnectionError, DaytonaError
from rigging.timing import ExponentialBackoff

logger = logging.getLogger(__name__)

# Throttling, gateway and availability responses can recover with backoff.
TRANSIENT_DAYTONA_STATUS = frozenset({429, 502, 503, 504})
TRANSIENT_RETRIES = 3


def is_transient_daytona_error(error: DaytonaError) -> bool:
    """Whether a request lost its connection, was throttled, or was temporarily unavailable.

    The provider may have run the request, so repeat only a request that is safe to run twice.
    """
    return isinstance(error, DaytonaConnectionError) or error.status_code in TRANSIENT_DAYTONA_STATUS


async def retry_transient[T](request: Callable[[], Awaitable[T]]) -> T:
    """Repeat a temporary provider failure. The request must be safe to run twice."""
    backoff = ExponentialBackoff(initial=0.5, maximum=4, factor=2, jitter=0.5)
    for _ in range(TRANSIENT_RETRIES):
        try:
            return await request()
        except DaytonaError as error:
            if not is_transient_daytona_error(error):
                raise
            delay = backoff.next_interval()
            logger.warning("Retrying a Daytona request after %s: delay=%.3f", type(error).__name__, delay)
            await sleep(delay)
    return await request()
