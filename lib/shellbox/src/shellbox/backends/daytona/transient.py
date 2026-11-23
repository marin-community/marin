# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Daytona requests that fail in transit, and their retry."""

import logging
from asyncio import sleep
from collections.abc import Awaitable, Callable

from daytona import DaytonaConnectionError, DaytonaError, DaytonaTimeoutError
from rigging.timing import ExponentialBackoff

logger = logging.getLogger(__name__)

# Gateway and availability statuses a Daytona API request may return transiently.
TRANSIENT_DAYTONA_STATUS = frozenset({502, 503, 504})
TRANSIENT_RETRIES = 3


def is_transient_daytona_error(error: DaytonaError) -> bool:
    """Whether a request failed in transit: a lost connection, or a gateway or availability status.

    The provider may have run the request, so repeat only a request that is safe to run twice.
    """
    return isinstance(error, DaytonaConnectionError) or error.status_code in TRANSIENT_DAYTONA_STATUS


def is_ambiguous_daytona_error(error: DaytonaError) -> bool:
    """Whether the provider may have run a request whose answer never arrived: a transit failure or a timeout.

    Repeat such a request only when it is idempotent, or read its effect back first.
    """
    return is_transient_daytona_error(error) or isinstance(error, DaytonaTimeoutError)


def retry_backoff() -> ExponentialBackoff:
    """The wait between repeats of a Daytona request: 0.5 s, doubling to 4 s, with jitter."""
    return ExponentialBackoff(initial=0.5, maximum=4, factor=2, jitter=0.5)


async def retry_transient[T](request: Callable[[], Awaitable[T]]) -> T:
    """Repeat a provider request that failed in transit. The request must be safe to run twice."""
    backoff = retry_backoff()
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
