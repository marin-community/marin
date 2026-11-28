# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Daytona requests that fail in transit, and their retry."""

import logging
from asyncio import sleep
from collections.abc import Awaitable, Callable

from daytona import DaytonaError
from rigging.task_errors import RetryDisposition
from rigging.timing import ExponentialBackoff

from shellbox.backends.daytona.errors import daytona_error_identity

logger = logging.getLogger(__name__)

TRANSIENT_RETRIES = 3


async def retry_transient[T](request: Callable[[], Awaitable[T]]) -> T:
    """Repeat a provider request that failed in transit. The request must be safe to run twice."""
    backoff = ExponentialBackoff(initial=0.5, maximum=4, factor=2, jitter=0.5)
    for _ in range(TRANSIENT_RETRIES):
        try:
            return await request()
        except DaytonaError as error:
            failure = daytona_error_identity(error, operation="request")
            if failure.retry_disposition is not RetryDisposition.TRANSIENT:
                raise
            delay = backoff.next_interval()
            logger.warning("Retrying a Daytona request after %s: delay=%.3f", type(error).__name__, delay)
            await sleep(delay)
    return await request()
