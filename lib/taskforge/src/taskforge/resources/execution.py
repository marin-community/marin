# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded retries shared by resource executors, outside the effect context."""

import logging
from asyncio import sleep
from asyncio import timeout as deadline
from collections.abc import Awaitable, Callable

from taskforge.effects.events import Event, Note, Retrying
from taskforge.effects.issues import Restart, Transient, describe_exception

type Notify = Callable[[Event[None]], None]
MAX_ATTEMPTS = 3


async def retry_effect[T](event: Event[T], operation: Callable[[], Awaitable[T]], notify: Notify) -> T:
    attempt = 1
    delay = 1.0
    while True:
        try:
            return await operation()
        except Transient as error:
            if attempt == MAX_ATTEMPTS:
                raise Restart(f"{type(event).__name__} exhausted its retry budget: {error}") from error
            notify(Retrying(event, attempt, str(error)))
            await sleep(delay)
            delay = min(60.0, delay * 2)
            attempt += 1


async def close_resource(action: Callable[[], Awaitable[None]], notify: Notify, *, timeout: float | None = None) -> None:
    """Warn about failed cleanup without replacing the result or the primary failure."""
    try:
        async with deadline(timeout):
            await action()
    except Exception as error:
        message = f"Cleanup failed: {describe_exception(error)}"
        logging.warning(message, exc_info=error)
        notify(Note(message))
