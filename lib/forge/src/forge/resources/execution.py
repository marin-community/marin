# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded retries shared by resource executors, outside the effect context."""

from collections.abc import Awaitable, Callable

from forge.effects.events import Event, Retrying
from forge.effects.issues import Restart, Transient

type Notify = Callable[[Event[None]], object]
MAX_ATTEMPTS = 3


async def retry_effect[T](event: Event[T], operation: Callable[[], Awaitable[T]], notify: Notify) -> T:
    attempt = 1
    while True:
        try:
            return await operation()
        except Transient as error:
            if attempt == MAX_ATTEMPTS:
                raise Restart(f"{type(event).__name__} exhausted its retry budget") from error
            notify(Retrying(event, attempt, str(error)))
            attempt += 1
