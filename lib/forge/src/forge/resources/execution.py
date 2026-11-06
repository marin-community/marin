# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded retries shared by resource executors, outside the effect context."""

from asyncio import sleep
from collections.abc import Awaitable, Callable

from forge.effects.events import Event, Retrying
from forge.effects.issues import Restart, Transient

type Notify = Callable[[Event[None]], object]
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


def has_cause(error: BaseException, kind: type[BaseException]) -> bool:
    seen: set[int] = set()
    cause: BaseException | None = error
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        if isinstance(cause, kind):
            return True
        cause = cause.__cause__
    return False
