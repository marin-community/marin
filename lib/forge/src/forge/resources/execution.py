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


def websocket_setup_failed(error: BaseException) -> bool:
    """Recognize Iris's pre-command websocket failure, including rollout wrappers."""
    seen: set[int] = set()
    cause: BaseException | None = error
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        detail = str(cause)
        # kubernetes 36.0.3 masks websocket construction failures with None.decode.
        # Accept the deployed controller's text and the repaired controller's marker.
        if detail.startswith("Iris exec failed: ") and (
            "'NoneType' object has no attribute 'decode'" in detail
            or detail.startswith("Iris exec failed: Websocket setup failed (status 0): ")
        ):
            return True
        cause = cause.__cause__
    return False
