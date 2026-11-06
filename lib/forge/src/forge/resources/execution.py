# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded retries shared by resource executors, outside the effect context."""

from asyncio import sleep
from collections.abc import Awaitable, Callable

from shellbox.machine import MachineTerminated

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


def exec_machine_lost(error: BaseException) -> bool:
    """A container that cannot be exec'd needs replacement, not an RPC retry."""
    seen: set[int] = set()
    cause: BaseException | None = error
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        if isinstance(cause, MachineTerminated):
            return True
        detail = str(cause)
        if detail.startswith("Iris exec failed: ") and (
            detail.startswith("Iris exec failed: Non-integer exec status:")
            or "error executing command in container" in detail
            or "cannot exec in a" in detail
            or "not running" in detail
        ):
            return True
        cause = cause.__cause__
    return False


def exec_setup_failed(error: BaseException) -> bool:
    """Recognize retryable Iris websocket setup failures, including rollout wrappers."""
    if exec_machine_lost(error):
        return False
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
