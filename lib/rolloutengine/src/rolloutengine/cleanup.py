# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded cleanup and ownership of operations that outlive cancellation."""

import asyncio
import logging
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)
_background_tasks: set[asyncio.Task[None]] = set()


def _retain_task(task: asyncio.Task[None]) -> None:
    _background_tasks.add(task)
    task.add_done_callback(_background_done)


def _background_done(task: asyncio.Task[None]) -> None:
    _background_tasks.discard(task)
    if not task.cancelled() and (error := task.exception()) is not None:
        logger.error("Background resource operation failed", exc_info=error)


@dataclass(frozen=True)
class CleanupError:
    """A cleanup action that failed, by operation name and exception type."""

    operation: str
    exception_type: str


@dataclass
class Cleanup:
    """Run cleanup actions, each within ``timeout`` seconds, and record their failures in ``errors``."""

    timeout: float
    errors: list[CleanupError] = field(default_factory=list)

    async def run(self, operation: str, action: Callable[[], Coroutine[Any, Any, None]]) -> Exception | None:
        """Run ``action`` within the deadline and return its failure, if any.

        Repeated cancellation cannot extend the deadline and is re-raised after the action ends.
        An action that ignores cancellation at the deadline is retained until it completes.
        """
        pending = asyncio.create_task(action())
        loop = asyncio.get_running_loop()
        end = loop.time() + self.timeout
        cancellation = None
        failure = None
        while not pending.done():
            remaining = end - loop.time()
            if remaining <= 0:
                pending.cancel()
                _retain_task(pending)
                failure = TimeoutError("Cleanup deadline expired")
                break
            try:
                await asyncio.wait((pending,), timeout=remaining)
            except asyncio.CancelledError as error:
                # Repeated cancellation cannot extend the cleanup deadline.
                cancellation = error
        if failure is None:
            try:
                pending.result()
            except asyncio.CancelledError as error:
                failure = RuntimeError("Cleanup operation cancelled")
                failure.__cause__ = error
            except Exception as error:
                failure = error
        if failure is not None:
            self.errors.append(CleanupError(operation, type(failure).__name__))
            logger.warning("Cleanup failed during %s", operation, exc_info=failure)
        if cancellation is not None:
            if failure is not None:
                raise cancellation from failure
            raise cancellation
        return failure


async def finish_cleanup(action: Callable[[], Coroutine[Any, Any, None]], *, timeout: float) -> None:
    """Finish resource cleanup within its deadline and propagate failures or repeated cancellation."""
    failure = await Cleanup(timeout).run("resource_close", action)
    if failure is not None:
        raise failure
