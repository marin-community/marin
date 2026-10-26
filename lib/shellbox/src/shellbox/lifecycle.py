# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Drain admitted commands before releasing a machine's resources."""

import asyncio
from collections.abc import Awaitable, Iterator
from contextlib import contextmanager
from typing import TypeVar

T = TypeVar("T")


async def finish_cleanup(cleanup: Awaitable[T]) -> T:
    """Finish bounded cleanup despite repeated cancellation, then propagate cancellation."""
    task = asyncio.ensure_future(cleanup)
    cancellation = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as error:
            cancellation = error
    result = task.result()
    if cancellation is not None:
        raise cancellation
    return result


class CommandLifetime:
    """Reject new commands after close and drain all admitted commands."""

    def __init__(self) -> None:
        self._tasks: set[asyncio.Task] = set()
        self._closed = False

    @contextmanager
    def admit(self) -> Iterator[None]:
        if self._closed:
            raise RuntimeError("Machine is closed")
        task = asyncio.current_task()
        assert task is not None
        self._tasks.add(task)
        try:
            yield
        finally:
            self._tasks.remove(task)

    async def close(self) -> None:
        self._closed = True
        tasks = self._tasks - {asyncio.current_task()}
        for task in tasks:
            task.cancel()
        if tasks:
            await finish_cleanup(asyncio.gather(*tasks, return_exceptions=True))
