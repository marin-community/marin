# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Drain admitted commands before releasing a machine's resources."""

import asyncio
from collections.abc import Awaitable
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
