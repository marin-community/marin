"""Return an active request's exact partial output before its caller's deadline."""

import asyncio
from collections.abc import Awaitable, Callable


async def finish_partial_on_timeout[T](
    request: Awaitable[T], timeout: float | None, finish_partial: Callable[[], Awaitable[bool]]
) -> tuple[T, bool]:
    """Finish an active decode, leaving an unstarted request to its caller's timeout.

    ``finish_partial`` must return false without aborting when no tokens exist.
    An outer cancellation still cancels the request; it never converts a shutdown
    into a successful partial response.
    """
    if timeout is None:
        return await request, False
    task = asyncio.ensure_future(request)
    try:
        done, _ = await asyncio.wait({task}, timeout=timeout)
        if task in done:
            return task.result(), False
        finished = await finish_partial()
        return await task, finished
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
