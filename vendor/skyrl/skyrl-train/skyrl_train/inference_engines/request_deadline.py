"""Return an active request's exact partial output before its caller's deadline."""

import asyncio
from collections.abc import Awaitable, Callable


PARTIAL_RESPONSE_RETURN_MARGIN = 30.0


def partial_response_timeout(execution_remaining: float | None) -> float | None:
    """Return an abort delay that leaves transport time inside the execution budget."""
    if execution_remaining is None:
        return None
    margin = min(PARTIAL_RESPONSE_RETURN_MARGIN, execution_remaining / 2)
    return max(0.0, execution_remaining - margin)


async def finish_partial_on_timeout[T](
    request: Awaitable[T], timeout: float | None, finish_partial: Callable[[], Awaitable[bool]],
    *, scheduled: asyncio.Event | None = None
) -> tuple[T, bool]:
    """Finish an active decode, leaving an unstarted request to its caller's timeout.

    ``finish_partial`` must return false without aborting when no tokens exist.
    An outer cancellation still cancels the request; it never converts a shutdown
    into a successful partial response.
    """
    if timeout is None:
        return await request, False
    task = asyncio.ensure_future(request)
    admission = None if scheduled is None else asyncio.create_task(scheduled.wait())
    try:
        if admission is not None:
            await asyncio.wait({task, admission}, return_when=asyncio.FIRST_COMPLETED)
        done, _ = await asyncio.wait({task}, timeout=timeout)
        if task in done:
            return task.result(), False
        finished = await finish_partial()
        return await task, finished
    finally:
        task.cancel()
        if admission is not None:
            admission.cancel()
        await asyncio.gather(task, *([] if admission is None else [admission]), return_exceptions=True)
