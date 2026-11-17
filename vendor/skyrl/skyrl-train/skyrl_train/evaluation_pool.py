"""Fair, bounded admission for a fixed set of evaluation requests."""

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from loguru import logger

T = TypeVar("T")


@dataclass
class _Request:
    run: Callable[[], Awaitable[Any]]
    result: asyncio.Future


@dataclass
class _SourceProgress:
    started: int = 0
    finished: int = 0
    first_start: float | None = None
    last_finish: float | None = None


class EvaluationPool:
    """Replenish free slots across sources without batch or cohort barriers.

    Each request retains its own result, including failures. Source limits constrain
    expensive sandbox providers without reserving idle capacity from other sources.
    The caller owns the frozen-policy session around this pool.
    """

    def __init__(self, concurrency: int):
        if concurrency < 1:
            raise ValueError("evaluation concurrency must be positive")
        self.concurrency = concurrency
        self._queues: dict[str, deque[_Request]] = {}
        self._limits: dict[str, int] = {}
        self._active: dict[str, int] = {}
        self._sources: deque[str] = deque()
        self._changed = asyncio.Condition()
        self._workers: list[asyncio.Task] = []
        self._closing = False
        self._progress: dict[str, _SourceProgress] = {}
        self._peak_active = 0
        self._sources_overlapped = False

    async def __aenter__(self):
        self._started_at = asyncio.get_running_loop().time()
        self._workers = [asyncio.create_task(self._worker()) for _ in range(self.concurrency)]
        return self

    async def __aexit__(self, *_args):
        self._closing = True
        for worker in self._workers:
            worker.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        for queue in self._queues.values():
            for request in queue:
                request.result.cancel()
        logger.info(
            "Evaluation admission summary: peak_active={} sources_overlapped={} sources={}",
            self._peak_active,
            self._sources_overlapped,
            self._progress,
        )

    async def run(self, source: str, run: Callable[[], Awaitable[T]], *, limit: int | None = None) -> T:
        result = asyncio.get_running_loop().create_future()
        async with self._changed:
            source_limit = self.concurrency if limit is None else limit
            if source_limit < 1:
                raise ValueError("evaluation source concurrency must be positive")
            if source not in self._queues:
                self._queues[source] = deque()
                self._limits[source] = source_limit
                self._active[source] = 0
                self._progress[source] = _SourceProgress()
                self._sources.append(source)
            if self._limits[source] != source_limit:
                raise ValueError(f"inconsistent concurrency limit for {source}")
            self._queues[source].append(_Request(run, result))
            self._changed.notify_all()
        return await result

    def _next(self) -> tuple[str, _Request] | None:
        for _ in range(len(self._sources)):
            source = self._sources[0]
            self._sources.rotate(-1)
            queue = self._queues[source]
            while queue and queue[0].result.cancelled():
                queue.popleft()
            if queue and self._active[source] < self._limits[source]:
                self._active[source] += 1
                return source, queue.popleft()
        return None

    async def _worker(self):
        while True:
            async with self._changed:
                source, request = await self._changed.wait_for(self._next)
            progress = self._progress[source]
            progress.started += 1
            if progress.first_start is None:
                progress.first_start = asyncio.get_running_loop().time() - self._started_at
            self._peak_active = max(self._peak_active, sum(self._active.values()))
            self._sources_overlapped |= sum(active > 0 for active in self._active.values()) > 1
            try:
                value = await request.run()
            except asyncio.CancelledError:
                if self._closing:
                    request.result.cancel()
                    raise
                current = asyncio.current_task()
                while current.cancelling():
                    current.uncancel()
                if not request.result.done():
                    request.result.set_exception(RuntimeError("evaluation provider cancelled an admitted request"))
            except Exception as error:
                if not request.result.done():
                    request.result.set_exception(error)
            else:
                if not request.result.done():
                    request.result.set_result(value)
            finally:
                progress.finished += 1
                progress.last_finish = asyncio.get_running_loop().time() - self._started_at
                async with self._changed:
                    self._active[source] -= 1
                    self._changed.notify_all()
