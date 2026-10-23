# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Async streams that connect the boxes of the RSI loop, so that the data drives the loop and no ``while`` does.

Each box of the loop is a function from a stream to a stream. ``Broadcast`` is the one edge that goes back: the
trainer's rollouts pass through it to the caller, and each listener gets a copy, for example the failure analysis.
``mix`` reads its sources in background tasks, so that a slow source, such as task synthesis, does not stop a fast
one, such as the seed tasks.
"""

import asyncio
import random
from collections.abc import AsyncIterable, AsyncIterator, Iterable
from contextlib import suppress
from dataclasses import dataclass

_END = object()


class Broadcast[T]:
    """Pass the items of a stream through, and give a copy of each item to every listener.

    Listen before the stream starts: a listener gets only the items that come after it starts to listen.
    """

    def __init__(self) -> None:
        self._listeners: list[asyncio.Queue[object]] = []

    def listen(self) -> AsyncIterator[T]:
        queue: asyncio.Queue[object] = asyncio.Queue()
        self._listeners.append(queue)
        return _drain(queue)

    async def publish(self, items: AsyncIterable[T]) -> AsyncIterator[T]:
        try:
            async for item in items:
                for queue in self._listeners:
                    queue.put_nowait(item)
                yield item
        finally:
            for queue in self._listeners:
                queue.put_nowait(_END)


async def _drain[T](queue: asyncio.Queue[object]) -> AsyncIterator[T]:
    while (item := await queue.get()) is not _END:
        yield item  # type: ignore[misc]


async def batched[T](items: AsyncIterable[T], size: int) -> AsyncIterator[list[T]]:
    """Lists of ``size`` items. A box that needs enough data to start waits here. A shorter rest is dropped: it comes
    when the loop ends, when no box can use it."""
    if size < 1:
        raise ValueError("size must be at least 1")
    batch: list[T] = []
    async for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []


async def from_iterable[T](items: Iterable[T]) -> AsyncIterator[T]:
    for item in items:
        yield item


@dataclass(frozen=True)
class Source[T]:
    """A stream for ``mix``, with the share of the mixed items that it gets while it has items ready."""

    items: AsyncIterable[T]
    share: float


MIX_BUFFER = 64
"""Items that ``mix`` reads ahead from each source."""


async def mix[T](*sources: Source[T], seed: int) -> AsyncIterator[T]:
    """The items of ``sources``, each drawn from a source with items ready, in proportion to the source shares.

    A source with no item ready gives its share to the others, so the stream never waits for a slow source while a
    faster one has items. The stream ends when all sources end.
    """
    if not sources or any(source.share <= 0 for source in sources):
        raise ValueError("mix needs sources with positive shares")
    rng = random.Random(seed)
    buffers: list[asyncio.Queue[object]] = [asyncio.Queue() for _ in sources]
    room = [asyncio.Semaphore(MIX_BUFFER) for _ in sources]
    ready = asyncio.Event()

    async def pump(index: int) -> None:
        try:
            async for item in sources[index].items:
                await room[index].acquire()
                buffers[index].put_nowait(item)
                ready.set()
        finally:
            buffers[index].put_nowait(_END)
            ready.set()

    pumps = [asyncio.create_task(pump(index)) for index in range(len(sources))]
    live = set(range(len(sources)))
    try:
        while live:
            available = [index for index in live if not buffers[index].empty()]
            if not available:
                ready.clear()
                await ready.wait()
                continue
            index = rng.choices(available, weights=[sources[i].share for i in available])[0]
            item = buffers[index].get_nowait()
            if item is _END:
                live.discard(index)
                await pumps[index]
                continue
            room[index].release()
            yield item  # type: ignore[misc]
    finally:
        for task in pumps:
            task.cancel()
        for task in pumps:
            with suppress(asyncio.CancelledError):
                await task
