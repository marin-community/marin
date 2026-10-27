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
from collections.abc import AsyncIterable, AsyncIterator, Iterable, Sequence
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


async def _wait_for_kept_shares(
    sources: Sequence["Source"],
    buffers: Sequence[asyncio.Queue[object]],
    live: set[int],
    empty_since: dict[int, float],
    filled: Sequence[asyncio.Event],
    rng: random.Random,
) -> None:
    """Draw a source by share among the live ones; if it is empty and keeps its share, wait for its next item.

    The wait ends when the drawn source gets an item or has been empty for its ``max_wait``. Only a mix with a source
    that waits draws here, so the order of a mix without waits stays the same.
    """
    loop = asyncio.get_running_loop()
    for index in live:
        if not buffers[index].empty():
            empty_since.pop(index, None)
    candidates = sorted(live)
    index = rng.choices(candidates, weights=[sources[i].share for i in candidates])[0]
    if not buffers[index].empty() or sources[index].max_wait <= 0:
        return
    since = empty_since.setdefault(index, loop.time())
    remaining = sources[index].max_wait - (loop.time() - since)
    if remaining <= 0:
        return
    filled[index].clear()
    if not buffers[index].empty():
        return
    with suppress(TimeoutError):
        await asyncio.wait_for(filled[index].wait(), remaining)


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
    """A stream for ``mix``, with the share of the mixed items that it gets while it has items ready.

    With ``max_wait`` (seconds), the mix keeps the share of an empty source: it waits for the source's next item until
    the source has been empty that long, and only then gives its share to the others, until the source has items again.
    """

    items: AsyncIterable[T]
    share: float
    max_wait: float = 0.0


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
    filled = [asyncio.Event() for _ in sources]

    async def pump(index: int) -> None:
        try:
            async for item in sources[index].items:
                await room[index].acquire()
                buffers[index].put_nowait(item)
                filled[index].set()
                ready.set()
        finally:
            buffers[index].put_nowait(_END)
            filled[index].set()
            ready.set()

    pumps = [asyncio.create_task(pump(index)) for index in range(len(sources))]
    live = set(range(len(sources)))
    empty_since: dict[int, float] = {}
    try:
        while live:
            if any(sources[index].max_wait > 0 for index in live):
                await _wait_for_kept_shares(sources, buffers, live, empty_since, filled, rng)
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
