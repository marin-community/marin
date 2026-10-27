# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``mix`` gives the share of a slow source away, unless the source keeps it with ``max_wait``."""

import asyncio
import time
from collections.abc import AsyncIterator
from itertools import count

from experiments.post_training.rsi.streams import Source, from_iterable, mix

ITEMS = 40


async def slow(delay: float) -> AsyncIterator[str]:
    """An endless source with one item every ``delay`` seconds."""
    for k in count():
        await asyncio.sleep(delay)
        yield f"slow-{k}"


async def first(max_wait: float, delay: float) -> list[str]:
    fast = from_iterable(f"fast-{k}" for k in range(10_000))
    stream = mix(Source(fast, share=0.5), Source(slow(delay), share=0.5, max_wait=max_wait), seed=0)
    items = []
    async for item in stream:
        items.append(item)
        if len(items) == ITEMS:
            return items
    return items


def slow_share(items: list[str]) -> float:
    return sum(item.startswith("slow") for item in items) / len(items)


def test_a_slow_source_gives_its_share_away_without_max_wait():
    assert slow_share(asyncio.run(first(max_wait=0.0, delay=0.01))) < 0.2


def test_a_slow_source_keeps_its_share_with_max_wait():
    assert 0.3 < slow_share(asyncio.run(first(max_wait=1.0, delay=0.01))) < 0.7


def test_the_mix_moves_on_once_a_source_has_waited_its_max_wait():
    start = time.monotonic()
    items = asyncio.run(first(max_wait=0.05, delay=10.0))
    # The slow source never delivers in time, so after one wait the fast source fills the stream.
    assert all(item.startswith("fast") for item in items)
    assert time.monotonic() - start < 2.0
