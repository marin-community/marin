# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import threading

import pytest

from levanter.utils.background_iterable import BackgroundIterator


@pytest.mark.parametrize("max_capacity", [-1, None, 10])
def test_yields_all_items_in_order(max_capacity):
    test_data = list(range(1, 101))

    assert list(BackgroundIterator(lambda: iter(test_data), max_capacity=max_capacity)) == test_data


@pytest.mark.parametrize("max_capacity", [-1, None, 10])
def test_empty_iteration(max_capacity):
    assert list(BackgroundIterator(lambda: iter([]), max_capacity=max_capacity)) == []


@pytest.mark.timeout(10)
@pytest.mark.parametrize("max_capacity", [1, None])
def test_buffered_iterator_stays_exhausted(max_capacity):
    iterator = BackgroundIterator(lambda: iter([1]), max_capacity=max_capacity)

    assert list(iterator) == [1]
    with pytest.raises(StopIteration):
        next(iterator)


@pytest.mark.parametrize("max_capacity", [-1, None, 10])
def test_exception_handling(max_capacity):
    def producer_with_exception():
        raise ValueError("Something went wrong!")

    with pytest.raises(ValueError):
        list(BackgroundIterator(producer_with_exception, max_capacity=max_capacity))


@pytest.mark.timeout(10)
@pytest.mark.parametrize("max_capacity", [1, None])
def test_buffered_iterator_stops_after_producer_error(max_capacity):
    def producer():
        yield 1
        raise ValueError("producer failed")

    iterator = BackgroundIterator(producer, max_capacity=max_capacity)

    assert next(iterator) == 1
    with pytest.raises(ValueError):
        next(iterator)
    with pytest.raises(StopIteration):
        next(iterator)


@pytest.mark.parametrize("max_capacity", [-1, None, 10])
def test_stop_event(max_capacity):
    def ongoing_process():
        while True:
            for item in range(1, 101):
                yield item

    iter1 = BackgroundIterator(ongoing_process, max_capacity=max_capacity)

    for _ in range(5):
        next(iter1)

    iter1.stop()

    # Try to get another item from the iterator (should raise StopIteration)
    # there's a bit of a race so we give it 2 tries, which is enough for the test
    with pytest.raises(StopIteration):
        next(iter1)
        next(iter1)


def test_stop_does_not_hang_when_producer_raises_into_full_queue():
    queue_filled = threading.Event()
    release_error = threading.Event()

    def producer():
        yield 1
        # resumed only after 1 is enqueued, so the capacity-1 queue is now full
        queue_filled.set()
        release_error.wait()
        raise ValueError("producer failed after stop")

    iterator = BackgroundIterator(producer, max_capacity=1)
    assert queue_filled.wait(timeout=10)
    iterator.stop(wait=False)
    release_error.set()

    assert iterator.thread is not None
    iterator.thread.join(timeout=10)
    assert not iterator.thread.is_alive()


@pytest.mark.asyncio
@pytest.mark.parametrize("max_capacity", [-1, None, 10])
async def test_async_yields_all_items_in_order(max_capacity):
    async def async_producer():
        for i in range(1, 101):
            yield i
            if i % 10 == 0:
                await asyncio.sleep(0.001)

    assert list(BackgroundIterator(async_producer, max_capacity=max_capacity)) == list(range(1, 101))


@pytest.mark.asyncio
@pytest.mark.parametrize("max_capacity", [-1, None, 10])
async def test_async_empty_iteration(max_capacity):
    async def async_producer():
        if False:
            yield

    assert list(BackgroundIterator(async_producer, max_capacity=max_capacity)) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("max_capacity", [-1, None, 10])
async def test_async_exception_handling(max_capacity):
    async def async_producer_with_exception():
        raise ValueError("Something went wrong!")
        yield 0  # have to make sure it's an async coroutine

    with pytest.raises(ValueError):
        list(BackgroundIterator(async_producer_with_exception, max_capacity=max_capacity))


@pytest.mark.asyncio
@pytest.mark.parametrize("max_capacity", [-1, None, 10])
async def test_async_stop_event(max_capacity):
    async def ongoing_async_process():
        while True:
            for item in range(1, 101):
                yield item

    iter1 = BackgroundIterator(ongoing_async_process, max_capacity=max_capacity)

    for _ in range(5):
        next(iter1)

    iter1.stop()

    with pytest.raises(StopIteration):
        next(iter1)
        next(iter1)
