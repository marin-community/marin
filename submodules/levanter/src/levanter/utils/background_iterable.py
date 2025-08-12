import asyncio
import queue
import sys
import threading
from typing import AsyncIterator, Callable, Iterable, Iterator, Optional, TypeVar, Union

import tblib

from levanter.utils.thread_utils import AsyncIteratorWrapper


Ex = TypeVar("Ex", covariant=True)


class BackgroundIterable(Iterable[Ex]):
    """
    A wrapper around an iterable that runs the iterable in a background thread and fills a queue with the results.

    This allows the iterable to be consumed in a separate thread, and for the results to be consumed in the main thread.
    This will only work particularly well if the main thread is doing some kind of IO or other blocking operation,
    like running XLA kernels...
    """

    def __init__(
        self,
        producer_fn: Callable[[], Union[Iterator[Ex], AsyncIterator[Ex]]],
        max_capacity: Optional[int] = None,
    ):
        self.max_capacity = max_capacity
        self._producer_fn = producer_fn

    def __iter__(self):
        return BackgroundIterator(self._producer_fn, self.max_capacity)


class BackgroundIterator(Iterator[Ex]):
    def __init__(
        self,
        producer_fn: Callable[[], Iterator[Ex] | AsyncIterator[Ex]] | Iterator[Ex] | AsyncIterator[Ex],
        max_capacity: Optional[int],
    ):
        self.max_capacity = max_capacity
        if not callable(producer_fn):
            self._producer_fn = lambda: producer_fn
        else:
            self._producer_fn = producer_fn
        self._stop_event = threading.Event()

        if self.max_capacity is None or self.max_capacity >= 0:
            self.q: queue.Queue = queue.Queue(self.max_capacity or 0)
            self.thread: Optional[threading.Thread] = threading.Thread(target=self._fill_queue_with_batches)
            self.thread.daemon = True
            self.thread.start()
        else:
            # No background thread; consume items on demand
            self.thread = None
            self.iterator = self._producer_fn()
            if not isinstance(self.iterator, Iterator):
                self.iterator = AsyncIteratorWrapper(self.iterator)

    def __iter__(self):
        return self

    def __next__(self):
        if self._stop_event.is_set():
            raise StopIteration
        if self.thread is not None:
            while not self._stop_event.is_set():
                batch = self.q.get()
                if batch is _SENTINEL:
                    raise StopIteration
                elif isinstance(batch, _ExceptionWrapper):
                    batch.reraise()
                return batch
        else:
            # Consume the iterator directly on demand
            try:
                return next(self.iterator)
            except StopIteration:
                raise
            except StopAsyncIteration:
                raise StopIteration
            except Exception as e:
                raise e
        raise StopIteration

    def __del__(self):
        self.stop()

    def stop(self, wait: bool = True):
        self._stop_event.set()
        # I'm getting an error that the thread is threading.current_thread(), which seems impossible
        if self.thread is not None and wait and self.thread != threading.current_thread():
            self.thread.join()

    def _fill_queue_with_batches(self):
        try:
            iterator = self._producer_fn()
            if isinstance(iterator, Iterator):
                self._produce_batches_sync(iterator)
            else:
                asyncio.run(self._produce_batches_async(iterator))
        except Exception:
            self.q.put(_ExceptionWrapper(sys.exc_info()))

    def _produce_batches_sync(self, iterator):
        try:
            for batch in iterator:
                while not self._stop_event.is_set():
                    try:
                        self.q.put(batch, block=True, timeout=1)
                        break
                    except queue.Full:
                        pass

                if self._stop_event.is_set():
                    break

            while not self._stop_event.is_set():
                try:
                    self.q.put(_SENTINEL, block=True, timeout=1)
                    break
                except queue.Full:
                    pass
        except Exception:
            self.q.put(_ExceptionWrapper(sys.exc_info()))

    async def _produce_batches_async(self, iterator):
        try:
            async for batch in iterator:
                while not self._stop_event.is_set():
                    try:
                        self.q.put(batch, block=True, timeout=1)
                        break
                    except queue.Full:
                        pass

                if self._stop_event.is_set():
                    break

            while not self._stop_event.is_set():
                try:
                    self.q.put(_SENTINEL, block=True, timeout=1)
                    break
                except queue.Full:
                    pass
        except Exception:
            self.q.put(_ExceptionWrapper(sys.exc_info()))


class _Sentinel:
    """A sentinel object for marking the end of a stream of data."""

    pass


_SENTINEL = _Sentinel()


class _ExceptionWrapper:
    """Wraps exception and original traceback in object for queue."""

    def __init__(self, exc_info):
        self.exc_type = exc_info[0]
        self.exc = exc_info[1]
        self.tb = tblib.Traceback(exc_info[2])

    def reraise(self):
        raise self.exc.with_traceback(self.tb.as_traceback())
