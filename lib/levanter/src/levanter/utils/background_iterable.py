# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextvars
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
        # Capture the parent thread's ContextVars + JAX mesh stack so the
        # prefetch thread runs under the same JAX mesh / sharding context.
        # threading.Thread does NOT propagate either by default; without
        # this, jits traced inside _fill_queue_with_batches see an empty
        # mesh and fall back to CPU, causing "Received incompatible devices"
        # when the resulting array meets TPU-resident data downstream.
        #
        # JAX stores its active-mesh stack in a `threading.local` subclass
        # (`jax._src.mesh.thread_resources`), NOT a ContextVar — so
        # `copy_context()` alone is insufficient. Snapshot the mesh stack
        # manually here and restore it inside the prefetch thread.
        self._captured_ctx = contextvars.copy_context()
        self._captured_jax_mesh_stack: Optional[list] = None
        self._captured_jax_mesh_env = None
        try:
            # Lazy import of a JAX private internal; guarded for version drift.
            from jax._src.mesh import thread_resources as _jax_thread_resources  # noqa: PLC0415

            self._captured_jax_mesh_stack = list(_jax_thread_resources.stack)
            self._captured_jax_mesh_env = _jax_thread_resources.env
        except Exception:
            pass

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
            batch = self.q.get()
            if batch is _SENTINEL:
                raise StopIteration
            if isinstance(batch, _ExceptionWrapper):
                batch.reraise()
            return batch
        # No background thread; consume the iterator on demand.
        try:
            return next(self.iterator)
        except StopAsyncIteration:
            raise StopIteration

    def __del__(self):
        self.stop()

    def qsize(self) -> int | None:
        """Current number of items in the prefetch queue, or None if unbuffered."""
        if self.thread is not None:
            return self.q.qsize()
        return None

    def stop(self, wait: bool = True):
        self._stop_event.set()
        # I'm getting an error that the thread is threading.current_thread(), which seems impossible
        if self.thread is not None and wait and self.thread != threading.current_thread():
            self.thread.join()

    def _fill_queue_with_batches(self):
        # Restore JAX's thread-local mesh stack in this producer thread,
        # since `threading.Thread` doesn't carry it across. This is the
        # actual fix for the eval-mesh ValueError; `copy_context` below
        # is belt-and-suspenders for any non-mesh ContextVars.
        mesh_stack = getattr(self, "_captured_jax_mesh_stack", None)
        mesh_env = getattr(self, "_captured_jax_mesh_env", None)
        if mesh_stack is not None:
            try:
                from jax._src.mesh import thread_resources as _jax_thread_resources  # noqa: PLC0415

                _jax_thread_resources.stack = list(mesh_stack)
                if mesh_env is not None:
                    _jax_thread_resources.env = mesh_env
            except Exception:
                pass
        ctx = getattr(self, "_captured_ctx", None)
        if ctx is None:
            self._fill_queue_with_batches_inner()
        else:
            ctx.run(self._fill_queue_with_batches_inner)

    def _fill_queue_with_batches_inner(self):
        try:
            iterator = self._producer_fn()
        except Exception:
            self.q.put(_ExceptionWrapper(sys.exc_info()))
            return

        if isinstance(iterator, Iterator):
            self._produce_batches_sync(iterator)
        else:
            asyncio.run(self._produce_batches_async(iterator))

    def _enqueue(self, item) -> bool:
        """Block until ``item`` is on the queue or stop is signaled.

        Returns True if the item was enqueued; False if stop was requested first.
        """
        while not self._stop_event.is_set():
            try:
                self.q.put(item, block=True, timeout=1)
                return True
            except queue.Full:
                continue
        return False

    def _produce_batches_sync(self, iterator):
        try:
            for batch in iterator:
                if not self._enqueue(batch):
                    return
            self._enqueue(_SENTINEL)
        except Exception:
            self.q.put(_ExceptionWrapper(sys.exc_info()))

    async def _produce_batches_async(self, iterator):
        try:
            async for batch in iterator:
                if not self._enqueue(batch):
                    return
            self._enqueue(_SENTINEL)
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
