import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator


# Create a ThreadPoolExecutor
_executor = ThreadPoolExecutor(max_workers=10)


def blocking_wait(coro):
    """
    This will only work if there are fewer than 10 levels of nested coroutines...
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        future = _executor.submit(lambda: asyncio.run(coro))
        return future.result()
    else:
        return asyncio.run(coro)


def future_from_value(value):
    future = asyncio.Future()
    future.set_result(value)
    return future


class AsyncIteratorWrapper(Iterator):
    def __init__(self, async_iter):
        self.async_iter = async_iter
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        self._exhausted = False  # Flag to indicate if the iterator is exhausted

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def _run_async_task(self, coro):
        if not self.loop.is_running() or not self.thread.is_alive():
            raise StopIteration
        try:
            future = asyncio.run_coroutine_threadsafe(coro, self.loop)
            return future.result()
        except (RuntimeError, asyncio.CancelledError):
            raise StopIteration

    def __iter__(self):
        return self

    def __next__(self):
        if self._exhausted:
            raise StopIteration
        try:
            return self._run_async_task(self.async_iter.__anext__())
        except StopAsyncIteration:
            self._exhausted = True  # Mark the iterator as exhausted
            if self.loop.is_running():
                self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join()
            raise StopIteration

    def close(self):
        """Close the event loop and thread gracefully."""
        if self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join()
        self.loop.close()


class ExceptionTrackingThread(threading.Thread):
    """A thread that will store exceptions that occur in the target function and
    re-raise them in the main thread."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._exception = None

    def run(self):
        try:
            super().run()
        except Exception as e:
            self._exception = e

    def join(self, *args, **kwargs):
        super().join(*args, **kwargs)
        if self._exception:
            raise self._exception

    def check_raise(self):
        if self._exception:
            raise self._exception
