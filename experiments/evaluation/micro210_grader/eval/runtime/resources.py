"""Resource isolation for ordinary functions, independent of their callers."""

from __future__ import annotations

import asyncio
import multiprocessing
import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from enum import StrEnum
from functools import cache
from typing import Any, TypeVar

Result = TypeVar("Result")


@cache
def prepare_nltk_tokenizers(tokenizers: tuple[str, ...]) -> None:
    """Prepare optional tokenizer data once in each grader process."""
    import nltk

    for tokenizer in tokenizers:
        try:
            nltk.data.find(f"tokenizers/{tokenizer}")
        except LookupError:
            nltk.download(tokenizer, quiet=True, raise_on_error=True)


class ExecutionMode(StrEnum):
    SERIAL = "serial"
    THREAD_SAFE = "thread_safe"
    PROCESS_ISOLATED = "process_isolated"
    SANDBOXED = "sandboxed"


class SerialIO:
    """Serialize blocking storage calls and join an active write on cancellation."""

    def __init__(self):
        self.admission = asyncio.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="evalchemy-io")

    async def call(self, function: Callable[..., Result], *args: Any) -> Result:
        async with self.admission:
            future = asyncio.get_running_loop().run_in_executor(self.executor, function, *args)
            cancelled = None
            while not future.done():
                try:
                    await asyncio.shield(future)
                except asyncio.CancelledError as error:
                    # A cancelled asyncio waiter cannot cancel a running storage
                    # write. Keep admission until it finishes, even if shutdown
                    # delivers another cancellation while we are joining it.
                    cancelled = error
            result = future.result()
            if cancelled is not None:
                raise cancelled
            return result

    def close(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=True)


def initialize_grader_worker() -> None:
    # The outer worker starts cleanly with spawn. Canonical code graders then
    # fork their local sandbox entrypoints from this worker's main thread.
    multiprocessing.set_start_method("fork", force=True)


class Resources:
    def __init__(self, workers: int = 1):
        self.workers = min(workers, os.cpu_count() or 1, 16)
        self.processes: ProcessPoolExecutor | None = None

    async def call(self, function: Callable[..., Result], args: tuple[Any, ...], mode: ExecutionMode) -> Result:
        if mode is ExecutionMode.THREAD_SAFE:
            return await asyncio.to_thread(function, *args)
        if mode in (ExecutionMode.PROCESS_ISOLATED, ExecutionMode.SANDBOXED):
            if self.processes is None:
                self.processes = ProcessPoolExecutor(
                    max_workers=self.workers, mp_context=multiprocessing.get_context("spawn"),
                    initializer=initialize_grader_worker,
                )
            return await asyncio.get_running_loop().run_in_executor(self.processes, function, *args)
        return function(*args)

    def close(self) -> None:
        if self.processes is not None:
            self.processes.shutdown(wait=True, cancel_futures=True)
            self.processes = None
