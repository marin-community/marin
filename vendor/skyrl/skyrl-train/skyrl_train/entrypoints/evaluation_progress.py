"""Forward existing evaluation progress records through the Ray driver."""

from collections.abc import Iterator
from contextlib import contextmanager
from queue import Empty
from typing import Protocol

import ray
from loguru import logger


class ProgressQueue(Protocol):
    def put(self, item: str) -> None: ...

    def get_nowait(self) -> str: ...


@contextmanager
def forward_evaluation_progress(queue: ProgressQueue) -> Iterator[None]:
    """Retain worker logs and forward only the existing throttled progress lines."""
    sink = logger.add(
        lambda message: queue.put(str(message)),
        filter=lambda record: record["name"] == "skyrl_train.utils.progress",
        format="{message}",
        catch=False,
    )
    try:
        yield
    finally:
        logger.remove(sink)


def wait_for_evaluation(result: ray.ObjectRef, queue: ProgressQueue) -> dict:
    """Drain worker progress into driver stdout until the evaluation returns."""
    while True:
        ready, _ = ray.wait([result], timeout=1)
        while True:
            try:
                line = queue.get_nowait()
            except Empty:
                break
            print(line, end="" if line.endswith("\n") else "\n", flush=True)
        if ready:
            return ray.get(result)
