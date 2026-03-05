# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from typing import TypeVar
from collections.abc import Iterable, Iterator

T = TypeVar("T")


def batched(iterable: Iterable[T], batch_size: int) -> Iterator[list[T]]:
    """Yields batches of the given size from the given iterable."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []

    if len(batch) > 0:
        yield batch
