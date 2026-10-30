# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Byte- and row-bounded tokenization batches."""

from collections.abc import Callable, Iterable, Iterator
from typing import TypeVar

_Item = TypeVar("_Item")


def bounded_batches(
    items: Iterable[_Item],
    *,
    max_rows: int,
    max_bytes: int,
    byte_size: Callable[[_Item], int],
) -> Iterator[list[_Item]]:
    """Yield ordered batches bounded by row count and estimated bytes.

    A single item larger than ``max_bytes`` is yielded alone.
    """
    pending: list[_Item] = []
    pending_bytes = 0
    for item in items:
        item_bytes = byte_size(item)
        if pending and pending_bytes + item_bytes > max_bytes:
            yield pending
            pending = []
            pending_bytes = 0
        pending.append(item)
        pending_bytes += item_bytes
        if len(pending) >= max_rows or pending_bytes >= max_bytes:
            yield pending
            pending = []
            pending_bytes = 0
    if pending:
        yield pending
