# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Data-only reader contract shared by file and shuffle inputs."""

from collections.abc import Callable, Generator, Iterable, Iterator
from contextlib import closing
from functools import partial
from typing import Any


def _single_row(item: Any) -> int:
    return 1


class Reader[T](Iterator[T]):
    """An input iterator with a best-effort row total.

    Readers fill ``total_rows`` when they open their input on first iteration.
    ``None`` means the total is unknown. Batch readers supply ``rows_per_item``
    so the input operation can count records without knowing the format.
    """

    def __init__(
        self,
        read: Callable[["Reader[T]"], Generator[T, None, None]],
        *,
        total_rows: int | None = None,
        rows_per_item: Callable[[T], int] = _single_row,
    ) -> None:
        self.total_rows = total_rows
        self.rows_per_item = rows_per_item
        self._items = read(self)

    def __next__(self) -> T:
        return next(self._items)

    def close(self) -> None:
        self._items.close()


def flat_map_reader[T, U](
    source: Reader[T], fn: Callable[[T], Iterable[U]], *, rows_per_item: Callable[[U], int] = _single_row
) -> Reader[U]:
    """Adapt reader output while retaining its row total and resource lifetime."""
    return Reader(partial(_mapped_items, source, fn), rows_per_item=rows_per_item)


def _mapped_items[T, U](source: Reader[T], fn: Callable[[T], Iterable[U]], reader: Reader[U]):
    with closing(source):
        for item in source:
            reader.total_rows = source.total_rows
            yield from fn(item)
        reader.total_rows = source.total_rows
