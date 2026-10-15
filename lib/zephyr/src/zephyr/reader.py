# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Data-only reader contract shared by file and shuffle inputs."""

from collections.abc import Callable, Generator, Iterator
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
