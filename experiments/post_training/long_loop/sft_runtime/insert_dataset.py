# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Spread a fixed number of extra rows evenly through a training stream without reordering it."""

from collections.abc import Sequence
from typing import TypeVar

from levanter.data.dataset import AsyncDataset

T = TypeVar("T")


class InsertedDataset(AsyncDataset[T]):
    """Put ``rows`` items of ``insert`` at evenly spaced positions in ``[0, total)``; ``base`` fills the rest.

    Position ``p < total`` is an insert slot when ``(p + 1) * rows // total > p * rows // total``. It takes
    ``insert[p * rows // total]``. Every other position takes the next ``base`` item in order. Two runs
    that differ only in the insert therefore read the same ``base`` sequence; the inserted run reads
    ``rows`` fewer base items by position ``total``.
    """

    def __init__(self, base: AsyncDataset[T], insert: AsyncDataset[T], *, rows: int, total: int):
        if not 0 < rows <= total:
            raise ValueError(f"Insert rows must be in (0, {total}], got {rows}")
        if base.first_readable_index() != 0 or insert.first_readable_index() != 0:
            raise ValueError("Inserted streams start at index 0")
        self.base = base
        self.insert = insert
        self.rows = rows
        self.total = total

    def source(self, index: int) -> tuple[bool, int]:
        """Return (is_insert, index into that source) for a stream position."""
        before = min(index, self.total) * self.rows // self.total
        if index < self.total and (index + 1) * self.rows // self.total > before:
            return True, before
        return False, index - before

    async def async_len(self) -> int:
        return await self.base.async_len() + self.rows

    def is_finite(self) -> bool:
        return self.base.is_finite()

    async def get_batch(self, indices: Sequence[int]) -> Sequence[T]:
        sources = [self.source(int(index)) for index in indices]
        inserted = [position for is_insert, position in sources if is_insert]
        based = [position for is_insert, position in sources if not is_insert]
        inserted_items = iter(await self.insert.get_batch(inserted) if inserted else [])
        based_items = iter(await self.base.get_batch(based) if based else [])
        return [next(inserted_items) if is_insert else next(based_items) for is_insert, _ in sources]
