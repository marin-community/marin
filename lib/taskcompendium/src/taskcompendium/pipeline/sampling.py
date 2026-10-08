# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Seeded, bounded samples of source rows, and merging them while retaining their full population counts."""

import hashlib
import heapq
from collections.abc import Callable, Iterable, Iterator


def seeded_order(identifier: str, seed: int) -> tuple[str, str]:
    """Order identifiers by their seeded SHA-256, breaking ties by the identifier itself."""
    return hashlib.sha256(f"{seed}:{identifier}".encode()).hexdigest(), identifier


def seeded_sample[Row](
    rows: Iterable[Row], *, size: int, key: Callable[[Row], tuple[str, str]]
) -> tuple[int, list[Row]]:
    """Count ``rows`` and keep the ``size`` that sort first by ``key``, holding at most ``size`` in memory."""
    count = 0

    def counted() -> Iterator[Row]:
        nonlocal count
        for row in rows:
            count += 1
            yield row

    selected = heapq.nsmallest(size, counted(), key=key)
    return count, selected


def merge_sample_rows[Row](
    samples: Iterable[tuple[int, Iterable[Row]]], *, size: int, key: Callable[[Row], tuple[str, str]]
) -> tuple[int, list[Row]]:
    count = 0

    def candidates() -> Iterator[Row]:
        nonlocal count
        for population_count, rows in samples:
            count += population_count
            yield from rows

    selected = heapq.nsmallest(size, candidates(), key=key)
    return count, selected
