# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Seeded sample order, and merging bounded source samples while retaining their full population counts."""

import hashlib
import heapq
from collections.abc import Callable, Iterable, Iterator


def seeded_order(identifier: str, seed: int) -> tuple[str, str]:
    """Order identifiers by their seeded SHA-256, breaking ties by the identifier itself."""
    return hashlib.sha256(f"{seed}:{identifier}".encode()).hexdigest(), identifier


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
