# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Data-driven resource sizing for release stages (DESIGN.md §7).

Curves are seeded from measured runs and recorded here with their provenance. No
stage carries its own stale default (the previous per-release workers hardcoded
workers=4 and a fixed batch size regardless of population).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class HistoryStagingSizing:
    """Worker parallelism for one history-staging run."""

    workers: int
    item_batch_rows: int


# Measured on the consumer37/bootstrap/r39 retained-history runs (2026-11-21..23):
# 4 clients, 1024-row item batches, over a 325-source prior population.
_MEASURED_WORKERS = 4
_MEASURED_ITEM_BATCH_ROWS = 1024
_MEASURED_POPULATION_ITEMS = 800_000
_MIN_WORKERS = 2
_MAX_WORKERS = 8


def history_staging_sizing(total_items: int) -> HistoryStagingSizing:
    """Scale history-copy workers with the prior population's item count.

    Workers grow linearly with the measured population up to the cap; small
    populations still get two workers so a stalled copy is always observable.
    """
    if total_items <= 0:
        raise ValueError(f"history staging sizing needs a positive item count, got {total_items}")
    scaled = round(_MEASURED_WORKERS * total_items / _MEASURED_POPULATION_ITEMS)
    workers = max(_MIN_WORKERS, min(_MAX_WORKERS, scaled))
    return HistoryStagingSizing(workers=workers, item_batch_rows=_MEASURED_ITEM_BATCH_ROWS)
