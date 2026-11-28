# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Data-driven resource sizing for release stages (DESIGN.md §7).

Curves are seeded from measured runs and recorded here with their provenance. No
stage carries its own stale default (the previous per-release workers hardcoded
workers=4 and a fixed batch size regardless of population).
"""

import math
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class HistoryStagingSizing:
    """Worker parallelism for one history-staging run."""

    workers: int
    item_batch_rows: int


# Measured on the consumer/bootstrap/retained-history runs (2026-11-21..23):
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


@dataclass(frozen=True)
class StageResources:
    """Cluster resources for one stage run; every value derives from the population."""

    cpu: int
    memory_gb: int
    disk_gb: int
    timeout_s: int
    # Worker parallelism for Zephyr-based stages (embedding, curriculum compose).
    workers: int = 0
    worker_cpu: int = 0
    worker_memory_gb: int = 0

    def as_resources(self) -> dict[str, int]:
        """The job-level resource dict for the cluster submission."""
        return {"cpu": self.cpu, "memory_gb": self.memory_gb, "disk_gb": self.disk_gb, "timeout_s": self.timeout_s}


# Measured anchors from the admitted-population release runs (2026-11-21..23).
# Census: one streaming job over 60,581 admitted tasks (16 CPU / 64 GB / 200 GB /
# 4 h). Embedding: 16 Zephyr workers of 4 CPU / 16 GB under a 64 CPU / 256 GB /
# 72 h job. Curriculum compose: 8 workers of 4 CPU / 16 GB under a 32 CPU /
# 128 GB / 72 h job over the full composed population (~4.29 M tasks).
_ANCHOR_TASKS = 60_581
_ANCHOR_COMPOSE_TASKS = 4_291_291
_ANCHOR_EMBED_WORKERS = 16
# The anchor embedding run took 15.4 h on 16 CPU workers: ~246 tasks per worker-hour.
_ANCHOR_EMBED_HOURS = 15.4
_EMBED_TASKS_PER_WORKER_HOUR = _ANCHOR_TASKS / (_ANCHOR_EMBED_WORKERS * _ANCHOR_EMBED_HOURS)
# Size the worker pool so the fresh embedding finishes in about this many hours.
_EMBED_TARGET_HOURS = 2
_ANCHOR_COMPOSE_WORKERS = 8
_WORKER_CPU = 4
_WORKER_MEMORY_GB = 16


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def census_sizing(total_tasks: int) -> StageResources:
    """Scale the census job with the measured population's task count.

    The census streams each population once, so core count stays at the measured
    shape while memory, disk and wall clock scale with the task count.
    """
    if total_tasks <= 0:
        raise ValueError(f"census sizing needs a positive task count, got {total_tasks}")
    share = total_tasks / _ANCHOR_TASKS
    return StageResources(
        cpu=16,
        memory_gb=_clamp(round(64 * share), 16, 512),
        disk_gb=_clamp(round(200 * share), 50, 1000),
        timeout_s=_clamp(round(14_400 * share), 3_600, 43_200),
    )


# Measured on the per-release corpus-assembly runs (2026-11-21..23): 4 prepare
# workers over the anchor population.
_ANCHOR_PREPARE_WORKERS = 4
_MIN_PREPARE_WORKERS = 2
_MAX_PREPARE_WORKERS = 8


def corpus_sizing(total_tasks: int) -> StageResources:
    """Scale the corpus-assembly job with the measured population's task count.

    The corpus streams each frozen population's specs once (the census resource
    shape), and its prepare workers scale from the measured per-release run
    (4 workers over the anchor population).
    """
    sizing = census_sizing(total_tasks)
    workers = _clamp(
        round(_ANCHOR_PREPARE_WORKERS * total_tasks / _ANCHOR_TASKS),
        _MIN_PREPARE_WORKERS,
        _MAX_PREPARE_WORKERS,
    )
    return replace(sizing, workers=workers)


# Measured on the R40 cut-v22 QC-browse copy (2026-11-25, job remote_job-810554cf on
# cw-us-east-02a): 220 panels copied through ONE paced SQL client in 128.8 minutes
# (~31,000 statements). One worker copies the anchor population in that anchor time;
# workers scale linearly with the panel count so a release holds the anchor wall clock.
_QC_BROWSE_ANCHOR_PANELS = 220
_QC_BROWSE_ANCHOR_WORKERS = 8
_QC_BROWSE_MIN_WORKERS = 2
_QC_BROWSE_MAX_WORKERS = 16


def qc_browse_sizing(total_panels: int) -> StageResources:
    """Scale the QC-browse copy's panel workers with the planned panel count.

    Panels are disjoint (one pipeline key each), so the copy fans out across paced
    SQL clients; CPU stays at the measured 4-core shape because the work is
    network-bound.
    """
    if total_panels < 0:
        raise ValueError(f"qc browse sizing needs a nonnegative panel count, got {total_panels}")
    scaled = round(_QC_BROWSE_ANCHOR_WORKERS * total_panels / _QC_BROWSE_ANCHOR_PANELS)
    workers = max(_QC_BROWSE_MIN_WORKERS, min(_QC_BROWSE_MAX_WORKERS, scaled))
    return StageResources(
        cpu=4,
        memory_gb=16,
        disk_gb=64,
        timeout_s=14_400,
        workers=workers,
    )


def embedding_sizing(total_tasks: int) -> StageResources:
    """Size the CPU embedding pool to finish in ~``_EMBED_TARGET_HOURS`` at the measured rate.

    GPU embedding has no measured run yet: when a valid backend parity receipt makes
    it eligible, the embed stage keeps this worker curve and requests the accelerator
    through the typed manifest runtime instead of guessing a resource shape here.
    """
    if total_tasks <= 0:
        raise ValueError(f"embedding sizing needs a positive task count, got {total_tasks}")
    share = total_tasks / _ANCHOR_TASKS
    workers = _clamp(math.ceil(total_tasks / (_EMBED_TASKS_PER_WORKER_HOUR * _EMBED_TARGET_HOURS)), 2, 32)
    return StageResources(
        cpu=_clamp(workers * _WORKER_CPU, 8, 64),
        memory_gb=_clamp(workers * _WORKER_MEMORY_GB, 32, 256),
        disk_gb=_clamp(round(64 * share), 64, 1000),
        timeout_s=_clamp(round(259_200 * share), 7_200, 259_200),
        workers=workers,
        worker_cpu=_WORKER_CPU,
        worker_memory_gb=_WORKER_MEMORY_GB,
    )


def curriculum_compose_sizing(composed_tasks: int) -> StageResources:
    """Scale the curriculum-compose job with the composed population size."""
    if composed_tasks <= 0:
        raise ValueError(f"curriculum compose sizing needs a positive task count, got {composed_tasks}")
    share = composed_tasks / _ANCHOR_COMPOSE_TASKS
    workers = _clamp(round(_ANCHOR_COMPOSE_WORKERS * share), 2, 32)
    return StageResources(
        cpu=_clamp(workers * _WORKER_CPU, 8, 64),
        memory_gb=_clamp(workers * _WORKER_MEMORY_GB, 32, 256),
        disk_gb=_clamp(round(64 * share), 64, 1000),
        timeout_s=_clamp(round(259_200 * share), 7_200, 259_200),
        workers=workers,
        worker_cpu=_WORKER_CPU,
        worker_memory_gb=_WORKER_MEMORY_GB,
    )


# Measured on the R40 cut-v22 staged QC panels and the catalog-load replay they feed
# (2026-11-25, job remote_job-74886a67): 211 fresh panels totalling 1,389,220 records
# (472,382 raw reviews, 472,382 items, 427,820 batch links, 15,717 batches), where
# ~205 panels hold ~100 records and five monsters hold ~80% of the raw reviews. The
# replay fans page-range segments out over paced SQL clients; workers scale with the
# record count to hold the anchor wall clock, capped by the app's statement latency.
_CATALOG_REPLAY_ANCHOR_RECORDS = 1_389_220
_CATALOG_REPLAY_ANCHOR_WORKERS = 8
_CATALOG_REPLAY_MIN_WORKERS = 2
_CATALOG_REPLAY_MAX_WORKERS = 16


def catalog_replay_sizing(total_records: int) -> StageResources:
    """Scale the catalog-load QC replay's segment workers with the record count."""
    if total_records <= 0:
        raise ValueError(f"catalog replay sizing needs a positive record count, got {total_records}")
    scaled = round(_CATALOG_REPLAY_ANCHOR_WORKERS * total_records / _CATALOG_REPLAY_ANCHOR_RECORDS)
    workers = max(_CATALOG_REPLAY_MIN_WORKERS, min(_CATALOG_REPLAY_MAX_WORKERS, scaled))
    return StageResources(
        cpu=4,
        memory_gb=16,
        disk_gb=64,
        timeout_s=14_400,
        workers=workers,
    )
