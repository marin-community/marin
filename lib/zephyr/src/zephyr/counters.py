# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""User-defined counters for Zephyr tasks.

Task code can increment named counters during execution; counters are
aggregated across all tasks and exposed via the coordinator's ``get_counters()``
actor method.

**Pipeline-level counters** (aggregated across all stages and shards)::

    from zephyr import counters

    counters.pipeline.update_counter("documents_processed", 100)
    counters.pipeline.set_counter("mem_bytes", current_rss)
    counters.pipeline.set_aggregation("mem_bytes", counters.Aggregation.MAX)

**Stage-level counters** (scoped to a specific stage)::

    stage = counters.stage("tokenize")
    stage.update_counter("tokens_produced", n)

    # Or use the current running stage automatically:
    stage = counters.current_stage()
    stage.update_counter("tokens_produced", n)

**Aggregation** — counters default to SUM across all worker snapshots.
Override per-counter using ``set_aggregation``::

    counters.pipeline.set_aggregation("peak_mem_bytes", counters.Aggregation.MAX)
    counters.pipeline.set_aggregation("cpu_percent", counters.Aggregation.AVERAGE)

Supported aggregations: ``SUM``, ``AVERAGE``, ``MAX``, ``MIN``.

Counter values are accumulated in-memory on each worker and sent to the
coordinator periodically via heartbeats and as a final snapshot on task
completion.

Outside of a Zephyr worker context, all calls are silent no-ops.
"""

import logging

from zephyr.worker_context import Aggregation, _worker_ctx_var

logger = logging.getLogger(__name__)


# Built-in pipeline counters emitted by zephyr's readers, writers, and planner.
# Defined here (rather than inline at each call site) so the name stays consistent
# across the modules that emit them — a typo at one site would otherwise create a
# silently-separate counter.
RECORDS_IN = "zephyr/records_in"
"""Records read by the loaders (jsonl/parquet/vortex/zip)."""
RECORDS_OUT = "zephyr/records_out"
"""Records written by the output writers (jsonl/parquet/vortex/binary)."""
PARTITIONS_SKIPPED = "zephyr/partitions_skipped"
"""Output partitions skipped because the target already existed (``skip_existing``)."""


class ScopedCounters:
    """Counter namespace scoped to a stage (or pipeline-level when stage is None).

    Instantiate via ``counters.pipeline``, ``counters.stage(name)``, or
    ``counters.current_stage()`` rather than directly.
    """

    def __init__(self, stage: str | None) -> None:
        self._stage = stage

    def set_counter(self, name: str, value: int | float) -> None:
        """Overwrite a named counter with ``value`` (not additive).

        Use for point-in-time metrics (cpu percent, current RSS) that should
        replace the previous value rather than accumulate. No-op outside a
        Zephyr worker context.
        """
        worker = _worker_ctx_var.get()
        if worker is None:
            return
        worker.set_counter(name, value, stage=self._stage)

    def update_counter(self, name: str, value: int | float) -> None:
        """Update a counter according to its aggregation mode.

        - ``SUM``: adds ``value`` (same as ``increment``).
        - ``MAX`` / ``MIN``: keeps the running max or min.
        - ``AVERAGE``: maintains a rolling average — ``value`` always holds
          the current mean, updated with each call.

        On first call (no prior entry) the counter is initialised to ``value``
        regardless of aggregation. No-op outside a Zephyr worker context.
        """
        worker = _worker_ctx_var.get()
        if worker is None:
            return
        worker.update_counter(name, value, stage=self._stage)

    def set_aggregation(self, name: str, agg: Aggregation) -> None:
        """Set the aggregation function used when reducing this counter across workers.

        Aggregation is per counter name (not per stage). Defaults to
        ``Aggregation.SUM`` if not set. No-op outside a Zephyr worker.
        """
        worker = _worker_ctx_var.get()
        if worker is None:
            return
        worker.set_aggregation(name, agg)

    def get_counters(self) -> dict[str, int | float]:
        """Return a snapshot of counters in this scope.

        Returns only counters whose stage matches this scope's stage.
        Returns an empty dict outside a worker context.
        """
        worker = _worker_ctx_var.get()
        if worker is None:
            return {}
        snap = worker.get_counter_snapshot()
        return {k: e.value for k, e in snap.counters.items() if e.stage == self._stage}


pipeline: ScopedCounters = ScopedCounters(stage=None)
"""Pipeline-level counter namespace. Counters here have no stage scope."""


def stage(name: str) -> ScopedCounters:
    """Return a ``ScopedCounters`` scoped to the named stage."""
    return ScopedCounters(stage=name)


def current_stage() -> ScopedCounters:
    """Return a ``ScopedCounters`` scoped to the stage currently running on this worker.

    Falls back to pipeline scope outside a Zephyr worker context.
    """
    worker = _worker_ctx_var.get()
    if worker is None:
        return pipeline
    return ScopedCounters(stage=worker.current_stage_name())
