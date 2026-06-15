# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Accelerator provisioning gauges, read from the controller's ``iris.provisioning``
finelog namespace.

The controller emits one structured row per slice provisioning outcome (the
autoscaler owns this — see ``iris.cluster.controller.autoscaler.provisioning``),
so this collector just windows and counts: no log parsing, no slice_id→pool
mapping, no create→outcome lookback. Each row already carries the authoritative
``resource_type`` / ``scale_group`` / ``zone`` and, for successes, the
create→ready latency.

I/O (the bounded finelog query) is separated from the pure ``aggregate`` so the
windowing/rollup logic is unit-testable without a live controller.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import NamedTuple

from finelog.client.log_client import LogClient
from sample import Sample

# Wire vocabulary for the iris.provisioning namespace. These mirror
# iris.cluster.controller.autoscaler.provisioning (PROVISIONING_NAMESPACE,
# ProvisioningOutcome) by value rather than importing them: this package pulls
# marin-iris from the rolling release, so importing the symbols would couple the
# probe's build to an iris release that postdates this change.
PROVISIONING_NAMESPACE = "iris.provisioning"
OUTCOME_READY = "ready"
OUTCOME_STOCKOUT = "stockout"  # create failed: no capacity
OUTCOME_ERROR = "error"  # create failed: other fault
OUTCOME_PREEMPTED = "preempted"  # reached ready, then lost at runtime
MAX_ROWS = 500_000

METRIC_READY = "provision_ready"
METRIC_STOCKOUT = "provision_stockout"
METRIC_ERROR = "provision_error"
METRIC_PREEMPTED = "provision_preempted"
METRIC_OUTCOMES = "provision_outcomes"  # create attempts resolved = ready + stockout + error
METRIC_LATENCY_SECONDS = "provision_latency_seconds"
METRIC_SUCCESS_RATIO = "provision_success_ratio"
METRIC_POOLS_PLACING = "provision_pools_placing"
METRIC_POOLS_STOCKOUT_DEAD = "provision_pools_stockout_dead"
METRIC_WINDOW_HOURS = "provision_window_hours"

# Label value marking the fleet-wide aggregate series (no pool labels).
FLEET = "fleet"
QUANTILES = {"p50": 0.50, "p95": 0.95}


class Row(NamedTuple):
    """One iris.provisioning row, in the column order selected by ``_query``."""

    resource_type: str
    scale_group: str
    zone: str
    outcome: str
    provision_latency_ms: int


# Columns selected from iris.provisioning, in order.
_COLUMNS = Row._fields


@dataclass(frozen=True)
class Pool:
    resource_type: str
    scale_group: str
    zone: str


@dataclass
class _Tally:
    ready: int = 0
    stockout: int = 0
    error: int = 0
    preempted: int = 0

    @property
    def create_attempts(self) -> int:
        """Resolved create attempts — excludes runtime deaths (preemptions)."""
        return self.ready + self.stockout + self.error


def _percentile(values_sorted: Sequence[float], q: float) -> float:
    """Nearest-rank percentile of a pre-sorted, non-empty sequence."""
    rank = max(0, min(len(values_sorted) - 1, math.ceil(q * len(values_sorted)) - 1))
    return values_sorted[rank]


def _latency_samples(labels: dict[str, str], latencies: list[float]) -> list[Sample]:
    if not latencies:
        return []
    ordered = sorted(latencies)
    return [
        Sample.of(METRIC_LATENCY_SECONDS, _percentile(ordered, q), quantile=name, **labels)
        for name, q in QUANTILES.items()
    ]


def _count_samples(labels: dict[str, str], tally: _Tally) -> list[Sample]:
    """The per-outcome count series, emitted identically for a pool and the fleet."""
    return [
        Sample.of(METRIC_READY, tally.ready, **labels),
        Sample.of(METRIC_STOCKOUT, tally.stockout, **labels),
        Sample.of(METRIC_ERROR, tally.error, **labels),
        Sample.of(METRIC_PREEMPTED, tally.preempted, **labels),
        Sample.of(METRIC_OUTCOMES, tally.create_attempts, **labels),
    ]


def _pool_samples(pool: Pool, tally: _Tally, latencies: list[float]) -> list[Sample]:
    labels = {"resource_type": pool.resource_type, "scale_group": pool.scale_group, "zone": pool.zone}
    return _count_samples(labels, tally) + _latency_samples(labels, latencies)


def aggregate(rows: Sequence[Row], *, window_hours: float) -> list[Sample]:
    """Roll up iris.provisioning rows (already windowed by the query) into gauges.

    Emits per-pool and fleet-wide series; create success rate is
    ready / (ready + stockout + error) — runtime deaths (preemptions) are counted
    separately and excluded from the rate. Latency quantiles use the create→ready
    latency of READY rows only.
    """
    tallies: dict[Pool, _Tally] = defaultdict(_Tally)
    latencies: dict[Pool, list[float]] = defaultdict(list)

    for resource_type, scale_group, zone, outcome, latency_ms in rows:
        pool = Pool(resource_type=resource_type, scale_group=scale_group, zone=zone)
        tally = tallies[pool]
        if outcome == OUTCOME_READY:
            tally.ready += 1
            latencies[pool].append(latency_ms / 1000)
        elif outcome == OUTCOME_STOCKOUT:
            tally.stockout += 1
        elif outcome == OUTCOME_ERROR:
            tally.error += 1
        elif outcome == OUTCOME_PREEMPTED:
            tally.preempted += 1

    samples: list[Sample] = []
    fleet = _Tally()
    fleet_latencies: list[float] = []
    pools_placing = 0
    pools_stockout_dead = 0
    for pool, tally in tallies.items():
        samples.extend(_pool_samples(pool, tally, latencies[pool]))
        fleet.ready += tally.ready
        fleet.stockout += tally.stockout
        fleet.error += tally.error
        fleet.preempted += tally.preempted
        fleet_latencies.extend(latencies[pool])
        if tally.ready > 0:
            pools_placing += 1
        elif tally.stockout + tally.error > 0:
            pools_stockout_dead += 1

    fleet_labels = {"scope": FLEET}
    samples.extend(_count_samples(fleet_labels, fleet))
    samples.extend(
        [
            Sample.of(METRIC_POOLS_PLACING, pools_placing, **fleet_labels),
            Sample.of(METRIC_POOLS_STOCKOUT_DEAD, pools_stockout_dead, **fleet_labels),
            Sample.of(METRIC_WINDOW_HOURS, window_hours, **fleet_labels),
        ]
    )
    if fleet.create_attempts > 0:
        samples.append(Sample.of(METRIC_SUCCESS_RATIO, fleet.ready / fleet.create_attempts, **fleet_labels))
    samples.extend(_latency_samples(fleet_labels, fleet_latencies))
    return samples


def _query(finelog: LogClient, cutoff: datetime) -> list[Row]:
    """Fetch iris.provisioning rows at/after ``cutoff`` (a tz-naive UTC datetime,
    matching the namespace's ``ts`` column)."""
    columns = ", ".join(_COLUMNS)
    table = finelog.query(
        f'SELECT {columns} FROM "{PROVISIONING_NAMESPACE}" '
        f"WHERE ts >= TIMESTAMP '{cutoff:%Y-%m-%d %H:%M:%S}' "
        f"ORDER BY ts",
        max_rows=MAX_ROWS,
    )
    return [Row(*row) for row in zip(*(table.column(c).to_pylist() for c in _COLUMNS), strict=True)]


def collect_provisioning(
    finelog: LogClient, *, window_hours: float, now: Callable[[], float] = time.time
) -> list[Sample]:
    """Query iris.provisioning over the trailing window and aggregate into gauges."""
    cutoff = datetime.fromtimestamp(now() - window_hours * 3600, tz=UTC).replace(tzinfo=None)
    return aggregate(_query(finelog, cutoff), window_hours=window_hours)
