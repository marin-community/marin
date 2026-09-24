# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded publication of externally collected metric snapshots into telemetry.

External collectors, such as the Prometheus scraper, produce ``MetricSnapshot``
values with explicit source semantics. ``MetricSnapshotPublisher`` validates and
enqueues them through the package's shared emit path, capping admission so one
oversized scrape cannot exhaust the export queue.
"""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import TypeVar

from rigging import telemetry
from rigging.telemetry import serialization


@dataclass(frozen=True)
class MetricSnapshot:
    """One externally collected metric value with explicit source semantics."""

    name: str
    value: float
    unit: str
    attributes: Mapping[str, str]
    source_kind: str
    source_temporality: str


@dataclass(frozen=True)
class HistogramSnapshot:
    """One cumulative explicit-bucket snapshot with per-bin integer counts."""

    name: str
    explicit_bounds: tuple[float, ...]
    bucket_counts: tuple[int, ...]
    count: int
    sum: float
    unit: str
    attributes: Mapping[str, str]
    timestamp_ms: int
    producer_epoch: str | None = None
    sequence: int | None = None


@dataclass(frozen=True)
class MetricPublishResult:
    """Bounded admission result for one metric snapshot publication.

    ``sample_limit_dropped_records`` never reached telemetry admission.
    ``telemetry_lost_records`` reached admission but could not enter the queue.
    """

    configured: bool
    enqueued_records: int
    sample_limit_dropped_records: int
    telemetry_lost_records: int


_Snapshot = TypeVar("_Snapshot")


def _publish_snapshots(
    snapshots: Sequence[_Snapshot],
    max_records: int,
    enqueue: Callable[["telemetry._Runtime", _Snapshot], int],
) -> MetricPublishResult:
    runtime = telemetry._runtime
    if runtime is None:
        return MetricPublishResult(False, 0, 0, 0)
    selected = snapshots[:max_records]
    enqueued = sum(enqueue(runtime, snapshot) for snapshot in selected)
    return MetricPublishResult(
        configured=True,
        enqueued_records=enqueued,
        sample_limit_dropped_records=len(snapshots) - len(selected),
        telemetry_lost_records=len(selected) - enqueued,
    )


class MetricSnapshotPublisher:
    """Publish bounded batches of externally collected metric snapshots."""

    def __init__(
        self,
        *,
        max_records: int,
        attributes: Mapping[str, str] | None = None,
    ) -> None:
        if max_records <= 0:
            raise ValueError("max_records must be positive")
        common_attributes = dict(attributes or {})
        serialization.validate_attributes(common_attributes)
        self._max_records = max_records
        self._attributes = common_attributes

    def publish(self, snapshots: Sequence[MetricSnapshot]) -> MetricPublishResult:
        """Validate and enqueue at most ``max_records`` snapshots without blocking."""
        return _publish_snapshots(snapshots, self._max_records, self._enqueue)

    def _enqueue(self, runtime: "telemetry._Runtime", snapshot: MetricSnapshot) -> int:
        """Return 1 when ``snapshot`` enters the queue, else 0, counting any loss."""
        try:
            if not isinstance(snapshot, MetricSnapshot):
                raise TypeError("snapshots must contain MetricSnapshot values")
            if snapshot.source_temporality not in {telemetry.CURRENT_SNAPSHOT, telemetry.CUMULATIVE_SNAPSHOT}:
                raise ValueError("source_temporality must be current_snapshot or cumulative_snapshot")
            attributes = {
                **snapshot.attributes,
                **self._attributes,
                **telemetry.snapshot_attributes(snapshot.source_kind, snapshot.source_temporality),
            }
        except Exception:
            # One malformed external series must not suppress the rest of the bounded batch.
            runtime.count_lost()
            return 0
        return int(
            telemetry._emit_to_runtime(
                runtime,
                "gauge",
                snapshot.name,
                value=snapshot.value,
                unit=snapshot.unit,
                attributes=attributes,
            )
        )


_MAX_HISTOGRAM_INTEGER = (1 << 63) - 1
_MAX_HISTOGRAM_BOUNDS = 512


def _histogram_body(snapshot: HistogramSnapshot) -> dict[str, object]:
    if not isinstance(snapshot, HistogramSnapshot):
        raise TypeError("snapshots must contain HistogramSnapshot values")
    if (
        isinstance(snapshot.timestamp_ms, bool)
        or not isinstance(snapshot.timestamp_ms, int)
        or not 0 <= snapshot.timestamp_ms <= _MAX_HISTOGRAM_INTEGER
    ):
        raise ValueError("timestamp_ms must be a nonnegative signed-64 integer")
    if (snapshot.producer_epoch is None) != (snapshot.sequence is None):
        raise ValueError("producer_epoch and sequence must be provided together")
    if snapshot.producer_epoch is not None:
        serialization.validate_string(snapshot.producer_epoch, "producer_epoch")
    if snapshot.sequence is not None:
        if (
            isinstance(snapshot.sequence, bool)
            or not isinstance(snapshot.sequence, int)
            or not 0 <= snapshot.sequence <= _MAX_HISTOGRAM_INTEGER
        ):
            raise ValueError("sequence must be a nonnegative signed-64 integer")
    bounds = snapshot.explicit_bounds
    if len(bounds) > _MAX_HISTOGRAM_BOUNDS or any(
        isinstance(bound, bool) or not isinstance(bound, (int, float)) or not math.isfinite(bound) for bound in bounds
    ):
        raise ValueError("explicit bounds must be finite numbers")
    if any(left >= right for left, right in pairwise(bounds)):
        raise ValueError("explicit bounds must increase strictly")
    counts = snapshot.bucket_counts
    if len(counts) != len(bounds) + 1 or any(
        isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= _MAX_HISTOGRAM_INTEGER
        for value in counts
    ):
        raise ValueError("bucket counts must be nonnegative signed-64 integers, one per bucket")
    if isinstance(snapshot.count, bool) or not isinstance(snapshot.count, int) or snapshot.count != sum(counts):
        raise ValueError("count must equal the sum of bucket counts")
    if snapshot.count > _MAX_HISTOGRAM_INTEGER:
        raise ValueError("count exceeds the signed-64 range")
    if isinstance(snapshot.sum, bool) or not isinstance(snapshot.sum, (int, float)) or not math.isfinite(snapshot.sum):
        raise ValueError("sum must be finite")
    body: dict[str, object] = {
        "encoding": "explicit_bucket_v1",
        "aggregation_temporality": "cumulative",
        "explicit_bounds": list(bounds),
        "bucket_counts": list(counts),
        "count": snapshot.count,
        "sum": snapshot.sum,
    }
    if snapshot.producer_epoch is not None:
        body["producer_epoch"] = snapshot.producer_epoch
        body["sequence"] = snapshot.sequence
    return body


class HistogramSnapshotPublisher:
    """Publish opt-in aggregate histograms without changing raw observations."""

    def __init__(self, *, max_records: int, attributes: Mapping[str, str] | None = None) -> None:
        if max_records <= 0:
            raise ValueError("max_records must be positive")
        common_attributes = dict(attributes or {})
        serialization.validate_attributes(common_attributes)
        self._max_records = max_records
        self._attributes = common_attributes

    def publish(self, snapshots: Sequence[HistogramSnapshot]) -> MetricPublishResult:
        """Admit at most ``max_records`` whole families, counting any rejected family."""
        return _publish_snapshots(snapshots, self._max_records, self._enqueue)

    def _enqueue(self, runtime: "telemetry._Runtime", snapshot: HistogramSnapshot) -> int:
        try:
            body = _histogram_body(snapshot)
            attributes = {
                **snapshot.attributes,
                **self._attributes,
                **telemetry.snapshot_attributes("histogram", telemetry.CUMULATIVE_SNAPSHOT),
            }
        except Exception:
            runtime.count_lost()
            return 0
        return int(
            telemetry._emit_histogram_to_runtime(
                runtime,
                snapshot.name,
                body=body,
                timestamp_ms=snapshot.timestamp_ms,
                unit=snapshot.unit,
                attributes=attributes,
            )
        )
