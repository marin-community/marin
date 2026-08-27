# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded publication of externally collected metric snapshots into telemetry.

External collectors, such as the Prometheus scraper, produce ``MetricSnapshot``
values with explicit source semantics. ``MetricSnapshotPublisher`` validates and
enqueues them through the package's shared emit path, capping admission so one
oversized scrape cannot exhaust the export queue.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

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
class MetricPublishResult:
    """Bounded admission result for one metric snapshot publication.

    ``sample_limit_dropped_records`` never reached telemetry admission.
    ``telemetry_lost_records`` reached admission but could not enter the queue.
    """

    configured: bool
    enqueued_records: int
    sample_limit_dropped_records: int
    telemetry_lost_records: int


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
        input_records = len(snapshots)
        runtime = telemetry._runtime
        if runtime is None:
            return MetricPublishResult(False, 0, 0, 0)

        selected = snapshots[: self._max_records]
        enqueued = sum(self._enqueue(runtime, snapshot) for snapshot in selected)
        return MetricPublishResult(
            configured=True,
            enqueued_records=enqueued,
            sample_limit_dropped_records=max(0, input_records - len(selected)),
            telemetry_lost_records=len(selected) - enqueued,
        )

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
