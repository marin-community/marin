# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded publication of externally collected metric snapshots into telemetry.

External collectors, such as the Prometheus scraper, produce ``MetricSnapshot``
values with explicit source semantics. ``MetricSnapshotPublisher`` validates and
enqueues them through the package's shared emit path, capping admission so one
oversized scrape cannot exhaust the export queue.
"""

import hashlib
import json
import math
import struct
import threading
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise

from rigging import telemetry
from rigging.telemetry import serialization

CUMULATIVE_HISTOGRAM_ENCODING = "explicit_bounds_cumulative_with_delta_v1"
CUMULATIVE_HISTOGRAM_BUNDLE_ENCODING = "explicit_bounds_cumulative_bundle_v2"
_MAX_SIGNED_64 = (1 << 63) - 1
_MAX_HISTOGRAM_BUCKETS = 128
_MAX_HISTOGRAMS_PER_BUNDLE = 64
_MAX_HISTOGRAM_BASELINES = 8_192
_HISTOGRAM_QUERY_DELIMITER = "|"
_HISTOGRAM_PUBLICATION_ATTRIBUTES = frozenset(
    {
        "histogram_collection_timestamp_ms",
        "histogram_publication_id",
        "histogram_sample_sequence",
    }
)


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


@dataclass(frozen=True)
class CumulativeHistogramSnapshot:
    """One source-provided cumulative histogram within a collection snapshot."""

    name: str
    finite_bounds: tuple[float, ...]
    cumulative_counts: tuple[int, ...]
    count: int
    total: float
    unit: str
    attributes: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class CumulativeHistogramBundleSnapshot:
    """A source-timestamped group of cumulative histograms from one producer."""

    name: str
    histograms: tuple[CumulativeHistogramSnapshot, ...]
    attributes: Mapping[str, str]
    timestamp_ms: int
    sample_sequence: int


@dataclass(frozen=True)
class _HistogramBaseline:
    timestamp_ms: int
    sample_sequence: int
    series: str
    schema: str
    cumulative_counts: tuple[int, ...]
    count: int
    total: float


@dataclass(frozen=True)
class _ValidatedHistogram:
    body: dict[str, object]
    component_kinds: tuple[str, ...]
    component_bounds: tuple[str, ...]
    series: str
    schema: str


@dataclass(frozen=True)
class _HistogramDelta:
    body: dict[str, object]
    component_values: tuple[str, ...]
    valid: bool
    advance_baseline: bool


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


class CumulativeHistogramBundleSnapshotPublisher:
    """Publish bounded producer snapshots without expanding histogram buckets."""

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
        self._baselines: OrderedDict[tuple[str, ...], _HistogramBaseline] = OrderedDict()
        self._baseline_lock = threading.Lock()

    def publish(self, snapshots: Sequence[CumulativeHistogramBundleSnapshot]) -> MetricPublishResult:
        """Validate and enqueue at most ``max_records`` whole producer bundles."""
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

    def _enqueue(self, runtime: "telemetry._Runtime", snapshot: CumulativeHistogramBundleSnapshot) -> int:
        with self._baseline_lock:
            try:
                body, baselines = _validated_histogram_bundle(snapshot, self._baselines)
                attributes = {
                    **snapshot.attributes,
                    **self._attributes,
                    **telemetry.snapshot_attributes("histogram_bundle", telemetry.CUMULATIVE_SNAPSHOT),
                    "histogram_encoding": CUMULATIVE_HISTOGRAM_BUNDLE_ENCODING,
                }
                serialization.validate_attributes(attributes)
            except (TypeError, ValueError):
                runtime.count_lost()
                return 0
            accepted = telemetry._emit_structured_event_to_runtime(
                runtime,
                snapshot.name,
                body=body,
                timestamp_ms=snapshot.timestamp_ms,
                attributes=attributes,
            )
            if accepted:
                for key, baseline in baselines.items():
                    self._baselines[key] = baseline
                    self._baselines.move_to_end(key)
                while len(self._baselines) > _MAX_HISTOGRAM_BASELINES:
                    self._baselines.popitem(last=False)
            return int(accepted)


def _validated_histogram_bundle(
    snapshot: CumulativeHistogramBundleSnapshot,
    baselines: Mapping[tuple[str, ...], _HistogramBaseline],
) -> tuple[dict[str, object], dict[tuple[str, ...], _HistogramBaseline]]:
    if not isinstance(snapshot, CumulativeHistogramBundleSnapshot):
        raise TypeError("snapshots must contain CumulativeHistogramBundleSnapshot values")
    serialization.validate_string(snapshot.name, "name")
    serialization.validate_attributes(snapshot.attributes)
    _signed_nonnegative_integer(snapshot.timestamp_ms, "timestamp_ms")
    sample_sequence = _signed_nonnegative_integer(snapshot.sample_sequence, "sample_sequence")
    if not snapshot.histograms:
        raise ValueError("histogram bundles must not be empty")
    if len(snapshot.histograms) > _MAX_HISTOGRAMS_PER_BUNDLE:
        raise ValueError(f"histogram bundles may contain at most {_MAX_HISTOGRAMS_PER_BUNDLE} histograms")

    histograms: dict[str, object] = {}
    next_baselines: dict[tuple[str, ...], _HistogramBaseline] = {}
    bundle_series = cumulative_histogram_series(
        {key: value for key, value in snapshot.attributes.items() if key not in _HISTOGRAM_PUBLICATION_ATTRIBUTES}
    )
    delta_family_names: list[str] = []
    delta_family_indexes: list[str] = []
    delta_component_kinds: list[str] = []
    delta_component_bounds: list[str] = []
    delta_component_values: list[str] = []
    for histogram in snapshot.histograms:
        if _HISTOGRAM_QUERY_DELIMITER in histogram.name:
            raise ValueError(f"histogram names must not contain {_HISTOGRAM_QUERY_DELIMITER!r} in a bundle")
        validated = _validated_histogram(histogram)
        if histogram.name in histograms:
            raise ValueError(f"duplicate histogram name: {histogram.name}")
        baseline_key = (
            snapshot.name,
            bundle_series,
            histogram.name,
        )
        next_baseline = _HistogramBaseline(
            timestamp_ms=snapshot.timestamp_ms,
            sample_sequence=sample_sequence,
            series=validated.series,
            schema=validated.schema,
            cumulative_counts=tuple(histogram.cumulative_counts),
            count=histogram.count,
            total=histogram.total,
        )
        delta = _histogram_delta(baselines.get(baseline_key), next_baseline)
        body = {**validated.body, **delta.body}
        if delta.valid:
            family_index = str(len(delta_family_names))
            delta_family_names.append(histogram.name)
            delta_family_indexes.extend((family_index,) * len(delta.component_values))
            delta_component_kinds.extend(validated.component_kinds)
            delta_component_bounds.extend(validated.component_bounds)
            delta_component_values.extend(delta.component_values)
        if delta.advance_baseline:
            next_baselines[baseline_key] = next_baseline
        histograms[histogram.name] = body
    bundle: dict[str, object] = {
        "encoding": CUMULATIVE_HISTOGRAM_BUNDLE_ENCODING,
        "histograms": histograms,
        "sample_sequence": sample_sequence,
    }
    if delta_component_values:
        query_index = {
            "delta_component_bounds_csv": _HISTOGRAM_QUERY_DELIMITER.join(delta_component_bounds),
            "delta_component_kinds_csv": _HISTOGRAM_QUERY_DELIMITER.join(delta_component_kinds),
            "delta_component_values_csv": _HISTOGRAM_QUERY_DELIMITER.join(delta_component_values),
            "delta_family_indexes_csv": _HISTOGRAM_QUERY_DELIMITER.join(delta_family_indexes),
            "delta_family_names_csv": _HISTOGRAM_QUERY_DELIMITER.join(delta_family_names),
        }
        for field, value in query_index.items():
            serialization.validate_string(value, field)
        bundle.update(query_index)
    return bundle, next_baselines


def _histogram_delta(
    previous: _HistogramBaseline | None,
    current: _HistogramBaseline,
) -> _HistogramDelta:
    """Compute one interval and whether admission should replace its baseline."""
    if previous is None:
        return _HistogramDelta({"delta_valid": False}, (), False, True)
    ordered = current.sample_sequence > previous.sample_sequence and current.timestamp_ms >= previous.timestamp_ms
    if not ordered:
        return _HistogramDelta({"delta_valid": False}, (), False, False)
    if current.series != previous.series or current.schema != previous.schema:
        return _HistogramDelta({"delta_valid": False}, (), False, True)
    monotonic = (
        len(current.cumulative_counts) == len(previous.cumulative_counts)
        and all(now >= before for now, before in zip(current.cumulative_counts, previous.cumulative_counts, strict=True))
        and current.count >= previous.count
        and current.total >= previous.total
    )
    if not monotonic:
        return _HistogramDelta({"delta_valid": False}, (), False, True)

    delta_counts = tuple(
        now - before for now, before in zip(current.cumulative_counts, previous.cumulative_counts, strict=True)
    )
    delta_count = current.count - previous.count
    delta_sum = current.total - previous.total
    component_values = (*map(str, delta_counts), str(delta_count), repr(delta_sum))
    return _HistogramDelta(
        {
            "delta_count": delta_count,
            "delta_cumulative_counts": delta_counts,
            "delta_from_sample_sequence": previous.sample_sequence,
            "delta_from_timestamp_ms": previous.timestamp_ms,
            "delta_sum": delta_sum,
            "delta_valid": True,
        },
        tuple(component_values),
        True,
        True,
    )


def _validated_histogram(snapshot: CumulativeHistogramSnapshot) -> _ValidatedHistogram:
    if not isinstance(snapshot, CumulativeHistogramSnapshot):
        raise TypeError("histograms must contain CumulativeHistogramSnapshot values")
    serialization.validate_string(snapshot.name, "name")
    if snapshot.unit:
        serialization.validate_string(snapshot.unit, "unit")
    attributes = dict(snapshot.attributes or {})
    serialization.validate_attributes(attributes)
    count = _signed_nonnegative_integer(snapshot.count, "count")

    bounds = tuple(_finite_float(bound, "finite bound") for bound in snapshot.finite_bounds)
    if any(left >= right for left, right in pairwise(bounds)):
        raise ValueError("finite bounds must be strictly increasing")
    counts = tuple(_signed_nonnegative_integer(value, "cumulative count") for value in snapshot.cumulative_counts)
    if len(counts) != len(bounds) + 1:
        raise ValueError("cumulative counts must contain one overflow bucket")
    if len(counts) > _MAX_HISTOGRAM_BUCKETS:
        raise ValueError(f"cumulative counts must contain at most {_MAX_HISTOGRAM_BUCKETS} buckets")
    if any(left > right for left, right in pairwise(counts)):
        raise ValueError("cumulative counts must be nondecreasing")
    if counts[-1] != count:
        raise ValueError("overflow bucket must equal count")
    total = _finite_float(snapshot.total, "sum")
    if total < 0:
        raise ValueError("sum must be nonnegative")
    component_kinds = ("bucket",) * len(counts) + ("count", "sum")
    component_bounds = (*map(repr, bounds), "+Inf", "_", "_")
    schema = _cumulative_histogram_schema(bounds)
    series = cumulative_histogram_series(attributes)

    return _ValidatedHistogram(
        body={
            "attributes": attributes,
            "count": count,
            "cumulative_counts": counts,
            "encoding": CUMULATIVE_HISTOGRAM_ENCODING,
            "finite_bounds": bounds,
            "schema": schema,
            "series": series,
            "sum": total,
            "unit": snapshot.unit,
        },
        component_kinds=component_kinds,
        component_bounds=component_bounds,
        series=series,
        schema=schema,
    )


def cumulative_histogram_schema(finite_bounds: Sequence[float]) -> str:
    """Return the encoding-specific identity for one exact finite-bound layout."""
    bounds = tuple(_finite_float(bound, "finite bound") for bound in finite_bounds)
    if any(left >= right for left, right in pairwise(bounds)):
        raise ValueError("finite bounds must be strictly increasing")
    return _cumulative_histogram_schema(bounds)


def _cumulative_histogram_schema(bounds: tuple[float, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(CUMULATIVE_HISTOGRAM_ENCODING.encode())
    for bound in bounds:
        digest.update(struct.pack("!d", bound))
    return digest.hexdigest()


def cumulative_histogram_series(attributes: Mapping[str, str]) -> str:
    """Return a stable identity for one exact attribute set."""
    normalized = dict(attributes)
    serialization.validate_attributes(normalized)
    canonical = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _signed_nonnegative_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= _MAX_SIGNED_64:
        raise ValueError(f"{field} must be a nonnegative signed 64-bit integer")
    return value


def _finite_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result
