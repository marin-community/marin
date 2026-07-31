# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded process-local telemetry primitives.

Instrument handles are safe to declare before an entrypoint configures
telemetry. Emission is a no-op until then and never performs I/O. A daemon
exporter snapshots bounded, delivery-homogeneous batches for a local agent.
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import math
import os
import re
import threading
import time
import uuid
from collections import Counter as Counts
from collections import deque
from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from enum import StrEnum
from typing import Any, Protocol, SupportsFloat

import requests

from rigging.auth import TokenProvider
from rigging.telemetry_catalog import load_catalog

_CATALOG = load_catalog()
CATALOG_VERSION = _CATALOG.catalog_version
DEFAULT_CARDINALITY_LIMIT = 100
DEFAULT_MAX_QUEUE_RECORDS = 10_000
DEFAULT_MAX_QUEUE_BYTES = 16 * 1024 * 1024
MAX_SHUTDOWN_TIMEOUT = timedelta(seconds=5)
_SCHEMA_VERSION = 1
_CONTENT_TYPE = "application/json"
_MAX_STRING_BYTES = 64 * 1024
_SERIES_ID_DOMAIN = b"finelog.telemetry.series.v1\0"
_BATCH_ID_PLACEHOLDER = "0" * 36
_MAX_RETRY_AFTER = timedelta(seconds=60)
_MAX_I64 = (1 << 63) - 1
_INSTRUMENT_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_SCOPE_NAME = re.compile(r"^[a-z][a-z0-9_.]*$")
_LOSS_CARDINALITY_OVERFLOW = "cardinality_overflow"
_LOSS_CONFIGURATION_CONFLICT = "configuration_conflict"
_LOSS_DESCRIPTOR_CONFLICT = "descriptor_conflict"
_LOSS_EVENT_QUEUE_OVERFLOW = "event_queue_overflow"
_LOSS_INERT_HANDLE = "inert_handle"
_LOSS_INVALID_CONFIGURATION = "invalid_configuration"
_LOSS_INVALID_DESCRIPTOR = "invalid_descriptor"
_LOSS_INVALID_EMISSION = "invalid_emission"
_LOSS_RUNTIME_CONTENTION = "runtime_contention"
_LOSS_EXPORT_RETRY = "export_retry"
_LOSS_EXPORT_TERMINAL = "export_terminal"
_LOSS_FORKED_PROCESS = "forked_process"
_RETRYABLE_HTTP_STATUS = frozenset({429, 502, 503, 504})
_RENEWABLE_CREDENTIAL_HTTP_STATUS = frozenset({401})
_SEVERITY_NUMBER = {
    "debug": 5,
    "info": 9,
    "warning": 13,
    "error": 17,
    "critical": 21,
}


class DeliveryClass(StrEnum):
    COALESCING = "coalescing"
    BUFFERED = "buffered"
    DURABLE = "durable"


_DELIVERY_LANES = (
    DeliveryClass.DURABLE,
    DeliveryClass.BUFFERED,
    DeliveryClass.COALESCING,
)


class InstrumentKind(StrEnum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"


class Temporality(StrEnum):
    CUMULATIVE = "cumulative"
    UNSPECIFIED = "unspecified"


class Maturity(StrEnum):
    EXPERIMENTAL = "experimental"
    STABLE = "stable"


class Severity(StrEnum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class AttributeSpec:
    name: str
    allowed_values: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MetricDescriptor:
    name: str
    scope: str
    description: str
    unit: str
    instrument_kind: InstrumentKind
    temporality: Temporality
    attributes: tuple[AttributeSpec, ...]
    buckets: tuple[float, ...]
    owner: str
    cadence: timedelta
    delivery_class: DeliveryClass
    cardinality_limit: int = DEFAULT_CARDINALITY_LIMIT
    maturity: Maturity = Maturity.EXPERIMENTAL


@dataclass(frozen=True, slots=True)
class EventDescriptor:
    event_name: str
    owner: str
    attribute_names: tuple[str, ...]
    delivery_class: DeliveryClass


@dataclass(frozen=True, slots=True)
class Resource:
    service_name: str
    service_instance_id: str
    role: str | None = None
    root_run_uid: str | None = None
    service_version: str | None = None
    run_id_alias: str | None = None
    iris_job_id: str | None = None
    iris_task_id: str | None = None
    task_index: int | None = None
    attempt_id: int | None = None
    attempt_uid: str | None = None
    worker_id: str | None = None
    node_id: str | None = None
    pod_uid: str | None = None
    container_id: str | None = None
    rank: int | None = None
    process_index: int | None = None
    actor_id: str | None = None
    engine_id: str | None = None
    repository: str | None = None
    git_revision: str | None = None
    image_digest: str | None = None
    model_id: str | None = None
    model_revision: str | None = None
    policy_step: int | None = None
    owner: str | None = None
    experiment_issue: int | None = None
    cluster: str | None = None
    entity_authority: str | None = None
    entity_type: str | None = None
    entity_uid: str | None = None


@dataclass(frozen=True, slots=True)
class AgentWriteRequest:
    endpoint: str
    batch_id: str
    body: bytes
    content_type: str
    token_provider: TokenProvider
    timeout: timedelta


@dataclass(frozen=True, slots=True)
class AgentWriteAck:
    batch_id: str
    status: str
    durability: str


class AgentTransport(Protocol):
    """Synchronous local-agent boundary used only by the exporter thread."""

    def write(self, request: AgentWriteRequest) -> AgentWriteAck:
        """Durably admit one immutable producer batch."""
        ...


class RetryableExportError(Exception):
    """The exact batch may be retried without changing its ID or bytes."""

    def __init__(self, message: str, *, retry_after: timedelta | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class TerminalExportError(Exception):
    """The agent rejected these bytes permanently."""


class RequestsAgentTransport:
    """POST immutable producer batches to the local telemetry agent."""

    def write(self, request: AgentWriteRequest) -> AgentWriteAck:
        token = request.token_provider.get_token()
        if not token:
            raise RetryableExportError("telemetry agent credential is unavailable")
        try:
            response = requests.post(
                request.endpoint,
                data=request.body,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": request.content_type,
                    "Idempotency-Key": request.batch_id,
                },
                timeout=request.timeout.total_seconds(),
            )
        except requests.RequestException as error:
            raise RetryableExportError("telemetry agent request failed") from error
        if response.status_code in _RETRYABLE_HTTP_STATUS | _RENEWABLE_CREDENTIAL_HTTP_STATUS:
            raise RetryableExportError(
                f"telemetry agent returned HTTP {response.status_code}",
                retry_after=_retry_after(response.headers.get("Retry-After")),
            )
        if 400 <= response.status_code < 500 or response.status_code == 500:
            raise TerminalExportError(f"telemetry agent returned HTTP {response.status_code}")
        if response.status_code not in (200, 201):
            raise RetryableExportError(f"telemetry agent returned HTTP {response.status_code}")
        try:
            payload = response.json()
            ack = AgentWriteAck(
                batch_id=payload["batch_id"],
                status=payload["status"],
                durability=payload["durability"],
            )
        except (KeyError, TypeError, ValueError) as error:
            raise RetryableExportError("telemetry agent returned an invalid acknowledgement") from error
        return ack


@dataclass(frozen=True, slots=True)
class HttpExporterConfig:
    endpoint: str
    token_provider: TokenProvider
    export_interval: timedelta = timedelta(seconds=5)
    request_timeout: timedelta = timedelta(seconds=2)
    shutdown_timeout: timedelta = timedelta(seconds=1)
    max_queue_records: int = DEFAULT_MAX_QUEUE_RECORDS
    max_queue_bytes: int = DEFAULT_MAX_QUEUE_BYTES


@dataclass(frozen=True, slots=True)
class RuntimeStatus:
    configured: bool
    stopped: bool
    service_instance_id: str | None
    accepted_emissions: int
    metric_series: int
    queued_events: int
    queued_event_bytes: int
    in_flight_batch_id: str | None
    in_flight_records: int
    in_flight_bytes: int
    losses: tuple[tuple[str, int], ...]


@dataclass(slots=True)
class _HistogramState:
    bucket_counts: list[int]
    count: int = 0
    sum: float = 0.0


@dataclass(frozen=True, slots=True)
class _EventRecord:
    event_name: str
    delivery_class: DeliveryClass
    severity: Severity
    outcome: str | None
    body: str | None
    trace_id: str | None
    span_id: str | None
    attributes: tuple[tuple[str, str], ...]
    event_ts_unix_nano: int
    estimated_bytes: int


SeriesKey = tuple[str, str, tuple[tuple[str, str], ...]]
MetricIdentity = tuple[InstrumentKind, SeriesKey]


@dataclass(frozen=True, slots=True)
class _HistogramSnapshot:
    bucket_counts: tuple[int, ...]
    count: int
    sum: float


@dataclass(frozen=True, slots=True)
class _MetricSnapshot:
    descriptor: MetricDescriptor
    key: SeriesKey
    value: float | _HistogramSnapshot
    state_kind: InstrumentKind


@dataclass(frozen=True, slots=True)
class _BatchReservation:
    batch_id: str
    delivery_class: DeliveryClass
    lane_index: int
    observed_ts_unix_nano: int
    events: tuple[_EventRecord, ...]
    metrics: tuple[_MetricSnapshot, ...]
    metric_start_cursor: int
    metric_total: int
    reserved_records: int
    reserved_bytes: int
    metric_first: bool
    next_metric_first: bool


@dataclass(frozen=True, slots=True)
class _PendingBatch:
    batch_id: str
    delivery_class: DeliveryClass
    body: bytes
    record_count: int
    next_metric_cursor: int
    lane_index: int
    next_metric_first: bool


@dataclass(frozen=True, slots=True)
class _BuildResult:
    pending: _PendingBatch | None
    unused_events: tuple[_EventRecord, ...]
    oversized_metrics: tuple[MetricIdentity, ...]
    next_metric_cursor: int
    next_metric_first: bool


_agent_transport: AgentTransport = RequestsAgentTransport()


class _Runtime:
    def __init__(self, resource: Resource, exporter: HttpExporterConfig) -> None:
        self.resource = resource
        self.exporter = exporter
        self.pid = os.getpid()
        self.lock = threading.Lock()
        self.stopped = False
        self.accepted_emissions = 0
        self.counters: dict[SeriesKey, float] = {}
        self.gauges: dict[SeriesKey, float] = {}
        self.histograms: dict[SeriesKey, _HistogramState] = {}
        self.series_per_descriptor: Counts[tuple[str, str]] = Counts()
        self.events: deque[_EventRecord] = deque()
        self.event_bytes = 0
        self.event_counts: Counts[DeliveryClass] = Counts()
        self.batch_envelope_bytes = len(_stable_json(_batch_envelope(_BATCH_ID_PLACEHOLDER)))
        self.start_ts_unix_nano = time.time_ns()
        self.reset_id = str(uuid.uuid4())
        self.building: _BatchReservation | None = None
        self.pending: _PendingBatch | None = None
        self._metric_cursor: dict[DeliveryClass, int] = {lane: 0 for lane in _DELIVERY_LANES}
        self._oversized_metrics: set[MetricIdentity] = set()
        self._metric_first: dict[DeliveryClass, bool] = {lane: False for lane in _DELIVERY_LANES}
        self._next_lane = 0
        self._retry_not_before = 0.0
        self._stop_event = threading.Event()
        self._export_thread = threading.Thread(
            target=self._export_loop,
            name="rigging-telemetry-exporter",
            daemon=True,
        )
        self._export_thread.start()

    def emit_counter(
        self,
        descriptor: MetricDescriptor,
        value: SupportsFloat,
        attributes: dict[str, Any],
    ) -> None:
        number = _finite_number(value)
        if number < 0:
            raise ValueError("counter increments must be non-negative")
        key = self._series_key(descriptor, attributes)
        if not self.lock.acquire(blocking=False):
            _record_loss(_LOSS_RUNTIME_CONTENTION)
            return
        try:
            if self.stopped or not self._admit_series(descriptor, key, self.counters):
                return
            self.counters[key] = self.counters.get(key, 0.0) + number
            self.accepted_emissions += 1
        finally:
            self.lock.release()

    def emit_gauge(
        self,
        descriptor: MetricDescriptor,
        value: SupportsFloat,
        attributes: dict[str, Any],
    ) -> None:
        number = _finite_number(value)
        key = self._series_key(descriptor, attributes)
        if not self.lock.acquire(blocking=False):
            _record_loss(_LOSS_RUNTIME_CONTENTION)
            return
        try:
            if self.stopped or not self._admit_series(descriptor, key, self.gauges):
                return
            self.gauges[key] = number
            self.accepted_emissions += 1
        finally:
            self.lock.release()

    def emit_histogram(
        self,
        descriptor: MetricDescriptor,
        value: SupportsFloat,
        attributes: dict[str, Any],
    ) -> None:
        number = _finite_number(value)
        key = self._series_key(descriptor, attributes)
        if not self.lock.acquire(blocking=False):
            _record_loss(_LOSS_RUNTIME_CONTENTION)
            return
        try:
            if self.stopped or not self._admit_series(descriptor, key, self.histograms):
                return
            state = self.histograms.get(key)
            if state is None:
                state = _HistogramState(bucket_counts=[0] * (len(descriptor.buckets) + 1))
                self.histograms[key] = state
            bucket = 0
            while bucket < len(descriptor.buckets) and number > descriptor.buckets[bucket]:
                bucket += 1
            state.bucket_counts[bucket] += 1
            state.count += 1
            state.sum += number
            self.accepted_emissions += 1
        finally:
            self.lock.release()

    def emit_event(
        self,
        event_name: str,
        delivery_class: DeliveryClass,
        severity: Severity,
        outcome: str | None,
        body: str | None,
        trace_id: str | None,
        span_id: str | None,
        attributes: dict[str, Any],
    ) -> None:
        descriptor = _EVENT_CATALOG.get(event_name)
        if descriptor is None:
            raise ValueError(f"event {event_name!r} is not in catalog {CATALOG_VERSION}")
        if delivery_class != descriptor.delivery_class:
            raise ValueError(f"event {event_name!r} requires delivery class {descriptor.delivery_class}")
        if not set(attributes).issubset(descriptor.attribute_names):
            raise ValueError(f"event {event_name!r} attributes must be a subset of {descriptor.attribute_names}")
        normalized = tuple(sorted((key, _event_value(value)) for key, value in attributes.items()))
        for name, value in (
            ("event_name", event_name),
            ("outcome", outcome),
            ("body", body),
            ("trace_id", trace_id),
            ("span_id", span_id),
            *normalized,
        ):
            if value is not None:
                _validate_text(name, value)
        record = _EventRecord(
            event_name=event_name,
            delivery_class=delivery_class,
            severity=severity,
            outcome=outcome,
            body=body,
            trace_id=trace_id,
            span_id=span_id,
            attributes=normalized,
            event_ts_unix_nano=time.time_ns(),
            estimated_bytes=0,
        )
        record = replace(
            record,
            estimated_bytes=len(
                _stable_json(
                    _event_wire_record(
                        self.resource,
                        record,
                        self.exporter.max_queue_records - 1,
                        _MAX_I64,
                    )
                )
            ),
        )
        if (
            self.exporter.max_queue_records < 1
            or self.batch_envelope_bytes + record.estimated_bytes > self.exporter.max_queue_bytes
        ):
            _record_loss(_LOSS_EXPORT_TERMINAL)
            return
        if not self.lock.acquire(blocking=False):
            _record_loss(_LOSS_RUNTIME_CONTENTION)
            return
        try:
            if self.stopped:
                return
            while not self._can_add_event(record):
                if not self._drop_oldest_event(record.delivery_class):
                    _record_loss(_LOSS_EVENT_QUEUE_OVERFLOW)
                    return
                _record_loss(_LOSS_EVENT_QUEUE_OVERFLOW)
            self._append_event(record)
            self.accepted_emissions += 1
        finally:
            self.lock.release()

    def _can_add_event(self, event: _EventRecord) -> bool:
        lane_count = self.event_counts[event.delivery_class]
        record_delta = 1
        byte_delta = event.estimated_bytes + (1 if lane_count else self.batch_envelope_bytes)
        return (
            self._accounted_records() + record_delta <= self.exporter.max_queue_records
            and self._accounted_bytes() + byte_delta <= self.exporter.max_queue_bytes
        )

    def _append_event(self, event: _EventRecord) -> None:
        self.events.append(event)
        self.event_bytes += event.estimated_bytes
        self.event_counts[event.delivery_class] += 1

    def _drop_oldest_event(self, lane: DeliveryClass) -> bool:
        for event in self.events:
            if event.delivery_class != lane:
                continue
            self.events.remove(event)
            self._untrack_event(event)
            return True
        return False

    def _untrack_event(self, event: _EventRecord) -> None:
        self.event_bytes -= event.estimated_bytes
        self.event_counts[event.delivery_class] -= 1
        if self.event_counts[event.delivery_class] == 0:
            del self.event_counts[event.delivery_class]

    def _queued_event_bytes(self) -> int:
        lanes = len(self.event_counts)
        commas = len(self.events) - lanes
        return self.event_bytes + commas + lanes * self.batch_envelope_bytes

    def _accounted_records(self) -> int:
        in_flight = (
            self.pending.record_count
            if self.pending is not None
            else self.building.reserved_records if self.building is not None else 0
        )
        return in_flight + len(self.events)

    def _accounted_bytes(self) -> int:
        in_flight = (
            len(self.pending.body)
            if self.pending is not None
            else self.building.reserved_bytes if self.building is not None else 0
        )
        return in_flight + self._queued_event_bytes()

    def shutdown(self) -> None:
        # Never acquire the aggregation lock: it may be held by a failed or
        # paused worker, and telemetry cannot extend process shutdown.
        self.stopped = True
        if self.pid != os.getpid():
            return
        self._stop_event.set()
        self._export_thread.join(self.exporter.shutdown_timeout.total_seconds())

    def _export_loop(self) -> None:
        while not self._stop_event.wait(self.exporter.export_interval.total_seconds()):
            self._export_once()
        self._export_once()

    def _export_once(self) -> None:
        if self.pid != os.getpid():
            _record_loss(_LOSS_FORKED_PROCESS)
            return
        if time.monotonic() < self._retry_not_before:
            return
        pending: _PendingBatch | None = None
        try:
            pending = self._pending_batch()
            if pending is None:
                return
            ack = _agent_transport.write(
                AgentWriteRequest(
                    endpoint=self.exporter.endpoint,
                    batch_id=pending.batch_id,
                    body=pending.body,
                    content_type=_CONTENT_TYPE,
                    token_provider=self.exporter.token_provider,
                    timeout=self.exporter.request_timeout,
                )
            )
            if (
                ack.batch_id != pending.batch_id
                or ack.status not in ("accepted", "duplicate")
                or ack.durability != "agent_wal"
            ):
                raise RetryableExportError("telemetry agent returned an unverifiable acknowledgement")
        except TerminalExportError:
            _record_loss(_LOSS_EXPORT_TERMINAL)
            self._settle_pending(pending)
            return
        except RetryableExportError as error:
            _record_loss(_LOSS_EXPORT_RETRY)
            if error.retry_after is not None:
                self._retry_not_before = time.monotonic() + error.retry_after.total_seconds()
            return
        except Exception:
            # Provider, transport, and fake failures are process-local telemetry
            # loss state. BaseException still terminates the worker naturally.
            _record_loss(_LOSS_EXPORT_RETRY)
            return
        self._retry_not_before = 0.0
        self._settle_pending(pending)

    def _pending_batch(self) -> _PendingBatch | None:
        pending = self.pending
        if pending is not None:
            return pending
        with self.lock:
            if self.pending is not None:
                return self.pending
            if self.building is not None:
                return None
            reservation = self._reserve_batch()
            self.building = reservation
        if reservation is None:
            return None
        try:
            result = self._build_batch(reservation)
        except Exception:
            self._cancel_reservation(reservation)
            raise
        dropped_events = 0
        with self.lock:
            if self.building is not reservation:
                return self.pending
            self.building = None
            self._oversized_metrics.update(result.oversized_metrics)
            self.pending = result.pending
            dropped_events = self._restore_reserved_events(
                result.unused_events,
                enforce_caps=result.pending is not None,
            )
            if result.pending is None:
                self._metric_cursor[reservation.delivery_class] = result.next_metric_cursor
                self._metric_first[reservation.delivery_class] = result.next_metric_first
                self._next_lane = (reservation.lane_index + 1) % len(_DELIVERY_LANES)
        for _ in result.oversized_metrics:
            _record_loss(_LOSS_EXPORT_TERMINAL)
        for _ in range(dropped_events):
            _record_loss(_LOSS_EVENT_QUEUE_OVERFLOW)
        return result.pending

    def _reserve_batch(self) -> _BatchReservation | None:
        lanes = _DELIVERY_LANES
        for offset in range(len(lanes)):
            lane_index = (self._next_lane + offset) % len(lanes)
            lane = lanes[lane_index]
            metrics, metric_start_cursor, metric_total = self._metric_snapshots(lane)
            events = tuple(event for event in self.events if event.delivery_class == lane)
            if not metrics and not events:
                continue
            lane_event_bytes = sum(event.estimated_bytes for event in events)
            remaining_count = len(self.events) - len(events)
            remaining_lanes = len(self.event_counts) - (1 if events else 0)
            remaining_bytes = self.event_bytes - lane_event_bytes
            remaining_commas = remaining_count - remaining_lanes
            remaining_usage = remaining_bytes + remaining_commas + remaining_lanes * self.batch_envelope_bytes
            reserved_records = self.exporter.max_queue_records - remaining_count
            reserved_bytes = self.exporter.max_queue_bytes - remaining_usage
            if reserved_records <= 0 or reserved_bytes < self.batch_envelope_bytes:
                continue
            metric_first = bool(metrics and events and self._metric_first[lane])
            next_metric_first = not metric_first if metrics and events else self._metric_first[lane]
            self._take_reserved_events(events)
            return _BatchReservation(
                batch_id=str(uuid.uuid4()),
                delivery_class=lane,
                lane_index=lane_index,
                observed_ts_unix_nano=time.time_ns(),
                events=events,
                metrics=metrics,
                metric_start_cursor=metric_start_cursor,
                metric_total=metric_total,
                reserved_records=reserved_records,
                reserved_bytes=reserved_bytes,
                metric_first=metric_first,
                next_metric_first=next_metric_first,
            )
        return None

    def _take_reserved_events(self, events: tuple[_EventRecord, ...]) -> None:
        selected_ids = {id(event) for event in events}
        if selected_ids:
            self.events = deque(event for event in self.events if id(event) not in selected_ids)
            for event in events:
                self._untrack_event(event)

    def _metric_snapshots(
        self,
        lane: DeliveryClass,
    ) -> tuple[tuple[_MetricSnapshot, ...], int, int]:
        snapshots: list[_MetricSnapshot] = []
        for values, state_kind in (
            (self.counters, InstrumentKind.COUNTER),
            (self.gauges, InstrumentKind.GAUGE),
            (self.histograms, InstrumentKind.HISTOGRAM),
        ):
            for key, value in values.items():
                if (state_kind, key) in self._oversized_metrics:
                    continue
                descriptor = _descriptors[(key[0], key[1])]
                if descriptor.delivery_class != lane:
                    continue
                if isinstance(value, _HistogramState):
                    snapshot_value: float | _HistogramSnapshot = _HistogramSnapshot(
                        bucket_counts=tuple(value.bucket_counts),
                        count=value.count,
                        sum=value.sum,
                    )
                else:
                    snapshot_value = value
                snapshots.append(
                    _MetricSnapshot(
                        descriptor=descriptor,
                        key=key,
                        value=snapshot_value,
                        state_kind=state_kind,
                    )
                )
        total = len(snapshots)
        if total == 0:
            return (), 0, 0
        start = self._metric_cursor[lane] % total
        return tuple(snapshots), start, total

    def _build_batch(
        self,
        reservation: _BatchReservation,
    ) -> _BuildResult:
        envelope = _batch_envelope(reservation.batch_id)
        encoded_bytes = len(_stable_json(envelope))
        records: list[dict[str, Any]] = []
        event_count = 0
        metric_count = 0
        oversized_metrics: list[MetricIdentity] = []

        sorted_metrics = sorted(
            reservation.metrics,
            key=lambda snapshot: (snapshot.key, snapshot.state_kind),
        )
        ordered_metrics = (
            sorted_metrics[reservation.metric_start_cursor :] + sorted_metrics[: reservation.metric_start_cursor]
        )

        def append_events() -> None:
            nonlocal encoded_bytes, event_count
            for event in reservation.events:
                record = _event_wire_record(
                    self.resource,
                    event,
                    len(records),
                    reservation.observed_ts_unix_nano,
                )
                record_bytes = len(_stable_json(record)) + (1 if records else 0)
                if not self._record_fits(
                    len(records),
                    encoded_bytes,
                    record_bytes,
                    reservation.reserved_records,
                    reservation.reserved_bytes,
                ):
                    break
                records.append(record)
                encoded_bytes += record_bytes
                event_count += 1

        def append_metrics() -> None:
            nonlocal encoded_bytes, metric_count
            for metric in ordered_metrics:
                record = _metric_wire_record(
                    self.resource,
                    metric.descriptor,
                    metric.key,
                    metric.value,
                    metric.state_kind,
                    self.start_ts_unix_nano,
                    self.reset_id,
                    reservation.observed_ts_unix_nano,
                )
                record["record_index"] = len(records)
                record_bytes = len(_stable_json(record)) + (1 if records else 0)
                if not self._record_fits(
                    len(records),
                    encoded_bytes,
                    record_bytes,
                    reservation.reserved_records,
                    reservation.reserved_bytes,
                ):
                    record["record_index"] = 0
                    single_record_bytes = len(_stable_json(record))
                    if not self._record_fits(
                        0,
                        len(_stable_json(envelope)),
                        single_record_bytes,
                        self.exporter.max_queue_records,
                        self.exporter.max_queue_bytes,
                    ):
                        oversized_metrics.append((metric.state_kind, metric.key))
                        metric_count += 1
                        continue
                    break
                records.append(record)
                encoded_bytes += record_bytes
                metric_count += 1

        if reservation.metric_first:
            append_metrics()
            append_events()
        else:
            append_events()
            append_metrics()

        unused_events = reservation.events[event_count:]
        next_metric_cursor = reservation.metric_start_cursor
        if reservation.metric_total:
            next_metric_cursor = (reservation.metric_start_cursor + metric_count) % reservation.metric_total
        if not records:
            return _BuildResult(
                pending=None,
                unused_events=unused_events,
                oversized_metrics=tuple(oversized_metrics),
                next_metric_cursor=next_metric_cursor,
                next_metric_first=reservation.next_metric_first,
            )
        envelope["records"] = records
        body = _stable_json(envelope)
        if len(records) > reservation.reserved_records or len(body) > reservation.reserved_bytes:
            raise AssertionError("telemetry batch builder exceeded its global queue reservation")
        return _BuildResult(
            pending=_PendingBatch(
                batch_id=reservation.batch_id,
                delivery_class=reservation.delivery_class,
                body=body,
                record_count=len(records),
                next_metric_cursor=next_metric_cursor,
                lane_index=reservation.lane_index,
                next_metric_first=reservation.next_metric_first,
            ),
            unused_events=unused_events,
            oversized_metrics=tuple(oversized_metrics),
            next_metric_cursor=next_metric_cursor,
            next_metric_first=reservation.next_metric_first,
        )

    def _record_fits(
        self,
        record_count: int,
        encoded_bytes: int,
        record_bytes: int,
        max_records: int,
        max_bytes: int,
    ) -> bool:
        return record_count < max_records and encoded_bytes + record_bytes <= max_bytes

    def _cancel_reservation(self, reservation: _BatchReservation) -> None:
        with self.lock:
            if self.building is reservation:
                self.building = None
                self._restore_reserved_events(reservation.events, enforce_caps=False)

    def _restore_reserved_events(
        self,
        events: tuple[_EventRecord, ...],
        *,
        enforce_caps: bool,
    ) -> int:
        dropped = 0
        existing = tuple(self.events)
        restored: list[_EventRecord] = []
        for event in events:
            if enforce_caps and not self._can_add_event(event):
                dropped += 1
                continue
            self._append_event(event)
            restored.append(event)
        if restored:
            self.events = deque((*restored, *existing))
        return dropped

    def _settle_pending(self, pending: _PendingBatch | None) -> None:
        if pending is None:
            return
        # This is the one blocking exporter coordination point: after a durable
        # agent ack, the daemon must retire exactly that immutable batch. It
        # never runs on an emission thread, and shutdown still caps its join.
        with self.lock:
            if self.pending is pending:
                self.pending = None
                self._metric_cursor[pending.delivery_class] = pending.next_metric_cursor
                self._metric_first[pending.delivery_class] = pending.next_metric_first
                self._next_lane = (pending.lane_index + 1) % len(_DELIVERY_LANES)

    def _series_key(
        self,
        descriptor: MetricDescriptor,
        attributes: dict[str, Any],
    ) -> SeriesKey:
        expected = {attribute.name: attribute for attribute in descriptor.attributes}
        if set(attributes) != set(expected):
            raise ValueError(f"attributes for {descriptor.scope}.{descriptor.name} must be {tuple(expected)}")
        normalized: list[tuple[str, str]] = []
        for name in expected:
            value = attributes[name]
            if not isinstance(value, str):
                raise ValueError(f"attribute {name!r} must be a string")
            if value not in expected[name].allowed_values:
                raise ValueError(f"attribute {name!r} has unapproved value {value!r}")
            normalized.append((name, value))
        return (descriptor.scope, descriptor.name, tuple(normalized))

    def _admit_series(
        self,
        descriptor: MetricDescriptor,
        key: SeriesKey,
        values: dict[SeriesKey, Any],
    ) -> bool:
        if key in values:
            return True
        descriptor_key = (descriptor.scope, descriptor.name)
        if self.series_per_descriptor[descriptor_key] >= descriptor.cardinality_limit:
            _record_loss(_LOSS_CARDINALITY_OVERFLOW)
            return False
        self.series_per_descriptor[descriptor_key] += 1
        return True


def _resource_record(resource: Resource) -> dict[str, Any]:
    return {key: value for key, value in asdict(resource).items() if value is not None}


def _batch_envelope(batch_id: str) -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "catalog_version": CATALOG_VERSION,
        "batch_id": batch_id,
        "records": [],
    }


def _stable_json(value: Mapping[str, Any] | list[Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _retry_after(value: str | None) -> timedelta | None:
    if value is None:
        return None
    try:
        delay = timedelta(seconds=max(0, int(value)))
    except ValueError:
        try:
            deadline = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if deadline.tzinfo is None:
            deadline = deadline.replace(tzinfo=UTC)
        delay = max(timedelta(0), deadline - datetime.now(UTC))
    return min(delay, _MAX_RETRY_AFTER)


def _metric_wire_record(
    resource: Resource,
    descriptor: MetricDescriptor,
    key: SeriesKey,
    value: float | _HistogramSnapshot,
    state_kind: InstrumentKind,
    start_ts_unix_nano: int,
    reset_id: str,
    observed_ts_unix_nano: int,
) -> dict[str, Any]:
    attributes = dict(key[2])
    metric: dict[str, Any] = {
        "scope": descriptor.scope,
        "name": descriptor.name,
        "description": descriptor.description,
        "unit": descriptor.unit,
        "instrument_kind": descriptor.instrument_kind,
        "temporality": descriptor.temporality,
        "series_id": _series_id(resource, descriptor, attributes),
        "attributes": attributes,
    }
    if state_kind == InstrumentKind.HISTOGRAM:
        if not isinstance(value, _HistogramSnapshot):
            raise AssertionError("histogram descriptor has scalar state")
        metric.update(
            {
                "start_ts_unix_nano": start_ts_unix_nano,
                "reset_id": reset_id,
                "count": value.count,
                "sum": value.sum,
                "explicit_bounds": descriptor.buckets,
                "bucket_counts": value.bucket_counts,
            }
        )
    else:
        if isinstance(value, _HistogramSnapshot):
            raise AssertionError("scalar descriptor has histogram state")
        metric["value"] = value
        if descriptor.temporality == Temporality.CUMULATIVE:
            metric["start_ts_unix_nano"] = start_ts_unix_nano
            metric["reset_id"] = reset_id
    return {
        "signal": "metric",
        "event_ts_unix_nano": observed_ts_unix_nano,
        "observed_ts_unix_nano": observed_ts_unix_nano,
        "delivery_class": descriptor.delivery_class,
        "resource": _resource_record(resource),
        "metric": metric,
    }


def _event_wire_record(
    resource: Resource,
    event: _EventRecord,
    record_index: int,
    observed_ts_unix_nano: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "event_name": event.event_name,
        "severity_number": _SEVERITY_NUMBER[event.severity],
        "severity_text": event.severity.upper(),
        "attributes": dict(event.attributes),
    }
    for key, value in (
        ("outcome", event.outcome),
        ("body", event.body),
        ("trace_id", event.trace_id),
        ("span_id", event.span_id),
    ):
        if value is not None:
            payload[key] = value
    return {
        "record_index": record_index,
        "signal": "event",
        "event_ts_unix_nano": event.event_ts_unix_nano,
        "observed_ts_unix_nano": observed_ts_unix_nano,
        "delivery_class": event.delivery_class,
        "resource": _resource_record(resource),
        "event": payload,
    }


def _series_id(
    resource: Resource,
    descriptor: MetricDescriptor,
    attributes: dict[str, str],
) -> str:
    fields = [
        descriptor.scope,
        descriptor.name,
        resource.cluster or "",
        resource.entity_authority or "",
        resource.entity_type or "",
        resource.entity_uid or "",
        resource.service_name,
        resource.service_instance_id,
        resource.attempt_uid or "",
        resource.actor_id or "",
        resource.engine_id or "",
        "" if resource.process_index is None else str(resource.process_index),
        "" if resource.rank is None else str(resource.rank),
        "",
        "",
        *(f"{key}={value}" for key, value in sorted(attributes.items())),
    ]
    digest = hashlib.sha256(_SERIES_ID_DOMAIN)
    for field in fields:
        encoded = field.encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _emit_metric(
    descriptor: MetricDescriptor | None,
    emit: Callable[[_Runtime, MetricDescriptor, SupportsFloat, dict[str, Any]], None],
    value: SupportsFloat,
    attributes: dict[str, Any],
) -> None:
    if descriptor is None:
        _record_loss(_LOSS_INERT_HANDLE)
        return
    runtime = _active_runtime()
    if runtime is None:
        return
    try:
        emit(runtime, descriptor, value, attributes)
    except Exception:
        _record_loss(_LOSS_INVALID_EMISSION)


class Counter:
    def __init__(self, descriptor: MetricDescriptor | None) -> None:
        self.descriptor = descriptor

    def add(self, value: SupportsFloat = 1, **attributes: Any) -> None:
        _emit_metric(self.descriptor, _Runtime.emit_counter, value, attributes)


class Gauge:
    def __init__(self, descriptor: MetricDescriptor | None) -> None:
        self.descriptor = descriptor

    def set(self, value: SupportsFloat, **attributes: Any) -> None:
        _emit_metric(self.descriptor, _Runtime.emit_gauge, value, attributes)


class Histogram:
    def __init__(self, descriptor: MetricDescriptor | None) -> None:
        self.descriptor = descriptor

    def record(self, value: SupportsFloat, **attributes: Any) -> None:
        _emit_metric(self.descriptor, _Runtime.emit_histogram, value, attributes)


@dataclass(frozen=True, slots=True)
class Meter:
    scope: str
    owner: str
    default_cadence: timedelta

    def counter(
        self,
        name: str,
        *,
        description: str,
        unit: str,
        attributes: tuple[AttributeSpec, ...] = (),
        delivery_class: DeliveryClass = DeliveryClass.BUFFERED,
        cardinality_limit: int = DEFAULT_CARDINALITY_LIMIT,
        maturity: Maturity = Maturity.EXPERIMENTAL,
    ) -> Counter:
        return counter(
            name,
            scope=self.scope,
            description=description,
            unit=unit,
            attributes=attributes,
            owner=self.owner,
            cadence=self.default_cadence,
            delivery_class=delivery_class,
            cardinality_limit=cardinality_limit,
            maturity=maturity,
        )

    def gauge(
        self,
        name: str,
        *,
        description: str,
        unit: str,
        attributes: tuple[AttributeSpec, ...] = (),
        delivery_class: DeliveryClass = DeliveryClass.COALESCING,
        cardinality_limit: int = DEFAULT_CARDINALITY_LIMIT,
        maturity: Maturity = Maturity.EXPERIMENTAL,
    ) -> Gauge:
        return gauge(
            name,
            scope=self.scope,
            description=description,
            unit=unit,
            attributes=attributes,
            owner=self.owner,
            cadence=self.default_cadence,
            delivery_class=delivery_class,
            cardinality_limit=cardinality_limit,
            maturity=maturity,
        )

    def histogram(
        self,
        name: str,
        *,
        description: str,
        unit: str,
        buckets: tuple[float, ...],
        attributes: tuple[AttributeSpec, ...] = (),
        delivery_class: DeliveryClass = DeliveryClass.BUFFERED,
        cardinality_limit: int = DEFAULT_CARDINALITY_LIMIT,
        maturity: Maturity = Maturity.EXPERIMENTAL,
    ) -> Histogram:
        return histogram(
            name,
            scope=self.scope,
            description=description,
            unit=unit,
            attributes=attributes,
            buckets=buckets,
            owner=self.owner,
            cadence=self.default_cadence,
            delivery_class=delivery_class,
            cardinality_limit=cardinality_limit,
            maturity=maturity,
        )


_state_lock = threading.RLock()
_loss_lock = threading.Lock()
_runtime: _Runtime | None = None
_descriptors: dict[tuple[str, str], MetricDescriptor] = {}
_losses: Counts[str] = Counts()
_logging_context: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "rigging_telemetry_logging_context",
    default=None,
)
_METRIC_CATALOG = {
    (entry.scope, entry.name): MetricDescriptor(
        name=entry.name,
        scope=entry.scope,
        description=entry.description,
        unit=entry.unit,
        instrument_kind=InstrumentKind(entry.instrument_kind),
        temporality=Temporality(entry.temporality),
        attributes=tuple(
            AttributeSpec(name=name, allowed_values=allowed_values) for name, allowed_values in entry.attributes
        ),
        buckets=entry.buckets,
        owner=entry.owner,
        cadence=timedelta(seconds=entry.cadence_seconds),
        delivery_class=DeliveryClass(entry.delivery_class),
        cardinality_limit=entry.cardinality_limit,
        maturity=Maturity(entry.maturity),
    )
    for entry in _CATALOG.metrics
}
_EVENT_CATALOG = {
    entry.event_name: EventDescriptor(
        event_name=entry.event_name,
        owner=entry.owner,
        attribute_names=entry.attributes,
        delivery_class=DeliveryClass(entry.delivery_class),
    )
    for entry in _CATALOG.events
}


def _after_fork_child() -> None:
    global _loss_lock, _runtime, _state_lock
    # A lock held by another parent thread stays permanently locked in the
    # single-threaded child. Replace coordination primitives and discard the
    # inherited runtime without touching either old lock.
    _state_lock = threading.RLock()
    _loss_lock = threading.Lock()
    _runtime = None
    _losses[_LOSS_FORKED_PROCESS] += 1


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_child)


def meter(*, scope: str, owner: str, default_cadence: timedelta) -> Meter:
    return Meter(scope=scope, owner=owner, default_cadence=default_cadence)


def counter(
    name: str,
    *,
    scope: str,
    description: str,
    unit: str,
    attributes: tuple[AttributeSpec, ...] = (),
    owner: str,
    cadence: timedelta,
    delivery_class: DeliveryClass = DeliveryClass.BUFFERED,
    cardinality_limit: int = DEFAULT_CARDINALITY_LIMIT,
    maturity: Maturity = Maturity.EXPERIMENTAL,
) -> Counter:
    descriptor = _declare(
        name=name,
        scope=scope,
        description=description,
        unit=unit,
        instrument_kind=InstrumentKind.COUNTER,
        temporality=Temporality.CUMULATIVE,
        attributes=attributes,
        buckets=(),
        owner=owner,
        cadence=cadence,
        delivery_class=delivery_class,
        cardinality_limit=cardinality_limit,
        maturity=maturity,
    )
    return Counter(descriptor)


def gauge(
    name: str,
    *,
    scope: str,
    description: str,
    unit: str,
    attributes: tuple[AttributeSpec, ...] = (),
    owner: str,
    cadence: timedelta,
    delivery_class: DeliveryClass = DeliveryClass.COALESCING,
    cardinality_limit: int = DEFAULT_CARDINALITY_LIMIT,
    maturity: Maturity = Maturity.EXPERIMENTAL,
) -> Gauge:
    descriptor = _declare(
        name=name,
        scope=scope,
        description=description,
        unit=unit,
        instrument_kind=InstrumentKind.GAUGE,
        temporality=Temporality.UNSPECIFIED,
        attributes=attributes,
        buckets=(),
        owner=owner,
        cadence=cadence,
        delivery_class=delivery_class,
        cardinality_limit=cardinality_limit,
        maturity=maturity,
    )
    return Gauge(descriptor)


def histogram(
    name: str,
    *,
    scope: str,
    description: str,
    unit: str,
    buckets: tuple[float, ...],
    attributes: tuple[AttributeSpec, ...] = (),
    owner: str,
    cadence: timedelta,
    delivery_class: DeliveryClass = DeliveryClass.BUFFERED,
    cardinality_limit: int = DEFAULT_CARDINALITY_LIMIT,
    maturity: Maturity = Maturity.EXPERIMENTAL,
) -> Histogram:
    descriptor = _declare(
        name=name,
        scope=scope,
        description=description,
        unit=unit,
        instrument_kind=InstrumentKind.HISTOGRAM,
        temporality=Temporality.CUMULATIVE,
        attributes=attributes,
        buckets=buckets,
        owner=owner,
        cadence=cadence,
        delivery_class=delivery_class,
        cardinality_limit=cardinality_limit,
        maturity=maturity,
    )
    return Histogram(descriptor)


def configure(
    *,
    exporter: HttpExporterConfig,
    resource: Resource | None = None,
    service_name: str | None = None,
    role: str | None = None,
    root_run_uid: str | None = None,
    service_instance_id: str | None = None,
    service_version: str | None = None,
) -> None:
    global _runtime
    try:
        _validate_exporter(exporter)
        with _state_lock:
            resolved = _resolve_resource(
                resource=resource,
                service_name=service_name,
                role=role,
                root_run_uid=root_run_uid,
                service_instance_id=service_instance_id,
                service_version=service_version,
            )
            if _runtime is None or _runtime.pid != os.getpid():
                _runtime = _Runtime(resolved, exporter)
                return
            if _runtime.resource == resolved and _runtime.exporter == exporter:
                return
            _record_loss(_LOSS_CONFIGURATION_CONFLICT)
    except Exception:
        _record_loss(_LOSS_INVALID_CONFIGURATION)


def shutdown() -> None:
    runtime = _runtime
    if runtime is None:
        return
    runtime.shutdown()


def event(
    event_name: str,
    *,
    delivery_class: DeliveryClass = DeliveryClass.BUFFERED,
    severity: Severity = Severity.INFO,
    outcome: str | None = None,
    body: str | None = None,
    trace_id: str | None = None,
    span_id: str | None = None,
    **attributes: Any,
) -> None:
    runtime = _active_runtime()
    if runtime is None:
        return
    try:
        runtime.emit_event(
            event_name,
            delivery_class,
            severity,
            outcome,
            body,
            trace_id,
            span_id,
            attributes,
        )
    except Exception:
        _record_loss(_LOSS_INVALID_EMISSION)


@contextlib.contextmanager
def logging_context(**fields: Any) -> Iterator[None]:
    current = _logging_context.get() or {}
    token = _logging_context.set({**current, **fields})
    try:
        yield
    finally:
        _logging_context.reset(token)


def current_logging_context() -> dict[str, Any]:
    return dict(_logging_context.get() or {})


def runtime_status() -> RuntimeStatus:
    with _state_lock:
        runtime = _runtime
    with _loss_lock:
        losses = tuple(sorted(_losses.items()))
    if runtime is None:
        return RuntimeStatus(
            configured=False,
            stopped=False,
            service_instance_id=None,
            accepted_emissions=0,
            metric_series=0,
            queued_events=0,
            queued_event_bytes=0,
            in_flight_batch_id=None,
            in_flight_records=0,
            in_flight_bytes=0,
            losses=losses,
        )
    pending = runtime.pending
    building = runtime.building
    return RuntimeStatus(
        configured=True,
        stopped=runtime.stopped,
        service_instance_id=runtime.resource.service_instance_id,
        accepted_emissions=runtime.accepted_emissions,
        metric_series=len(runtime.counters) + len(runtime.gauges) + len(runtime.histograms),
        queued_events=len(runtime.events),
        queued_event_bytes=runtime._queued_event_bytes(),
        in_flight_batch_id=(
            pending.batch_id if pending is not None else building.batch_id if building is not None else None
        ),
        in_flight_records=(
            pending.record_count if pending is not None else building.reserved_records if building is not None else 0
        ),
        in_flight_bytes=(
            len(pending.body) if pending is not None else building.reserved_bytes if building is not None else 0
        ),
        losses=losses,
    )


def _declare(
    *,
    name: str,
    scope: str,
    description: str,
    unit: str,
    instrument_kind: InstrumentKind,
    temporality: Temporality,
    attributes: tuple[AttributeSpec, ...],
    buckets: tuple[float, ...],
    owner: str,
    cadence: timedelta,
    delivery_class: DeliveryClass,
    cardinality_limit: int,
    maturity: Maturity,
) -> MetricDescriptor | None:
    try:
        descriptor = MetricDescriptor(
            name=name,
            scope=scope,
            description=description,
            unit=unit,
            instrument_kind=instrument_kind,
            temporality=temporality,
            attributes=attributes,
            buckets=buckets,
            owner=owner,
            cadence=cadence,
            delivery_class=delivery_class,
            cardinality_limit=cardinality_limit,
            maturity=maturity,
        )
        _validate_descriptor(descriptor)
        key = (descriptor.scope, descriptor.name)
        if _METRIC_CATALOG.get(key) != descriptor:
            raise ValueError(f"metric {descriptor.scope}.{descriptor.name} is not in {CATALOG_VERSION}")
        with _state_lock:
            existing = _descriptors.get(key)
            if existing is None:
                _descriptors[key] = descriptor
                return descriptor
            if existing == descriptor:
                return existing
            _record_loss(_LOSS_DESCRIPTOR_CONFLICT)
            return None
    except Exception:
        _record_loss(_LOSS_INVALID_DESCRIPTOR)
        return None


def _validate_descriptor(descriptor: MetricDescriptor) -> None:
    if not _INSTRUMENT_NAME.fullmatch(descriptor.name):
        raise ValueError(f"invalid instrument name {descriptor.name!r}")
    if not _SCOPE_NAME.fullmatch(descriptor.scope):
        raise ValueError(f"invalid instrumentation scope {descriptor.scope!r}")
    if not descriptor.description:
        raise ValueError("description is required")
    if not descriptor.unit:
        raise ValueError("unit is required")
    if not descriptor.owner:
        raise ValueError("owner is required")
    if descriptor.cadence <= timedelta(0):
        raise ValueError("cadence must be positive")
    if descriptor.cardinality_limit <= 0:
        raise ValueError("cardinality_limit must be positive")
    names = [attribute.name for attribute in descriptor.attributes]
    if len(names) != len(set(names)):
        raise ValueError("attribute names must be unique")
    for attribute in descriptor.attributes:
        if not _INSTRUMENT_NAME.fullmatch(attribute.name):
            raise ValueError(f"invalid attribute name {attribute.name!r}")
        if not attribute.allowed_values or len(attribute.allowed_values) != len(set(attribute.allowed_values)):
            raise ValueError(f"attribute {attribute.name!r} needs unique allowed values")
    if descriptor.instrument_kind == InstrumentKind.HISTOGRAM:
        if not descriptor.buckets:
            raise ValueError("histogram buckets are required")
        if any(not math.isfinite(bound) for bound in descriptor.buckets):
            raise ValueError("histogram buckets must be finite")
        if any(left >= right for left, right in zip(descriptor.buckets, descriptor.buckets[1:], strict=False)):
            raise ValueError("histogram buckets must be strictly increasing")
    elif descriptor.buckets:
        raise ValueError("only histograms may define buckets")


def _validate_exporter(exporter: HttpExporterConfig) -> None:
    if not exporter.endpoint:
        raise ValueError("exporter endpoint is required")
    if exporter.token_provider is None:
        raise ValueError("exporter token_provider is required")
    if (
        exporter.export_interval <= timedelta(0)
        or exporter.request_timeout <= timedelta(0)
        or exporter.shutdown_timeout < timedelta(0)
    ):
        raise ValueError("exporter durations must be positive")
    if exporter.shutdown_timeout > MAX_SHUTDOWN_TIMEOUT:
        raise ValueError("exporter shutdown_timeout must not exceed five seconds")
    if exporter.max_queue_records <= 0 or exporter.max_queue_bytes <= 0:
        raise ValueError("exporter queue bounds must be positive")
    minimum_batch_bytes = len(_stable_json(_batch_envelope(_BATCH_ID_PLACEHOLDER)))
    if exporter.max_queue_bytes < minimum_batch_bytes:
        raise ValueError(f"exporter max_queue_bytes must be at least {minimum_batch_bytes}")


def _resolve_resource(
    *,
    resource: Resource | None,
    service_name: str | None,
    role: str | None,
    root_run_uid: str | None,
    service_instance_id: str | None,
    service_version: str | None,
) -> Resource:
    convenience_used = any(
        value is not None for value in (service_name, role, root_run_uid, service_instance_id, service_version)
    )
    if resource is not None:
        if convenience_used:
            raise ValueError("resource cannot be combined with convenience identity fields")
        _validate_resource(resource)
        return resource
    if service_name is None:
        raise ValueError("service_name is required")
    if service_instance_id is None and _runtime is not None and _runtime.pid == os.getpid():
        current = _runtime.resource
        if (
            current.service_name == service_name
            and current.role == role
            and current.root_run_uid == root_run_uid
            and current.service_version == service_version
        ):
            service_instance_id = current.service_instance_id
    resolved = Resource(
        service_name=service_name,
        service_instance_id=service_instance_id or str(uuid.uuid4()),
        role=role,
        root_run_uid=root_run_uid,
        service_version=service_version,
    )
    _validate_resource(resolved)
    return resolved


def _validate_resource(resource: Resource) -> None:
    if not resource.service_name:
        raise ValueError("service_name is required")
    if not resource.service_instance_id:
        raise ValueError("service_instance_id is required")
    for name, value in asdict(resource).items():
        if isinstance(value, str):
            _validate_text(name, value)


def _active_runtime() -> _Runtime | None:
    # CPython reference reads are atomic. Emission deliberately does not take
    # the configuration lock: a concurrent configure either publishes the
    # complete runtime before this read or this call observes the prior value.
    runtime = _runtime
    if runtime is None or runtime.stopped:
        return None
    if runtime.pid != os.getpid():
        _record_loss(_LOSS_FORKED_PROCESS)
        return None
    return runtime


def _finite_number(value: SupportsFloat) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("metric values must be finite")
    return number


def _event_value(value: Any) -> str:
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    raise ValueError("event attributes must be scalar")


def _validate_text(name: str, value: str) -> None:
    if len(value.encode()) > _MAX_STRING_BYTES:
        raise ValueError(f"{name} exceeds {_MAX_STRING_BYTES} UTF-8 bytes")


def _record_loss(reason: str) -> None:
    if not _loss_lock.acquire(blocking=False):
        return
    try:
        _losses[reason] += 1
    finally:
        _loss_lock.release()
