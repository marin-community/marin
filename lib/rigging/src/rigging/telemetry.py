# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded process-local telemetry primitives.

Instrument handles are safe to declare before an entrypoint configures
telemetry. Emission is a no-op until then and never performs I/O.
"""

from __future__ import annotations

import contextlib
import contextvars
import math
import re
import threading
import uuid
from collections import Counter as Counts
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum
from typing import Any, SupportsFloat

from rigging.telemetry_catalog import load_catalog

_CATALOG = load_catalog()
CATALOG_VERSION = _CATALOG.catalog_version
DEFAULT_CARDINALITY_LIMIT = 100
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


class DeliveryClass(StrEnum):
    COALESCING = "coalescing"
    BUFFERED = "buffered"
    DURABLE = "durable"


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
    entity_authority: str | None = None
    entity_type: str | None = None
    entity_uid: str | None = None


@dataclass(frozen=True, slots=True)
class HttpExporterConfig:
    endpoint: str
    export_interval: timedelta
    request_timeout: timedelta
    shutdown_timeout: timedelta
    max_queue_records: int
    max_queue_bytes: int


@dataclass(frozen=True, slots=True)
class RuntimeStatus:
    configured: bool
    stopped: bool
    service_instance_id: str | None
    accepted_emissions: int
    metric_series: int
    queued_events: int
    queued_event_bytes: int
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
    estimated_bytes: int


SeriesKey = tuple[str, str, tuple[tuple[str, str], ...]]


class _Runtime:
    def __init__(self, resource: Resource, exporter: HttpExporterConfig) -> None:
        self.resource = resource
        self.exporter = exporter
        self.lock = threading.Lock()
        self.stopped = False
        self.accepted_emissions = 0
        self.counters: dict[SeriesKey, float] = {}
        self.gauges: dict[SeriesKey, float] = {}
        self.histograms: dict[SeriesKey, _HistogramState] = {}
        self.series_per_descriptor: Counts[tuple[str, str]] = Counts()
        self.events: deque[_EventRecord] = deque()
        self.event_bytes = 0

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
        estimated_bytes = (
            len(event_name)
            + len(body or "")
            + len(outcome or "")
            + len(trace_id or "")
            + len(span_id or "")
            + sum(len(key) + len(value) for key, value in normalized)
        )
        record = _EventRecord(
            event_name=event_name,
            delivery_class=delivery_class,
            severity=severity,
            outcome=outcome,
            body=body,
            trace_id=trace_id,
            span_id=span_id,
            attributes=normalized,
            estimated_bytes=estimated_bytes,
        )
        if not self.lock.acquire(blocking=False):
            _record_loss(_LOSS_RUNTIME_CONTENTION)
            return
        try:
            if self.stopped:
                return
            while self.events and (
                len(self.events) >= self.exporter.max_queue_records
                or self.event_bytes + estimated_bytes > self.exporter.max_queue_bytes
            ):
                dropped = self.events.popleft()
                self.event_bytes -= dropped.estimated_bytes
                _record_loss(_LOSS_EVENT_QUEUE_OVERFLOW)
            if self.exporter.max_queue_records == 0 or estimated_bytes > self.exporter.max_queue_bytes:
                _record_loss(_LOSS_EVENT_QUEUE_OVERFLOW)
                return
            self.events.append(record)
            self.event_bytes += estimated_bytes
            self.accepted_emissions += 1
        finally:
            self.lock.release()

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
            if _runtime is None:
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
    runtime.stopped = True


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
            losses=losses,
        )
    with runtime.lock:
        return RuntimeStatus(
            configured=True,
            stopped=runtime.stopped,
            service_instance_id=runtime.resource.service_instance_id,
            accepted_emissions=runtime.accepted_emissions,
            metric_series=len(runtime.counters) + len(runtime.gauges) + len(runtime.histograms),
            queued_events=len(runtime.events),
            queued_event_bytes=runtime.event_bytes,
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
    if (
        exporter.export_interval <= timedelta(0)
        or exporter.request_timeout <= timedelta(0)
        or exporter.shutdown_timeout < timedelta(0)
    ):
        raise ValueError("exporter durations must be positive")
    if exporter.max_queue_records < 0 or exporter.max_queue_bytes < 0:
        raise ValueError("exporter queue bounds must be non-negative")


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
    if service_instance_id is None and _runtime is not None:
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


def _active_runtime() -> _Runtime | None:
    # CPython reference reads are atomic. Emission deliberately does not take
    # the configuration lock: a concurrent configure either publishes the
    # complete runtime before this read or this call observes the prior value.
    runtime = _runtime
    if runtime is None or runtime.stopped:
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


def _record_loss(reason: str) -> None:
    if not _loss_lock.acquire(blocking=False):
        return
    try:
        _losses[reason] += 1
    finally:
        _loss_lock.release()
