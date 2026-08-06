# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded, best-effort process telemetry exported directly to Finelog."""

import atexit
import enum
import logging
import math
import threading
import time
import uuid
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

import requests
import zstandard

from rigging.telemetry import serialization
from rigging.telemetry.compression import FINELOG_ZSTD_LEVEL
from rigging.timing import ExponentialBackoff

logger = logging.getLogger(__name__)

DEFAULT_MAX_QUEUE_RECORDS = 10_000
DEFAULT_MAX_QUEUE_BYTES = 16 << 20
DEFAULT_MAX_BATCH_RECORDS = 1_000
DEFAULT_MAX_BATCH_BYTES = 1 << 20
CURRENT_SNAPSHOT = "current_snapshot"
CUMULATIVE_SNAPSHOT = "cumulative_snapshot"
_MAX_REQUEST_BYTES = 4 << 20
_MAX_SHUTDOWN_TIMEOUT = 5.0
_WARNING_INTERVAL = 60.0
_EVENT_KIND = "event"


class TelemetryRole(StrEnum):
    """Stock process roles for telemetry resources."""

    CONTROLLER = "controller"
    COORDINATOR = "coordinator"
    DRIVER = "driver"
    TRAINER = "trainer"
    ROLLOUT = "rollout"
    INFERENCE = "inference"
    ENVIRONMENT = "environment"
    STORAGE = "storage"
    WEIGHT = "weight"
    WORKER = "worker"
    EVALUATOR = "evaluator"


@dataclass(frozen=True)
class TelemetryStatus:
    """Process-local exporter state."""

    configured: bool
    queued_records: int
    queued_bytes: int
    lost_records: int
    export_attempts: int
    export_failures: int
    export_retries: int
    rejected_records: int
    last_success_time_seconds: float | None
    oldest_queued_record_age_seconds: float


@dataclass(frozen=True)
class _Config:
    endpoint: str
    service: str
    attributes: dict[str, str]
    max_queue_records: int
    max_queue_bytes: int
    max_batch_records: int
    max_batch_bytes: int
    connect_timeout: float
    request_timeout: float
    retry_initial: float
    retry_maximum: float

    def validate(self) -> None:
        if not self.endpoint.startswith(("http://", "https://")):
            raise ValueError("endpoint must use http:// or https://")
        serialization.validate_string(self.service, "service")
        serialization.validate_attributes(self.attributes)
        limits = (
            self.max_queue_records,
            self.max_queue_bytes,
            self.max_batch_records,
            self.max_batch_bytes,
        )
        if any(value <= 0 for value in limits):
            raise ValueError("queue and batch limits must be positive")
        if self.max_batch_records > self.max_queue_records or self.max_batch_bytes > self.max_queue_bytes:
            raise ValueError("batch limits must not exceed queue limits")
        if self.max_batch_bytes > _MAX_REQUEST_BYTES:
            raise ValueError(f"max_batch_bytes must not exceed {_MAX_REQUEST_BYTES}")
        timings = (self.connect_timeout, self.request_timeout, self.retry_initial, self.retry_maximum)
        if not all(math.isfinite(value) for value in timings):
            raise ValueError("timeouts and retry intervals must be finite")
        if min(self.connect_timeout, self.request_timeout, self.retry_initial) <= 0:
            raise ValueError("timeouts and retry_initial must be positive")
        if self.retry_maximum < self.retry_initial:
            raise ValueError("retry_maximum must be at least retry_initial")


@dataclass(frozen=True)
class _Handle:
    name: str
    kind: str
    unit: str

    def _emit(self, value: float) -> None:
        _emit(self.kind, self.name, value=value, unit=self.unit)


class Counter(_Handle):
    """A counter whose calls emit deltas."""

    def add(self, value: float = 1.0, *, attributes: Mapping[str, str] | None = None) -> None:
        _emit(self.kind, self.name, value=value, unit=self.unit, attributes=attributes)


class Gauge(_Handle):
    """A gauge whose calls emit current values."""

    def set(self, value: float, *, attributes: Mapping[str, str] | None = None) -> None:
        _emit(self.kind, self.name, value=value, unit=self.unit, attributes=attributes)


class Histogram(_Handle):
    """A histogram whose calls emit individual observations."""

    def record(self, value: float, *, attributes: Mapping[str, str] | None = None) -> None:
        _emit(self.kind, self.name, value=value, unit=self.unit, attributes=attributes)


def snapshot_attributes(source_kind: str, temporality: str) -> dict[str, str]:
    """Return the canonical attributes for a source-provided snapshot."""
    return {"source_kind": source_kind, "source_temporality": temporality}


class _Response(Protocol):
    status_code: int
    headers: Mapping[str, str]

    def json(self) -> Any: ...


class _Transport(Protocol):
    def post(self, endpoint: str, body: bytes, batch_id: str, timeout: tuple[float, float]) -> _Response: ...

    def close(self) -> None: ...


class _RequestsTransport:
    def __init__(self) -> None:
        self._session = requests.Session()
        self._compressor = zstandard.ZstdCompressor(level=FINELOG_ZSTD_LEVEL)

    def post(self, endpoint: str, body: bytes, batch_id: str, timeout: tuple[float, float]) -> requests.Response:
        return self._session.post(
            endpoint,
            data=self._compressor.compress(body),
            headers={
                "Content-Encoding": "zstd",
                "Content-Type": "application/json",
                "Idempotency-Key": batch_id,
            },
            timeout=timeout,
        )

    def close(self) -> None:
        self._session.close()


@dataclass(frozen=True)
class _PendingBatch:
    batch_id: str
    body: bytes
    record_count: int
    record_bytes: int


@dataclass(frozen=True)
class _QueuedRecord:
    body: bytes
    enqueued_at: float


class _DeliveryOutcome(enum.Enum):
    DELIVERED = enum.auto()
    REJECTED = enum.auto()
    STOPPED = enum.auto()


class _Runtime:
    def __init__(self, config: _Config, transport: _Transport) -> None:
        self.config = config
        self._transport = transport
        self._records: deque[_QueuedRecord] = deque()
        self._condition = threading.Condition()
        self._resident_records = 0
        self._resident_bytes = 0
        self._lost_records = 0
        self._export_attempts = 0
        self._export_failures = 0
        self._export_retries = 0
        self._rejected_records = 0
        self._last_success_time_seconds: float | None = None
        self._oldest_queued_at: float | None = None
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="rigging-telemetry", daemon=True)
        self._thread.start()

    def emit(self, record: bytes) -> bool:
        """Return whether ``record`` was queued; a false result is counted as lost."""
        with self._condition:
            if self._stop or len(record) + self._empty_envelope_size() > self.config.max_batch_bytes:
                self._lost_records += 1
                return False
            if (
                self._resident_records >= self.config.max_queue_records
                or self._resident_bytes + len(record) > self.config.max_queue_bytes
            ):
                self._lost_records += 1
                return False
            queued = _QueuedRecord(record, time.monotonic())
            self._records.append(queued)
            if self._oldest_queued_at is None:
                self._oldest_queued_at = queued.enqueued_at
            self._resident_records += 1
            self._resident_bytes += len(record)
            self._condition.notify()
            return True

    def count_lost(self, records: int = 1) -> None:
        """Count records that could not be validated or queued for export."""
        with self._condition:
            self._lost_records += records

    def status(self) -> TelemetryStatus:
        with self._condition:
            oldest_age = 0.0
            if self._oldest_queued_at is not None:
                oldest_age = max(0.0, time.monotonic() - self._oldest_queued_at)
            return TelemetryStatus(
                True,
                self._resident_records,
                self._resident_bytes,
                self._lost_records,
                self._export_attempts,
                self._export_failures,
                self._export_retries,
                self._rejected_records,
                self._last_success_time_seconds,
                oldest_age,
            )

    def stop(self, timeout: float) -> None:
        with self._condition:
            self._stop = True
            self._condition.notify_all()
        self._thread.join(max(0.0, timeout))

    def flush(self, timeout: float) -> bool:
        """Wait until all resident records settle without stopping the exporter."""
        with self._condition:
            return self._condition.wait_for(lambda: self._resident_records == 0, timeout=timeout)

    def _run(self) -> None:
        try:
            while True:
                batch = self._next_batch()
                if batch is None:
                    return
                outcome = self._deliver(batch)
                self._settle(batch, lost=outcome is not _DeliveryOutcome.DELIVERED)
                if outcome is _DeliveryOutcome.STOPPED or (outcome is _DeliveryOutcome.REJECTED and self._stopping()):
                    self._abandon_remaining()
                    return
        except Exception:
            _warn("telemetry exporter stopped after an internal failure")
            self._abandon_remaining()
        finally:
            try:
                self._transport.close()
            except Exception:
                _warn("telemetry transport failed to close")

    def _next_batch(self) -> _PendingBatch | None:
        with self._condition:
            while not self._records and not self._stop:
                self._condition.wait()
            if not self._records:
                return None
            batch_id = str(uuid.uuid4())
            selected: list[_QueuedRecord] = []
            body_size = self._empty_envelope_size()
            while self._records and len(selected) < self.config.max_batch_records:
                record = self._records[0]
                extra = len(record.body) + (1 if selected else 0)
                if selected and body_size + extra > self.config.max_batch_bytes:
                    break
                if not selected and body_size + extra > self.config.max_batch_bytes:
                    self._records.popleft()
                    self._resident_records -= 1
                    self._resident_bytes -= len(record.body)
                    self._lost_records += 1
                    continue
                selected.append(self._records.popleft())
                body_size += extra
            if not selected:
                return None
        bodies = [record.body for record in selected]
        body = self._batch_body(batch_id, bodies)
        return _PendingBatch(batch_id, body, len(selected), sum(map(len, bodies)))

    def _batch_body(self, batch_id: str, records: list[bytes]) -> bytes:
        resource = serialization.json_bytes({"attributes": self.config.attributes, "service": self.config.service})
        return b"".join(
            (
                b'{"batch_id":"',
                batch_id.encode(),
                b'","records":[',
                b",".join(records),
                b'],"resource":',
                resource,
                b',"version":1}',
            )
        )

    def _empty_envelope_size(self) -> int:
        return len(self._batch_body(str(uuid.UUID(int=0)), []))

    def max_record_bytes(self) -> int:
        return min(self.config.max_queue_bytes, self.config.max_batch_bytes - self._empty_envelope_size())

    def _deliver(self, batch: _PendingBatch) -> _DeliveryOutcome:
        backoff = ExponentialBackoff(
            initial=self.config.retry_initial,
            maximum=self.config.retry_maximum,
            factor=2.0,
            jitter=0.1,
        )
        attempt = 0
        while True:
            attempt += 1
            with self._condition:
                self._export_attempts += 1
                if attempt > 1:
                    self._export_retries += 1
            try:
                response = self._transport.post(
                    self.config.endpoint,
                    batch.body,
                    batch.batch_id,
                    (self.config.connect_timeout, self.config.request_timeout),
                )
                if response.status_code == 200 and _valid_ack(response, batch.batch_id):
                    with self._condition:
                        self._last_success_time_seconds = time.time()
                    return _DeliveryOutcome.DELIVERED
                with self._condition:
                    self._export_failures += 1
                if 400 <= response.status_code < 500 and response.status_code != 429:
                    with self._condition:
                        self._rejected_records += batch.record_count
                    _warn(f"telemetry batch rejected with HTTP {response.status_code}")
                    return _DeliveryOutcome.REJECTED
            except Exception:
                with self._condition:
                    self._export_failures += 1
            if self._stopping():
                return _DeliveryOutcome.STOPPED
            delay = backoff.next_interval()
            with self._condition:
                self._condition.wait_for(lambda: self._stop, timeout=delay)

    def _settle(self, batch: _PendingBatch, *, lost: bool) -> None:
        with self._condition:
            self._resident_records -= batch.record_count
            self._resident_bytes -= batch.record_bytes
            if lost:
                self._lost_records += batch.record_count
            self._oldest_queued_at = self._records[0].enqueued_at if self._records else None
            self._condition.notify_all()

    def _abandon_remaining(self) -> None:
        with self._condition:
            abandoned = len(self._records)
            self._records.clear()
            self._resident_records -= abandoned
            self._resident_bytes = 0
            self._lost_records += abandoned
            self._oldest_queued_at = None
            self._condition.notify_all()

    def _stopping(self) -> bool:
        with self._condition:
            return self._stop


# The public API is an implicit process singleton: configure once, then use
# module-level handles without passing runtime state through every call site.
_configuration_lock = threading.Lock()
_runtime: _Runtime | None = None
_last_warning: float | None = None
_warning_lock = threading.Lock()


def counter(name: str, *, unit: str = "") -> Counter:
    """Declare a counter that emits deltas when configured."""
    return Counter(name, "counter", unit)


def gauge(name: str, *, unit: str = "") -> Gauge:
    """Declare a gauge that emits current values when configured."""
    return Gauge(name, "gauge", unit)


def histogram(name: str, *, unit: str = "") -> Histogram:
    """Declare a histogram that emits individual observations when configured."""
    return Histogram(name, "histogram", unit)


def event(
    name: str,
    body: serialization.EventBody,
    *,
    attributes: Mapping[str, str] | None = None,
) -> None:
    """Emit one structured event when configured."""
    _emit(_EVENT_KIND, name, body=body, attributes=attributes)


def configure(
    *,
    endpoint: str,
    service: str,
    attributes: Mapping[str, str] | None = None,
    max_queue_records: int = DEFAULT_MAX_QUEUE_RECORDS,
    max_queue_bytes: int = DEFAULT_MAX_QUEUE_BYTES,
    max_batch_records: int = DEFAULT_MAX_BATCH_RECORDS,
    max_batch_bytes: int = DEFAULT_MAX_BATCH_BYTES,
    connect_timeout: float = 1.0,
    request_timeout: float = 5.0,
    retry_initial: float = 0.1,
    retry_maximum: float = 5.0,
) -> None:
    """Configure the process-wide exporter; the first valid configuration wins."""
    global _runtime
    try:
        config = _Config(
            endpoint=endpoint,
            service=service,
            attributes=dict(attributes or {}),
            max_queue_records=max_queue_records,
            max_queue_bytes=max_queue_bytes,
            max_batch_records=max_batch_records,
            max_batch_bytes=max_batch_bytes,
            connect_timeout=connect_timeout,
            request_timeout=request_timeout,
            retry_initial=retry_initial,
            retry_maximum=retry_maximum,
        )
        config.validate()
    except Exception as error:
        _warn(f"telemetry export disabled by invalid configuration: {error}")
        return
    with _configuration_lock:
        if _runtime is not None:
            if _runtime.config != config:
                _warn("telemetry is already configured; keeping the first configuration")
            return
        try:
            _runtime = _Runtime(config, _RequestsTransport())
        except Exception as error:
            _warn(f"telemetry export could not start: {error}")


def shutdown(timeout: float = 5.0) -> None:
    """Attempt a bounded flush and disable the process exporter."""
    global _runtime
    with _configuration_lock:
        runtime, _runtime = _runtime, None
    if runtime is not None:
        invalid_timeout = False
        try:
            budget = float(timeout)
        except Exception:
            budget = 0.0
            invalid_timeout = True
        if not math.isfinite(budget) or budget < 0:
            budget = 0.0
            invalid_timeout = True
        else:
            budget = min(budget, _MAX_SHUTDOWN_TIMEOUT)
        if invalid_timeout:
            _warn("telemetry shutdown received an invalid timeout; skipping the final flush")
        try:
            runtime.stop(budget)
        except Exception:
            _warn("telemetry exporter failed during shutdown")


def flush(timeout: float = 5.0) -> bool:
    """Wait for queued telemetry to settle without disabling the exporter."""
    runtime = _runtime
    return runtime is None or runtime.flush(timeout)


def runtime_status() -> TelemetryStatus:
    """Return a bounded snapshot of process-local queue and loss state."""
    runtime = _runtime
    return runtime.status() if runtime is not None else TelemetryStatus(False, 0, 0, 0, 0, 0, 0, 0, None, 0.0)


def record_runtime_health() -> TelemetryStatus:
    """Publish one bounded snapshot of direct exporter loss and freshness."""
    status = runtime_status()
    if not status.configured:
        return status
    current = snapshot_attributes("gauge", CURRENT_SNAPSHOT)
    cumulative = snapshot_attributes("counter", CUMULATIVE_SNAPSHOT)
    gauge("queue_depth", unit="{record}").set(
        status.queued_records,
        attributes={**current, "queue_kind": "telemetry_export"},
    )
    gauge("telemetry_queue_bytes", unit="By").set(status.queued_bytes, attributes=current)
    gauge("telemetry_lost_records", unit="{record}").set(status.lost_records, attributes=cumulative)
    gauge("telemetry_export_attempts", unit="{attempt}").set(status.export_attempts, attributes=cumulative)
    gauge("telemetry_export_failures", unit="{attempt}").set(status.export_failures, attributes=cumulative)
    gauge("telemetry_export_retries", unit="{attempt}").set(status.export_retries, attributes=cumulative)
    gauge("telemetry_rejected_records", unit="{record}").set(status.rejected_records, attributes=cumulative)
    gauge("telemetry_oldest_queued_age_seconds", unit="s").set(
        status.oldest_queued_record_age_seconds,
        attributes=current,
    )
    if status.last_success_time_seconds is not None:
        gauge("progress_time_seconds", unit="s").set(
            status.last_success_time_seconds,
            attributes={**current, "progress_kind": "telemetry_export"},
        )
    return status


def _emit(
    kind: str,
    name: str,
    *,
    value: float | None = None,
    body: serialization.EventBody | None = None,
    unit: str = "",
    attributes: Mapping[str, str] | None = None,
) -> None:
    runtime = _runtime
    if runtime is None:
        return
    _emit_to_runtime(runtime, kind, name, value=value, body=body, unit=unit, attributes=attributes)


def _emit_to_runtime(
    runtime: _Runtime,
    kind: str,
    name: str,
    *,
    value: float | None = None,
    body: serialization.EventBody | None = None,
    unit: str = "",
    attributes: Mapping[str, str] | None = None,
) -> bool:
    """Return whether one validated record was queued in the selected runtime."""
    try:
        serialization.validate_string(name, "name")
        if unit:
            serialization.validate_string(unit, "unit")
        attrs = dict(attributes or {})
        serialization.validate_attributes(attrs)
        record: dict[str, Any] = {
            "attributes": attrs,
            "kind": kind,
            "name": name,
            "timestamp_ms": int(time.time() * 1_000),
        }
        if kind == _EVENT_KIND:
            record_budget = runtime.max_record_bytes()
            fixed_size = len(serialization.json_bytes_bounded({**record, "body": None}, record_budget))
            if body is None:
                raise ValueError("event body is required")
            record["body"] = serialization.event_fields(body, record_budget - fixed_size)
        else:
            numeric = float(value)  # type: ignore[arg-type]
            if not math.isfinite(numeric):
                raise ValueError("metric value must be finite")
            record["unit"] = unit
            record["value"] = numeric
        return runtime.emit(serialization.json_bytes_bounded(record, runtime.max_record_bytes()))
    except Exception:
        runtime.count_lost()
        return False


def _valid_ack(response: _Response, batch_id: str) -> bool:
    try:
        payload = response.json()
        return payload.get("batch_id") == batch_id and payload.get("status") == "accepted"
    except Exception:
        return False


def _warn(message: str) -> None:
    global _last_warning
    now = time.monotonic()
    with _warning_lock:
        if _last_warning is not None and now - _last_warning < _WARNING_INTERVAL:
            return
        _last_warning = now
    try:
        logger.warning(message)
    except Exception:
        pass


atexit.register(shutdown)
