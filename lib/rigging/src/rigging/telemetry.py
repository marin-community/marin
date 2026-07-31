# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded, best-effort process telemetry exported directly to Finelog."""

import atexit
import json
import logging
import math
import threading
import time
import uuid
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import requests

from rigging.timing import ExponentialBackoff

logger = logging.getLogger(__name__)

DEFAULT_MAX_QUEUE_RECORDS = 10_000
DEFAULT_MAX_QUEUE_BYTES = 16 << 20
DEFAULT_MAX_BATCH_RECORDS = 1_000
DEFAULT_MAX_BATCH_BYTES = 1 << 20
_MAX_REQUEST_BYTES = 4 << 20
_MAX_SHUTDOWN_TIMEOUT = 5.0
_WARNING_INTERVAL = 60.0
_MAX_ATTRIBUTES = 64
_MAX_STRING_LENGTH = 4_096


@dataclass(frozen=True)
class TelemetryStatus:
    """Process-local exporter state."""

    configured: bool
    queued_records: int
    queued_bytes: int
    lost_records: int


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
        _validate_string(self.service, "service")
        _validate_attributes(self.attributes)
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

    def post(self, endpoint: str, body: bytes, batch_id: str, timeout: tuple[float, float]) -> requests.Response:
        return self._session.post(
            endpoint,
            data=body,
            headers={"Content-Type": "application/json", "Idempotency-Key": batch_id},
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


class _Runtime:
    def __init__(self, config: _Config, transport: _Transport) -> None:
        self.config = config
        self._transport = transport
        self._records: deque[bytes] = deque()
        self._condition = threading.Condition()
        self._resident_records = 0
        self._resident_bytes = 0
        self._lost_records = 0
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="rigging-telemetry", daemon=True)
        self._thread.start()

    def emit(self, record: bytes) -> None:
        with self._condition:
            if self._stop or len(record) + self._empty_envelope_size() > self.config.max_batch_bytes:
                self._lost_records += 1
                return
            if (
                self._resident_records >= self.config.max_queue_records
                or self._resident_bytes + len(record) > self.config.max_queue_bytes
            ):
                self._lost_records += 1
                return
            self._records.append(record)
            self._resident_records += 1
            self._resident_bytes += len(record)
            self._condition.notify()

    def status(self) -> TelemetryStatus:
        with self._condition:
            return TelemetryStatus(True, self._resident_records, self._resident_bytes, self._lost_records)

    def stop(self, timeout: float) -> None:
        with self._condition:
            self._stop = True
            self._condition.notify_all()
        self._thread.join(max(0.0, timeout))

    def _run(self) -> None:
        try:
            while True:
                batch = self._next_batch()
                if batch is None:
                    return
                delivered = self._deliver(batch)
                self._settle(batch, lost=not delivered)
                if self._stopping():
                    self._abandon_remaining()
                    return
        except Exception:
            _warn("telemetry exporter stopped after an internal failure")
            self._abandon_remaining()
        finally:
            try:
                self._transport.close()
            except Exception:
                pass

    def _next_batch(self) -> _PendingBatch | None:
        with self._condition:
            while not self._records and not self._stop:
                self._condition.wait()
            if not self._records:
                return None
            batch_id = str(uuid.uuid4())
            selected: list[bytes] = []
            body_size = self._empty_envelope_size()
            while self._records and len(selected) < self.config.max_batch_records:
                record = self._records[0]
                extra = len(record) + (1 if selected else 0)
                if selected and body_size + extra > self.config.max_batch_bytes:
                    break
                if not selected and body_size + extra > self.config.max_batch_bytes:
                    self._records.popleft()
                    self._resident_records -= 1
                    self._resident_bytes -= len(record)
                    self._lost_records += 1
                    continue
                selected.append(self._records.popleft())
                body_size += extra
            if not selected:
                return None
        body = self._batch_body(batch_id, selected)
        return _PendingBatch(batch_id, body, len(selected), sum(map(len, selected)))

    def _batch_body(self, batch_id: str, records: list[bytes]) -> bytes:
        resource = _json_bytes({"attributes": self.config.attributes, "service": self.config.service})
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

    def _deliver(self, batch: _PendingBatch) -> bool:
        backoff = ExponentialBackoff(
            initial=self.config.retry_initial,
            maximum=self.config.retry_maximum,
            factor=2.0,
            jitter=0.1,
        )
        while True:
            try:
                response = self._transport.post(
                    self.config.endpoint,
                    batch.body,
                    batch.batch_id,
                    (self.config.connect_timeout, self.config.request_timeout),
                )
                if response.status_code == 200 and _valid_ack(response, batch.batch_id):
                    return True
                if 400 <= response.status_code < 500 and response.status_code != 429:
                    _warn(f"telemetry batch rejected with HTTP {response.status_code}")
                    return False
            except Exception:
                pass
            if self._stopping():
                return False
            delay = backoff.next_interval()
            with self._condition:
                self._condition.wait_for(lambda: self._stop, timeout=delay)

    def _settle(self, batch: _PendingBatch, *, lost: bool) -> None:
        with self._condition:
            self._resident_records -= batch.record_count
            self._resident_bytes -= batch.record_bytes
            if lost:
                self._lost_records += batch.record_count

    def _abandon_remaining(self) -> None:
        with self._condition:
            abandoned = len(self._records)
            self._records.clear()
            self._resident_records -= abandoned
            self._resident_bytes = 0
            self._lost_records += abandoned

    def _stopping(self) -> bool:
        with self._condition:
            return self._stop


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


def event(name: str, body: Any, *, attributes: Mapping[str, str] | None = None) -> None:
    """Emit one structured event when configured."""
    _emit("event", name, body=body, attributes=attributes)


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
    """Configure the process exporter; invalid configuration leaves it inert."""
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
        try:
            budget = float(timeout)
        except Exception:
            budget = 0.0
        budget = min(budget, _MAX_SHUTDOWN_TIMEOUT) if math.isfinite(budget) and budget > 0 else 0.0
        try:
            runtime.stop(budget)
        except Exception:
            pass


def runtime_status() -> TelemetryStatus:
    """Return a bounded snapshot of process-local queue and loss state."""
    runtime = _runtime
    return runtime.status() if runtime is not None else TelemetryStatus(False, 0, 0, 0)


def _emit(
    kind: str,
    name: str,
    *,
    value: float | None = None,
    body: Any = None,
    unit: str = "",
    attributes: Mapping[str, str] | None = None,
) -> None:
    runtime = _runtime
    if runtime is None:
        return
    try:
        _validate_string(name, "name")
        if unit:
            _validate_string(unit, "unit")
        attrs = dict(attributes or {})
        _validate_attributes(attrs)
        record: dict[str, Any] = {
            "attributes": attrs,
            "kind": kind,
            "name": name,
            "timestamp_ms": int(time.time() * 1_000),
        }
        if kind == "event":
            record_budget = runtime.max_record_bytes()
            fixed_size = len(_json_bytes_bounded({**record, "body": None}, record_budget))
            _validate_event_body(body, record_budget - fixed_size)
            record["body"] = body
        else:
            numeric = float(value)  # type: ignore[arg-type]
            if not math.isfinite(numeric):
                raise ValueError("metric value must be finite")
            record["unit"] = unit
            record["value"] = numeric
        runtime.emit(_json_bytes_bounded(record, runtime.max_record_bytes()))
    except Exception:
        with runtime._condition:
            runtime._lost_records += 1


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()


def _json_bytes_bounded(value: Any, limit: int) -> bytes:
    encoded = bytearray()
    encoder = json.JSONEncoder(allow_nan=False, separators=(",", ":"), sort_keys=True)
    for chunk in encoder.iterencode(value):
        chunk_bytes = chunk.encode()
        if len(encoded) + len(chunk_bytes) > limit:
            raise ValueError("encoded telemetry record exceeds the batch limit")
        encoded.extend(chunk_bytes)
    return bytes(encoded)


def _validate_event_body(value: Any, budget: int) -> None:
    if value is None:
        raise ValueError("event body must not be None")

    def consume(remaining: int, amount: int) -> int:
        if amount > remaining:
            raise ValueError("event body exceeds the record byte budget")
        return remaining - amount

    def string_size(item: Any, field: str, remaining: int, punctuation: int) -> int:
        if not isinstance(item, str) or (field == "event body key" and not item):
            raise ValueError(f"{field} must be a nonempty string")
        if len(item) > _MAX_STRING_LENGTH:
            raise ValueError(f"{field} exceeds {_MAX_STRING_LENGTH} bytes")
        if len(item) + punctuation > remaining:
            raise ValueError("event body exceeds the record byte budget")
        encoded_size = len(item.encode())
        if encoded_size > _MAX_STRING_LENGTH:
            raise ValueError(f"{field} exceeds {_MAX_STRING_LENGTH} bytes")
        return encoded_size + punctuation

    def visit(item: Any, depth: int, remaining_nodes: int, remaining_bytes: int) -> tuple[int, int]:
        if remaining_nodes <= 0:
            raise ValueError("event body exceeds the record node budget")
        if depth > 32:
            raise ValueError("event body exceeds JSON depth 32")
        remaining_nodes -= 1
        if isinstance(item, str):
            remaining_bytes = consume(remaining_bytes, string_size(item, "event body string", remaining_bytes, 2))
        elif item is None:
            remaining_bytes = consume(remaining_bytes, 4)
        elif isinstance(item, bool):
            remaining_bytes = consume(remaining_bytes, 5)
        elif isinstance(item, int):
            if not -(1 << 63) <= item <= (1 << 64) - 1:
                raise ValueError("event body integer is outside serde_json's exact range")
            remaining_bytes = consume(remaining_bytes, 20)
        elif isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError("event body numbers must be finite")
            remaining_bytes = consume(remaining_bytes, 32)
        elif isinstance(item, dict):
            remaining_bytes = consume(remaining_bytes, 2)
            for index, (key, child) in enumerate(item.items()):
                if remaining_nodes <= 0:
                    raise ValueError("event body exceeds the record node budget")
                remaining_nodes -= 1
                key_size = string_size(key, "event body key", remaining_bytes, 3 + bool(index))
                remaining_bytes = consume(remaining_bytes, key_size)
                remaining_nodes, remaining_bytes = visit(child, depth + 1, remaining_nodes, remaining_bytes)
        elif isinstance(item, list | tuple):
            remaining_bytes = consume(remaining_bytes, 2)
            for index, child in enumerate(item):
                if index:
                    remaining_bytes = consume(remaining_bytes, 1)
                remaining_nodes, remaining_bytes = visit(child, depth + 1, remaining_nodes, remaining_bytes)
        else:
            raise ValueError("event body must contain only JSON values")
        return remaining_nodes, remaining_bytes

    visit(value, 0, budget, budget)


def _validate_string(value: str, field: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > _MAX_STRING_LENGTH
        or len(value.encode()) > _MAX_STRING_LENGTH
    ):
        raise ValueError(f"{field} must be a nonempty string of at most {_MAX_STRING_LENGTH} bytes")


def _validate_attributes(attributes: Mapping[str, str]) -> None:
    if len(attributes) > _MAX_ATTRIBUTES:
        raise ValueError(f"attributes may contain at most {_MAX_ATTRIBUTES} entries")
    for key, value in attributes.items():
        _validate_string(key, "attribute key")
        _validate_string(value, "attribute value")


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
