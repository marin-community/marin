# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import pytest
import requests
from rigging import telemetry


@dataclass
class FakeResponse:
    status_code: int
    payload: dict[str, object]
    headers: dict[str, str]

    def json(self) -> dict[str, object]:
        return self.payload


TransportOutcome = Callable[[str], FakeResponse]


def status_outcome(status_code: int) -> TransportOutcome:
    return lambda batch_id: FakeResponse(
        status_code,
        {"batch_id": batch_id, "status": "accepted"} if status_code == 200 else {"error": {"code": "rejected"}},
        {},
    )


def invalid_ack_outcome(batch_id: str) -> FakeResponse:
    del batch_id
    return FakeResponse(200, {"batch_id": "wrong", "status": "accepted"}, {})


def error_outcome(error: BaseException) -> TransportOutcome:
    def raise_error(batch_id: str) -> FakeResponse:
        del batch_id
        raise error

    return raise_error


class RecordingTransport:
    def __init__(self, outcomes: list[TransportOutcome] | None = None) -> None:
        self.outcomes = deque(outcomes or [status_outcome(200)])
        self.requests: list[tuple[str, bytes, str, tuple[float, float]]] = []
        self.accepted = threading.Event()
        self.rejected = threading.Event()
        self.closed = threading.Event()

    def post(self, endpoint: str, body: bytes, batch_id: str, timeout: tuple[float, float]) -> FakeResponse:
        self.requests.append((endpoint, body, batch_id, timeout))
        outcome = self.outcomes.popleft() if self.outcomes else status_outcome(200)
        response = outcome(batch_id)
        if response.status_code == 200 and response.payload.get("batch_id") == batch_id:
            self.accepted.set()
        elif response.status_code >= 400:
            self.rejected.set()
        return response

    def close(self) -> None:
        self.closed.set()


class BlockingTransport(RecordingTransport):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def post(self, endpoint: str, body: bytes, batch_id: str, timeout: tuple[float, float]) -> FakeResponse:
        self.requests.append((endpoint, body, batch_id, timeout))
        self.started.set()
        self.release.wait()
        self.accepted.set()
        return FakeResponse(200, {"batch_id": batch_id, "status": "accepted"}, {})


class RaisingHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        raise RuntimeError("logging is broken")


class CountingList(list):
    def __init__(self) -> None:
        super().__init__([0] * 10_000)
        self.visited = 0

    def __iter__(self):
        for value in super().__iter__():
            self.visited += 1
            yield value


class CountingDict(dict):
    def __init__(self) -> None:
        super().__init__((str(index), 0) for index in range(10_000))
        self.visited = 0

    def items(self):
        for item in super().items():
            self.visited += 1
            yield item


@pytest.fixture(autouse=True)
def reset_telemetry():
    telemetry.shutdown(0.01)
    yield
    telemetry.shutdown(0.1)


def configure(monkeypatch: pytest.MonkeyPatch, transport: RecordingTransport, **overrides: object) -> None:
    monkeypatch.setattr(telemetry, "_RequestsTransport", lambda: transport)
    options = {
        "endpoint": "http://finelog.test/v1/telemetry",
        "service": "test-service",
        "retry_initial": 0.001,
        "retry_maximum": 0.002,
    }
    options.update(overrides)
    telemetry.configure(**options)


def test_unconfigured_instruments_are_noops() -> None:
    telemetry.counter("requests").add()
    telemetry.gauge("workers").set(2)
    telemetry.histogram("latency", unit="ms").record(1.5)
    telemetry.event("ready", {"worker": 3})

    assert telemetry.runtime_status() == telemetry.TelemetryStatus(False, 0, 0, 0)


def test_invalid_configuration_stays_inert(caplog: pytest.LogCaptureFixture) -> None:
    telemetry.configure(endpoint="file:///tmp/telemetry", service="test")
    telemetry.counter("requests").add()

    assert telemetry.runtime_status().configured is False
    assert "invalid configuration" in caplog.text


def test_retry_reuses_exact_batch_id_and_body(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = RecordingTransport(
        [error_outcome(requests.ConnectionError("offline")), invalid_ack_outcome, status_outcome(200)]
    )
    configure(monkeypatch, transport)

    telemetry.counter("requests", unit="request").add(2, attributes={"route": "/chat"})

    assert transport.accepted.wait(1)
    telemetry.shutdown(0.2)
    assert len(transport.requests) == 3
    assert len({request[2] for request in transport.requests}) == 1
    assert len({request[1] for request in transport.requests}) == 1
    payload = json.loads(transport.requests[0][1])
    assert payload["batch_id"] == transport.requests[0][2]
    assert payload["records"][0]["value"] == 2


def test_network_wait_never_runs_on_emitting_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = BlockingTransport()
    configure(monkeypatch, transport, max_batch_records=1)
    counter = telemetry.counter("requests")
    counter.add()
    assert transport.started.wait(1)

    emitter = threading.Thread(target=counter.add)
    emitter.start()
    emitter.join(0.2)

    assert not emitter.is_alive()
    assert telemetry.runtime_status().queued_records == 2
    transport.release.set()


@pytest.mark.parametrize(
    ("queue_records", "queue_bytes"),
    [(2, 1 << 20), (100, 300)],
)
def test_pending_and_queued_records_share_hard_caps(
    monkeypatch: pytest.MonkeyPatch, queue_records: int, queue_bytes: int
) -> None:
    transport = BlockingTransport()
    configure(
        monkeypatch,
        transport,
        max_queue_records=queue_records,
        max_queue_bytes=queue_bytes,
        max_batch_records=1,
        max_batch_bytes=queue_bytes,
    )
    metric = telemetry.gauge("worker.load")
    metric.set(1)
    assert transport.started.wait(1)

    threads = [threading.Thread(target=metric.set, args=(value,)) for value in range(40)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(0.5)

    status = telemetry.runtime_status()
    assert status.queued_records <= queue_records
    assert status.queued_bytes <= queue_bytes
    assert status.lost_records > 0
    transport.release.set()


def test_terminal_response_drops_batch_and_records_loss(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = RecordingTransport([status_outcome(422)])
    configure(monkeypatch, transport)

    telemetry.event("invalid", {"reason": "test"})

    assert transport.rejected.wait(1)
    deadline = time.monotonic() + 1
    while telemetry.runtime_status().queued_records and time.monotonic() < deadline:
        threading.Event().wait(0.001)
    status = telemetry.runtime_status()
    assert status.queued_records == 0
    assert status.lost_records == 1


def test_nonserializable_event_is_lost_without_escaping(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = RecordingTransport()
    configure(monkeypatch, transport)

    telemetry.event("bad", object())

    assert telemetry.runtime_status().lost_records == 1
    assert not transport.requests


def deep_event_body() -> object:
    body: object = "leaf"
    for _ in range(33):
        body = {"child": body}
    return body


@pytest.mark.parametrize(
    "body",
    [
        None,
        {"outer": "x" * 4_097},
        {"k" * 4_097: "value"},
        deep_event_body(),
        {"integer": -(1 << 63) - 1},
        {"integer": 1 << 64},
    ],
)
def test_invalid_event_body_cannot_poison_valid_metric(monkeypatch: pytest.MonkeyPatch, body: object) -> None:
    transport = RecordingTransport()
    configure(monkeypatch, transport)

    telemetry.event("invalid", body)
    telemetry.counter("valid").add()

    assert transport.accepted.wait(1)
    payload = json.loads(transport.requests[0][1])
    assert [record["name"] for record in payload["records"]] == ["valid"]
    assert telemetry.runtime_status().lost_records == 1


@pytest.mark.parametrize("body", [CountingList(), CountingDict()])
def test_event_validation_stops_at_record_node_budget(
    monkeypatch: pytest.MonkeyPatch, body: CountingList | CountingDict
) -> None:
    transport = RecordingTransport()
    configure(
        monkeypatch,
        transport,
        max_queue_bytes=300,
        max_batch_bytes=300,
    )

    telemetry.event("oversized", body)
    telemetry.counter("valid").add()

    assert transport.accepted.wait(1)
    assert body.visited <= 300
    assert [record["name"] for record in json.loads(transport.requests[0][1])["records"]] == ["valid"]


def test_event_validation_stops_at_cumulative_string_bytes() -> None:
    class CountingString(str):
        encode_calls = 0

        def encode(self, encoding: str = "utf-8", errors: str = "strict") -> bytes:
            type(self).encode_calls += 1
            return super().encode(encoding, errors)

    body = [CountingString("x" * 30) for _ in range(4)]

    with pytest.raises(ValueError, match="record byte budget"):
        telemetry._validate_event_body(body, 100)

    assert CountingString.encode_calls == 3


def test_event_integer_boundaries_match_serde_json(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = RecordingTransport()
    configure(monkeypatch, transport)

    telemetry.event("integer.bounds", {"minimum": -(1 << 63), "maximum": (1 << 64) - 1})

    assert transport.accepted.wait(1)
    body = json.loads(transport.requests[0][1])["records"][0]["body"]
    assert body == {"minimum": -(1 << 63), "maximum": (1 << 64) - 1}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("connect_timeout", float("nan")),
        ("request_timeout", float("inf")),
        ("retry_initial", float("-inf")),
        ("retry_maximum", float("nan")),
    ],
)
def test_nonfinite_network_configuration_stays_inert(monkeypatch: pytest.MonkeyPatch, field: str, value: float) -> None:
    transport = RecordingTransport()
    configure(monkeypatch, transport, **{field: value})

    assert telemetry.runtime_status().configured is False
    assert not transport.requests


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), float("-inf"), -1, "bad", None])
def test_invalid_shutdown_budget_is_bounded(monkeypatch: pytest.MonkeyPatch, timeout: object) -> None:
    transport = BlockingTransport()
    configure(monkeypatch, transport)
    telemetry.event("stuck", {})
    assert transport.started.wait(1)

    started = time.monotonic()
    telemetry.shutdown(timeout)  # type: ignore[arg-type]
    elapsed = time.monotonic() - started
    transport.release.set()

    assert elapsed < 0.2
    assert telemetry.runtime_status().configured is False


def test_huge_shutdown_budget_is_clamped_and_stop_failure_is_contained(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: list[float] = []

    class RaisingRuntime:
        def stop(self, timeout: float) -> None:
            observed.append(timeout)
            raise OverflowError("platform timer overflow")

    monkeypatch.setattr(telemetry, "_runtime", RaisingRuntime())

    telemetry.shutdown(1e300)

    assert observed == [5.0]
    assert telemetry.runtime_status().configured is False


def test_raising_log_handler_cannot_escape_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    handler = RaisingHandler()
    telemetry.logger.addHandler(handler)
    monkeypatch.setattr(telemetry, "_last_warning", None)
    try:
        telemetry.configure(endpoint="invalid", service="test")
    finally:
        telemetry.logger.removeHandler(handler)

    assert telemetry.runtime_status().configured is False


def test_raising_log_handler_cannot_stop_exporter_settlement(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = RecordingTransport([status_outcome(422)])
    configure(monkeypatch, transport)
    handler = RaisingHandler()
    telemetry.logger.addHandler(handler)
    monkeypatch.setattr(telemetry, "_last_warning", None)
    try:
        telemetry.event("rejected", {})
        assert transport.rejected.wait(1)
        deadline = time.monotonic() + 1
        while telemetry.runtime_status().queued_records and time.monotonic() < deadline:
            threading.Event().wait(0.001)
    finally:
        telemetry.logger.removeHandler(handler)

    assert telemetry.runtime_status().queued_records == 0
    assert telemetry.runtime_status().lost_records == 1


def test_first_warning_is_emitted_before_one_minute_of_process_uptime(monkeypatch: pytest.MonkeyPatch) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(telemetry, "_last_warning", None)
    monkeypatch.setattr(telemetry.time, "monotonic", lambda: 10.0)
    monkeypatch.setattr(telemetry.logger, "warning", warnings.append)

    telemetry.configure(endpoint="invalid", service="test")

    assert warnings == ["telemetry export disabled by invalid configuration: endpoint must use http:// or https://"]


def test_same_configuration_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    transports: list[RecordingTransport] = []

    def factory() -> RecordingTransport:
        transport = RecordingTransport()
        transports.append(transport)
        return transport

    monkeypatch.setattr(telemetry, "_RequestsTransport", factory)
    options = {
        "endpoint": "http://finelog.test/v1/telemetry",
        "service": "test-service",
        "retry_initial": 0.001,
        "retry_maximum": 0.002,
    }
    telemetry.configure(**options)
    telemetry.configure(**options)
    telemetry.event("ready", {})

    assert transports[0].accepted.wait(1)
    assert len(transports) == 1


def test_shutdown_returns_within_budget_when_transport_is_stuck(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = BlockingTransport()
    configure(monkeypatch, transport)
    telemetry.event("stuck", {})
    assert transport.started.wait(1)

    started = time.monotonic()
    telemetry.shutdown(0.03)
    elapsed = time.monotonic() - started
    transport.release.set()

    assert elapsed < 0.2
    assert telemetry.runtime_status().configured is False
