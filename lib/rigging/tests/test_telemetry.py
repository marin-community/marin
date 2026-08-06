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
import zstandard
from rigging import telemetry


@dataclass
class FakeResponse:
    status_code: int
    payload: dict[str, object]
    headers: dict[str, str]

    def json(self) -> dict[str, object]:
        return self.payload


TransportOutcome = Callable[[str], FakeResponse]


def status_outcome(status_code: int, *, headers: dict[str, str] | None = None) -> TransportOutcome:
    return lambda batch_id: FakeResponse(
        status_code,
        {"batch_id": batch_id, "status": "accepted"} if status_code == 200 else {"error": {"code": "rejected"}},
        headers or {},
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


class RecordingSession:
    def __init__(self, outcomes: list[TransportOutcome] | None = None) -> None:
        self.outcomes = deque(outcomes or [status_outcome(200)])
        self.requests: list[tuple[str, bytes, dict[str, str], tuple[float, float]]] = []
        self.closed = False

    def post(
        self,
        endpoint: str,
        *,
        data: bytes,
        headers: dict[str, str],
        timeout: tuple[float, float],
    ) -> FakeResponse:
        self.requests.append((endpoint, data, headers, timeout))
        batch_id = headers["Idempotency-Key"]
        outcome = self.outcomes.popleft() if self.outcomes else status_outcome(200)
        return outcome(batch_id)

    def close(self) -> None:
        self.closed = True


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
    telemetry.event("ready", telemetry.serialization.EventBody({"worker": 3}))

    assert telemetry.runtime_status() == telemetry.TelemetryStatus(False, 0, 0, 0, 0, 0, 0, 0, None, 0.0)


def test_requests_transport_sends_zstd_body(monkeypatch: pytest.MonkeyPatch) -> None:
    session = RecordingSession([status_outcome(200, headers={"Accept-Encoding": "zstd"})])
    monkeypatch.setattr(telemetry.requests, "Session", lambda: session)
    transport = telemetry._RequestsTransport()
    record = b'{"name":"worker_cpu","value":1}'
    body = b'{"records":[' + b",".join([record] * 100) + b"]}"

    transport.post("http://finelog/v1/telemetry", body, "batch-1", (1.0, 2.0))

    assert len(session.requests) == 1
    endpoint, compressed, headers, timeout = session.requests[0]
    assert endpoint == "http://finelog/v1/telemetry"
    assert headers == {
        "Content-Encoding": "zstd",
        "Content-Type": "application/json",
        "Idempotency-Key": "batch-1",
    }
    assert timeout == (1.0, 2.0)
    assert zstandard.ZstdDecompressor().decompress(compressed) == body


def test_requests_transport_falls_back_until_server_advertises_zstd(monkeypatch: pytest.MonkeyPatch) -> None:
    session = RecordingSession(
        [
            status_outcome(400),
            status_outcome(200),
            status_outcome(200, headers={"Accept-Encoding": "zstd"}),
            status_outcome(200, headers={"Accept-Encoding": "zstd"}),
        ]
    )
    monkeypatch.setattr(telemetry.requests, "Session", lambda: session)
    transport = telemetry._RequestsTransport()
    body = b'{"records":[{"name":"worker_cpu","value":1}]}'

    transport.post("http://finelog/v1/telemetry", body, "batch-1", (1.0, 2.0))
    transport.post("http://finelog/v1/telemetry", body, "batch-2", (1.0, 2.0))
    transport.post("http://finelog/v1/telemetry", body, "batch-3", (1.0, 2.0))

    assert len(session.requests) == 4
    assert session.requests[0][2]["Content-Encoding"] == "zstd"
    assert zstandard.ZstdDecompressor().decompress(session.requests[0][1]) == body
    assert session.requests[1][1] == body
    assert "Content-Encoding" not in session.requests[1][2]
    assert session.requests[2][1] == body
    assert "Content-Encoding" not in session.requests[2][2]
    assert session.requests[3][2]["Content-Encoding"] == "zstd"
    assert zstandard.ZstdDecompressor().decompress(session.requests[3][1]) == body


def test_invalid_configuration_stays_inert(caplog: pytest.LogCaptureFixture) -> None:
    telemetry.configure(endpoint="file:///tmp/telemetry", service="test")
    telemetry.counter("requests").add()

    assert telemetry.runtime_status().configured is False
    assert "invalid configuration" in caplog.text


def test_custom_resource_role_is_exported(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = RecordingTransport()
    configure(monkeypatch, transport, attributes={"role": "skyrl_driver"})

    telemetry.counter("requests").add()

    assert transport.accepted.wait(1)
    payload = json.loads(transport.requests[0][1])
    assert payload["resource"]["attributes"]["role"] == "skyrl_driver"


def test_retry_reuses_exact_batch_id_and_body(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = RecordingTransport(
        [error_outcome(requests.ConnectionError("offline")), invalid_ack_outcome, status_outcome(200)]
    )
    configure(monkeypatch, transport)

    telemetry.counter("requests", unit="request").add(2, attributes={"route": "/chat"})

    assert transport.accepted.wait(1)
    status = telemetry.runtime_status()
    telemetry.shutdown(0.2)
    delivery_requests = transport.requests
    assert len(delivery_requests) == 3
    assert len({request[2] for request in delivery_requests}) == 1
    assert len({request[1] for request in delivery_requests}) == 1
    payload = json.loads(transport.requests[0][1])
    assert payload["batch_id"] == transport.requests[0][2]
    assert payload["records"][0]["value"] == 2
    assert status.export_attempts == 3
    assert status.export_failures == 2
    assert status.export_retries == 2
    assert status.last_success_time_seconds is not None


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

    telemetry.event("invalid", telemetry.serialization.EventBody({"reason": "test"}))

    assert transport.rejected.wait(1)
    deadline = time.monotonic() + 1
    while telemetry.runtime_status().queued_records and time.monotonic() < deadline:
        threading.Event().wait(0.001)
    status = telemetry.runtime_status()
    assert status.queued_records == 0
    assert status.lost_records == 1
    assert status.rejected_records == 1


def test_exporter_health_records_queue_loss_retry_and_freshness(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = RecordingTransport([error_outcome(requests.ConnectionError("offline")), status_outcome(200)])
    configure(monkeypatch, transport, max_batch_records=20)
    telemetry.counter("application_progress").add()
    assert transport.accepted.wait(1)

    status = telemetry.record_runtime_health()
    deadline = time.monotonic() + 1
    while telemetry.runtime_status().queued_records and time.monotonic() < deadline:
        threading.Event().wait(0.001)

    records = [record for request in transport.requests for record in json.loads(request[1])["records"]]
    by_name = {record["name"]: record for record in records}
    assert status.export_attempts == 2
    assert status.export_failures == 1
    assert status.export_retries == 1
    assert by_name["telemetry_export_attempts"]["value"] == 2
    assert by_name["telemetry_export_failures"]["value"] == 1
    assert by_name["telemetry_export_retries"]["value"] == 1
    for name in (
        "telemetry_lost_records",
        "telemetry_export_attempts",
        "telemetry_export_failures",
        "telemetry_export_retries",
        "telemetry_rejected_records",
    ):
        assert by_name[name]["attributes"] == {
            "source_kind": "counter",
            "source_temporality": "cumulative_snapshot",
        }
    for name in ("queue_depth", "telemetry_queue_bytes", "telemetry_oldest_queued_age_seconds"):
        assert by_name[name]["attributes"]["source_kind"] == "gauge"
        assert by_name[name]["attributes"]["source_temporality"] == "current_snapshot"
    assert by_name["progress_time_seconds"]["attributes"] == {
        "progress_kind": "telemetry_export",
        "source_kind": "gauge",
        "source_temporality": "current_snapshot",
    }


def test_invalid_typed_event_cannot_poison_valid_metric(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = RecordingTransport()
    configure(monkeypatch, transport)

    telemetry.event("invalid", telemetry.serialization.EventBody({"value": float("nan")}))
    telemetry.counter("valid").add()

    assert transport.accepted.wait(1)
    payload = json.loads(transport.requests[0][1])
    assert [record["name"] for record in payload["records"]] == ["valid"]
    assert telemetry.runtime_status().lost_records == 1


def test_event_integer_boundaries_match_serde_json(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = RecordingTransport()
    configure(monkeypatch, transport)

    telemetry.event(
        "integer.bounds",
        telemetry.serialization.EventBody({"minimum": -(1 << 63), "maximum": (1 << 64) - 1}),
    )

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
    telemetry.event("stuck", telemetry.serialization.EventBody({}))
    assert transport.started.wait(1)

    started = time.monotonic()
    telemetry.shutdown(timeout)  # type: ignore[arg-type]
    elapsed = time.monotonic() - started
    transport.release.set()

    assert elapsed < 0.2
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
        telemetry.event("rejected", telemetry.serialization.EventBody({}))
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
    telemetry.event("ready", telemetry.serialization.EventBody({}))

    assert transports[0].accepted.wait(1)
    assert len(transports) == 1


def test_shutdown_returns_within_budget_when_transport_is_stuck(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = BlockingTransport()
    configure(monkeypatch, transport)
    telemetry.event("stuck", telemetry.serialization.EventBody({}))
    assert transport.started.wait(1)

    started = time.monotonic()
    telemetry.shutdown(0.03)
    elapsed = time.monotonic() - started
    transport.release.set()

    assert elapsed < 0.2
    assert telemetry.runtime_status().configured is False


def test_flush_drains_records_without_disabling_export(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = RecordingTransport()
    configure(monkeypatch, transport, max_batch_records=1)
    counter = telemetry.counter("flushed_record")

    counter.add(1)
    assert telemetry.flush(1.0)
    assert telemetry.runtime_status().configured
    counter.add(2)
    assert telemetry.flush(1.0)

    records = [record for request in transport.requests for record in json.loads(request[1])["records"]]
    assert [record["value"] for record in records] == [1, 2]


def test_flush_timeout_leaves_exporter_running(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = BlockingTransport()
    configure(monkeypatch, transport)
    telemetry.counter("stuck_record").add()
    assert transport.started.wait(1)

    started = time.monotonic()
    assert not telemetry.flush(0.03)
    elapsed = time.monotonic() - started

    assert elapsed < 0.2
    assert telemetry.runtime_status().configured
    transport.release.set()
    assert telemetry.flush(1.0)


def test_shutdown_drains_multiple_queued_batches_after_success(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = BlockingTransport()
    configure(monkeypatch, transport, max_batch_records=1)
    counter = telemetry.counter("terminal_record")
    counter.add(1)
    counter.add(2)
    counter.add(3)
    assert transport.started.wait(1)

    shutdown_thread = threading.Thread(target=telemetry.shutdown, args=(1.0,))
    shutdown_thread.start()
    deadline = time.monotonic() + 1
    while telemetry.runtime_status().configured and time.monotonic() < deadline:
        threading.Event().wait(0.001)
    assert telemetry.runtime_status().configured is False
    transport.release.set()
    shutdown_thread.join(1)

    assert not shutdown_thread.is_alive()
    records = [record for request in transport.requests for record in json.loads(request[1])["records"]]
    assert [record["value"] for record in records] == [1, 2, 3]
