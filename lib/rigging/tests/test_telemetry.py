# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
import threading
from collections import Counter
from dataclasses import replace
from datetime import timedelta

import pytest
from rigging import telemetry
from rigging.auth import StaticTokenProvider

_TOKEN_PROVIDER = StaticTokenProvider("producer-token")


class FakeAgentTransport:
    def __init__(self, failures: int = 0) -> None:
        self.failures = failures
        self.requests: list[telemetry.AgentWriteRequest] = []
        self.accepted = threading.Event()

    def write(self, request: telemetry.AgentWriteRequest) -> telemetry.AgentWriteAck:
        self.requests.append(request)
        if len(self.requests) <= self.failures:
            raise telemetry.RetryableExportError("retry")
        self.accepted.set()
        return telemetry.AgentWriteAck(
            batch_id=request.batch_id,
            status="accepted",
            durability="agent_wal",
        )


class FakeHttpResponse:
    def __init__(
        self,
        status_code: int,
        payload: dict[str, str] | None = None,
        *,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self.payload = payload
        self.headers = headers or {}

    def json(self) -> dict[str, str]:
        if self.payload is None:
            raise ValueError("response is not JSON")
        return self.payload


class RenewableTokenProvider:
    def __init__(self, tokens: list[str | None]) -> None:
        self.tokens = tokens
        self.calls = 0

    def get_token(self) -> str | None:
        index = min(self.calls, len(self.tokens) - 1)
        self.calls += 1
        return self.tokens[index]


@pytest.fixture(autouse=True)
def isolated_telemetry(monkeypatch):
    transport = FakeAgentTransport()
    monkeypatch.setattr(telemetry, "_runtime", None)
    monkeypatch.setattr(telemetry, "_descriptors", {})
    monkeypatch.setattr(telemetry, "_losses", Counter())
    monkeypatch.setattr(telemetry, "_agent_transport", transport)
    yield transport
    telemetry.shutdown()


def exporter(
    *,
    max_queue_records: int = 10,
    max_queue_bytes: int = telemetry.DEFAULT_MAX_QUEUE_BYTES,
    export_interval: timedelta = timedelta(seconds=5),
    shutdown_timeout: timedelta = timedelta(seconds=1),
) -> telemetry.HttpExporterConfig:
    return telemetry.HttpExporterConfig(
        endpoint="http://127.0.0.1:4318",
        token_provider=_TOKEN_PROVIDER,
        export_interval=export_interval,
        request_timeout=timedelta(seconds=2),
        shutdown_timeout=shutdown_timeout,
        max_queue_records=max_queue_records,
        max_queue_bytes=max_queue_bytes,
    )


def _lane_index(delivery_class: telemetry.DeliveryClass) -> int:
    return telemetry._DELIVERY_LANES.index(delivery_class)


def runtime_meter() -> telemetry.Meter:
    return telemetry.meter(
        scope="telemetry.runtime",
        owner="rigging",
        default_cadence=timedelta(seconds=10),
    )


def emissions_counter() -> telemetry.Counter:
    return runtime_meter().counter(
        "emissions",
        description="Telemetry emissions accepted by the process runtime",
        unit="{emission}",
        attributes=(telemetry.AttributeSpec("signal", ("metric", "event", "log", "artifact")),),
        cardinality_limit=4,
        maturity=telemetry.Maturity.STABLE,
    )


def queue_records_gauge() -> telemetry.Gauge:
    return runtime_meter().gauge(
        "queue_records",
        description="Telemetry records waiting for background export",
        unit="{record}",
        cardinality_limit=1,
        maturity=telemetry.Maturity.STABLE,
    )


def export_duration_histogram() -> telemetry.Histogram:
    return runtime_meter().histogram(
        "export_duration",
        description="Telemetry export request duration",
        unit="s",
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
        attributes=(telemetry.AttributeSpec("outcome", ("success", "failure")),),
        cardinality_limit=2,
        maturity=telemetry.Maturity.STABLE,
    )


def hold_lock(lock, acquired: threading.Event, release: threading.Event) -> None:
    with lock:
        acquired.set()
        assert release.wait(timeout=5)


def invoke_capturing_interrupts(call, failures: list[BaseException]) -> None:
    try:
        call()
    except BaseException as error:
        failures.append(error)


def test_handle_declared_before_configuration_starts_emitting_after_configuration():
    requests = emissions_counter()

    requests.add(signal="metric")
    assert telemetry.runtime_status().accepted_emissions == 0

    telemetry.configure(service_name="inference", exporter=exporter())
    requests.add(signal="metric")

    status = telemetry.runtime_status()
    assert status.accepted_emissions == 1
    assert status.metric_series == 1
    assert status.service_instance_id


def test_conflicting_configuration_keeps_the_first_runtime_active():
    queue = queue_records_gauge()
    telemetry.configure(service_name="worker", role="trainer", exporter=exporter())
    first_instance = telemetry.runtime_status().service_instance_id

    telemetry.configure(service_name="other-worker", role="rollout", exporter=exporter())
    queue.set(3)

    status = telemetry.runtime_status()
    assert status.service_instance_id == first_instance
    assert status.accepted_emissions == 1
    assert dict(status.losses)["configuration_conflict"] == 1


def test_repeated_convenience_configuration_reuses_auto_instance_id():
    telemetry.configure(service_name="worker", role="trainer", exporter=exporter())
    first_instance = telemetry.runtime_status().service_instance_id

    telemetry.configure(service_name="worker", role="trainer", exporter=exporter())

    status = telemetry.runtime_status()
    assert status.service_instance_id == first_instance
    assert "configuration_conflict" not in dict(status.losses)


def test_invalid_emission_is_bounded_loss_state_without_synchronous_logging():
    class RaisingHandler(logging.Handler):
        def emit(self, record):
            raise AssertionError("telemetry emission invoked logging")

    handler = RaisingHandler()
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        requests = emissions_counter()
        telemetry.configure(service_name="worker", exporter=exporter())

        requests.add(1, signal="unbounded-value")
        requests.add(float("nan"), signal="metric")
    finally:
        root.removeHandler(handler)

    status = telemetry.runtime_status()
    assert status.accepted_emissions == 0
    assert dict(status.losses)["invalid_emission"] == 2


def test_emission_does_not_catch_process_interrupts():
    class InterruptingNumber:
        def __float__(self):
            raise KeyboardInterrupt

    requests = emissions_counter()
    telemetry.configure(service_name="worker", exporter=exporter())

    with pytest.raises(KeyboardInterrupt):
        requests.add(InterruptingNumber())


def test_event_queue_drops_oldest_within_record_and_byte_bounds():
    telemetry.configure(
        service_name="worker",
        exporter=exporter(max_queue_records=2),
    )

    telemetry.event(
        "telemetry.runtime.gap",
        delivery_class=telemetry.DeliveryClass.DURABLE,
        reason="queue_overflow",
        dropped_records=1,
    )
    telemetry.event(
        "telemetry.runtime.gap",
        delivery_class=telemetry.DeliveryClass.DURABLE,
        reason="queue_overflow",
        dropped_records=2,
    )
    telemetry.event(
        "telemetry.runtime.gap",
        delivery_class=telemetry.DeliveryClass.DURABLE,
        reason="queue_overflow",
        dropped_records=3,
    )

    status = telemetry.runtime_status()
    assert status.queued_events == 2
    assert status.queued_event_bytes <= telemetry.DEFAULT_MAX_QUEUE_BYTES
    assert dict(status.losses)["event_queue_overflow"] == 1


def test_undeclared_event_is_rejected_into_bounded_loss_state():
    telemetry.configure(service_name="worker", exporter=exporter())

    telemetry.event("worker.ad_hoc", arbitrary="value")

    status = telemetry.runtime_status()
    assert status.queued_events == 0
    assert dict(status.losses)["invalid_emission"] == 1


def test_emission_returns_while_internal_coordination_locks_are_held():
    counter = emissions_counter()
    gauge = queue_records_gauge()
    histogram = export_duration_histogram()
    telemetry.configure(
        service_name="worker",
        exporter=exporter(shutdown_timeout=timedelta(milliseconds=50)),
    )
    assert telemetry._runtime is not None

    valid_emissions = (
        lambda: counter.add(signal="metric"),
        lambda: gauge.set(1),
        lambda: histogram.record(0.5, outcome="success"),
        lambda: telemetry.event(
            "telemetry.runtime.gap",
            delivery_class=telemetry.DeliveryClass.DURABLE,
            reason="queue_overflow",
            dropped_records=1,
        ),
    )
    invalid_emissions = (
        lambda: counter.add(-1, signal="metric"),
        lambda: gauge.set(float("nan")),
        lambda: histogram.record(float("nan"), outcome="success"),
        lambda: telemetry.event("undeclared.event"),
    )
    cases = (
        (telemetry._state_lock, valid_emissions),
        (telemetry._runtime.lock, valid_emissions),
        (telemetry._loss_lock, invalid_emissions),
    )

    for lock, emissions in cases:
        acquired = threading.Event()
        release = threading.Event()

        holder = threading.Thread(target=hold_lock, args=(lock, acquired, release))
        holder.start()
        assert acquired.wait(timeout=5)
        try:
            for emit in emissions:
                failures: list[BaseException] = []
                caller = threading.Thread(
                    target=invoke_capturing_interrupts,
                    args=(emit, failures),
                )
                caller.start()
                caller.join(timeout=0.2)
                assert not caller.is_alive(), f"emission blocked on {lock!r}"
                assert failures == []
        finally:
            release.set()
            holder.join(timeout=5)

    for lock in (telemetry._state_lock, telemetry._runtime.lock):
        acquired = threading.Event()
        release = threading.Event()

        holder = threading.Thread(target=hold_lock, args=(lock, acquired, release))
        holder.start()
        assert acquired.wait(timeout=5)
        try:
            caller = threading.Thread(target=telemetry.shutdown)
            caller.start()
            caller.join(timeout=0.2)
            assert not caller.is_alive(), f"shutdown blocked on {lock!r}"
        finally:
            release.set()
            holder.join(timeout=5)


def test_logging_context_nests_and_restores_values():
    with telemetry.logging_context(root_run_uid="run-1"):
        assert telemetry.current_logging_context() == {"root_run_uid": "run-1"}
        with telemetry.logging_context(worker_id="worker-2"):
            assert telemetry.current_logging_context() == {
                "root_run_uid": "run-1",
                "worker_id": "worker-2",
            }
        assert telemetry.current_logging_context() == {"root_run_uid": "run-1"}

    assert telemetry.current_logging_context() == {}


def test_shutdown_turns_existing_handles_back_into_no_ops():
    requests = emissions_counter()
    telemetry.configure(service_name="worker", exporter=exporter())
    requests.add(signal="metric")

    telemetry.shutdown()
    requests.add(signal="metric")

    status = telemetry.runtime_status()
    assert status.stopped
    assert status.accepted_emissions == 1


def test_retry_reuses_exact_batch_id_and_bytes(isolated_telemetry, monkeypatch):
    transport = FakeAgentTransport(failures=1)
    monkeypatch.setattr(telemetry, "_agent_transport", transport)
    requests = emissions_counter()
    telemetry.configure(service_name="worker", exporter=exporter())
    requests.add(signal="metric")
    assert telemetry._runtime is not None

    telemetry._runtime._export_once()
    telemetry._runtime._export_once()

    assert len(transport.requests) == 2
    first, second = transport.requests
    assert first.batch_id
    assert first.batch_id == second.batch_id
    assert first.body == second.body
    batch = json.loads(first.body)
    assert batch["batch_id"] == first.batch_id
    assert len(batch["records"]) == 1
    record = batch["records"][0]
    assert record["delivery_class"] == "buffered"
    assert "delivery_class" not in record["metric"]
    assert dict(telemetry.runtime_status().losses)["export_retry"] == 1


def test_renewed_credential_and_unverifiable_acks_retain_exact_pending_batch(monkeypatch):
    provider = RenewableTokenProvider([None, "renewed-token"])
    responses = [
        FakeHttpResponse(200, {}),
        FakeHttpResponse(
            201,
            {
                "batch_id": "wrong-batch",
                "status": "accepted",
                "durability": "agent_wal",
            },
        ),
    ]
    posts: list[dict] = []

    def post(url, **kwargs):
        posts.append({"url": url, **kwargs})
        if responses:
            return responses.pop(0)
        return FakeHttpResponse(
            201,
            {
                "batch_id": kwargs["headers"]["Idempotency-Key"],
                "status": "accepted",
                "durability": "agent_wal",
            },
        )

    monkeypatch.setattr(telemetry.requests, "post", post)
    monkeypatch.setattr(telemetry, "_agent_transport", telemetry.RequestsAgentTransport())
    telemetry.configure(
        service_name="worker",
        exporter=replace(exporter(), token_provider=provider),
    )
    telemetry.event(
        "telemetry.runtime.gap",
        delivery_class=telemetry.DeliveryClass.DURABLE,
        reason="queue_overflow",
        dropped_records=1,
    )
    assert telemetry._runtime is not None

    telemetry._runtime._export_once()
    pending = telemetry._runtime.pending
    assert pending is not None
    expected_id = pending.batch_id
    expected_body = pending.body

    telemetry._runtime._export_once()
    assert telemetry._runtime.pending is pending
    telemetry._runtime._export_once()
    assert telemetry._runtime.pending is pending
    telemetry._runtime._export_once()

    assert telemetry._runtime.pending is None
    assert provider.calls == 4
    assert len(posts) == 3
    assert {call["headers"]["Idempotency-Key"] for call in posts} == {expected_id}
    assert {call["data"] for call in posts} == {expected_body}
    assert dict(telemetry.runtime_status().losses)["export_retry"] == 3


@pytest.mark.parametrize("status_code", [401, 429, 502, 503, 504])
def test_retryable_http_statuses_retain_batch(status_code, monkeypatch):
    monkeypatch.setattr(
        telemetry.requests,
        "post",
        lambda *args, **kwargs: FakeHttpResponse(status_code),
    )
    request = telemetry.AgentWriteRequest(
        endpoint="http://127.0.0.1:4318",
        batch_id="batch",
        body=b"body",
        content_type="application/json",
        token_provider=_TOKEN_PROVIDER,
        timeout=timedelta(seconds=1),
    )

    with pytest.raises(telemetry.RetryableExportError):
        telemetry.RequestsAgentTransport().write(request)


@pytest.mark.parametrize("status_code", [400, 403, 404, 409, 413, 415, 422, 500])
def test_terminal_http_statuses_settle_with_loss(status_code, monkeypatch):
    monkeypatch.setattr(
        telemetry.requests,
        "post",
        lambda *args, **kwargs: FakeHttpResponse(status_code),
    )
    monkeypatch.setattr(telemetry, "_agent_transport", telemetry.RequestsAgentTransport())
    telemetry.configure(service_name="worker", exporter=exporter())
    telemetry.event(
        "telemetry.runtime.gap",
        delivery_class=telemetry.DeliveryClass.DURABLE,
        reason="queue_overflow",
        dropped_records=1,
    )
    assert telemetry._runtime is not None

    telemetry._runtime._export_once()

    assert telemetry._runtime.pending is None
    assert telemetry.runtime_status().queued_events == 0
    assert dict(telemetry.runtime_status().losses)["export_terminal"] == 1


def test_retry_after_defers_the_exact_pending_batch(monkeypatch):
    now = [100.0]
    posts: list[bytes] = []

    def post(url, **kwargs):
        posts.append(kwargs["data"])
        if len(posts) == 1:
            return FakeHttpResponse(429, headers={"Retry-After": "7"})
        return FakeHttpResponse(
            200,
            {
                "batch_id": kwargs["headers"]["Idempotency-Key"],
                "status": "duplicate",
                "durability": "agent_wal",
            },
        )

    monkeypatch.setattr(telemetry.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(telemetry.requests, "post", post)
    monkeypatch.setattr(telemetry, "_agent_transport", telemetry.RequestsAgentTransport())
    counter = emissions_counter()
    telemetry.configure(service_name="worker", exporter=exporter())
    counter.add(signal="metric")
    assert telemetry._runtime is not None

    telemetry._runtime._export_once()
    pending = telemetry._runtime.pending
    assert pending is not None
    assert telemetry._runtime._retry_not_before == 107.0

    now[0] = 106.0
    telemetry._runtime._export_once()
    assert len(posts) == 1
    assert telemetry._runtime.pending is pending

    now[0] = 107.0
    telemetry._runtime._export_once()
    assert telemetry._runtime.pending is None
    assert posts == [pending.body, pending.body]


def test_delivery_classes_form_homogeneous_batches(isolated_telemetry):
    counter = emissions_counter()
    gauge = queue_records_gauge()
    telemetry.configure(service_name="worker", exporter=exporter())
    counter.add(signal="metric")
    gauge.set(2)
    telemetry.event(
        "telemetry.runtime.gap",
        delivery_class=telemetry.DeliveryClass.DURABLE,
        reason="queue_overflow",
        dropped_records=1,
    )
    assert telemetry._runtime is not None

    telemetry._runtime._export_once()
    telemetry._runtime._export_once()
    telemetry._runtime._export_once()

    batches = [json.loads(request.body) for request in isolated_telemetry.requests]
    assert {batch["records"][0]["delivery_class"] for batch in batches} == {
        "buffered",
        "coalescing",
        "durable",
    }
    for batch in batches:
        assert len({record["delivery_class"] for record in batch["records"]}) == 1


def test_in_flight_ack_never_removes_newer_overflow_survivors(monkeypatch):
    class BlockingTransport(FakeAgentTransport):
        def __init__(self) -> None:
            super().__init__()
            self.entered = threading.Event()
            self.release = threading.Event()

        def write(self, request: telemetry.AgentWriteRequest) -> telemetry.AgentWriteAck:
            self.requests.append(request)
            self.entered.set()
            assert self.release.wait(timeout=5)
            self.accepted.set()
            return telemetry.AgentWriteAck(request.batch_id, "accepted", "agent_wal")

    transport = BlockingTransport()
    monkeypatch.setattr(telemetry, "_agent_transport", transport)
    telemetry.configure(
        service_name="worker",
        exporter=exporter(max_queue_records=2),
    )
    telemetry.event(
        "telemetry.runtime.gap",
        delivery_class=telemetry.DeliveryClass.DURABLE,
        reason="queue_overflow",
        dropped_records=1,
    )
    assert telemetry._runtime is not None
    exporter_thread = threading.Thread(target=telemetry._runtime._export_once)
    exporter_thread.start()
    assert transport.entered.wait(timeout=5)

    for dropped_records in (2, 3, 4):
        telemetry.event(
            "telemetry.runtime.gap",
            delivery_class=telemetry.DeliveryClass.DURABLE,
            reason="queue_overflow",
            dropped_records=dropped_records,
        )
    transport.release.set()
    assert transport.accepted.wait(timeout=5)
    exporter_thread.join(timeout=5)
    with telemetry._runtime.lock:
        survivors = [dict(event.attributes)["dropped_records"] for event in telemetry._runtime.events]
    assert survivors == ["4"]


def test_shutdown_returns_within_budget_when_transport_never_returns(monkeypatch):
    class StuckTransport:
        def __init__(self) -> None:
            self.entered = threading.Event()
            self.release = threading.Event()

        def write(self, request: telemetry.AgentWriteRequest) -> telemetry.AgentWriteAck:
            self.entered.set()
            assert self.release.wait(timeout=5)
            return telemetry.AgentWriteAck(request.batch_id, "accepted", "agent_wal")

    transport = StuckTransport()
    monkeypatch.setattr(telemetry, "_agent_transport", transport)
    telemetry.configure(
        service_name="worker",
        exporter=exporter(
            export_interval=timedelta(milliseconds=1),
            shutdown_timeout=timedelta(milliseconds=50),
        ),
    )
    telemetry.event(
        "telemetry.runtime.gap",
        delivery_class=telemetry.DeliveryClass.DURABLE,
        reason="queue_overflow",
        dropped_records=1,
    )
    assert transport.entered.wait(timeout=5)

    caller = threading.Thread(target=telemetry.shutdown)
    caller.start()
    caller.join(timeout=0.2)
    try:
        assert not caller.is_alive()
        assert telemetry.runtime_status().stopped
    finally:
        transport.release.set()
        caller.join(timeout=5)


def test_batch_encoding_happens_outside_aggregation_lock_and_reserves_capacity(monkeypatch):
    counter = emissions_counter()
    telemetry.configure(service_name="worker", exporter=exporter(max_queue_records=2))
    counter.add(signal="metric")
    assert telemetry._runtime is not None

    entered = threading.Event()
    release = threading.Event()
    original_stable_json = telemetry._stable_json

    def blocking_stable_json(value):
        if isinstance(value, dict) and "records" in value and not release.is_set():
            entered.set()
            assert release.wait(timeout=5)
        return original_stable_json(value)

    monkeypatch.setattr(telemetry, "_stable_json", blocking_stable_json)
    exporter_thread = threading.Thread(target=telemetry._runtime._export_once)
    exporter_thread.start()
    assert entered.wait(timeout=5)

    callers = [
        threading.Thread(target=lambda: counter.add(signal="metric")),
        threading.Thread(
            target=lambda: telemetry.event(
                "telemetry.runtime.gap",
                delivery_class=telemetry.DeliveryClass.DURABLE,
                reason="queue_overflow",
                dropped_records=1,
            )
        ),
    ]
    for caller in callers:
        caller.start()
        caller.join(timeout=0.2)
        assert not caller.is_alive()
    assert dict(telemetry.runtime_status().losses)["event_queue_overflow"] == 1

    release.set()
    exporter_thread.join(timeout=5)
    assert not exporter_thread.is_alive()


def test_mixed_lane_building_retry_and_success_stay_within_combined_cap(monkeypatch):
    transport = FakeAgentTransport(failures=1)
    monkeypatch.setattr(telemetry, "_agent_transport", transport)
    counter = emissions_counter()
    telemetry.configure(service_name="worker", exporter=exporter(max_queue_records=2))
    counter.add(signal="metric")
    telemetry.event(
        "telemetry.runtime.gap",
        delivery_class=telemetry.DeliveryClass.DURABLE,
        reason="queue_overflow",
        dropped_records=1,
    )
    assert telemetry._runtime is not None
    runtime = telemetry._runtime
    runtime._next_lane = _lane_index(telemetry.DeliveryClass.BUFFERED)

    entered = threading.Event()
    release = threading.Event()
    original_stable_json = telemetry._stable_json

    def blocking_stable_json(value):
        if isinstance(value, dict) and "records" in value and not release.is_set():
            entered.set()
            assert release.wait(timeout=5)
        return original_stable_json(value)

    monkeypatch.setattr(telemetry, "_stable_json", blocking_stable_json)
    pending_results: list[telemetry._PendingBatch | None] = []
    builder = threading.Thread(target=lambda: pending_results.append(runtime._pending_batch()))
    builder.start()
    assert entered.wait(timeout=5)

    status = telemetry.runtime_status()
    assert status.in_flight_records == 1
    assert status.in_flight_bytes == runtime.building.reserved_bytes
    assert status.queued_events == 1
    assert runtime._accounted_records() == 2
    assert runtime._accounted_bytes() <= runtime.exporter.max_queue_bytes

    telemetry.event(
        "telemetry.runtime.gap",
        delivery_class=telemetry.DeliveryClass.DURABLE,
        reason="queue_overflow",
        dropped_records=2,
    )
    assert runtime._accounted_records() == 2
    assert runtime._accounted_bytes() <= runtime.exporter.max_queue_bytes
    with runtime.lock:
        assert [dict(event.attributes)["dropped_records"] for event in runtime.events] == ["2"]

    release.set()
    builder.join(timeout=5)
    assert not builder.is_alive()
    assert pending_results[0] is runtime.pending
    assert runtime._accounted_records() == 2
    assert runtime._accounted_bytes() <= runtime.exporter.max_queue_bytes

    runtime._export_once()
    pending = runtime.pending
    assert pending is not None
    assert runtime._accounted_records() == 2
    runtime._export_once()
    assert runtime.pending is None
    runtime._export_once()

    assert len(transport.requests) == 3
    assert transport.requests[0].batch_id == transport.requests[1].batch_id
    assert transport.requests[0].body == transport.requests[1].body
    assert json.loads(transport.requests[0].body)["records"][0]["signal"] == "metric"
    durable_record = json.loads(transport.requests[2].body)["records"][0]
    assert durable_record["event"]["attributes"]["dropped_records"] == "2"
    assert dict(telemetry.runtime_status().losses) == {
        "event_queue_overflow": 1,
        "export_retry": 1,
    }


@pytest.mark.parametrize("replacement_extra_byte", [0, 1])
def test_mixed_lane_exact_byte_cap_survives_build_restoration_and_retry(
    replacement_extra_byte,
    monkeypatch,
):
    event_name = "telemetry.runtime.gap"
    durable_descriptor = telemetry._EVENT_CATALOG[event_name]
    buffered_descriptor = replace(durable_descriptor, delivery_class=telemetry.DeliveryClass.BUFFERED)
    resource = telemetry.Resource(service_name="worker", service_instance_id="instance")

    telemetry.configure(resource=resource, exporter=exporter(max_queue_records=2))
    telemetry.event(
        event_name,
        delivery_class=telemetry.DeliveryClass.DURABLE,
        body="d" * 16,
        reason="queue_overflow",
        dropped_records=1,
    )
    monkeypatch.setitem(telemetry._EVENT_CATALOG, event_name, buffered_descriptor)
    telemetry.event(
        event_name,
        delivery_class=telemetry.DeliveryClass.BUFFERED,
        body="b" * 16,
        reason="queue_overflow",
        dropped_records=2,
    )
    assert telemetry._runtime is not None
    exact_mixed_lane_bytes = telemetry._runtime._queued_event_bytes()
    telemetry.shutdown()

    transport = FakeAgentTransport(failures=1)
    monkeypatch.setattr(telemetry, "_runtime", None)
    monkeypatch.setattr(telemetry, "_losses", Counter())
    monkeypatch.setattr(telemetry, "_agent_transport", transport)
    monkeypatch.setitem(telemetry._EVENT_CATALOG, event_name, durable_descriptor)
    telemetry.configure(
        resource=resource,
        exporter=exporter(max_queue_records=2, max_queue_bytes=exact_mixed_lane_bytes),
    )
    telemetry.event(
        event_name,
        delivery_class=telemetry.DeliveryClass.DURABLE,
        body="d" * 16,
        reason="queue_overflow",
        dropped_records=1,
    )
    monkeypatch.setitem(telemetry._EVENT_CATALOG, event_name, buffered_descriptor)
    telemetry.event(
        event_name,
        delivery_class=telemetry.DeliveryClass.BUFFERED,
        body="b" * 16,
        reason="queue_overflow",
        dropped_records=2,
    )
    assert telemetry._runtime is not None
    runtime = telemetry._runtime
    assert runtime._queued_event_bytes() == exact_mixed_lane_bytes
    assert runtime._accounted_bytes() == exact_mixed_lane_bytes

    entered = threading.Event()
    release = threading.Event()
    original_stable_json = telemetry._stable_json
    build_errors: list[Exception] = []

    def fail_after_reservation(value):
        if isinstance(value, dict) and "records" in value:
            entered.set()
            assert release.wait(timeout=5)
            raise ValueError("injected encoding failure")
        return original_stable_json(value)

    monkeypatch.setattr(telemetry, "_stable_json", fail_after_reservation)

    def build_pending() -> None:
        try:
            runtime._pending_batch()
        except Exception as error:
            build_errors.append(error)

    builder = threading.Thread(target=build_pending)
    builder.start()
    assert entered.wait(timeout=5)
    assert runtime._accounted_bytes() == exact_mixed_lane_bytes
    assert telemetry.runtime_status().in_flight_bytes == runtime.building.reserved_bytes

    telemetry.event(
        event_name,
        delivery_class=telemetry.DeliveryClass.BUFFERED,
        body="r" * (16 + replacement_extra_byte),
        reason="queue_overflow",
        dropped_records=3,
    )
    assert runtime._accounted_bytes() <= exact_mixed_lane_bytes

    release.set()
    builder.join(timeout=5)
    assert not builder.is_alive()
    assert [str(error) for error in build_errors] == ["injected encoding failure"]
    assert runtime.building is None
    assert runtime.pending is None
    if replacement_extra_byte:
        assert len(runtime.events) == 1
        telemetry.event(
            event_name,
            delivery_class=telemetry.DeliveryClass.BUFFERED,
            body="b" * 16,
            reason="queue_overflow",
            dropped_records=2,
        )
    assert runtime._queued_event_bytes() == exact_mixed_lane_bytes
    assert runtime._accounted_bytes() == exact_mixed_lane_bytes

    monkeypatch.setattr(telemetry, "_stable_json", original_stable_json)
    runtime._export_once()
    pending = runtime.pending
    assert pending is not None
    assert runtime._accounted_bytes() == exact_mixed_lane_bytes
    runtime._export_once()
    assert runtime.pending is None
    runtime._export_once()

    assert len(transport.requests) == 3
    assert transport.requests[0].batch_id == transport.requests[1].batch_id
    assert transport.requests[0].body == transport.requests[1].body
    buffered_record = json.loads(transport.requests[2].body)["records"][0]
    expected_dropped_records = "2" if replacement_extra_byte else "3"
    assert buffered_record["event"]["attributes"]["dropped_records"] == expected_dropped_records
    expected_overflows = 2 if replacement_extra_byte else 1
    assert dict(telemetry.runtime_status().losses) == {
        "event_queue_overflow": expected_overflows,
        "export_retry": 1,
    }


def test_failed_batch_build_restores_selected_events_without_crossing_caps(monkeypatch):
    telemetry.configure(service_name="worker", exporter=exporter(max_queue_records=2))
    for dropped_records in (1, 2):
        telemetry.event(
            "telemetry.runtime.gap",
            delivery_class=telemetry.DeliveryClass.DURABLE,
            reason="queue_overflow",
            dropped_records=dropped_records,
        )
    assert telemetry._runtime is not None
    runtime = telemetry._runtime
    original_stable_json = telemetry._stable_json

    def fail_batch_encoding(value):
        if isinstance(value, dict) and "records" in value:
            raise ValueError("injected encoding failure")
        return original_stable_json(value)

    monkeypatch.setattr(telemetry, "_stable_json", fail_batch_encoding)
    with pytest.raises(ValueError, match="injected encoding failure"):
        runtime._pending_batch()

    assert runtime.building is None
    assert runtime.pending is None
    assert runtime._accounted_records() == 2
    assert runtime._accounted_bytes() <= runtime.exporter.max_queue_bytes
    assert [dict(event.attributes)["dropped_records"] for event in runtime.events] == ["1", "2"]

    monkeypatch.setattr(telemetry, "_stable_json", original_stable_json)
    runtime._export_once()
    assert runtime.pending is None


def test_event_exact_byte_boundary_ships_and_oversized_event_does_not_pin(monkeypatch, isolated_telemetry):
    resource = telemetry.Resource(service_name="worker", service_instance_id="instance")
    large_body = "x" * 2_000
    large_event = {
        "event_name": "telemetry.runtime.gap",
        "delivery_class": telemetry.DeliveryClass.DURABLE,
        "reason": "queue_overflow",
        "dropped_records": 1,
        "body": large_body,
    }
    telemetry.configure(resource=resource, exporter=exporter())
    telemetry.event(**large_event)
    assert telemetry._runtime is not None
    pending = telemetry._runtime._pending_batch()
    assert pending is not None
    exact_bytes = len(pending.body)
    telemetry._runtime._settle_pending(pending)
    telemetry.shutdown()

    monkeypatch.setattr(telemetry, "_runtime", None)
    telemetry.configure(resource=resource, exporter=exporter(max_queue_records=1, max_queue_bytes=exact_bytes))
    telemetry.event(**large_event)
    assert telemetry._runtime is not None
    telemetry._runtime._export_once()
    assert len(isolated_telemetry.requests[-1].body) == exact_bytes

    telemetry.shutdown()
    monkeypatch.setattr(telemetry, "_runtime", None)
    telemetry.configure(resource=resource, exporter=exporter(max_queue_records=1, max_queue_bytes=exact_bytes - 1))
    telemetry.event(**large_event)
    telemetry.event(
        "telemetry.runtime.gap",
        delivery_class=telemetry.DeliveryClass.DURABLE,
        reason="queue_overflow",
        dropped_records=2,
    )
    assert telemetry._runtime is not None
    assert telemetry.runtime_status().queued_events == 1
    telemetry._runtime._export_once()

    record = json.loads(isolated_telemetry.requests[-1].body)["records"][0]
    assert "body" not in record["event"]
    assert record["event"]["attributes"]["dropped_records"] == "2"
    assert dict(telemetry.runtime_status().losses)["export_terminal"] == 1


def test_same_lane_event_traffic_cannot_starve_metric_cursor(monkeypatch, isolated_telemetry):
    event_name = "telemetry.runtime.gap"
    descriptor = telemetry._EVENT_CATALOG[event_name]
    monkeypatch.setitem(
        telemetry._EVENT_CATALOG,
        event_name,
        replace(descriptor, delivery_class=telemetry.DeliveryClass.BUFFERED),
    )
    counter = emissions_counter()
    telemetry.configure(service_name="worker", exporter=exporter(max_queue_records=1))
    counter.add(signal="event")
    counter.add(signal="metric")
    assert telemetry._runtime is not None

    signals: list[str] = []
    for dropped_records in range(4):
        telemetry.event(
            event_name,
            delivery_class=telemetry.DeliveryClass.BUFFERED,
            reason="queue_overflow",
            dropped_records=dropped_records,
        )
        telemetry._runtime._export_once()
        signals.append(json.loads(isolated_telemetry.requests[-1].body)["records"][0]["signal"])

    assert signals == ["event", "metric", "event", "metric"]
    metric_records = [
        json.loads(request.body)["records"][0]
        for request in isolated_telemetry.requests
        if json.loads(request.body)["records"][0]["signal"] == "metric"
    ]
    assert [record["metric"]["attributes"]["signal"] for record in metric_records] == ["event", "metric"]


def test_successful_ack_waits_for_and_then_retires_exact_pending_batch(monkeypatch):
    transport = FakeAgentTransport()
    monkeypatch.setattr(telemetry, "_agent_transport", transport)
    counter = emissions_counter()
    telemetry.configure(service_name="worker", exporter=exporter())
    counter.add(signal="metric")
    assert telemetry._runtime is not None
    pending = telemetry._runtime._pending_batch()
    assert pending is not None

    acquired = threading.Event()
    release = threading.Event()
    holder = threading.Thread(target=hold_lock, args=(telemetry._runtime.lock, acquired, release))
    holder.start()
    assert acquired.wait(timeout=5)
    exporter_thread = threading.Thread(target=telemetry._runtime._export_once)
    exporter_thread.start()
    assert transport.accepted.wait(timeout=5)
    assert exporter_thread.is_alive()

    release.set()
    holder.join(timeout=5)
    exporter_thread.join(timeout=5)
    assert not exporter_thread.is_alive()
    assert telemetry._runtime.pending is None
    assert [request.batch_id for request in transport.requests] == [pending.batch_id]


def test_metric_snapshots_chunk_under_exact_record_and_body_caps(isolated_telemetry):
    counter = emissions_counter()
    histogram = export_duration_histogram()
    config = exporter(max_queue_records=1, max_queue_bytes=6_500)
    telemetry.configure(
        resource=telemetry.Resource(
            service_name="s" * 4_000,
            service_instance_id="instance",
        ),
        exporter=config,
    )
    counter.add(signal="metric")
    histogram.record(0.5, outcome="success")
    assert telemetry._runtime is not None

    telemetry._runtime._export_once()
    telemetry._runtime._export_once()

    assert len(isolated_telemetry.requests) == 2
    batches = [json.loads(request.body) for request in isolated_telemetry.requests]
    assert all(len(batch["records"]) == 1 for batch in batches)
    assert all(len(request.body) <= config.max_queue_bytes for request in isolated_telemetry.requests)
    assert {batch["records"][0]["metric"]["name"] for batch in batches} == {
        "emissions",
        "export_duration",
    }


def test_individually_oversized_metric_is_suppressed_instead_of_pinning_cursor(isolated_telemetry):
    counter = emissions_counter()
    histogram = export_duration_histogram()
    telemetry.configure(service_name="worker", exporter=exporter(max_queue_bytes=200))
    counter.add(signal="metric")
    histogram.record(0.5, outcome="success")
    assert telemetry._runtime is not None

    telemetry._runtime._export_once()
    telemetry._runtime._export_once()

    assert isolated_telemetry.requests == []
    assert telemetry._runtime.pending is None
    assert telemetry._runtime.building is None
    assert len(telemetry._runtime._oversized_metrics) == 2
    assert dict(telemetry.runtime_status().losses)["export_terminal"] == 2


def test_status_does_not_wait_for_aggregation_lock():
    telemetry.configure(service_name="worker", exporter=exporter())
    assert telemetry._runtime is not None
    acquired = threading.Event()
    release = threading.Event()
    holder = threading.Thread(target=hold_lock, args=(telemetry._runtime.lock, acquired, release))
    holder.start()
    assert acquired.wait(timeout=5)
    try:
        result: list[telemetry.RuntimeStatus] = []
        caller = threading.Thread(target=lambda: result.append(telemetry.runtime_status()))
        caller.start()
        caller.join(timeout=0.2)
        assert not caller.is_alive()
        assert result[0].configured
    finally:
        release.set()
        holder.join(timeout=5)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX at-fork hooks")
@pytest.mark.filterwarnings("ignore:This process.*multi-threaded.*fork.*:DeprecationWarning")
def test_fork_resets_held_module_lock_records_loss_and_allows_child_configuration():
    telemetry.configure(service_name="parent", exporter=exporter())
    acquired = threading.Event()
    release = threading.Event()
    holder = threading.Thread(target=hold_lock, args=(telemetry._state_lock, acquired, release))
    holder.start()
    assert acquired.wait(timeout=5)

    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(read_fd)
        try:
            before = telemetry.runtime_status()
            telemetry.configure(service_name="child", exporter=exporter())
            after = telemetry.runtime_status()
            payload = json.dumps(
                {
                    "before_configured": before.configured,
                    "before_losses": dict(before.losses),
                    "after_configured": after.configured,
                    "after_service_instance_id": after.service_instance_id,
                }
            ).encode()
            os.write(write_fd, payload)
            os._exit(0)
        except BaseException:
            os._exit(70)

    os.close(write_fd)
    child_output: list[bytes] = []
    child_status: list[tuple[int, int]] = []

    def collect_child() -> None:
        child_output.append(os.read(read_fd, 64 * 1024))
        child_status.append(os.waitpid(pid, 0))

    collector = threading.Thread(target=collect_child)
    collector.start()
    collector.join(timeout=2)
    try:
        assert not collector.is_alive(), "child blocked on a lock inherited across fork"
        assert os.waitstatus_to_exitcode(child_status[0][1]) == 0
        payload = json.loads(child_output[0])
        assert not payload["before_configured"]
        assert payload["before_losses"]["forked_process"] == 1
        assert payload["after_configured"]
        assert payload["after_service_instance_id"]
    finally:
        if collector.is_alive():
            os.kill(pid, 9)
            collector.join(timeout=5)
        os.close(read_fd)
        release.set()
        holder.join(timeout=5)


def test_pid_mismatch_disables_inherited_runtime_and_child_reconfigure_replaces_it():
    counter = emissions_counter()
    telemetry.configure(service_name="worker", exporter=exporter())
    assert telemetry._runtime is not None
    inherited = telemetry._runtime
    inherited.pid = -1
    inherited._stop_event.set()

    counter.add(signal="metric")
    inherited.shutdown()
    assert inherited.accepted_emissions == 0
    assert dict(telemetry.runtime_status().losses)["forked_process"] == 1

    telemetry.configure(service_name="worker", exporter=exporter())
    assert telemetry._runtime is not inherited
    assert telemetry._runtime is not None
    assert telemetry._runtime.pid != -1
    assert telemetry._runtime.resource.service_instance_id != inherited.resource.service_instance_id


def test_shutdown_budget_over_five_seconds_is_rejected():
    telemetry.configure(
        service_name="worker",
        exporter=exporter(shutdown_timeout=timedelta(seconds=6)),
    )
    status = telemetry.runtime_status()
    assert not status.configured
    assert dict(status.losses)["invalid_configuration"] == 1


def test_queue_byte_cap_below_empty_batch_envelope_keeps_runtime_inert():
    counter = emissions_counter()

    telemetry.configure(
        service_name="worker",
        exporter=exporter(max_queue_bytes=1),
    )
    counter.add(signal="metric")

    status = telemetry.runtime_status()
    assert not status.configured
    assert status.accepted_emissions == 0
    assert dict(status.losses)["invalid_configuration"] == 1
