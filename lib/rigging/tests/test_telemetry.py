# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import logging
import threading
from collections import Counter
from datetime import timedelta

import pytest
from rigging import telemetry


@pytest.fixture(autouse=True)
def isolated_telemetry(monkeypatch):
    monkeypatch.setattr(telemetry, "_runtime", None)
    monkeypatch.setattr(telemetry, "_descriptors", {})
    monkeypatch.setattr(telemetry, "_losses", Counter())


def exporter(*, max_queue_records: int = 10, max_queue_bytes: int = 1_000) -> telemetry.HttpExporterConfig:
    return telemetry.HttpExporterConfig(
        endpoint="http://127.0.0.1:4318",
        export_interval=timedelta(seconds=5),
        request_timeout=timedelta(seconds=2),
        shutdown_timeout=timedelta(seconds=1),
        max_queue_records=max_queue_records,
        max_queue_bytes=max_queue_bytes,
    )


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
    assert status.queued_event_bytes <= 1_000
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
    telemetry.configure(service_name="worker", exporter=exporter())
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
