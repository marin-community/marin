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


def test_handle_declared_before_configuration_starts_emitting_after_configuration():
    requests = telemetry.meter(
        scope="test.inference",
        owner="test",
        default_cadence=timedelta(seconds=15),
    ).counter(
        "requests",
        description="Completed requests",
        unit="{request}",
    )

    requests.add()
    assert telemetry.runtime_status().accepted_emissions == 0

    telemetry.configure(service_name="inference", exporter=exporter())
    requests.add()

    status = telemetry.runtime_status()
    assert status.accepted_emissions == 1
    assert status.metric_series == 1
    assert status.service_instance_id


def test_conflicting_configuration_keeps_the_first_runtime_active():
    meter = telemetry.meter(
        scope="test.worker",
        owner="test",
        default_cadence=timedelta(seconds=15),
    )
    queue = meter.gauge("queue_depth", description="Queue depth", unit="{request}")
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
        outcome = telemetry.AttributeSpec("outcome", ("success", "failure"))
        requests = telemetry.meter(
            scope="test.inference",
            owner="test",
            default_cadence=timedelta(seconds=15),
        ).counter(
            "requests",
            description="Completed requests",
            unit="{request}",
            attributes=(outcome,),
        )
        telemetry.configure(service_name="worker", exporter=exporter())

        requests.add(1, outcome="unbounded-value")
        requests.add(float("nan"), outcome="success")
    finally:
        root.removeHandler(handler)

    status = telemetry.runtime_status()
    assert status.accepted_emissions == 0
    assert dict(status.losses)["invalid_emission"] == 2


def test_emission_does_not_catch_process_interrupts():
    class InterruptingNumber:
        def __float__(self):
            raise KeyboardInterrupt

    requests = telemetry.meter(
        scope="test.inference",
        owner="test",
        default_cadence=timedelta(seconds=15),
    ).counter("requests", description="Completed requests", unit="{request}")
    telemetry.configure(service_name="worker", exporter=exporter())

    with pytest.raises(KeyboardInterrupt):
        requests.add(InterruptingNumber())


def test_event_queue_drops_oldest_within_record_and_byte_bounds():
    telemetry.configure(
        service_name="worker",
        exporter=exporter(max_queue_records=2),
    )

    telemetry.event("telemetry.runtime.gap", reason="queue_overflow", dropped_records=1)
    telemetry.event("telemetry.runtime.gap", reason="queue_overflow", dropped_records=2)
    telemetry.event("telemetry.runtime.gap", reason="queue_overflow", dropped_records=3)

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
    meter = telemetry.meter(
        scope="test.contention",
        owner="test",
        default_cadence=timedelta(seconds=15),
    )
    counter = meter.counter("requests", description="Requests", unit="{request}")
    gauge = meter.gauge("queue_depth", description="Queue depth", unit="{request}")
    histogram = meter.histogram(
        "request_duration",
        description="Request latency",
        unit="s",
        buckets=(0.1, 1.0),
    )
    telemetry.configure(service_name="worker", exporter=exporter())
    assert telemetry._runtime is not None

    valid_emissions = (
        lambda: counter.add(),
        lambda: gauge.set(1),
        lambda: histogram.record(0.5),
        lambda: telemetry.event(
            "telemetry.runtime.gap",
            reason="queue_overflow",
            dropped_records=1,
        ),
    )
    invalid_emissions = (
        lambda: counter.add(-1),
        lambda: gauge.set(float("nan")),
        lambda: histogram.record(float("nan")),
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

        def hold_lock():
            with lock:
                acquired.set()
                assert release.wait(timeout=5)

        holder = threading.Thread(target=hold_lock)
        holder.start()
        assert acquired.wait(timeout=5)
        try:
            for emit in emissions:
                failures = []

                def run_emission():
                    try:
                        emit()
                    except BaseException as error:
                        failures.append(error)

                caller = threading.Thread(target=run_emission)
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

        def hold_shutdown_lock():
            with lock:
                acquired.set()
                assert release.wait(timeout=5)

        holder = threading.Thread(target=hold_shutdown_lock)
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
    requests = telemetry.meter(
        scope="test.inference",
        owner="test",
        default_cadence=timedelta(seconds=15),
    ).counter("requests", description="Completed requests", unit="{request}")
    telemetry.configure(service_name="worker", exporter=exporter())
    requests.add()

    telemetry.shutdown()
    requests.add()

    status = telemetry.runtime_status()
    assert status.stopped
    assert status.accepted_emissions == 1
