# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import pytest
from rigging import probes, telemetry
from rigging import timing as rigging_timing
from rigging.testing import RecordingTelemetryTransport


@pytest.fixture(autouse=True)
def reset_telemetry() -> Iterator[None]:
    telemetry.shutdown(0.01)
    yield
    telemetry.shutdown(0.1)


@dataclass(frozen=True)
class CommandRequest:
    argv: tuple[str, ...]
    timeout: float
    max_output_bytes: int
    thread: threading.Thread


class RecordingRunner:
    def __init__(self, results: list[probes.CommandResult]) -> None:
        self.results = iter(results)
        self.requests: list[CommandRequest] = []

    def run(self, argv: tuple[str, ...], *, timeout: float, max_output_bytes: int) -> probes.CommandResult:
        self.requests.append(CommandRequest(argv, timeout, max_output_bytes, threading.current_thread()))
        return next(self.results)

    def cancel(self, timeout: float) -> None:
        del timeout


class RecordingSink:
    def __init__(self) -> None:
        self.samples: list[probes.ProbeSample] = []
        self._condition = threading.Condition()

    def emit(self, sample: probes.ProbeSample) -> None:
        with self._condition:
            self.samples.append(sample)
            self._condition.notify_all()

    def wait_for(self, count: int, timeout: float = 1.0) -> list[probes.ProbeSample]:
        with self._condition:
            assert self._condition.wait_for(lambda: len(self.samples) >= count, timeout)
            return list(self.samples)


class FakeClock:
    def __init__(self, *, monotonic: float = 100.0, wall_time: float = 1_000.0, automatic_waits: int = 0) -> None:
        self.monotonic_value = monotonic
        self.wall_time_value = wall_time
        self.automatic_waits = automatic_waits
        self.waits: list[float] = []

    def monotonic(self) -> float:
        return self.monotonic_value

    def time(self) -> float:
        return self.wall_time_value

    def wait(self, event: threading.Event, timeout: float) -> bool:
        self.waits.append(timeout)
        if self.automatic_waits:
            self.automatic_waits -= 1
            self.monotonic_value += timeout
            self.wall_time_value += timeout
            return False
        return event.wait()

    def advance(self, seconds: float) -> None:
        self.monotonic_value += seconds
        self.wall_time_value += seconds


class AdvancingRunner(RecordingRunner):
    def __init__(self, results: list[probes.CommandResult], clock: FakeClock, first_advance: float) -> None:
        super().__init__(results)
        self.clock = clock
        self.first_advance = first_advance

    def run(self, argv: tuple[str, ...], *, timeout: float, max_output_bytes: int) -> probes.CommandResult:
        result = super().run(argv, timeout=timeout, max_output_bytes=max_output_bytes)
        if len(self.requests) == 1:
            self.clock.advance(self.first_advance)
        return result


@dataclass
class StaticProbe:
    name: str
    interval: float
    collection: probes.ProbeCollection

    def collect(self, runner: probes.CommandRunner) -> probes.ProbeCollection:
        del runner
        return self.collection


@dataclass
class SubprocessProbe:
    name: str
    interval: float
    pid_path: Path

    def collect(self, runner: probes.CommandRunner) -> probes.ProbeCollection:
        runner.run(
            (
                sys.executable,
                "-c",
                (
                    "import os, pathlib, sys, threading; "
                    "pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); "
                    "threading.Event().wait(60)"
                ),
                str(self.pid_path),
            ),
            timeout=60.0,
            max_output_bytes=128,
        )
        return probes.ProbeCollection.succeeded()


class _AcceptedResponse:
    status_code = 200
    headers: ClassVar[dict[str, str]] = {}

    def __init__(self, batch_id: str) -> None:
        self.batch_id = batch_id

    def json(self) -> dict[str, str]:
        return {"batch_id": self.batch_id, "status": "accepted"}


class BlockingTelemetryTransport:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def post(self, endpoint: str, body: bytes, batch_id: str, timeout: tuple[float, float]) -> _AcceptedResponse:
        del endpoint, body, timeout
        self.started.set()
        self.release.wait()
        return _AcceptedResponse(batch_id)

    def close(self) -> None:
        pass


def metric(collection: probes.ProbeCollection, name: str, **attributes: str) -> probes.ProbeMetric:
    return next(
        item
        for item in collection.metrics
        if item.name == name and all(item.attributes.get(key) == value for key, value in attributes.items())
    )


def documented_nccl_ras_payload() -> dict[str, Any]:
    """NVIDIA's documented NCCL 2.28.7+ JSON status shape."""
    return {
        "nccl_version": "2.29.1",
        "cuda_runtime_version": 13000,
        "cuda_driver_version": 13000,
        "timestamp": "2025-12-19 13:06:53",
        "communicators_count": 1,
        "communicators": [
            {
                "hash": "0xae94423cfbb2ef4a",
                "secondary_hash": "0xb7e7187447156001:0xb8242ed28a71381e",
                "size": 2,
                "ranks_count": 1,
                "missing_ranks_count": 1,
                "ranks": [
                    {
                        "rank": 0,
                        "host": "172.16.64.245",
                        "pid": 1524344,
                        "cuda_dev": 0,
                        "nvml_dev": 0,
                        "status": {
                            "init_state": 0,
                            "async_error": 0,
                            "finalize_called": False,
                            "destroy_flag": False,
                            "abort_flag": False,
                        },
                        "collective_counts": {
                            "Broadcast": 0,
                            "Reduce": 0,
                            "AllGather": 4,
                            "ReduceScatter": 7,
                            "AllReduce": 12,
                        },
                    }
                ],
                "missing_ranks": [
                    {
                        "rank": 1,
                        "host": "172.16.64.245",
                        "pid": 1524345,
                        "cuda_dev": 1,
                        "nvml_dev": 1,
                        "status": {"unresponsive": True, "considered_dead": False},
                    }
                ],
            }
        ],
        "ras": {"collection_time_sec": 0.125, "timeouts_count": 2},
    }


def test_import_does_not_start_probe_threads() -> None:
    code = """
import threading
before = {thread.ident for thread in threading.enumerate()}
from rigging import probes
after = [thread for thread in threading.enumerate() if thread.ident not in before]
assert after == [], after
"""

    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=3)

    assert result.returncode == 0, result.stderr


def test_probe_subprocess_runs_only_on_isolated_daemon_thread(capsys: pytest.CaptureFixture[str]) -> None:
    runner = RecordingRunner([probes.CommandResult(probes.CommandStatus.NOT_FOUND)])
    sink = RecordingSink()

    manager = probes.start(probes.nvidia_smi(), runner=runner, sink=sink)
    sample = sink.wait_for(1)[0]
    manager.shutdown(timeout=0.1)

    request_thread = runner.requests[0].thread
    assert request_thread is not threading.current_thread()
    assert request_thread.daemon
    assert sample.outcome is probes.ProbeOutcome.UNAVAILABLE
    assert sample.reason is probes.ProbeReason.TOOL_MISSING
    assert capsys.readouterr() == ("", "")


def test_monotonic_cadence_is_anchored_before_collection() -> None:
    clock = FakeClock(automatic_waits=1)
    sink = RecordingSink()

    class AdvancingProbe(StaticProbe):
        def collect(self, runner: probes.CommandRunner) -> probes.ProbeCollection:
            del runner
            clock.advance(7.0)
            return self.collection

    probe = AdvancingProbe("cadence", 10.0, probes.ProbeCollection.succeeded())
    manager = probes.start(probe, runner=RecordingRunner([]), sink=sink, clock=clock)
    sink.wait_for(2)
    manager.shutdown(timeout=0.1)

    assert clock.waits[:2] == pytest.approx([3.0, 3.0])


def test_on_demand_sample_wakes_only_the_named_probe() -> None:
    sink = RecordingSink()
    manager = probes.start(
        StaticProbe("first", 3_600.0, probes.ProbeCollection.succeeded()),
        StaticProbe("second", 3_600.0, probes.ProbeCollection.succeeded()),
        runner=RecordingRunner([]),
        sink=sink,
    )
    sink.wait_for(2)

    manager.sample("first")
    samples = sink.wait_for(3)
    manager.shutdown(timeout=0.1)

    assert [sample.probe for sample in samples].count("first") == 2
    assert [sample.probe for sample in samples].count("second") == 1


def test_blocked_probe_does_not_delay_peer_or_bounded_shutdown() -> None:
    release = threading.Event()
    entered = threading.Event()

    class BlockingProbe(StaticProbe):
        def collect(self, runner: probes.CommandRunner) -> probes.ProbeCollection:
            del runner
            entered.set()
            release.wait()
            return self.collection

    sink = RecordingSink()
    manager = probes.start(
        BlockingProbe("blocked", 600.0, probes.ProbeCollection.succeeded()),
        StaticProbe("healthy", 600.0, probes.ProbeCollection.succeeded()),
        runner=RecordingRunner([]),
        sink=sink,
    )
    assert entered.wait(1)
    samples = sink.wait_for(1)

    started = time.monotonic()
    manager.shutdown(timeout=0.02)
    elapsed = time.monotonic() - started
    status = manager.status()
    release.set()

    assert samples[0].probe == "healthy"
    assert elapsed < 0.2
    assert status.live_workers >= 1
    assert status.sink_failures == 0
    assert status.telemetry_lost_records == 0


def test_shutdown_cancels_and_reaps_all_production_subprocesses_within_shared_budget(tmp_path: Path) -> None:
    pid_paths = [tmp_path / "first.pid", tmp_path / "second.pid"]
    sink = RecordingSink()
    manager = probes.start(
        *(SubprocessProbe(f"process-{index}", 600.0, path) for index, path in enumerate(pid_paths)),
        sink=sink,
    )
    deadline = time.monotonic() + 2.0
    while not all(path.exists() for path in pid_paths) and time.monotonic() < deadline:
        threading.Event().wait(0.001)
    assert all(path.exists() for path in pid_paths)

    started = time.monotonic()
    manager.shutdown(timeout=0.5)
    elapsed = time.monotonic() - started

    assert elapsed < 0.7
    assert manager.status().live_workers == 0
    for path in pid_paths:
        with pytest.raises(ProcessLookupError):
            os.kill(int(path.read_text()), 0)


def test_shutdown_gate_prevents_subprocess_start_after_stop() -> None:
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    class GateRunner(RecordingRunner):
        def __init__(self) -> None:
            super().__init__([probes.CommandResult(probes.CommandStatus.COMPLETED, b"", 0)])
            self.cancelled = False
            self.starts = 0

        def run(self, argv: tuple[str, ...], *, timeout: float, max_output_bytes: int) -> probes.CommandResult:
            del argv, timeout, max_output_bytes
            if self.cancelled:
                return probes.CommandResult(probes.CommandStatus.CANCELLED)
            self.starts += 1
            return next(self.results)

        def cancel(self, timeout: float) -> None:
            del timeout
            self.cancelled = True

    class GatedProbe(StaticProbe):
        def collect(self, runner: probes.CommandRunner) -> probes.ProbeCollection:
            entered.set()
            release.wait()
            runner.run(("would-start",), timeout=60.0, max_output_bytes=128)
            finished.set()
            return self.collection

    runner = GateRunner()
    manager = probes.start(
        GatedProbe("gated", 600.0, probes.ProbeCollection.succeeded()),
        runner=runner,
        sink=RecordingSink(),
    )
    assert entered.wait(1)

    manager.shutdown(timeout=0.01)
    release.set()
    assert finished.wait(1)

    assert runner.starts == 0


@pytest.mark.parametrize("result_kind", ["metrics", "event"])
def test_oversized_probe_result_becomes_explicit_failure(result_kind: str) -> None:
    if result_kind == "metrics":
        oversized = tuple(
            probes.ProbeMetric("value", probes.MetricKind.GAUGE, float(index))
            for index in range(probes.MAX_RESULT_METRICS + 1)
        )
        collection = probes.ProbeCollection(probes.ProbeOutcome.SUCCEEDED, metrics=oversized)
        empty_field = "metrics"
    else:
        collection = probes.ProbeCollection.succeeded(
            events=(probes.ProbeEvent("diagnostic", {"raw": "x" * (probes.MAX_FIELD_BYTES + 1)}),)
        )
        empty_field = "events"
    sink = RecordingSink()
    manager = probes.start(
        StaticProbe("oversized", 600.0, collection),
        runner=RecordingRunner([]),
        sink=sink,
    )

    sample = sink.wait_for(1)[0]
    manager.shutdown(timeout=0.1)

    assert sample.outcome is probes.ProbeOutcome.FAILED
    assert sample.reason is probes.ProbeReason.RESULT_LIMIT
    assert getattr(sample, empty_field) == ()


def test_malformed_probe_result_is_distinct_from_size_limit() -> None:
    collection = probes.ProbeCollection.succeeded(
        metrics=(probes.ProbeMetric("invalid", probes.MetricKind.GAUGE, float("nan")),)
    )
    sink = RecordingSink()
    manager = probes.start(
        StaticProbe("invalid", 600.0, collection),
        runner=RecordingRunner([]),
        sink=sink,
    )

    sample = sink.wait_for(1)[0]
    manager.shutdown(timeout=0.1)

    assert sample.outcome is probes.ProbeOutcome.FAILED
    assert sample.reason is probes.ProbeReason.INVALID_RESULT


def test_probe_exception_reports_bounded_error_type() -> None:
    class RaisingProbe(StaticProbe):
        def collect(self, runner: probes.CommandRunner) -> probes.ProbeCollection:
            del runner
            raise ValueError("raw diagnostic is not retained")

    sink = RecordingSink()
    manager = probes.start(
        RaisingProbe("raising", 600.0, probes.ProbeCollection.succeeded()),
        runner=RecordingRunner([]),
        sink=sink,
    )

    sample = sink.wait_for(1)[0]
    manager.shutdown(timeout=0.1)

    assert sample.reason is probes.ProbeReason.INTERNAL_ERROR
    assert sample.error_type == "ValueError"


def test_sink_failure_is_counted_without_stopping_probe() -> None:
    class FailingOnceSink:
        def __init__(self) -> None:
            self.calls = 0
            self.first_attempt = threading.Event()
            self.recovered = threading.Event()

        def emit(self, sample: probes.ProbeSample) -> None:
            del sample
            self.calls += 1
            if self.calls == 1:
                self.first_attempt.set()
                raise RuntimeError("sink failed")
            self.recovered.set()

    sink = FailingOnceSink()
    manager = probes.start(
        StaticProbe("sink", 600.0, probes.ProbeCollection.succeeded()),
        runner=RecordingRunner([]),
        sink=sink,
    )
    assert sink.first_attempt.wait(1)

    manager.sample("sink")
    assert sink.recovered.wait(1)
    status = manager.status()
    manager.shutdown(timeout=0.1)

    assert status.sink_failures == 1
    assert status.sink_error_types == ("RuntimeError",)
    assert status.telemetry_lost_records == 0


def test_bounded_subprocess_runner_stops_output_overflow() -> None:
    result = probes.BoundedSubprocessRunner().run(
        (sys.executable, "-c", "import os; os.write(1, b'x' * 8192)"),
        timeout=1.0,
        max_output_bytes=128,
    )

    assert result.status is probes.CommandStatus.OUTPUT_LIMIT
    assert len(result.stdout) <= 128


def test_bounded_subprocess_runner_enforces_hard_timeout() -> None:
    started = time.monotonic()
    result = probes.BoundedSubprocessRunner().run(
        (sys.executable, "-c", "import threading; threading.Event().wait(60)"),
        timeout=0.05,
        max_output_bytes=128,
    )
    elapsed = time.monotonic() - started

    assert result.status is probes.CommandStatus.TIMED_OUT
    assert elapsed < 0.5


def test_default_sink_emits_common_probe_health_signals(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = RecordingTelemetryTransport()
    monkeypatch.setattr(telemetry, "_RequestsTransport", lambda: transport)
    telemetry.configure(
        endpoint="http://finelog.test/v1/telemetry",
        service="probe-test",
        retry_initial=0.001,
        retry_maximum=0.002,
    )
    collection = probes.ProbeCollection.succeeded(
        metrics=(probes.ProbeMetric("device_value", probes.MetricKind.GAUGE, 3.0),)
    )

    manager = probes.start(StaticProbe("health", 600.0, collection), runner=RecordingRunner([]))
    probe_runs = transport.record("probe_runs", {"probe": "health", "outcome": "succeeded"})
    transport.record("probe_duration_seconds", {"probe": "health", "outcome": "succeeded"})
    probe_up = transport.record("probe_up", {"probe": "health"})
    last_success = transport.record("probe_last_success_time_seconds", {"probe": "health"})
    transport.record("device_value", {})
    manager.shutdown(timeout=0.1)
    terminal = transport.record("probe_terminal", {"probe": "health", "reason": "clean_shutdown"})

    assert probe_runs["attributes"] == {"outcome": "succeeded", "probe": "health"}
    assert probe_up["value"] == 1.0
    assert last_success["value"] > 0
    assert terminal["body"] == {"state": "clean_shutdown"}
    names = [record["name"] for record in transport.records]
    assert names.index("device_value") < names.index("probe_up")


def test_manager_status_reports_telemetry_queue_loss_separately_from_sink_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = BlockingTelemetryTransport()
    monkeypatch.setattr(telemetry, "_RequestsTransport", lambda: transport)
    telemetry.configure(
        endpoint="http://finelog.test/v1/telemetry",
        service="probe-test",
        max_queue_records=1,
        max_batch_records=1,
        retry_initial=0.001,
        retry_maximum=0.002,
    )
    telemetry.gauge("occupy_queue").set(1)
    assert transport.started.wait(1)
    manager = probes.start(
        StaticProbe(
            "loss",
            600.0,
            probes.ProbeCollection.succeeded(metrics=(probes.ProbeMetric("payload", probes.MetricKind.GAUGE, 1.0),)),
        ),
        runner=RecordingRunner([]),
    )
    deadline = time.monotonic() + 1.0
    while manager.status().telemetry_lost_records == 0 and time.monotonic() < deadline:
        threading.Event().wait(0.001)
    status = manager.status()
    manager.shutdown(timeout=0.1)
    transport.release.set()

    assert status.telemetry_lost_records > 0
    assert status.sink_failures == 0


def test_nvidia_smi_normalizes_stable_device_identity_and_slow_health() -> None:
    output = (
        b"GPU-f81d4fae, 00000000:17:00.0, NVIDIA H100 80GB HBM3, 580.65.06, "
        b"81559, Default, Disabled, 700.00, 96.00.68.00.01, 2, 5, 1, Yes, 3, 4, Yes, No\n"
    )
    runner = RecordingRunner([probes.CommandResult(probes.CommandStatus.COMPLETED, output, 0)])

    collection = probes.nvidia_smi().collect(runner)

    inventory = metric(collection, "hardware_inventory")
    identity = {"gpu_uuid": "GPU-f81d4fae", "pci_bus_id": "00000000:17:00.0"}
    assert collection.outcome is probes.ProbeOutcome.SUCCEEDED
    assert inventory.attributes | identity == inventory.attributes
    assert inventory.attributes["gpu_model"] == "NVIDIA H100 80GB HBM3"
    assert metric(collection, "gpu_memory_total_bytes", **identity).value == 81_559 * 1024**2
    assert metric(collection, "gpu_ecc_uncorrected_errors", **identity).value == 2
    assert metric(collection, "gpu_retired_pages", error_kind="single_bit_ecc").value == 5
    assert metric(collection, "gpu_row_remapped_rows", error_kind="uncorrectable").value == 4
    assert metric(collection, "gpu_retired_pages_pending", **identity).value == 1
    assert metric(collection, "gpu_row_remap_pending", **identity).value == 1
    assert inventory.attributes["source_temporality"] == telemetry.CURRENT_SNAPSHOT
    assert metric(collection, "gpu_memory_total_bytes").attributes["source_temporality"] == telemetry.CURRENT_SNAPSHOT
    assert metric(collection, "gpu_retired_pages_pending").attributes["source_temporality"] == telemetry.CURRENT_SNAPSHOT
    assert metric(collection, "gpu_row_remap_failures").attributes["source_temporality"] == telemetry.CURRENT_SNAPSHOT
    assert runner.requests[0].timeout == pytest.approx(5.0)


def test_nvidia_smi_retries_baseline_inventory_with_remaining_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = FakeClock()
    monkeypatch.setattr(rigging_timing.time, "monotonic", clock.monotonic)

    baseline = b"GPU-f81d4fae, 00000000:17:00.0, NVIDIA H100 80GB HBM3, 580.65.06, 81559\n"
    runner = AdvancingRunner(
        [
            probes.CommandResult(probes.CommandStatus.COMPLETED, b"unsupported field", 2),
            probes.CommandResult(probes.CommandStatus.COMPLETED, baseline, 0),
        ],
        clock,
        2.0,
    )

    collection = probes.nvidia_smi().collect(runner)

    assert collection.outcome is probes.ProbeOutcome.SUCCEEDED
    assert metric(collection, "hardware_inventory").attributes["gpu_uuid"] == "GPU-f81d4fae"
    assert metric(collection, "gpu_memory_total_bytes").value == 81_559 * 1024**2
    assert not collection.events
    assert len(runner.requests) == 2
    assert runner.requests[0].timeout == pytest.approx(5.0)
    assert runner.requests[1].timeout == pytest.approx(3.0)
    assert runner.requests[1].argv[1] == "--query-gpu=uuid,pci.bus_id,name,driver_version,memory.total"


def test_nvidia_smi_no_devices_is_not_applicable() -> None:
    runner = RecordingRunner([probes.CommandResult(probes.CommandStatus.COMPLETED, b"", 0)])

    collection = probes.nvidia_smi().collect(runner)

    assert collection.outcome is probes.ProbeOutcome.NOT_APPLICABLE
    assert collection.reason is probes.ProbeReason.NO_DEVICES


def test_nccl_ras_prefers_json_and_reports_progress_without_latency() -> None:
    output = json.dumps(documented_nccl_ras_payload()).encode()
    runner = RecordingRunner([probes.CommandResult(probes.CommandStatus.COMPLETED, output, 0)])

    collection = probes.nccl_ras().collect(runner)

    all_reduce = metric(collection, "collective_operations", collective="AllReduce", rank="0")
    assert collection.outcome is probes.ProbeOutcome.SUCCEEDED
    assert all_reduce.kind is probes.MetricKind.GAUGE
    assert all_reduce.value == 12
    assert all_reduce.attributes["source_temporality"] == "cumulative_snapshot"
    assert all_reduce.attributes["communicator_hash"] == "0xae94423cfbb2ef4a"
    assert metric(collection, "communicators").value == 1
    assert metric(collection, "communicators").attributes["source_temporality"] == telemetry.CURRENT_SNAPSHOT
    assert metric(collection, "communicator_ranks", rank_state="total").value == 2
    assert metric(collection, "communicator_ranks", rank_state="reported").value == 1
    assert metric(collection, "communicator_ranks", rank_state="missing").value == 1
    assert metric(collection, "communicator_ranks", rank_state="unresponsive").value == 1
    assert metric(collection, "communicator_rank_status", rank="0").attributes["init_state"] == "0"
    assert metric(collection, "ras_collection_duration_seconds").value == pytest.approx(0.125)
    assert metric(collection, "ras_collection_timeouts").value == 2
    assert all("latency" not in item.name for item in collection.metrics)
    assert len(runner.requests) == 1
    request = runner.requests[0]
    argv = request.argv
    assert argv[:2] == ("ncclras", "-v")
    assert argv[-2:] == ("-f", "json")
    assert argv[argv.index("-t") + 1] == "6"
    assert request.timeout == pytest.approx(8.0)


def test_nccl_ras_marks_documented_legacy_text_unavailable_with_remaining_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock()
    monkeypatch.setattr(rigging_timing.time, "monotonic", clock.monotonic)

    output = b"""Job summary
===========

Communicators... (0.00s)
=============

Group     Comms     Nodes     Ranks     Ranks     Ranks    Status  Errors
    #  in group  per comm  per node  per comm  in group
    0         8         4         1         4        32   RUNNING      OK
"""
    runner = AdvancingRunner(
        [
            probes.CommandResult(probes.CommandStatus.COMPLETED, b"", 2),
            probes.CommandResult(probes.CommandStatus.COMPLETED, output, 0),
        ],
        clock,
        3.0,
    )

    collection = probes.nccl_ras().collect(runner)

    assert collection.outcome is probes.ProbeOutcome.UNAVAILABLE
    assert collection.reason is probes.ProbeReason.UNSUPPORTED_FORMAT
    assert len(runner.requests) == 2
    assert "-f" not in runner.requests[1].argv
    assert runner.requests[0].timeout == pytest.approx(8.0)
    assert runner.requests[1].timeout == pytest.approx(5.0)


def test_nccl_ras_rejects_synthesized_json_shape() -> None:
    synthesized = json.dumps(
        {
            "version": {"nccl": "2.28.9"},
            "communicators": [{"comm_hash": "0xabc", "ranks": {"total": 8}, "operations": {}}],
        }
    ).encode()
    runner = RecordingRunner([probes.CommandResult(probes.CommandStatus.COMPLETED, synthesized, 0)])

    collection = probes.nccl_ras().collect(runner)

    assert collection.outcome is probes.ProbeOutcome.FAILED
    assert collection.reason is probes.ProbeReason.PARSE_ERROR
    assert not collection.metrics
    assert len(runner.requests) == 1


def test_nccl_ras_timeout_is_explicit_progress_health_not_application_error() -> None:
    runner = RecordingRunner([probes.CommandResult(probes.CommandStatus.TIMED_OUT)])

    collection = probes.nccl_ras().collect(runner)

    assert collection.outcome is probes.ProbeOutcome.TIMED_OUT
    assert collection.reason is probes.ProbeReason.SUBPROCESS_TIMEOUT
    timeout_count = metric(collection, "ras_query_timeouts")
    assert timeout_count.kind is probes.MetricKind.COUNTER
    assert timeout_count.value == 1


def test_nccl_ras_emits_bounded_mismatch_and_dead_peer_anomalies() -> None:
    payload = documented_nccl_ras_payload()
    communicator = payload["communicators"][0]
    communicator["size"] = 3
    communicator["ranks_count"] = 2
    communicator["missing_ranks_count"] = 1
    communicator["ranks"].append(
        {
            "rank": 2,
            "host": "172.16.64.246",
            "pid": 1524346,
            "cuda_dev": 0,
            "nvml_dev": 0,
            "status": {
                "init_state": 0,
                "async_error": 0,
                "finalize_called": False,
                "destroy_flag": False,
                "abort_flag": False,
            },
            "collective_counts": {"AllReduce": 11},
        }
    )
    communicator["missing_ranks"][0]["status"]["considered_dead"] = True
    payload["diagnostic"] = "raw output must not be retained"
    output = json.dumps(payload).encode()
    runner = RecordingRunner([probes.CommandResult(probes.CommandStatus.COMPLETED, output, 0)])

    collection = probes.nccl_ras().collect(runner)

    anomalies = {event.attributes["anomaly"] for event in collection.events}
    encoded_events = json.dumps(
        [{"body": dict(event.body), "attributes": dict(event.attributes)} for event in collection.events]
    )
    assert anomalies == {"incomplete_communicator", "unresponsive_rank", "dead_peer", "collective_mismatch"}
    assert "raw output must not be retained" not in encoded_events


@pytest.mark.parametrize(
    ("factory", "expected_reason"),
    [
        (probes.nvidia_smi, probes.ProbeReason.TOOL_MISSING),
        (probes.nccl_ras, probes.ProbeReason.TOOL_MISSING),
    ],
)
def test_missing_probe_tool_is_explicit_unavailable(factory, expected_reason: probes.ProbeReason) -> None:
    runner = RecordingRunner([probes.CommandResult(probes.CommandStatus.NOT_FOUND)])

    collection = factory().collect(runner)

    assert collection.outcome is probes.ProbeOutcome.UNAVAILABLE
    assert collection.reason is expected_reason
