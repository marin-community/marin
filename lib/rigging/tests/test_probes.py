# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field

import pytest
from rigging import probes, telemetry


@pytest.fixture(autouse=True)
def reset_telemetry() -> Iterator[None]:
    telemetry.shutdown(0.01)
    yield
    telemetry.shutdown(0.1)


class RecordingRunner:
    def __init__(self, results: list[probes.CommandResult]) -> None:
        self.results = iter(results)
        self.requests: list[tuple[tuple[str, ...], float, int, threading.Thread]] = []

    def run(self, argv: tuple[str, ...], *, timeout: float, max_output_bytes: int) -> probes.CommandResult:
        self.requests.append((argv, timeout, max_output_bytes, threading.current_thread()))
        return next(self.results)


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


@dataclass
class AcceptedResponse:
    batch_id: str
    status_code: int = 200
    headers: dict[str, str] = field(default_factory=dict)

    def json(self) -> dict[str, str]:
        return {"batch_id": self.batch_id, "status": "accepted"}


class CapturingTransport:
    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []
        self._condition = threading.Condition()

    def post(self, endpoint: str, body: bytes, batch_id: str, timeout: tuple[float, float]) -> AcceptedResponse:
        del endpoint, timeout
        with self._condition:
            self.records.extend(json.loads(body)["records"])
            self._condition.notify_all()
        return AcceptedResponse(batch_id)

    def close(self) -> None:
        pass

    def wait_for_names(self, expected: set[str], timeout: float = 1.0) -> list[dict[str, object]]:
        with self._condition:
            assert self._condition.wait_for(
                lambda: expected <= {str(record["name"]) for record in self.records},
                timeout,
            )
            return list(self.records)


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


@dataclass
class StaticProbe:
    name: str
    interval: float
    collection: probes.ProbeCollection

    def collect(self, runner: probes.CommandRunner) -> probes.ProbeCollection:
        del runner
        return self.collection


def metric(sample: probes.ProbeCollection, name: str, **attributes: str) -> probes.ProbeMetric:
    return next(
        item
        for item in sample.metrics
        if item.name == name and all(item.attributes.get(key) == value for key, value in attributes.items())
    )


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

    request_thread = runner.requests[0][3]
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
    release.set()

    assert samples[0].probe == "healthy"
    assert elapsed < 0.2


def test_oversized_custom_result_becomes_explicit_failure() -> None:
    oversized = tuple(
        probes.ProbeMetric("value", probes.MetricKind.GAUGE, float(index))
        for index in range(probes.MAX_RESULT_METRICS + 1)
    )
    sink = RecordingSink()
    manager = probes.start(
        StaticProbe("oversized", 600.0, probes.ProbeCollection(probes.ProbeOutcome.SUCCEEDED, metrics=oversized)),
        runner=RecordingRunner([]),
        sink=sink,
    )

    sample = sink.wait_for(1)[0]
    manager.shutdown(timeout=0.1)

    assert sample.outcome is probes.ProbeOutcome.FAILED
    assert sample.reason is probes.ProbeReason.RESULT_LIMIT
    assert sample.metrics == ()


def test_oversized_event_body_becomes_explicit_failure() -> None:
    collection = probes.ProbeCollection.succeeded(
        events=(probes.ProbeEvent("diagnostic", {"raw": "x" * (probes.MAX_FIELD_BYTES + 1)}),)
    )
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
    assert sample.events == ()


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
    transport = CapturingTransport()
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
    records = transport.wait_for_names(
        {
            "probe_runs",
            "probe_duration_seconds",
            "probe_up",
            "probe_last_success_time_seconds",
            "device_value",
        }
    )
    manager.shutdown(timeout=0.1)
    records = transport.wait_for_names({"probe_terminal"})

    by_name = {str(record["name"]): record for record in records}
    assert by_name["probe_runs"]["attributes"] == {"outcome": "succeeded", "probe": "health"}
    assert by_name["probe_up"]["value"] == 1.0
    assert by_name["probe_last_success_time_seconds"]["value"] > 0
    assert by_name["probe_terminal"]["attributes"] == {"probe": "health", "reason": "clean_shutdown"}


def test_hardware_probe_default_cadence_is_ten_minutes() -> None:
    assert probes.nvidia_smi().interval == 600.0
    assert probes.nccl_ras().interval == 600.0


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
    assert runner.requests[0][1:] == (5.0, probes.MAX_SUBPROCESS_OUTPUT_BYTES, threading.current_thread())


def test_nvidia_smi_no_devices_is_not_applicable() -> None:
    runner = RecordingRunner([probes.CommandResult(probes.CommandStatus.COMPLETED, b"", 0)])

    collection = probes.nvidia_smi().collect(runner)

    assert collection.outcome is probes.ProbeOutcome.NOT_APPLICABLE
    assert collection.reason is probes.ProbeReason.NO_DEVICES


def test_nccl_ras_prefers_json_and_reports_progress_without_latency() -> None:
    output = json.dumps(
        {
            "version": {"nccl": "2.28.9", "cuda_runtime": "13.0", "cuda_driver": "13.0"},
            "communicators": [
                {
                    "comm_hash": "0xabc",
                    "secondary_hash": "0xdef",
                    "state": "active",
                    "ranks": {"total": 8, "missing": 0, "unresponsive": 0, "considered_dead": 0},
                    "initialization": "complete",
                    "async_error": "none",
                    "operations": {"AllReduce": 12, "AllGather": 4, "ReduceScatter": 7},
                }
            ],
        }
    ).encode()
    runner = RecordingRunner([probes.CommandResult(probes.CommandStatus.COMPLETED, output, 0)])

    collection = probes.nccl_ras().collect(runner)

    all_reduce = metric(collection, "collective_operations", collective="AllReduce")
    assert collection.outcome is probes.ProbeOutcome.SUCCEEDED
    assert all_reduce.kind is probes.MetricKind.GAUGE
    assert all_reduce.value == 12
    assert all_reduce.attributes["source_temporality"] == "cumulative_snapshot"
    assert all_reduce.attributes["communicator_hash"] == "0xabc"
    assert metric(collection, "communicator_ranks", rank_state="total").value == 8
    assert all("latency" not in item.name for item in collection.metrics)
    argv, timeout, _, _ = runner.requests[0]
    assert argv[:2] == ("ncclras", "-v")
    assert argv[-2:] == ("-f", "json")
    assert argv[argv.index("-t") + 1] == "6"
    assert timeout == 8.0


def test_nccl_ras_falls_back_to_bounded_legacy_summary() -> None:
    output = b"""NCCL RAS (v2.27.5)
CUDA runtime version: 12.8
CUDA driver version: 12.8
Communicator hash=0xabc secondary_hash=0xdef state=active
Ranks: total=8 missing=1 unresponsive=0 considered_dead=0
Initialization: complete
Async error: none
Operations:
  AllReduce: 17
  AllGather: 4
"""
    runner = RecordingRunner(
        [
            probes.CommandResult(probes.CommandStatus.COMPLETED, b"", 2),
            probes.CommandResult(probes.CommandStatus.COMPLETED, output, 0),
        ]
    )

    collection = probes.nccl_ras().collect(runner)

    assert collection.outcome is probes.ProbeOutcome.SUCCEEDED
    assert metric(collection, "collective_operations", collective="AllReduce").value == 17
    assert metric(collection, "communicator_ranks", rank_state="missing").value == 1
    assert collection.events[0].attributes["anomaly"] == "incomplete_communicator"
    assert len(runner.requests) == 2
    assert "-f" not in runner.requests[1][0]


def test_nccl_ras_timeout_is_explicit_progress_health_not_application_error() -> None:
    runner = RecordingRunner([probes.CommandResult(probes.CommandStatus.TIMED_OUT)])

    collection = probes.nccl_ras().collect(runner)

    assert collection.outcome is probes.ProbeOutcome.TIMED_OUT
    assert collection.reason is probes.ProbeReason.SUBPROCESS_TIMEOUT
    timeout_count = metric(collection, "ras_query_timeouts")
    assert timeout_count.kind is probes.MetricKind.COUNTER
    assert timeout_count.value == 1


def test_nccl_ras_emits_bounded_mismatch_and_dead_peer_anomalies() -> None:
    output = json.dumps(
        {
            "communicators": [
                {
                    "comm_hash": "0xabc",
                    "secondary_hash": "0xdef",
                    "state": "active",
                    "ranks": {"total": 8, "missing": 0, "unresponsive": 1, "considered_dead": 1},
                    "mismatch": True,
                    "operations": {"AllReduce": 12},
                    "diagnostic": "raw output must not be retained",
                }
            ]
        }
    ).encode()
    runner = RecordingRunner([probes.CommandResult(probes.CommandStatus.COMPLETED, output, 0)])

    collection = probes.nccl_ras().collect(runner)

    anomalies = {event.attributes["anomaly"] for event in collection.events}
    encoded_events = json.dumps(
        [{"body": dict(event.body), "attributes": dict(event.attributes)} for event in collection.events]
    )
    assert anomalies == {"unresponsive_rank", "dead_peer", "collective_mismatch"}
    assert "raw output must not be retained" not in encoded_events
    assert all(len(json.dumps(dict(event.body)).encode()) <= probes.MAX_EVENT_BODY_BYTES for event in collection.events)


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
