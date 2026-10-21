# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import json
import stat
import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from finelog.client import FlushResult
from rigging import telemetry
from rigging.telemetry.probes import nccl_ras
from rigging.telemetry.probes.nccl_client import NCCL_RAS_ENABLE_ENV
from rigging.testing import RecordingTelemetryTransport, nccl_ras_payload

from levanter.callbacks import ProgressEvent
from levanter.callbacks.progress_watchdog import ProgressTimeout
from levanter.tracker import BackgroundTracker, NoopTracker, current_tracker, telemetry as tracker_telemetry
from levanter.tracker.histogram import SummaryStats
from levanter.tracker.telemetry import TelemetryConfig, TelemetryTracker, TrainingPhase


@dataclass(frozen=True)
class _MetricWrite:
    name: str
    method: str
    step: int | None
    value: float | None = None
    summary: SummaryStats | None = None


class _RecordingWriter:
    def __init__(self):
        self.rows: list[_MetricWrite] = []
        self.closed = False
        self.flushes = 0

    def scalar(self, name: str, value: float, *, step: int | None, **_kwargs) -> None:
        self.rows.append(_MetricWrite(name, "scalar", step, value=value))

    def summary(self, name: str, stats: SummaryStats, *, step: int | None, **_kwargs) -> None:
        self.rows.append(_MetricWrite(name, "summary", step, summary=stats))

    def flush(self, _timeout: float | None = None) -> FlushResult:
        self.flushes += 1
        return FlushResult.SUCCEEDED

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def writer() -> _RecordingWriter:
    return _RecordingWriter()


@pytest.fixture
def exported(monkeypatch):
    telemetry.shutdown(0)
    transport = RecordingTelemetryTransport()
    monkeypatch.setattr(telemetry, "_RequestsTransport", lambda: transport)
    telemetry.configure(
        endpoint="http://finelog/v1/telemetry",
        service="levanter",
        attributes={"run": "run-42"},
    )
    yield transport
    telemetry.shutdown(1)


def _values(records: list[dict]) -> dict[str, float]:
    return {record["name"]: record["value"] for record in records}


def _install_fake_nccl_client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    detail: nccl_ras.RasDetail,
) -> None:
    report = nccl_ras.reduce_response(json.dumps(nccl_ras_payload()).encode(), detail=detail)
    output = nccl_ras.NcclRasClientOutput.success(report).to_string()
    command = tmp_path / "nccl-client"
    command.write_text(f"#!{sys.executable}\nprint({output!r})\n")
    command.chmod(command.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setattr(tracker_telemetry.nccl, "_CLIENT_COMMAND", (str(command),))


def test_log_writes_scalars_with_step_as_a_column(writer):
    tracker = TelemetryTracker(writer)
    tracker.log(
        {
            "train/loss": jnp.float32(1.25),
            "throughput": np.float64(7),
            "global_step": 10,
            "step": 10,
        },
        step=10,
    )

    train_loss = next(row for row in writer.rows if row.name == "train_loss")
    throughput = next(row for row in writer.rows if row.name == "throughput")
    assert train_loss.value == 1.25
    assert train_loss.step == 10
    assert throughput.value == 7.0
    assert throughput.step == 10
    assert not any(row.name in {"step", "global_step"} for row in writer.rows)
    tracker.finish()


def test_complex_scalar_does_not_poison_valid_metrics_in_same_log_call(writer):
    tracker = TelemetryTracker(writer)
    tracker.log({"unsupported": np.complex64(1 + 2j), "valid": 3.0}, step=None)

    assert next(row for row in writer.rows if row.name == "valid").value == 3.0
    assert not any(row.name == "unsupported" for row in writer.rows)
    tracker.finish()


def test_summary_stats_are_one_summary_write(writer):
    tracker = TelemetryTracker(writer)
    values = jnp.asarray(np.linspace(0.0, 1.0, 1000))
    stats = SummaryStats.from_array(values, num_bins=64)
    assert stats.histogram is not None, "a summary carrying no histogram would pass this vacuously"

    tracker.log({"grad": stats}, step=10)
    grad = [row for row in writer.rows if row.name == "grad"]
    assert len(grad) == 1
    assert grad[0].method == "summary"
    assert grad[0].step == 10
    assert grad[0].summary is stats
    tracker.finish()


def test_training_progress_and_phase_carry_the_current_step(writer, monkeypatch):
    monkeypatch.setattr("levanter.tracker.telemetry.time", lambda: 1234.5)
    tracker = TelemetryTracker(writer)
    tracker.log({"train/loss": 1.25}, step=3)
    tracker.finish()

    progress = [row for row in writer.rows if row.name == "progress_time_seconds"]
    assert progress[-1] == _MetricWrite("progress_time_seconds", "scalar", 3, value=1234.5)
    phases = [row for row in writer.rows if row.name == "phase"]
    assert phases[-1] == _MetricWrite("phase", "scalar", 3, value=TrainingPhase.FINISHED)


def test_nonprimary_process_does_not_create_a_metrics_writer(monkeypatch):
    monkeypatch.setattr(tracker_telemetry.jax, "process_index", lambda: 1)
    monkeypatch.setattr(tracker_telemetry.runtime_telemetry, "configure", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        tracker_telemetry.LevanterMetricsWriter,
        "from_iris",
        lambda *_args, **_kwargs: pytest.fail("nonprimary process created a Finelog metrics writer"),
    )
    tracker = TelemetryConfig().init("run-42")

    tracker.log({"train/loss": 1.25, "throughput": 7.0}, step=3)
    assert isinstance(tracker, NoopTracker)


def test_metrics_connection_failure_does_not_stop_training(monkeypatch):
    def fail_to_connect(*_args, **_kwargs):
        raise ConnectionError("Finelog unavailable")

    monkeypatch.setattr(tracker_telemetry.jax, "process_index", lambda: 0)
    monkeypatch.setattr(tracker_telemetry.runtime_telemetry, "configure", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        tracker_telemetry.LevanterMetricsWriter,
        "from_iris",
        fail_to_connect,
    )

    tracker = TelemetryConfig().init("run-42")

    tracker.log({"train/loss": 1.25}, step=3)
    assert isinstance(tracker, NoopTracker)


def test_non_status_metrics_are_sampled_every_ten_steps_while_status_metrics_are_not(writer):
    tracker = TelemetryTracker(writer)

    tracker.log({"train/loss": 1.25, "throughput": 7.0}, step=1)
    tracker.log({"train/loss": 1.0, "throughput": 8.0}, step=2)
    tracker.log({"train/loss": 0.75, "throughput": 9.0}, step=10)
    assert [row.value for row in writer.rows if row.name == "train_loss"] == [1.25, 1.0, 0.75]
    assert [row.value for row in writer.rows if row.name == "throughput"] == [9.0]
    assert [row.value for row in writer.rows if row.name == "phase"] == [
        TrainingPhase.INITIALIZING,
        TrainingPhase.TRAINING,
    ]
    tracker.finish()


def test_phase_is_republished_while_a_job_initializes(writer):
    """A job that hangs before its first step must stay enrolled.

    Stalled-training detection finds a job by its newest `phase` row. Written
    only on transition, an initializing job's sole row ages out of any bounded
    window and the job goes silent exactly when it is stuck.
    """
    tracker = TelemetryTracker(writer, heartbeat_interval=0.01)  # never logs a step

    for _ in range(100):
        phases = [row for row in writer.rows if row.name == "phase"]
        if len(phases) >= 3:
            break
        tracker_telemetry.threading.Event().wait(0.005)
    assert len(phases) >= 3, "phase should be republished, not written once"
    assert all(row.value == TrainingPhase.INITIALIZING for row in phases)
    tracker.finish()


def test_finish_stops_the_phase_heartbeat(writer):
    """A finished run has nothing left to enroll, so the worker must not outlive it."""
    tracker = TelemetryTracker(writer, heartbeat_interval=0.01)

    for _ in range(100):
        if len([row for row in writer.rows if row.name == "phase"]) >= 2:
            break
        tracker_telemetry.threading.Event().wait(0.005)

    tracker.finish()
    phase_count_after_finish = len([row for row in writer.rows if row.name == "phase"])
    tracker_telemetry.threading.Event().wait(0.03)

    assert len([row for row in writer.rows if row.name == "phase"]) == phase_count_after_finish
    assert writer.closed


def test_gpu_primary_process_exports_nccl_ras_until_finish(exported, writer, monkeypatch, tmp_path):
    _install_fake_nccl_client(monkeypatch, tmp_path, nccl_ras.RasDetail.PERIODIC)
    monkeypatch.setenv(NCCL_RAS_ENABLE_ENV, "1")
    monkeypatch.setattr(tracker_telemetry.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(tracker_telemetry.jax, "process_index", lambda: 0)

    tracker = TelemetryTracker(writer)
    rank = exported.record(
        "communicator_rank_status",
        {"communicator_hash": "0xae94423cfbb2ef4a", "rank": "1"},
    )

    tracker.finish()

    assert rank["attributes"]["rank_host"] == "10.0.0.2"


def test_gpu_nonprimary_process_does_not_duplicate_nccl_ras_polling(exported, writer, monkeypatch):
    monkeypatch.setenv(NCCL_RAS_ENABLE_ENV, "1")
    monkeypatch.setattr(tracker_telemetry.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(tracker_telemetry.jax, "process_index", lambda: 1)
    monkeypatch.setattr(
        tracker_telemetry.nccl,
        "start",
        lambda **_kwargs: pytest.fail("nonprimary process started NCCL RAS polling"),
    )

    tracker = TelemetryTracker(writer)
    tracker.finish()

    assert not any(record["name"] == "communicators" for record in exported.records)


def test_stall_diagnostic_exports_fresh_detail_without_disabling_telemetry(exported, writer, monkeypatch, tmp_path):
    _install_fake_nccl_client(monkeypatch, tmp_path, nccl_ras.RasDetail.STALL)
    monkeypatch.setenv(NCCL_RAS_ENABLE_ENV, "1")
    monkeypatch.setattr(tracker_telemetry.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(tracker_telemetry.jax, "process_index", lambda: 0)
    tracker = BackgroundTracker(TelemetryTracker(writer))

    with current_tracker(tracker):
        tracker_telemetry.capture_stall_diagnostics(
            ProgressTimeout(ProgressEvent.TRAIN_STEP_STARTED, timedelta(minutes=15).total_seconds(), 900)
        )

    stall = next(
        record
        for record in exported.records
        if record["name"] == "communicators"
        and record["attributes"].get("trigger") == "stall"
        and record["attributes"].get("detail") == "stall"
    )
    assert stall["value"] == 1
    assert telemetry.runtime_status().configured
    tracker.finish()
