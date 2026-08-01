# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp
import numpy as np
import pytest
from rigging import telemetry
from rigging.testing import RecordingTelemetryTransport

from levanter.tracker import telemetry as tracker_telemetry
from levanter.tracker.histogram import SummaryStats
from levanter.tracker.telemetry import TelemetryTracker, TrainingPhase


@pytest.fixture
def exported(monkeypatch):
    telemetry.shutdown(0)
    transport = RecordingTelemetryTransport()
    monkeypatch.setattr(telemetry, "_RequestsTransport", lambda: transport)
    telemetry.configure(endpoint="http://finelog/v1/telemetry", service="levanter", attributes={"run": "run-42"})
    yield transport
    telemetry.shutdown(1)


def _values(records: list[dict]) -> dict[str, float]:
    return {record["name"]: record["value"] for record in records}


def test_log_exports_jax_scalar_snapshots_without_service_prefix(exported):
    tracker = TelemetryTracker()
    tracker.log({"train/loss": jnp.float32(1.25), "throughput": np.float64(7)}, step=3)

    records = exported.wait_for(7)
    values = _values(records)
    assert values["train_loss"] == 1.25
    assert values["throughput"] == 7.0
    assert values["step"] == 3.0
    train_loss = next(record for record in records if record["name"] == "train_loss")
    assert train_loss["attributes"] == {"source_kind": "gauge", "source_temporality": "current_snapshot"}


def test_complex_scalar_does_not_poison_valid_metrics_in_same_log_call(exported):
    tracker = TelemetryTracker()
    tracker.log({"unsupported": np.complex64(1 + 2j), "valid": 3.0}, step=None)

    records = exported.wait_for(3)
    values = _values(records)
    assert values["valid"] == 3.0
    assert "unsupported" not in values


def test_summary_stats_exports_preaggregated_current_gauges(exported):
    tracker = TelemetryTracker()
    values = jnp.asarray(np.linspace(0.0, 1.0, 1000))
    tracker.log({"grad": SummaryStats.from_array(values, num_bins=4)}, step=1)

    records = exported.wait_for(10)
    grad = [record for record in records if record["name"].startswith("grad_")]
    assert _values(grad) == pytest.approx(
        {
            "grad_mean": 0.5,
            "grad_min": 0.0,
            "grad_max": 1.0,
            "grad_variance": 0.0833,
            "grad_rms": 0.5774,
            "grad_count": 1000,
            "grad_sum": 500.0,
        },
        abs=1e-3,
    )
    assert all(record["kind"] == "gauge" for record in grad)
    assert all(record["attributes"]["source_temporality"] == "current_snapshot" for record in grad)


def test_summary_stats_does_not_export_a_row_per_histogram_bucket(exported):
    """One row per bin, per metric, per step is what saturated the finelog hub."""
    tracker = TelemetryTracker()
    values = jnp.asarray(np.linspace(0.0, 1.0, 1000))
    stats = SummaryStats.from_array(values, num_bins=64)
    assert stats.histogram is not None, "a summary without a histogram would pass this test vacuously"

    tracker.log({"grad": stats}, step=1)

    records = exported.wait_for(10)
    assert [record["name"] for record in records if record["name"].endswith("_bucket")] == []
    assert not any("le" in record["attributes"] for record in records)


def test_training_progress_and_phase_are_current_snapshots(exported, monkeypatch):
    monkeypatch.setattr("levanter.tracker.telemetry.time", lambda: 1234.5)
    tracker = TelemetryTracker()
    tracker.log({"train/loss": 1.25}, step=3)
    tracker.finish()

    values = _values(exported.wait_for(7))
    assert values["progress_time_seconds"] == 1234.5
    assert values["phase"] == TrainingPhase.FINISHED


@pytest.fixture
def fast_heartbeat(monkeypatch):
    heartbeat = tracker_telemetry._PhaseHeartbeat(interval=0.01)
    monkeypatch.setattr(tracker_telemetry, "_HEARTBEAT", heartbeat)
    yield heartbeat
    heartbeat.stop()


def test_phase_is_republished_while_a_job_initializes(exported, fast_heartbeat):
    """A job that hangs before its first step must stay enrolled.

    Stalled-training detection finds a job by its newest `phase` row. Written
    only on transition, an initializing job's sole row ages out of any bounded
    window and the job goes silent exactly when it is stuck.
    """
    TelemetryTracker()  # never logs a step

    phases = [record for record in exported.wait_for(6) if record["name"] == "phase"]
    assert len(phases) >= 3, "phase should be republished, not written once"
    assert all(record["value"] == TrainingPhase.INITIALIZING for record in phases)


def test_finish_stops_the_phase_heartbeat(exported, fast_heartbeat):
    """A finished run has nothing left to enroll, so the worker must not outlive it."""
    tracker = TelemetryTracker()
    exported.wait_for(4)
    assert fast_heartbeat.running

    tracker.finish()

    assert not fast_heartbeat.running
