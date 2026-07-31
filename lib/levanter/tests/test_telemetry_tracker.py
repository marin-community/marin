# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp
import numpy as np
import pytest
from rigging import telemetry
from rigging.testing import RecordingTelemetryTransport

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

    records = exported.wait_for(15)
    buckets = [record for record in records if record["name"] == "grad_bucket"]
    assert len(buckets) == 5
    assert next(record for record in buckets if record["attributes"]["le"] == "+Inf")["value"] == 1000
    assert all(record["kind"] == "gauge" for record in buckets)
    assert all(record["attributes"]["source_temporality"] == "current_snapshot" for record in buckets)
    assert _values(records)["grad_count"] == 1000


def test_training_progress_and_phase_are_current_snapshots(exported, monkeypatch):
    monkeypatch.setattr("levanter.tracker.telemetry.time", lambda: 1234.5)
    tracker = TelemetryTracker()
    tracker.log({"train/loss": 1.25}, step=3)
    tracker.finish()

    values = _values(exported.wait_for(7))
    assert values["progress_time_seconds"] == 1234.5
    assert values["phase"] == TrainingPhase.FINISHED
