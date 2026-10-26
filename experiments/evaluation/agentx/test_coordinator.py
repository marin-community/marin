# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Control-plane interruption regressions; these tests allocate no hardware."""

import dataclasses
import io
import json
import os
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import zstandard
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from iris.client.workload import JobStatus, ResourceRequest
from iris.cluster.types import JobName
from iris.resources.state import FederationState, JobState
from rigging.timing import Timestamp

from experiments.evaluation.agentx.collect import (
    collect_terminal,
    read_record,
    retain_processed_artifacts,
    unpack,
    validate_window,
    write_record,
)
from experiments.evaluation.agentx.control import coordinator_plan, materialize, pending_result
from experiments.evaluation.agentx.coordinator import recover_results, resolve_plan, run_points
from experiments.evaluation.agentx.processes import logged_process
from experiments.evaluation.agentx.recipe import SERIES_PATH


def job_status(job_id: JobName, state: JobState) -> JobStatus:
    return JobStatus(
        job_id=job_id,
        state=state,
        exit_code=0,
        error_message="",
        submitted_at=Timestamp.from_seconds(1000),
        started_at=None,
        finished_at=None,
        ports={},
        status_message="",
        build_metrics=None,
        failure_count=0,
        preemption_count=0,
        tasks=(),
        name=job_id.name,
        resources=ResourceRequest(cpu_millicores=32000, memory_bytes=0, disk_bytes=0),
        task_state_counts={},
        task_count=1,
        completed_count=int(state == JobState.SUCCEEDED),
        pending_reason="",
        has_children=False,
        parent_job_id=None,
        backend_id="fixture",
        execution_cluster_id="fixture",
        federation_state=FederationState.LOCAL,
        submitting_user="fixture",
    )


class FakeController:
    """Iris RPC boundary with terminal scheduling failures and hierarchical ownership."""

    def __init__(self, plan):
        self.plan = plan
        self.jobs = {plan["job_id"]: job_status(JobName.from_string(plan["job_id"]), JobState.RUNNING)}
        self.waited = []
        self.stop_after = None

    def job_status(self, job_id):
        if str(job_id) not in self.jobs:
            raise ConnectError(code=Code.NOT_FOUND, message="No such fixture job")
        return self.jobs[str(job_id)]

    def submit(self, **request):
        if self.stop_after is not None and len(self.waited) >= self.stop_after:
            raise InterruptedError("Coordinator lost before the next submission")
        job_id = JobName.from_string(self.plan["job_id"]).child(request["name"])
        retained = read_record(str(Path(self.plan["retention_uri"]) / "plan.json"))
        assert retained["points"] == self.plan["points"]
        # Sequential ownership: the previous child is terminal before submission.
        assert all(status.state != JobState.PENDING for status in self.jobs.values())
        self.jobs[str(job_id)] = job_status(job_id, JobState.PENDING)

        def wait(**_kwargs):
            self.jobs[str(job_id)] = dataclasses.replace(self.jobs[str(job_id)], state=JobState.FAILED)
            self.waited.append(str(job_id))
            return self.jobs[str(job_id)]

        return SimpleNamespace(job_id=job_id, wait=wait)

    def list_tasks(self, job_id):
        return self.jobs[str(job_id)].tasks


@pytest.fixture
def saved_batch(tmp_path):
    source = {
        "kind": "github-actions",
        "run_id": 123,
        "run_attempt": 2,
        "repository": "fixture/fixture",
        "head_sha": "1" * 40,
    }
    submission = coordinator_plan(source, "h100", "cw-rno2a", "batch", str(tmp_path / "retained"), None)
    plan = resolve_plan(submission, {"key": "fixture", "archive_sha256": "a" * 64})
    return plan, str(tmp_path / "submission.json")


def test_coordinator_retains_each_terminal_outcome_and_runs_points_serially(saved_batch, tmp_path):
    plan, _ = saved_batch
    controller = FakeController(plan)
    with pytest.raises(RuntimeError, match="points failed"):
        run_points(controller, plan, tmp_path, tmp_path)
    result = read_record(pending_result(plan["root"]))
    assert result["source"] == plan["source"]
    assert len(controller.waited) == 2
    # The one-point result was deliverable before the second point completed.
    completed = list((tmp_path / "retained" / "completed").glob("*.json"))
    assert sorted(len(json.loads(path.read_text())["points"]) for path in completed) == [1, 2]
    assert all(point["receipt"]["status"] == "failed" for point in result["points"])


def test_coordinator_loss_keeps_completed_results_and_manual_recovery_never_submits(saved_batch, tmp_path):
    plan, descriptor = saved_batch
    write_record(descriptor, plan)
    controller = FakeController(plan)
    controller.stop_after = 1
    with pytest.raises(InterruptedError):
        run_points(controller, plan, tmp_path, tmp_path)
    completed_uri = pending_result(plan["root"])
    first_result = read_record(completed_uri)
    controller.jobs[plan["job_id"]] = dataclasses.replace(controller.jobs[plan["job_id"]], state=JobState.FAILED)
    recovered_uri = recover_results(controller, read_record(descriptor), tmp_path, tmp_path)
    assert read_record(recovered_uri) == first_result
    assert len(controller.jobs) == 2  # Failed parent and its only submitted child.
    assert len(controller.waited) == 1


def test_recovery_refuses_a_live_coordinator(saved_batch, tmp_path):
    plan, _ = saved_batch
    controller = FakeController(plan)
    with pytest.raises(ValueError, match="still live"):
        recover_results(controller, plan, tmp_path, tmp_path)
    assert len(controller.jobs) == 1


def test_recovery_after_preparation_loss_publishes_no_measurements(saved_batch, tmp_path):
    plan, _ = saved_batch
    controller = FakeController(plan)
    controller.jobs[plan["job_id"]] = dataclasses.replace(controller.jobs[plan["job_id"]], state=JobState.FAILED)
    result = read_record(recover_results(controller, plan, tmp_path, tmp_path))
    assert result["points"] == []
    assert result["source"] == plan["source"]
    assert list(controller.jobs) == [plan["job_id"]]


def test_overlapping_sources_retain_independent_completed_results(saved_batch, tmp_path):
    plan, _ = saved_batch
    second = resolve_plan(
        coordinator_plan(plan["source"] | {"run_id": 124}, "h100", "cw-rno2a", "batch", plan["root"], 8),
        plan["points"][0]["manifest"]["workload"],
    )
    for submission in (plan, second):
        with pytest.raises(RuntimeError):
            run_points(FakeController(submission), submission, tmp_path, tmp_path)
    sources = {
        json.loads(path.read_text())["source"]["run_id"] for path in (tmp_path / "retained" / "completed").glob("*.json")
    }
    assert sources == {123, 124}


def test_delivery_acknowledgement_does_not_change_results_or_create_execution_state(saved_batch, tmp_path):
    plan, _ = saved_batch
    with pytest.raises(RuntimeError):
        run_points(FakeController(plan), plan, tmp_path, tmp_path)
    uri = pending_result(plan["root"])
    original = read_record(uri)
    result = materialize(uri, tmp_path / "import")
    assert result == read_record(str(tmp_path / "import" / "agentx-source.json")) == original
    write_record(str(tmp_path / "retained" / "exported" / Path(uri).name), {"artifact_url": "github/artifact"})
    assert pending_result(plan["root"]) != uri
    assert read_record(uri) == original
    # Explicit delivery retry recreates the same original source identity.
    assert materialize(uri, tmp_path / "retry") == result


def test_reprocessing_preserves_bundles_referenced_by_earlier_completed_results(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    aggregate = artifacts / "agg.json"
    completed = []
    for value in [1, 2]:
        aggregate.write_text(json.dumps({"value": value}))
        uri, digest = retain_processed_artifacts(artifacts, tmp_path / "working.tar.zst", str(tmp_path / "retained"))
        result_uri = str(tmp_path / f"result-{value}.json")
        write_record(
            result_uri,
            {
                "source": {"kind": "fixture"},
                "points": [{"receipt": {"status": "valid", "artifacts_uri": uri, "artifacts_sha256": digest}}],
            },
        )
        completed.append(result_uri)
    # Recovery can regenerate a point after its previous descriptor was delivered.
    for value, result_uri in enumerate(completed, 1):
        output = tmp_path / f"delivery-{value}"
        materialize(result_uri, output)
        assert json.loads((output / "agg.json").read_text()) == {"value": value}


@pytest.fixture
def exported_window(tmp_path, saved_batch):
    batch, _ = saved_batch
    manifest = batch["points"][0]["manifest"]
    (tmp_path / "attempt.json").write_text(json.dumps(manifest | {"status": "succeeded", "iris_attempt_uid": "winner"}))
    artifacts = tmp_path / "aiperf_artifacts"
    artifacts.mkdir()
    duration = json.loads(SERIES_PATH.read_text())["duration"]
    export = {
        "metadata": {
            "submission_valid": True,
            "metric_duration_coverage": [
                {
                    "phase_name": "profiling",
                    "expected_duration_seconds": duration,
                    "required_ratio": 0.95,
                    "ttft_ratio": 0.99,
                    "inter_token_latency_ratio": 0.98,
                }
            ],
        },
        "was_cancelled": False,
        "start_time": "2026-01-01T00:00:00Z",
        "end_time": "2026-01-01T00:15:00Z",
        "benchmark_duration": {"avg": 895.0},
    }
    (artifacts / "profile_export_aiperf.json").write_text(json.dumps(export))
    requests = [
        {"metadata": {"benchmark_phase": "warmup"}, "error": {"type": "empty-one-token-response"}},
        {"metadata": {"benchmark_phase": "profiling"}, "error": None},
    ]
    (artifacts / "profile_export.jsonl").write_text("".join(json.dumps(row) + "\n" for row in requests))
    return tmp_path, manifest, export


def test_complete_window_uses_phase_coverage_and_excludes_warmup_errors(exported_window):
    directory, manifest, export = exported_window
    result = validate_window(directory, manifest, "winner")
    assert result["profiling_requests"] == 1
    assert result["coverage"] == export["metadata"]["metric_duration_coverage"][0]
    assert result["submission_valid"] is True


@pytest.mark.parametrize("cancelled,valid", [(True, True), (False, False)])
def test_interrupted_or_scenario_invalid_window_is_not_publishable(exported_window, cancelled, valid):
    directory, manifest, export = exported_window
    export["was_cancelled"] = cancelled
    export["metadata"]["submission_valid"] = valid
    (directory / "aiperf_artifacts" / "profile_export_aiperf.json").write_text(json.dumps(export))
    with pytest.raises(ValueError, match="complete window"):
        validate_window(directory, manifest, "winner")


def test_child_failure_reaches_iris_exit_status_and_both_log_destinations(tmp_path, capfd):
    output = tmp_path / "attempt.log"
    with logged_process(
        [sys.executable, "-c", 'print("attempt diagnostic", flush=True); raise SystemExit(17)'], dict(os.environ), output
    ) as process:
        assert process.wait(timeout=10) == 17
    assert output.read_text() == "attempt diagnostic\n"
    assert "attempt diagnostic" in capfd.readouterr().out


def test_scheduling_failure_retains_diagnostics_without_waiting_for_a_nonexistent_archive(saved_batch, tmp_path):
    batch, _ = saved_batch
    point = batch["points"][0]
    status = job_status(JobName.from_string(batch["job_id"] + "/" + point["name"]), JobState.FAILED)
    receipt = collect_terminal(status, point["manifest"], str(tmp_path / "retained"), tmp_path, tmp_path, 1001)
    assert receipt["status"] == "failed"
    assert receipt["attempt_uid"] is None
    assert read_record(receipt["iris_status_uri"])["state"] == "failed"


def test_error_ratio_is_checked_separately_from_agentx_scenario_validity(exported_window):
    directory, manifest, _ = exported_window
    records = directory / "aiperf_artifacts" / "profile_export.jsonl"
    success = {"metadata": {"benchmark_phase": "profiling"}, "error": None}
    failure = success | {"error": {"type": "InvalidInferenceResultError"}}
    records.write_text("".join(json.dumps(row) + "\n" for row in [success] * 99 + [failure]))
    result = validate_window(directory, manifest, "winner")
    assert result["profiling_errors"] == 1
    assert result["profiling_error_ratio"] == 0.01
    with records.open("a") as stream:
        stream.write(json.dumps(failure) + "\n")
    with pytest.raises(ValueError, match="error ratio"):
        validate_window(directory, manifest, "winner")


def test_iris_archive_can_be_read_without_host_zstd_or_tar(tmp_path, monkeypatch):
    # Iris task images provide the Python codec, not the zstd CLI used by Actions.
    content = b'{"iris_attempt_uid":"original-attempt","status":"failed"}\n'
    wire = io.BytesIO()
    with tarfile.open(fileobj=wire, mode="w") as archive:
        entry = tarfile.TarInfo("point/attempt.json")
        entry.size = len(content)
        archive.addfile(entry, io.BytesIO(content))
    incoming = tmp_path / "outputs.tar.zst"
    incoming.write_bytes(zstandard.compress(wire.getvalue()))
    monkeypatch.setenv("PATH", str(tmp_path / "no-external-tools"))
    unpack(incoming, tmp_path / "retained")
    assert (tmp_path / "retained" / "point" / "attempt.json").read_bytes() == content
