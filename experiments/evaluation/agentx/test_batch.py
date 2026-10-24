# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Control-plane interruption regressions; these tests allocate no hardware."""

import dataclasses
import json
import os
import sys
from types import SimpleNamespace

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from iris.client.workload import JobStatus, ResourceRequest
from iris.cluster.types import JobName
from iris.resources.state import FederationState, JobState
from rigging.timing import Timestamp

from experiments.evaluation.agentx.batch import (
    advance_batch,
    cancel_batch,
    create_batch,
    read_record,
    write_record,
)
from experiments.evaluation.agentx.collect import collect_terminal, validate_window
from experiments.evaluation.agentx.run import logged_process
from experiments.evaluation.agentx.submit import SERIES_PATH


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
    """The Iris RPC boundary: stable logical jobs survive a lost local response."""

    def __init__(self) -> None:
        self.jobs = {}
        self.submitted_manifests = {}
        self.lose_submit_response = False
        self.lose_cancel_response = False

    def job_status(self, job_id):
        if str(job_id) not in self.jobs:
            raise ConnectError(code=Code.NOT_FOUND, message="No such fixture job")
        return self.jobs[str(job_id)]

    def submit(self, **request):
        job_id = JobName.from_string(f"/fixture/{request['name']}")
        self.jobs.setdefault(str(job_id), job_status(job_id, JobState.PENDING))
        self.submitted_manifests.setdefault(
            str(job_id), json.loads(request["entrypoint"].workdir_files["agentx-point.json"])
        )
        if self.lose_submit_response:
            self.lose_submit_response = False
            raise TimeoutError("Fixture lost the response after Iris accepted the job")
        return SimpleNamespace(job_id=job_id)

    def cancel_job(self, job_id):
        self.jobs[str(job_id)] = dataclasses.replace(self.jobs[str(job_id)], state=JobState.KILLED)
        if self.lose_cancel_response:
            self.lose_cancel_response = False
            raise TimeoutError("Fixture lost the cancellation acknowledgement")

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
    batch = create_batch(source, ["h100"], "cw-rno2a", "batch", "fixture", str(tmp_path / "retained"), 1000, None)
    checkpoint = str(tmp_path / "checkpoint.json")
    write_record(checkpoint, batch)
    return batch, checkpoint


def no_collection(status, manifest, retention):
    raise AssertionError("An unfinished job has no publishable window")


def test_lost_submit_response_recovers_same_queued_job_and_frozen_inputs(saved_batch):
    batch, checkpoint = saved_batch
    controller = FakeController()
    controller.lose_submit_response = True
    with pytest.raises(TimeoutError):
        advance_batch(controller, batch, checkpoint, no_collection, 1001)
    recovered = read_record(checkpoint)
    advance_batch(controller, recovered, checkpoint, no_collection, 1000 + 30 * 3600)
    assert len(controller.jobs) == 1
    point = recovered["points"][0]
    assert point["state"] == "pending"
    assert point["manifest"] == controller.submitted_manifests[point["job_id"]]
    assert recovered["points"][1]["state"] == "unsubmitted"


def test_collection_timeout_reuses_retained_result_before_advancing(saved_batch, tmp_path):
    batch, checkpoint = saved_batch
    controller = FakeController()
    advance_batch(controller, batch, checkpoint, no_collection, 1001)
    point = batch["points"][0]
    controller.jobs[point["job_id"]] = dataclasses.replace(controller.jobs[point["job_id"]], state=JobState.SUCCEEDED)
    retained = tmp_path / "retained-receipt.json"

    def interrupted_collection(status, manifest, retention):
        write_record(str(retained), {"status": "valid", "sha256": "a" * 64, "attempt_uid": "completed-attempt"})
        raise TimeoutError("Fixture control process ended after retaining a completed result")

    with pytest.raises(TimeoutError):
        advance_batch(controller, batch, checkpoint, interrupted_collection, 1002)
    recovered = read_record(checkpoint)
    advance_batch(controller, recovered, checkpoint, lambda *_: read_record(str(retained)), 1003)
    assert len(controller.jobs) == 1
    assert recovered["points"][0]["receipt"]["attempt_uid"] == "completed-attempt"
    assert recovered["points"][0]["state"] == "collected"
    advance_batch(controller, recovered, checkpoint, no_collection, 1004)
    assert len(controller.jobs) == 2
    assert recovered["points"][1]["state"] == "submitted"


def test_cancellation_survives_lost_ack_without_starting_another_point(saved_batch):
    batch, checkpoint = saved_batch
    controller = FakeController()
    advance_batch(controller, batch, checkpoint, no_collection, 1001)
    controller.lose_cancel_response = True
    with pytest.raises(TimeoutError):
        cancel_batch(controller, batch, checkpoint)
    recovered = read_record(checkpoint)
    advance_batch(controller, recovered, checkpoint, no_collection, 1002)
    assert recovered["status"] == "cancelled"
    assert len(controller.jobs) == 1
    assert next(iter(controller.jobs.values())).state == JobState.KILLED
    assert recovered["points"][1]["state"] == "unsubmitted"


def test_native_success_with_invalid_export_stays_failed(saved_batch):
    batch, checkpoint = saved_batch
    controller = FakeController()
    advance_batch(controller, batch, checkpoint, no_collection, 1001)
    point = batch["points"][0]
    controller.jobs[point["job_id"]] = dataclasses.replace(controller.jobs[point["job_id"]], state=JobState.SUCCEEDED)
    advance_batch(controller, batch, checkpoint, lambda *_: {"status": "failed", "reason": "invalid window"}, 1002)
    assert read_record(checkpoint)["points"][0]["state"] == "failed"


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
    directory, manifest, _ = exported_window
    result = validate_window(directory, manifest, "winner")
    assert result["profiling_requests"] == 1
    assert result["coverage"]["expected_duration_seconds"] == 900
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
    status = job_status(JobName.from_string(point["job_id"]), JobState.FAILED)
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
