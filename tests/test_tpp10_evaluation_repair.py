# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from iris.client.client import IrisClient
from iris.cluster.client.job_info import JobInfo
from iris.resources.state import JobState
from marin.execution.artifact import read_record
from marin.execution.fingerprint import fingerprint_hash
from marin.execution.step_status import STATUS_SUCCESS, StatusFile

from experiments.domain_phase_mix import repair_tpp10_evaluation as repair


@pytest.fixture
def recovery_case(tmp_path, monkeypatch):
    monkeypatch.setattr(repair.experiment, "PREFIX", str(tmp_path / "receipts"))
    output = tmp_path / "training"
    checkpoint = output / "checkpoints" / "step-9"
    checkpoint.mkdir(parents=True)
    (checkpoint / "metadata.json").write_text(json.dumps({"step": 9, "is_temporary": False}))
    payload = '{"frozen":true}'
    fingerprint = fingerprint_hash(payload)
    request = {
        "run_name": "survey",
        "output_path": str(output),
        "total_steps": 10,
        "fingerprint": fingerprint,
        "domain": "wikipedia",
    }
    training_plan = {
        "design_sha256": "design",
        "runtime_versions": {},
        "code_sha256": {},
        "extension_code_sha256": {},
        "plan_sha256": "original",
    }
    (output / "verified_runtime.json").write_text(
        json.dumps({"design_sha256": "design", "versions": {}, "code_sha256": {}})
    )
    (output / "domain_runtime.json").write_text(json.dumps({"domain": "wikipedia", "extension_code_sha256": {}}))
    metrics = {
        repair.experiment.PRIMARY_METRIC: 1.0,
        **{f"eval/uncheatable_eval/{name}/bpb": 2.0 for name in repair.original_eval.COMPONENTS},
        **{f"eval/{tag}/loss": 3.0 for tag in repair.SURVEY_TAGS},
    }
    (output / "checkpoints" / "eval_metrics.jsonl").write_text(json.dumps({"step": 9, **metrics}) + "\n")
    plan = {
        "repair_sha256": "repair",
        "source_coordinator": "/test/parent",
        "artifacts": {
            "plan": training_plan,
            "rows": [
                {
                    "request": request,
                    "iris_job": "/test/parent/survey",
                    "record": {
                        "name": "survey",
                        "output_path": str(output),
                        "version": "2026.09.11",
                        "fingerprint": fingerprint,
                        "fingerprint_payload": payload,
                    },
                }
            ],
        },
    }
    client = MagicMock(spec=IrisClient)
    client.job_state.return_value = JobState.SUCCEEDED
    api = MagicMock()
    api.run.return_value.state = "finished"
    api.run.return_value.summary = metrics.copy()
    monkeypatch.setattr(repair.wandb, "Api", lambda **_: api)
    monkeypatch.setattr(
        repair, "get_job_info", lambda: JobInfo(task_id=repair.JobName.from_wire("/test/repair").task(0))
    )
    return plan, client, api, output


def test_recovery_finalizes_verified_training_and_is_idempotent(recovery_case):
    plan, client, _, output = recovery_case
    before = (output / "checkpoints" / "eval_metrics.jsonl").read_bytes()
    result = repair.recover_records(plan, client)
    assert result["survey"]["state"] == "verified"
    assert StatusFile(str(output), "test").status == STATUS_SUCCESS
    record = read_record(str(output))
    assert record.fingerprint == plan["artifacts"]["rows"][0]["request"]["fingerprint"]
    first = (output / ".artifact.json").read_bytes()
    assert repair.recover_records(plan, client)["survey"]["state"] == "verified"
    assert (output / ".artifact.json").read_bytes() == first
    assert (output / "checkpoints" / "eval_metrics.jsonl").read_bytes() == before


def test_recovery_leaves_live_training_unmodified(recovery_case):
    plan, client, _, output = recovery_case
    client.job_state.return_value = JobState.RUNNING
    assert repair.recover_records(plan, client)["survey"]["state"] == "running"
    assert read_record(str(output)) is None
    assert StatusFile(str(output), "test").status is None


def test_recovery_rejects_stale_wandb_before_writing_completion(recovery_case):
    plan, client, api, output = recovery_case
    api.run.return_value.summary[repair.experiment.PRIMARY_METRIC] = 3.0
    status = repair.recover_records(plan, client)["survey"]
    assert status["state"] == "wandb_mismatch"
    assert status["metrics"][repair.experiment.PRIMARY_METRIC] == {"wandb": "3.0", "saved": 1.0}
    assert read_record(str(output)) is None
    assert StatusFile(str(output), "test").status is None


def test_recovery_leaves_completion_to_live_parent(recovery_case):
    plan, client, _, output = recovery_case
    client.job_state.side_effect = lambda name: (
        JobState.RUNNING if name.to_wire() == "/test/parent" else JobState.SUCCEEDED
    )
    assert repair.recover_records(plan, client)["survey"]["state"] == "verified_output"
    assert read_record(str(output)) is None
    assert StatusFile(str(output), "test").status is None


def test_recovery_requires_saved_token_losses(recovery_case):
    plan, client, _, output = recovery_case
    path = output / "checkpoints" / "eval_metrics.jsonl"
    metrics = json.loads(path.read_text())
    del metrics[f"eval/{repair.SURVEY_TAGS[-1]}/loss"]
    path.write_text(json.dumps(metrics) + "\n")
    with pytest.raises(ValueError, match="Missing final metrics"):
        repair.recover_records(plan, client)
    assert read_record(str(output)) is None


def test_coordinator_retirement_preserves_unfinished_descendants(recovery_case):
    plan, client, _, _ = recovery_case
    statuses = {"survey": {"state": "verified_output"}}
    client.job_state.return_value = JobState.RUNNING
    child = SimpleNamespace(job_id=repair.JobName.from_wire("/test/parent/survey"), state=JobState.RUNNING)
    client.list_jobs.return_value = [child]
    repair.retire_finished_coordinator(plan, statuses, client)
    client.cancel_job.assert_not_called()
    child.state = JobState.SUCCEEDED
    repair.retire_finished_coordinator(plan, statuses, client)
    client.cancel_job.assert_called_once_with(repair.JobName.from_wire("/test/parent"))


def test_corrected_metrics_refuses_a_partial_component_set():
    counts = {tag: {"tokens": 10, "bytes": 20} for tag in repair.SURVEY_TAGS}
    metrics = {f"eval/{tag}/loss": 1.0 for tag in repair.SURVEY_TAGS[:-1]}
    with pytest.raises(KeyError):
        repair.corrected_metrics(metrics, counts)


def test_retirement_cannot_cancel_the_repair_itself(recovery_case, monkeypatch):
    plan, client, _, _ = recovery_case
    monkeypatch.setattr(
        repair, "get_job_info", lambda: JobInfo(task_id=repair.JobName.from_wire("/test/parent/repair").task(0))
    )
    with pytest.raises(ValueError, match="outside the original coordinator"):
        repair.retire_finished_coordinator(plan, {"survey": {"state": "verified_output"}}, client)
    client.cancel_job.assert_not_called()


def test_nonfinite_wandb_value_is_reported_without_mutating_training(recovery_case):
    plan, client, api, output = recovery_case
    api.run.return_value.summary[repair.experiment.PRIMARY_METRIC] = float("nan")
    status = repair.recover_records(plan, client)["survey"]
    assert status["state"] == "wandb_mismatch"
    assert status["metrics"][repair.experiment.PRIMARY_METRIC]["wandb"] == "nan"
    assert read_record(str(output)) is None


def test_corrected_results_identify_omissions_and_unfinalized_records(recovery_case):
    plan, _, _, _ = recovery_case
    plan["audits"] = []
    counts = {tag: {"tokens": 10, "bytes": 20} for tag in repair.SURVEY_TAGS}
    result = repair.collect_corrected(plan, counts, {"survey": {"state": "running"}})
    assert result["complete"] is False
    assert result["rows"] == []
    assert result["omitted"] == {"survey": {"state": "running"}}
    result = repair.collect_corrected(plan, counts, {"survey": {"state": "verified_output"}})
    assert len(result["rows"]) == result["expected_rows"] == 1
    assert result["complete"] is False
    assert result["artifact_recovery_pending"] == ["survey"]
    assert result["rows"][0]["completion_record_verified"] is False


def test_recovery_rejects_changed_checkpoint_before_writing_completion(recovery_case):
    plan, client, _, output = recovery_case
    (output / "checkpoints" / "step-9" / "metadata.json").write_text(json.dumps({"step": 8, "is_temporary": False}))
    with pytest.raises(ValueError, match="final permanent checkpoint"):
        repair.recover_records(plan, client)
    assert read_record(str(output)) is None


def test_recovery_does_not_take_an_active_artifact_lease(recovery_case):
    plan, client, _, output = recovery_case
    status = StatusFile(str(output), "original-coordinator")
    assert status.try_acquire_lock()
    try:
        assert repair.recover_records(plan, client)["survey"]["state"] == "active_lease"
        assert read_record(str(output)) is None
        assert status.active_lock_holder() == "original-coordinator"
    finally:
        status.release_lock()


def test_recovery_rejects_conflicting_final_evaluations(recovery_case):
    plan, client, _, output = recovery_case
    with (output / "checkpoints" / "eval_metrics.jsonl").open("a") as handle:
        handle.write(json.dumps({"step": 9, repair.experiment.PRIMARY_METRIC: 99.0}) + "\n")
    with pytest.raises(ValueError, match="conflicting final metric"):
        repair.recover_records(plan, client)
    assert read_record(str(output)) is None
