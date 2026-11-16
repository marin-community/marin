# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recorded grades survive approved teardown failures without rewriting raw Results."""

import hashlib
import io
import json
import tarfile

import pytest
from finestore.rl.curation import ContentEvidenceReference
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import ExecutionRequirements
from taskcompendium.pipeline.oracle_filter import (
    OracleExclusionReason,
    OracleTrialReceipt,
    PostGradeTeardownEvidence,
    ResourceCalibrationReceipt,
    oracle_failure_reason,
)

from experiments.post_training.task_curation.post_grade_teardowns import (
    bind_post_grade_teardowns,
    require_post_grade_teardown,
)


def archived_case(tmp_path, reward, mutation=None):
    root = "postgrade/trial"
    verdict = {"status": "scored", "reward": reward, "detail": {"exit_code": 0}}
    cleanup = "in _stop_environment\nharbor.trial.errors.EnvironmentStopError: Environment teardown failed\n"
    result = {
        "id": "recorded-original",
        "task_name": "scale-swe-verified-fixture",
        "task_checksum": "a" * 64,
        "trial_uri": f"file:///tmp/{root}",
        "agent_info": {"name": "oracle"},
        "config": {
            "task": {},
            "agent": {"name": "oracle"},
            "environment": {"type": "daytona"},
            "resolved_timeouts": {"agent": 900, "verifier": 600},
        },
        "verifier_result": None,
        "completion": None,
        "exception_info": {
            "exception_type": "EnvironmentStopError",
            "exception_message": "Environment teardown failed",
            "exception_traceback": cleanup,
        },
        "agent_execution": {"started_at": "2026-11-16T01:00:00Z", "finished_at": "2026-11-16T01:00:01Z"},
        "verifier": {"started_at": "2026-11-16T01:00:02Z", "finished_at": "2026-11-16T01:00:03Z"},
        "finished_at": "2026-11-16T01:00:04Z",
    }
    if mutation == "late_phase":
        result["verifier"]["finished_at"] = "2026-11-16T01:00:05Z"
    if mutation == "setup_failure":
        result["exception_info"]["exception_type"] = "EnvironmentSetupError"
    files = {
        "result.json": json.dumps(result).encode(),
        "config.json": json.dumps(result["config"]).encode(),
        "verifier/verdict.json": json.dumps(verdict).encode(),
        "verifier/reward.txt": str(1 - reward if mutation == "reward_conflict" else reward).encode(),
        "agent/exit-code.txt": b"1" if mutation == "nonzero_command" else b"0",
        "exception.txt": cleanup.encode(),
    }
    archive = tmp_path / "trial.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        for name, data in files.items():
            member = tarfile.TarInfo(f"{root}/{name}")
            member.size = len(data)
            output.addfile(member, io.BytesIO(data))
    reference = ContentEvidenceReference(uri=str(archive), sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    digest = canonical_sha256(result)
    cases = {
        digest: {"task_id": result["task_name"], "reward": reward},
        **{str(i) * 64: {"task_id": f"scale-swe-verified-other-{i}", "reward": 1} for i in range(3)},
    }
    decision = tmp_path / "decision.json"
    decision.write_text(json.dumps({"authority_message": "7cf3d8d1fd2b9bb4c9998a4b758746d4", "cases": cases}))

    def member(name):
        return ContentEvidenceReference(
            uri=f"{reference.uri}#{root}/{name}", sha256=hashlib.sha256(files[name]).hexdigest()
        )

    evidence = PostGradeTeardownEvidence(
        result_sha256=digest,
        archive=reference,
        result_artifact=member("result.json"),
        verdict_artifact=member("verifier/verdict.json"),
        reward_artifact=member("verifier/reward.txt"),
        oracle_exit_artifact=member("agent/exit-code.txt"),
        teardown_exception_artifact=member("exception.txt"),
        decision=ContentEvidenceReference(uri=str(decision), sha256=hashlib.sha256(decision.read_bytes()).hexdigest()),
    )
    trial = OracleTrialReceipt(
        path=f"/tmp/{root}/result.json",
        result=result,
        verdict=verdict,
        oracle_exit_code=0,
        result_sha256=digest,
        execution_requirements=ExecutionRequirements(
            cpus=1, memory_mb=2048, storage_mb=2048, agent_timeout_sec=900, verifier_timeout_sec=600
        ),
    )
    return trial, evidence


@pytest.mark.parametrize("reward", [0.0, 1.0])
def test_original_recorded_grade_is_bound_without_replay_or_raw_result_edit(tmp_path, reward):
    trial, evidence = archived_case(tmp_path, reward)
    assert not trial.scored
    require_post_grade_teardown(evidence, trial)
    calibration = ResourceCalibrationReceipt(
        declared_requirements=trial.execution_requirements,
        verifier={"mode": "shared"},
        oracle_kind="original",
        attempts=(trial,),
    )
    original = tmp_path / "original.json"
    original.write_text(json.dumps({"source-task": {"calibration": calibration.model_dump(mode="json")}}))
    original_bytes = original.read_bytes()
    output = tmp_path / "bound.json"
    bind_post_grade_teardowns(original, {trial.result_sha256: evidence}, output)
    bound = ResourceCalibrationReceipt.model_validate(json.loads(output.read_text())["source-task"]["calibration"])
    observed = bound.attempts[-1]
    assert original.read_bytes() == original_bytes
    assert observed.result == trial.result and observed.verdict == trial.verdict
    assert observed.scored and bound.passed is (reward == 1)
    assert observed.result["completion"] is None and observed.result["verifier_result"] is None
    if reward == 0:
        assert oracle_failure_reason(observed, None) is OracleExclusionReason.ORACLE_FAILED
    assert len(bound.attempts) == 1


@pytest.mark.parametrize("mutation", ["late_phase", "reward_conflict", "nonzero_command", "setup_failure"])
def test_completed_grade_proof_rejects_another_phase_score_or_original_exit(tmp_path, mutation):
    trial, evidence = archived_case(tmp_path, 1.0, mutation)
    with pytest.raises(ValueError, match=r"Recorded|Archived|recorded"):
        require_post_grade_teardown(evidence, trial)
