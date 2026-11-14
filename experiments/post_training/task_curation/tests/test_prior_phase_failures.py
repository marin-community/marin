# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A preserved source deadline survives a later production cleanup cancellation."""

import json
from pathlib import Path

import pytest
from taskcompendium.grading_environment import GradingEnvironmentMode
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import ExecutionRequirements
from taskcompendium.pipeline.oracle_filter import (
    InfrastructureRetryApproval,
    OracleTrialReceipt,
    ResourceCalibrationReceipt,
    VerifierResourceProfile,
    require_infrastructure_retry_parent,
)
from taskcompendium.pipeline.resource_calibration import FailureSignal

from experiments.post_training.task_curation.oracle_filter import main, trial_result_paths
from experiments.post_training.task_curation.prior_phase_failures import bind_prior_phase_failures

from .prior_phase_fixtures import preserved_phase_failure


@pytest.fixture
def phase_case(tmp_path):
    original = json.loads((Path(__file__).parent / "fixtures/prior-phase-timeout.json").read_text())
    trial = OracleTrialReceipt(
        path=original["result"]["trial_uri"].removeprefix("file://") + "/result.json",
        result=original["result"],
        verdict=None,
        oracle_exit_code=None,
        result_sha256=canonical_sha256(original["result"]),
        execution_requirements=ExecutionRequirements(
            cpus=1, memory_mb=2048, storage_mb=2048, agent_timeout_sec=900, verifier_timeout_sec=600
        ),
    )
    calibration = ResourceCalibrationReceipt(
        declared_requirements=None,
        preserved_requirements={"agent_timeout_sec": 900, "verifier_timeout_sec": 600},
        oracle_kind="original",
        verifier=VerifierResourceProfile(mode=GradingEnvironmentMode.SHARED),
        attempts=(trial,),
    )
    task_map = tmp_path / "original-map.json"
    task_map.write_text(json.dumps({"smartie": {"calibration": calibration.model_dump(mode="json")}}))
    return original, task_map, trial


def test_observed_original_deadline_survives_cleanup_and_cannot_be_replayed(tmp_path, phase_case, monkeypatch):
    original, task_map, trial = phase_case
    old_bytes = task_map.read_bytes()
    evidence = preserved_phase_failure(
        tmp_path, original, trial, archive_root="/tmp/atlas-smartie-oracles-v3/calibrated"
    )
    output = tmp_path / "bound-map.json"
    evidence_path = tmp_path / "phase-failures.json"
    evidence_path.write_text(json.dumps([evidence.model_dump(mode="json")]))
    monkeypatch.setattr(
        "sys.argv",
        [
            "oracle_filter",
            "bind-phase-failures",
            "--task-map",
            str(task_map),
            "--evidence",
            str(evidence_path),
            "--output",
            str(output),
        ],
    )
    main()
    calibration = ResourceCalibrationReceipt.model_validate(json.loads(output.read_text())["smartie"]["calibration"])
    actual = calibration.attempts[0]
    assert actual.result == trial.result
    assert actual.failure_signal is FailureSignal.AGENT_TIMEOUT
    assert not calibration.passed and not actual.scored
    assert task_map.read_bytes() == old_bytes
    approval = InfrastructureRetryApproval(
        classification="our_infrastructure",
        parent_result_sha256=trial.result_sha256,
        failure_signature="cleanup cancellation",
        decision_ref="fixture-decision",
    )
    with pytest.raises(ValueError, match="cannot replay an observed source-declared phase deadline"):
        require_infrastructure_retry_parent(calibration, approval)


@pytest.mark.parametrize(
    ("changed", "message"),
    [
        ("another_trial", "archive config differs from the actual production Trial"),
        ("provider_failure", "does not prove the reported original phase deadline"),
        ("source_budget", "original source-declared deadline"),
        ("calibrator_budget", "original source-declared deadline"),
    ],
)
def test_unbound_archive_or_budget_cannot_convert_cleanup_into_source_failure(tmp_path, phase_case, changed, message):
    original, task_map, trial = phase_case
    evidence = preserved_phase_failure(
        tmp_path, original, trial, archive_root="/tmp/atlas-smartie-oracles-v3/calibrated", changed=changed
    )
    if changed in {"source_budget", "calibrator_budget"}:
        document = json.loads(task_map.read_text())
        if changed == "source_budget":
            document["smartie"]["calibration"]["preserved_requirements"]["agent_timeout_sec"] = 300
        else:
            del document["smartie"]["calibration"]["preserved_requirements"]["agent_timeout_sec"]
        task_map.write_text(json.dumps(document))
    output = tmp_path / "bound-map.json"
    with pytest.raises(ValueError, match=message):
        bind_prior_phase_failures(task_map, {trial.result_sha256: evidence}, output)
    assert not output.exists()


def test_lifecycle_result_remains_collectable_when_no_selected_result_exists(tmp_path, phase_case):
    original, _, _ = phase_case
    trial = tmp_path / "jobs/oracle/attempts/000"
    trial.mkdir(parents=True)
    (trial / "config.json").write_text(json.dumps(original["config"]))
    lifecycle = trial / "lifecycle-result.json"
    lifecycle.write_text(json.dumps(original["result"]))
    assert trial_result_paths(tmp_path / "jobs") == [lifecycle]
    selected = trial / "result.json"
    selected.write_bytes(lifecycle.read_bytes())
    assert trial_result_paths(tmp_path / "jobs") == [selected]
