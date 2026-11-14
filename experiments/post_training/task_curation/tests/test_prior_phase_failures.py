# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A preserved source deadline survives a later production cleanup cancellation."""

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest
import zstandard
from finestore.rl.curation import ContentEvidenceReference
from taskcompendium.grading_environment import GradingEnvironmentMode
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import ExecutionRequirements
from taskcompendium.pipeline.oracle_filter import (
    InfrastructureRetryApproval,
    OracleTrialReceipt,
    PriorPhaseFailureReceipt,
    ResourceCalibrationReceipt,
    VerifierResourceProfile,
    require_infrastructure_retry_parent,
)
from taskcompendium.pipeline.resource_calibration import FailureSignal

from experiments.post_training.task_curation.oracle_filter import trial_result_paths
from experiments.post_training.task_curation.prior_phase_failures import bind_prior_phase_failures


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


def phase_evidence(tmp_path, original, trial, *, changed=None):
    member = original["archive_member"]
    config = json.loads(json.dumps(original["config"]))
    if changed == "another_trial":
        config["task"]["path"] = "/another-task"
    exception = original["exception_text"].encode()
    if changed == "provider_failure":
        exception = b"RuntimeError: Machine is closed\n"
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, data in ((member, exception), (str(Path(member).with_name("config.json")), json.dumps(config).encode())):
            item = tarfile.TarInfo(name)
            item.size = len(data)
            archive.addfile(item, io.BytesIO(data))
    archive_path = tmp_path / "preserved.tar.zst"
    payload = zstandard.ZstdCompressor().compress(buffer.getvalue())
    archive_path.write_bytes(payload)
    return PriorPhaseFailureReceipt(
        phase="agent",
        timeout_sec=900,
        result_sha256=trial.result_sha256,
        archive=ContentEvidenceReference(uri=str(archive_path), sha256=hashlib.sha256(payload).hexdigest()),
        exception=ContentEvidenceReference(uri=f"{archive_path}#{member}", sha256=hashlib.sha256(exception).hexdigest()),
        archive_member=member,
        archive_root="/tmp/atlas-smartie-oracles-v3/calibrated",
    )


def test_observed_original_deadline_survives_cleanup_and_cannot_be_replayed(tmp_path, phase_case):
    original, task_map, trial = phase_case
    old_bytes = task_map.read_bytes()
    evidence = phase_evidence(tmp_path, original, trial)
    output = tmp_path / "bound-map.json"
    bind_prior_phase_failures(task_map, {trial.result_sha256: evidence}, output)
    calibration = ResourceCalibrationReceipt.model_validate(json.loads(output.read_text())["smartie"]["calibration"])
    actual = calibration.attempts[0]
    assert actual.result == trial.result
    assert actual.result_sha256 == trial.result_sha256
    assert actual.result["exception_info"]["exception_type"] == "CancelledError"
    assert actual.failure_signal is FailureSignal.AGENT_TIMEOUT
    assert not calibration.passed and not actual.scored
    assert task_map.read_bytes() == old_bytes
    approval = InfrastructureRetryApproval(
        classification="our_infrastructure", parent_result_sha256=trial.result_sha256,
        failure_signature="cleanup cancellation", decision_ref="fixture-decision",
    )
    with pytest.raises(ValueError, match="cannot replay an observed source-declared phase deadline"):
        require_infrastructure_retry_parent(calibration, approval)


@pytest.mark.parametrize("changed", ["another_trial", "provider_failure", "source_budget"])
def test_unbound_archive_or_budget_cannot_convert_cleanup_into_source_failure(tmp_path, phase_case, changed):
    original, task_map, trial = phase_case
    evidence = phase_evidence(tmp_path, original, trial, changed=changed)
    if changed == "source_budget":
        document = json.loads(task_map.read_text())
        document["smartie"]["calibration"]["preserved_requirements"]["agent_timeout_sec"] = 300
        task_map.write_text(json.dumps(document))
    output = tmp_path / "bound-map.json"
    with pytest.raises(ValueError):
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
