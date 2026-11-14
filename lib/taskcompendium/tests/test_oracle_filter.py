# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Oracle admission rejects missing execution evidence and stale converted states."""

import json
from pathlib import Path

import pytest
from finestore.rl.curation import ContentEvidenceReference
from pydantic import ValidationError

from taskcompendium.grading_environment import GradingEnvironmentMode
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import ResourceGroups, Source
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import RawRow
from taskcompendium.pipeline.oracle_filter import (
    MechanicalOracleTaskReceipt,
    OracleExclusionReason,
    OracleFilterReceipt,
    OracleTrialReceipt,
    ResourceCalibrationReceipt,
    VerifierResourceProfile,
    load_oracle_filter,
    next_trial_requirements,
    require_oracle_filter,
)
from taskcompendium.pipeline.resource_calibration import FailureSignal, ResourceDimension, ResourceUsage
from taskcompendium.runtime.resources import inline_resource

from .oracle_filter_fixtures import oracle_receipt
from .pipeline_stages import SOURCE_FILES
from .test_source_processing import apple_rows, convert_script_graded


def task():
    return convert_script_graded(
        RawRow(
            "oracle",
            Source(dataset=SOURCE_FILES.dataset, revision="1", row="0", importer_revision="1"),
            apple_rows(1)[0],
        ),
        ConversionContext({}, None),
    )


def test_oracle_filter_requires_scored_success_instead_of_reward_only_receipts():
    current = task()
    with pytest.raises(ValueError, match="full-population Harbor oracle filter"):
        require_oracle_filter(current, None)
    assert require_oracle_filter(current, oracle_receipt([current]))
    assert not require_oracle_filter(current, oracle_receipt([current], failed=frozenset({current.id})))
    document = oracle_receipt([current]).model_dump(mode="json")
    document["tasks"][current.id]["trials"][0]["verdict"] = {"status": "infra_error", "reward": 1}
    document["tasks"][current.id]["calibration"]["attempts"][0]["verdict"] = {"status": "infra_error", "reward": 1}
    document["tasks"][current.id]["exclusion_reason"] = OracleExclusionReason.INFRASTRUCTURE.value
    with pytest.raises(ValueError, match="infrastructure retry budget is incomplete"):
        OracleFilterReceipt.model_validate(document)
    document["tasks"][current.id]["calibration"]["attempts"] *= 3
    assert not require_oracle_filter(current, OracleFilterReceipt.model_validate(document))


def test_changed_task_or_incomplete_population_cannot_reuse_an_oracle_receipt():
    current = task()
    receipt = oracle_receipt([current])
    changed = current.model_copy(update={"labels": {"changed": "after trial"}})
    with pytest.raises(ValueError, match="changed after oracle filtering"):
        require_oracle_filter(changed, receipt)
    document = receipt.model_dump(mode="json")
    document["input_rows"] += 1
    with pytest.raises(ValidationError, match="full input population"):
        OracleFilterReceipt.model_validate(document)


@pytest.mark.parametrize(
    "dimension,log,oom_kills,expected",
    [
        (ResourceDimension.MEMORY, "", 1, {"memory_mb": 4096}),
        (ResourceDimension.STORAGE, "ENOSPC", 0, {"storage_mb": 4096}),
    ],
)
def test_fixed_separate_grader_failures_do_not_increase_actor_resources(dimension, log, oom_kills, expected):
    current = task()
    document = oracle_receipt([current]).tasks[current.id].calibration.attempts[0].model_dump(mode="json")
    document["result"]["exception_info"] = {"exception_type": "VerifierRuntimeError"}
    document["result"]["verifier_execution"] = {"started_at": "2026-11-12T13:00:00Z"}
    document["result_sha256"] = canonical_sha256(document["result"])
    document["failure_log"] = log
    document["verifier_resource_usage"] = {"oom_kills": oom_kills}
    trial = OracleTrialReceipt.model_validate(document)
    separate = VerifierResourceProfile(mode=GradingEnvironmentMode.SEPARATE, fixed={dimension: 2048})
    assert next_trial_requirements(trial, {}, separate) is None
    calibration = ResourceCalibrationReceipt(
        declared_requirements=None,
        verifier=separate,
        oracle_kind="original",
        attempts=(trial,),
    )
    assert not calibration.passed
    shared = VerifierResourceProfile(mode=GradingEnvironmentMode.SHARED, fixed={dimension: 2048})
    assert next_trial_requirements(trial, {}, shared) == current.execution_requirements.model_copy(update=expected)


def test_calibration_only_oracle_cannot_supply_admission_evidence():
    current = task().model_copy(update={"resources": ResourceGroups()})
    with pytest.raises(ValueError, match="full-population Harbor oracle filter"):
        require_oracle_filter(current, None)
    receipt = oracle_receipt([current], executability=False)
    assert receipt.tasks[current.id].trials == ()
    assert receipt.tasks[current.id].calibration.passed
    assert not receipt.tasks[current.id].passed
    assert not require_oracle_filter(current, receipt)
    document = receipt.model_dump(mode="json")
    document["tasks"][current.id]["calibration"]["oracle_kind"] = "original"
    with pytest.raises(ValidationError, match=r"calibration-only|Calibration-only"):
        OracleFilterReceipt.model_validate(document)


@pytest.mark.parametrize("reward", [0, 0.25, 1])
def test_no_oracle_strong_model_proof_records_reward_without_using_it_as_executability(reward):
    current = task().model_copy(update={"resources": ResourceGroups()})
    document = oracle_receipt([current]).model_dump(mode="json")
    entry = document["tasks"][current.id]
    for trial in (entry["calibration"]["attempts"][0], entry["executability_run"]["trial"]):
        trial["verdict"]["reward"] = reward
        trial["result"]["verifier_result"]["rewards"]["reward"] = reward
        trial["result_sha256"] = canonical_sha256(trial["result"])
    receipt = OracleFilterReceipt.model_validate(document)
    assert require_oracle_filter(current, receipt)
    assert not receipt.tasks[current.id].calibration.passed


@pytest.mark.parametrize("changed", ["model", "temperature", "turn_limit", "provider_failure"])
def test_no_oracle_control_cannot_admit_changed_actor_or_provider_failure(changed):
    current = task().model_copy(update={"resources": ResourceGroups()})
    document = oracle_receipt([current]).model_dump(mode="json")
    entry = document["tasks"][current.id]
    for trial in (entry["calibration"]["attempts"][0], entry["executability_run"]["trial"]):
        agent = trial["result"]["config"]["agent"]
        if changed == "model":
            agent["model_name"] = "other-model"
        elif changed == "temperature":
            agent["kwargs"]["request_parameters"]["temperature"] = 0
        elif changed == "turn_limit":
            agent["kwargs"]["max_turns"] = 30
        else:
            trial["result"]["exception_info"] = {"exception_type": "SourceContractError"}
        trial["result_sha256"] = canonical_sha256(trial["result"])
    entry["exclusion_reason"] = OracleExclusionReason.EXECUTABILITY_RUN_FAILED.value
    assert not require_oracle_filter(current, OracleFilterReceipt.model_validate(document))


@pytest.mark.parametrize("original_oracle", [True, False])
def test_historical_v2_receipt_is_readable_but_cannot_establish_new_admission(original_oracle):
    current = task()
    if not original_oracle:
        current = current.model_copy(update={"resources": ResourceGroups()})
    document = oracle_receipt([current]).model_dump(mode="json")
    document["version"] = 2
    for entry in document["tasks"].values():
        entry.pop("outcome")
        entry.pop("executability_run")
    receipt = OracleFilterReceipt.model_validate(document)
    with pytest.raises(ValueError, match=r"version 3.*archival"):
        require_oracle_filter(current, receipt)


@pytest.mark.parametrize("mismatch", ["actor_effective", "verifier_effective", "source_runtime_sha256", "missing"])
def test_declared_network_permission_requires_matching_production_observation(mismatch):
    current = task()
    current = current.model_copy(
        update={"resources": current.resources.model_copy(update={"worker": (inline_resource("input.txt", b"input"),)})}
    )
    runtime = {"task_machine": {"network": "allow"}, "verifier_machine": {"network": "deny"}}
    document = oracle_receipt([current]).model_dump(mode="json")
    entry = document["tasks"][current.id]
    entry["runtime_sha256"] = canonical_sha256(runtime)
    packet = {
        "source_runtime_sha256": canonical_sha256(runtime),
        "actor_required": True,
        "actor_declared": "allow",
        "verifier_declared": "deny",
        "actor_effective": "allow",
        "verifier_effective": "deny",
    }
    for trial in (entry["trials"][0], entry["calibration"]["attempts"][0]):
        trial["network_policy"] = dict(packet)
    assert require_oracle_filter(current, OracleFilterReceipt.model_validate(document), runtime)
    for trial in (entry["trials"][0], entry["calibration"]["attempts"][0]):
        if mismatch == "missing":
            trial["network_policy"] = None
        else:
            trial["network_policy"][mismatch] = {
                "actor_effective": "deny",
                "verifier_effective": "allow",
                "source_runtime_sha256": "0" * 64,
            }[mismatch]
    assert not require_oracle_filter(current, OracleFilterReceipt.model_validate(document), runtime)


def test_mechanical_exclusion_keeps_uncalibrated_task_and_checks_both_input_identities():
    current = task().model_copy(update={"execution_requirements": None})
    original = current.model_dump_json()
    runtime = {"task_machine": {"network": "allow"}}
    mechanical = MechanicalOracleTaskReceipt(
        task_sha256=canonical_sha256(current.model_dump(mode="json")),
        runtime_sha256=canonical_sha256(runtime),
        has_oracle=True,
        exclusion_reason=OracleExclusionReason.TOOL_PROVIDER_REQUIRED,
        evidence=(ContentEvidenceReference(uri="s3://fixtures/capability-proof.json", sha256="c" * 64),),
    )
    receipt = OracleFilterReceipt(
        recipe_identity=None,
        input_cohort_sha256="a" * 64,
        input_rows=1,
        conversion_rejections=0,
        tasks={current.id: mechanical},
    )
    assert not require_oracle_filter(current, receipt, runtime)
    assert current.model_dump_json() == original
    assert receipt.uniform_requirements is None
    with pytest.raises(ValueError, match="Runtime changed"):
        require_oracle_filter(current, receipt, {"task_machine": {"network": "deny"}})
    with pytest.raises(ValueError, match="Task changed"):
        require_oracle_filter(current.model_copy(update={"labels": {"changed": "input"}}), receipt, runtime)
    document = mechanical.model_dump(mode="json")
    document["evidence"] = []
    with pytest.raises(ValidationError, match="at least 1"):
        MechanicalOracleTaskReceipt.model_validate(document)
    document["evidence"] = mechanical.model_dump(mode="json")["evidence"]
    document["calibration"] = {"invented": "trial"}
    with pytest.raises(ValidationError, match="Extra inputs"):
        MechanicalOracleTaskReceipt.model_validate(document)


@pytest.mark.parametrize("exception_type", ["DiskLimitExceeded", "SandboxDiskFullError", "AgentDiskFullError"])
def test_disk_limit_evidence_doubles_disk_even_with_an_infrastructure_verdict(exception_type):
    current = task()
    document = oracle_receipt([current]).tasks[current.id].calibration.attempts[0].model_dump(mode="json")
    document["result"]["exception_info"] = {"exception_type": exception_type}
    document["result_sha256"] = canonical_sha256(document["result"])
    document["verdict"] = {"status": "infra_error"}
    trial = OracleTrialReceipt.model_validate(document)
    assert trial.failure_signal is FailureSignal.DISK_FULL
    assert next_trial_requirements(trial, {}, VerifierResourceProfile(mode=GradingEnvironmentMode.SHARED)) == (
        current.execution_requirements.model_copy(update={"storage_mb": current.execution_requirements.storage_mb * 2})
    )


@pytest.mark.parametrize(
    "usage,exception,expected",
    [
        (ResourceUsage(oom_kills=1), None, FailureSignal.OOM),
        (ResourceUsage(), {"exception_type": "VerifierTimeoutError"}, FailureSignal.VERIFIER_TIMEOUT),
        (ResourceUsage(), {"exception_type": "VerifierRuntimeError"}, FailureSignal.INFRASTRUCTURE),
    ],
)
def test_structured_infrastructure_verdict_preserves_measured_resource_failures(usage, exception, expected):
    current = task()
    trial = oracle_receipt([current]).tasks[current.id].calibration.attempts[0]
    result = {**trial.result, "exception_info": exception}
    trial = trial.model_copy(
        update={
            "result": result,
            "result_sha256": canonical_sha256(result),
            "verdict": {"status": "infra_error", "reward": 0, "detail": {}},
            "verifier_resource_usage": usage,
        }
    )
    assert trial.failure_signal is expected


@pytest.mark.parametrize("failure", ["daytona_pre_oracle_counters", "local_disk_probe_loop"])
@pytest.mark.parametrize("stage", ["host", "source_command", "after_source"])
def test_pre_oracle_host_failures_retry_without_reclassifying_source_failures(failure, stage):
    frozen = json.loads((Path(__file__).parent / "fixtures/host_pre_oracle_failures.json").read_text())[failure]
    current = task()
    document = oracle_receipt([current]).tasks[current.id].calibration.attempts[0].model_dump(mode="json")
    result = document["result"]
    result["config"]["environment"]["type"] = frozen["backend"]
    result["exception_info"] = frozen["exception_info"]
    result["verifier_result"] = None
    document["verdict"] = None
    document["oracle_exit_code"] = 0 if stage == "after_source" else None
    if stage == "source_command":
        result["exception_info"]["exception_traceback"] = "Traceback: source command raised this message"
    document["result_sha256"] = canonical_sha256(result)
    trial = OracleTrialReceipt.model_validate(document)
    assert trial.failure_signal is (FailureSignal.INFRASTRUCTURE if stage == "host" else FailureSignal.NON_RESOURCE)
    if stage == "host":
        historical = ResourceCalibrationReceipt(
            declared_requirements=current.execution_requirements,
            oracle_kind="original",
            verifier=VerifierResourceProfile(mode=GradingEnvironmentMode.SHARED),
            attempts=(trial,),
        )
        assert not ResourceCalibrationReceipt.model_validate_json(historical.model_dump_json()).passed


def test_public_filter_load_rejects_omitted_measured_verifier_allocation(tmp_path):
    current = task()
    document = oracle_receipt([current]).model_dump(mode="json")
    entry = document["tasks"][current.id]
    entry["calibration"]["verifier"]["mode"] = "separate"
    for trial in (entry["trials"][0], entry["calibration"]["attempts"][0]):
        trial["result"]["config"]["task"] = {
            "verifier_execution_requirements": {
                **trial["execution_requirements"],
                "memory_mb": trial["execution_requirements"]["memory_mb"] * 2,
            }
        }
        trial["result_sha256"] = canonical_sha256(trial["result"])
        trial.pop("verifier_execution_requirements", None)
    path = tmp_path / "missing-verifier-allocation.json"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="Verifier allocation differs from the actual production Trial config"):
        load_oracle_filter(str(path))
