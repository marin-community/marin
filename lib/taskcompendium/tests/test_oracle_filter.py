# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Oracle admission rejects missing execution evidence and stale converted states."""

import pytest
from pydantic import ValidationError

from taskcompendium.grading_environment import GradingEnvironmentMode
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import ResourceGroups, Source
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import RawRow
from taskcompendium.pipeline.oracle_filter import (
    OracleExclusionReason,
    OracleFilterReceipt,
    OracleTrialReceipt,
    ResourceCalibrationReceipt,
    VerifierResourceProfile,
    next_trial_requirements,
    require_oracle_filter,
)
from taskcompendium.pipeline.resource_calibration import FailureSignal, ResourceDimension

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
    receipt = oracle_receipt([current], no_oracle_liveness=False)
    assert receipt.tasks[current.id].trials == ()
    assert receipt.tasks[current.id].calibration.passed
    assert not receipt.tasks[current.id].passed
    assert not require_oracle_filter(current, receipt)
    document = receipt.model_dump(mode="json")
    document["tasks"][current.id]["calibration"]["oracle_kind"] = "original"
    with pytest.raises(ValidationError, match=r"calibration-only|Calibration-only"):
        OracleFilterReceipt.model_validate(document)


@pytest.mark.parametrize(
    "changes",
    [
        {},
        {"submission_kind": "candidate"},
        {"exit_code": 1},
        {"reward": 1},
        {"tests_executed": 0},
        {"task_sha256": "other-task"},
        {"runtime_sha256": "other-runtime"},
    ],
)
def test_no_oracle_admission_requires_exact_empty_submission_liveness(changes):
    current = task().model_copy(update={"resources": ResourceGroups()})
    document = oracle_receipt([current]).model_dump(mode="json")
    entry = document["tasks"][current.id]
    entry["no_oracle_liveness"].update(changes)
    if changes:
        entry["exclusion_reason"] = OracleExclusionReason.NO_ORACLE_LIVENESS.value
    receipt = OracleFilterReceipt.model_validate(document)
    assert require_oracle_filter(current, receipt) is (not changes)


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
