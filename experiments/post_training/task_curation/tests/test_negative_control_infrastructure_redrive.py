# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Infrastructure NoP re-drives preserve the failed physical control and its budget."""

import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest
from finestore.rl.curation import ContentEvidenceReference
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import ExecutionRequirements
from taskcompendium.pipeline.oracle_filter import (
    NegativeControlApplicability,
    NegativeControlInfrastructureApproval,
    OracleExclusionReason,
    OracleNegativeControlReceipt,
    OracleTaskReceipt,
    OracleTrialReceipt,
    ResourceCalibrationReceipt,
)

from experiments.post_training.task_curation.negative_controls import (
    negative_control_from_archive,
    require_negative_control,
)

FIXTURES = Path(__file__).parent / "fixtures/negative_controls"
COLLECT_ERROR = "Cannot collect grading inputs: exit=1"


def physical_control(tmp_path, name, *, infrastructure, archive_failure=False, **redrive):
    """Derive a synthetic failure from the retained production preparation fixture."""
    index = json.loads((FIXTURES / "packet-runtime-fallback-index.json").read_text())["source"]
    archive = tmp_path / f"{name}.tar.gz"
    with tarfile.open(FIXTURES / "packet-runtime-fallback.tar.gz") as original:
        control = json.load(original.extractfile("nop-control/negative-control.json"))
        result = control["trial_result"]
        result["id"] = name
        result["trial_uri"] = f"file:///tmp/{name}/nop-control/trials/nop"
        control["trial_result_path"] = f"/tmp/{name}/nop-control/trials/nop/result.json"
        verdict = control["physical_verdict"]
        if infrastructure:
            result["exception_info"] = {
                "exception_type": "VerifierRuntimeError",
                "exception_message": COLLECT_ERROR,
            }
            result["verifier_result"] = None
            verdict = {"status": "infra_error", "reward": None, "detail": {"error": COLLECT_ERROR}}
            control["physical_verdict"] = verdict
            control["verifier_exit_code"] = 1
        if archive_failure:
            result["exception_info"] = {
                "exception_type": "DownloadVerifierDirError",
                "exception_message": "Failed to download verifier directory from environment",
                "exception_traceback": "RuntimeError: Failed to archive /logs/verifier: No space left on device",
            }
            result["verifier_result"] = None
            result["completion"] = None
            verdict = None
            control["physical_verdict"] = None
            control["verifier_exit_code"] = None
        result_bytes = json.dumps(result).encode()
        control["trial_result_file_sha256"] = hashlib.sha256(result_bytes).hexdigest()
        updates = {
            "nop-control/negative-control.json": json.dumps(control).encode(),
            "nop-control/trials/nop/result.json": result_bytes,
            "nop-control/trials/nop/verifier/verdict.json": json.dumps(verdict).encode(),
        }
        with tarfile.open(archive, "w:gz") as output:
            for member in original.getmembers():
                if archive_failure and member.name.startswith("nop-control/trials/nop/verifier/"):
                    continue
                payload = updates.get(member.name)
                if payload is None and member.isfile():
                    payload = original.extractfile(member).read()
                if payload is not None:
                    member.size = len(payload)
                output.addfile(member, io.BytesIO(payload) if payload is not None else None)
    return negative_control_from_archive(
        ContentEvidenceReference(uri=str(archive), sha256=hashlib.sha256(archive.read_bytes()).hexdigest()),
        "nop-control/negative-control.json",
        source_task_sha256=index["source_task_sha256"],
        source_task_checksum=index["source_task_checksum"],
        execution_requirements=ExecutionRequirements.model_validate(index["execution_requirements"]),
        **redrive,
    )


def approval(tmp_path, parent):
    error = (parent.trial.verdict or {}).get("detail", {}).get("error")
    diagnosis = tmp_path / "diagnosis.json"
    diagnosis.write_text(json.dumps({"classification": "our_infrastructure", "parent": parent.trial.result_sha256}))
    decision = tmp_path / "decision.json"
    decision.write_text(json.dumps({"scope": "one same-payload same-budget NoP-only retry"}))
    reference = ContentEvidenceReference(uri=str(decision), sha256=hashlib.sha256(decision.read_bytes()).hexdigest())
    return reference, NegativeControlInfrastructureApproval(
        approval={
            "classification": "our_infrastructure",
            "parent_result_sha256": parent.trial.result_sha256,
            "failure_signature": f"{parent.trial.result['exception_info']['exception_type']}: " f"{error}",
            "decision_ref": str(decision),
        },
        diagnosis=ContentEvidenceReference(
            uri=str(diagnosis), sha256=hashlib.sha256(diagnosis.read_bytes()).hexdigest()
        ),
    )


def source_receipt(control, reward):
    verifier = control.trial.verifier_execution_requirements
    assert verifier is not None
    result = control.trial.model_dump(mode="json")["result"]
    result["agent_info"]["name"] = "oracle"
    result["config"]["agent"] = {"name": "oracle"}
    result["exception_info"] = None
    result["completion"] = {"trajectory_present": False}
    result["verifier_result"] = {"rewards": {"reward": reward}, "exit_code": 0}
    source = OracleTrialReceipt(
        path="oracle/result.json",
        result=result,
        verdict={"status": "scored", "reward": reward},
        oracle_exit_code=0,
        result_sha256=canonical_sha256(result),
        execution_requirements=control.trial.execution_requirements,
        verifier_execution_requirements=control.trial.verifier_execution_requirements,
    )
    return OracleTaskReceipt(
        task_sha256=control.consumed_task_sha256,
        source_task_sha256=control.source_task_sha256,
        source_task_checksum=control.source_task_checksum,
        runtime_sha256=control.runtime_sha256,
        harbor_task_checksum=control.consumed_task_checksum,
        has_oracle=True,
        trials=(source,),
        calibration=ResourceCalibrationReceipt(
            declared_requirements=source.execution_requirements,
            verifier={"mode": "separate", "fixed": verifier.model_dump()},
            oracle_kind="original",
            attempts=(source,),
        ),
        negative_control_applicability=NegativeControlApplicability.WORKSPACE,
        negative_control=control,
        exclusion_reason=None if control.conclusive else OracleExclusionReason.NEGATIVE_CONTROL_PENDING,
    )


@pytest.mark.parametrize("repeat_infrastructure", [False, True])
def test_diagnosed_infrastructure_control_can_repeat_once_without_synthesizing_source_failure(
    tmp_path, repeat_infrastructure
):
    parent = physical_control(tmp_path, "parent", infrastructure=True)
    decision, diagnosed = approval(tmp_path, parent)
    with pytest.raises(ValueError, match="inconclusive native signal"):
        physical_control(tmp_path, "unapproved", infrastructure=False, redrive_parent=parent, redrive_decision=decision)
    repeat = physical_control(
        tmp_path,
        "repeat",
        infrastructure=repeat_infrastructure,
        redrive_parent=parent,
        redrive_decision=decision,
        redrive_infrastructure=diagnosed,
    )
    retained = OracleNegativeControlReceipt.model_validate_json(repeat.model_dump_json())
    require_negative_control(retained)
    assert retained.redrive_parent == parent
    assert retained.annotation is None and not retained.passed(None)
    assert retained.conclusive is not repeat_infrastructure
    assert retained.score == (None if repeat_infrastructure else 0)
    source = source_receipt(retained, 1)
    assert source.original_oracle_passed and source.passed is not repeat_infrastructure
    with pytest.raises(ValueError, match="passing original oracle"):
        source_receipt(retained, 0)
    document = retained.model_dump(mode="json")
    document["redrive_parent"] = document.copy()
    with pytest.raises(ValueError, match="Only one approved re-drive"):
        OracleNegativeControlReceipt.model_validate(document)


@pytest.mark.parametrize("changed", ["backend", "budget", "another_parent"])
def test_infrastructure_redrive_rejects_another_provider_budget_or_failed_parent(tmp_path, changed):
    parent = physical_control(tmp_path, "parent", infrastructure=True)
    decision, diagnosed = approval(tmp_path, parent)
    repeat = physical_control(
        tmp_path,
        "repeat",
        infrastructure=False,
        redrive_parent=parent,
        redrive_decision=decision,
        redrive_infrastructure=diagnosed,
    )
    document = repeat.model_dump(mode="json")
    if changed == "backend":
        result = document["trial"]["result"]
        result["config"]["environment"]["kwargs"] = {"provisioning": "snapshots"}
        document["trial"]["result_sha256"] = canonical_sha256(result)
    elif changed == "budget":
        document["trial"]["execution_requirements"]["memory_mb"] *= 2
    else:
        another = physical_control(tmp_path, "another-parent", infrastructure=True)
        document["redrive_parent"] = another.model_dump(mode="json")
    with pytest.raises(ValueError, match=r"re-drive|contract"):
        OracleNegativeControlReceipt.model_validate(document)


def test_infrastructure_redrive_rechecks_immutable_diagnosis_bytes(tmp_path):
    parent = physical_control(tmp_path, "parent", infrastructure=True)
    decision, diagnosed = approval(tmp_path, parent)
    repeat = physical_control(
        tmp_path,
        "repeat",
        infrastructure=False,
        redrive_parent=parent,
        redrive_decision=decision,
        redrive_infrastructure=diagnosed,
    )
    Path(diagnosed.diagnosis.uri).write_text("unrelated diagnosis")
    with pytest.raises(ValueError, match="diagnosis bytes changed"):
        require_negative_control(repeat)


def test_repeated_native_signal_classified_as_infrastructure_still_remains_pending(tmp_path):
    item = json.loads((FIXTURES / "redrives-index.json").read_text())[0]
    requirements = ExecutionRequirements(
        cpus=1, memory_mb=2048, storage_mb=2048, agent_timeout_sec=900, verifier_timeout_sec=600
    )

    def collect(role, **kwargs):
        archive = tmp_path / f"{role}.tar.zst"
        archive.write_bytes(gzip.decompress((FIXTURES / f"redrive-0-{role}.tar.zst.gz").read_bytes()))
        return negative_control_from_archive(
            ContentEvidenceReference(uri=str(archive), sha256=item[f"{role}_archive"]["sha256"]),
            item["control_member"],
            source_task_sha256="a" * 64,
            source_task_checksum="b" * 64,
            execution_requirements=requirements,
            **kwargs,
        )

    parent = collect("parent")
    decision, diagnosed = approval(tmp_path, parent)
    repeat = collect("repeat", redrive_parent=parent, redrive_decision=decision, redrive_infrastructure=diagnosed)
    require_negative_control(repeat)
    assert repeat.signal_exit_code == parent.signal_exit_code
    assert repeat.annotation is None and not repeat.conclusive and not repeat.passed(None)


@pytest.mark.parametrize("repeat_archive_failure", [False, True])
def test_approved_verifier_archive_failure_retries_same_control_and_preserves_another_failure(
    tmp_path, repeat_archive_failure
):
    parent = physical_control(tmp_path, "archive-parent", infrastructure=False, archive_failure=True)
    assert not parent.conclusive and parent.trial.result["completion"] is None
    decision, diagnosed = approval(tmp_path, parent)
    repeat = physical_control(
        tmp_path,
        "archive-repeat",
        infrastructure=False,
        archive_failure=repeat_archive_failure,
        redrive_parent=parent,
        redrive_decision=decision,
        redrive_infrastructure=diagnosed,
    )
    require_negative_control(repeat)
    assert repeat.redrive_parent == parent and repeat.trial.execution_requirements == parent.trial.execution_requirements
    assert repeat.conclusive is not repeat_archive_failure
    assert repeat.annotation is None and not repeat.passed(None)
    assert source_receipt(repeat, 1).passed is not repeat_archive_failure
    assert parent.trial.result["completion"] is None and parent.trial.result["verifier_result"] is None
    if repeat_archive_failure:
        assert repeat.trial.result["completion"] is None and repeat.score is None
