# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect unchanged physical nop trials from retained production archives."""

import hashlib
import json
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from urllib.parse import unquote, urlparse

from finestore.rl.curation import ContentEvidenceReference
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import ExecutionRequirements
from taskcompendium.pipeline.oracle_filter import NativeVerifierGrade, OracleNegativeControlReceipt, OracleTrialReceipt

from experiments.post_training.task_curation.archive_evidence import verified_archive_members
from experiments.post_training.task_curation.harbor_inspection import harbor_inspection


def negative_control_from_archive(
    archive: ContentEvidenceReference,
    control_member: str,
    *,
    source_task_sha256: str,
    source_task_checksum: str,
    execution_requirements: ExecutionRequirements,
    redrive_parent: OracleNegativeControlReceipt | None = None,
    redrive_decision: ContentEvidenceReference | None = None,
) -> OracleNegativeControlReceipt:
    """Bind an archived setup/nop/result tuple without changing its source result."""
    root = PurePosixPath(control_member).parent
    control_bytes = verified_archive_members(archive, {control_member})[control_member]
    control = json.loads(control_bytes)
    physical = PurePosixPath(control["trial_result_path"])
    candidates = [
        index for index in range(len(physical.parts)) if physical.parts[index : index + len(root.parts)] == root.parts
    ]
    if len(candidates) != 1:
        raise ValueError("Negative-control archive root does not identify one physical Trial")
    result_member = str(PurePosixPath(*physical.parts[candidates[0] :]))
    trial_root = PurePosixPath(result_member).parent
    setup_member = str(root / "setup-receipt.json")
    script_member = str(root / "original-public-setup.sh")
    verdict_member = str(trial_root / "verifier/verdict.json")
    rewards = {str(trial_root / "verifier" / name) for name in ("reward.json", "reward.txt")}
    members = verified_archive_members(
        archive,
        {control_member, result_member, setup_member, script_member},
        optional_names=frozenset({verdict_member, *rewards}),
    )
    result = json.loads(members[result_member])
    setup = json.loads(members[setup_member])
    verdict = json.loads(members[verdict_member]) if verdict_member in members else None
    if control["trial_result"] != result or control["physical_verdict"] != verdict or control["setup"] != setup:
        raise ValueError("Negative-control manifest differs from its archived production artifacts")
    if control["trial_result_file_sha256"] != hashlib.sha256(members[result_member]).hexdigest():
        raise ValueError("Negative-control result bytes differ from the archived file")
    trial_uri = urlparse(result["trial_uri"])
    result_path = PurePosixPath(control["trial_result_path"])
    if (
        trial_uri.scheme != "file"
        or result_path.parent != PurePosixPath(unquote(trial_uri.path))
        or result_path.parts[-len(PurePosixPath(result_member).parts) :] != PurePosixPath(result_member).parts
    ):
        raise ValueError("Negative-control archive belongs to another physical production Trial")
    if (
        control["control"] != "original-public-setup-then-nop"
        or control["native_package_unchanged"] is not True
        or control["source_commands_outside_trial"] != 0
        or setup["source_commands_outside_trial"] != 0
        or setup["setup_sha256"] != hashlib.sha256(members[script_member]).hexdigest()
        or setup["event"] != "agent_start"
        or setup["trial_id"] != result["trial_name"]
        or setup["source_native_checksum"] != control["task_checksum"]
        or setup["source_agent_timeout_sec"] != execution_requirements.agent_timeout_sec
    ):
        raise ValueError("Negative control does not preserve the original package and public setup boundary")

    def reference(name: str) -> ContentEvidenceReference:
        return ContentEvidenceReference(uri=f"{archive.uri}#{name}", sha256=hashlib.sha256(members[name]).hexdigest())

    native_grade = None
    if verdict is None and result.get("exception_info") is None and (result.get("verifier_result") or {}).get("rewards"):
        available_rewards = rewards & members.keys()
        if not available_rewards:
            raise ValueError("Native negative control is missing its physical production reward artifact")
        with TemporaryDirectory(prefix="atlas-nop-rewards-") as temporary:
            directory = Path(temporary)
            (directory / "verifier").mkdir()
            for member in available_rewards:
                (directory / "verifier" / PurePosixPath(member).name).write_bytes(members[member])
            parsed = harbor_inspection("native-rewards", [str(directory)])[str(directory)]
        member = str(trial_root / "verifier" / parsed["artifact_name"])
        native_grade = NativeVerifierGrade(
            rewards=parsed["rewards"],
            artifact_path=f"{archive.uri}#{member}",
            artifact_sha256=parsed["artifact_sha256"],
            result_sha256=canonical_sha256(result),
            task_checksum=result["task_checksum"],
        )
    trial = OracleTrialReceipt(
        path=f"{archive.uri}#{result_member}",
        result=result,
        result_sha256=canonical_sha256(result),
        verdict=verdict,
        oracle_exit_code=None,
        execution_requirements=execution_requirements,
        verifier_execution_requirements=(
            ExecutionRequirements.model_validate(result["config"]["task"]["verifier_execution_requirements"])
            if result["config"].get("task", {}).get("verifier_execution_requirements") is not None
            else None
        ),
        native_grade=native_grade,
    )
    receipt = OracleNegativeControlReceipt(
        source_task_sha256=source_task_sha256,
        source_task_checksum=source_task_checksum,
        runtime_sha256=control["runtime_sha256"],
        consumed_task_sha256=control["task_sha256"],
        consumed_task_checksum=control["task_checksum"],
        archive=archive,
        control=reference(control_member),
        result_artifact=reference(result_member),
        setup_artifact=reference(setup_member),
        verdict_artifact=reference(verdict_member) if verdict_member in members else None,
        trial=trial,
        score=(verdict or {}).get("reward"),
        verifier_exit_code=control["verifier_exit_code"],
        setup_exit_code=setup["setup_exit_code"],
        original_public_setup=setup["setup_from_original_public_instruction"],
        redrive_parent=redrive_parent,
        redrive_decision=redrive_decision,
    )
    if receipt.accepted_source_error:
        receipt = receipt.model_copy(update={"annotation": "negative_control_verifier_error"})
    elif receipt.accepted_source_deadline:
        receipt = receipt.model_copy(update={"annotation": "negative_control_verifier_timeout"})
    return receipt


def require_negative_control(receipt: OracleNegativeControlReceipt) -> None:
    """Recollect exact archived bytes before a control can participate in admission."""
    if receipt.redrive_parent is not None:
        require_negative_control(receipt.redrive_parent)
        parent = receipt.redrive_parent
        parent_root = PurePosixPath(parent.control.uri.split("#", 1)[1]).parent
        root = PurePosixPath(receipt.control.uri.split("#", 1)[1]).parent
        parent_name = str(parent_root / "original-public-setup.sh")
        name = str(root / "original-public-setup.sh")
        if verified_archive_members(parent.archive, {parent_name})[parent_name] != (
            verified_archive_members(receipt.archive, {name})[name]
        ):
            raise ValueError("Negative-control re-drive changed the original public setup")
        assert receipt.redrive_decision is not None
        decision = StoragePath(receipt.redrive_decision.uri).read_bytes()
        if hashlib.sha256(decision).hexdigest() != receipt.redrive_decision.sha256:
            raise ValueError("Negative-control re-drive decision bytes changed")
    prefix = f"{receipt.archive.uri}#"
    if not receipt.control.uri.startswith(prefix):
        raise ValueError("Negative-control reference differs from its immutable archive")
    observed = negative_control_from_archive(
        receipt.archive,
        receipt.control.uri.removeprefix(prefix),
        source_task_sha256=receipt.source_task_sha256,
        source_task_checksum=receipt.source_task_checksum,
        execution_requirements=receipt.trial.execution_requirements,
        redrive_parent=receipt.redrive_parent,
        redrive_decision=receipt.redrive_decision,
    )
    if observed != receipt:
        raise ValueError("Negative-control receipt differs from its retained physical evidence")
