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
from taskcompendium.harbor.adapter import actor_machine_required
from taskcompendium.harbor.negative_control import (
    NegativeControlSetupKind,
    negative_control_setup_kind,
    public_task_preparation,
)
from taskcompendium.harbor.runtime import TaskNetworkPolicyReceipt, TaskSpecRuntimeBinding
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import ExecutionRequirements, TaskSpec
from taskcompendium.pipeline.oracle_filter import NativeVerifierGrade, OracleNegativeControlReceipt, OracleTrialReceipt

from experiments.post_training.task_curation.archive_evidence import verified_archive_members
from experiments.post_training.task_curation.harbor_inspection import harbor_inspection


def _require_task_spec_preparation(
    members: dict[str, bytes],
    root: PurePosixPath,
    trial_root: PurePosixPath,
    control: dict,
    result: dict,
    setup: dict,
    source_task_sha256: str,
    execution_requirements: ExecutionRequirements,
) -> TaskNetworkPolicyReceipt:
    preparation = json.loads(members[str(root / "task-spec-preparation.json")])
    specification_bytes = members[str(root / "source-task-spec.json")]
    task = TaskSpec.model_validate_json(specification_bytes)
    network = TaskNetworkPolicyReceipt.model_validate_json(members[str(trial_root / "network-policy.json")])
    binding = (
        TaskSpecRuntimeBinding.model_validate(preparation["source_runtime"])
        if preparation["source_runtime"] is not None
        else None
    )
    if binding is not None:
        binding.require_task(task)
    if (
        negative_control_setup_kind(task) is not NegativeControlSetupKind.TASK_SPEC_PREPARATION
        or preparation != control["preparation"]
        or preparation["version"] != 1
        or preparation["setup_kind"] != NegativeControlSetupKind.TASK_SPEC_PREPARATION.value
        or preparation["task_spec_file"] != "source-task-spec.json"
        or preparation["task_spec_file_sha256"] != hashlib.sha256(specification_bytes).hexdigest()
        or preparation["task_spec_sha256"] != source_task_sha256
        or canonical_sha256(task.model_dump(mode="json")) != source_task_sha256
        or preparation["public_preparation_sha256"] != canonical_sha256(public_task_preparation(task))
        or preparation["source_native_checksum"] != control["task_checksum"]
        or preparation["runtime_sha256"] != control["runtime_sha256"]
        or preparation["source_runtime"] != result["config"]["task"].get("source_runtime")
        or preparation["execution_requirements"] != execution_requirements.model_dump(mode="json")
        or preparation["verifier_execution_requirements"] != result["config"]["task"]["verifier_execution_requirements"]
        or preparation["actor_required"] != actor_machine_required(task)
        or network.actor_required != preparation["actor_required"]
        or network.source_runtime_sha256 != control["runtime_sha256"]
        or network.actor_declared != (binding.network("task_machine") if binding is not None else None)
        or network.verifier_declared != (binding.network("verifier_machine") if binding is not None else None)
        or setup["setup_exit_code"] is not None
    ):
        raise ValueError("Negative control changed its actual TaskSpec preparation or source contract")
    if setup["preparation_completed"] and (
        setup["event"] != "agent_start"
        or control["hook_calls"] != 1
        or setup["environment_setup"] != result["environment_setup"]
        or result["environment_setup"]["finished_at"] is None
        or (
            network.actor_required
            and network.actor_declared is not None
            and network.actor_effective != network.actor_declared
        )
    ):
        raise ValueError("Negative control has no matching successful production TaskSpec preparation")
    return network


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
    setup_kind = NegativeControlSetupKind(control["setup_kind"]) if control.get("setup_kind") is not None else None
    converted = setup_kind is NegativeControlSetupKind.TASK_SPEC_PREPARATION
    descriptor_member = str(root / ("task-spec-preparation.json" if converted else "original-public-setup.sh"))
    verdict_member = str(trial_root / "verifier/verdict.json")
    rewards = {str(trial_root / "verifier" / name) for name in ("reward.json", "reward.txt")}
    preparation_names = (
        {str(root / "source-task-spec.json"), str(trial_root / "network-policy.json")} if converted else set()
    )
    members = verified_archive_members(
        archive,
        {control_member, result_member, setup_member, descriptor_member, *preparation_names},
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
        or setup["setup_sha256"] != hashlib.sha256(members[descriptor_member]).hexdigest()
        or (setup["event"] != "agent_start" and setup.get("preparation_completed") is not False)
        or setup["trial_id"] != result["trial_name"]
        or setup["source_native_checksum"] != control["task_checksum"]
        or setup["source_agent_timeout_sec"] != execution_requirements.agent_timeout_sec
    ):
        raise ValueError("Negative control does not preserve the original package and public setup boundary")
    network = (
        _require_task_spec_preparation(
            members, root, trial_root, control, result, setup, source_task_sha256, execution_requirements
        )
        if converted
        else None
    )
    if setup_kind is not None and setup["setup_kind"] != setup_kind.value:
        raise ValueError("Negative-control preparation kind differs from its actual setup receipt")
    if not converted and control.get("preparation") is not None:
        raise ValueError("Native negative controls cannot claim TaskSpec preparation")

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
        network_policy=network,
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
        setup_kind=setup_kind,
        preparation_artifact=reference(descriptor_member) if converted else None,
        verdict_artifact=reference(verdict_member) if verdict_member in members else None,
        trial=trial,
        score=(verdict or {}).get("reward"),
        verifier_exit_code=control["verifier_exit_code"],
        setup_exit_code=setup["setup_exit_code"],
        original_public_setup=(
            setup["preparation_completed"] if converted else setup["setup_from_original_public_instruction"]
        ),
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
        parent_name = str(
            parent_root
            / (
                "task-spec-preparation.json"
                if parent.setup_kind is NegativeControlSetupKind.TASK_SPEC_PREPARATION
                else "original-public-setup.sh"
            )
        )
        name = str(
            root
            / (
                "task-spec-preparation.json"
                if receipt.setup_kind is NegativeControlSetupKind.TASK_SPEC_PREPARATION
                else "original-public-setup.sh"
            )
        )
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
