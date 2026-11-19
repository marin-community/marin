# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Physical nop archives emitted by the offline Harbor provider fixture."""

import hashlib
import io
import json
import tarfile
from copy import deepcopy
from pathlib import Path

import zstandard
from finestore.rl.curation import ContentEvidenceReference
from taskcompendium.harbor.adapter import actor_machine_required
from taskcompendium.harbor.negative_control import NegativeControlSetupKind, public_task_preparation
from taskcompendium.harbor.runtime import TaskNetworkPolicyReceipt, TaskSpecRuntimeBinding
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.oracle_filter import (
    NegativeControlApplicability,
    OracleExclusionReason,
    OracleTaskReceipt,
    ResourceCalibrationReceipt,
)

from experiments.post_training.task_curation.negative_controls import negative_control_from_archive
from experiments.post_training.task_curation.oracle_filter import filter_cohort
from lib.taskcompendium.tests.oracle_filter_fixtures import negative_control_policy


def filter_with_negative_controls(cohort, task_map, jobs, output, *args, **kwargs):
    """Supply explicit offline-provider nop evidence to existing source-result cases."""
    if "negative_control_policy" not in kwargs:
        output.parent.mkdir(parents=True, exist_ok=True)
        decision = output.parent / "global-policy-fixture.json"
        payload = b'{"scope":"global_workspace","fixture":true}\n'
        decision.write_bytes(payload)
        kwargs["negative_control_policy"] = negative_control_policy().model_copy(
            update={"decision": ContentEvidenceReference(uri=str(decision), sha256=hashlib.sha256(payload).hexdigest())}
        )
    mappings = json.loads(task_map.read_text())
    if kwargs.get("native_success_policies") is None and all(
        "calibration" in mapping and not ResourceCalibrationReceipt.model_validate(mapping["calibration"]).passed
        for mapping in mappings.values()
    ):
        return filter_cohort(cohort, task_map, jobs, output, *args, **kwargs)
    pending = output.parent / "nop-fixture-input/tasks.parquet"
    receipt = filter_cohort(cohort, task_map, jobs, pending, *args, **kwargs)
    records = []
    for task_id, entry in receipt.tasks.items():
        if not isinstance(entry, OracleTaskReceipt) or (
            entry.negative_control_applicability is not NegativeControlApplicability.WORKSPACE
            or entry.exclusion_reason is not OracleExclusionReason.NEGATIVE_CONTROL_PENDING
        ):
            continue
        source = entry.calibration.attempts[-1]
        result = deepcopy(source.result)
        result.update(
            id=f"nop-{result['id']}",
            trial_name="nop",
            trial_uri=f"file:///fixture/{task_id}/trials/nop",
            agent_info={"name": "nop"},
            exception_info=None,
            completion={"trajectory_present": False},
            finished_at="2026-11-14T00:00:00Z",
            verifier_result={"rewards": {"reward": 0}, "exit_code": 0},
        )
        result["config"]["agent"] = {"name": "nop"}
        result["config"]["trial_name"] = "nop"
        specification_path = Path(mappings[task_id]["task_path"]) / "task-spec.json"
        converted = specification_path.is_file()
        setup_kind = (
            NegativeControlSetupKind.TASK_SPEC_PREPARATION if converted else NegativeControlSetupKind.NATIVE_SCRIPT
        )
        preparation = None
        preparation_members = {}
        setup_script = b"true\n"
        if converted:
            specification_bytes = specification_path.read_bytes()
            task = TaskSpec.model_validate_json(specification_bytes)
            binding_data = result["config"]["task"].get("source_runtime")
            binding = TaskSpecRuntimeBinding.model_validate(binding_data) if binding_data is not None else None
            preparation = {
                "version": 1,
                "setup_kind": setup_kind.value,
                "task_spec_file": "source-task-spec.json",
                "task_spec_file_sha256": hashlib.sha256(specification_bytes).hexdigest(),
                "task_spec_sha256": canonical_sha256(task.model_dump(mode="json")),
                "runtime_sha256": entry.runtime_sha256,
                "source_runtime": binding_data,
                "source_native_checksum": entry.harbor_task_checksum,
                "actor_required": actor_machine_required(task),
                "execution_requirements": source.execution_requirements.model_dump(mode="json"),
                "verifier_execution_requirements": result["config"]["task"].get("verifier_execution_requirements"),
                "public_preparation_sha256": canonical_sha256(public_task_preparation(task)),
            }
            setup_script = json.dumps(preparation).encode()
            result["environment_setup"] = {"started_at": "2026-11-13T23:59:00Z", "finished_at": "2026-11-13T23:59:01Z"}
            network = TaskNetworkPolicyReceipt(
                source_runtime_sha256=entry.runtime_sha256,
                actor_required=actor_machine_required(task),
                actor_declared=binding.network("task_machine") if binding else None,
                verifier_declared=binding.network("verifier_machine") if binding else None,
                actor_effective=binding.network("task_machine") if binding and actor_machine_required(task) else None,
                verifier_effective=binding.network("verifier_machine") if binding else None,
            )
            preparation_members = {
                f"{task_id}/source-task-spec.json": specification_bytes,
                f"{task_id}/trials/nop/network-policy.json": network.model_dump_json().encode(),
            }
        root = f"{task_id}"
        result_member = f"{root}/trials/nop/result.json"
        result_bytes = json.dumps(result).encode()
        verdict = {"status": "scored", "reward": 0, "detail": {"total": 1, "passed": 0}}
        setup = {
            "event": "agent_start",
            "trial_id": "nop",
            "source_native_checksum": entry.harbor_task_checksum,
            "setup_sha256": hashlib.sha256(setup_script).hexdigest(),
            "setup_kind": setup_kind.value,
            "setup_exit_code": None if converted else 0,
            "preparation_completed": True,
            "source_agent_timeout_sec": source.execution_requirements.agent_timeout_sec,
            "setup_from_original_public_instruction": not converted,
            "source_commands_outside_trial": 0,
        }
        if converted:
            setup["environment_setup"] = result["environment_setup"]
        control = {
            "task_sha256": entry.task_sha256,
            "runtime_sha256": entry.runtime_sha256,
            "task_checksum": entry.harbor_task_checksum,
            "control": "original-public-setup-then-nop",
            "native_package_unchanged": True,
            "setup": setup,
            "setup_kind": setup_kind.value,
            "preparation": preparation,
            "hook_calls": 1,
            "physical_verdict": verdict,
            "verifier_exit_code": 0,
            "trial_result_path": f"/fixture/{result_member}",
            "trial_result_file_sha256": hashlib.sha256(result_bytes).hexdigest(),
            "trial_result": result,
            "source_commands_outside_trial": 0,
        }
        members = {
            result_member: result_bytes,
            f"{root}/trials/nop/verifier/verdict.json": json.dumps(verdict).encode(),
            f"{root}/setup-receipt.json": json.dumps(setup).encode(),
            f"{root}/{'task-spec-preparation.json' if converted else 'original-public-setup.sh'}": setup_script,
            f"{root}/negative-control.json": json.dumps(control).encode(),
            **preparation_members,
        }
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as archive:
            for name, payload in members.items():
                item = tarfile.TarInfo(name)
                item.size = len(payload)
                archive.addfile(item, io.BytesIO(payload))
        payload = zstandard.ZstdCompressor().compress(buffer.getvalue())
        archive_path = pending.parent / f"{task_id}.tar.zst"
        archive_path.write_bytes(payload)
        proof = negative_control_from_archive(
            ContentEvidenceReference(uri=str(archive_path), sha256=hashlib.sha256(payload).hexdigest()),
            f"{root}/negative-control.json",
            source_task_sha256=entry.source_task_sha256,
            source_task_checksum=entry.source_task_checksum,
            execution_requirements=source.execution_requirements,
        )
        records.append({"task_id": task_id, "receipt": proof.model_dump(mode="json")})
    controls = pending.parent / "controls.jsonl"
    controls.write_text("".join(json.dumps(record) + "\n" for record in records))
    return filter_cohort(cohort, task_map, jobs, output, *args, negative_controls=controls, **kwargs)
