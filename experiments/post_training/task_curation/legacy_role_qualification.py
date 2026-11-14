# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind completed legacy passes to a reviewed, frozen allocation audit."""

import hashlib
import json

from finestore.rl.curation import ContentEvidenceReference
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import ExecutionRequirements, TaskSpec
from taskcompendium.pipeline.legacy_roles import (
    LegacyRoleAdoptionDecision,
    LegacyRoleAllocation,
    LegacyRoleQualificationReceipt,
    LegacySmallRoleAdoptionDecision,
    LegacySmallRoleScope,
    require_legacy_adoption_scope,
)
from taskcompendium.pipeline.oracle_filter import ResourceCalibrationReceipt

from experiments.post_training.task_curation.archive_evidence import verified_archive_members
from experiments.post_training.task_curation.calibration_projection import calibrated_task_spec


class LegacyRoleQualificationEvidence(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    input_partition: ContentEvidenceReference
    audit: ContentEvidenceReference
    decision: ContentEvidenceReference
    small_scope: LegacySmallRoleScope | None = Field(default=None, exclude_if=lambda value: value is None)


def verified_legacy_evidence_bytes(reference: ContentEvidenceReference) -> bytes:
    """Read an immutable evidence object and reject changed bytes."""
    data = StoragePath(reference.uri).read_bytes()
    if hashlib.sha256(data).hexdigest() != reference.sha256:
        raise ValueError(f"Legacy qualification evidence bytes changed: {reference.uri}")
    return data


def legacy_role_allocation(observation: dict | None, deadlines: ExecutionRequirements) -> LegacyRoleAllocation | None:
    """Record captured capacity with the original phase deadlines."""
    if observation is None:
        return None
    return LegacyRoleAllocation(
        execution_requirements=ExecutionRequirements.model_validate(
            {**deadlines.model_dump(), **{name: observation[name] for name in ("cpus", "memory_mb", "storage_mb")}}
        ),
        network=observation["network"],
        gpus=observation["gpus"],
        machine_spec_sha256=canonical_sha256(observation),
    )


def _require_small_partition(inputs: dict, audit: dict, manifest: dict, decision: LegacySmallRoleAdoptionDecision):
    entries = {entry["task_id"]: entry for entry in inputs["entries"]}
    rows = {row["task_id"]: row for row in audit["rows"]}
    total = decision.adopted_count + decision.excluded_scored_failures
    files = {item["path"]: item["sha256"] for item in manifest["files"]}
    if (
        len(entries) != len(inputs["entries"])
        or len(rows) != len(audit["rows"])
        or entries.keys() != rows.keys()
        or len(entries) != total
        or inputs["tasks"] != total
        or inputs["source_commands_executed"] != 0
        or manifest["version"] != 1
        or manifest["producer_revision"] != decision.producer_revision
        or manifest["compatible_completed_passes"] != decision.adopted_count
        or manifest["scored_failure_held_unchanged"] != decision.excluded_scored_failures
        or manifest["providers"] != 0
        or manifest["source_commands"] != 0
        or files.get("index.json") != decision.input_index_sha256
        or files.get("partition-v2.json") != decision.input_partition_sha256
        or audit["counts"]
        != {"compatible_completed_pass": decision.adopted_count, "held": decision.excluded_scored_failures}
    ):
        raise ValueError("Legacy small adoption requires its complete offline partition and exact outcome counts")
    passes = failures = 0
    for task_id, entry in entries.items():
        row = rows[task_id]
        if (
            {key: row[key] for key in entry} != entry
            or not entry["scored"]
            or row["observation"]["source_commands"] != 0
            or row["observation"]["provider_creations"] != 0
        ):
            raise ValueError("Legacy small audit changed its frozen membership or performed execution")
        if entry["passed"] and row["outcome"] == "compatible_completed_pass":
            passes += 1
        elif not entry["passed"] and row["outcome"] == "held":
            failures += 1
        else:
            raise ValueError("Legacy small audit changed a scored source outcome")
    if (passes, failures) != (decision.adopted_count, decision.excluded_scored_failures):
        raise ValueError("Legacy small audit differs from the adopted pass/failure partition")


def legacy_role_qualifications(
    evidence: LegacyRoleQualificationEvidence, task_ids: frozenset[str]
) -> dict[str, LegacyRoleQualificationReceipt]:
    """Verify audit membership and original bytes before qualifying the requested subset."""
    inputs = json.loads(verified_legacy_evidence_bytes(evidence.input_partition))
    audit = json.loads(verified_legacy_evidence_bytes(evidence.audit))
    decision = TypeAdapter(LegacyRoleAdoptionDecision | LegacySmallRoleAdoptionDecision).validate_json(
        verified_legacy_evidence_bytes(evidence.decision)
    )
    require_legacy_adoption_scope(decision, evidence.input_partition, evidence.audit, evidence.small_scope)
    if decision.producer_revision != inputs["backend_revision"]:
        raise ValueError("Legacy adoption decision scope differs from the requested partition or audit")
    if isinstance(decision, LegacySmallRoleAdoptionDecision):
        assert evidence.small_scope is not None
        verified_legacy_evidence_bytes(evidence.small_scope.ruling)
        manifest = json.loads(verified_legacy_evidence_bytes(evidence.small_scope.audit_manifest))
        _require_small_partition(inputs, audit, manifest, decision)
    else:
        verified_legacy_evidence_bytes(decision.ruling)
    if (
        audit["version"] != 2
        or audit["input_partition_sha256"] != evidence.input_partition.sha256
        or audit["producer_revision"] != inputs["backend_revision"]
        or audit["source_commands"] != 0
        or audit["provider_creations"] != 0
    ):
        raise ValueError("Legacy qualification requires its exact frozen offline allocation audit")
    producer = ContentEvidenceReference(uri=inputs["producer_manifest_uri"], sha256=inputs["producer_manifest_sha256"])
    closure = ContentEvidenceReference(uri=inputs["closure_uri"], sha256=inputs["closure_sha256"])
    verified_legacy_evidence_bytes(producer)
    verified_legacy_evidence_bytes(closure)
    rows = {row["task_id"]: row for row in audit["rows"]}
    if len(rows) != len(audit["rows"]):
        raise ValueError("Legacy audit repeats a task")
    qualified = {}
    for entry in inputs["entries"]:
        task_id = entry["task_id"]
        if task_id not in task_ids and evidence.small_scope is None:
            continue
        row = rows[task_id]
        if row["outcome"] != "compatible_completed_pass" and evidence.small_scope is None:
            continue
        if {key: row[key] for key in entry} != entry or not entry["scored"]:
            raise ValueError("Legacy allocation audit changed its original input or source outcome")
        observation = row["observation"]
        mapping = json.loads(
            verified_legacy_evidence_bytes(
                ContentEvidenceReference(uri=entry["receipt_uri"], sha256=entry["receipt_sha256"])
            )
        )["receipt"]
        calibration = ResourceCalibrationReceipt.model_validate(mapping["calibration"])
        final = calibration.attempts[-1]
        if (
            calibration.oracle_kind != "original"
            or calibration.passed != entry["passed"]
            or not final.scored
            or (not calibration.passed and evidence.small_scope is None)
        ):
            raise ValueError("Legacy qualification cannot change the original scored source outcome")
        archive = ContentEvidenceReference(uri=entry["trial_archive_uri"], sha256=entry["trial_archive_sha256"])
        members = verified_archive_members(archive, {"original-packet/task-spec.json", "original-packet/task.toml"})
        task = TaskSpec.model_validate_json(members["original-packet/task-spec.json"])
        if (
            task.id != task_id
            or canonical_sha256(task.model_dump(mode="json")) != entry["task_sha256"]
            or observation["packet_task_sha256"] != entry["task_sha256"]
            or observation["packet_checksum"] != mapping["source_task_checksum"]
            or observation["raw_task_checksum"] != mapping["task_checksum"]
            or observation["archive_sha256"] != archive.sha256
            or observation["raw_result_sha256"] != final.result_sha256
            or observation["phase_deadlines"] != final.execution_requirements.model_dump(mode="json")
            or observation["source_runtime"]["runtime_sha256"] != entry["runtime_sha256"]
            or mapping["runtime_sha256"] != entry["runtime_sha256"]
        ):
            raise ValueError("Legacy audit belongs to another original packet or final production trial")
        if not calibration.passed:
            continue
        deadlines = final.execution_requirements
        verifier = legacy_role_allocation(observation["verifier"], deadlines)
        if verifier is None:
            raise ValueError("Legacy qualification requires an observed verifier MachineSpec")
        published = calibrated_task_spec(
            task, deadlines, verifier.execution_requirements, members["original-packet/task.toml"]
        )
        qualification = LegacyRoleQualificationReceipt(
            version=1,
            producer_revision=inputs["backend_revision"],
            source_task_sha256=entry["task_sha256"],
            source_task_checksum=mapping["source_task_checksum"],
            consumed_task_sha256=observation["packet_task_sha256"],
            consumed_task_checksum=mapping["task_checksum"],
            published_task_sha256=canonical_sha256(published.model_dump(mode="json")),
            result_sha256=final.result_sha256,
            calibration_sha256=canonical_sha256(calibration.model_dump(mode="json")),
            input_cohort_sha256=entry["cohort_sha256"],
            source_runtime=observation["source_runtime"],
            actor=legacy_role_allocation(observation["actor"], deadlines),
            oracle_helper=legacy_role_allocation(observation["oracle_helper"], deadlines),
            verifier=verifier,
            producer_manifest=producer,
            producer_closure=closure,
            input_partition=evidence.input_partition,
            audit=evidence.audit,
            decision=evidence.decision,
            adoption_decision=decision,
            small_scope=evidence.small_scope,
        )
        if task_id in qualified:
            raise ValueError("Legacy input partition repeats a task")
        if task_id in task_ids:
            qualified[task_id] = qualification
    return qualified
