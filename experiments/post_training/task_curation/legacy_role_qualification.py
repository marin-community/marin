# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind completed legacy passes to a reviewed, frozen allocation audit."""

import hashlib
import json

from finestore.rl.curation import ContentEvidenceReference
from pydantic import BaseModel, ConfigDict
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import ExecutionRequirements, TaskSpec
from taskcompendium.pipeline.legacy_roles import LegacyRoleAllocation, LegacyRoleQualificationReceipt
from taskcompendium.pipeline.oracle_filter import ResourceCalibrationReceipt

from experiments.post_training.task_curation.archive_evidence import verified_archive_members
from experiments.post_training.task_curation.calibration_projection import calibrated_task_spec


class LegacyRoleQualificationEvidence(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    input_partition: ContentEvidenceReference
    audit: ContentEvidenceReference
    decision: ContentEvidenceReference


def _verified_bytes(reference: ContentEvidenceReference) -> bytes:
    data = StoragePath(reference.uri).read_bytes()
    if hashlib.sha256(data).hexdigest() != reference.sha256:
        raise ValueError(f"Legacy qualification evidence bytes changed: {reference.uri}")
    return data


def _allocation(observation: dict | None, deadlines: ExecutionRequirements) -> LegacyRoleAllocation | None:
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


def legacy_role_qualifications(
    evidence: LegacyRoleQualificationEvidence, task_ids: frozenset[str]
) -> dict[str, LegacyRoleQualificationReceipt]:
    """Verify audit membership and original bytes before qualifying the requested subset."""
    inputs = json.loads(_verified_bytes(evidence.input_partition))
    audit = json.loads(_verified_bytes(evidence.audit))
    _verified_bytes(evidence.decision)
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
    _verified_bytes(producer)
    _verified_bytes(closure)
    rows = {row["task_id"]: row for row in audit["rows"]}
    if len(rows) != len(audit["rows"]):
        raise ValueError("Legacy audit repeats a task")
    qualified = {}
    for entry in inputs["entries"]:
        task_id = entry["task_id"]
        if task_id not in task_ids:
            continue
        row = rows[task_id]
        if row["outcome"] != "compatible_completed_pass":
            continue
        if {key: row[key] for key in entry} != entry or not entry["passed"] or not entry["scored"]:
            raise ValueError("Legacy allocation audit changed its original input or source outcome")
        observation = row["observation"]
        mapping = json.loads(
            _verified_bytes(ContentEvidenceReference(uri=entry["receipt_uri"], sha256=entry["receipt_sha256"]))
        )["receipt"]
        calibration = ResourceCalibrationReceipt.model_validate(mapping["calibration"])
        if not calibration.passed or calibration.oracle_kind != "original":
            raise ValueError("Legacy qualification cannot promote a source failure or calibration-only trial")
        final = calibration.attempts[-1]
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
        ):
            raise ValueError("Legacy audit belongs to another original packet or final production trial")
        deadlines = final.execution_requirements
        verifier = _allocation(observation["verifier"], deadlines)
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
            actor=_allocation(observation["actor"], deadlines),
            oracle_helper=_allocation(observation["oracle_helper"], deadlines),
            verifier=verifier,
            producer_manifest=producer,
            producer_closure=closure,
            input_partition=evidence.input_partition,
            audit=evidence.audit,
            decision=evidence.decision,
        )
        if task_id in qualified:
            raise ValueError("Legacy input partition repeats a task")
        qualified[task_id] = qualification
    return qualified
