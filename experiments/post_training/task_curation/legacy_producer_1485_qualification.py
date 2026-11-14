# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify the separate producer-1485 adoption against unchanged production evidence."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import PurePosixPath

from finestore.rl.curation import ContentEvidenceReference
from pydantic import BaseModel, ConfigDict
from taskcompendium.harbor.runtime import TaskSpecRuntimeBinding
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.legacy_producer_1485 import (
    LegacyProducer1485AdoptionDecision,
    LegacyProducer1485QualificationReceipt,
)
from taskcompendium.pipeline.oracle_filter import ResourceCalibrationReceipt

from experiments.post_training.task_curation.archive_evidence import verified_archive_members
from experiments.post_training.task_curation.calibration_projection import calibrated_task_spec
from experiments.post_training.task_curation.legacy_role_qualification import (
    legacy_role_allocation,
    verified_legacy_evidence_bytes,
)

PRODUCER_MODULES = {
    "harbor.trial.taskcompendium": "vendor/harbor/src/harbor/trial/taskcompendium.py",
    "rolloutengine.machines": "lib/rolloutengine/src/rolloutengine/machines.py",
}


class LegacyProducer1485QualificationEvidence(BaseModel):
    """Immutable inputs for the independent producer-1485 adoption."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    input_partition: ContentEvidenceReference
    audit: ContentEvidenceReference
    decision: ContentEvidenceReference


@dataclass(frozen=True)
class _Producer1485Scope:
    entries: dict[str, dict]
    rows: dict[str, dict]
    unscored: frozenset[str]
    decision: LegacyProducer1485AdoptionDecision
    producer: ContentEvidenceReference
    closure: ContentEvidenceReference
    backend: dict


def _producer_1485_scope(evidence: LegacyProducer1485QualificationEvidence) -> _Producer1485Scope:
    inputs = json.loads(verified_legacy_evidence_bytes(evidence.input_partition))
    audit = json.loads(verified_legacy_evidence_bytes(evidence.audit))
    decision = LegacyProducer1485AdoptionDecision.model_validate_json(verified_legacy_evidence_bytes(evidence.decision))
    verified_legacy_evidence_bytes(decision.ruling)
    if (
        decision.input_index_sha256 != evidence.input_partition.sha256
        or decision.audit_sha256 != evidence.audit.sha256
        or inputs["backend_revision"] != decision.producer_revision
        or audit["producer_revision"] != decision.producer_revision
        or audit["input_partition_sha256"] != evidence.input_partition.sha256
        or inputs["tasks"] != decision.adopted_count + decision.excluded_unscored
        or inputs["passing_observations"] != decision.adopted_count
        or inputs["unscored_observations"] != decision.excluded_unscored
        or audit["counts"] != {"compatible_completed_pass": decision.adopted_count}
    ):
        raise ValueError("Producer-1485 qualification requires its exact adopted partition and audit")
    entries = {entry["task_id"]: entry for entry in inputs["entries"]}
    rows = {row["task_id"]: row for row in audit["rows"]}
    if len(entries) != len(inputs["entries"]) or len(entries) != inputs["tasks"] or len(rows) != len(audit["rows"]):
        raise ValueError("Producer-1485 evidence repeats or omits a task")
    passing = {task_id for task_id, entry in entries.items() if entry["passed"] and entry["scored"]}
    unscored = {task_id for task_id, entry in entries.items() if not entry["passed"] and not entry["scored"]}
    if len(passing) != decision.adopted_count or len(unscored) != decision.excluded_unscored or rows.keys() != passing:
        raise ValueError("Producer-1485 adoption cannot change a scored or unscored source outcome")
    producer = ContentEvidenceReference(uri=inputs["producer_manifest_uri"], sha256=inputs["producer_manifest_sha256"])
    closure = ContentEvidenceReference(uri=inputs["closure_uri"], sha256=inputs["closure_sha256"])
    manifest = json.loads(verified_legacy_evidence_bytes(producer))
    closed = json.loads(verified_legacy_evidence_bytes(closure))
    backend_ref = ContentEvidenceReference(uri=inputs["backend_identity_uri"], sha256=inputs["backend_identity_sha256"])
    backend = json.loads(verified_legacy_evidence_bytes(backend_ref))
    if (
        manifest["revision"] != decision.producer_revision
        or backend["revision"] != decision.producer_revision
        or closed["backend_revision"] != decision.producer_revision
        or closed["backend_identity_uri"] != backend_ref.uri
        or closed["backend_identity_sha256"] != backend_ref.sha256
        or closed["producer_manifest_uri"] != producer.uri
        or closed["producer_manifest_sha256"] != producer.sha256
    ):
        raise ValueError("Producer-1485 capture belongs to another frozen runtime")
    return _Producer1485Scope(entries, rows, frozenset(unscored), decision, producer, closure, backend)


def legacy_producer_1485_qualifications(
    evidence: LegacyProducer1485QualificationEvidence, task_ids: frozenset[str]
) -> dict[str, LegacyProducer1485QualificationReceipt]:
    """Qualify only adopted passes after checking the complete pass/unscored partition."""
    scope = _producer_1485_scope(evidence)
    qualified = {}
    for task_id, entry in scope.entries.items():
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
            or final.scored != entry["scored"]
            or final.result_sha256 != entry["parent_result_sha256"]
            or mapping["task_sha256"] != entry["task_sha256"]
            or mapping["runtime_sha256"] != entry["runtime_sha256"]
            or mapping["source_task_checksum"] != entry["source_task_checksum"]
            or mapping["task_checksum"] != entry["task_checksum"]
        ):
            raise ValueError("Producer-1485 evidence changed an original source task or outcome")
        if task_id in scope.unscored:
            if any(attempt.scored for attempt in calibration.attempts):
                raise ValueError("Producer-1485 unscored exclusion contains a scored attempt")
            continue
        qualification = _qualified_1485_pass(scope, evidence, entry, mapping, calibration)
        if task_id in task_ids:
            qualified[task_id] = qualification
    return qualified


def _qualified_1485_pass(
    scope: _Producer1485Scope,
    evidence: LegacyProducer1485QualificationEvidence,
    entry: dict,
    mapping: dict,
    calibration: ResourceCalibrationReceipt,
) -> LegacyProducer1485QualificationReceipt:
    task_id = entry["task_id"]
    final = calibration.attempts[-1]
    row = scope.rows[task_id]
    observation = row["observation"]
    if (
        {key: row[key] for key in entry} != entry
        or row["outcome"] != "compatible_completed_pass"
        or row["hold_reasons"]
        or observation["source_commands"] != 0
        or observation["provider_creations"] != 0
        or any(
            observation["module_files"][module]["sha256"] != scope.backend["files_sha256"][path]
            for module, path in PRODUCER_MODULES.items()
        )
    ):
        raise ValueError("Producer-1485 audit differs from its actual offline constructor capture")
    archive = ContentEvidenceReference(uri=entry["trial_archive_uri"], sha256=entry["trial_archive_sha256"])
    runtime_evidence = row["archived_source_runtime"]
    consumed_runtime = runtime_evidence["consumed_member"]
    consumed_spec = str(PurePosixPath(consumed_runtime).with_name("task-spec.json"))
    result_parts = PurePosixPath(final.path).parts
    result_member = PurePosixPath("trials", *result_parts[result_parts.index("attempt-00") :])
    verdict_member = str(result_member.parent / "verifier/verdict.json")
    exit_member = str(result_member.parent / "agent/exit-code.txt")
    members = verified_archive_members(
        archive,
        {
            "original-packet/task-spec.json",
            "original-packet/task.toml",
            "original-packet/source-runtime.json",
            consumed_runtime,
            consumed_spec,
            str(result_member),
            verdict_member,
            exit_member,
        },
    )
    original_runtime = members["original-packet/source-runtime.json"]
    task = TaskSpec.model_validate_json(members["original-packet/task-spec.json"])
    binding = TaskSpecRuntimeBinding.model_validate_json(original_runtime)
    binding.require_task(task)
    if (
        runtime_evidence["original_member"] != "original-packet/source-runtime.json"
        or runtime_evidence["original_and_consumed_bytes_equal"] is not True
        or original_runtime != members[consumed_runtime]
        or hashlib.sha256(original_runtime).hexdigest() != runtime_evidence["sha256"]
        or members["original-packet/task-spec.json"] != members[consumed_spec]
        or task.id != task_id
        or canonical_sha256(task.model_dump(mode="json")) != entry["task_sha256"]
        or binding.task_sha256 != entry["task_sha256"]
        or binding.runtime_sha256 != entry["runtime_sha256"]
        or binding.model_dump(mode="json") != observation["source_runtime"]
        or observation["packet_task_sha256"] != entry["task_sha256"]
        or observation["packet_checksum"] != mapping["source_task_checksum"]
        or observation["raw_task_checksum"] != mapping["task_checksum"]
        or observation["archive_sha256"] != archive.sha256
        or observation["raw_result_sha256"] != final.result_sha256
        or observation["phase_deadlines"] != final.execution_requirements.model_dump(mode="json")
        or final.result["config"]["task"].get("source_runtime") is not None
        or len(calibration.attempts) != 1
        or final.verifier_execution_requirements is not None
        or canonical_sha256(json.loads(members[str(result_member)])) != final.result_sha256
        or json.loads(members[verdict_member]) != final.verdict
        or int(members[exit_member]) != final.oracle_exit_code
    ):
        raise ValueError("Producer-1485 qualification changed the original/consumed packet, runtime or Result")
    verifier = legacy_role_allocation(observation["verifier"], final.execution_requirements)
    if verifier is None:
        raise ValueError("Producer-1485 qualification requires the captured verifier MachineSpec")
    published = calibrated_task_spec(
        task, final.execution_requirements, verifier.execution_requirements, members["original-packet/task.toml"]
    )
    return LegacyProducer1485QualificationReceipt(
        version=1,
        producer_revision=scope.decision.producer_revision,
        source_task_sha256=entry["task_sha256"],
        source_task_checksum=mapping["source_task_checksum"],
        consumed_task_sha256=observation["packet_task_sha256"],
        consumed_task_checksum=mapping["task_checksum"],
        published_task_sha256=canonical_sha256(published.model_dump(mode="json")),
        result_sha256=final.result_sha256,
        calibration_sha256=canonical_sha256(calibration.model_dump(mode="json")),
        input_cohort_sha256=entry["cohort_sha256"],
        source_runtime=binding,
        actor=legacy_role_allocation(observation["actor"], final.execution_requirements),
        oracle_helper=legacy_role_allocation(observation["oracle_helper"], final.execution_requirements),
        verifier=verifier,
        producer_manifest=scope.producer,
        producer_closure=scope.closure,
        input_partition=evidence.input_partition,
        audit=evidence.audit,
        decision=evidence.decision,
        adoption_decision=scope.decision,
    )
