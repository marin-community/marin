# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind named-snapshot accounting to exact declarative execution evidence."""

import hashlib
import json
from typing import Literal

from finestore.rl.curation import ActorCensusInput, ContentEvidenceReference
from marin.execution.artifact import ArtifactRecord
from pydantic import BaseModel, ConfigDict, Field
from rigging.filesystem.storage_path import StoragePath
from shellbox.backends.daytona.ownership import OwnershipResolution
from taskcompendium.identity import task_spec_sha256
from taskcompendium.models import TaskSpec

from experiments.post_training.task_curation.census_binding import receipt_from_census
from experiments.post_training.task_curation.lexical_artifacts import ActorCensus, SourceTextPopulation
from experiments.post_training.task_curation.oracle_filter import row_runtime_hash
from experiments.post_training.task_curation.published_census import population_rows


class DeclarativeTaskExecution(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str
    task_sha256: str
    runtime_sha256: str
    receipt: ContentEvidenceReference
    start: ContentEvidenceReference


class DeclarativeProvisioning(BaseModel):
    """A complete task/runtime-bound execution panel using the declarative factory."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str
    input: ActorCensusInput
    factory: Literal["shellbox.backends.daytona.machine.DaytonaDeclarativeMachineFactory"]
    executor_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    executor_script: ContentEvidenceReference
    snapshot_census: ContentEvidenceReference
    tasks: tuple[DeclarativeTaskExecution, ...]


def _read_reference(reference: ContentEvidenceReference) -> bytes:
    raw = StoragePath(reference.uri).read_bytes()
    if hashlib.sha256(raw).hexdigest() != reference.sha256:
        raise ValueError(f"Declarative execution evidence digest differs: {reference.uri}")
    return raw


def bind_declarative_census(
    census: ActorCensus, qualification: ContentEvidenceReference, output_path: str
) -> ActorCensus:
    """Count zero named snapshots only after every original contract executed declaratively.

    Existing image identities, refusals and source inputs remain archived in the
    original census. This binding never clears an execution or mapping failure.
    """
    proof = DeclarativeProvisioning.model_validate_json(_read_reference(qualification))
    expected_input = receipt_from_census(census.model_dump(mode="json"), None).input
    if (
        proof.source_id != census.source_id
        or proof.input != expected_input
        or proof.snapshot_census.uri != expected_input.evidence_url
        or output_path == census.path
    ):
        raise ValueError("Declarative qualification belongs to another census")
    # The snapshot record is retained byte-for-byte; the caller cannot substitute
    # the new accounting artifact for that historical identity census.
    snapshot_record = _read_reference(proof.snapshot_census)
    record = ArtifactRecord.model_validate_json(snapshot_record)
    if record.result != census.result_payload():
        raise ValueError("Declarative qualification differs from the archived snapshot census")
    if (
        census.daytona_execution_failures
        or census.execution_failures
        or census.mapping_failures
        or set(census.daytona_unsafe_reasons) - {"too_many_snapshots"}
    ):
        raise ValueError("Declarative accounting cannot clear execution or mapping failures")
    _read_reference(proof.executor_script)
    population = SourceTextPopulation.raw_load(census.input_artifact)
    tasks = {task.task_id: task for task in proof.tasks}
    if len(tasks) != len(proof.tasks) or len(tasks) != census.task_count:
        raise ValueError("Declarative executions do not cover the complete census")
    seen = set()
    starts = {}
    for row in population_rows(population):
        spec = TaskSpec.model_validate_json(row["task_spec"])
        evidence = tasks.get(spec.id)
        if (
            evidence is None
            or spec.id in seen
            or evidence.task_sha256 != task_spec_sha256(spec)
            or evidence.runtime_sha256 != row_runtime_hash(row)
        ):
            raise ValueError(f"Declarative execution differs from the frozen task/runtime: {spec.id}")
        seen.add(spec.id)
        start_key = (evidence.start.uri, evidence.start.sha256)
        if start_key not in starts:
            starts[start_key] = json.loads(_read_reference(evidence.start))
        start = starts[start_key]
        if (
            start["provisioning"] != "declarative"
            or start["source_revision"] != proof.executor_revision
            or spec.id not in start["task_ids"]
            or start["task_count"] != len(set(start["task_ids"]))
            or not start["native_budgets_unchanged"]
        ):
            raise ValueError(f"Execution was not declarative at the pinned native contract: {spec.id}")
        raw = _read_reference(evidence.receipt)
        receipt = json.loads(raw)
        ownership = OwnershipResolution.model_validate(receipt["ownership"])
        if (
            receipt["source_id"] != census.source_id
            or receipt["contract"]
            != {
                "task_id": evidence.task_id,
                "task_sha256": evidence.task_sha256,
                "runtime_sha256": evidence.runtime_sha256,
            }
            or receipt.get("producer_error") is not None
            or receipt["ownership_status"] != "all_owned_ids_absent"
            or not ownership.evidence.sandbox_ids
            or not ownership.resolved
        ):
            raise ValueError(f"Declarative execution is incomplete or belongs to another task: {spec.id}")
    if seen != tasks.keys():
        raise ValueError("Declarative executions omit census tasks")
    return census.model_copy(
        update={
            "path": output_path,
            "daytona_safety": "safe" if census.daytona_machine_slots else "not_applicable",
            "daytona_snapshot_count": 0,
            "snapshot_sizes": {},
            "daytona_unsafe_reasons": [],
            "daytona_unsafe_details": {},
            "daytona_unsafe_counts": {},
            "daytona_count_basis": (
                f"Complete {census.task_count}-task census with physical declarative execution; "
                f"zero named snapshots required by {proof.factory}. "
                f"Exact task/runtime and owned-ID evidence: {qualification.uri} SHA256 {qualification.sha256}; "
                f"original image identities: {proof.snapshot_census.uri} SHA256 {proof.snapshot_census.sha256}."
            ),
        }
    )
