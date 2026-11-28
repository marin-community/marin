# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind ingest-time license provenance and seal dataset-level acquisition refusals."""

import hashlib
from dataclasses import replace
from enum import StrEnum
from pathlib import Path

from finestore.rl.licenses import LicenseCheck, LicenseCheckVerdict, LicenseReceipt, check_license
from pydantic import BaseModel, ConfigDict, Field
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.post_training.task_curation.backend_census_manifest import seal_manifest
from experiments.post_training.task_curation.ingest_stages.graph import StageJob
from experiments.post_training.task_curation.ingest_stages.ingest_config import IngestConfig
from experiments.post_training.task_curation.ingest_stages.receipts import SHA256
from experiments.post_training.task_curation.license_binding import read_license_receipts
from experiments.post_training.task_curation.license_check import CHECKER_VERSION, check_target, input_target
from experiments.post_training.task_curation.license_publication import require_publication_license
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef, read_ref
from infra.marina.apps.rl_data_catalog.server.licensing import source_license_ban
from infra.marina.apps.rl_data_catalog.server.source_identity import SourceId


class LicenseContext(BaseModel):
    """Host-derived registry evidence, frozen alongside the queue claim before submission."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: SourceId
    policy_sha256: str = Field(pattern=SHA256)
    registry_sha256: str = Field(pattern=SHA256)
    ingest_record: LicenseReceipt | None
    repackager: bool


class LicenseDisposition(StrEnum):
    PERMIT = "permit"
    BAN = "ban"


class LicenseResolution(BaseModel):
    """Pin the resolver's original evidence and timestamp across interrupted attempts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    input_sha256: str = Field(pattern=SHA256)
    check: LicenseCheck
    receipt: LicenseReceipt | None


class LicenseAssessment(BaseModel):
    """A dataset-level outcome; a ban is evidence, never a completed license stage."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    input_sha256: str = Field(pattern=SHA256)
    checker_version: str
    source_id: SourceId
    dataset_id: str
    revision: str
    inherited: bool
    disposition: LicenseDisposition
    receipt: LicenseReceipt | None
    checks: dict[str, LicenseCheck]
    refusals: tuple[str, ...]


class LicenseBlocked(ValueError):
    """Expose the sealed refusal without printing license documents or resolver errors."""

    def __init__(self, assessment: ArtifactRef):
        super().__init__("license gate bans acquisition; inspect the sealed assessment")
        self.assessment = assessment


def capture_license_context(policy: IngestConfig, index: Path) -> LicenseContext:
    """Capture the existing receipt and identify known repackagers from folder provenance."""
    payload = index.read_bytes()
    records = read_license_receipts(index)
    if index.read_bytes() != payload:
        raise ValueError("license registry pointer changed while capturing ingest provenance")
    root = policy.source_id.split("/", 1)[0]
    repackager = any(
        key.split("/", 1)[0] == root
        and record.resolved_source is not None
        and record.resolved_source.dataset_id != policy.upstream.repository
        for key, record in records.items()
    )
    return LicenseContext(
        source_id=policy.source_id,
        policy_sha256=policy.digest(),
        registry_sha256=hashlib.sha256(payload).hexdigest(),
        ingest_record=records.get(policy.source_id),
        repackager=repackager,
    )


def resolve_new_license(job: StageJob) -> LicenseResolution:
    """Resolve primary evidence once and bind it to this stage's exact inputs."""
    policy = job.execution.policy
    path = prefix_join(job.output_root, "resolution.json")
    target = StoragePath(path)
    if target.exists():
        resolution = LicenseResolution.model_validate_json(target.read_bytes())
    else:
        upstream = policy.upstream
        license_target = replace(input_target(f"{upstream.repository}@{upstream.revision}"), source_id=policy.source_id)
        result = check_target(license_target, None)
        resolution = LicenseResolution(
            input_sha256=job.inputs.digest(),
            check=LicenseCheck.model_validate({key: result[key] for key in LicenseCheck.model_fields}),
            receipt=LicenseReceipt.model_validate(result["receipt"]) if result["receipt"] is not None else None,
        )
    expected = (policy.source_id, policy.upstream.repository, policy.upstream.revision)
    if (
        resolution.input_sha256 != job.inputs.digest()
        or (
            resolution.check.source_id,
            resolution.check.dataset_id,
            resolution.check.revision,
        )
        != expected
    ):
        raise ValueError("license resolution belongs to another execution or upstream pin")
    receipt = resolution.receipt
    if receipt is not None and (receipt.source_id, receipt.dataset_id, receipt.revision) != expected:
        raise ValueError("license evidence belongs to another upstream pin")
    if receipt is None and resolution.check.verdict is LicenseCheckVerdict.PASS:
        raise ValueError("license resolution claims permission without evidence")
    seal_manifest(resolution.model_dump(mode="json"), path)
    return resolution


def assess_license(job: StageJob, context: LicenseContext) -> LicenseAssessment:
    """Reuse the canonical resolver and use-scoped gates; never create approval decisions."""
    policy = job.execution.policy
    if context.source_id != policy.source_id or context.policy_sha256 != policy.digest():
        raise ValueError("license context belongs to another ingest policy")
    receipt = context.ingest_record
    inherited = receipt is not None
    checks: dict[str, LicenseCheck] = {}
    refusals: list[str] = []
    dataset_id, revision = policy.upstream.repository, policy.upstream.revision
    if receipt is None and context.repackager:
        refusals.append("repackaged folder has no registered upstream lineage receipt")
    elif receipt is None:
        resolution = resolve_new_license(job)
        receipt = resolution.receipt
        if receipt is None:
            checks["resolution"] = resolution.check
            refusals.append("primary license evidence is unresolved")
        elif receipt.bundled_third_party and receipt.resolved_source is None:
            refusals.append("new aggregate source requires registered per-folder upstream lineage")
        elif policy.source_id.startswith("Marin:") and "openmdw-1.1" not in receipt.license_ids:
            refusals.append("first-party input has no recorded OpenMDW 1.1 grant")
    if receipt is not None:
        if receipt.source_id != policy.source_id:
            raise ValueError("license registry receipt belongs to another source")
        if inherited:
            # The ingest-provenance ruling carries the original record unchanged
            # across repairs. A publication pin is not grounds for re-resolution.
            dataset_id, revision = receipt.dataset_id, receipt.revision
        if context.repackager and receipt.resolved_source is None:
            refusals.append("repackaged folder record does not identify its original upstream")
        for use in policy.intended_use:
            checks[use.value] = check_license(
                receipt, source_license_ban(policy.source_id, policy.upstream.repository, receipt, use.value)
            )
            try:
                require_publication_license(receipt, policy.source_id, policy.upstream.repository, use.value)
            except ValueError:
                # Gate refusals are a normal policy outcome. Preserve the typed
                # check, but do not print its quotes or turn it into permission.
                refusals.append(f"canonical license gate refused {use.value}")
    return LicenseAssessment(
        input_sha256=job.inputs.digest(),
        checker_version=CHECKER_VERSION,
        source_id=policy.source_id,
        dataset_id=dataset_id,
        revision=revision,
        inherited=inherited,
        disposition=LicenseDisposition.BAN if refusals else LicenseDisposition.PERMIT,
        receipt=receipt,
        checks=checks,
        refusals=tuple(refusals),
    )


def license_stage(job: StageJob) -> dict[str, ArtifactRef]:
    """Seal refusal evidence before raising; only permission completes the graph stage."""
    context = LicenseContext.model_validate(read_ref(job.inputs.artifacts["license-context"]))
    path = prefix_join(job.output_root, "assessment.json")
    assessment = assess_license(job, context)
    reference = seal_manifest(assessment.model_dump(mode="json"), path)
    if assessment.disposition is LicenseDisposition.BAN:
        raise LicenseBlocked(reference)
    return {"assessment": reference}
