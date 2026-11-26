# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind read-back retention releases while preserving the original source evidence."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from finestore.rl.curation import (
    ContentEvidenceReference,
    DatasetRelease,
    DeconReceiptStatus,
    DecontaminatedRelease,
    LexicalContaminationEvidence,
    LexicalReceipt,
    LexicalUse,
    NoOracleExecutabilityPolicy,
    NoOracleStaticRetentionEvidence,
    PipelineReference,
    WithinSourceDedupEvidence,
)
from pydantic import BaseModel, ConfigDict, Field, model_validator
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.census_binding import ActorCensusReceipt, bind_actor_census
from experiments.post_training.task_curation.datasets.daytona_measurements import actor_census_receipts
from experiments.post_training.task_curation.full_content_binding import RetainedLineageProof
from experiments.post_training.task_curation.license_binding import licensed_source
from experiments.post_training.task_curation.receipt_shards import write_receipt_shards
from experiments.post_training.task_curation.remediation_publish import ReleasePlan
from experiments.post_training.task_curation.retained_grading import RetainedGrading
from experiments.post_training.task_curation.retained_release import (
    RETAINED_IMPORT_VERSION,
    RetainedCohort,
    import_retained,
    import_taskspec_carry,
)
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, SourceReference
from experiments.post_training.task_curation.within_source_binding import bind_within_source_dedup
from infra.marina.apps.rl_data_catalog.server.admission import require_private_release

RETAINED_RECEIPTS_PER_SHARD = 16


class ArchivePin(BaseModel):
    repo: str
    revision: str


class PublicationReceipt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str
    plan: ReleasePlan
    repo: str
    revision: str
    url: str
    private: bool
    format: str
    original_private_pin: ArchivePin
    tasks_parquet_sha256: str
    retained_content_sha256: str
    verified_retained_tasks: int
    specs_output_dir: str
    evidence_url: str
    lexical: LexicalContaminationEvidence

    @property
    def release(self) -> DatasetRelease:
        return DatasetRelease(repo=self.repo, revision=self.revision, url=self.url)

    @model_validator(mode="after")
    def _frozen_population(self) -> "PublicationReceipt":
        plan, lexical = self.plan, self.lexical
        if not self.private or self.format != "taskcompendium" or not self.repo.startswith("open-athena/"):
            raise ValueError("Retention requires a private open-athena TaskCompendium publication")
        if self.source_id != plan.source_id or self.verified_retained_tasks != plan.retained_count:
            raise ValueError("Publication readback differs from the retained plan")
        if (
            lexical.input_dataset_id,
            lexical.input_dataset_revision,
            lexical.converter_revision,
            lexical.measured_tasks,
            lexical.contaminated_tasks,
        ) != (
            plan.source_dataset_id,
            plan.source_revision,
            plan.converter_revision,
            plan.raw_count,
            plan.raw_lexical_count,
        ):
            raise ValueError("Publication lexical evidence differs from the frozen input")
        lexical_drops = plan.unique_lexical_count if plan.lexical_use is LexicalUse.TRAIN else 0
        if (
            plan.raw_count != plan.retained_count + plan.combined_drop_count
            or plan.raw_count != plan.unique_count + plan.within_source_duplicates
            or plan.combined_drop_count != plan.within_source_duplicates + lexical_drops
        ):
            raise ValueError("Publication counts do not conserve the frozen population")
        return self


class StaticRetentionBinding(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    policy: NoOracleExecutabilityPolicy
    evidence: NoOracleStaticRetentionEvidence

    @model_validator(mode="after")
    def _same_policy(self) -> "StaticRetentionBinding":
        if self.evidence.policy_sha256 != self.policy.sha256:
            raise ValueError("Static retention evidence belongs to another policy")
        return self


class RetainedGradingCarry(BaseModel):
    """Retained grading evidence carried forward through verified content identity.

    Coordinator ruling 2026-11-25 ~22:20 EST (derived from the user's 2026-11-25 ~20:00
    principle that a previously admitted source is re-admitted unless its code is
    broken): a re-pinned release whose content is byte-identical to the prior admitted
    pin carries the prior pin's retained grading instead of regrading the population.
    The carry is explicit so it is never mistaken for a fresh census: it names the
    carried-from pin and the content-identity proof digest, and it admits only when the
    grader/verifier identity is unchanged and the converter change touched format only.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    carried_from_repo: str
    carried_from_revision: str
    identity_proof_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    grader_identity_unchanged: bool
    converter_format_only: bool
    tasks: int = Field(gt=0)

    @model_validator(mode="after")
    def _admission_conditions(self) -> "RetainedGradingCarry":
        if not self.grader_identity_unchanged or not self.converter_format_only:
            raise ValueError(
                "A carried retained release admits only with unchanged grader identity and a "
                "format-only converter change; anything else fails closed"
            )
        return self


class RetainedReleaseReceipt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str
    publication: PublicationReceipt
    actor: ActorCensusReceipt
    grading: RetainedGrading
    original_pipeline: PipelineReference
    original_info: SourceInfo
    retained_dedup: WithinSourceDedupEvidence
    static_retention: StaticRetentionBinding | None = None
    carry: RetainedGradingCarry | None = None

    @model_validator(mode="after")
    def _exact_retained_pin(self) -> "RetainedReleaseReceipt":
        publication = self.publication
        if self.source_id != publication.source_id or self.actor.source_id != self.source_id:
            raise ValueError("Retention lineage contains another source")
        if self.grading.source_id != self.source_id or self.grading.input != self.actor.input:
            raise ValueError("Retained grading evidence belongs to another actor census input")
        # The importer identity the new pin's census and dedup must describe: the retained
        # importer for a fresh retention, or the publication's own conversion format for a
        # carried one (the carried release was converted by its declared format, not by the
        # retained importer).
        importer = RETAINED_IMPORT_VERSION if self.carry is None else publication.format
        if (
            self.actor.input.dataset_id,
            self.actor.input.dataset_revision,
            self.actor.input.converter_version,
            self.actor.input.measured_tasks,
        ) != (publication.repo, publication.revision, importer, publication.plan.retained_count):
            raise ValueError("Retained census does not cover the new pin and importer")
        dedup = self.retained_dedup
        if (
            dedup.source_dataset_id,
            dedup.source_revision,
            dedup.converter_version,
            dedup.raw_count,
            dedup.unique_count,
            dedup.within_source_duplicates,
        ) != (
            publication.repo,
            publication.revision,
            importer,
            publication.plan.retained_count,
            publication.plan.retained_count,
            0,
        ):
            raise ValueError("Active dedup evidence must describe the retained population")
        if self.carry is not None:
            if self.carry.tasks != self.actor.input.measured_tasks:
                raise ValueError("The carried grading population differs from the new census count")
        if self.static_retention is not None:
            static = self.static_retention.evidence
            if (
                static.source_id,
                static.source_release,
                static.current_input,
                static.retained_count,
                static.intended_use.value,
            ) != (
                self.source_id,
                publication.release,
                self.actor.input,
                publication.plan.retained_count,
                publication.plan.lexical_use.value,
            ):
                raise ValueError("Static retention evidence belongs to another retained population or Use")
        return self


class ArchivedRetainedBinding(BaseModel):
    """An explicitly authorized retained baseline independent of a live converter."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    retained: RetainedReleaseReceipt
    authority: ContentEvidenceReference

    @model_validator(mode="after")
    def _recorded_original(self) -> "ArchivedRetainedBinding":
        record = self.retained
        plan, pipeline = record.publication.plan, record.original_pipeline
        actor = record.original_info.actor_census_input
        if record.original_info.id != record.source_id or actor is None:
            raise ValueError("Archived retention lacks its original source census")
        if (pipeline.source, pipeline.revision, pipeline.version) != (
            plan.source_dataset_id,
            plan.source_revision,
            plan.converter_version,
        ) or (
            actor.dataset_id,
            actor.dataset_revision,
            actor.converter_version,
            actor.converter_revision,
            actor.task_specs_sha256,
            actor.measured_tasks,
        ) != (
            plan.source_dataset_id,
            plan.source_revision,
            plan.converter_version,
            plan.converter_revision,
            plan.task_specs_sha256,
            plan.raw_count,
        ):
            raise ValueError("Archived retention differs from its recorded conversion")
        return self


def release_lexical_evidence(publication: PublicationReceipt) -> LexicalContaminationEvidence:
    """Keep raw measurements and attach disjoint duplicate-first retention accounting."""
    plan = publication.plan
    if plan.lexical_use is not LexicalUse.TRAIN:
        return publication.lexical
    receipt = LexicalReceipt(
        use=LexicalUse.TRAIN,
        status=DeconReceiptStatus.REMEDIATED if plan.raw_lexical_count else DeconReceiptStatus.CLEAN,
        retained_tasks=plan.retained_count,
        dropped_tasks=plan.unique_lexical_count,
        duplicate_dropped_tasks=plan.within_source_duplicates,
        release=DecontaminatedRelease(
            private=True,
            format="taskcompendium",
            dataset_id=publication.repo,
            dataset_revision=publication.revision,
            drop_manifest_uri=plan.combined_drop_manifest_uri,
            original_private_dataset_id=publication.original_private_pin.repo if plan.raw_lexical_count else None,
            original_private_dataset_revision=(
                publication.original_private_pin.revision if plan.raw_lexical_count else None
            ),
            duplicate_manifest_uri=plan.drop_manifest_uri,
            dedup_census_uri=plan.evidence_url,
            original_dataset_id=plan.source_dataset_id,
            original_dataset_revision=plan.source_revision,
        ),
    )
    return LexicalContaminationEvidence.model_validate(
        {**publication.lexical.model_dump(mode="json"), "receipt": receipt}
    )


def retained_lineage(receipt: RetainedReleaseReceipt) -> RetainedLineageProof | None:
    """The byte-preserving subset proof for a zero-drop retained release, else None.

    The publication readback verified every retained task hash against the frozen
    original conversion; a release that dropped nothing therefore carries the
    original population byte-for-byte, and its archived original-pin evidence
    stays valid. Releases with any drop (duplicate or lexical) return None:
    their archived evidence describes a different population.
    """
    plan, publication = receipt.publication.plan, receipt.publication
    if plan.combined_drop_count != 0 or publication.verified_retained_tasks != plan.raw_count:
        return None
    actor, dedup = receipt.original_info.actor_census_input, receipt.original_info.within_source_dedup
    if actor is None or dedup is None:
        return None
    return RetainedLineageProof(
        source_id=receipt.source_id,
        original_dataset_id=plan.source_dataset_id,
        original_dataset_revision=plan.source_revision,
        original_input=actor,
        original_unique_count=dedup.unique_count,
        retained_content_sha256=publication.retained_content_sha256,
    )


def bind_retained_release(source: RlDataSource, receipt: RetainedReleaseReceipt) -> RlDataSource[RetainedCohort]:
    """Select the new configured pin only when the archived conversion binding still matches."""
    publication = receipt.publication
    plan = publication.plan
    dataset = source.dataset
    if dataset is None or (source.info.id, dataset.name, dataset.revision, source.version) != (
        receipt.source_id,
        plan.source_dataset_id,
        plan.source_revision,
        plan.converter_version,
    ):
        raise ValueError(f"Retention belongs to a different conversion input: {source.info.id}")
    if source.pipeline_reference != receipt.original_pipeline:
        raise ValueError(f"Retention lineage belongs to a different pipeline: {source.info.id}")
    original_input = source.info.actor_census_input
    if original_input is None or (
        original_input.task_specs_sha256,
        original_input.converter_revision,
        original_input.measured_tasks,
    ) != (plan.task_specs_sha256, plan.converter_revision, plan.raw_count):
        raise ValueError(f"Retention belongs to a different converted population: {source.info.id}")
    return _retained_population(
        source.info,
        receipt,
        name=source.name,
        dataset=dataset,
        inherited_admission=source.current_release if source.declaration.status == "Available" else None,
    )


def _retained_population(
    info: SourceInfo,
    receipt: RetainedReleaseReceipt,
    *,
    name: str,
    dataset: SourceReference,
    inherited_admission: DatasetRelease | None,
) -> RlDataSource[RetainedCohort]:
    publication = receipt.publication
    plan = publication.plan
    # A carried release was converted in its own declared format, not by the retained
    # importer; its population config and import step declare that format (ruling
    # 2026-11-25 ~22:20 EST). A fresh retention keeps the retained importer identity.
    version = RETAINED_IMPORT_VERSION if receipt.carry is None else publication.format
    config = RetainedCohort(
        name=name + "-retained",
        version=version,
        release=publication.release,
        original_dataset=SourceReference(plan.source_dataset_id, plan.source_revision, dataset.url),
        expected_rows=plan.retained_count,
        tasks_parquet_sha256=publication.tasks_parquet_sha256,
        intended_use=IntendedUse.TRAIN if plan.lexical_use is LexicalUse.TRAIN else IntendedUse.EVAL,
        evidence_url=publication.evidence_url,
    )
    info = replace(
        info,
        current_release=publication.release,
        upstream_url=info.upstream_url or dataset.url,
        inherited_admission=inherited_admission,
        dataset=config.dataset,
        count=plan.retained_count,
        within_source_dedup=receipt.retained_dedup,
        lexical_contamination=release_lexical_evidence(publication),
        grading_environment=receipt.grading.evidence,
        converted_examples=None,
        daytona_run_evidence=None,
        notes=(
            f"remediated: dedup/decon. Retained TaskSpec and runtime bytes match the original pin; "
            f"original oversight and quality evidence apply through this subset lineage: {publication.evidence_url}. "
            + info.notes
        ),
    )
    retained = RlDataSource(info=info, config=config, pipeline=import_retained)
    if receipt.carry is not None:
        # The carried release's tasks.parquet carries the taskspec schema (no runtime
        # column); its import reads that schema with the census-measured empty shapes.
        retained = replace(retained, pipeline=import_taskspec_carry)
    retained = bind_actor_census(retained, receipt.actor)
    if receipt.static_retention is None:
        return retained
    return replace(
        retained,
        info=replace(
            retained.info,
            no_oracle_executability_policy=receipt.static_retention.policy,
            no_oracle_static_retention=receipt.static_retention.evidence,
        ),
    )


def bind_archived_retained_release(binding: ArchivedRetainedBinding) -> RlDataSource[RetainedCohort]:
    """Select only the frozen private pin recorded in an authorized archived baseline."""
    record = binding.retained
    pipeline = record.original_pipeline
    assert pipeline.source is not None and pipeline.revision is not None
    dataset = SourceReference(
        pipeline.source,
        pipeline.revision,
        f"https://huggingface.co/datasets/{pipeline.source}/tree/{pipeline.revision}",
    )
    return _retained_population(
        record.original_info,
        record,
        name=pipeline.name,
        dataset=dataset,
        inherited_admission=None,
    )


def measured_intake_source(
    source: RlDataSource, publication: PublicationReceipt, actor: ActorCensusReceipt
) -> RlDataSource:
    """Bind a pending converter's measured original input without adding it to the catalog."""
    source = bind_actor_census(licensed_source(source), actor)
    plan = publication.plan
    if source.info.actor_census_input is None or (
        source.info.id,
        actor.input.task_specs_sha256,
        actor.input.converter_revision,
        actor.input.measured_tasks,
    ) != (plan.source_id, plan.task_specs_sha256, plan.converter_revision, plan.raw_count):
        raise ValueError("Pending retention requires its complete original actor census")
    return bind_within_source_dedup(
        source,
        WithinSourceDedupEvidence(
            raw_count=plan.raw_count,
            unique_count=plan.unique_count,
            within_source_duplicates=plan.within_source_duplicates,
            source_dataset_id=plan.source_dataset_id,
            source_revision=plan.source_revision,
            converter_revision=plan.converter_revision,
            converter_version=plan.converter_version,
            drop_manifest_uri=plan.drop_manifest_uri,
            evidence_url=plan.evidence_url,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--releases", type=Path, nargs="+", required=True)
    parser.add_argument("--censuses", type=Path, nargs="+", required=True)
    parser.add_argument("--grading", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--static-retention-bindings", type=Path, help="Per-source typed policy/evidence objects")
    parser.add_argument("--pending-source", action="append", default=[], help="Explicit pending Atlas source ID")
    args = parser.parse_args()
    static_retention = (
        {}
        if args.static_retention_bindings is None
        else {
            key: StaticRetentionBinding.model_validate(value)
            for key, value in json.loads(args.static_retention_bindings.read_text()).items()
        }
    )
    # Registry loading depends on this module once retained receipts are installed.
    from experiments.post_training.task_curation.sources import (  # noqa: PLC0415
        PENDING_MEASUREMENT_SOURCES,
        all_sources,
    )

    sources = all_sources()
    pending = {source.info.id: source for source in PENDING_MEASUREMENT_SOURCES}
    if unknown := set(args.pending_source) - pending.keys():
        raise ValueError(f"Unknown pending source IDs: {sorted(unknown)}")
    original_actors = actor_census_receipts() if args.pending_source else {}
    censuses = {record["source_id"]: record for path in args.censuses for record in json.loads(path.read_text())}
    grading = {row["source_id"]: RetainedGrading.model_validate(row) for row in json.loads(args.grading.read_text())}
    receipts = {}
    for path in args.releases:
        for record in json.loads(path.read_text()):
            publication = PublicationReceipt.model_validate(record)
            source = (
                measured_intake_source(
                    pending[publication.source_id], publication, original_actors[publication.source_id]
                )
                if publication.source_id in args.pending_source
                else sources[publication.source_id]
            )
            measured = censuses[publication.source_id]
            receipt = RetainedReleaseReceipt(
                source_id=publication.source_id,
                publication=publication,
                actor=ActorCensusReceipt.model_validate(measured["actor"]),
                grading=grading[publication.source_id],
                original_pipeline=source.pipeline_reference,
                original_info=source.info,
                retained_dedup=WithinSourceDedupEvidence.model_validate(measured["dedup"]),
                static_retention=static_retention.get(publication.source_id),
            )
            bound = bind_retained_release(source, receipt)
            if bound.current_release != publication.release:
                raise ValueError("Configured retained release differs from publication")
            if publication.plan.lexical_use is LexicalUse.TRAIN:
                require_private_release(bound.info.lexical_contamination, publication.repo, publication.revision)
            receipts[publication.source_id] = receipt.model_dump(mode="json")
    if unknown := static_retention.keys() - receipts.keys():
        raise ValueError(f"Static retention bindings have no selected publication: {sorted(unknown)}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_receipt_shards(args.output_dir, receipts, records_per_shard=RETAINED_RECEIPTS_PER_SHARD)


if __name__ == "__main__":
    main()
