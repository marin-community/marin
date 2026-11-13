# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind read-back retention releases while preserving the original source evidence."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from finestore.rl.curation import (
    DatasetRelease,
    DeconReceiptStatus,
    DecontaminatedRelease,
    LexicalContaminationEvidence,
    LexicalReceipt,
    LexicalUse,
    OracleValidation,
    PipelineReference,
    WithinSourceDedupEvidence,
)
from pydantic import BaseModel, ConfigDict, model_validator
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.census_binding import ActorCensusReceipt, bind_actor_census
from experiments.post_training.task_curation.decontamination import require_private_release
from experiments.post_training.task_curation.receipt_shards import write_receipt_shards
from experiments.post_training.task_curation.remediation_publish import ReleasePlan
from experiments.post_training.task_curation.retained_grading import RetainedGradingReceipt
from experiments.post_training.task_curation.retained_release import (
    RETAINED_IMPORT_VERSION,
    RetainedCohort,
    import_retained,
)
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, TaskFormat


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


class RetainedReleaseReceipt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str
    publication: PublicationReceipt
    actor: ActorCensusReceipt
    grading: RetainedGradingReceipt
    original_pipeline: PipelineReference
    original_info: SourceInfo
    retained_dedup: WithinSourceDedupEvidence

    @model_validator(mode="after")
    def _exact_retained_pin(self) -> "RetainedReleaseReceipt":
        publication = self.publication
        if self.source_id != publication.source_id or self.actor.source_id != self.source_id:
            raise ValueError("Retention lineage contains another source")
        if self.grading.source_id != self.source_id or self.grading.input != self.actor.input:
            raise ValueError("Retained grading evidence belongs to another actor census input")
        if (
            self.actor.input.dataset_id,
            self.actor.input.dataset_revision,
            self.actor.input.converter_version,
            self.actor.input.measured_tasks,
        ) != (publication.repo, publication.revision, RETAINED_IMPORT_VERSION, publication.plan.retained_count):
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
            RETAINED_IMPORT_VERSION,
            publication.plan.retained_count,
            publication.plan.retained_count,
            0,
        ):
            raise ValueError("Active dedup evidence must describe the retained population")
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
    config = RetainedCohort(
        name=source.name + "-retained",
        version=RETAINED_IMPORT_VERSION,
        release=publication.release,
        expected_rows=plan.retained_count,
        tasks_parquet_sha256=publication.tasks_parquet_sha256,
        intended_use=IntendedUse.TRAIN if plan.lexical_use is LexicalUse.TRAIN else IntendedUse.EVAL,
        evidence_url=publication.evidence_url,
    )
    info = replace(
        source.info,
        current_release=publication.release,
        dataset=config.dataset,
        count=plan.retained_count,
        task_format=TaskFormat.TASKCOMPENDIUM,
        within_source_dedup=receipt.retained_dedup,
        lexical_contamination=release_lexical_evidence(publication),
        oversight=None,
        oracle_validation=OracleValidation.SKIPPED,
        quality_result=None,
        independent_qc=None,
        grading_environment=receipt.grading.evidence,
        converted_examples=None,
        daytona_run_evidence=None,
        notes=(
            source.info.notes + f" Retention lineage and exact readback: {publication.evidence_url}. "
            "Original conversion and qualification remain archived; this release adds mechanical retention only."
        ),
    )
    retained = RlDataSource(info=info, config=config, pipeline=import_retained)
    return bind_actor_census(retained, receipt.actor)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--releases", type=Path, nargs="+", required=True)
    parser.add_argument("--censuses", type=Path, nargs="+", required=True)
    parser.add_argument("--grading", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    # Registry loading depends on this module once retained receipts are installed.
    from experiments.post_training.task_curation.sources import all_sources  # noqa: PLC0415

    sources = all_sources()
    censuses = {record["source_id"]: record for path in args.censuses for record in json.loads(path.read_text())}
    grading = {
        row["source_id"]: RetainedGradingReceipt.model_validate(row) for row in json.loads(args.grading.read_text())
    }
    receipts = {}
    for path in args.releases:
        for record in json.loads(path.read_text()):
            publication = PublicationReceipt.model_validate(record)
            source = sources[publication.source_id]
            measured = censuses[publication.source_id]
            receipt = RetainedReleaseReceipt(
                source_id=publication.source_id,
                publication=publication,
                actor=ActorCensusReceipt.model_validate(measured["actor"]),
                grading=grading[publication.source_id],
                original_pipeline=source.pipeline_reference,
                original_info=source.info,
                retained_dedup=WithinSourceDedupEvidence.model_validate(measured["dedup"]),
            )
            bound = bind_retained_release(source, receipt)
            if bound.current_release != publication.release:
                raise ValueError("Configured retained release differs from publication")
            if publication.plan.lexical_use is LexicalUse.TRAIN:
                require_private_release(bound.info.lexical_contamination, publication.repo, publication.revision)
            receipts[publication.source_id] = receipt.model_dump(mode="json")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_receipt_shards(args.output_dir, receipts)


if __name__ == "__main__":
    main()
