# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind private oracle-filtered releases independently of lexical retention."""

from collections import Counter
from dataclasses import replace
from pathlib import Path

import pyarrow.parquet as pq
from finestore.rl.curation import (
    OracleFiltrationEvidence,
    PipelineReference,
    WithinSourceDedupEvidence,
)
from pydantic import BaseModel, ConfigDict, model_validator
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.identity import canonical_sha256
from taskcompendium.pipeline.models import IntendedUse
from taskcompendium.pipeline.oracle_filter import MechanicalOracleTaskReceipt, OracleExclusionReason, OracleFilterReceipt

from experiments.post_training.task_curation.census_binding import bind_actor_census
from experiments.post_training.task_curation.oracle_filter import row_runtime, row_runtime_hash, row_task
from experiments.post_training.task_curation.retained_release import (
    RETAINED_IMPORT_VERSION,
    RetainedCohort,
    import_retained,
)
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, SourceReference, TaskFormat


class OracleFilteredReleaseReceipt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str
    original_pipeline: PipelineReference
    original_info: SourceInfo
    filtration: OracleFiltrationEvidence
    output_dedup: WithinSourceDedupEvidence

    @model_validator(mode="after")
    def _exact_population(self) -> "OracleFilteredReleaseReceipt":
        evidence, dedup = self.filtration, self.output_dedup
        if self.source_id != self.original_info.id:
            raise ValueError("Oracle filtration belongs to another source")
        if (self.original_pipeline.source, self.original_pipeline.revision) != (
            evidence.original_release.repo,
            evidence.original_release.revision,
        ):
            raise ValueError("Oracle filtration belongs to another original pin")
        original = self.original_info.within_source_dedup
        if original is None or original.unique_count != evidence.original_count:
            raise ValueError("Oracle filtration does not cover the original retained population")
        for reason in evidence.exclusion_counts:
            OracleExclusionReason(reason)
        if (
            dedup.source_dataset_id,
            dedup.source_revision,
            dedup.converter_version,
            dedup.raw_count,
            dedup.unique_count,
            dedup.within_source_duplicates,
        ) != (
            evidence.filtered_release.repo,
            evidence.filtered_release.revision,
            RETAINED_IMPORT_VERSION,
            evidence.admitted_count,
            evidence.admitted_count,
            0,
        ):
            raise ValueError("Output uniqueness evidence does not cover the filtered pin")
        return self


def verify_oracle_release(
    evidence: OracleFiltrationEvidence, original_cohort: Path, oracle_filter: Path, output_cohort: Path
) -> OracleFilterReceipt:
    """Verify complete source IDs, canonical outcomes and exact published task/runtime bytes."""
    for path, reference in (
        (original_cohort, evidence.original_cohort),
        (oracle_filter, evidence.oracle_filter),
        (output_cohort, evidence.tasks_parquet),
    ):
        if file_sha256(StoragePath(str(path))) != reference.sha256:
            raise ValueError("Oracle publication evidence bytes changed")
    receipt = OracleFilterReceipt.model_validate_json(oracle_filter.read_text())
    if (receipt.version, receipt.input_cohort_sha256, receipt.input_rows, receipt.conversion_rejections) != (
        evidence.oracle_filter_version,
        evidence.original_cohort.sha256,
        evidence.original_count,
        0,
    ):
        raise ValueError("Canonical oracle receipt belongs to another complete input")
    seen = set()
    for batch in pq.ParquetFile(original_cohort).iter_batches(batch_size=16):
        for row in batch.to_pylist():
            task = row_task(row)
            if task.id in seen or task.id not in receipt.tasks:
                raise ValueError("Original task coverage is duplicated or incomplete")
            entry = receipt.tasks[task.id]
            source_sha = (
                entry.task_sha256 if isinstance(entry, MechanicalOracleTaskReceipt) else entry.source_task_sha256
            )
            if (
                source_sha != canonical_sha256(task.model_dump(mode="json"))
                or row_runtime_hash(row) != entry.runtime_sha256
            ):
                raise ValueError("Original task/runtime differs from canonical filtration")
            seen.add(task.id)
    if seen != receipt.tasks.keys() or len(seen) != evidence.original_count:
        raise ValueError("Oracle receipt omits original IDs")
    kept = set()
    for batch in pq.ParquetFile(output_cohort).iter_batches(batch_size=16):
        for row in batch.to_pylist():
            task = row_task(row)
            if task.id in kept or not receipt.admits(task, row_runtime(row)):
                raise ValueError("Published task/runtime lacks exact canonical admission")
            kept.add(task.id)
    if kept != {tid for tid, entry in receipt.tasks.items() if entry.passed} or len(kept) != evidence.admitted_count:
        raise ValueError("Published subset differs from canonical passing outcomes")
    counts = Counter(
        entry.exclusion_reason.value for entry in receipt.tasks.values() if entry.exclusion_reason is not None
    )
    if dict(counts) != evidence.exclusion_counts:
        raise ValueError("Published exclusions differ from canonical typed outcomes")
    if evidence.admitted_count and receipt.version != 4:
        raise ValueError("New admitted releases require current oracle/control evidence")
    if any(reason in counts for reason in (OracleExclusionReason.INFRASTRUCTURE, "negative_control_pending")):
        raise ValueError("Unresolved oracle/control infrastructure cannot be published")
    return receipt


def bind_oracle_filtered_release(
    source: RlDataSource, receipt: OracleFilteredReleaseReceipt
) -> RlDataSource[RetainedCohort]:
    """Select verified output bytes and clear qualification from the previous population."""
    evidence = receipt.filtration
    if source.info.id != receipt.source_id or source.pipeline_reference != receipt.original_pipeline:
        raise ValueError("Oracle release belongs to another configured conversion")
    if (
        source.current_release != evidence.original_release
        or source.info.within_source_dedup != receipt.original_info.within_source_dedup
    ):
        raise ValueError("Oracle release belongs to another original population")
    config = RetainedCohort(
        name=source.name + "-oracle-filtered",
        version=RETAINED_IMPORT_VERSION,
        release=evidence.filtered_release,
        original_dataset=SourceReference(
            evidence.original_release.repo, evidence.original_release.revision, evidence.original_release.url
        ),
        expected_rows=evidence.admitted_count,
        tasks_parquet_sha256=evidence.tasks_parquet.sha256,
        intended_use=IntendedUse.EVAL if "eval" in source.info.tags else IntendedUse.TRAIN,
        evidence_url=evidence.validation.uri,
    )
    tags = source.info.tags
    if evidence.admitted_count == 0:
        tags = tuple(tag for tag in tags if tag != "deprecated")
        if "excluded" not in tags:
            tags = (*tags, "excluded")
    info = replace(
        source.info,
        current_release=evidence.filtered_release,
        inherited_admission=None,
        dataset=config.dataset,
        count=evidence.admitted_count,
        task_format=TaskFormat.TASKCOMPENDIUM,
        oracle_filtration=evidence,
        within_source_dedup=receipt.output_dedup,
        lexical_contamination=None,
        grading_environment=None,
        oracle_validation=None,
        oversight=None,
        quality_result=None,
        quality_result_finalization=None,
        quality_result_waiver=None,
        full_content_result_binding=None,
        independent_qc=None,
        converted_examples=None,
        daytona_run_evidence=None,
        tags=tags,
        notes=(
            f"Oracle filtration retained {evidence.admitted_count}/{evidence.original_count}; "
            f"typed exclusions: {dict(evidence.exclusion_counts)}. Evidence: {evidence.validation.uri}. "
            "Earlier qualification and retention accounting are archived in the oracle release receipt. "
            + source.info.notes
        ),
    )
    return bind_actor_census(RlDataSource(info=info, config=config, pipeline=import_retained), None)
