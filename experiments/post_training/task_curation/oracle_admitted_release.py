# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind qualified oracle subsets to their freshly measured private pins."""

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Literal

import pyarrow.parquet as pq
from finestore.rl.curation import (
    ContentEvidenceReference,
    DatasetRelease,
    LexicalContaminationEvidence,
    OracleFiltrationEvidence,
    OracleValidation,
    OversightEvidence,
    OversightLevel,
    PipelineReference,
    WithinSourceDedupEvidence,
)
from huggingface_hub import HfApi, hf_hub_download
from pydantic import BaseModel, ConfigDict, Field, model_validator
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.execution_applicability import is_nonagentic_task
from taskcompendium.grading_environment import shared_integrity_issue
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.oracle_filter import OracleExclusionReason, require_oracle_filter

from experiments.post_training.task_curation.census_binding import ActorCensusReceipt, bind_actor_census
from experiments.post_training.task_curation.decontamination import require_private_release
from experiments.post_training.task_curation.full_content_binding import (
    bind_full_content_result,
    matches_full_content_input,
)
from experiments.post_training.task_curation.full_content_results import FullContentResultBinding
from experiments.post_training.task_curation.oracle_release import verify_oracle_release
from experiments.post_training.task_curation.prebuilt import (
    PrebuiltCohort,
    import_oracle_harbor_cohort,
    import_oracle_harbor_prebuilt,
    import_taskcompendium_cohort,
    import_taskcompendium_prebuilt,
    require_runnable_grader,
)
from experiments.post_training.task_curation.receipt_shards import write_receipt_shards
from experiments.post_training.task_curation.retained_grading import RetainedGradingReceipt
from experiments.post_training.task_curation.sample_qc_binding import SampleQCReceipt, bind_sample_result
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, TaskFormat


class OracleAdmittedReleaseReceipt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str
    task_format: Literal[TaskFormat.HARBOR, TaskFormat.TASKCOMPENDIUM]
    original_pipeline: PipelineReference
    original_info: SourceInfo
    filtration: OracleFiltrationEvidence
    execution_release: DatasetRelease | None
    output_dedup: WithinSourceDedupEvidence
    actor: ActorCensusReceipt
    grading: RetainedGradingReceipt
    lexical: LexicalContaminationEvidence
    sample: SampleQCReceipt
    repair_evidence: tuple[ContentEvidenceReference, ...] = Field(min_length=1)
    consumer: ContentEvidenceReference
    full_content: FullContentResultBinding | None = None

    @model_validator(mode="after")
    def _qualified_population(self) -> "OracleAdmittedReleaseReceipt":
        evidence, actor = self.filtration, self.actor.input
        release = evidence.filtered_release
        original = self.original_info.within_source_dedup
        if (
            self.source_id != self.original_info.id
            or original is None
            or original.unique_count != evidence.original_count
        ):
            raise ValueError("Oracle release does not cover its original source population")
        if (self.original_pipeline.source, self.original_pipeline.revision) != (
            evidence.original_release.repo,
            evidence.original_release.revision,
        ):
            raise ValueError("Oracle release belongs to another original pin")
        if evidence.admitted_count <= 0 or evidence.oracle_filter_version != 4:
            raise ValueError("Admissions require nonempty current oracle/control filtration")
        if any(
            reason in evidence.exclusion_counts
            for reason in (OracleExclusionReason.INFRASTRUCTURE, OracleExclusionReason.NEGATIVE_CONTROL_PENDING)
        ):
            raise ValueError("Pending source/control outcomes cannot qualify a release")
        if (self.source_id, release.repo, release.revision, evidence.admitted_count) != (
            self.actor.source_id,
            actor.dataset_id,
            actor.dataset_revision,
            actor.measured_tasks,
        ):
            raise ValueError("Fresh actor census belongs to another filtered pin")
        if self.grading.source_id != self.source_id or self.grading.input != actor:
            raise ValueError("Grading census belongs to another filtered population")
        required_shared = self.grading.evidence.required_counts.get("shared", 0)
        integrity = self.actor.integrity_contracts
        if required_shared and (
            integrity is None
            or integrity.required_shared_tasks != required_shared
            or integrity.shared_missing_contract_tasks
        ):
            raise ValueError("SHARED integrity census does not qualify the filtered population")
        dedup = self.output_dedup
        if (
            dedup.source_dataset_id,
            dedup.source_revision,
            dedup.converter_revision,
            dedup.converter_version,
            dedup.raw_count,
            dedup.unique_count,
            dedup.within_source_duplicates,
        ) != (
            release.repo,
            release.revision,
            actor.converter_revision,
            actor.converter_version,
            evidence.admitted_count,
            evidence.admitted_count,
            0,
        ):
            raise ValueError("Output uniqueness census differs from the fresh actor input")
        lexical = self.lexical
        if (
            lexical.input_dataset_id,
            lexical.input_dataset_revision,
            lexical.converter_revision,
            lexical.measured_tasks,
        ) != (
            release.repo,
            release.revision,
            actor.converter_revision,
            actor.measured_tasks,
        ):
            raise ValueError("Lexical measurement belongs to another filtered population")
        require_private_release(lexical, release.repo, release.revision)
        if lexical.receipt is None or lexical.receipt.retained_tasks != evidence.admitted_count:
            raise ValueError("Lexical receipt does not cover the published subset")
        panel, review = self.sample.review.panel, self.sample.review
        if (
            self.sample.source_id,
            panel.dataset_id,
            panel.dataset_revision,
            panel.converter_version,
            panel.task_specs_sha256,
            panel.retained_tasks,
        ) != (
            self.source_id,
            release.repo,
            release.revision,
            actor.converter_version,
            actor.task_specs_sha256,
            actor.measured_tasks,
        ):
            raise ValueError("SAMPLE content review belongs to another filtered population")
        if self.full_content is not None and (
            self.full_content.source_id != self.source_id or self.full_content.input != actor
        ):
            raise ValueError("Whole-source content review belongs to another filtered population")
        sample_size = min(100, evidence.admitted_count)
        if (
            not review.panel_coverage_complete
            or review.attempted_count != sample_size
            or panel.sample_seed != 0
            or panel.sample_size != sample_size
        ):
            raise ValueError("Filtered SAMPLE observations are incomplete")
        return self


def bind_oracle_admitted_release(
    source: RlDataSource, receipt: OracleAdmittedReleaseReceipt
) -> RlDataSource[PrebuiltCohort]:
    """Select the native consumer with qualification bound to the new population."""
    evidence = receipt.filtration
    if (
        source.info.id != receipt.source_id
        or source.pipeline_reference != receipt.original_pipeline
        or source.current_release != evidence.original_release
        or source.info.within_source_dedup != receipt.original_info.within_source_dedup
    ):
        raise ValueError("Oracle release belongs to another configured original population")
    release = evidence.filtered_release
    config = PrebuiltCohort(
        name=source.name + "-oracle-filtered",
        version=receipt.actor.input.converter_version,
        repo=release.repo,
        revision=release.revision,
        expected_rows=evidence.admitted_count,
        intended_use=source.config.intended_use,
    )
    info = replace(
        source.info,
        current_release=release,
        inherited_admission=None,
        dataset=config.dataset,
        count=evidence.admitted_count,
        task_format=receipt.task_format,
        oracle_filtration=evidence,
        within_source_dedup=receipt.output_dedup,
        lexical_contamination=receipt.lexical,
        grading_environment=receipt.grading.evidence,
        oracle_validation=OracleValidation.EXECUTED,
        oversight=OversightEvidence(
            level=OversightLevel.SAMPLE,
            campaign=evidence.validation.uri,
            recipe_revision=receipt.actor.input.converter_revision,
            reviewed=receipt.sample.review.attempted_count,
            admitted=evidence.admitted_count,
            verification=(
                f"Full original oracles and workspace no-op controls: {evidence.oracle_filter.uri}; "
                f"fresh private-pin census and native consumer: {receipt.consumer.uri}; "
                f"seed-zero SAMPLE content: {receipt.sample.review.path}."
            ),
        ),
        quality_result=None,
        quality_result_finalization=None,
        quality_result_waiver=None,
        full_content_result_binding=None,
        independent_qc=None,
        converted_examples=None,
        daytona_run_evidence=evidence.validation.uri,
        tags=tuple(tag for tag in source.info.tags if tag not in {"excluded", "deprecated"}),
        notes=(
            f"Oracle/control filtration retained {evidence.admitted_count}/{evidence.original_count}. "
            f"Typed exclusions: {dict(evidence.exclusion_counts)}. "
            "Original source and prior retention evidence remain archived in the release receipt. " + source.info.notes
        ),
    )
    pipeline = (
        import_oracle_harbor_prebuilt if receipt.task_format is TaskFormat.HARBOR else import_taskcompendium_prebuilt
    )
    bound = bind_actor_census(RlDataSource(info=info, config=config, pipeline=pipeline), receipt.actor)
    return bind_full_content_result(bind_sample_result(bound, receipt.sample), receipt.full_content)


def bind_selected_source_content(
    original: RlDataSource,
    selected: RlDataSource,
    sample: SampleQCReceipt | None,
    full_content: FullContentResultBinding | None,
) -> RlDataSource:
    """Attach current content after pin selection and validate archived original evidence."""
    dataset = selected.dataset
    if sample is not None and (
        dataset is None
        or (sample.review.panel.dataset_id, sample.review.panel.dataset_revision, sample.review.panel.converter_version)
        != (dataset.name, dataset.revision, selected.version)
    ):
        bind_sample_result(original, sample)
    else:
        selected = bind_sample_result(selected, sample)
    if full_content is not None and not matches_full_content_input(selected, full_content):
        bind_full_content_result(original, full_content)
    else:
        selected = bind_full_content_result(selected, full_content)
    return selected


class OracleAdmittedReleaseInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str
    task_format: Literal[TaskFormat.HARBOR, TaskFormat.TASKCOMPENDIUM]
    filtration: OracleFiltrationEvidence
    execution_release: DatasetRelease | None
    qualification: ContentEvidenceReference
    repair_evidence: tuple[ContentEvidenceReference, ...] = Field(min_length=1)
    original_cohort: Path
    oracle_filter: Path
    output_cohort: Path


def main() -> None:
    # Receipt reconstruction loads the source inventory through its census dependencies.
    from experiments.post_training.task_curation.sample_qc_receipts import sample_qc_receipt  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    # Source declarations import the installed oracle-release binders.
    from experiments.post_training.task_curation.sources import sources_before_oracle_filtration  # noqa: PLC0415

    item = OracleAdmittedReleaseInput.model_validate_json(args.manifest.read_bytes())
    source = sources_before_oracle_filtration()[item.source_id]
    release = item.filtration.filtered_release
    if not HfApi().repo_info(release.repo, repo_type="dataset", revision=release.revision).private:
        raise ValueError("Filtered publication must remain private")
    if item.execution_release is not None:
        executed = item.execution_release
        input_path = hf_hub_download(executed.repo, "tasks.parquet", repo_type="dataset", revision=executed.revision)
        if file_sha256(StoragePath(input_path)) != item.filtration.original_cohort.sha256:
            raise ValueError("Execution-input pin differs from the canonical pre-filtration cohort")
    qualified = StoragePath(item.qualification.uri)
    if file_sha256(qualified) != item.qualification.sha256:
        raise ValueError("Fresh qualification metadata bytes changed")
    metadata = json.loads(qualified.read_text())
    if metadata["source_id"] != item.source_id or metadata["release"] != item.filtration.filtered_release.model_dump(
        mode="json"
    ):
        raise ValueError("Fresh qualification describes another source pin")
    observed_sample = sample_qc_receipt(metadata["sample"]["review"]["path"])
    if observed_sample != SampleQCReceipt.model_validate(metadata["sample"]):
        raise ValueError("Fresh SAMPLE binding differs from its physical observations")
    for evidence in item.repair_evidence:
        if file_sha256(StoragePath(evidence.uri)) != evidence.sha256:
            raise ValueError("Source repair provenance bytes changed")
    receipt = OracleAdmittedReleaseReceipt(
        source_id=item.source_id,
        task_format=item.task_format,
        original_pipeline=source.pipeline_reference,
        original_info=source.info,
        filtration=item.filtration,
        execution_release=item.execution_release,
        output_dedup=WithinSourceDedupEvidence.model_validate(metadata["dedup"]),
        actor=ActorCensusReceipt.model_validate(metadata["actor"]),
        grading=RetainedGradingReceipt.model_validate(metadata["grading"]),
        lexical=LexicalContaminationEvidence.model_validate(metadata["lexical"]),
        sample=observed_sample,
        repair_evidence=item.repair_evidence,
        consumer=item.qualification,
        full_content=(
            FullContentResultBinding.model_validate(metadata["full_content"])
            if metadata.get("full_content") is not None
            else None
        ),
    )
    verified = verify_oracle_release(item.filtration, item.original_cohort, item.oracle_filter, item.output_cohort)
    if verified.version != 4:
        raise ValueError("Admitted releases require current complete oracle/control evidence")
    if receipt.full_content is not None:
        full = receipt.full_content
        for proof in (full.contract_index, full.review_receipt, full.records, full.waiver):
            if proof is not None and file_sha256(StoragePath(proof.uri)) != proof.sha256:
                raise ValueError("Whole-source content evidence bytes changed")
    bound = bind_oracle_admitted_release(source, receipt)
    load = import_oracle_harbor_cohort if receipt.task_format is TaskFormat.HARBOR else import_taskcompendium_cohort
    published = Path(load(bound.config))
    if file_sha256(StoragePath(str(published))) != item.filtration.tasks_parquet.sha256:
        raise ValueError("Consumer readback differs from the filtered output")
    for batch in pq.ParquetFile(published).iter_batches(batch_size=32):
        for row in batch.to_pylist():
            task = TaskSpec.model_validate_json(row["task_spec"])
            require_runnable_grader(task)
            if (not is_nonagentic_task(task) and task.execution_requirements is None) or shared_integrity_issue(task):
                raise ValueError(f"Published task lacks its required execution/integrity contract: {task.id}")
            if not require_oracle_filter(task, verified, json.loads(row["runtime"])):
                raise ValueError(f"Published task/runtime lacks canonical oracle admission: {task.id}")
    if bound.current_release != item.filtration.filtered_release or bound.admission_blockers:
        raise ValueError(f"Filtered source remains unqualified: {bound.admission_blockers}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_receipt_shards(args.output_dir, {item.source_id: receipt.model_dump(mode="json")})


if __name__ == "__main__":
    main()
