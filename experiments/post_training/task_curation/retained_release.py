# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import a verified retention release without changing its stored task contracts."""

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from finestore.rl.curation import DatasetRelease, LexicalContaminationEvidence, OversightEvidence
from huggingface_hub import hf_hub_download
from marin.execution.lazy import ArtifactStep
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.audit_schema import NORMALIZED_SCHEMA
from taskcompendium.pipeline.conversion import normalization_columns
from taskcompendium.pipeline.models import IntendedUse, SourceStatus
from taskcompendium.pipeline.source_processing import SourceProcessingMode

from experiments.post_training.task_curation.campaign import CampaignArtifact, PipelineResult
from experiments.post_training.task_curation.config import PipelineOptions
from experiments.post_training.task_curation.decontamination import require_private_release
from experiments.post_training.task_curation.prebuilt import require_runnable_grader
from experiments.post_training.task_curation.source import RlDataSource, SourceReference

RETAINED_IMPORT_VERSION = "2026.11.13.1"
RETAINED_ADMISSION_VERSION = "2026.11.13.2"
RETAINED_SCHEMA = NORMALIZED_SCHEMA.append(pa.field("runtime", pa.large_string()))


@dataclass(frozen=True)
class RetainedCohort:
    name: str
    version: str
    release: DatasetRelease
    original_dataset: SourceReference
    expected_rows: int
    tasks_parquet_sha256: str
    intended_use: IntendedUse
    evidence_url: str

    @property
    def dataset(self) -> SourceReference:
        return SourceReference(self.release.repo, self.release.revision, self.release.url)

    @property
    def files(self) -> tuple[str, ...]:
        return ("tasks.parquet",)


@dataclass(frozen=True)
class RetainedImport:
    cohort: RetainedCohort
    output_path: str
    mode: SourceProcessingMode
    oversight: OversightEvidence | None
    inherited_admission: DatasetRelease | None
    lexical_contamination: LexicalContaminationEvidence | None


def run_retained_import(invocation: RetainedImport) -> CampaignArtifact:
    """Read exact published bytes, preserving the original qualification provenance."""
    cohort = invocation.cohort
    admitted = invocation.mode is not SourceProcessingMode.QUICK
    if admitted:
        if cohort.expected_rows <= 0:
            raise ValueError("An admitted source requires a nonempty retained population")
        require_private_release(invocation.lexical_contamination, cohort.release.repo, cohort.release.revision)
    local = Path(
        hf_hub_download(cohort.release.repo, "tasks.parquet", repo_type="dataset", revision=cohort.release.revision)
    )
    if file_sha256(StoragePath(str(local))) != cohort.tasks_parquet_sha256:
        raise ValueError("Retained release differs from its publication readback receipt")
    output = StoragePath(invocation.output_path)
    normalized = output / "normalize/part-000000.parquet"
    seen = set()
    with normalized.open("wb", auto_mkdir=True) as stream, pq.ParquetWriter(stream, RETAINED_SCHEMA) as writer:
        for batch in pq.ParquetFile(local).iter_batches(batch_size=16):
            records = []
            for row in batch.to_pylist():
                encoded = row["task_spec"]
                task = TaskSpec.model_validate_json(encoded)
                if admitted:
                    require_runnable_grader(task)
                if task.id in seen or hashlib.sha256(encoded.encode()).hexdigest() != row["task_sha256"]:
                    raise ValueError(f"Repeated or changed retained task: {task.id}")
                seen.add(task.id)
                records.append(
                    {
                        **normalization_columns(task.id, task.source, task, None, (), task.source.row),
                        "task_json": encoded,
                        "runtime": row["runtime"],
                    }
                )
            writer.write_table(pa.Table.from_pylist(records, schema=RETAINED_SCHEMA))
    if len(seen) != cohort.expected_rows:
        raise ValueError("Retained release count differs from its frozen plan")
    with local.open("rb") as source, (output / "tasks.parquet").open("wb") as destination:
        shutil.copyfileobj(source, destination)
    outputs = {"tasks": str(output / "tasks.parquet"), "normalize": str(normalized.parent)}
    if admitted:
        final = output / "final/part-000000.parquet"
        with normalized.open("rb") as source, final.open("wb", auto_mkdir=True) as destination:
            shutil.copyfileobj(source, destination)
        outputs["final"] = str(final.parent)
    manifest = {
        "mode": invocation.mode.value,
        "status": SourceStatus.COMPLETED.value,
        "intended_use": cohort.intended_use.value,
        "input_rows": len(seen),
        "converted_rows": len(seen),
        "admitted_rows": len(seen) if admitted else 0,
        "qualification_newly_executed": False,
        "inherited_oversight": invocation.oversight.model_dump(mode="json") if invocation.oversight else None,
        "inherited_admission": (
            invocation.inherited_admission.model_dump(mode="json") if invocation.inherited_admission else None
        ),
        "original_dataset": {
            "repo": cohort.original_dataset.name,
            "revision": cohort.original_dataset.revision,
            "url": cohort.original_dataset.url,
        },
        "lexical_contamination": (
            invocation.lexical_contamination.model_dump(mode="json") if invocation.lexical_contamination else None
        ),
        "rejections": {},
        "repo": cohort.release.repo,
        "revision": cohort.release.revision,
        "tasks_parquet_sha256": cohort.tasks_parquet_sha256,
        "retention_evidence": cohort.evidence_url,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return CampaignArtifact(
        status="completed",
        result=PipelineResult(
            status=SourceStatus.COMPLETED,
            outputs=outputs,
            evidence={"manifest": str(output / "manifest.json")},
            stages=("download", "validate", "normalize"),
        ),
    )


def import_retained(source: RlDataSource[RetainedCohort], options: PipelineOptions) -> ArtifactStep[CampaignArtifact]:
    """Admit a verified subset through its existing qualification and fresh release gates."""
    if options.mode is not SourceProcessingMode.QUICK:
        if "excluded" in source.info.tags:
            raise ValueError(f"Retained cohort is excluded pending qualification: {source.name}")
        if source.admission_blockers:
            raise ValueError(f"Source {source.name} cannot admit tasks: {'; '.join(source.admission_blockers)}")
    config = source.config
    assert config is not None

    def build_config(context) -> RetainedImport:
        return RetainedImport(
            config,
            context.output_path,
            options.mode,
            source.info.oversight,
            source.info.inherited_admission,
            source.info.lexical_contamination,
        )

    return ArtifactStep(
        name=f"retained/{options.mode.value}/{config.release.repo}/{config.release.revision}",
        version=RETAINED_IMPORT_VERSION if options.mode is SourceProcessingMode.QUICK else RETAINED_ADMISSION_VERSION,
        artifact_type=CampaignArtifact,
        run=run_retained_import,
        build_config=build_config,
        deps=(),
    )
