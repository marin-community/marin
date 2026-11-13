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
from finestore.rl.curation import DatasetRelease
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
from experiments.post_training.task_curation.source import RlDataSource, SourceReference

RETAINED_IMPORT_VERSION = "2026.11.13.1"
RETAINED_SCHEMA = NORMALIZED_SCHEMA.append(pa.field("runtime", pa.large_string()))


@dataclass(frozen=True)
class RetainedCohort:
    name: str
    version: str
    release: DatasetRelease
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


def run_retained_import(invocation: RetainedImport) -> CampaignArtifact:
    """Stream the exact published TaskSpec bytes into the converted inspection view."""
    cohort = invocation.cohort
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
    manifest = {
        "mode": SourceProcessingMode.QUICK.value,
        "status": SourceStatus.COMPLETED.value,
        "intended_use": cohort.intended_use.value,
        "input_rows": len(seen),
        "converted_rows": len(seen),
        "admitted_rows": 0,
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
            outputs={"tasks": str(output / "tasks.parquet"), "normalize": str(normalized.parent)},
            evidence={"manifest": str(output / "manifest.json")},
            stages=("download", "validate", "normalize"),
        ),
    )


def import_retained(source: RlDataSource[RetainedCohort], options: PipelineOptions) -> ArtifactStep[CampaignArtifact]:
    """Expose mechanical retention for inspection while execution qualification remains separate."""
    if options.mode is not SourceProcessingMode.QUICK:
        raise ValueError("Retention publication requires separate SAMPLE/FULL execution qualification")
    config = source.config
    assert config is not None

    def build_config(context) -> RetainedImport:
        return RetainedImport(config, context.output_path)

    return ArtifactStep(
        name=f"retained/{config.release.repo}/{config.release.revision}",
        version=RETAINED_IMPORT_VERSION,
        artifact_type=CampaignArtifact,
        run=run_retained_import,
        build_config=build_config,
        deps=(),
    )
