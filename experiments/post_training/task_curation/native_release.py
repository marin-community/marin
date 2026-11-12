# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import private native Harbor inspection releases without claiming admission."""

import hashlib
import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from marin.execution.lazy import ArtifactStep
from taskcompendium.grading_environment import (
    GradingEnvironmentStatus,
    assess_grading_environment,
    count_grading_environment,
    grading_environment_census,
)
from taskcompendium.models import TaskSpec, missing_environment
from taskcompendium.pipeline.audit_schema import NORMALIZED_SCHEMA
from taskcompendium.pipeline.conversion import normalization_columns
from taskcompendium.pipeline.models import SourceStatus
from taskcompendium.pipeline.source_processing import SourceProcessingMode

from experiments.post_training.task_curation.campaign import CampaignArtifact, PipelineResult
from experiments.post_training.task_curation.config import PipelineOptions
from experiments.post_training.task_curation.prebuilt import PrebuiltCohort
from experiments.post_training.task_curation.source import RlDataSource


@dataclass(frozen=True)
class NativeQuickImport:
    output_path: str
    cohort: PrebuiltCohort


def run_native_import(invocation: NativeQuickImport) -> CampaignArtifact:
    config = invocation.cohort
    local = Path(hf_hub_download(config.repo, "tasks.parquet", repo_type="dataset", revision=config.revision))
    provenance = hf_hub_download(config.repo, "manifest.json", repo_type="dataset", revision=config.revision)
    parquet = pq.ParquetFile(local)
    if parquet.metadata.num_rows != config.expected_rows:
        raise ValueError("Native release population differs from its pinned declaration")
    output = Path(invocation.output_path)
    normalized = output / "normalize" / "part-00000.parquet"
    normalized.parent.mkdir(parents=True)
    seen = set()
    counts: Counter[str] = Counter()
    with pq.ParquetWriter(normalized, NORMALIZED_SCHEMA) as writer:
        for batch in parquet.iter_batches(batch_size=32):
            records = []
            for row in batch.to_pylist():
                serialized = row["task_spec"]
                if hashlib.sha256(serialized.encode()).hexdigest() != row["task_sha256"]:
                    raise ValueError("Native TaskSpec differs from its published content hash")
                task = TaskSpec.model_validate_json(serialized)
                if task.id in seen:
                    raise ValueError(f"Duplicate native task ID: {task.id}")
                seen.add(task.id)
                if "harbor" not in task.tags or missing_environment(task.model_dump(mode="json")):
                    raise ValueError(f"Incomplete native TaskSpec: {task.id}")
                assessment = assess_grading_environment(task)
                if assessment.status is GradingEnvironmentStatus.MISMATCH:
                    raise ValueError(f"Native grading environment mismatch: {task.id}")
                count_grading_environment(counts, assessment.model_dump(mode="json"))
                records.append(
                    {
                        **normalization_columns(task.id, task.source, task, None, (), row["path"]),
                        "task_json": serialized,
                    }
                )
            writer.write_table(pa.Table.from_pylist(records, schema=NORMALIZED_SCHEMA))
    shutil.copyfile(local, output / "tasks.parquet")
    shutil.copyfile(provenance, output / "upstream-conversion-manifest.json")
    manifest = {
        "mode": "quick",
        "intended_use": "train",
        "task_format": "harbor",
        "input_rows": len(seen),
        "converted_rows": len(seen),
        "rejections": {},
        "source_dataset": config.repo,
        "source_revision": config.revision,
        "recipe_revision": config.version,
        "reviewed": False,
        "verified": False,
        "grading_environment": grading_environment_census(counts),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return CampaignArtifact(
        status="completed",
        result=PipelineResult(
            status=SourceStatus.COMPLETED,
            outputs={"tasks": str(output / "tasks.parquet"), "normalize": str(normalized.parent)},
            evidence={
                "manifest": str(output / "manifest.json"),
                "upstream_conversion": str(output / "upstream-conversion-manifest.json"),
            },
            stages=("download", "validate", "convert"),
        ),
    )


def import_native_quick(
    source: RlDataSource[PrebuiltCohort], options: PipelineOptions
) -> ArtifactStep[CampaignArtifact]:
    """Register a lossless native import whose release has only QUICK evidence."""
    if options.mode is not SourceProcessingMode.QUICK:
        raise ValueError("Native inspection releases require separate SAMPLE/FULL qualification before admission")
    config = source.config
    assert config is not None

    def build_config(context) -> NativeQuickImport:
        return NativeQuickImport(context.output_path, config)

    return ArtifactStep(
        name=config.name,
        version=config.version,
        artifact_type=CampaignArtifact,
        run=run_native_import,
        build_config=build_config,
        deps=(),
    )
