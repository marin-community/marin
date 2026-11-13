# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import an already-published TaskCompendium cohort as a runnable source.

A converted dataset is published once and then read back, never regenerated: the
published revision is the artifact a reader and a run both consume. This stage
downloads the pinned revision through the cached Hugging Face login, validates every
row as a task spec under the admission predicates, and records what it admitted.

It converts nothing, so it cannot improve on the published cohort's review; the
source's oversight level and evidence describe the qualification the publish recorded.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from marin.execution.lazy import ArtifactStep
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.pipeline.audit_schema import NORMALIZED_SCHEMA
from taskcompendium.pipeline.conversion import normalization_columns
from taskcompendium.pipeline.models import SourceStatus
from taskcompendium.pipeline.source_processing import SourceProcessingMode

from experiments.post_training.task_curation.calibforge_publish import TASKS_FILENAME
from experiments.post_training.task_curation.campaign import CampaignArtifact, PipelineResult
from experiments.post_training.task_curation.config import PipelineOptions
from experiments.post_training.task_curation.environment_fidelity import missing_environment
from experiments.post_training.task_curation.lowered_envelope import LOWERED_TASK_SPEC_COLUMN, LoweredSpec
from experiments.post_training.task_curation.source import EXCLUDED_TAG, RlDataSource, SourceReference


@dataclass(frozen=True)
class PrebuiltCohort:
    """Where a published cohort lives and how much of it this source expects."""

    name: str
    version: str
    repo: str
    revision: str
    expected_rows: int

    @property
    def dataset(self) -> SourceReference:
        return SourceReference(
            self.repo, self.revision, f"https://huggingface.co/datasets/{self.repo}/tree/{self.revision}"
        )

    @property
    def files(self) -> tuple[str, ...]:
        return (TASKS_FILENAME,)


def import_cohort(config: PrebuiltCohort) -> str:
    """Download the pinned cohort, validate every row, and return the local path.

    A row without a usable environment, or a cohort whose size differs from the
    recorded count, fails here: admitting it would put tasks into training that the
    published cohort never qualified.
    """
    local = Path(
        hf_hub_download(repo_id=config.repo, repo_type="dataset", filename=TASKS_FILENAME, revision=config.revision)
    )
    rows = pq.read_table(local).to_pylist()
    if len(rows) != config.expected_rows:
        raise ValueError(f"{config.repo}@{config.revision} has {len(rows)} rows, expected {config.expected_rows}")
    for row in rows:
        # A lowered row is the rollout envelope: the task spec plus its machine shapes.
        envelope = json.loads(row[LOWERED_TASK_SPEC_COLUMN])
        lowered = LoweredSpec.from_document(envelope)
        if not envelope.get("runtime"):
            raise ValueError(f"{row['path']}: envelope carries no runtime machines")
        reason = missing_environment(lowered.task)
        if reason is not None:
            raise ValueError(f"{row['path']}: {reason}")
    return str(local)


@dataclass(frozen=True)
class PrebuiltImport:
    output_path: str
    cohort: PrebuiltCohort
    mode: SourceProcessingMode


def import_prebuilt(source: RlDataSource[PrebuiltCohort], options: PipelineOptions) -> ArtifactStep[CampaignArtifact]:
    """Read the published cohort back and admit its tasks."""
    if EXCLUDED_TAG in source.info.tags:
        raise ValueError(f"Published cohort is excluded pending qualification: {source.name}")
    config = source.config
    assert config is not None
    if options.mode is SourceProcessingMode.QUICK:
        raise ValueError(f"A published cohort is imported for admission, not for a quick pass: {source.name}")

    def build_config(context) -> PrebuiltImport:
        return PrebuiltImport(context.output_path, config, options.mode)

    def run(invocation: PrebuiltImport) -> CampaignArtifact:
        local = Path(import_cohort(invocation.cohort))
        destination = StoragePath(invocation.output_path)
        destination.mkdirs(exist_ok=True)
        tasks_path = destination / TASKS_FILENAME
        tasks_path.upload_from(str(local))
        tasks = []
        for row in pq.read_table(local).to_pylist():
            task = LoweredSpec.from_document(json.loads(row[LOWERED_TASK_SPEC_COLUMN])).task
            tasks.append(normalization_columns(task.id, task.source, task, None, (), row["path"]))
        for view in ("normalize", "final"):
            folder = destination / view
            folder.mkdirs(exist_ok=True)
            with (folder / "part-00000.parquet").open("wb") as handle:
                pq.write_table(pa.Table.from_pylist(tasks, schema=NORMALIZED_SCHEMA), handle)
        manifest = {
            "mode": invocation.mode.value,
            "intended_use": "train",
            "input_rows": len(tasks),
            "converted_rows": len(tasks),
            "admitted_rows": len(tasks),
            "rejections": {},
            "status": "completed",
            "repo": config.repo,
            "revision": config.revision,
            "row_count": config.expected_rows,
            "tasks_sha256": file_sha256(tasks_path),
        }
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return CampaignArtifact(
            status="completed",
            result=PipelineResult(
                status=SourceStatus.COMPLETED,
                outputs={"tasks": str(destination / TASKS_FILENAME)},
                evidence={"manifest": str(destination / "manifest.json")},
                stages=("download", "validate"),
            ),
        )

    return ArtifactStep(
        name=config.name,
        version=config.version,
        artifact_type=CampaignArtifact,
        run=run,
        build_config=build_config,
        deps=(),
    )
