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

import hashlib
import json
import shutil
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from marin.execution.lazy import ArtifactStep
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.audit_schema import NORMALIZED_SCHEMA
from taskcompendium.pipeline.conversion import normalization_columns
from taskcompendium.pipeline.models import SourceStatus
from taskcompendium.pipeline.source_processing import SourceProcessingMode

from experiments.post_training.task_curation.campaign import CampaignArtifact, PipelineResult
from experiments.post_training.task_curation.config import PipelineOptions
from experiments.post_training.task_curation.environment_fidelity import missing_environment
from experiments.post_training.task_curation.source import RlDataSource, SourceReference

TASKS_FILENAME = "tasks.parquet"


@dataclass(frozen=True)
class PrebuiltCohort:
    """Where a published cohort lives and how much of it this source expects."""

    name: str
    version: str
    repo: str
    revision: str
    expected_rows: int
    repo_type: str = "dataset"

    @property
    def dataset(self) -> SourceReference:
        return SourceReference(
            self.repo, self.revision, f"https://huggingface.co/datasets/{self.repo}/tree/{self.revision}"
        )

    @property
    def files(self) -> tuple[str, ...]:
        return (TASKS_FILENAME,)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def import_cohort(config: PrebuiltCohort) -> str:
    """Download the pinned cohort, validate every row, and return the local path.

    A row without a usable environment, or a cohort whose size differs from the
    recorded count, fails here: admitting it would put tasks into training that the
    published cohort never qualified.
    """
    local = Path(
        hf_hub_download(
            repo_id=config.repo, repo_type=config.repo_type, filename=TASKS_FILENAME, revision=config.revision
        )
    )
    rows = pq.read_table(local).to_pylist()
    if len(rows) != config.expected_rows:
        raise ValueError(f"{config.repo}@{config.revision} has {len(rows)} rows, expected {config.expected_rows}")
    for row in rows:
        # A lowered row is the rollout envelope: the task spec plus its machine shapes.
        envelope = json.loads(row["lowered_task_spec"])
        task = envelope.get("task") or {}
        if not envelope.get("runtime"):
            raise ValueError(f"{row['path']}: envelope carries no runtime machines")
        spec = TaskSpec.model_validate(task)
        reason = missing_environment(spec.model_dump(mode="json"))
        if reason is not None:
            raise ValueError(f"{row['path']}: {reason}")
    return str(local)


@dataclass(frozen=True)
class PrebuiltImport:
    output_path: str


def import_prebuilt(source: RlDataSource[PrebuiltCohort], options: PipelineOptions) -> ArtifactStep[CampaignArtifact]:
    """Read the published cohort back and admit its tasks."""
    if "excluded" in source.info.tags:
        raise ValueError(f"Published cohort is excluded pending qualification: {source.name}")
    config = source.config
    assert config is not None
    if options.mode is SourceProcessingMode.QUICK:
        raise ValueError(f"A published cohort is imported for admission, not for a quick pass: {source.name}")

    def build_config(context) -> PrebuiltImport:
        return PrebuiltImport(context.output_path)

    def run(invocation: PrebuiltImport) -> CampaignArtifact:
        local = Path(import_cohort(config))
        destination = Path(invocation.output_path)
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local, destination / TASKS_FILENAME)
        tasks = []
        for row in pq.read_table(local).to_pylist():
            task = TaskSpec.model_validate(json.loads(row["lowered_task_spec"])["task"])
            tasks.append(normalization_columns(task.id, task.source, task, None, (), row["path"]))
        for view in ("normalize", "final"):
            folder = destination / view
            folder.mkdir()
            pq.write_table(pa.Table.from_pylist(tasks, schema=NORMALIZED_SCHEMA), folder / "part-00000.parquet")
        manifest = {
            "mode": options.mode.value,
            "intended_use": "train",
            "input_rows": len(tasks),
            "converted_rows": len(tasks),
            "admitted_rows": len(tasks),
            "status": "completed",
            "repo": config.repo,
            "revision": config.revision,
            "row_count": config.expected_rows,
            "tasks_sha256": _sha256(destination / TASKS_FILENAME),
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
        run=partial(run),
        build_config=build_config,
        deps=(),
    )
