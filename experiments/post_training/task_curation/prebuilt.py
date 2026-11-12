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
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from marin.execution.lazy import ArtifactStep
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.source_processing import SourceProcessingMode
from taskcompendium.runtime.resources import resource_bytes

from experiments.post_training.task_curation.campaign import CampaignArtifact, PipelineResult
from experiments.post_training.task_curation.config import PipelineOptions
from experiments.post_training.task_curation.environment_fidelity import missing_environment
from experiments.post_training.task_curation.source import RlDataSource, SourceReference

TASKS_FILENAME = "tasks.parquet"
CONTROLS_FILENAME = "controls.jsonl"


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


def import_harbor_cohort(config: PrebuiltCohort) -> str:
    """Import native Harbor packages only when exact package controls admit every row."""
    local = Path(
        hf_hub_download(
            repo_id=config.repo, repo_type=config.repo_type, filename=TASKS_FILENAME, revision=config.revision
        )
    )
    controls_path = Path(
        hf_hub_download(
            repo_id=config.repo, repo_type=config.repo_type, filename=CONTROLS_FILENAME, revision=config.revision
        )
    )
    controls = {}
    for line in controls_path.read_text().splitlines():
        receipt = json.loads(line)
        task_id = receipt["task_id"]
        if task_id in controls:
            raise ValueError(f"Duplicate Harbor control: {task_id}")
        controls[task_id] = receipt
    seen = set()
    for batch in pq.ParquetFile(local).iter_batches(batch_size=32):
        for row in batch.to_pylist():
            serialized = row["task_spec"]
            task = TaskSpec.model_validate_json(serialized)
            if task.id in seen:
                raise ValueError(f"Duplicate Harbor task: {task.id}")
            seen.add(task.id)
            receipt = controls[task.id]
            if hashlib.sha256(serialized.encode()).hexdigest() != receipt["task_sha256"]:
                raise ValueError(f"Harbor package differs from its control: {task.id}")
            outcomes = receipt["controls"]
            if not (
                receipt["status"] == "admitted"
                and outcomes["bug"]["status"] == outcomes["reference"]["status"] == "scored"
                and outcomes["bug"]["reward"] == 0
                and outcomes["reference"]["reward"] == 1
            ):
                raise ValueError(f"Harbor controls do not admit task: {task.id}")
            files = {resource.path: resource for resource in task.resources.verifier}
            required = {"task.toml", "instruction.md", "tests/test.sh"}
            if "harbor" not in task.tags or not required <= files.keys():
                raise ValueError(f"Incomplete native Harbor package: {task.id}")
            environment = tomllib.loads(resource_bytes(files["task.toml"]).decode())["environment"]
            if not environment.get("docker_image") and "environment/Dockerfile" not in files:
                raise ValueError(f"Harbor package declares no environment: {task.id}")
    if seen != controls.keys() or len(seen) != config.expected_rows:
        raise ValueError(f"Harbor population differs from its controls or expected count: {config.repo}")
    return str(local)


def import_prebuilt(source: RlDataSource[PrebuiltCohort], options: PipelineOptions) -> ArtifactStep[CampaignArtifact]:
    """Read the published TaskCompendium cohort back and admit its tasks."""
    return prebuilt_artifact(source, options, import_cohort)


def import_harbor_prebuilt(
    source: RlDataSource[PrebuiltCohort], options: PipelineOptions
) -> ArtifactStep[CampaignArtifact]:
    """Retain native Harbor packages and their exact bug/reference admission contracts."""
    return prebuilt_artifact(source, options, import_harbor_cohort)


def prebuilt_artifact(
    source: RlDataSource[PrebuiltCohort], options: PipelineOptions, load: Callable[[PrebuiltCohort], str]
) -> ArtifactStep[CampaignArtifact]:
    """Publish a validated cohort without rewriting its task bytes."""
    config = source.config
    assert config is not None
    if options.mode is SourceProcessingMode.QUICK:
        raise ValueError(f"A published cohort is imported for admission, not for a quick pass: {source.name}")

    def build_config(context) -> str:
        return context.output_path

    def run(output_path: str) -> CampaignArtifact:
        local = Path(load(config))
        destination = Path(output_path)
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local, destination / TASKS_FILENAME)
        manifest = {
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
                status="completed",
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
