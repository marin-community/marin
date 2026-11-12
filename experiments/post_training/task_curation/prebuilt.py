# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import an already-published TaskCompendium cohort as a runnable source.

A converted dataset is published once and then read back, never regenerated: the
published revision is the artifact a reader and a run both consume. This stage
downloads the pinned revision through the cached Hugging Face login, validates every
row as a task spec under the admission predicates, and records what it admitted.
Native Harbor cohorts use ``import_harbor_prebuilt`` and require exact-package
control receipts for admission.

It converts nothing, so it cannot improve on the published cohort's review; the
source's oversight level and evidence describe the qualification the publish recorded.
"""

import hashlib
import json
import shutil
import tomllib
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from functools import partial
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from marin.execution.lazy import ArtifactStep
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.grading_environment import (
    GradingEnvironmentStatus,
    assess_grading_environment,
    count_grading_environment,
    grading_environment_census,
)
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.audit_schema import NORMALIZED_SCHEMA
from taskcompendium.pipeline.conversion import normalization_columns
from taskcompendium.pipeline.models import SourceStatus
from taskcompendium.pipeline.oracle_filter import has_oracle_solution, load_oracle_filter, require_oracle_filter
from taskcompendium.pipeline.source_processing import SourceProcessingMode
from taskcompendium.runtime.resources import resource_bytes

from experiments.post_training.task_curation.campaign import CampaignArtifact, PipelineResult
from experiments.post_training.task_curation.config import PipelineOptions
from experiments.post_training.task_curation.environment_fidelity import missing_environment
from experiments.post_training.task_curation.source import RlDataSource, SourceReference, TaskFormat

TASKS_FILENAME = "tasks.parquet"
CONTROLS_FILENAME = "controls.jsonl"
ORACLE_FILTER_FILENAME = "oracle-filter.json"


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


def import_taskcompendium_cohort(config: PrebuiltCohort) -> str:
    """Validate canonical tasks without requiring a machine for conversation-only graders."""
    local = Path(
        hf_hub_download(
            repo_id=config.repo, repo_type=config.repo_type, filename=TASKS_FILENAME, revision=config.revision
        )
    )
    seen = set()
    for batch in pq.ParquetFile(local).iter_batches(batch_size=32):
        for row in batch.to_pylist():
            serialized = row["task_spec"]
            task = TaskSpec.model_validate_json(serialized)
            if hashlib.sha256(serialized.encode()).hexdigest() != row["task_sha256"]:
                raise ValueError(f"Published task differs from its recorded hash: {task.id}")
            if task.id in seen:
                raise ValueError(f"Duplicate published task: {task.id}")
            seen.add(task.id)
            reason = missing_environment(task.model_dump(mode="json"))
            if reason is not None:
                raise ValueError(f"{task.id}: {reason}")
    if len(seen) != config.expected_rows:
        raise ValueError(f"{config.repo}@{config.revision} has {len(seen)} rows, expected {config.expected_rows}")
    return str(local)


class PrebuiltEncoding(StrEnum):
    LOWERED = "lowered_task_spec"
    CANONICAL = "task_spec"


@dataclass(frozen=True)
class PrebuiltImport:
    output_path: str
    cohort: PrebuiltCohort
    mode: SourceProcessingMode
    task_format: TaskFormat
    encoding: PrebuiltEncoding


def import_prebuilt(source: RlDataSource[PrebuiltCohort], options: PipelineOptions) -> ArtifactStep[CampaignArtifact]:
    """Read the pinned lowered cohort into admitted corpus views."""
    return prebuilt_artifact(
        source, options, import_cohort, task_format=TaskFormat.TASKCOMPENDIUM, encoding=PrebuiltEncoding.LOWERED
    )


def import_harbor_prebuilt(
    source: RlDataSource[PrebuiltCohort], options: PipelineOptions
) -> ArtifactStep[CampaignArtifact]:
    """Admit native Harbor packages through exact bug/reference control receipts."""
    return prebuilt_artifact(
        source, options, import_harbor_cohort, task_format=TaskFormat.HARBOR, encoding=PrebuiltEncoding.CANONICAL
    )


def import_taskcompendium_prebuilt(
    source: RlDataSource[PrebuiltCohort], options: PipelineOptions
) -> ArtifactStep[CampaignArtifact]:
    """Admit a qualified canonical TaskCompendium release without lowering its tasks again."""
    return prebuilt_artifact(
        source,
        options,
        import_taskcompendium_cohort,
        task_format=TaskFormat.TASKCOMPENDIUM,
        encoding=PrebuiltEncoding.CANONICAL,
    )


QUICK_VIEW_VERSION = "2026.11.12"
"""Format version of the converted view a QUICK pass writes; the cohort revision is in the step identity."""
COHORT_INTENDED_USE = "train"
"""Published campaign cohorts are RL training populations."""


def cohort_task_json(tasks: Path) -> list[str]:
    """Each row's TaskSpec as canonical JSON, from a lowered envelope or a native Harbor package."""
    columns = pq.ParquetFile(tasks).schema_arrow.names
    rows = pq.read_table(tasks).to_pylist()
    if "lowered_task_spec" in columns:
        return [TaskSpec.model_validate(json.loads(row["lowered_task_spec"])["task"]).model_dump_json() for row in rows]
    return [TaskSpec.model_validate_json(row["task_spec"]).model_dump_json() for row in rows]


def _quick_view(config: PrebuiltCohort, tasks: Path, destination: str) -> CampaignArtifact:
    """Write the validated cohort as the converted view that QUICK consumers, such as embedding, read.

    QUICK admits nothing. Every published row is a converted task, so there are no rejections.
    """
    task_json = cohort_task_json(tasks)
    root = StoragePath(destination)
    normalize = root / "normalize"
    with (normalize / "part-00000.parquet").open("wb") as handle:
        pq.write_table(pa.table({"task_json": task_json}), handle)
    manifest = {
        "status": "completed",
        "mode": SourceProcessingMode.QUICK.value,
        "repo": config.repo,
        "revision": config.revision,
        "intended_use": COHORT_INTENDED_USE,
        "input_rows": len(task_json),
        "converted_rows": len(task_json),
        "rejections": {},
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return CampaignArtifact(
        status="completed",
        result=PipelineResult(
            status="completed",
            outputs={"normalize": str(normalize)},
            evidence={"manifest": str(root / "manifest.json")},
            stages=("download", "validate", "normalize"),
        ),
    )


def prebuilt_artifact(
    source: RlDataSource[PrebuiltCohort],
    options: PipelineOptions,
    load: Callable[[PrebuiltCohort], str],
    *,
    task_format: TaskFormat,
    encoding: PrebuiltEncoding,
) -> ArtifactStep[CampaignArtifact]:
    """Publish validated rows and corpus views while preserving their controlled package bytes."""
    if "excluded" in source.info.tags:
        raise ValueError(f"Published cohort is excluded pending qualification: {source.name}")
    if options.mode is not SourceProcessingMode.QUICK and source.admission_blockers:
        raise ValueError(f"Source {source.name} cannot admit tasks: {'; '.join(source.admission_blockers)}")
    config = source.config
    assert config is not None
    quick = options.mode is SourceProcessingMode.QUICK

    def build_config(context) -> PrebuiltImport:
        return PrebuiltImport(context.output_path, config, options.mode, task_format, encoding)

    def run(invocation: PrebuiltImport) -> CampaignArtifact:
        local = Path(load(invocation.cohort))
        if invocation.mode is SourceProcessingMode.QUICK:
            return _quick_view(invocation.cohort, local, invocation.output_path)
        destination = Path(invocation.output_path)

        destination.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local, destination / TASKS_FILENAME)
        tasks = []
        grading: Counter[str] = Counter()
        oracle_filter = None
        for row in pq.read_table(local).to_pylist():
            if invocation.encoding is PrebuiltEncoding.CANONICAL:
                serialized = row["task_spec"]
                task = TaskSpec.model_validate_json(serialized)
                original_path = task.source.row
            else:
                task = TaskSpec.model_validate(json.loads(row["lowered_task_spec"])["task"])
                serialized = task.model_dump_json()
                original_path = row["path"]
            assessment = assess_grading_environment(task)
            count_grading_environment(grading, assessment.model_dump(mode="json"))
            if assessment.status is GradingEnvironmentStatus.MISMATCH:
                raise ValueError(f"grading_environment_mismatch: {task.id}: {assessment.model_dump_json()}")
            if has_oracle_solution(task):
                if oracle_filter is None:
                    receipt_path = hf_hub_download(
                        repo_id=invocation.cohort.repo,
                        repo_type=invocation.cohort.repo_type,
                        revision=invocation.cohort.revision,
                        filename=ORACLE_FILTER_FILENAME,
                    )
                    oracle_filter = load_oracle_filter(receipt_path)
                    shutil.copyfile(receipt_path, destination / ORACLE_FILTER_FILENAME)
                if not require_oracle_filter(
                    task,
                    oracle_filter,
                    (
                        json.loads(row["lowered_task_spec"])["runtime"]
                        if invocation.encoding is PrebuiltEncoding.LOWERED
                        else None
                    ),
                ):
                    raise ValueError(f"Published cohort includes a failed oracle task: {task.id}")
            tasks.append(
                {
                    **normalization_columns(task.id, task.source, task, None, (), original_path),
                    "task_json": serialized,
                }
            )
        for view in ("normalize", "final"):
            folder = destination / view
            folder.mkdir()
            pq.write_table(pa.Table.from_pylist(tasks, schema=NORMALIZED_SCHEMA), folder / "part-00000.parquet")
        manifest = {
            "mode": invocation.mode.value,
            "task_format": invocation.task_format.value,
            "intended_use": "train",
            "input_rows": len(tasks),
            "converted_rows": len(tasks),
            "admitted_rows": len(tasks),
            "rejections": {},
            "grading_environment": grading_environment_census(grading),
            "status": "completed",
            "repo": config.repo,
            "revision": config.revision,
            "row_count": config.expected_rows,
            "oracle_filter": oracle_filter.model_dump(mode="json") if oracle_filter else None,
            "tasks_sha256": file_sha256(StoragePath(str(destination / TASKS_FILENAME))),
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
        version=QUICK_VIEW_VERSION if quick else config.version,
        artifact_type=CampaignArtifact,
        run=partial(run),
        build_config=build_config,
        deps=(),
    )
