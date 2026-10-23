# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select the pinned Snowball tasks and publish canonical TaskSpecs."""

import json
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.remote import remote
from marin.experiment.namespacing import user_owned_name
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.convert.structure_closure import structure_closure_task
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import SCHEMA_VERSION, ScriptGrader, Source, TaskSpec
from taskcompendium.pipeline.fingerprints import semantic_digest
from taskcompendium.pipeline.models import NormalizedTask, RawRow
from taskcompendium.pipeline.sampling import seeded_order, seeded_sample
from verifyit.json_objects import unique_object
from zephyr.writers import write_parquet_file

TRAIN_FILENAME = "train.parquet"
VALIDATION_FILENAME = "validation.parquet"
FULL_VALIDATION_FILENAME = "full_validation.parquet"
TASK_SPEC_COLUMN = "task_spec"
TASK_SCHEMA = pa.schema([pa.field(TASK_SPEC_COLUMN, pa.string(), nullable=False)])

# This URI identifies the binding in the immutable source rows. Its missing bytes
# are not used at training time; the replacement lock is retained on S3.
SOURCE_GRADER_LOCK = (
    "gs://marin-us-central2/tmp/ttl=7d/rl-data-curation-validation/rl-curation-full-20261011/"
    "images/env-857da253ce96109b/2026.10.08/requirements.lock"
)
DERIVED_GRADER_LOCK = (
    "s3://marin-us-east-02a/marin/rl-data/environments/snowball-inventory-grader/2026.10.20.1/requirements.lock"
)
SOURCE_SCHEMA_VERSION = "0.27"
FAMILIES = ("calendar", "competitive_coding", "reasoning_gym", "stem", "structural")
TASKS_PER_FAMILY = 20
SPLIT_SEED = 20261023
TRAINING_ROWS = 100_000
FULL_VALIDATION_ROWS = 910


@dataclass(frozen=True)
class IngestConfig:
    training: str
    training_rows: int
    validation: str
    validation_rows: int
    output_path: str


@dataclass(frozen=True)
class HeldoutSelection:
    full: tuple[TaskSpec, ...]
    sample: tuple[TaskSpec, ...]
    family_counts: dict[str, int]


def current_task(document: dict[str, Any]) -> TaskSpec:
    """Import a source task and bind the reconstructed script-grader lock."""
    if document["schema_version"] != SOURCE_SCHEMA_VERSION or SCHEMA_VERSION != "0.28":
        raise ValueError("The pinned inventory requires the reviewed TaskSpec 0.27 to 0.28 import")
    task = TaskSpec.model_validate({**document, "schema_version": SCHEMA_VERSION})
    grader = task.grader
    if not isinstance(grader, ScriptGrader) or grader.environment.packages_lock != SOURCE_GRADER_LOCK:
        return task
    return task.model_copy(
        update={
            "grader": grader.model_copy(
                update={"environment": grader.environment.model_copy(update={"packages_lock": DERIVED_GRADER_LOCK})}
            )
        }
    )


def native_task(raw: str, task_id: str) -> TaskSpec:
    document = json.loads(raw, object_pairs_hook=unique_object)
    if document["task"]["schema_version"] != SOURCE_SCHEMA_VERSION:
        raise ValueError("The pinned native source TaskSpec version changed")
    converted = structure_closure_task(
        RawRow(task_id, Source.model_validate(document["task"]["source"]), {"lowered_task_spec": raw})
    )
    if not isinstance(converted, NormalizedTask):
        raise ValueError(f"Native structure-closure import failed for {task_id}: {converted}")
    return converted.task


def training_task(row: dict[str, Any]) -> TaskSpec:
    if row["contract_kind"] == "task_spec":
        task = current_task(json.loads(row["task_json"], object_pairs_hook=unique_object))
    elif row["contract_kind"] == "native-structure_closure":
        task = native_task(row["lowered_task_spec"], row["task_id"])
    else:
        raise ValueError(f"Unsupported inventory contract: {row['contract_kind']}")
    if task.id != row["task_id"]:
        raise ValueError("Training import changed the selected task ID")
    return task


def validation_task(row: dict[str, Any]) -> TaskSpec:
    raw = row["lowered_task_spec"]
    document = json.loads(raw, object_pairs_hook=unique_object)
    if document["task"]["source"]["dataset"] == "structure-closure":
        return native_task(raw, row["task_id"])
    task = current_task(document["task"])
    if task.id != row["task_id"] or document["runtime"]["task_machine"] is not None:
        raise ValueError("Heldout source task ID or machine contract changed")
    if document["session"]["task_session"] != "shellbox" or document["session"]["max_turns"] != 1:
        raise ValueError("The heldout source must use the one-turn Shellbox contract")
    return task


def select_heldout(rows: list[dict[str, Any]]) -> HeldoutSelection:
    """Keep all heldout tasks and select 20 per family in source order."""
    full = tuple(validation_task(row) for row in rows)
    if len({task.id for task in full}) != len(full):
        raise ValueError("Heldout task IDs are not unique")
    if {row["evaluation_family"] for row in rows} != set(FAMILIES):
        raise ValueError("Heldout source families changed")
    selected: set[str] = set()
    for family in FAMILIES:
        count, sample = seeded_sample(
            (row for row in rows if row["evaluation_family"] == family),
            size=TASKS_PER_FAMILY,
            key=lambda row: seeded_order(row["task_id"], SPLIT_SEED),
        )
        if count < TASKS_PER_FAMILY:
            raise ValueError(f"Heldout source lacks enough {family} tasks")
        selected.update(row["task_id"] for row in sample)
    return HeldoutSelection(
        full=full,
        sample=tuple(task for task in full if task.id in selected),
        family_counts=dict(Counter(row["evaluation_family"] for row in rows if row["task_id"] in selected)),
    )


def task_records(tasks: Iterator[TaskSpec]) -> Iterator[dict[str, str]]:
    for task in tasks:
        yield {TASK_SPEC_COLUMN: task.model_dump_json()}


def write_training(config: IngestConfig, full: tuple[TaskSpec, ...], destination: Path) -> tuple[int, int]:
    """Write training tasks after auditing them against the full heldout set."""
    heldout_ids = {task.id for task in full}
    full_public = {semantic_digest(task, False) for task in full}
    full_contexts = {canonical_sha256(task.context.model_dump(mode="json")) for task in full}
    train_ids: set[str] = set()
    rebound_count = 0

    def training_tasks() -> Iterator[TaskSpec]:
        nonlocal rebound_count
        with StoragePath(config.training).open("rb") as source:
            parquet = pq.ParquetFile(source)
            if parquet.metadata.num_rows != config.training_rows:
                raise ValueError("Training source row count changed")
            for batch in parquet.iter_batches(batch_size=256):
                for row in batch.to_pylist():
                    task = training_task(row)
                    if (
                        task.id in heldout_ids
                        or semantic_digest(task, False) in full_public
                        or (canonical_sha256(task.context.model_dump(mode="json")) in full_contexts)
                    ):
                        raise ValueError("Training source overlaps the original heldout set")
                    if task.id in train_ids:
                        raise ValueError("Training source task IDs are not unique")
                    if (
                        row["contract_kind"] == "task_spec"
                        and isinstance(task.grader, ScriptGrader)
                        and task.grader.environment.packages_lock == DERIVED_GRADER_LOCK
                    ):
                        original = json.loads(row["task_json"], object_pairs_hook=unique_object)
                        if original["grader"]["environment"]["packages_lock"] == SOURCE_GRADER_LOCK:
                            rebound_count += 1
                    train_ids.add(task.id)
                    yield task

    write_parquet_file(task_records(training_tasks()), str(destination), schema=TASK_SCHEMA)
    if len(train_ids) != config.training_rows:
        raise ValueError("Training import did not preserve every source row")
    return len(train_ids), rebound_count


def prepare(config: IngestConfig) -> None:
    """Publish the complete training pool and a fixed, disjoint heldout sample."""
    with StoragePath(config.validation).open("rb") as source:
        heldout = pq.read_table(source).to_pylist()
    if len(heldout) != config.validation_rows:
        raise ValueError("Heldout source row count changed")
    selection = select_heldout(heldout)
    with TemporaryDirectory(prefix="snowball-inventory-") as temporary:
        directory = Path(temporary)
        training_rows, rebound_count = write_training(config, selection.full, directory / TRAIN_FILENAME)
        write_parquet_file(
            task_records(iter(selection.sample)), str(directory / VALIDATION_FILENAME), schema=TASK_SCHEMA
        )
        write_parquet_file(
            task_records(iter(selection.full)), str(directory / FULL_VALIDATION_FILENAME), schema=TASK_SCHEMA
        )
        summary = {
            "training_source": config.training,
            "validation_source": config.validation,
            "training_rows": training_rows,
            "validation_rows": len(selection.sample),
            "full_validation_rows": len(selection.full),
            "validation_family_counts": selection.family_counts,
            "validation_task_ids": [task.id for task in selection.sample],
            "split_seed": SPLIT_SEED,
            "environment_derivation": {
                "kind": "derived-deployment-environment",
                "source_packages_lock": SOURCE_GRADER_LOCK,
                "replacement_packages_lock": DERIVED_GRADER_LOCK,
                "affected_rows": rebound_count,
                "source_task_schema_version": SOURCE_SCHEMA_VERSION,
                "derived_task_schema_version": SCHEMA_VERSION,
            },
        }
        destination = StoragePath(config.output_path)
        destination.mkdirs()
        for name in (TRAIN_FILENAME, VALIDATION_FILENAME, FULL_VALIDATION_FILENAME):
            (destination / name).upload_from(str(directory / name))
        (destination / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def data_step(version: str, training_uri: str, validation_uri: str) -> ArtifactStep[Artifact]:
    """Bind the pinned 100k training source and original 910-row heldout source."""
    training = ArtifactStep.adopt("documents/snowball-inventory-training-source", "2026.10.19.2", training_uri)
    validation = ArtifactStep.adopt("documents/snowball-inventory-validation-source", "2026.10.20.2", validation_uri)

    def build_config(ctx: StepContext) -> IngestConfig:
        return IngestConfig(
            ctx.artifact_path(training),
            TRAINING_ROWS,
            ctx.artifact_path(validation),
            FULL_VALIDATION_ROWS,
            ctx.output_path,
        )

    return ArtifactStep(
        name=user_owned_name("documents/snowball-inventory"),
        version=version,
        artifact_type=Artifact,
        deps=(training, validation),
        run=remote(prepare, resources=ResourceConfig.with_cpu(cpu=4, ram="16g", disk="8g")),
        build_config=build_config,
    )
