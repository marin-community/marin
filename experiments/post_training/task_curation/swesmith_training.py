# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Derive an executable selection from the released SWESmith training wrappers."""

import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.parquet as pq
from finestore.rl.strata import SelectionDefinition, SelectionParent, SelectionTaskRow, read_selection_manifest
from marin.execution.lazy import OUT, ArtifactStep, apply
from rigging.filesystem.storage_path import StoragePath
from rigging.filesystem.transfer import copy
from rolloutengine.models import MachineRuntimeSpec
from shellbox.machine import Backend, NetworkPolicy
from taskcompendium.identity import canonical_sha256
from taskcompendium.importers.harbor.package import JobSettings, PackageTests, lower_package
from taskcompendium.models import AgentMachine, FileReward, NoGrader, ScriptGrader, TaskSpec
from taskcompendium.runtime.resources import resource_bytes

from experiments.evaluation.benchmarks.harbor import image_lock
from experiments.post_training.task_curation import CANONICAL_ARTIFACT_ROOT
from experiments.post_training.task_curation.selection import (
    SUMMARY_FILENAME,
    TASK_MANIFEST_FILENAME,
    TASK_MANIFEST_SCHEMA,
    TRAIN_FILENAME,
    GraderKind,
)
from experiments.post_training.task_curation.task_store import PARQUET_BATCH_ROWS, TASK_SPEC_COLUMN

SWESMITH_DATASET = "long-loop-smartie-swesmith"
SWESMITH_PI_VERSION = "0.87.0"
SWESMITH_MACHINE = MachineRuntimeSpec(
    backend=Backend.DAYTONA,
    network=NetworkPolicy.ALLOW,
    cpus=None,
    memory_mb=None,
    storage_mb=None,
    gpus=0,
    user="longloop-agent",
    startup_timeout=1800,
    cleanup_timeout=120,
)
VERIFIER_TIMEOUT = 900
_PI_LOCK = "environment/long-loop-pi/package-lock.json"
_PI_LOCK_PACKAGE = "node_modules/@earendil-works/pi-coding-agent"


def lower_swesmith_task(task: TaskSpec, machine: MachineRuntimeSpec, image_digests: Mapping[str, str]) -> TaskSpec:
    """Lower one sealed wrapper using its own bootstrap, image, and verifier."""
    if task.source.dataset != SWESMITH_DATASET or not isinstance(task.grader, NoGrader) or "harbor" not in task.tags:
        raise ValueError(f"Expected a released SWESmith task wrapper: {task.id}")
    with TemporaryDirectory(prefix="swesmith-package-") as directory:
        package = Path(directory) / "task"
        for resource in task.resources.verifier:
            path = package / resource.path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(resource_bytes(resource))
            if resource.mode is not None:
                path.chmod(int(resource.mode, 8))
        lock = json.loads((package / _PI_LOCK).read_text())
        if lock.get("packages", {}).get(_PI_LOCK_PACKAGE, {}).get("version") != SWESMITH_PI_VERSION:
            raise ValueError(f"SWESmith package does not lock Pi {SWESMITH_PI_VERSION}: {task.id}")
        lowered = lower_package(
            package,
            task_id=task.id,
            source=task.source,
            rule=PackageTests(),
            image_digests=image_digests,
            job=JobSettings(
                verifier_timeout=VERIFIER_TIMEOUT, agent_user=machine.user, setup_timeout=machine.startup_timeout
            ),
        )
    native, episode = lowered.task, lowered.episode
    if native.environment_requirements.setup_commands[1:] != task.environment_requirements.setup_commands:
        raise ValueError(f"Archived setup does not match the admitted SWESmith setup: {task.id}")
    if (episode.network, episode.cpus, episode.memory_mb, episode.storage_mb) != (
        machine.network,
        machine.cpus,
        machine.memory_mb,
        machine.storage_mb,
    ):
        raise ValueError(f"SWESmith package resources differ from the dataset deployment: {task.id}")
    if not isinstance(native.grader, ScriptGrader) or not isinstance(native.grader.placement, AgentMachine):
        raise ValueError(f"SWESmith requires a shared package verifier: {task.id}")
    reward = native.grader.reward
    assert isinstance(reward, FileReward)
    grader = native.grader.model_copy(update={"reward": reward.model_copy(update={"pass_above": 0.0})})
    return native.model_copy(update={"tags": task.tags, "labels": task.labels, "grader": grader})


def _convert_training(
    source: StoragePath,
    destination: StoragePath,
    manifest: dict[str, SelectionTaskRow],
    machine: MachineRuntimeSpec,
    image_digests: Mapping[str, str],
) -> int:
    converted = 0
    with (source / TRAIN_FILENAME).open("rb") as stream, (destination / TRAIN_FILENAME).open("wb") as output:
        training = pq.ParquetFile(stream)
        schema = training.schema_arrow
        index = schema.get_field_index(TASK_SPEC_COLUMN)
        with pq.ParquetWriter(output, schema) as writer:
            for batch in training.iter_batches(batch_size=PARQUET_BATCH_ROWS):
                table = pa.Table.from_batches([batch])
                columns = table[TASK_SPEC_COLUMN].to_pylist()
                for row_index, dataset in enumerate(table["source"].to_pylist()):
                    if dataset != SWESMITH_DATASET:
                        continue
                    serialized = columns[row_index]
                    task = TaskSpec.model_validate_json(serialized)
                    row = manifest[task.id]
                    if row.references or row.task_digest != canonical_sha256(json.loads(serialized)):
                        raise ValueError(f"SWESmith task has reference scores or a stale selection digest: {task.id}")
                    native = lower_swesmith_task(task, machine, image_digests)
                    columns[row_index] = native.model_dump_json()
                    manifest[task.id] = row.model_copy(
                        update={
                            "task_digest": canonical_sha256(json.loads(columns[row_index])),
                            "grader_kind": GraderKind.IMAGE,
                        }
                    )
                    converted += 1
                writer.write_table(
                    table.set_column(index, schema.field(index), pa.array(columns, type=schema.field(index).type))
                )
    return converted


def _derive_selection(
    selection: str,
    output: str,
    selection_name: str,
    selection_version: str,
    machine: MachineRuntimeSpec,
    image_digests: Mapping[str, str],
) -> None:
    source, destination = StoragePath(selection), StoragePath(output)
    summary = json.loads((source / SUMMARY_FILENAME).read_text())
    if summary.get("harness_graded_datasets") != [SWESMITH_DATASET]:
        raise ValueError(f"Selection must contain only the SWESmith harness-graded dataset: {selection}")
    counts = summary["grader_kind_counts"]["train"]
    definition = SelectionDefinition.model_validate(summary["selection_strata"])
    with (source / TASK_MANIFEST_FILENAME).open("rb") as stream:
        rows = [SelectionTaskRow.model_validate(row) for row in pq.read_table(stream).to_pylist()]
    manifest = {row.task_id: row for row in rows}
    if len(manifest) != len(rows):
        raise ValueError(f"Selection contains duplicate manifest IDs: {selection}")
    destination.mkdirs()
    converted = _convert_training(source, destination, manifest, machine, image_digests)
    if converted == 0 or converted != counts[GraderKind.HARNESS]:
        raise ValueError(f"Expected {counts[GraderKind.HARNESS]} SWESmith wrappers, converted {converted}: {selection}")
    for filename in set((*definition.task_files, *definition.active_files)) - {TRAIN_FILENAME}:
        copy(str(source / filename), str(destination / filename))
    with (destination / TASK_MANIFEST_FILENAME).open("wb") as stream:
        pq.write_table(
            pa.Table.from_pylist([row.model_dump(mode="json") for row in manifest.values()], TASK_MANIFEST_SCHEMA),
            stream,
        )
    counts[GraderKind.HARNESS] = 0
    counts[GraderKind.IMAGE] += converted
    del summary["harness_graded_datasets"]
    summary["selection_strata"] = definition.model_copy(
        update={
            "name": selection_name,
            "version": selection_version,
            "parent": SelectionParent(root=selection, name=definition.name, version=definition.version),
        }
    ).model_dump(mode="json")
    (destination / SUMMARY_FILENAME).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    read_selection_manifest(output)


def native_swesmith_selection(selection: ArtifactStep) -> ArtifactStep:
    """Build a native Pi selection with the parent selection's exact membership and task IDs."""
    name = f"{selection.name}-pi"
    step = apply(
        name,
        _derive_selection,
        version=selection.version,
        selection=selection,
        output=OUT,
        selection_name=name.rsplit("/", 1)[-1],
        selection_version=selection.version,
        machine=SWESMITH_MACHINE,
        image_digests=image_lock(SWESMITH_DATASET),
    )
    return replace(step, output_path_prefix=CANONICAL_ARTIFACT_ROOT)
