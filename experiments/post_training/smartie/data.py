# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert admitted curation rows without changing their task or grader contracts."""

import hashlib
import json
import logging
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import click
import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath
from rolloutengine.lowering import lower_task
from rolloutengine.spec import LoweredTaskSpec, MachineRuntimeSpec, TaskRuntimeSpec, TaskSessionSpec
from shellbox.backends.daytona.machine import DaytonaMachineFactory
from shellbox.machine import MachineFactory
from taskcompendium.models import EnvironmentRequirements, ScriptGrader, TaskSpec, VerifyitGrader
from taskcompendium.pipeline.sampling import seeded_order

from experiments.post_training.smartie.deployment import ImageDeployment, LockImage


@dataclass(frozen=True)
class ConversionFailure:
    task_id: str
    reason: str


@dataclass(frozen=True)
class ConvertedSource:
    tasks: tuple[LoweredTaskSpec, ...]
    failures: tuple[ConversionFailure, ...]


@dataclass(frozen=True)
class AdmittedInput:
    root: str
    expected_rows: int
    validation_rows: int


def convert_admitted_source(
    manifest: Mapping[str, Any],
    rows: Iterable[Mapping[str, Any]],
    *,
    machine: MachineRuntimeSpec,
    session: TaskSessionSpec,
    factories: Mapping[str, MachineFactory],
    deployment: ImageDeployment | None = None,
) -> ConvertedSource:
    """Check a FULL source and collect all row failures before publication.

    Callers must read rows from the manifest's final view, whose producer keeps
    only admitted rows. A nonempty failure list forbids publication as training data.
    """
    if manifest["mode"] != "full" or manifest["status"] != "completed":
        raise ValueError("Training requires completed FULL curation")
    if manifest["intended_use"] != "train" or manifest["unprocessed_rows"] != 0:
        raise ValueError("The source must be fully processed and training-approved")
    tasks = []
    failures = []
    seen = set()
    count = 0
    for row in rows:
        count += 1
        task_id = str(row["task_id"])
        try:
            task = TaskSpec.model_validate_json(row["task_json"])
            if task.id != task_id or task_id in seen:
                raise ValueError("Task identity mismatch or duplicate task ID")
            seen.add(task_id)
            if task.source.dataset != manifest["source_dataset"] or task.source.revision != manifest["source_revision"]:
                raise ValueError("Task source does not match its admission manifest")
            if deployment is not None:
                task = deployment.task(task)
            grader = task.grader
            verifier_environment = grader.environment if isinstance(grader, VerifyitGrader | ScriptGrader) else None
            runtime = TaskRuntimeSpec(
                task_machine=None if task.environment_requirements == EnvironmentRequirements() else machine,
                verifier_machine=None if verifier_environment is None else machine,
            )
            tasks.append(lower_task(task, runtime, session, factories=factories, sessions={}, providers={}))
        except (ValueError, NotImplementedError, KeyError, TypeError, OSError) as error:
            # Preserve every conversion failure; never publish a partial passing subset.
            failures.append(ConversionFailure(task_id, str(error)))
    expected = manifest["admission_counts"]["admitted"]
    if count != expected:
        failures.append(ConversionFailure("<source>", f"Expected {expected} admitted rows, read {count}"))
    return ConvertedSource(tuple(tasks), tuple(failures))


def split_real_tasks(
    tasks: Iterable[LoweredTaskSpec], *, validation_count: int, seed: int
) -> tuple[tuple[LoweredTaskSpec, ...], tuple[LoweredTaskSpec, ...]]:
    """Split unique task identities reproducibly, independently of input shard order."""
    ordered = sorted(tasks, key=lambda task: seeded_order(task.task.id, seed))
    identities = [task.task.id for task in ordered]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate task identities across the source mixture")
    if not 0 < validation_count < len(ordered):
        raise ValueError("Training and validation must both contain real tasks")
    return tuple(ordered[validation_count:]), tuple(ordered[:validation_count])


def build_real_dataset(
    sources: tuple[AdmittedInput, ...],
    output: StoragePath,
    *,
    seed: int,
    machine: MachineRuntimeSpec,
    session: TaskSessionSpec,
    factories: Mapping[str, MachineFactory],
    images: tuple[LockImage, ...] = (),
) -> dict[str, Any]:
    """Audit every source, then publish native tasks only if the entire batch passes."""
    if output.exists():
        raise FileExistsError(f"Dataset output must be new: {output}")
    output.mkdirs(exist_ok=False)
    training = []
    validation = []
    evidence = []
    failures = []
    identities = set()
    deployment = ImageDeployment(images)
    for source in sources:
        root = StoragePath(source.root)
        try:
            manifest_bytes = (root / "manifest.json").read_bytes()
            manifest = json.loads(manifest_bytes)
            if manifest["datasets"]["final"].rstrip("/") != str(root / "final").rstrip("/"):
                raise ValueError("Admission manifest points to a different final view")
            if manifest["admission_counts"]["admitted"] != source.expected_rows:
                raise ValueError("Admission count differs from the selected source contract")
            paths = sorted((root / "final/*.parquet").glob(), key=str)
            rows = []
            hashes = {}
            for path in paths:
                with path.open("rb") as stream:
                    hashes[str(path)] = hashlib.file_digest(stream, "sha256").hexdigest()
                    stream.seek(0)
                    rows.extend(pq.read_table(stream).to_pylist())
            converted = convert_admitted_source(
                manifest, rows, machine=machine, session=session, factories=factories, deployment=deployment
            )
            failures.extend({"source": source.root, **asdict(failure)} for failure in converted.failures)
            if converted.failures:
                continue
            train, val = split_real_tasks(converted.tasks, validation_count=source.validation_rows, seed=seed)
            source_ids = {task.task.id for task in converted.tasks}
            if identities.intersection(source_ids):
                raise ValueError("Task identities overlap across admitted sources")
            identities.update(source_ids)
            training.extend(train)
            validation.extend(val)
            evidence.append(
                {
                    **asdict(source),
                    "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                    "shard_sha256": hashes,
                    "train_ids": [task.task.id for task in train],
                    "validation_ids": [task.task.id for task in val],
                }
            )
        except (ValueError, KeyError, TypeError, OSError, NotImplementedError) as error:
            # Source failures are reported together; successful sources are not published alone.
            failures.append({"source": source.root, "task_id": "<source>", "reason": str(error)})
    report = {
        "sources": evidence,
        "failures": failures,
        "seed": seed,
        "image_deployments": {lock: asdict(image) for lock, image in deployment.evidence.items()},
    }
    (output / "conversion-report.json").write_text(json.dumps(report, indent=2))
    if failures:
        summary = Counter(failure["reason"].splitlines()[0] for failure in failures)
        for reason, count in summary.most_common(20):
            logging.warning("Conversion failure: count=%d reason=%s", count, reason[:500])
        raise ValueError(f"Dataset publication rejected: {len(failures)} failures; see {output}/conversion-report.json")
    if not training or not validation:
        raise ValueError("Cannot publish empty real-task splits")
    for name, tasks in (("train", training), ("validation", validation)):
        table = pa.table({"lowered_task_spec": [task.model_dump_json() for task in tasks]})
        with (output / f"{name}.parquet").open("wb") as stream:
            pq.write_table(table, stream)
    report.update(train_ids=[task.task.id for task in training], validation_ids=[task.task.id for task in validation])
    (output / "manifest.json").write_text(json.dumps(report, indent=2))
    return report


@click.command()
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--output", required=True, help="New immutable output directory or object-store prefix.")
def main(config_path: Path, output: str) -> None:
    config = json.loads(config_path.read_text())
    report = build_real_dataset(
        tuple(AdmittedInput(**source) for source in config["sources"]),
        StoragePath(output),
        seed=config["seed"],
        machine=MachineRuntimeSpec.model_validate(config["machine"]),
        session=TaskSessionSpec.model_validate(config["session"]),
        factories={"daytona": DaytonaMachineFactory()},
        images=tuple(LockImage(**image) for image in config.get("images", [])),
    )
    click.echo(json.dumps({"train": len(report["train_ids"]), "validation": len(report["validation_ids"])}))


if __name__ == "__main__":
    main()
