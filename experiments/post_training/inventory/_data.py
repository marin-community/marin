# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read the pinned source release in the frozen SkyRL CPU environment."""

import argparse
import hashlib
import json
import shutil
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pyarrow.parquet as pq
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.provenance import launch_provenance
from rolloutengine.harness.assistant_prefix import ASSISTANT_PREFIX_HARNESS
from rolloutengine.models import MachineRuntimeSpec, RolloutLimits, TaskRolloutSpec
from skyrl_gym.rollout_specs import SHELLBOX_HARNESS, rollout_spec
from skyrl_train.dataset.tasks import write_tasks
from taskcompendium.convert.structure_closure import structure_closure_task
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import SCHEMA_VERSION, ScriptGrader, Source, TaskSpec, VerifyitGrader, grades_with_judge
from taskcompendium.pipeline.fingerprints import semantic_digest
from taskcompendium.pipeline.models import NormalizedTask, RawRow
from taskcompendium.pipeline.sampling import seeded_order
from verifyit.json_objects import unique_object

from experiments.post_training.inventory import FULL_VALIDATION_FILENAME, TRAIN_FILENAME, VALIDATION_FILENAME

SOURCE_GRADER_LOCK = (
    "gs://marin-us-central2/tmp/ttl=7d/rl-data-curation-validation/rl-curation-full-20261011/"
    "images/env-857da253ce96109b/2026.10.08/requirements.lock"
)
DERIVED_GRADER_LOCK = (
    "s3://marin-us-east-02a/marin/rl-data/environments/snowball-inventory-grader/2026.10.20.1/requirements.lock"
)
DERIVED_GRADER_FILES_SHA256 = {
    "environment.json": "f6f0eab7f90c26885055fda7c2299589b2ea49fb61d3c417b6c996773b3fb704",
    "files.tar": "1be88375408d32ef739f183a8db63ea50c4dec16413467a612e76e96986c78e9",
    "provenance.json": "27d51b9c3482bee6e5278bdbb793f195e4ccf7b802b6a25dc90cbd9294041751",
    "requirements.lock": "10d9d138afef895dd28cae2885fab99651c7e962b09ce9e961de9a1d0b2e84a4",
}

FAMILIES = ("calendar", "competitive_coding", "reasoning_gym", "stem", "structural")
TASKS_PER_FAMILY = 20
SPLIT_SEED = 20261023
LIMITS = RolloutLimits(
    max_turns=1,
    max_total_tokens=None,
    model_turn_timeout=1800,
    tool_turn_timeout=90,
    total_turn_timeout=2400,
    attempt_timeout=3600,
    verifier_timeout=300,
    cleanup_timeout=60,
)
IMAGE_VERIFIER = MachineRuntimeSpec(
    backend="daytona",
    cpus=2,
    memory_mb=4096,
    storage_mb=10240,
    gpus=0,
    network="allow",
    user=None,
    startup_timeout=600,
    cleanup_timeout=60,
)

LOCAL_VERIFIER = IMAGE_VERIFIER.model_copy(
    update={"backend": "local", "cpus": None, "memory_mb": None, "storage_mb": None}
)


def current_task(document: dict[str, Any]) -> TaskSpec:
    """Import the source and bind the approved reconstructed script-grader environment."""
    if document["schema_version"] != "0.27" or SCHEMA_VERSION != "0.28":
        raise ValueError("The pinned inventory requires the reviewed TaskSpec 0.27 to 0.28 import")
    task = TaskSpec.model_validate({**document, "schema_version": SCHEMA_VERSION})
    grader = task.grader
    if not isinstance(grader, ScriptGrader) or grader.environment.packages_lock != SOURCE_GRADER_LOCK:
        return task
    derived = task.model_copy(
        update={
            "grader": grader.model_copy(
                update={"environment": grader.environment.model_copy(update={"packages_lock": DERIVED_GRADER_LOCK})}
            )
        }
    )
    restored = derived.model_dump(mode="json")
    restored["grader"]["environment"]["packages_lock"] = SOURCE_GRADER_LOCK
    if restored != task.model_dump(mode="json"):
        raise ValueError("Reconstructed grader environment changed fields other than its lock binding")
    return derived


def native_task(raw: str, task_id: str) -> TaskSpec:
    document = json.loads(raw, object_pairs_hook=unique_object)
    if document["task"]["schema_version"] != "0.27":
        raise ValueError("The pinned native source TaskSpec version changed")
    converted = structure_closure_task(
        RawRow(task_id, Source.model_validate(document["task"]["source"]), {"lowered_task_spec": raw})
    )
    if not isinstance(converted, NormalizedTask):
        raise ValueError(f"Native structure-closure import failed for {task_id}: {converted}")
    return converted.task


def lower(task: TaskSpec, harness: str) -> TaskRolloutSpec:
    grader = task.grader
    verifier = None
    if (
        isinstance(grader, ScriptGrader | VerifyitGrader)
        and grader.environment is not None
        and not grades_with_judge(grader)
    ):
        verifier = IMAGE_VERIFIER if grader.environment.docker_image is not None else LOCAL_VERIFIER
    return rollout_spec(
        task,
        {**LIMITS.model_dump(), "task_session": harness, "command_timeout": 30, "shell_tool": {}},
        verifier_machine=verifier,
    )


def training_task(row: dict[str, Any]) -> TaskRolloutSpec:
    if row["contract_kind"] == "task_spec":
        task = current_task(json.loads(row["task_json"], object_pairs_hook=unique_object))
        harness = SHELLBOX_HARNESS
    elif row["contract_kind"] == "native-structure_closure":
        task = native_task(row["lowered_task_spec"], row["task_id"])
        harness = ASSISTANT_PREFIX_HARNESS
    else:
        raise ValueError(f"Unsupported inventory contract: {row['contract_kind']}")
    if task.id != row["task_id"]:
        raise ValueError("Training import changed the selected task ID")
    return lower(task, harness)


def validation_task(row: dict[str, Any]) -> TaskRolloutSpec:
    raw = row["lowered_task_spec"]
    document = json.loads(raw, object_pairs_hook=unique_object)
    if document["task"]["source"]["dataset"] == "structure-closure":
        return lower(native_task(raw, row["task_id"]), ASSISTANT_PREFIX_HARNESS)
    task = current_task(document["task"])
    if task.id != row["task_id"] or document["runtime"]["task_machine"] is not None:
        raise ValueError("Heldout source task ID or machine contract changed")
    if document["session"]["task_session"] != SHELLBOX_HARNESS or document["session"]["max_turns"] != 1:
        raise ValueError("The heldout source must use the one-turn Shellbox contract")
    return lower(task, SHELLBOX_HARNESS)


def verified_source(uri: str, sha256: str, destination: Path) -> None:
    digest = hashlib.sha256()
    with StoragePath(uri).open("rb") as source, destination.open("wb") as target:
        while chunk := source.read(1024 * 1024):
            target.write(chunk)
            digest.update(chunk)
    if digest.hexdigest() != sha256:
        raise ValueError("Inventory source bytes differ from the pinned release")


def prepare(
    training: str,
    training_sha256: str,
    training_rows: int,
    validation: str,
    validation_sha256: str,
    validation_rows: int,
    output: str,
) -> None:
    """Write every training task and a deterministic disjoint five-family heldout set."""
    with TemporaryDirectory(prefix="snowball-inventory-") as temporary:
        directory = Path(temporary)
        train_source, validation_source = directory / "source-train.parquet", directory / "source-validation.parquet"
        verified_source(training, training_sha256, train_source)
        verified_source(validation, validation_sha256, validation_source)
        if pq.ParquetFile(train_source).metadata.num_rows != training_rows:
            raise ValueError("Training source row count changed")
        heldout = pq.read_table(validation_source).to_pylist()
        if len(heldout) != validation_rows:
            raise ValueError("Heldout source row count changed")
        full = [validation_task(row) for row in heldout]
        by_id = {spec.task.id: spec for spec in full}
        if len(by_id) != len(full):
            raise ValueError("Heldout task IDs are not unique")
        if {row["evaluation_family"] for row in heldout} != set(FAMILIES):
            raise ValueError("Heldout source families changed")
        selected: set[str] = set()
        for family in FAMILIES:
            candidates = [row for row in heldout if row["evaluation_family"] == family]
            if len(candidates) < TASKS_PER_FAMILY:
                raise ValueError(f"Heldout source lacks enough {family} tasks")
            selected.update(
                row["task_id"]
                for row in sorted(candidates, key=lambda row: seeded_order(row["task_id"], SPLIT_SEED))[
                    :TASKS_PER_FAMILY
                ]
            )
        fixed = [spec for spec in full if spec.task.id in selected]
        full_public = {semantic_digest(spec.task, False) for spec in full}
        full_contexts = {canonical_sha256(spec.task.context.model_dump(mode="json")) for spec in full}
        train_ids: set[str] = set()
        environment_digests: dict[str, dict[str, str]] = {}

        def training_specs() -> Iterator[TaskRolloutSpec]:
            for batch in pq.ParquetFile(train_source).iter_batches(batch_size=256):
                for row in batch.to_pylist():
                    spec = training_task(row)
                    grader = spec.task.grader
                    if isinstance(grader, ScriptGrader) and grader.environment.packages_lock == DERIVED_GRADER_LOCK:
                        original = json.loads(row["task_json"], object_pairs_hook=unique_object)
                        if original["grader"]["environment"]["packages_lock"] == SOURCE_GRADER_LOCK:
                            environment_digests[spec.task.id] = {
                                "source_task_sha256": canonical_sha256(original),
                                "derived_task_sha256": canonical_sha256(spec.task.model_dump(mode="json")),
                            }
                    if (
                        spec.task.id in by_id
                        or semantic_digest(spec.task, False) in full_public
                        or canonical_sha256(spec.task.context.model_dump(mode="json")) in full_contexts
                    ):
                        raise ValueError("Training source overlaps the original heldout set")
                    if spec.task.id in train_ids:
                        raise ValueError("Training source task IDs are not unique")
                    train_ids.add(spec.task.id)
                    yield spec

        outputs = {TRAIN_FILENAME: training_specs(), VALIDATION_FILENAME: fixed, FULL_VALIDATION_FILENAME: full}
        for name, specs in outputs.items():
            write_tasks(directory / name, specs)
        if len(train_ids) != training_rows:
            raise ValueError("Training import did not preserve every source row")
        output_hashes = {}
        for name in outputs:
            with (directory / name).open("rb") as contents:
                output_hashes[name] = hashlib.file_digest(contents, "sha256").hexdigest()
        summary = {
            "training_source_sha256": training_sha256,
            "validation_source_sha256": validation_sha256,
            "training_rows": len(train_ids),
            "validation_rows": len(fixed),
            "full_validation_rows": len(full),
            "validation_family_counts": dict(
                Counter(row["evaluation_family"] for row in heldout if row["task_id"] in selected)
            ),
            "validation_task_ids": [spec.task.id for spec in fixed],
            "split_seed": SPLIT_SEED,
            "import_changes": [
                "TaskSpec.schema_version:0.27->0.28",
                "exact missing ScriptGrader lock replaced by the approved derived environment",
                "native structure_closure->assistant_prefix using shared importer",
                "registry-image graders use Daytona; package-based graders use the local backend",
                "local grader machines request no per-machine CPU, memory, or storage allocations",
            ],
            "environment_derivation": {
                "kind": "derived-deployment-environment",
                "source_packages_lock": SOURCE_GRADER_LOCK,
                "replacement_packages_lock": DERIVED_GRADER_LOCK,
                "expected_replacement_files_sha256": DERIVED_GRADER_FILES_SHA256,
                "replacement_source_commit": "af479a4deec470dda976b6f91e6e4369981619be",
                "ingestion_source_commit": launch_provenance().base_commit,
                "affected_rows": len(environment_digests),
                "source_task_schema_version": "0.27",
                "derived_task_schema_version": SCHEMA_VERSION,
                "task_spec_digests": environment_digests,
            },
            "files": output_hashes,
        }
        destination = StoragePath(output)
        destination.mkdirs()
        for name in outputs:
            with (directory / name).open("rb") as source, (destination / name).open("wb") as target:
                shutil.copyfileobj(source, target)
        (destination / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def main() -> None:
    configure_coreweave_s3()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", required=True)
    parser.add_argument("--training-sha256", required=True)
    parser.add_argument("--training-rows", type=int, required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--validation-sha256", required=True)
    parser.add_argument("--validation-rows", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prepare(**vars(args))


if __name__ == "__main__":
    main()
