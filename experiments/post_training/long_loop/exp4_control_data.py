# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Partition the original Snowball Exp4 bytes without adopting later grader repairs."""

import hashlib
import json
import os
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import click
import draccus
import pyarrow.parquet as pq
from marin.execution.build_context import resolve_version
from marin.execution.lazy import OUT, ArtifactStep, apply
from marin.experiment.cli import build_options
from marin.rl.curriculum import (
    CurriculumEntry,
    CurriculumStage,
    PoolRow,
    RlCurriculum,
    SamplingMode,
    draw_validation_split,
    validation_payload,
)
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath, prefix_join
from taskcompendium.identity import canonical_sha256

from experiments.post_training.long_loop.exp3_data import PreparedPools
from experiments.post_training.long_loop.exp4_data import _write_pool_file
from experiments.post_training.long_loop.run_spec import FilePin, read_pin

PARENT_PLAN = FilePin(
    "s3://marin-us-east-08a/marin/long-loop/config/2026.11.19/exp4-recovery82-plan.json",
    "a0ddb064354834de62de27c68102b548009ceed476064f68fed5934bc0d177ca",
)
PARENT_CURRICULUM = "long-loop/exp4-rl@2026.11.18:cde16ecb"
OUTPUT_PREFIX = "s3://marin-us-east-08a/marin/users/benfeuer/mochi-exp4-dspark"
VALIDATION_SIZE = 200


def partition_rows(rows: list[dict], parent: dict, version: str) -> PreparedPools:
    """Keep each serialized task unchanged while removing a deterministic holdout."""
    entries = tuple(draccus.decode(CurriculumEntry, entry) for entry in parent["entries"])
    weights = {entry["label"]: entry["weight"] for entry in parent["entries"]}
    identities = {label: [] for label in weights}
    for row in rows:
        task = json.loads(row["task_spec"])
        identities[row["teacher_route"]].append(PoolRow(id=task["id"], sha256=canonical_sha256(task)))
    split = draw_validation_split(weights, identities, size=VALIDATION_SIZE, seed=parent["seed"])
    curriculum = RlCurriculum(
        name="long-loop/exp4-rl-v5-dspark",
        version=version,
        mode=SamplingMode(parent["mode"]),
        entries=entries,
        stages=tuple(draccus.decode(CurriculumStage, stage) for stage in parent["stages"]),
        seed=parent["seed"],
        validation=split.validation,
    )
    held_ids = {(row.label, row.id) for row in split.validation.rows}
    training = {label: [] for label in weights}
    heldout = []
    train_ids, train_hashes, val_ids, val_hashes = set(), set(), set(), set()
    for row in rows:
        task = json.loads(row["task_spec"])
        if (row["teacher_route"], task["id"]) in held_ids:
            heldout.append(row)
            val_ids.add(task["id"])
            val_hashes.add(canonical_sha256(task))
        else:
            training[row["teacher_route"]].append(row)
            train_ids.add(task["id"])
            train_hashes.add(canonical_sha256(task))
    if train_ids & val_ids or train_hashes & val_hashes:
        raise ValueError("Original Exp4 training and new validation overlap")
    return PreparedPools(curriculum, training, heldout)


def serialized_rows_sha256(rows: list[dict]) -> str:
    inventory = sorted((row["teacher_route"], hashlib.sha256(row["task_spec"].encode()).hexdigest()) for row in rows)
    return hashlib.sha256(json.dumps(inventory, separators=(",", ":")).encode()).hexdigest()


def prepare_pool(*, output: str, plan: str, curriculum_version: str) -> None:
    """Verify the recorded input parquets and publish only a new train/holdout partition."""
    source = read_pin(FilePin(plan, PARENT_PLAN.sha256), directory=Path.cwd())
    if source["curriculum"]["identity"] != PARENT_CURRICULUM:
        raise ValueError("The recorded Snowball Exp4 curriculum changed")
    rows = []
    sources = []
    for label in ("agentic", "nonagentic"):
        dataset = source["datasets"][label]
        uri = prefix_join(dataset["uri"], dataset["data_file"])
        pin = next(member for member in dataset["files"] if member["uri"] == uri)
        fs, path = filesystem_for(uri)
        with fs.open(path, "rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != pin["sha256"]:
            raise ValueError(f"Original Snowball Exp4 parquet changed: {uri}")
        count = 0
        with fs.open(path, "rb") as stream:
            for batch in pq.ParquetFile(stream).iter_batches(batch_size=128):
                batch_rows = batch.to_pylist()
                if set(batch.schema.names) != {"task_spec", "teacher_route"}:
                    raise ValueError("Original Exp4 parquet has an unexpected schema")
                rows.extend(batch_rows)
                count += len(batch_rows)
        if count != dataset["rows"]:
            raise ValueError(f"Original Snowball Exp4 row count changed: {uri}")
        sources.append({**pin, "rows": count})
    prepared = partition_rows(rows, source["curriculum"], curriculum_version)
    original_inventory = serialized_rows_sha256(rows)
    partitioned = [row for members in prepared.training.values() for row in members] + prepared.validation
    if serialized_rows_sha256(partitioned) != original_inventory:
        raise ValueError("The split changed original serialized TaskSpec bytes")
    destination = StoragePath(output)
    destination.mkdirs(exist_ok=True)
    training_files = {
        label: f"source-{index:03d}/train.parquet" for index, label in enumerate(sorted(prepared.training))
    }
    files = {
        training_files[label]: _write_pool_file(destination, training_files[label], members)
        for label, members in prepared.training.items()
    }
    files["validation.parquet"] = _write_pool_file(destination, "validation.parquet", prepared.validation)
    definition = asdict(prepared.curriculum)
    definition.pop("fingerprint")
    payload = json.dumps(definition, indent=2).encode()
    (destination / "curriculum.json").write_bytes(payload)
    files["curriculum.json"] = hashlib.sha256(payload).hexdigest()
    agentic = set(source["datasets"]["agentic"]["teacher_routes"])
    (destination / "manifest.json").write_text(
        json.dumps(
            {
                "curriculum": prepared.curriculum.identity,
                "curriculum_definition": definition,
                "parent_curriculum": PARENT_CURRICULUM,
                "parent_plan": asdict(PARENT_PLAN),
                "validation": validation_payload(prepared.curriculum.validation),
                "composition": dict(Counter(row["teacher_route"] for row in rows)),
                "train_composition": {label: len(members) for label, members in prepared.training.items()},
                "training_files": training_files,
                "routes": {label: "agentic" if label in agentic else "nonagentic" for label in prepared.training},
                "files": files,
                "rows": len(rows),
                "sources": sources,
                "serialized_task_inventory_sha256": original_inventory,
                "train_validation_id_overlap": 0,
                "train_validation_hash_overlap": 0,
                "task_content_changes": (
                    "None: serialized task_spec strings preserved exactly; only 200 heldout rows removed from training"
                ),
                "known_comparison_difference": (
                    "Snowball Exp4 used an external 120-prompt holdout (seed20261109); "
                    "this derived curriculum owns200 (seed20261020), minimum180valid"
                ),
            },
            indent=2,
        )
    )


def pool_steps(version: str) -> dict[str, ArtifactStep]:
    """Bind the immutable historical input record to the new split artifact."""
    plan = ArtifactStep.adopt(
        "long-loop-exp4/recorded-inputs", "2026.11.19.82", PARENT_PLAN.path, config=asdict(PARENT_PLAN)
    )
    return {
        "pools": apply(
            "long-loop/mochi-exp4-original-rl-pools",
            prepare_pool,
            version=version,
            output=OUT,
            plan=plan,
            curriculum_version=version,
        )
    }


@click.command(help=__doc__)
@build_options
def main() -> dict[str, ArtifactStep]:
    configure_coreweave_s3()
    os.environ["MARIN_PREFIX"] = OUTPUT_PREFIX
    return pool_steps(resolve_version("mochi-exp4-original-data", None))


if __name__ == "__main__":
    main()
