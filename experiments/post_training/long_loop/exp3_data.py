# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize the historical Exp3 RL pools with their curriculum-owned holdout."""

import hashlib
import json
import os
from collections import Counter
from dataclasses import asdict

import click
from marin.execution.build_context import resolve_version
from marin.execution.lazy import OUT, ArtifactStep, apply
from marin.experiment.cli import build_options
from marin.rl.curriculum import PoolRow, RlCurriculum, draw_validation_split, validation_payload
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath, prefix_join
from taskcompendium.identity import canonical_sha256

from experiments.post_training.long_loop.exp4_data import write_pool_file
from experiments.post_training.task_curation.datasets.long_loop_rl import (
    AGENTIC_RELEASE_URL,
    AGENTIC_TRAIN_ROWS,
    AGENTIC_TRAIN_SHA256,
    NONAGENTIC_RELEASE_URL,
    NONAGENTIC_TRAIN_ROWS,
    NONAGENTIC_TRAIN_SHA256,
    RELEASE_SOURCE_ROUTES,
    stream_task_specs,
)
from experiments.post_training.task_curation.exp3_curriculum import (
    SEED,
    VALIDATION_SIZE,
    WEIGHTS,
    exp3_curriculum_from_validation,
)


def partition_tasks(tasks: dict[str, list[dict]]) -> tuple[RlCurriculum, dict[str, list[dict]], list[dict]]:
    """Extract the deterministic holdout without changing any task document."""
    identities = {
        route: tuple(PoolRow(id=task["id"], sha256=canonical_sha256(task)) for task in rows)
        for route, rows in tasks.items()
    }
    split = draw_validation_split(dict(WEIGHTS), identities, size=VALIDATION_SIZE, seed=SEED)
    curriculum = exp3_curriculum_from_validation(split.validation)
    held_ids = {(row.label, row.id) for row in split.validation.rows}
    training: dict[str, list[dict]] = {route: [] for route in tasks}
    heldout = []
    for route, rows in tasks.items():
        for task in rows:
            row = {"task_spec": json.dumps(task), "teacher_route": route}
            if (route, task["id"]) in held_ids:
                heldout.append(row)
            else:
                training[route].append(row)
    train_tasks = [json.loads(row["task_spec"]) for rows in training.values() for row in rows]
    val_tasks = [json.loads(row["task_spec"]) for row in heldout]
    if {task["id"] for task in train_tasks} & {task["id"] for task in val_tasks}:
        raise ValueError("Exp3 holdout and training overlap by task ID")
    if {canonical_sha256(task) for task in train_tasks} & {canonical_sha256(task) for task in val_tasks}:
        raise ValueError("Exp3 holdout and training overlap by task content hash")
    return curriculum, training, heldout


def prepare_pool(*, output: str, agentic: str, blend: str) -> None:
    """Verify the original two training parquets and publish the disjoint split in-region."""
    tasks: dict[str, list[dict]] = {route: [] for route in WEIGHTS}
    source_files = []
    for uri, expected_hash in [(agentic, AGENTIC_TRAIN_SHA256), (blend, NONAGENTIC_TRAIN_SHA256)]:
        fs, path = filesystem_for(uri)
        with fs.open(path, "rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected_hash:
            raise ValueError(f"Exp3 training parquet digest differs: {uri}")
        source_files.append({"uri": uri, "sha256": actual})
        for task in stream_task_specs(fs, path):
            tasks[RELEASE_SOURCE_ROUTES[task["source"]["dataset"]]].append(task)
    composition = {route: len(rows) for route, rows in tasks.items()}
    if composition != {"agentic": AGENTIC_TRAIN_ROWS, "nonagentic": NONAGENTIC_TRAIN_ROWS}:
        raise ValueError(f"Exp3 training population changed: {composition}")
    curriculum, training, heldout = partition_tasks(tasks)
    destination = StoragePath(output)
    destination.mkdirs(exist_ok=True)
    files = {
        f"{route}/train.parquet": write_pool_file(destination, f"{route}/train.parquet", rows)
        for route, rows in training.items()
    }
    files["validation.parquet"] = write_pool_file(destination, "validation.parquet", heldout)
    document = json.dumps(asdict(curriculum), indent=2).encode()
    (destination / "curriculum.json").write_bytes(document)
    files["curriculum.json"] = hashlib.sha256(document).hexdigest()
    (destination / "manifest.json").write_text(
        json.dumps(
            {
                "curriculum": curriculum.identity,
                "validation": validation_payload(curriculum.validation),
                "composition": composition,
                "train_composition": {route: len(rows) for route, rows in training.items()},
                "source_composition": dict(
                    Counter(task["source"]["dataset"] for rows in tasks.values() for task in rows)
                ),
                "routes": {**{route: route for route in WEIGHTS}, **dict(RELEASE_SOURCE_ROUTES)},
                "files": files,
                "rows": sum(composition.values()),
                "sources": source_files,
                "train_validation_id_overlap": 0,
                "train_validation_hash_overlap": 0,
            },
            indent=2,
        )
    )


def pool_steps(version: str) -> dict[str, ArtifactStep]:
    """Adopt the two original Exp3 training releases and materialize their current holdout."""
    sources = [
        (prefix_join(AGENTIC_RELEASE_URL, "agentic-train.parquet"), AGENTIC_TRAIN_SHA256),
        (prefix_join(NONAGENTIC_RELEASE_URL, "nonagentic-train.parquet"), NONAGENTIC_TRAIN_SHA256),
    ]
    adopted = tuple(
        ArtifactStep.adopt(f"long-loop-exp3/admitted/{index}", version, source=uri, config={"sha256": sha})
        for index, (uri, sha) in enumerate(sources)
    )
    return {
        "pools": apply(
            "long-loop-exp3/rl-pools", prepare_pool, version=version, output=OUT, agentic=adopted[0], blend=adopted[1]
        )
    }


@click.command(help=__doc__)
@build_options
def main() -> dict[str, ArtifactStep]:
    configure_coreweave_s3()
    os.environ["MARIN_PREFIX"] = "s3://marin-us-east-08a/marin/users/benfeuer/mochi-v5-exp3-rl"
    return pool_steps(resolve_version("mochi-v5-exp3-data", None))


if __name__ == "__main__":
    main()
