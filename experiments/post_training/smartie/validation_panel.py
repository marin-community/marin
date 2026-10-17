# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish a fixed, family-balanced validation panel from the admitted mixed cohort."""

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from types import MappingProxyType

import click
import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath
from rigging.filesystem.transfer import copy
from rolloutengine.spec import LoweredTaskSpec
from taskcompendium.grader import grader_config

REAL_ROUTE_COUNTS = MappingProxyType({"calendar": 167, "competitive_code": 167, "reasoning_gym": 167, "stem_mcqa": 166})
STRUCTURAL_COUNT = 333
RANK_SALT = "smartie-fast-val-v2"


def selected_structural_ids(groups: Mapping[str, Sequence[str]], count: int) -> set[str]:
    """Rank task IDs within each family/control group and assign near-equal quotas."""
    if not groups or count < 1:
        raise ValueError("Structural groups and selection count must be nonempty")
    keys = sorted(groups)
    base, remainder = divmod(count, len(keys))
    selected = set()
    for index, key in enumerate(keys):
        quota = base + (index < remainder)
        candidates = sorted(
            groups[key],
            key=lambda task_id: (hashlib.sha256(f"{RANK_SALT}:{task_id}".encode()).hexdigest(), task_id),
        )
        if len(candidates) < quota:
            raise ValueError(f"Insufficient structural rows for {key}: {len(candidates)} < {quota}")
        selected.update(candidates[:quota])
    if len(selected) != count:
        raise ValueError("Structural task identities overlap across groups")
    return selected


def file_hash(path: StoragePath) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build_validation_panel(source: StoragePath, output: StoragePath) -> dict:
    """Keep all held-out real tasks and sample structural controls without changing training bytes."""
    if output.exists():
        raise FileExistsError(f"Validation panel output must be new: {output}")
    manifest_bytes = (source / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest["inputs"]["validation"]["routes"] != {
        **REAL_ROUTE_COUNTS,
        "structural": manifest["inputs"]["validation"]["structural"]["rows"],
    }:
        raise ValueError("Mixed validation routes differ from the admitted panel contract")
    validation_path = source / "validation.parquet"
    with validation_path.open("rb") as stream:
        source_validation_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        stream.seek(0)
        table = pq.read_table(stream)
    groups: dict[str, list[str]] = defaultdict(list)
    row_ids = []
    real_counts = Counter()
    for row in table.to_pylist():
        lowered = LoweredTaskSpec.model_validate_json(row["lowered_task_spec"])
        task_id = lowered.task.id
        row_ids.append(task_id)
        route = row["teacher_route"]
        if route == "structural":
            extras = grader_config(lowered.task)["extras"]
            family = extras["family"]
            control = extras["control"]
            if not isinstance(family, str) or not isinstance(control, str):
                raise ValueError(f"Structural family/control missing for {task_id}")
            groups[f"{family}/{control}"].append(task_id)
        else:
            real_counts[route] += 1
    if len(set(row_ids)) != len(row_ids) or dict(real_counts) != REAL_ROUTE_COUNTS:
        raise ValueError("Mixed validation task identities or real-route counts differ")
    structural_ids = selected_structural_ids(groups, STRUCTURAL_COUNT)
    selected = table.filter(
        pa.array(
            [
                route != "structural" or task_id in structural_ids
                for task_id, route in zip(row_ids, table["teacher_route"].to_pylist(), strict=True)
            ]
        )
    )
    if selected.num_rows != sum(REAL_ROUTE_COUNTS.values()) + STRUCTURAL_COUNT:
        raise ValueError("Validation panel row count differs from the selection contract")

    output.mkdirs(exist_ok=False)
    train_source = source / "train.parquet"
    train_output = output / "train.parquet"
    copy(str(train_source), str(train_output))
    source_train_hash = file_hash(train_source)
    if file_hash(train_output) != source_train_hash:
        raise ValueError("Training bytes changed while creating the validation panel")
    with (output / "validation.parquet").open("wb") as stream:
        pq.write_table(selected, stream)
    report = {
        "source": str(source),
        "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "source_validation_sha256": source_validation_hash,
        "source_train_sha256": source_train_hash,
        "selection": "SHA-256 rank by fixed salt and task ID within each structural family/control group",
        "rank_salt": RANK_SALT,
        "real_routes": dict(REAL_ROUTE_COUNTS),
        "structural_groups": {key: {"eligible": len(ids)} for key, ids in sorted(groups.items())},
        "selected_structural_ids": sorted(structural_ids),
        "train_rows": sum(
            entry["rows"]
            for entry in manifest["inputs"]["train"].values()
            if isinstance(entry, dict) and "rows" in entry
        ),
        "validation_rows": selected.num_rows,
        "output_sha256": {
            "train.parquet": file_hash(train_output),
            "validation.parquet": file_hash(output / "validation.parquet"),
        },
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


@click.command()
@click.option("--source", required=True)
@click.option("--output", required=True)
def main(source: str, output: str) -> None:
    report = build_validation_panel(StoragePath(source), StoragePath(output))
    click.echo(json.dumps({"train": report["train_rows"], "validation": report["validation_rows"]}))


if __name__ == "__main__":
    main()
