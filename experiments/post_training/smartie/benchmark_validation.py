# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze target-benchmark samples for Smartie's diagnostic inline validation.

Run with the pinned config/external/MarinSkyRL project, which supplies Harbor's
task identity types. Selection uses task IDs only, without model outcomes.
"""

import hashlib
import json
import shutil
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from types import MappingProxyType

import click
import pyarrow as pa
import pyarrow.parquet as pq
import shortuuid
from harbor_config.models.task.id import GitTaskId
from rolloutengine.spec import LoweredTaskSpec
from shellbox.image import DockerfileSource, image_source_key
from taskcompendium.grader import grader_config
from taskcompendium.harbor.snapshots import file_map_snapshot

from experiments.post_training.smartie.validation_panel import selected_structural_ids

BENCHMARKS = MappingProxyType({"bfcl_parity": "1.0", "swebench-verified": "1.0", "openthoughts-tblite": "2.0"})
NATIVE_REAL_COUNT = 667
NATIVE_STRUCTURAL_COUNT = 243
SELECTION_SALT = "smartie-target-benchmarks-20261015-v1"
SAMPLES_PER_BENCHMARK = 30


def build_native_panel(source: Path, output: Path) -> dict:
    """Keep all 667 native real tasks and 243 family-balanced structural tasks."""
    table = pq.read_table(source / "validation.parquet")
    rows = table.to_pylist()
    ids = [json.loads(row["lowered_task_spec"])["task"]["id"] for row in rows]
    groups = defaultdict(list)
    selected = set()
    for row, identity in zip(rows, ids, strict=True):
        if row["teacher_route"] != "structural":
            selected.add(identity)
        else:
            extras = grader_config(LoweredTaskSpec.model_validate_json(row["lowered_task_spec"]).task)["extras"]
            groups[f"{extras['family']}/{extras['control']}"].append(identity)
    if len(selected) != NATIVE_REAL_COUNT or len(set(ids)) != len(ids):
        raise ValueError("Expected a unique native holdout with 667 real tasks")
    selected.update(selected_structural_ids(groups, NATIVE_STRUCTURAL_COUNT))
    training_ids = {
        json.loads(payload)["task"]["id"]
        for payload in pq.read_table(source / "train.parquet", columns=["lowered_task_spec"])[
            "lowered_task_spec"
        ].to_pylist()
    }
    if selected & training_ids:
        raise ValueError("Native validation overlaps training IDs")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(source / "train.parquet", output / "train.parquet")
    pq.write_table(table.filter(pa.array([identity in selected for identity in ids])), output / "validation.parquet")
    hashes = {}
    for split in ("train", "validation"):
        with (output / f"{split}.parquet").open("rb") as stream:
            hashes[split] = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {
        "source": str(source),
        "selection": "Reuse native panel's fixed hash rank and family/control quotas",
        "native_real": NATIVE_REAL_COUNT,
        "native_structural": NATIVE_STRUCTURAL_COUNT,
        "external_benchmark_tasks": len(BENCHMARKS) * SAMPLES_PER_BENCHMARK,
        "selected_native_ids": sorted(selected),
        "sha256": hashes,
        "training_id_overlap": 0,
        "training_bytes_unchanged": True,
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def selected_tasks(registry: list[dict]) -> list[dict]:
    """Select a fixed equal-sized panel from each pinned benchmark release."""
    selected = []
    for name, version in BENCHMARKS.items():
        matches = [item for item in registry if item["name"] == name and item["version"] == version]
        if len(matches) != 1:
            raise ValueError(f"Expected one registry release for {name}@{version}")
        candidates = matches[0]["tasks"]
        if len({task["name"] for task in candidates}) != len(candidates):
            raise ValueError(f"Duplicate task identities in {name}@{version}")
        if len(candidates) < SAMPLES_PER_BENCHMARK:
            raise ValueError(f"Too few tasks in {name}@{version}")
        ranked = sorted(
            candidates,
            key=lambda task: hashlib.sha256(f"{SELECTION_SALT}:{name}:{task['name']}".encode()).hexdigest(),
        )
        selected.extend({"benchmark": name, "version": version, **task} for task in ranked[:SAMPLES_PER_BENCHMARK])
    return selected


def freeze_selection(registry_path: Path, cache: Path, output: Path) -> dict:
    """Record pinned task packages and build identities, failing on missing inputs."""
    registry_bytes = registry_path.read_bytes()
    rows = selected_tasks(json.loads(registry_bytes))
    missing = []
    for row in rows:
        identity = GitTaskId(**{key: row[key] for key in ("git_url", "git_commit_id", "path")})
        directory = cache / shortuuid.uuid(str(identity)) / identity.path.name
        if not (directory / "task.toml").is_file():
            missing.append(str(directory))
            continue
        paths = sorted(directory.rglob("*"))
        if any(path.is_symlink() for path in paths):
            raise ValueError(f"Task package contains symlinks: {directory}")
        files = {str(path.relative_to(directory)): path.read_bytes() for path in paths if path.is_file()}
        modes = {str(path.relative_to(directory)): path.stat().st_mode & 0o777 for path in paths if path.is_file()}
        snapshot = file_map_snapshot(files, modes, "task")
        row["local_package"] = str(directory)
        row["package_snapshot"] = {name: asdict(value) for name, value in snapshot.items()}
        context = directory / "environment"
        row["image_build_identity"] = image_source_key(DockerfileSource(context, context / "Dockerfile"))
    if missing:
        raise FileNotFoundError(f"Missing pinned Harbor packages: {missing}")
    report = {
        "purpose": "Diagnostic inline validation; benchmark policy harness and settings are not required",
        "selection_salt": SELECTION_SALT,
        "selection": "First 30 SHA-256-ranked task names per benchmark, independent of model outcomes",
        "registry_sha256": hashlib.sha256(registry_bytes).hexdigest(),
        "training_use": "Forbidden; validation only",
        "status": "Selected source packages; execution adapters and live verification remain required",
        "tasks": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Selection manifest must be new: {output}")
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


@click.command(help=__doc__)
@click.option("--registry", "registry_path", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--cache", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--output", type=click.Path(path_type=Path), required=True)
def main(registry_path: Path, cache: Path, output: Path) -> None:
    report = freeze_selection(registry_path, cache, output)
    click.echo(f"Selected {len(report['tasks'])} validation-only benchmark tasks: {output}")


if __name__ == "__main__":
    main()
