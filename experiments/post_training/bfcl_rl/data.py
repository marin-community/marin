# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize the BFCL complement without regenerating or rewriting tasks."""

import hashlib
import json
import shutil
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

FULL_TASK_COUNT = 3641
PARITY_TASK_COUNT = 123
DATASET_COMMIT = "6bedd7878dc5d6f3456b4d80b781eb3c2d84f262"
PARTITION_MANIFEST_SHA256 = "59abec59ea96a73266257bb14299cda0c3f99a8f3f79f2ced83bc417098fa1e1"


@dataclass(frozen=True)
class TaskIdentity:
    name: str
    source_id: str
    digest: str


@dataclass(frozen=True)
class BFCLPartition:
    dataset_commit: str
    complement: tuple[TaskIdentity, ...]
    parity: tuple[TaskIdentity, ...]


def task_identity(path: Path) -> TaskIdentity:
    """Hash all task files with their relative paths and retain the original source ID."""
    config = tomllib.loads((path / "task.toml").read_text())
    digest = hashlib.sha256()
    for file in sorted(path.rglob("*")):
        if file.is_file():
            relative_path = file.relative_to(path).as_posix().encode()
            content = file.read_bytes()
            digest.update(len(relative_path).to_bytes(8, "big"))
            digest.update(relative_path)
            digest.update(len(content).to_bytes(8, "big"))
            digest.update(content)
    return TaskIdentity(path.name, config["metadata"]["source_id"], digest.hexdigest())


def bfcl_partition(full_root: Path, parity_root: Path, excluded_ids_file: Path) -> BFCLPartition:
    """Audit the pinned full/parity inputs against the complement exclusion manifest."""
    full = tuple(task_identity(path) for path in sorted(full_root.iterdir()) if path.is_dir())
    parity = tuple(task_identity(path) for path in sorted(parity_root.iterdir()) if path.is_dir())
    full_by_id = {task.source_id: task for task in full}
    parity_by_id = {task.source_id: task for task in parity}
    excluded = frozenset(
        line.strip()
        for line in excluded_ids_file.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if len(full) != FULL_TASK_COUNT or len(parity) != PARITY_TASK_COUNT:
        raise ValueError(f"BFCL source count changed: full={len(full)}, parity={len(parity)}")
    if len(full_by_id) != len(full) or len(parity_by_id) != len(parity):
        raise ValueError("BFCL source IDs must be unique")
    if parity_by_id.keys() != excluded:
        raise ValueError("parity tasks differ from the complement exclusion manifest")
    for source_id, task in parity_by_id.items():
        if full_by_id.get(source_id) != task:
            raise ValueError(f"parity task differs from its pinned full-dataset counterpart: {source_id}")
    complement = tuple(task for task in full if task.source_id not in excluded)
    return BFCLPartition(DATASET_COMMIT, complement, parity)


def materialize_complement(partition: BFCLPartition, full_root: Path, output_root: Path) -> None:
    """Copy audited complement tasks unchanged and write their manifest beside the task directory."""
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True)
    for task in partition.complement:
        target = output_root / task.name
        shutil.copytree(full_root / task.name, target)
        if task_identity(target) != task:
            raise ValueError(f"BFCL task content changed during materialization: {task.name}")
    manifest_path = output_root.parent / f"{output_root.name}-manifest.json"
    manifest_path.write_text(json.dumps(asdict(partition), indent=2, sort_keys=True) + "\n")
