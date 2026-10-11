# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize audited BFCL training tasks after reserving protocol diagnostics."""

import argparse
import hashlib
import json
from pathlib import Path

from experiments.post_training.bfcl_rl.data import BFCLPartition, TaskIdentity, materialize_complement, task_identity


def materialize_training_tasks(source: Path, partition_file: Path, diagnostic_manifest: Path, output: Path) -> dict:
    """Copy the unchanged training subset with immutable split provenance."""
    if output.exists():
        raise FileExistsError(output)
    partition_bytes = partition_file.read_bytes()
    diagnostic_bytes = diagnostic_manifest.read_bytes()
    partition = json.loads(partition_bytes)
    diagnostic = json.loads(diagnostic_bytes)
    if hashlib.sha256(partition_bytes).hexdigest() != diagnostic["source_partition_sha256"]:
        raise ValueError("Diagnostic reservations refer to a different BFCL partition")
    approved = tuple(TaskIdentity(**item) for item in partition["complement"])
    parity = tuple(TaskIdentity(**item) for item in partition["parity"])
    approved_ids = {item.source_id for item in approved}
    parity_ids = {item.source_id for item in parity}
    train_ids = set(diagnostic["train_task_ids"])
    reserved_ids = set(diagnostic["diagnostic_reserved_task_ids"])
    if (
        train_ids & reserved_ids
        or (train_ids | reserved_ids) != approved_ids
        or approved_ids & parity_ids
        or parity_ids != set(diagnostic["parity_task_ids"])
    ):
        raise ValueError("Training and diagnostic IDs must partition the complement and exclude BFCL-Parity")
    selected = tuple(item for item in approved if item.source_id in train_ids)
    failures = []
    for expected in selected:
        try:
            actual = task_identity(source / expected.name)
        except (OSError, ValueError, KeyError) as error:
            failures.append({"task": expected.name, "error": str(error)})
            continue
        if actual != expected:
            failures.append({"task": expected.name, "error": "task identity differs from pinned partition"})
    if failures:
        raise ValueError(json.dumps({"conversion_failures": failures}, indent=2))
    materialize_complement(BFCLPartition(partition["dataset_commit"], selected, parity), source, output / "tasks")
    manifest = {
        "diagnostic_version": diagnostic["version"],
        "source_partition_sha256": hashlib.sha256(partition_bytes).hexdigest(),
        "diagnostic_manifest_sha256": hashlib.sha256(diagnostic_bytes).hexdigest(),
        "source_dataset_commit": partition["dataset_commit"],
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "training_tasks": len(selected),
        "train_task_ids": sorted(train_ids),
        "diagnostic_reserved_task_ids": sorted(reserved_ids),
        "parity_task_ids": sorted(parity_ids),
        "transformation": "unchanged task files; exclude diagnostic reservations",
    }
    with (output / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--diagnostic-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = materialize_training_tasks(args.source, args.partition, args.diagnostic_manifest, args.output)
    print(json.dumps({"training_tasks": manifest["training_tasks"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
