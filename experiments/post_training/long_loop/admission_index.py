# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Commit one complete receipt index across original and recovered CPU shards."""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.long_loop.rl_dataset_admission import file_digest
from experiments.post_training.long_loop.swe_verified_data import RUNTIME


def committed_task(root: str, task_id: str) -> dict | None:
    """Read a task only after the worker published its complete evidence inventory."""
    path = StoragePath(root)
    marker = path / "evidence-inventory.json"
    if not marker.exists():
        return None
    inventory_raw = marker.read_bytes()
    inventory = json.loads(inventory_raw)
    raw = (path / "receipt.json").read_bytes()
    checksum = hashlib.sha256(raw).hexdigest()
    if checksum != inventory["receipt.json"]["sha256"] or json.loads(raw)["task_id"] != task_id:
        raise ValueError(f"Published control receipt differs from its commit inventory: {root}")
    return {
        "task_id": task_id,
        "uri": root,
        "receipt_sha256": checksum,
        "inventory_sha256": hashlib.sha256(inventory_raw).hexdigest(),
    }


def finalize_admission(sources: list[str], output: str, workers: int) -> dict:
    """Reject missing or duplicate commits; reference existing evidence without copying it."""
    expected = set()
    candidates = []
    source_pins = []
    manifest = None
    for index, source in enumerate(sources):
        root = StoragePath(source)
        raw = (root / "plan.json").read_bytes()
        plan = json.loads(raw)
        if not plan["submitted"] or plan["runtime_sha256"] != file_digest(RUNTIME):
            raise ValueError("Admission plan is incomplete or uses a different grader")
        if json.loads((root / "input-exclusions.json").read_text()):
            raise ValueError("Source-pin exclusions require explicit resolution")
        task_ids = [task_id for shard in plan["shards"] for task_id in shard["task_ids"]]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("A source admission plan assigns a task more than once")
        if index == 0:
            expected = set(task_ids)
            manifest = plan["manifest_sha256"]
        elif not set(task_ids) <= expected:
            raise ValueError("Recovery plan introduces tasks outside the original corpus")
        source_pins.append({"uri": str(root / "plan.json"), "sha256": hashlib.sha256(raw).hexdigest()})
        for shard in plan["shards"]:
            for task_id in shard["task_ids"]:
                key = hashlib.sha256(task_id.encode()).hexdigest()
                candidates.append((str(root / "shards" / f"{shard['index']:04d}" / "tasks" / key), task_id))

    receipts = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for receipt in pool.map(lambda pair: committed_task(*pair), candidates):
            if receipt is None:
                continue
            task_id = receipt["task_id"]
            if task_id in receipts:
                raise ValueError(f"Duplicate committed controls require reconciliation: {task_id}")
            receipts[task_id] = receipt
    if set(receipts) != expected:
        raise ValueError(f"Admission lacks committed controls for {len(expected - set(receipts))} tasks")
    result = {
        "complete": True,
        "manifest_sha256": manifest,
        "runtime_sha256": file_digest(RUNTIME),
        "sources": source_pins,
        "tasks": [receipts[task_id] for task_id in sorted(receipts)],
    }
    target = StoragePath(output) / "index.json"
    raw = json.dumps(result, indent=2).encode()
    if target.exists():
        if target.read_bytes() != raw:
            raise ValueError("Completed admission index is immutable")
    else:
        target.write_bytes(raw)
    return result


def main() -> None:
    configure_coreweave_s3()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    result = finalize_admission(args.source, args.output, args.workers)
    print(json.dumps({"committed_tasks": len(result["tasks"])}))


if __name__ == "__main__":
    main()
