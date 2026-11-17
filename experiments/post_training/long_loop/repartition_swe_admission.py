# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare smaller recovery inputs after selected CPU workers are terminal."""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.long_loop.admission_index import committed_task
from experiments.post_training.long_loop.rl_dataset_admission import file_digest
from experiments.post_training.long_loop.swe_verified_data import RUNTIME


def recovery_inputs(
    source: Path, admission_prefix: str, indices: list[int], terminal: Path, output: Path, workers: int
) -> dict:
    """Preserve committed tasks and reassign only controls without a completed inventory."""
    root = StoragePath(admission_prefix)
    raw = (root / "plan.json").read_bytes()
    plan = json.loads(raw)
    if not plan["submitted"] or plan["runtime_sha256"] != file_digest(RUNTIME):
        raise ValueError("Recovery requires a fully submitted plan using the current grader")
    if file_digest(source / "manifest.json") != plan["manifest_sha256"]:
        raise ValueError("Recovery source differs from the original admission corpus")
    selected = [shard for shard in plan["shards"] if shard["index"] in indices]
    if {shard["index"] for shard in selected} != set(indices):
        raise ValueError("Recovery requested an unknown shard")
    states = json.loads(terminal.read_text())
    if any(states[shard["job_id"]]["state"] not in ("killed", "succeeded", "failed") for shard in selected):
        raise ValueError("Every reassigned worker must be terminal before reading its final commits")
    candidates = [
        (
            str(root / "shards" / f"{shard['index']:04d}" / "tasks" / hashlib.sha256(task_id.encode()).hexdigest()),
            task_id,
        )
        for shard in selected
        for task_id in shard["task_ids"]
    ]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        committed = [row for row in pool.map(lambda pair: committed_task(*pair), candidates) if row is not None]
    preserved = {row["task_id"] for row in committed}
    pending = {task_id for _, task_id in candidates} - preserved
    original = json.loads((source / "manifest.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    found = set()
    for name in ("agentic-train.parquet", "validation.parquet"):
        if file_digest(source / name) != original["files"][name]["sha256"]:
            raise ValueError("Recovery source parquet changed")
        rows = []
        for batch in pq.ParquetFile(source / name).iter_batches(batch_size=32):
            for row in batch.to_pylist():
                task_id = json.loads(row["task_spec"])["id"]
                if task_id in pending:
                    rows.append(row)
                    found.add(task_id)
        pq.write_table(
            pa.Table.from_pylist(rows, schema=pq.read_schema(source / name)), output / name, compression="zstd"
        )
        files[name] = {"sha256": file_digest(output / name), "rows": len(rows)}
    if found != pending:
        raise ValueError("Some recovery tasks are absent from the pinned corpus")
    (output / "terminal-evidence.json").write_bytes(terminal.read_bytes())
    files["terminal-evidence.json"] = {"sha256": file_digest(output / "terminal-evidence.json")}
    result = {
        "files": files,
        "original_manifest_sha256": plan["manifest_sha256"],
        "original_plan_uri": str(root / "plan.json"),
        "original_plan_sha256": hashlib.sha256(raw).hexdigest(),
        "terminal_evidence_sha256": file_digest(terminal),
        "runtime_sha256": file_digest(RUNTIME),
        "selected_indices": indices,
        "preserved_commits": committed,
        "pending_task_ids": sorted(pending),
        "policy": "Only incomplete CPU controls are restarted; original evidence is never overwritten",
    }
    (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    configure_coreweave_s3()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--admission-prefix", required=True)
    parser.add_argument("--index", type=int, action="append", required=True)
    parser.add_argument("--terminal-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    result = recovery_inputs(
        args.input, args.admission_prefix, args.index, args.terminal_evidence, args.output, args.workers
    )
    print(json.dumps({"preserved": len(result["preserved_commits"]), "pending": len(result["pending_task_ids"])}))


if __name__ == "__main__":
    main()
