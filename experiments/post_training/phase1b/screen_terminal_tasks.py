# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Split terminal task IDs and screen their prompts against Stage 0 terminal users.

The split groups prompts sharing any normalized 13-gram, so no train/held-out
pair can share one. The output records every Stage 0 match, including near-copy
share, and excludes all matched tasks conservatively.
"""

import argparse
import gzip
import hashlib
import json
from collections import defaultdict

import pyarrow.parquet as pq
from rigging.filesystem.buckets import filesystem_for

from experiments.post_training.probe_sets.builder import overlap_grams

SPLIT_SALT = "phase1b-20261108"


def task_components(rows: list[dict]) -> dict[str, str]:
    """Map each task ID to the lowest ID in its 13-gram overlap component."""
    parent = {row["task_id"]: row["task_id"] for row in rows}

    def find(task_id: str) -> str:
        while parent[task_id] != task_id:
            parent[task_id] = parent[parent[task_id]]
            task_id = parent[task_id]
        return task_id

    seen: dict[tuple[str, ...], str] = {}
    for row in rows:
        task_id = row["task_id"]
        for gram in overlap_grams(row["description"]):
            other = seen.setdefault(gram, task_id)
            left, right = find(task_id), find(other)
            if left != right:
                parent[max(left, right)] = min(left, right)
    return {task_id: find(task_id) for task_id in parent}


def stage0_match_shares(rows: list[dict], stage0_url: str) -> tuple[dict[str, float], int]:
    """Stream Stage 0 user turns and count the share of task 13-grams matched."""
    owners: dict[tuple[str, ...], set[str]] = defaultdict(set)
    sizes = {}
    for row in rows:
        grams = overlap_grams(row["description"])
        sizes[row["task_id"]] = len(grams)
        for gram in grams:
            owners[gram].add(row["task_id"])
    matched: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    filesystem, path = filesystem_for(stage0_url)
    count = 0
    with filesystem.open(path, "rb") as raw, gzip.open(raw, "rt") as stream:
        for line in stream:
            record = json.loads(line)
            user_text = "\n".join(m["content"] for m in record["messages"] if m["role"] == "user")
            for gram in overlap_grams(user_text):
                for task_id in owners.get(gram, ()):
                    matched[task_id].add(gram)
            count += 1
    return {task_id: len(grams) / sizes[task_id] for task_id, grams in matched.items()}, count


def split_manifest(rows: list[dict], shares: dict[str, float], stage0_rows: int, *, source: str) -> dict:
    """Make the screened, reproducible task-ID split and its screening counts."""
    components = task_components(rows)
    excluded = set(shares)
    train, held_out = [], []
    for task_id in sorted(components):
        if task_id in excluded:
            continue
        digest = hashlib.sha256(f"{SPLIT_SALT}:{components[task_id]}".encode()).hexdigest()
        (held_out if int(digest, 16) % 5 == 0 else train).append(task_id)
    return {
        "source": source,
        "split_salt": SPLIT_SALT,
        "stage0_rows": stage0_rows,
        "source_tasks": len(rows),
        "overlap_components": len(set(components.values())),
        "excluded_stage0_any_13gram": len(excluded),
        "stage0_match_shares": shares,
        "train_task_ids": train,
        "held_out_task_ids": held_out,
        "train_tasks": len(train),
        "held_out_tasks": len(held_out),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-url", required=True)
    parser.add_argument("--prompt-column", choices=("description", "instruction"), default="description")
    parser.add_argument("--stage0-url", required=True)
    parser.add_argument("--output-url", required=True)
    args = parser.parse_args()
    tasks_filesystem, tasks_path = filesystem_for(args.tasks_url)
    with tasks_filesystem.open(tasks_path, "rb") as source:
        rows = pq.read_table(source, columns=["task_id", args.prompt_column]).to_pylist()
    if args.prompt_column != "description":
        rows = [{"task_id": row["task_id"], "description": row[args.prompt_column]} for row in rows]
    shares, count = stage0_match_shares(rows, args.stage0_url)
    manifest = split_manifest(rows, shares, count, source=args.tasks_url)
    filesystem, path = filesystem_for(args.output_url)
    with filesystem.open(path, "wb") as output:
        output.write(json.dumps(manifest, sort_keys=True, indent=2).encode())
    print(json.dumps({key: value for key, value in manifest.items() if not isinstance(value, (dict, list))}))


if __name__ == "__main__":
    main()
