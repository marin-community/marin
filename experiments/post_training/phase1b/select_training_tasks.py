# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select a deterministic TMax training batch from the screened Phase 1b split."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import yaml

from experiments.post_training.phase1b.select_terminal_tasks import HARBOR_DATASET, HARBOR_RELEASE, complexity_label

TRAIN_TASKS = 1000
SELECTION_SALT = "phase1b-sft-train-seed0-20261108"


def selection_key(task_id: str) -> str:
    return hashlib.sha256(f"{SELECTION_SALT}:{task_id}".encode()).hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--offset", type=int, default=0, help="Start after this many screened train tasks")
    parser.add_argument("--count", type=int, default=TRAIN_TASKS, help="Number of tasks in this batch")
    args = parser.parse_args()
    if args.offset < 0 or args.count < 1:
        parser.error("offset must be nonnegative and count must be positive")

    rows = pq.read_table(
        args.tasks, columns=["task_id", "instruction", "domain", "task_complexity", "harbor_task_name"]
    ).to_pylist()
    split = json.loads(args.split.read_text())
    if len(rows) != split["source_tasks"] or len({row["task_id"] for row in rows}) != len(rows):
        raise ValueError("Task table and screened split disagree")
    train_ids = set(split["train_task_ids"])
    held_ids = set(split["held_out_task_ids"])
    if train_ids & held_ids or len(train_ids) != split["train_tasks"]:
        raise ValueError("Training and held-out task IDs overlap or have the wrong count")
    candidates = [row for row in rows if row["task_id"] in train_ids]
    if len(candidates) != len(train_ids):
        raise ValueError("Screened training task IDs are missing from the task table")
    selected = sorted(candidates, key=lambda row: selection_key(row["task_id"]))[args.offset : args.offset + args.count]
    if len(selected) != args.count or len({row["harbor_task_name"] for row in selected}) != args.count:
        raise ValueError("Training panel size or Harbor task names are wrong")

    config = yaml.safe_load(args.base_config.read_text())
    datasets = config.get("datasets")
    if len(datasets) != 1 or datasets[0]["name"] != HARBOR_DATASET or datasets[0]["ref"] != HARBOR_RELEASE:
        raise ValueError("Expected the pinned TMax v1.1 dataset release")
    datasets[0]["task_names"] = [row["harbor_task_name"] for row in selected]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "harbor-train.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    manifest = {
        "source": str(args.tasks),
        "screened_split": str(args.split),
        "harbor_dataset": HARBOR_DATASET,
        "harbor_release": HARBOR_RELEASE,
        "selection_salt": SELECTION_SALT,
        "offset": args.offset,
        "tasks": len(selected),
        "task_ids": [row["task_id"] for row in selected],
        "harbor_task_names": [row["harbor_task_name"] for row in selected],
        "heldout_overlap": len({row["task_id"] for row in selected} & held_ids),
        "domain_counts": dict(sorted(Counter(row["domain"] for row in selected).items())),
        "complexity_counts": dict(sorted(Counter(complexity_label(row["task_complexity"]) for row in selected).items())),
        "prompt_sha256": (
            hashlib.sha256("".join(f"{row['task_id']}\t{row['instruction']}\n" for row in selected).encode()).hexdigest()
        ),
        "task_table_sha256": sha256_file(args.tasks),
        "split_sha256": sha256_file(args.split),
    }
    (args.output_dir / "selection.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in manifest.items() if not isinstance(value, (dict, list))}))


if __name__ == "__main__":
    main()
