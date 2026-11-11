# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select a balanced held-out live panel from screened TMax Harbor tasks."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import yaml

HARBOR_RELEASE = "sha256:119bd1ba7bb6b948f9313a6379e3182d30cfe1afd182167162d8a6240ab164e4"
HARBOR_DATASET = "tmax/tmax-13k-v1.1"
SELECTION_SALT = "phase1b-heldout-20261108"
LIVE_TASKS = 100
SMOKE_TASKS = 4


def complexity_label(value: str) -> str:
    for label in ("short", "moderate", "complex", "intricate"):
        if value.startswith(f"{label} task"):
            return label
    raise ValueError(f"unknown task complexity: {value}")


def selection_key(task_id: str) -> str:
    return hashlib.sha256(f"{SELECTION_SALT}:{task_id}".encode()).hexdigest()


def balanced_live_rows(rows: list[dict], held_ids: set[str]) -> list[dict]:
    """Pick two per domain/complexity cell, then fill by salted hash to 100."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        if row["task_id"] in held_ids:
            groups[(row["domain"], complexity_label(row["task_complexity"]))].append(row)
    if len(groups) != 36:
        raise ValueError(f"expected all 36 domain/complexity cells, found {len(groups)}")
    selected = []
    for key in sorted(groups):
        members = sorted(groups[key], key=lambda row: selection_key(row["task_id"]))
        if len(members) < 2:
            raise ValueError(f"held-out cell {key} has fewer than two tasks")
        selected.extend(members[:2])
    chosen_ids = {row["task_id"] for row in selected}
    remaining = sorted(
        (row for row in rows if row["task_id"] in held_ids - chosen_ids),
        key=lambda row: selection_key(row["task_id"]),
    )
    selected.extend(remaining[: LIVE_TASKS - len(selected)])
    if len(selected) != LIVE_TASKS:
        raise ValueError(f"only {len(selected)} held-out tasks available")
    return sorted(selected, key=lambda row: row["task_id"])


def harbor_config(base: dict, task_names: list[str]) -> dict:
    """Use the same Terminus-2/Daytona settings as the pinned TB2 config."""
    config = dict(base)
    config["datasets"] = [{"name": HARBOR_DATASET, "ref": HARBOR_RELEASE, "task_names": task_names}]
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = pq.read_table(
        args.tasks, columns=["task_id", "instruction", "domain", "task_complexity", "harbor_task_name"]
    ).to_pylist()
    split = json.loads(args.split.read_text())
    if split["source_tasks"] != len(rows):
        raise ValueError("screened source task count differs from the task table")
    if len({row["task_id"] for row in rows}) != len(rows):
        raise ValueError("task IDs are not unique")
    held_ids = set(split["held_out_task_ids"])
    selected = balanced_live_rows(rows, held_ids)
    live_names = [row["harbor_task_name"] for row in selected]
    if len(set(live_names)) != LIVE_TASKS:
        raise ValueError("selected Harbor names are not unique")
    smoke_ids = set()
    for label in ("short", "moderate", "complex", "intricate"):
        candidates = [row for row in selected if complexity_label(row["task_complexity"]) == label]
        smoke_ids.add(min(candidates, key=lambda row: selection_key(row["task_id"]))["task_id"])
    smoke_names = [row["harbor_task_name"] for row in selected if row["task_id"] in smoke_ids]
    if len(smoke_names) != SMOKE_TASKS:
        raise ValueError("smoke selection must contain one task per complexity")
    base = yaml.safe_load(args.base_config.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "harbor-live.yaml").write_text(yaml.safe_dump(harbor_config(base, live_names), sort_keys=False))
    (args.output_dir / "harbor-smoke.yaml").write_text(yaml.safe_dump(harbor_config(base, smoke_names), sort_keys=False))
    manifest = {
        "source": str(args.tasks),
        "stage0_split": str(args.split),
        "harbor_dataset": HARBOR_DATASET,
        "harbor_release": HARBOR_RELEASE,
        "selection_salt": SELECTION_SALT,
        "live_prompt_sha256": (
            hashlib.sha256("".join(f"{row['task_id']}\t{row['instruction']}\n" for row in selected).encode()).hexdigest()
        ),
        "live_task_ids": [row["task_id"] for row in selected],
        "smoke_task_ids": sorted(smoke_ids),
        "live_domain_counts": dict(sorted(Counter(row["domain"] for row in selected).items())),
        "live_complexity_counts": dict(
            sorted(Counter(complexity_label(row["task_complexity"]) for row in selected).items())
        ),
    }
    (args.output_dir / "selection.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in manifest.items() if key not in ("live_task_ids", "smoke_task_ids")}))


if __name__ == "__main__":
    main()
