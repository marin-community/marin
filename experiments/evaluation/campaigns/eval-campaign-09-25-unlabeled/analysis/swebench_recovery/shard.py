# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare disjoint recovery jobs from an audited, stopped source run."""

import argparse
import copy
import hashlib
import json
import runpy
from pathlib import Path

import yaml

COMBINE = runpy.run_path(str(Path(__file__).with_name("combine.py")))


def partition(report: dict, config: dict, count: int) -> tuple[dict, list[dict]]:
    """Retain audited outcomes and assign every other task to exactly one job."""
    source = report["manifest"]
    all_tasks = {row["task_name"] for row in source["retained_trials"]} | set(source["recovery_tasks"])
    if len(all_tasks) != 500 or count < 2:
        raise ValueError("expected a 500-task cohort and at least two shards")
    rows = [{key: value for key, value in row.items() if key != "result"} for row in report["trials"]]
    if len({row["task_name"] for row in rows}) != len(rows):
        raise ValueError("duplicate audited outcomes")
    if any(row["task_name"] not in all_tasks for row in rows):
        raise ValueError("unknown audited task")
    retained = [row for row in rows if COMBINE["PREPARE"]["recovery_reason"](row) is None]
    if len(retained) != report["n_scored"]:
        raise ValueError("audit coverage changed")
    remaining = sorted(all_tasks - {row["task_name"] for row in retained})
    if len(remaining) < count:
        raise ValueError("not enough remaining tasks for nonempty shards")
    manifest = {
        **source,
        "retained_trials": retained,
        "recovery_tasks": remaining,
        "excluded_trials": [row for row in rows if row not in retained],
        "source_must_be_terminal_before_launch": True,
    }
    manifest.pop("source_summary", None)
    manifest.pop("source_runtime", None)
    canonical = copy.deepcopy(config)
    canonical["datasets"][0]["task_names"] = remaining
    shards = []
    for index in range(count):
        shard = copy.deepcopy(canonical)
        shard["datasets"][0]["task_names"] = remaining[index::count]
        shards.append(shard)
    COMBINE["validate_shards"](canonical, shards, remaining)
    return manifest, shards


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--source-audit-archive", required=True, help="Durable archive of the exact input audit")
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--source-root", required=True, help="Stopped Iris root whose audited outcomes are retained")
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = args.audit.read_bytes()
    config = yaml.safe_load(args.base_config.read_text())
    manifest, shards = partition(json.loads(payload), config, args.count)
    manifest["audit_sha256"] = hashlib.sha256(payload).hexdigest()
    manifest["source_audit_archive"] = args.source_audit_archive
    manifest["base_config_sha256"] = hashlib.sha256(args.base_config.read_bytes()).hexdigest()
    manifest["source_root"] = args.source_root
    config["datasets"][0]["task_names"] = manifest["recovery_tasks"]
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
    (args.output / "swebench-verified-recovery.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    for index, shard in enumerate(shards):
        (args.output / f"swebench-shard-{index}.yaml").write_text(yaml.safe_dump(shard, sort_keys=False))
    print(
        json.dumps(
            {
                "retained": len(manifest["retained_trials"]),
                "shard_sizes": [len(s["datasets"][0]["task_names"]) for s in shards],
            }
        )
    )


if __name__ == "__main__":
    main()
