# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recreate original-policy subset configs without validation-profile overrides."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True, help="Downloaded index repository root")
    parser.add_argument("--budget", type=int, choices=(210, 840), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.index / "index/tasks.jsonl").read_text().splitlines()]
    rows = sorted(rows, key=lambda row: row["selection_order"])[: args.budget]
    assert len(rows) == args.budget and len({r["task_id"] for r in rows}) == args.budget
    groups = defaultdict(list)
    for row in rows:
        groups[row["benchmark"]].append(row)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for benchmark, selected in sorted(groups.items()):
        source = selected[0]["policy"]
        if source["kind"] == "harbor":
            config = yaml.safe_load(source["original_config"])
            assert len(config["datasets"]) == 1
            config["datasets"][0]["task_names"] = sorted(row["policy"]["task_name"] for row in selected)
        else:
            config = yaml.safe_load(source["original_parent"]["config_text"])
            assert len(config["tasks"]) == 1
            keys = sorted({key for row in selected for key in row["policy"]["source_keys"]})
            config.setdefault("task_options", {}).setdefault(config["tasks"][0], {})["selected_source_keys"] = keys
        path = args.output / (benchmark + ".yaml")
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        manifest.append(
            {"benchmark": benchmark, "kind": source["kind"], "task_count": len(selected), "config": path.name}
        )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"tasks": args.budget, "benchmarks": len(groups), "output": str(args.output)}))


if __name__ == "__main__":
    main()
