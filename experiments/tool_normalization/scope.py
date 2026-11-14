# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze eligible source shards and locate diagnostic pilot rows without inference."""

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

EXCLUDED_SOURCES = frozenset(
    f"nemotron_sft_v3/agentic_{version}/{subset}"
    for version in ("v1", "v2")
    for subset in ("interactive_agent", "tool_calling")
)
PYTHON_TOOLS = frozenset(
    (
        "stateful_python_code_exec",
        "arc_python_executor",
        "evaluate_code",
        "sandbox_exec",
        "python_interpreter",
        "jupyter_kernel_run",
    )
)


def family(kind: str, tool: str) -> str:
    if kind != "native_function":
        return "terminal"
    if tool in PYTHON_TOOLS:
        return "python"
    if tool in ("bash", "execute_bash"):
        return "shell"
    if tool in ("str_replace_editor", "edit", "write", "read", "grep", "glob"):
        return "files"
    return "control_search"


def retained_inventory(inventory: dict) -> dict:
    retained = [item for item in inventory["sources"] if item["source"] not in EXCLUDED_SOURCES]
    excluded = [item for item in inventory["sources"] if item["source"] in EXCLUDED_SOURCES]
    return {
        "sources": retained,
        "excluded": [item["source"] for item in excluded],
        "retained_rows": sum(item["ledger"]["total_num_rows"] for item in retained),
        "excluded_rows": sum(item["ledger"]["total_num_rows"] for item in excluded),
        "catalog": inventory["catalog"],
        "catalog_sha256": inventory["catalog_sha256"],
    }


def candidate_pool(shards: list[dict]) -> list[dict]:
    """Deduplicate the scan's diagnostic examples by original row identity."""
    candidates = {}
    for shard in sorted(shards, key=lambda item: (item["source"], item["path"])):
        if shard["source"] in EXCLUDED_SOURCES:
            continue
        if not shard["complete"]:
            raise ValueError("Cannot sample incomplete scan evidence")
        for key, example in shard["examples"].items():
            kind, tool = key.split("\t", 1)
            identity = (shard["source"], shard["path"], example["row"])
            row = candidates.setdefault(
                identity,
                {
                    "source": identity[0],
                    "shard": identity[1],
                    "row": identity[2],
                    "observed_tools": [],
                    "families": [],
                },
            )
            row["observed_tools"].append(key)
            row["families"].append(family(kind, tool))
    return [
        {**row, "observed_tools": sorted(set(row["observed_tools"])), "families": sorted(set(row["families"]))}
        for row in sorted(candidates.values(), key=lambda row: (row["source"], row["shard"], row["row"]))
    ]


def pilot_candidates(shards: list[dict], seed: int, per_family: int) -> list[dict]:
    """Select distinct source rows; force coverage of observed retained tool names."""
    ordered = candidate_pool(shards)
    random.Random(seed).shuffle(ordered)
    selected = []
    covered = set()
    counts = Counter()
    for row in ordered:
        names = set(row["observed_tools"])
        families = set(row["families"])
        if names - covered or any(counts[name] < per_family for name in families) or len(selected) < 5 * per_family:
            selected.append({**row, "observed_tools": sorted(names), "families": sorted(families)})
            covered.update(names)
            counts.update(families)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--per-family", type=int, required=True)
    args = parser.parse_args()
    inventory_path = args.scan / "inventory.json"
    inventory = retained_inventory(json.loads(inventory_path.read_text()))
    inventory["inventory_sha256"] = hashlib.sha256(inventory_path.read_bytes()).hexdigest()
    shards = [json.loads(path.read_text()) for path in sorted((args.scan / "shards").glob("*.json"))]
    pilot = pilot_candidates(shards, args.seed, args.per_family)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "retained-inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    (args.output / "pilot-candidates.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "rows": pilot,
                "sampling": "Diagnostic first-observed examples; not a prevalence sample; lengths pending acquisition",
                "long_trace_selection_pending": True,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "retained_rows": inventory["retained_rows"],
                "excluded_rows": inventory["excluded_rows"],
                "pilot_candidates": len(pilot),
            }
        )
    )


if __name__ == "__main__":
    main()
