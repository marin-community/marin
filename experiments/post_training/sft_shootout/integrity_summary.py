# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Summarize a model-free integrity run (``harbor_glm.py --agent oracle|null``) over pool tasks.

An intact task scores 1 under the oracle (its ``solution/solve.sh`` passes its own verifier) and 0
under the null agent (the unmodified environment fails). Writes ``integrity.jsonl`` (task_id, reward,
exception, as_expected) next to the run and prints the counts and every task that is not as expected.

    uv run python experiments/post_training/sft_shootout/integrity_summary.py <run prefix> --expect 1
"""

import argparse
import json
from collections import Counter

from rigging.filesystem.storage_path import StoragePath


def trial_rows(root: StoragePath) -> list[dict]:
    """One row per task: its final trial result (Harbor retries are inside the trial folder)."""
    rows = []
    for directory, _, names in root.walk():
        if "result.json" not in names or "/attempts/" in f"{directory}/" or str(directory) == str(root):
            continue
        result = json.loads(StoragePath(f"{directory}/result.json").read_text())
        if "task_name" not in result:
            continue
        rows.append(
            {
                "task_id": result["task_name"].rsplit("/", 1)[-1],
                "reward": ((result.get("verifier_result") or {}).get("rewards") or {}).get("reward"),
                "exception": (result.get("exception_info") or {}).get("exception_type"),
            }
        )
    return sorted(rows, key=lambda row: row["task_id"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("prefix")
    parser.add_argument("--expect", type=float, required=True, help="Reward of an intact task: 1 (oracle) or 0 (null).")
    args = parser.parse_args()
    prefix = args.prefix.rstrip("/")
    rows = trial_rows(StoragePath(prefix))
    for row in rows:
        row["as_expected"] = row["exception"] is None and row["reward"] == args.expect
    with StoragePath(f"{prefix}/integrity.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps({
        "run": prefix,
        "tasks": len(rows),
        "as_expected": sum(row["as_expected"] for row in rows),
        "rewards": Counter(str(row["reward"]) for row in rows),
        "exceptions": Counter(row["exception"] for row in rows if row["exception"]),
    }))
    for row in rows:
        if not row["as_expected"]:
            print(json.dumps(row))


if __name__ == "__main__":
    main()
