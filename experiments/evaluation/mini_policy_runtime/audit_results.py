# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Summarize generation and verification evidence without printing model endpoints or credentials."""

import argparse
import json
from collections import Counter
from pathlib import Path

from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath


def audit(result_uri: str, expected_rollouts: int) -> dict:
    fs, pattern = filesystem_for(str(StoragePath(result_uri) / "exports" / "dumped_evals" / "eval_only" / "*.jsonl"))
    summaries = []
    failures = Counter()
    for path in sorted(fs.glob(pattern)):
        if path.endswith("/aggregated_results.jsonl"):
            continue
        rows = [json.loads(line) for line in fs.cat_file(path).decode().splitlines() if line.strip()]
        verified = [row for row in rows if row["verification"]["status"] == "verified"]
        for row in rows:
            if row["verification"]["status"] != "verified":
                failures[row.get("exception_type") or row["verification"]["status"]] += 1
        summaries.append(
            {
                "file": path.rsplit("/", 1)[-1],
                "rollouts": len(rows),
                "verified": len(verified),
                "mean_verified_reward": (
                    sum(row["verification"]["score"] for row in verified) / len(verified) if verified else None
                ),
                "response_tokens": sum(len(row["response_token_ids"]) for row in rows),
                "stop_reasons": dict(Counter(row["stop_reason"] for row in rows)),
            }
        )
    completed = sum(row["rollouts"] for row in summaries)
    return {
        "result_uri": result_uri,
        "expected_rollouts": expected_rollouts,
        "completed_rollouts": completed,
        "all_verified": completed == expected_rollouts and not failures,
        "unverified_types": dict(failures),
        "benchmarks": summaries,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-uri", required=True)
    parser.add_argument("--expected-rollouts", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.result_uri, args.expected_rollouts)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
