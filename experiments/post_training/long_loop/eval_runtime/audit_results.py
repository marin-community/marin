# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Summarize generation and verification evidence without printing model endpoints or credentials."""

import argparse
import json
from collections import Counter
from pathlib import Path

from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath


def audit(
    result_uri: str,
    expected_rollouts: int,
    expected_by_data_source: dict[str, int] | None = None,
    expected_identities: list[dict] | None = None,
) -> dict:
    fs, pattern = filesystem_for(str(StoragePath(result_uri) / "exports" / "dumped_evals" / "eval_only" / "*.jsonl"))
    summaries = []
    failures = Counter()
    source_counts = Counter()
    identities = Counter()
    unscored = []
    for path in sorted(fs.glob(pattern)):
        if path.endswith("/aggregated_results.jsonl"):
            continue
        rows = [json.loads(line) for line in fs.cat_file(path).decode().splitlines() if line.strip()]
        source_counts.update(row["data_source"] for row in rows)
        if expected_identities is not None:
            identities.update((row["data_source"], str(row["uid"])) for row in rows)
        verified = [row for row in rows if row["verification"]["status"] == "verified"]
        for row in rows:
            if row["verification"]["status"] != "verified":
                failures[row.get("exception_type") or row["verification"]["status"]] += 1
                unscored.append(
                    {
                        "data_source": row["data_source"],
                        "uid": str(row["uid"]),
                        "exception_type": row.get("exception_type"),
                    }
                )
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
    coverage_matches = expected_by_data_source is None or dict(source_counts) == expected_by_data_source
    identities_match = expected_identities is None or identities == Counter(
        (row["data_source"], row["dataset_row_uid"]) for row in expected_identities
    )
    return {
        "identity_coverage_matches": identities_match,
        "result_uri": result_uri,
        "expected_rollouts": expected_rollouts,
        "completed_rollouts": completed,
        "verified_rollouts": sum(row["verified"] for row in summaries),
        "unscored_attempts": unscored,
        "all_verified": completed == expected_rollouts and not failures and coverage_matches and identities_match,
        "attempts_by_data_source": dict(source_counts),
        "expected_by_data_source": expected_by_data_source,
        "source_coverage_matches": coverage_matches,
        "unverified_types": dict(failures),
        "benchmarks": summaries,
    }


def require_evaluation_coverage(evaluation: dict) -> None:
    """Require full scoring or an exact, explicitly accepted set of unscored attempts."""
    if evaluation["all_verified"]:
        return
    unscored = evaluation["unscored_attempts"]
    accepted = evaluation.get("accepted_unscored_attempts", [])
    if (
        evaluation["completed_rollouts"] != evaluation["expected_rollouts"]
        or not evaluation["source_coverage_matches"]
        or not evaluation["identity_coverage_matches"]
        or not accepted
        or sorted(json.dumps(row, sort_keys=True) for row in unscored)
        != sorted(json.dumps(row, sort_keys=True) for row in accepted)
    ):
        raise ValueError("Evaluation has missing, unexpected or unaccepted unscored attempts")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-uri", required=True)
    parser.add_argument("--expected-rollouts", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.result_uri, args.expected_rollouts)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
