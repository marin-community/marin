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
    *,
    results_uri: str | None = None,
) -> dict:
    results = StoragePath(results_uri or str(StoragePath(result_uri) / "exports/dumped_evals/eval_only"))
    fs, pattern = filesystem_for(str(results / "*.jsonl"))
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
    identity_missing = []
    identity_unexpected = []
    if expected_identities is not None:
        expected_counter = Counter((row["data_source"], row["dataset_row_uid"]) for row in expected_identities)
        identity_missing = sorted(
            (
                {"data_source": source, "dataset_row_uid": uid}
                for (source, uid), count in (expected_counter - identities).items()
                for _ in range(count)
            ),
            key=lambda row: (row["data_source"], row["dataset_row_uid"]),
        )
        identity_unexpected = sorted(
            (
                {"data_source": source, "dataset_row_uid": uid}
                for (source, uid), count in (identities - expected_counter).items()
                for _ in range(count)
            ),
            key=lambda row: (row["data_source"], row["dataset_row_uid"]),
        )
    return {
        "identity_coverage_matches": identities_match,
        "identity_missing": identity_missing,
        "identity_unexpected": identity_unexpected,
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


POLICY_VALID_SEAL_FRACTION = 0.90
POLICY_GRADEABLE_EXCEPTIONS = frozenset(
    {"malformed_model_response", "AgentTimeoutError", "NonZeroAgentExitCodeError", "ContextLengthExceededError"}
)


def require_evaluation_coverage(evaluation: dict) -> None:
    """Require full scoring, an exact accepted unscored set, or a policy-valid partial seal.

    Applies the Micro210 eval scoring policy (#10197): unnamed errors and
    infrastructure-classified exceptions count as absent from the scored pool; grade-able
    exceptions (for example Harbor ``AgentTimeoutError``, which runs the verifier) do not.
    The evaluation seals when at least 90% of expected attempts are validly scored and no
    attempt identity is unexpected. The classification is recorded on the evaluation, never
    fabricated.
    """
    if evaluation["all_verified"]:
        return
    unscored = evaluation["unscored_attempts"]
    accepted = evaluation.get("accepted_unscored_attempts", [])
    if accepted and sorted(json.dumps(row, sort_keys=True) for row in unscored) == sorted(
        json.dumps(row, sort_keys=True) for row in accepted
    ):
        return
    gradeable = sorted(
        json.dumps(row, sort_keys=True) for row in unscored if row.get("exception_type") in POLICY_GRADEABLE_EXCEPTIONS
    )
    missing_attempts = evaluation["expected_rollouts"] - evaluation["completed_rollouts"]
    valid = evaluation["verified_rollouts"]
    expected = evaluation["expected_rollouts"]
    receipt = {
        "policy": "#10197",
        "valid_seal_fraction": POLICY_VALID_SEAL_FRACTION,
        "valid_scored": valid,
        "expected_attempts": expected,
        "valid_fraction": (valid / expected) if expected else 0.0,
        "infrastructure_absent": len(unscored) - len(gradeable) + max(missing_attempts, 0),
        "missing_attempts": max(missing_attempts, 0),
        "gradeable_unscored": gradeable,
        "identity_unexpected": evaluation.get("identity_unexpected", []),
        "identity_missing_count": len(evaluation.get("identity_missing", [])),
    }
    evaluation["unscored_policy"] = receipt
    if (
        gradeable
        or evaluation.get("identity_unexpected")
        or missing_attempts < 0
        or valid < POLICY_VALID_SEAL_FRACTION * expected
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
