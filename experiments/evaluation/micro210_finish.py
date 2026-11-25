# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Preserve completed Micro210 evidence and prepare its policy-coverage and zero-score review."""

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from skyrl_gym.verification import VerificationResult, VerificationStatus
from skyrl_train.trajectory_runners.trajectory_processing import get_trajectory_passes, normalized_verifier_scores
from skyrl_train.utils.harbor_errors import ErrorCategory, error_category

from experiments.evaluation.micro210_policy import POLICY_REFERENCE, PREPARED_COMMIT, PREPARED_REPO, read_micro210_policy

EXPECTED_ROWS = 434
MINIMUM_COVERAGE = 0.9


def summarize_rows(expected: dict[str, str], rows: list[dict]) -> dict:
    """Reconcile trial identities and apply the runtime's verdict and masking semantics."""
    seen = set()
    assigned = Counter(expected.values())
    scores = defaultdict(list)
    passes = Counter()
    unavailable = []
    zero_ledger = []
    for row in rows:
        uid, source = str(row["uid"]), row["data_source"]
        if uid in seen or expected.get(uid) != source:
            raise ValueError(f"Unexpected or duplicate Micro210 trial: {source}/{uid}")
        seen.add(uid)
        verdict = row["verification"]
        verification = (
            VerificationResult(**{**verdict, "status": VerificationStatus(verdict["status"])}) if verdict else None
        )
        batch = {
            "rewards": [row["score"]],
            "verification_results": [verification],
            "exclude_from_baseline": [row["error_treatment"] == "mask"],
        }
        score = normalized_verifier_scores(batch)[0]
        exception = row["exception_type"]
        if (
            score is not None
            and exception
            and exception != "malformed_model_response"
            and error_category(exception) in {ErrorCategory.UNKNOWN, ErrorCategory.INFRASTRUCTURE}
        ):
            raise ValueError(f"Infrastructure error unexpectedly carries a scored verdict: {exception}/{uid}")
        if score is None:
            unavailable.append({"uid": uid, "source": source, "exception_type": row["exception_type"]})
            continue
        scores[source].append(score)
        passes[source] += get_trajectory_passes(batch)[0]
        if score == 0:
            response = row["output_response"]
            zero_ledger.append(
                {
                    "uid": uid,
                    "source": source,
                    "exception_type": row["exception_type"],
                    "verification": verdict,
                    "server_error": row["server_error"],
                    "response_chars": len(response),
                    "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
                }
            )
    unavailable.extend(
        {"uid": uid, "source": expected[uid], "exception_type": "MissingTrial"} for uid in sorted(set(expected) - seen)
    )
    benchmarks = [
        {
            "source": source,
            "assigned": count,
            "scored": len(scores[source]),
            "passed": passes[source],
            "unavailable": count - len(scores[source]),
            "pass_at_1": passes[source] / len(scores[source]) if scores[source] else None,
            "mean_verifier_score": math.fsum(scores[source]) / len(scores[source]) if scores[source] else None,
        }
        for source, count in sorted(assigned.items())
    ]
    scored = sum(len(values) for values in scores.values())
    correct = sum(passes.values())
    return {
        "assigned": len(expected),
        "observed": len(seen),
        "scored": scored,
        "passed": correct,
        "coverage": scored / len(expected),
        "coverage_eligible": scored >= math.ceil(MINIMUM_COVERAGE * len(expected)),
        "pass_at_1": correct / scored if scored else None,
        "mean_verifier_score": (
            math.fsum(value for values in scores.values() for value in values) / scored if scored else None
        ),
        "per_benchmark": benchmarks,
        "unavailable": unavailable,
        "zero_score_ledger": zero_ledger,
        "zero_score_benchmarks": [row["source"] for row in benchmarks if row["pass_at_1"] == 0],
        "seal_status": "AWAITING_AGENT_TRACE_REVIEW",
        "policy": asdict(POLICY_REFERENCE),
    }


def finish(output: str, attempt: str, job_id: str) -> dict:
    """Prepare policy coverage and zero-score evidence in the native verdict environment."""
    configure_coreweave_s3()
    policy = read_micro210_policy(POLICY_REFERENCE)
    prepared = (
        StoragePath(f"hf://datasets/{PREPARED_REPO}@{PREPARED_COMMIT}") / policy["evaluation"]["data_relative_path"]
    )
    raw = prepared.read_bytes()
    if hashlib.sha256(raw).hexdigest() != policy["prepared_rollouts_sha256"]:
        raise ValueError("Micro210 prepared tasks changed after evaluation")
    table = pq.read_table(pa.BufferReader(raw)).to_pylist()
    if len(table) != EXPECTED_ROWS:
        raise ValueError(f"Frozen prepared Micro210 cohort is not {EXPECTED_ROWS} rows")
    expected = {
        str(index): json.loads(row["task_rollout_spec"])["task"]["source"]["dataset"] for index, row in enumerate(table)
    }
    rows = []
    source = StoragePath(output) / "exports" / "dumped_evals" / attempt / "eval_only"
    for path in sorted((source / "*.jsonl").glob(), key=str):
        if path.name != "aggregated_results.jsonl":
            rows.extend(json.loads(line) for line in path.read_bytes().splitlines() if line.strip())
    if not rows:
        raise ValueError(f"No completed Micro210 rows at {source}")
    summary = summarize_rows(expected, rows)
    summary.update(
        job_id=job_id,
        attempt_id=attempt,
        result_source=str(source),
        canonical_evaluation=str(StoragePath(output) / "canonical-evaluation.json"),
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    summary = finish(args.output, args.attempt, args.job_id)
    args.receipt.write_text(json.dumps(summary, sort_keys=True, indent=2))
    print(
        json.dumps(
            {key: summary[key] for key in ("job_id", "scored", "assigned", "coverage", "pass_at_1", "seal_status")}
        )
    )


if __name__ == "__main__":
    main()
