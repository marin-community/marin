# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit fixed evaluation cohorts with SkyRL's exact FTPO tail-loop detector."""

import argparse
import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from rigging.filesystem.storage_path import StoragePath
from rigging.filesystem.s3_compat import configure_coreweave_s3
from skyrl_train.config.reward_shaping import LoopCreditConfig
from skyrl_train.trajectory_runners.trajectory_reward_shaping import tail_loop


@dataclass(frozen=True)
class Incidence:
    N: int
    confirmed: int
    unassessed: int
    UB: int
    passes: int
    unavailable: int
    errors: int
    length_stop: int
    open_think: int
    length_p50: float | None
    length_p90: float | None
    length_p99: float | None


def prompt_hash(token_ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(token_ids, separators=(",", ":")).encode()).hexdigest()


def audit_cohort(cohort: Mapping[str, Any], rows: Iterable[dict[str, Any]]) -> Incidence:
    """Count missing or damaged generations in the loop upper bound, never as passes."""
    expected = {row["task_id"]: row["prompt_hash"] for row in cohort["rows"]}
    if len(expected) != len(cohort["rows"]):
        raise ValueError("cohort contains duplicate task IDs")
    indexed = {}
    for row in rows:
        task_id = row.get("task_id")
        if task_id not in expected:
            continue
        if task_id in indexed:
            raise ValueError(f"multiple generations for cohort task {task_id}; select one evaluation session")
        indexed[task_id] = row
    config = LoopCreditConfig(
        max_period_tokens=cohort["max_period_tokens"], tail_tokens=cohort["budget"], minimum_occurrences=8
    )
    confirmed = unassessed = passes = unavailable = errors = length_stop = open_think = 0
    lengths = []
    for task_id, expected_hash in expected.items():
        row = indexed.get(task_id)
        if row is None:
            unassessed += 1
            unavailable += 1
            continue
        tokens = row.get("response_token_ids")
        prompt = row.get("prompt_token_ids")
        valid_tokens = isinstance(tokens, list) and bool(tokens) and all(type(token) is int for token in tokens)
        valid_prompt = isinstance(prompt, list) and bool(prompt) and prompt_hash(prompt) == expected_hash
        damaged = (
            not valid_tokens
            or not valid_prompt
            or bool(row.get("server_error"))
            or row.get("error_treatment") not in (None, "none")
            or bool(row.get("exception_type"))
            or row.get("is_last_step", True) is not True
        )
        if damaged:
            unassessed += 1
        elif tail_loop(tokens, [1] * len(tokens), config) is not None:
            confirmed += 1
        verification = row.get("verification") or {}
        status = verification.get("status")
        grader_error = status == "error" or bool(row.get("exception_type")) or bool(row.get("server_error"))
        errors += int(grader_error)
        unavailable += int(not grader_error and status != "verified")
        passes += int(not damaged and not grader_error and verification.get("passed") is True)
        if valid_tokens:
            lengths.append(len(tokens))
        at_cap = row.get("stop_reason") in ("length", "max_tokens", "max_length") or (
            valid_tokens and len(tokens) >= cohort["budget"]
        )
        length_stop += int(at_cap)
        response = row.get("output_response") or ""
        open_think += int(at_cap and response.rfind("<|start_think|>") > response.rfind("<|end_think|>"))
    quantiles = np.percentile(lengths, [50, 90, 99]).tolist() if lengths else [None] * 3
    return Incidence(
        len(expected), confirmed, unassessed, confirmed + unassessed, passes, unavailable, errors,
        length_stop, open_think, *quantiles,
    )


def read_rows(pattern: str) -> Iterable[dict[str, Any]]:
    for path in sorted(StoragePath(pattern).glob(), key=str):
        if path.name == "aggregated_results.jsonl":
            continue
        with path.open("r") as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--input", action="append", required=True, help="COHORT=JSONL_GLOB; one saved session per cohort")
    parser.add_argument("--output", required=True, help="JSON output URI; Markdown is written beside it")
    args = parser.parse_args()
    configure_coreweave_s3()
    manifest = json.loads(StoragePath(args.manifest).read_text())
    results = {}
    for item in args.input:
        name, pattern = item.split("=", 1)
        if name in results:
            raise ValueError(f"duplicate input for {name}")
        results[name] = asdict(audit_cohort(manifest["cohorts"][name], read_rows(pattern)))
    document = {"manifest": args.manifest, "cohort_hashes": {
        name: manifest["cohorts"][name]["cohort_hash"] for name in results
    }, "domains": results}
    StoragePath(args.output).write_text(json.dumps(document, indent=2) + "\n")
    columns = ("N", "confirmed", "unassessed", "UB", "passes", "unavailable", "errors", "length_stop", "open_think")
    lines = ["| Cohort | " + " | ".join(columns) + " |", "| --- | " + " | ".join(["---:"] * len(columns)) + " |"]
    lines.extend("| " + name + " | " + " | ".join(str(row[key]) for key in columns) + " |" for name, row in results.items())
    markdown = "\n".join(lines) + "\n"
    StoragePath(args.output.removesuffix(".json") + ".md").write_text(markdown)
    print(markdown)


if __name__ == "__main__":
    main()
