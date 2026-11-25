# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit fixed evaluation cohorts with SkyRL's exact FTPO tail-loop detector."""

import argparse
import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from itertools import chain
from typing import Any

import numpy as np
from rigging.filesystem.storage_path import StoragePath
from rigging.filesystem.s3_compat import configure_coreweave_s3
from skyrl_train.config.reward_shaping import LoopCreditConfig
from skyrl_train.trajectory_runners.trajectory_reward_shaping import tail_loop
from verifyit.tool_call_markup import THINK_CLOSE, THINK_OPEN
from skyrlrecipe.cohort_identity import cohort_hash, prompt_hash

INITIAL_PANEL_ROWS = 128
EXTENDED_PANEL_ROWS = 256
MINIMUM_TARGET_LOOPS = 10


@dataclass(frozen=True)
class Incidence:
    """Exact loop counts, including UB, the confirmed-plus-unassessed upper bound."""
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
        unavailable += int(not grader_error and (status != "verified" or type(verification.get("passed")) is not bool))
        passes += int(not damaged and not grader_error and status == "verified" and verification.get("passed") is True)
        if valid_tokens:
            lengths.append(len(tokens))
        at_cap = row.get("stop_reason") in ("length", "max_tokens", "max_length") or (
            valid_tokens and len(tokens) >= cohort["budget"]
        )
        length_stop += int(at_cap)
        response = (row.get("input_prompt") or "") + (row.get("output_response") or "")
        open_think += int(at_cap and response.rfind(THINK_OPEN) > response.rfind(THINK_CLOSE))
    quantiles = np.percentile(lengths, [50, 90, 99]).tolist() if lengths else [None] * 3
    return Incidence(
        len(expected), confirmed, unassessed, confirmed + unassessed, passes, unavailable, errors,
        length_stop, open_think, *quantiles,
    )


@dataclass(frozen=True)
class DomainAudit:
    incidence: Incidence
    cohort_hash: str
    target_status: str


def audit_domain(
    name: str, cohort: Mapping[str, Any], initial_rows: list[dict[str, Any]],
    reserve_rows: Iterable[dict[str, Any]] | None = None,
) -> DomainAudit:
    """Apply the fixed-reserve rule before deciding whether a 32K domain is a target."""
    selected = cohort
    result = audit_cohort(selected, initial_rows)
    status = "target"
    if name != "math16k" and result.confirmed < MINIMUM_TARGET_LOOPS:
        if result.N != INITIAL_PANEL_ROWS or len(cohort["reserve"]) != EXTENDED_PANEL_ROWS - INITIAL_PANEL_ROWS:
            raise ValueError(f"{name} must have the fixed initial panel and complete reserve")
        status = "extend"
        if reserve_rows is not None:
            selected = {**cohort, "rows": [*cohort["rows"], *cohort["reserve"]]}
            result = audit_cohort(selected, chain(initial_rows, reserve_rows))
            status = "target" if result.confirmed >= MINIMUM_TARGET_LOOPS else "non-target"
    elif reserve_rows is not None:
        raise ValueError(f"{name} does not qualify for extension")
    digest = cohort_hash(row["prompt_hash"] for row in selected["rows"])
    return DomainAudit(result, digest, status)


def audit_candidate(cohort: Mapping[str, Any], rows: Iterable[dict[str, Any]], baseline: DomainAudit) -> DomainAudit:
    """Keep the sealed baseline's prompt set and target decision for every candidate."""
    if baseline.target_status == "extend":
        raise ValueError("candidate evaluation requires a sealed baseline")
    selected = cohort
    if baseline.incidence.N == len(cohort["rows"]) + len(cohort["reserve"]):
        selected = {**cohort, "rows": [*cohort["rows"], *cohort["reserve"]]}
    digest = cohort_hash(row["prompt_hash"] for row in selected["rows"])
    if digest != baseline.cohort_hash:
        raise ValueError("candidate prompt hashes differ from the sealed baseline")
    return DomainAudit(audit_cohort(selected, rows), digest, baseline.target_status)


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
    parser.add_argument("--reserve-input", action="append", default=[], help="COHORT=JSONL_GLOB; fixed reserve session")
    parser.add_argument("--baseline", help="Sealed baseline audit JSON; fixes each candidate's cohort and target status")
    parser.add_argument("--output", required=True, help="JSON output URI; Markdown is written beside it")
    args = parser.parse_args()
    configure_coreweave_s3()
    manifest_payload = StoragePath(args.manifest).read_bytes()
    manifest = json.loads(manifest_payload)
    baseline = json.loads(StoragePath(args.baseline).read_text()) if args.baseline else None
    results = {}
    hashes = {}
    statuses = {}
    reserves = dict(item.split("=", 1) for item in args.reserve_input)
    reserve_sources = reserves.copy()
    input_sources = {}
    cached_rows = {}
    for item in args.input:
        name, pattern = item.split("=", 1)
        if name in results:
            raise ValueError(f"duplicate input for {name}")
        input_sources[name] = pattern
        if pattern not in cached_rows:
            cached_rows[pattern] = list(read_rows(pattern))
        reserve_rows = read_rows(reserves.pop(name)) if name in reserves else None
        if baseline is None:
            audited = audit_domain(name, manifest["cohorts"][name], cached_rows[pattern], reserve_rows)
        else:
            reference = DomainAudit(
                Incidence(**baseline["domains"][name]), baseline["cohort_hashes"][name], baseline["target_status"][name]
            )
            audited = audit_candidate(
                manifest["cohorts"][name], chain(cached_rows[pattern], reserve_rows or ()), reference
            )
        results[name] = asdict(audited.incidence)
        hashes[name] = audited.cohort_hash
        statuses[name] = audited.target_status
    if reserves:
        raise ValueError(f"reserve inputs without initial inputs: {sorted(reserves)}")
    document = {
        "manifest": args.manifest, "manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
        "baseline": args.baseline, "inputs": input_sources, "reserve_inputs": reserve_sources,
        "cohort_hashes": hashes, "domains": results, "target_status": statuses,
    }
    StoragePath(args.output).write_text(json.dumps(document, indent=2) + "\n")
    columns = ("N", "confirmed", "unassessed", "UB", "passes", "unavailable", "errors", "length_stop", "open_think")
    lines = ["| Cohort | " + " | ".join(columns) + " |", "| --- | " + " | ".join(["---:"] * len(columns)) + " |"]
    lines.extend("| " + name + " | " + " | ".join(str(row[key]) for key in columns) + " |" for name, row in results.items())
    markdown = "UB is the loop upper bound: confirmed + unassessed.\n\n" + "\n".join(lines) + "\n\n"
    markdown += "Target status: " + ", ".join(f"{name}={status}" for name, status in statuses.items()) + ".\n"
    StoragePath(args.output.removesuffix(".json") + ".md").write_text(markdown)
    print(markdown)


if __name__ == "__main__":
    main()
