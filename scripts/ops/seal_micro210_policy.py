# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Seal the Micro210 spec policy per the 2026-11-24 user directive.

The in-region gate (job a8) exposed three task-environment defects. Rather than building
replacement task images, the directive drops the defective terminal-bench and ot-tblite
slots with no replacement and swaps the broken HumanEval/130 for the deterministic pool
order's first image-free same-family candidate (HumanEval/51). This script applies those
changes to the a8-gated prepared table, verifies every kept row's TaskSpec is
byte-identical to what a8 graded, binds the a8 receipt to the sealed table (plus a fresh
receipt for the swapped row), and records every removal and swap with its reason in
task-swaps.json.
"""

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from experiments.evaluation.micro210_gate import expectations
from experiments.evaluation.micro210_replacements import humanevalplus_candidate

# The a8-gated prepared table digest this seal binds to.
PREPARED_DIGEST = "17652701330da48d58924cd6639a8899225d374fc5d048c6e4771220e76e7312"

DROPPED_TASKS = {
    "tb2-recovery/terminal-bench/mcmc-sampling-stan": (
        "task environment defect (a8 gate): the harbor environment cannot install its RStan "
        "dependencies (StanHeaders/RcppParallel source builds fail), so oracle and NoP never "
        "complete; dropped with no replacement — the 2026-11-24 user directive seals Micro210 "
        "without task-image builds"
    ),
    "ot-tblite-recovery/pdf-table-parsing": (
        "task environment defect (a8 gate): the harbor environment lacks the camelot module at "
        "verifier time; every same-family replacement row requires building a task image, which "
        "the directive excludes, so the slot is dropped with no replacement (2026-11-24 user "
        "directive)"
    ),
}

SWAPPED_TASK = "humanevalplus/HumanEval/130"
SWAP_SOURCE = "HumanEval/51"
SWAP_REASON = (
    "task environment defect (a8 gate): the row's oracle answer scores 0.0 in the spec-only "
    "gate; replaced with the deterministic seed-20261124 pool order's first image-free "
    "same-family candidate (no image build required)"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row_key(row: dict) -> tuple[str, int]:
    return (row["task_id"], row["repetition_index"])


def load_and_bind(prepared_table: Path, a8_receipt: Path) -> tuple[pa.Table, str, dict]:
    """Load the a8-gated table and verify the receipt binds to exactly these bytes."""
    digest = sha256_file(prepared_table)
    receipt = json.loads(a8_receipt.read_text())
    if digest != PREPARED_DIGEST:
        raise ValueError(f"Prepared table digest {digest} does not match the pinned a8 table {PREPARED_DIGEST}")
    if receipt["prepared_rollouts_sha256"] != digest:
        raise ValueError("The a8 receipt does not bind to this prepared table")
    return pq.read_table(prepared_table), digest, receipt


def apply_directive(table: pa.Table) -> tuple[pa.Table, dict, list[dict]]:
    """Drop the directive's slots, swap the broken HumanEval row, return the sealed table."""
    rows = table.to_pylist()
    swaps = {"dropped": [], "swapped": []}
    kept = []
    for benchmark, reason in DROPPED_TASKS.items():
        dropped = [row for row in rows if row["task_id"] == benchmark]
        if not dropped:
            raise ValueError(f"The directive's drop target {benchmark} is not in the prepared table")
        swaps["dropped"].append(
            {
                "task_id": benchmark,
                "repetitions": len(dropped),
                "reason": reason,
                "replacement": None,
            }
        )
    template = next(row for row in rows if row["task_id"] == SWAPPED_TASK)
    broken_rows = [row for row in rows if row["task_id"] == SWAPPED_TASK]
    swaps["swapped"].append(
        {
            "task_id": SWAPPED_TASK,
            "repetitions": len(broken_rows),
            "reason": SWAP_REASON,
            "replacement": {"task_id": None, "source_task_id": SWAP_SOURCE},
        }
    )

    candidate = humanevalplus_candidate(SWAP_SOURCE, template)
    rollout = json.loads(candidate.task_rollout_spec)
    rollout["task"] = json.loads(candidate.task_spec)
    rollout_spec = json.dumps(rollout, sort_keys=True)
    swaps["swapped"][0]["replacement"]["task_id"] = candidate.task_id

    replaced_ids = {SWAPPED_TASK, *DROPPED_TASKS}
    for row in rows:
        if row["task_id"] not in replaced_ids:
            kept.append(row)
            continue
        if row["task_id"] != SWAPPED_TASK:
            continue
        kept.append(
            {
                **row,
                "task_id": candidate.task_id,
                "source_task_id": candidate.source_task_id,
                "task_spec": candidate.task_spec,
                "task_rollout_spec": rollout_spec,
            }
        )
    if any(row["task_id"] == SWAPPED_TASK for row in kept):
        raise ValueError("The broken HumanEval row survived the seal")
    if sum(1 for row in kept if row["task_id"] == candidate.task_id) != len(broken_rows):
        raise ValueError("The swap did not preserve the broken task's repetition count")
    return pa.Table.from_pylist(kept, schema=table.schema), swaps, kept


def binding_report(prepared_rows: list[dict], sealed_rows: list[dict]) -> dict:
    """Verify every kept row's TaskSpec is byte-identical to what a8 graded."""
    prepared_specs = {_row_key(row): row["task_spec"] for row in prepared_rows}
    unchanged = 0
    for row in sealed_rows:
        key = _row_key(row)
        if key not in prepared_specs:
            continue
        if prepared_specs[key] != row["task_spec"]:
            raise ValueError(f"Kept row {key} drifted from the a8-gated TaskSpec")
        unchanged += 1
    return {
        "kept_rows_with_unchanged_spec": unchanged,
        "a8_rows": len(prepared_rows),
        "sealed_rows": len(sealed_rows),
    }


def merge_receipts(a8_receipt: dict, replacement_receipt: dict, sealed_rows: list[dict], sealed_digest: str) -> dict:
    """Bind a8's per-row results (plus the fresh replacement results) to the sealed table."""
    replaced_ids = {SWAPPED_TASK, *DROPPED_TASKS}
    stale_failures = [item for item in a8_receipt["failures"] if item["task_id"] not in replaced_ids]
    if stale_failures:
        raise ValueError(f"a8 failures outside the directive's scope: {stale_failures[:3]}")

    replacement_results = {}
    for item in replacement_receipt["rows"]:
        key = (item["task_id"], item["repetition_index"])
        replacement_results[key] = item
        if item.get("failed"):
            raise ValueError(f"The replacement row failed its gate: {json.dumps(item)[:400]}")
        oracle_minimum, nop_maximum = expectations("humanevalplus")
        if item["oracle_reward"] < oracle_minimum or item["nop_reward"] > (nop_maximum or 0.0):
            raise ValueError(f"The replacement row is not eligible: {json.dumps(item)[:400]}")
    if replacement_receipt.get("failures"):
        raise ValueError("The replacement gate receipt carries failures")
    if replacement_receipt["grader_library_manifest_sha256"] != a8_receipt["grader_library_manifest_sha256"]:
        raise ValueError("The replacement gate ran with a different grader library than a8")

    results = []
    a8_rows = {(item["task_id"], item["repetition_index"]): item for item in a8_receipt["rows"]}
    for row in sealed_rows:
        key = _row_key(row)
        if key in replacement_results:
            results.append(replacement_results[key])
            continue
        if key not in a8_rows:
            raise ValueError(f"Sealed row {key} has neither an a8 result nor a fresh gate result")
        results.append(a8_rows[key])

    summary = (
        f"{sum(1 for item in results if not item.get('failed'))} graded rows, "
        f"0 failures, {a8_receipt['deferred_harbor_rows']} harbor rows deferred"
    )
    return {
        "prepared_rollouts_sha256": sealed_digest,
        "grader_library_manifest_sha256": a8_receipt["grader_library_manifest_sha256"],
        "dependencies": a8_receipt["dependencies"],
        "rows": results,
        "failures": [],
        "deferred_harbor_rows": a8_receipt["deferred_harbor_rows"],
        "summary": summary,
        "pass": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-table", type=Path, required=True, help="The a8-gated prepared table")
    parser.add_argument("--a8-receipt", type=Path, required=True, help="a8's green-oracle gate receipt")
    parser.add_argument(
        "--replacement-receipt",
        type=Path,
        help="The in-region gate receipt for the swapped row; without it only the table is sealed",
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    table, prepared_digest, a8_receipt = load_and_bind(arguments.prepared_table, arguments.a8_receipt)
    prepared_rows = table.to_pylist()
    sealed_table, swaps, sealed_rows = apply_directive(table)
    bindings = binding_report(prepared_rows, sealed_rows)
    if len(sealed_rows) != len(prepared_rows) - 6:
        raise ValueError(f"Sealed row count {len(sealed_rows)} does not match the directive arithmetic")

    arguments.output.mkdir(parents=True, exist_ok=True)
    sealed_path = arguments.output / "rollouts.parquet"
    pq.write_table(sealed_table, sealed_path)
    sealed_digest = sha256_file(sealed_path)
    swaps["prepared_rollouts_sha256"] = prepared_digest
    swaps["sealed_rollouts_sha256"] = sealed_digest
    swaps.update(bindings)
    (arguments.output / "task-swaps.json").write_text(json.dumps(swaps, indent=2))
    swapped_rows = [row for row in sealed_rows if row["task_id"] == swaps["swapped"][0]["replacement"]["task_id"]]
    pq.write_table(
        pa.Table.from_pylist(swapped_rows, schema=table.schema), arguments.output / "gate-replacement-rows.parquet"
    )

    print(f"sealed table: {sealed_digest} ({len(sealed_rows)} rows, a8 had {len(prepared_rows)})")
    print(f"binding: {json.dumps(bindings)}")
    for entry in swaps["dropped"]:
        print(f"drop: {entry['task_id']} ({entry['repetitions']} rows)")
    for entry in swaps["swapped"]:
        print(f"swap: {entry['task_id']} -> {entry['replacement']['task_id']}")

    if arguments.replacement_receipt is None:
        print("receipt: not merged (gate the swapped row in-region, then re-run with --replacement-receipt)")
        return
    replacement_receipt = json.loads(arguments.replacement_receipt.read_text())
    receipt = merge_receipts(a8_receipt, replacement_receipt, sealed_rows, sealed_digest)
    receipt_path = arguments.output / "gate-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2))
    print(f"receipt: {receipt_path} — {receipt['summary']}")


if __name__ == "__main__":
    main()
