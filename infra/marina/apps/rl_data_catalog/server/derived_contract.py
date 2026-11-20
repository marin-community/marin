# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Expose grader profiles only for an exact current TaskSpec census input."""

from collections import defaultdict
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Connection

from .contract_selection import INPUT_FIELDS, select_contract_receipts
from .source_identity import canonical_source_id

PROFILE_FIELDS = (
    "grader_type",
    "grader_type_counts",
    "judge_rubric_counts",
    "judge_reference_count",
    "judge_no_reference_count",
    "derived_agentic",
    "derived_turns",
    "derived_reply_only",
    "derived_workspace_graded",
    "derived_contract_receipt",
)


def contract_records(connection: Connection) -> dict[str, list[dict]]:
    records: dict[str, list[dict]] = defaultdict(list)
    for source_id, receipt in connection.execute(
        text("SELECT source_id,receipt FROM atlas_derived_contracts ORDER BY receipt_id")
    ).tuples():
        records[canonical_source_id(source_id)].append(receipt)
    return records


def contract_projection(declaration: dict[str, Any], receipts: list[dict]) -> dict[str, Any]:
    """Keep stale or unmeasured populations pending instead of inheriting declared tags."""
    binding = declaration.get("actor_census_input")
    matches = [
        row
        for row in receipts
        if binding
        and canonical_source_id(row["source_id"]) == declaration["id"]
        and all(row["input"][key] == binding[key] for key in INPUT_FIELDS)
    ]
    fields = dict(
        grader_type="",
        grader_type_counts={},
        judge_rubric_counts={},
        judge_reference_count=0,
        judge_no_reference_count=0,
        derived_agentic=None,
        derived_turns="",
        derived_reply_only=None,
        derived_workspace_graded=None,
    )
    if not matches:
        # Distinguish "not yet computed for this pin" from a genuine pending state: a receipt
        # measured at an older pin is the drift the re-pin flow must repair, not an unmeasured
        # source. The UI renders the last-computed pin beside the Pending chip.
        stale = [row for row in receipts if canonical_source_id(row["source_id"]) == declaration["id"]]
        if stale:
            last = sorted(stale, key=lambda row: row.get("receipt_id") or "")[-1]
            fields["derived_contract_pending_reason"] = {
                "state": "stale-pin",
                "last_computed_dataset_revision": last["input"].get("dataset_revision"),
            }
        result = {**declaration, **fields}
        result.pop("derived_contract_receipt", None)
        return result
    matches = select_contract_receipts(declaration["id"], matches)
    receipt = matches[0]
    summary = receipt["summary"]
    count = summary["task_count"]
    types = summary["grader_type_counts"]
    if not count or sum(types.values()) != count:
        raise ValueError(f"Derived grader counts do not partition the population: {declaration['id']}")
    fields.update(
        grader_type=next(iter(types)) if len(types) == 1 else "mixed",
        grader_type_counts=types,
        judge_rubric_counts=summary["judge_rubric_counts"],
        judge_reference_count=summary["judge_reference_count"],
        judge_no_reference_count=summary["judge_no_reference_count"],
        derived_agentic=summary["agentic_count"] > 0,
        derived_turns=(
            "single-turn"
            if summary["single_turn_count"] == count
            else "multi-turn" if summary["single_turn_count"] == 0 else "mixed"
        ),
        derived_reply_only=summary["reply_only_count"] == count,
        derived_workspace_graded=summary["workspace_graded_count"] == count,
        derived_contract_receipt={
            "uri": receipt["evidence_url"],
            "input": {key: receipt["input"][key] for key in INPUT_FIELDS},
            "scanned_task_specs_sha256": receipt["scanned_task_specs_sha256"],
        },
    )
    return {**declaration, **fields}
