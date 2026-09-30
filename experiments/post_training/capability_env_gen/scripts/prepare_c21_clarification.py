#!/usr/bin/env python3
"""Prepare the hash-bound c21 slot-9 verifier-contract clarification input."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from capability_pipeline.inference import atomic_json, digest
from capability_pipeline.synthesis import load_accepted

CAPABILITY_ID = "c21.spreadsheets.calculate"
SLOT = 9
SOURCE_PROPOSAL_HASH = (
    "cbb84ed8e089542d2ead41b282e6c3d32abff30b4c6597cc13a03a3ae446d8a3"
)
AUDIT_PATH = Path("docs/audits/c21_c35_construction_002.md")
AUDIT_SHA256 = "406eb1bbeb924fff4c414210f5ce4bfe1cf6c6e21e8e2a8876c526bfad846b22"

VERIFIER_CONTRACT_CLARIFICATION = """BLOCKING C21 SLOT-9 VERIFIER-CONTRACT CLARIFICATION.
This is a hash-changing clarification of the same realistic workbook regression
task, not permission to simplify it. Preserve the container solver environment,
pinned LibreOffice recalculation path, workbook/data volume, live formula
inspection, all five mutation scenarios, independent Python oracle, clean and
held-out workbooks, formula-diff checks, findings deliverable, eight rubric
criteria and their original weights, binding detection requirement, difficulty,
and public workflow. Do not replace the workbook with a toy, reduce the mutation
set, accept cached values, weaken oracle coverage, remove the held-out variant,
or turn the task into direct formula repair without a reusable harness.

Resolve the construction-discovered cardinality contradiction explicitly. The
planted defect is exactly all 36 formulas in Valuation!D/E/G5:16 (96 truncated
Ledger range references), and the smallest complete repair changes all 36 and
only those 36 formulas from the row-104 boundary to the intended row-504
boundary. Replace every accepted `<= 12 changed cells` statement with this exact
36-cell contract. Minimality passes only when the changed formula-cell set equals
the private 36-cell defect set, no formula is replaced by a value, and no other
cell changes. Preserve the 5% minimality weight. Preserve the accepted easy
row-90 positive control as an actual control rather than omitting it.

Replace lexical scenario-coverage inspection with private execution evidence.
The evaluator must instrument the pinned /opt/recalc/recalc.py boundary in a
private, candidate-unwritable trace location, delegate to the exact real engine,
snapshot/hash every workbook passed to it, and inspect the traced workbooks to
prove the submitted harness actually executes every required scenario on fresh
copies: append at least three ledger rows beyond the original extent; change a
receipt UnitCost for both a moving-average and FIFO item; set a receipt Qty to
zero; create an issue that drives balance negative; and blank a receipt UnitCost.
The evidence must prove each scenario reached real recalculation and the harness
checked its required invariant/flag. Comments, dead strings, imports, function
names, source tokens, or an uncalled subprocess path never earn coverage. The
trace/instrumentation must be inaccessible to the solver and cannot trust a
solver-authored report as proof of execution.

Add the exact combined shortcut control described by the bound audit: a harness
that only fingerprints row-104 aggregate formulas and the held-out static-value
cell, prints a canned pinpointing diagnostic, passes repaired/clean workbooks,
fails buggy/variant workbooks, and includes all mutation/recalc vocabulary in
dead text, while never mutating or recalculating. It must fail execution-backed
scenario coverage and receive at most the binding-failure cap of 0.2. Retain the
existing cached-value, hardcoded/fingerprint, pasted-value, over-broad, missing,
and malformed controls; do not substitute the new control for them.

Keep the original reward weights exactly: detection .25, green-on-fix .15,
no-false-positive .15, held-out generality .15, execution-backed scenario
coverage .10, oracle .10, exact-36 minimality .05, findings .05. Preserve the
tiered held-out generality partial credit and overall continuous reward. Full
detection and full execution-backed scenario coverage remain binding; failure of
either caps reward at 0.2. Missing/malformed candidate deliverables are semantic
graded failures. Missing private fixtures/tools, unavailable recalculation
service, evaluator timeout/crash, or internal checker error are infrastructure
outcomes with null reward, never candidate zero. The repaired proposal must
state these boundaries without dropping or merging criteria.

Fresh review must verify the exact 36-cell repair, all eight weights, all five
executed mutation classes, private trace integrity, every retained control, the
new static-fingerprint/dead-token control, semantic-versus-infrastructure
outcomes, and unchanged task scope/difficulty. Any residual `<=12` claim,
lexical coverage path, omitted control, weakened criterion, or scope reduction
requires repair or rejection rather than acceptance."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(source: Path, audit: Path = AUDIT_PATH) -> tuple[list[dict], dict]:
    if sha256(audit) != AUDIT_SHA256:
        raise ValueError("c21/c35 audit hash differs from the pinned input")
    items = load_accepted(source, allow_revoked=True)
    matches = [
        item
        for item in items
        if item["proposal"]["capability_id"] == CAPABILITY_ID
        and item["proposal"]["slot"] == SLOT
    ]
    if len(matches) != 1:
        raise ValueError("source must contain exactly one c21 slot-9 candidate")
    original = matches[0]
    if original["proposal_hash"] != SOURCE_PROPOSAL_HASH:
        raise ValueError("c21 source proposal hash differs from clarification input")

    candidate = copy.deepcopy(original)
    context = candidate["construction_context"]
    if VERIFIER_CONTRACT_CLARIFICATION not in context["portfolio_issues"]:
        context["portfolio_issues"].append(VERIFIER_CONTRACT_CLARIFICATION)
    prior_admission = copy.deepcopy(candidate.get("admission"))
    context["clarification_lineage"] = {
        "source_proposal_hash": SOURCE_PROPOSAL_HASH,
        "source_admission": prior_admission,
        "construction_audit": {"path": str(audit), "sha256": AUDIT_SHA256},
        "reuse_policy": (
            "Preserve prior construction artifacts only as lineage and repair inputs; "
            "fresh hash-bound construction and validation are mandatory."
        ),
    }
    context["verifier_contract_clarification"] = {
        "defect_cells": "Valuation!D/E/G5:16",
        "defect_cell_count": 36,
        "truncated_reference_count": 96,
        "old_ledger_boundary": 104,
        "intended_ledger_boundary": 504,
        "minimality": "changed_formula_cell_set_equals_private_defect_set",
        "formula_to_value_forbidden": True,
        "outside_change_forbidden": True,
        "scenario_evidence": {
            "kind": "private_recalculation_execution_trace",
            "candidate_writable": False,
            "fresh_copy_required": True,
            "required_scenarios": [
                "append_at_least_three_rows_beyond_original_extent",
                "change_ma_and_fifo_receipt_unit_cost",
                "zero_receipt_quantity",
                "negative_running_balance_issue",
                "blank_receipt_unit_cost",
            ],
            "lexical_or_self_report_evidence_allowed": False,
        },
        "weights": {
            "detection": 0.25,
            "green_on_fix": 0.15,
            "no_false_positive": 0.15,
            "generality": 0.15,
            "scenario_coverage": 0.10,
            "oracle_match": 0.10,
            "minimality": 0.05,
            "findings": 0.05,
        },
        "binding_criteria": ["detection", "scenario_coverage"],
        "binding_failure_reward_cap": 0.2,
        "semantic_candidate_failure": "graded_zero_or_rubric_reward",
        "private_or_checker_infrastructure_failure": "null_infra_error",
        "required_new_control": "static_formula_fingerprint_with_dead_coverage_tokens",
        "preserve_easy_row90_positive_control": True,
    }
    context["source_files"]["c21_c35_construction_002.md"] = AUDIT_SHA256
    candidate["admission"] = {
        "state": "pending",
        "scope": "individual_construction",
        "clarification_reason": (
            "36-cell minimal repair and execution-backed mutation verification"
        ),
        "source_proposal_hash": SOURCE_PROPOSAL_HASH,
    }

    result = [candidate]
    manifest = {
        "schema_version": "c21-slot9-verifier-clarification-input-v1",
        "source_path": str(source),
        "source_sha256": sha256(source),
        "source_proposal_hash": SOURCE_PROPOSAL_HASH,
        "construction_audit_path": str(audit),
        "construction_audit_sha256": AUDIT_SHA256,
        "clarification_sha256": hashlib.sha256(
            VERIFIER_CONTRACT_CLARIFICATION.encode()
        ).hexdigest(),
        "candidate_digest": digest(result),
        "prior_admission_rounds": len(prior_admission.get("history", [])),
        "inference_performed": False,
        "next_stage": (
            "fresh construction admission review, one GLM repair, fresh independent "
            "review; no construction or acceptance claim"
        ),
    }
    return result, manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, default=AUDIT_PATH)
    args = parser.parse_args(argv)
    result, manifest = prepare(args.source, args.audit)
    atomic_json(args.output, result)
    atomic_json(args.output.with_suffix(".manifest.json"), manifest)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
