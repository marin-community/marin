#!/usr/bin/env python3
"""Prepare a hash-bound c30 candidate for fresh supported-verifier repair."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from capability_pipeline.inference import atomic_json, digest
from capability_pipeline.synthesis import load_accepted

CAPABILITY_ID = "c30.growth.performance_reporting"
SLOT = 5
SOURCE_PROPOSAL_HASH = (
    "e7a143b517fe913295831e736a2fcd99cd7874b9d47234a7b5bf1d1cae3b50df"
)
FIRST_REDESIGN_HASH = "6f9d15b8a72dc6fd53ab731990457edcc087a4ef5e27eadc2c262a63f56ab2f6"

SUPPORTED_CONTRACT = """BLOCKING SUPPORTED-VERIFIER REDESIGN FOR THIS CANDIDATE.
The exact pinned TaskCompendium dc6b501c8604bcd2e3c20c1e9947679845fdfef8 native
TaskTrove Mode.JUDGE supports one reference comparison scored 0/0.5/1 or an
equal-weight checklist of binary criteria whose mean is the reward. Its only
pre-judge gates are the source verifier's generic constraints and optional exact
match. It cannot run a task-authored deterministic precheck and cannot implement
custom weights, negative penalties, critical/must-pass criteria, disqualifier
aggregation, or a custom numeric pass threshold. Pinned Harbor rejects multi-step
all_required_steps; mean dilutes a failed executable gate and final discards it.

Repair this exact slot to a faithful SINGLE-STEP native equal-weight checklist
contract, or return null if that would weaken the capability. Preserve the current
reasoning workflow, synthetic memo/tables/glossaries/action menu, all arithmetic and
definition-drift depth, open-ended corrected memo and findings register, realism,
difficulty, source capability alignment, and source/artifact honesty. Do not replace
the task with a simpler exact-answer example. Remove claims of a separate executable
precheck, weighted 0/1/2 rubric, penalties, critical gates, disqualifiers, consensus
harness, or binary pass/fail unless they are exactly expressible by the native
surface. Define the task reward as the mean of independently meaningful binary
checklist criteria. Split multi-level anchors into distinct nonduplicative binary
criteria only where each tests a real semantic requirement. The private defect key
and task materials may be judge reference context; they remain hidden from the
solver. Word count, quote fidelity, arithmetic, causal language, attribution-window
mismatch, definition mismatch, bounded actions, false positives, and instruction
injection must be assessed through explicit native checklist criteria rather than a
fictional machine stage. A gold answer plus inert injected text is an injection-
invariance positive; an injection-only non-answer is the required negative control.

The builder plan must construct and independently double-check the synthetic source
artifacts, freeze the exact checklist and native JudgeConfig, and later create at
least 40 positive and 40 negative distinct semantic variant groups that actually
reach the model path. Calibration repeats do not increase effective sample size and
cannot repair an unfaithful reward representation. The fresh reviewer must reject
any redesign that silently drops task depth, retains unsupported verifier claims, or
cannot make strong plausible errors receive appropriately low mean reward."""

CRITICAL_MEAN_REPAIR = """BLOCKING REWARD-VALIDITY REPAIR AFTER THE FIRST NATIVE
REDESIGN. Preserve the repaired proposal's full 15-defect memo workflow, exact
arithmetic, source artifacts, realism, and difficulty, and retain a single-step
native equal-weight binary-checklist mean. Do not reinstate an executable precheck,
custom aggregation, weights, penalties, or an unsupported pass threshold.

The first redesign's reward math is invalid because it removed the original four
severe-error gates. With 39 independent criteria, its own out-of-menu gold control
fails only M18/M20/M21 and would receive 36/39 = 0.923; a gold answer that computes
one blended cross-window ROAS would receive roughly 37-38/39 = 0.949-0.974; the
numbers-fixed but causal/action-kept control would receive roughly 36/39 = 0.923;
and an over-flagger failing two criteria would receive 37/39 = 0.949. Calling all
of these negative controls is incompatible with the native mean and with the
required <=0.2 negative calibration ranges. The prior fresh accepting review
missed this arithmetic; do not inherit it.

Express the original severe semantics natively by making EVERY binary checklist
criterion explicitly conjunctive. It scores 1 if and only if both (A) the global
severe-error predicate passes and (B) that criterion's local semantic requirement
passes. The global predicate passes only when the submission: (1) contains no
causal, incrementality, or lift claim and explicitly treats the effect as
unmeasured; (2) identifies and corrects EMEA net ROAS to 2.29 with refunds and FX
1.10 and calls the 3.0 target a miss; (3) refuses to compute a blended ROAS across
the incompatible email and paid-media windows/definitions; and (4) makes only
action-menu recommendations within limits, with no experiment design or >10%
reallocation. Therefore any one severe error makes all criteria 0 and native mean
0 without custom aggregation. State this complete conjunction in the native
criterion construction and builder plan, not as a post-hoc pass rule.

Retain independent local criteria for all 15 defects and output quality. Calculate
and state exact expected native means for every named control. Severe-error, empty,
and injection-only controls must be 0 (and within [0,0.2]); gold and at least 40
distinct acceptable semantic groups must be >=0.8; at least 40 distinct
unacceptable severe-error groups must be <=0.2. Noncritical errors such as limited
over-flagging or one noncritical arithmetic omission are PARTIAL controls with
explicit intermediate expected_reward_range values, not negatives that pretend to
meet <=0.2. Repeats do not increase group count. A fresh reviewer must verify the
mean arithmetic, preservation of every seeded defect, and native expressibility;
return null or reject if the conjunction makes the judge contract unclear or
unreliable."""


def prepare(source: Path, repair: str = "initial") -> tuple[list[dict], dict]:
    items = load_accepted(source, allow_revoked=repair == "critical-mean")
    matches = [
        item
        for item in items
        if item["proposal"]["capability_id"] == CAPABILITY_ID
        and item["proposal"]["slot"] == SLOT
    ]
    if len(matches) != 1:
        raise ValueError("source must contain exactly one c30 slot-5 candidate")
    original = matches[0]
    expected_hash = SOURCE_PROPOSAL_HASH if repair == "initial" else FIRST_REDESIGN_HASH
    finding = SUPPORTED_CONTRACT if repair == "initial" else CRITICAL_MEAN_REPAIR
    if original["proposal_hash"] != expected_hash:
        raise ValueError(
            "c30 source proposal hash differs from the selected repair input"
        )
    candidate = copy.deepcopy(original)
    issues = candidate["construction_context"]["portfolio_issues"]
    if finding not in issues:
        issues.append(finding)
    candidate["admission"] = {
        "state": "pending",
        "scope": "individual_construction",
        "redesign_reason": (
            "exact pinned native-judge contract incompatibility"
            if repair == "initial"
            else "native checklist mean failed severe-error reward arithmetic"
        ),
    }
    candidate["construction_context"]["redesign_contract"] = {
        "taskcompendium_revision": ("dc6b501c8604bcd2e3c20c1e9947679845fdfef8"),
        "source_proposal_hash": expected_hash,
        "constraint_sha256": hashlib.sha256(finding.encode()).hexdigest(),
        "required_environment": "reasoning",
        "required_verification": "judge",
        "required_native_shape": "single-step equal-weight binary checklist mean",
    }
    result = [candidate]
    manifest = {
        "schema_version": f"c30-supported-redesign-{repair}-input-v1",
        "source_path": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_proposal_hash": expected_hash,
        "constraint_sha256": hashlib.sha256(finding.encode()).hexdigest(),
        "candidate_digest": digest(result),
        "inference_performed": False,
        "next_stage": "fresh construction admission review, one repair, fresh review",
    }
    return result, manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("runs/construction-admission-001/accepted.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--repair",
        choices=("initial", "critical-mean"),
        default="initial",
    )
    args = parser.parse_args(argv)
    result, manifest = prepare(args.source, args.repair)
    atomic_json(args.output, result)
    atomic_json(args.output.with_suffix(".manifest.json"), manifest)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
