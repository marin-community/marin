#!/usr/bin/env python3
"""Prepare the hash-bound c02 slot-10 score-contract clarification input."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from capability_pipeline.inference import atomic_json, digest
from capability_pipeline.synthesis import load_accepted

CAPABILITY_ID = "c02.flaky_test_repair"
SLOT = 10
SOURCE_PROPOSAL_HASH = (
    "d5aa409874d71bb4ea4f46bcc7627020ccc8d09948524b2255b22cd29d36c683"
)
AUDIT_PATH = Path("docs/audits/c02_construction_001.md")
AUDIT_SHA256 = "3c1761ebad7fa254ff5806309f0baf953a7f734f819e23bfa1fc33164740e212"

SCORE_CONTRACT_CLARIFICATION = """BLOCKING C02 SLOT-10 SCORE-CONTRACT CLARIFICATION.
This is a reward-contract clarification of the same difficult task, not permission
to simplify or redesign it. Preserve the original container solver environment,
mixed four-mechanism legacy-suite workflow, product-source freeze, task scope,
deliverables, controlled experiments, 120-run evaluator campaign, all machine
gates and probes, machine-generated judge context, partial-credit behavior, and
positive/negative controls. Composition of the private executable prepass with
the judge is authorized and required. Do not convert this to a native-only judge,
drop a criterion, merge anchors, make criteria binary, discard partial credit, or
replace the realistic task with a smaller example.

The rubric has exactly thirteen scored items, each anchored 0-3: four mechanism-
attribution items, one isolating-experiment item, four repair-quality items, one
assertion-preservation item, one campaign-design/greenness item, one internal-
consistency item, and one evidence-discipline item. Preserve every item, anchor,
and relative weight. Represent each 0-3 anchor as an `anchor_group` of three
ordered cumulative threshold bits: score 0 = 000, score 1 = 100, score 2 = 110,
and score 3 = 111. A nonmonotone group is invalid, never coerced to a score. The
thirteen groups therefore retain all partial-credit anchors and contribute 39
equally weighted raw threshold bits. Per judge, raw_score is the sum of these
bits and ranges from 0 to 39. The reported 24-point score is exactly
normalized_score = raw_score * 24 / 39, without rounding before the pass decision.
The pass fraction is 5/6: normalized_score >= 20, equivalently integer raw_score
>= 33. Passing additionally requires zero disqualifiers, the first cumulative
bit in every one of the four repair-quality groups equal to one, and campaign
greenness not machine-capped at zero. The executable product-edit,
skipped/deleted/xfail, assertion-weakening, mutant, fabricated-log,
unbounded-wait and campaign gates retain their original fail-closed effects;
normalization must not dilute or bypass them.

The repaired proposal must state this exact reproducible consensus protocol,
which is an authorized hash-bound verifier extension rather than stock native
binary-mean behavior. Under each composed step's judge config, use
`consensus: {mode: two_then_third, initial_samples: 2,
disagreement_tolerance: 0.0, resolution: median}` and 13 `anchor_groups`, each
with a stable rubric id and the three low-to-high threshold criterion indices.
The native `JudgeModelPolicy.samples` must equal `initial_samples`. Two
independent blinded initial judges each emit all 39 threshold criteria; their
thirteen 0-3 item scores, raw totals and normalized totals are derived from the
monotone groups, alongside disqualifiers, repair-first-bit flags and campaign-cap
state. If ANY threshold criterion differs between the two initial judges (absolute criterion
disagreement greater than zero), invoke one independent third judge to score the
complete 39-criterion rubric. Resolve every threshold criterion by the median of
its three binary scores, validate monotonicity of each resolved anchor group,
then derive the thirteen item scores, final raw score, normalized score and pass
decision. If all initial criterion scores agree, their common vector is final
and no third judge runs. Never average totals
or criterion vectors. The result must retain `initial_judgments`,
`disagreed_indices`, `adjudicator_judgments`, actual `judge_call_count`, and the
resolved criterion scores. Missing judge fields or nonmonotone anchor groups are
`invalid_task`; judge/verifier infrastructure failures are `infra_error`; both
carry null reward, never zero and never pass. Preserve judge independence and the original
calibration/anchor separation. The fresh reviewer must verify the 39-to-24
arithmetic, 5/6 threshold, integer threshold equivalence, all thirteen anchors,
every hard gate and the complete `two_then_third` median policy. Any missing item,
changed weight, weakened gate, ambiguous aggregation, assumption that unextended
stock native mode supports this contract, or task simplification requires repair
or rejection, not acceptance."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(source: Path, audit: Path = AUDIT_PATH) -> tuple[list[dict], dict]:
    if sha256(audit) != AUDIT_SHA256:
        raise ValueError("c02 construction audit hash differs from the pinned input")
    items = load_accepted(source, allow_revoked=True)
    matches = [
        item
        for item in items
        if item["proposal"]["capability_id"] == CAPABILITY_ID
        and item["proposal"]["slot"] == SLOT
    ]
    if len(matches) != 1:
        raise ValueError("source must contain exactly one c02 slot-10 candidate")
    original = matches[0]
    if original["proposal_hash"] != SOURCE_PROPOSAL_HASH:
        raise ValueError("c02 source proposal hash differs from the clarification input")

    candidate = copy.deepcopy(original)
    context = candidate["construction_context"]
    issues = context["portfolio_issues"]
    if SCORE_CONTRACT_CLARIFICATION not in issues:
        issues.append(SCORE_CONTRACT_CLARIFICATION)
    prior_admission = copy.deepcopy(candidate.get("admission"))
    context["clarification_lineage"] = {
        "source_proposal_hash": SOURCE_PROPOSAL_HASH,
        "source_admission": prior_admission,
        "construction_audit": {
            "path": str(audit),
            "sha256": AUDIT_SHA256,
        },
        "reuse_policy": (
            "Preserve prior construction artifacts as lineage and repair inputs; "
            "no old artifact is accepted under the clarified proposal hash without "
            "fresh hash-bound construction and validation."
        ),
    }
    context["score_contract_clarification"] = {
        "scored_items": 13,
        "item_range": [0, 3],
        "raw_max": 39,
        "normalization": "raw_score * 24 / 39",
        "normalized_threshold": 20,
        "minimum_integer_raw_score": 33,
        "pass_fraction": "5/6",
        "anchor_groups": [
            {
                "id": group_id,
                "threshold_bits": 3,
                "criterion_index_order": "low_to_high",
            }
            for group_id in (
                "mechanism_attribution_a",
                "mechanism_attribution_b",
                "mechanism_attribution_c",
                "mechanism_attribution_d",
                "isolating_experiment",
                "repair_quality_a",
                "repair_quality_b",
                "repair_quality_c",
                "repair_quality_d",
                "assertion_preservation",
                "campaign_design_greenness",
                "internal_consistency",
                "evidence_discipline",
            )
        ],
        "anchor_encoding": {
            "0": "000",
            "1": "100",
            "2": "110",
            "3": "111",
            "invalid_nonmonotone_outcome": "invalid_task_with_null_reward",
        },
        "preserve_partial_credit": True,
        "preserve_machine_gates": True,
        "critical_repair_condition": (
            "the first cumulative bit in every repair-quality anchor group is 1"
        ),
        "judge_protocol": {
            "mode": "two_then_third",
            "initial_samples": 2,
            "judge_model_policy_samples": 2,
            "disagreement_tolerance": 0.0,
            "third_trigger": "any criterion disagreement greater than tolerance",
            "adjudicator_samples": 1,
            "adjudicator_scope": "complete thirty-nine-criterion native pass",
            "resolution": "per-criterion median across three binary judgments",
            "no_disagreement": "common initial vector is final",
            "required_result_detail": [
                "initial_judgments",
                "disagreed_indices",
                "adjudicator_judgments",
                "judge_call_count",
                "resolved_criterion_scores",
            ],
            "missing_or_nonmonotone": "invalid_task_with_null_reward",
            "infrastructure_failure": "infra_error_with_null_reward",
            "forbid_total_or_vector_mean": True,
        },
        "requires_hash_bound_verifier_extension": True,
    }
    context["source_files"]["c02_construction_audit.md"] = AUDIT_SHA256
    candidate["admission"] = {
        "state": "pending",
        "scope": "individual_construction",
        "clarification_reason": "unresolved 39-point rubric versus 24-point threshold",
        "source_proposal_hash": SOURCE_PROPOSAL_HASH,
    }

    result = [candidate]
    clarification_sha256 = hashlib.sha256(
        SCORE_CONTRACT_CLARIFICATION.encode()
    ).hexdigest()
    manifest = {
        "schema_version": "c02-slot10-score-clarification-input-v1",
        "source_path": str(source),
        "source_sha256": sha256(source),
        "source_proposal_hash": SOURCE_PROPOSAL_HASH,
        "construction_audit_path": str(audit),
        "construction_audit_sha256": AUDIT_SHA256,
        "clarification_sha256": clarification_sha256,
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
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("runs/construction-admission-001/accepted.json"),
    )
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
