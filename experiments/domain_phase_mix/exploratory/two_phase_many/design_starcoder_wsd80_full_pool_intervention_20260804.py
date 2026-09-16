# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

"""Freeze the WSD80 physical-full-pool repetition intervention."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    design_starcoder_wsd80_batch_repetition_intervention_20260804 as parent_design,
)

DESIGN_VERSION = "2026-08-04-v2"
CONDITION_ID = "fullpool"
CONDITION_FAMILY = "physical_full_pool"
EXPECTED_RUN_COUNT = len(parent_design.POLICIES) * len(parent_design.PAIR_SEEDS)
PRIMARY_METRIC = "eval/paloma/dolma_100_programing_languages-llama3/bpb"
EQUIVALENCE_MARGIN_BPB = 0.001
TEST_ALPHA = 0.05
TRAINING_CODE_BASE_COMMIT = "86c4154a638577e9a6cfc2083abb8399527671eb"

OUTPUT_PATH = Path(__file__).resolve().parents[2] / "starcoder_wsd80_full_pool_design_20260804.json"
ARTIFACT_DIR = Path(__file__).parent / "reference_outputs" / "starcoder_wsd80_full_pool_design_20260804"
CSV_PATH = ARTIFACT_DIR / "run_manifest.csv"
REPORT_PATH = ARTIFACT_DIR / "report.md"


def _physical_pool_usage(policy: parent_design.PolicySpec) -> dict[str, float | int]:
    beta_0 = float(parent_design.PHASE_0_FRACTION)
    beta_1 = 1.0 - beta_0
    phase_0 = float(policy.phase_0_starcoder)
    phase_1 = float(policy.phase_1_starcoder)
    tokens = parent_design.MATERIALIZED_TOKENS

    starcoder_phase_0_tokens = beta_0 * phase_0 * tokens
    starcoder_phase_1_tokens = beta_1 * phase_1 * tokens
    nemotron_phase_0_tokens = beta_0 * (1.0 - phase_0) * tokens
    nemotron_phase_1_tokens = beta_1 * (1.0 - phase_1) * tokens
    starcoder_total_tokens = starcoder_phase_0_tokens + starcoder_phase_1_tokens
    nemotron_total_tokens = nemotron_phase_0_tokens + nemotron_phase_1_tokens

    starcoder_pool_fraction = starcoder_total_tokens / parent_design.STARCODER_SOURCE_TOKENS
    nemotron_pool_fraction = nemotron_total_tokens / parent_design.NEMOTRON_SOURCE_TOKENS
    if not 0.0 <= starcoder_pool_fraction < 1.0:
        raise ValueError(f"{policy.policy_id}: StarCoder physical source would restart")
    if not 0.0 <= nemotron_pool_fraction < 1.0:
        raise ValueError(f"{policy.policy_id}: Nemotron physical source would restart")

    return {
        "starcoder_phase_0_tokens": round(starcoder_phase_0_tokens),
        "starcoder_phase_1_tokens": round(starcoder_phase_1_tokens),
        "starcoder_total_tokens": round(starcoder_total_tokens),
        "starcoder_physical_pool_fraction": starcoder_pool_fraction,
        "nemotron_phase_0_tokens": round(nemotron_phase_0_tokens),
        "nemotron_phase_1_tokens": round(nemotron_phase_1_tokens),
        "nemotron_total_tokens": round(nemotron_total_tokens),
        "nemotron_physical_pool_fraction": nemotron_pool_fraction,
    }


def build_payload() -> dict[str, Any]:
    """Build and validate the immutable full-pool design payload."""
    parent_payload = parent_design.build_payload()
    base_condition = parent_design.CONDITIONS[0]
    if base_condition.condition_id != "base":
        raise ValueError("Parent design no longer starts with the historical base condition")
    schedule = parent_design._schedule(base_condition)
    if schedule["materialized_tokens"] != parent_design.MATERIALIZED_TOKENS:
        raise ValueError("Historical high-D token budget drifted")

    target_budget = parent_design.MATERIALIZED_TOKENS
    target_budget_multiplier = target_budget / parent_design.BASE_TARGET_BUDGET
    if target_budget_multiplier <= 0.0 or target_budget_multiplier >= 1.0:
        raise ValueError("Full-pool target-budget multiplier must lie strictly between zero and one")

    policy_rows = [parent_design._policy_row(policy) for policy in parent_design.POLICIES]
    if abs(float(policy_rows[0]["aggregate_starcoder"]) - float(policy_rows[1]["aggregate_starcoder"])) > 1e-12:
        raise ValueError("Policies A and B are not aggregate matched")

    rows: list[dict[str, Any]] = []
    for policy, policy_row in zip(parent_design.POLICIES, policy_rows, strict=True):
        usage = _physical_pool_usage(policy)
        for seed in parent_design.PAIR_SEEDS:
            rows.append(
                {
                    "run_name": f"fullpool_{policy.policy_id}_s{seed}",
                    "condition_id": CONDITION_ID,
                    "condition_family": CONDITION_FAMILY,
                    "policy_id": policy.policy_id,
                    "policy_role": policy.role,
                    "pair_seed": seed,
                    "data_seed": seed,
                    "simulated_epoch_subset_seed": seed,
                    "phase_0_starcoder": policy_row["phase_0_starcoder"],
                    "phase_1_starcoder": policy_row["phase_1_starcoder"],
                    "aggregate_starcoder": policy_row["aggregate_starcoder"],
                    "batch_size": base_condition.batch_size,
                    "total_steps": schedule["total_steps"],
                    "boundary_step": schedule["boundary_step"],
                    "warmup_steps": schedule["warmup_steps"],
                    "decay_steps": schedule["decay_steps"],
                    "eval_interval_steps": schedule["eval_interval_steps"],
                    "materialized_tokens": schedule["materialized_tokens"],
                    "muon_learning_rate": schedule["muon_learning_rate"],
                    "adam_learning_rate": schedule["adam_learning_rate"],
                    "target_budget": target_budget,
                    "target_budget_multiplier": target_budget_multiplier,
                    "unique_pool_scale_relative": 1.0 / target_budget_multiplier,
                    "simulated_epoch_scale_relative": target_budget_multiplier,
                    **usage,
                }
            )

    if len(rows) != EXPECTED_RUN_COUNT or len({row["run_name"] for row in rows}) != EXPECTED_RUN_COUNT:
        raise ValueError("Full-pool run manifest has the wrong cardinality")
    for seed in parent_design.PAIR_SEEDS:
        seed_rows = [row for row in rows if row["pair_seed"] == seed]
        if len(seed_rows) != len(parent_design.POLICIES):
            raise ValueError(f"Seed {seed}: incomplete paired policy triplet")

    payload: dict[str, Any] = {
        "design_version": DESIGN_VERSION,
        "description": "Fixed-arm WSD80 full-cache-support intervention with no exact source-index restart.",
        "parent_design_sha256": parent_payload["design_sha256"],
        "training_code_base_commit": TRAINING_CODE_BASE_COMMIT,
        "expected_run_count": EXPECTED_RUN_COUNT,
        "pair_seeds": list(parent_design.PAIR_SEEDS),
        "training_environment": parent_payload["training_environment"],
        "fixed_training": parent_payload["fixed_training"],
        "condition": {
            "condition_id": CONDITION_ID,
            "family": CONDITION_FAMILY,
            "batch_size": base_condition.batch_size,
            "target_budget": target_budget,
            "target_budget_multiplier": target_budget_multiplier,
            "unique_pool_scale_relative": 1.0 / target_budget_multiplier,
            "simulated_epoch_scale_relative": target_budget_multiplier,
            "interpretation": (
                "Make every physical source-cache index eligible; the consumed stream cannot wrap within any "
                "fixed policy."
            ),
            **schedule,
        },
        "policies": policy_rows,
        "analysis": {
            "primary_metric": PRIMARY_METRIC,
            "primary_estimand": "delta_order(fullpool) = mean_s[loss(B_agg018,s) - loss(A_phase,s)]",
            "equivalence_margin_bpb": EQUIVALENCE_MARGIN_BPB,
            "alpha": TEST_ALPHA,
            "primary_test": (
                "TOST equivalence test on the six paired full-pool differences against "
                "[-0.001,+0.001] BPB at alpha=0.05"
            ),
            "primary_decision_rule": {
                "collapse": (
                    "Declare the residual order gain practically equivalent to zero only when the 90% confidence "
                    "interval for delta_order(fullpool) lies wholly inside [-0.001,+0.001] BPB."
                ),
                "material_persistence": (
                    "Declare a materially persistent phase-order gain only when the 90% confidence interval lies "
                    "wholly above +0.001 BPB."
                ),
                "otherwise": "Report the mechanism test as inconclusive.",
            },
            "equivalence_margin_rationale": (
                "0.001 BPB is a preregistered practical-null scale, about one sixth of the independent 0.006101-BPB "
                "fresh-seed policy-class gain at this N,D cell; it is not estimated from either intervention panel."
            ),
            "secondary_treatment_estimand": (
                "gamma_fullpool = mean_s[(loss(B,s)-loss(A,s))_fullpool - (loss(B,s)-loss(A,s))_base]"
            ),
            "secondary_test": "two-sided one-sample t-test and 95% confidence interval for the six gamma_s values",
            "sensitivity_reporting": (
                "Before inspecting full-pool outcomes, compute the achieved minimum detectable effect from the "
                "completed base-condition paired SD; report it without changing the frozen margin or decision rule."
            ),
            "secondary_estimands": [
                "delta_aggregate = loss(B_agg018) - loss(C_tied070)",
                "delta_global = loss(C_tied070) - loss(A_phase)",
            ],
            "selection_policy": (
                "All policies, seeds, estimands, and thresholds are frozen before outcomes are inspected."
            ),
        },
        "caveats": [
            (
                "No source-cache index repeats, but the underlying corpora may contain duplicate or "
                "near-duplicate content."
            ),
            (
                "The intervention rules out exact finite-pool restart, not semantic redundancy or "
                "distributional overfitting."
            ),
            (
                "The base and full-pool conditions consume different StarCoder content: the intervention replaces "
                "repeated traversal of a smaller block sample with one traversal of a larger block sample. Content "
                "identity and exact repetition therefore change together."
            ),
            (
                "Nominal A/B aggregate weights match exactly, but mixture-block integer allocation creates the same "
                "small realized aggregate discrepancy in both conditions; the cross-condition contrast cancels it."
            ),
            (
                "The fixed A/B/C arms measure gain at one aggregate and contrast; they do not "
                "re-identify a global optimum."
            ),
        ],
        "runs": rows,
    }
    payload["design_sha256"] = parent_design.canonical_sha256(payload)
    return payload


def _write_report(payload: dict[str, Any]) -> None:
    policy_lines = []
    base_condition = parent_design.CONDITIONS[0]
    for policy in parent_design.POLICIES:
        usage = _physical_pool_usage(policy)
        base_epochs = parent_design._epoch_summary(base_condition, policy)
        starcoder_epochs = base_epochs["starcoder_phase_0_epochs"] + base_epochs["starcoder_phase_1_epochs"]
        policy_lines.append(
            "| {policy} | {starcoder:.3f}B | {starcoder_fraction:.3f}% | {base_epochs:.2f}x | {nemotron:.3f}B | "
            "{nemotron_fraction:.4f}% |".format(
                policy=policy.policy_id,
                starcoder=float(usage["starcoder_total_tokens"]) / 1e9,
                starcoder_fraction=100.0 * float(usage["starcoder_physical_pool_fraction"]),
                base_epochs=starcoder_epochs,
                nemotron=float(usage["nemotron_total_tokens"]) / 1e9,
                nemotron_fraction=100.0 * float(usage["nemotron_physical_pool_fraction"]),
            )
        )
    condition = payload["condition"]
    REPORT_PATH.write_text(
        "\n".join(
            [
                "# StarCoder WSD80 physical-full-pool intervention",
                "",
                "This frozen 18-run extension uses the same high-D model, schedule, A/B/C policies, and six paired "
                "seeds as the parent batch/repetition panel. The only intervention is the source pool.",
                "",
                "## Intervention",
                "",
                f"- Materialized tokens: `{condition['materialized_tokens']:,}`",
                f"- Target budget: `{condition['target_budget']:,}`",
                "- Simulated-data slice ratio: `1.0` (every physical cache index is eligible)",
                (
                    "- Eligible cache-support scale relative to the historical panel: "
                    f"`{condition['unique_pool_scale_relative']:.2f}x`"
                ),
                f"- Simulated-epoch scale relative to the historical panel: "
                f"`{condition['simulated_epoch_scale_relative']:.6f}x`",
                "- Runs: `3 policies x 6 paired seeds = 18`",
                "",
                "| policy | StarCoder used | StarCoder pool | base StarCoder epochs | Nemotron used | Nemotron pool |",
                "|---|---:|---:|---:|---:|---:|",
                *policy_lines,
                "",
                "All consumed source fractions are below one, so Levanter's restart sampler never wraps a "
                "source-cache index. The 773x figure is the increase in eligible cache support, not the number of "
                "additional examples consumed; for A and B, the intervention removes about 4.76 traversals of the "
                "historical StarCoder subset.",
                "",
                "## Frozen analysis",
                "",
                "The primary metric is Paloma Dolma 100 Programming Languages BPB. Define `delta_order = L(B)-L(A)`. "
                "The confirmatory test is a paired TOST at alpha 0.05: practical equivalence requires the 90% "
                "confidence interval for `delta_order(fullpool)` to lie wholly inside +/-0.001 BPB. A 90% interval "
                "wholly above +0.001 BPB establishes material persistence; every other outcome is inconclusive.",
                "",
                "The secondary attribution contrast uses the same six seeds in the already-frozen base condition: "
                "`gamma_fullpool = delta_order(fullpool)-delta_order(base)`. Before inspecting full-pool outcomes, "
                "the base paired SD will be used to report achieved sensitivity, without changing the decision rule.",
                "",
                (
                    "This is a no-exact-repeat test, not a claim that semantic duplication or all "
                    "forms of overfitting vanish."
                ),
                "The intervention also changes which StarCoder blocks are consumed, so content identity and exact "
                "repetition are not separately randomized.",
                "",
                f"Parent design SHA-256: `{payload['parent_design_sha256']}`",
                f"Training-code base commit: `{payload['training_code_base_commit']}`",
                f"Design SHA-256: `{payload['design_sha256']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    payload = build_payload()
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(payload["runs"][0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(payload["runs"])
    _write_report(payload)
    print(
        json.dumps(
            {
                "design_path": str(OUTPUT_PATH),
                "manifest_path": str(CSV_PATH),
                "report_path": str(REPORT_PATH),
                "run_count": len(payload["runs"]),
                "design_sha256": payload["design_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
