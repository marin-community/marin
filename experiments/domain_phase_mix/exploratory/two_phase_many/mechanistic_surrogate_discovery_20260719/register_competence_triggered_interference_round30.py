# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2"]
# ///
"""Preregister competence-triggered gradient interference before fitting."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
REGISTRY = OUTPUT_ROOT / "approach_registry.csv"
LEDGER = OUTPUT_ROOT / "data_use_ledger.csv"


def main() -> None:
    registry = pd.read_csv(REGISTRY)
    row = {
        "id": "CTGI",
        "family": "Competence-triggered gradient interference",
        "relationship_to_prior": (
            "FSC gates specialist acquisition on foundation competence, while PLAFK and factorized flow apply stationary "
            "broad-to-specialist forgetting. CTGI introduces a sign-changing transition: broad data is harmless to the "
            "specialist state before general competence matures and destructive afterward."
        ),
        "materially_new_mechanism": (
            "A bounded general-competence state crosses a threshold that activates destructive gradient interference "
            "from broad data on specialist competence."
        ),
        "mechanistic_premise": (
            "Early broad gradients build reusable features; after those features mature, broad and specialist gradients "
            "conflict, so specialist examples become more valuable late in training."
        ),
        "governing_equations": (
            "dg/dt=a[(1-p)+r p](1-g); ds/dt=b p g(1-s)-c(1-p)s sigmoid((g-theta)/delta); "
            "Y=b0+A[(1-q)(1-g_T)+q(1-s_T)], A>=0. The exact interference ablation is c=0."
        ),
        "latent_state": "General competence g in [0,1] and specialist competence s in [0,1].",
        "state_transition": (
            "Autonomous bounded acquisition plus a competence-triggered specialist-forgetting hazard, integrated continuously across phases."
        ),
        "response_link": "One nonnegative amplitude on a fixed convex combination of unresolved general and specialist error, plus an intercept.",
        "additional_degrees_of_freedom": (
            "General acquisition rate, rare-data general efficiency, specialist acquisition rate, interference rate, "
            "competence threshold, fixed transition softness, evaluation mixture, one amplitude, intercept, and ridge."
        ),
        "units_and_symmetries": (
            "g,s,p,q,theta are dimensionless; rates are per normalized training time; delta is a dimensionless competence width; "
            "A and b0 have BPB units. Fixed state endpoints remove affine state symmetries."
        ),
        "single_phase_restriction": (
            "For tied phase weights the autonomous flow composes across the artificial boundary; the same CTGI law can be refit on tied policies."
        ),
        "starcoder_signature": (
            "Compared with c=0, active interference should create a late-specialization advantage and place the WSD optimum "
            "on the p1>p0 side while allowing the cosine optimum to remain nearer tied."
        ),
        "catastrophic_optimism_resolution": (
            "Policies that continue broad training after specialist competence forms pay an explicit bounded forgetting hazard instead of receiving free aggregate exposure credit."
        ),
        "response_compression_resolution": (
            "Threshold activation expands response differences among high-aggregate policies according to when broad exposure occurs."
        ),
        "scale_transfer_expectation": (
            "The threshold is a dimensionless competence level; rates may change with tokens per parameter, but the sign switch should transfer if gradient conflict is the mechanism."
        ),
        "cheapest_falsification": (
            "Reject if c=0 wins globally on either StarCoder schedule. If active, require nested selection on both schedules, "
            "compatible thresholds, shape RMSE within 5% of frozen references, and raw optimum distance at most 0.15."
        ),
        "status": "active_frozen_round30",
        "status_evidence": "Frozen before CTGI fitting; no new Delphi heldout or adversarial evaluation was read.",
    }
    registry = registry.loc[~registry["id"].eq("CTGI")]
    registry = pd.concat([registry, pd.DataFrame([row], columns=registry.columns)], ignore_index=True)
    registry.to_csv(REGISTRY, index=False)

    ledger = pd.read_csv(LEDGER)
    ledger_row = {
        "timestamp": datetime.now(UTC).isoformat(),
        "round_id": "round_30_batch_preregistration",
        "candidate_id": "CTGI",
        "candidate_family": "Competence-triggered gradient interference",
        "hyperparameters": (
            "general_rate={1,4,16}, rare_general_efficiency={0.3,1}, specialist_rate={1,4,16}, "
            "interference={0,2,8}, threshold={0.25,0.6}, softness=0.1, eval={0.2,0.5,0.8}, ridge={0.1,1}; "
            "128 RK4 steps/unit time"
        ),
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": True,
        "observations_inspiring_mechanism": (
            "StarCoder asymmetry and the failure of stationary forgetting motivate testing whether interference begins only after general competence matures."
        ),
        "novelty_class": "Competence-dependent sign change in broad-to-specialist gradient interaction",
        "evaluation_status": "frozen before two-stage StarCoder evaluation",
        "evidence_path": "round30_competence_triggered_interference_starcoder/report.md",
        "notes": "No adversarial outcomes will be read during the StarCoder gate.",
    }
    identity = ["round_id", "candidate_id", "evaluation_status"]
    existing = set(map(tuple, ledger[identity].itertuples(index=False, name=None)))
    if tuple(ledger_row[column] for column in identity) not in existing:
        ledger = pd.concat([ledger, pd.DataFrame([ledger_row], columns=ledger.columns)], ignore_index=True)
        ledger.to_csv(LEDGER, index=False)


if __name__ == "__main__":
    main()
