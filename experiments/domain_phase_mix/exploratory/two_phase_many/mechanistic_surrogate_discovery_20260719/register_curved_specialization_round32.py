# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2"]
# ///
"""Preregister curved-manifold specialization flow before fitting."""

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
        "id": "CMSF",
        "family": "Curved-manifold specialization flow",
        "relationship_to_prior": (
            "FPCGF uses a multiplicative production bottleneck and FSC uses a phenomenological acquisition gate. CMSF "
            "derives curriculum effects from conservative gradient flow toward a nonlinear specialist manifold s=g^m."
        ),
        "materially_new_mechanism": (
            "The rare-task optimum is a curved manifold coupling specialist state to general competence. Its gradient is "
            "weak and geometrically different before general features form, then becomes effective late."
        ),
        "mechanistic_premise": (
            "Specialist features are compositions of general features. Broad training can first build the substrate; rare "
            "training is most effective after that substrate exists, while continued broad training regularizes the specialist head away."
        ),
        "governing_equations": (
            "L_b=(g-1)^2/2+lambda s^2/2; L_r=rho(g-1)^2/2+(s-g^m)^2/2; "
            "d(g,s)/dt=-k[(1-p)grad L_b+p grad L_r]; Y=b0+A[(1-q)L_b+qL_r], A>=0. "
            "The exact linear-manifold ablation is m=1."
        ),
        "latent_state": "Dimensionless general capability g and specialist capability s.",
        "state_transition": "Autonomous gradient flow in a mixture-weighted sum of broad and specialist task potentials.",
        "response_link": "One nonnegative amplitude on the same evaluation-weighted task potential used by the transition, plus intercept.",
        "additional_degrees_of_freedom": (
            "Manifold power, broad specialist regularization, rare general weight, flow speed, evaluation mixture, one amplitude, intercept, and ridge."
        ),
        "units_and_symmetries": (
            "g,s,p,q are dimensionless; fixing both task optima and the coefficient of (s-g^m)^2 removes scale symmetry; "
            "k is per normalized time; A and b0 have BPB units."
        ),
        "single_phase_restriction": (
            "A tied policy gives one autonomous task-potential flow for unit duration; the same restricted law can be independently refit on tied observations."
        ),
        "starcoder_signature": (
            "m>1 should make early broad data prepare g and late rare data fit s, rotating the WSD optimum toward p1>p0; "
            "m=1 should be insufficient if nonlinear compositional prerequisites are real."
        ),
        "catastrophic_optimism_resolution": (
            "Policies missing either the general substrate or specialist alignment retain explicit task-potential error; no output interaction grants free credit."
        ),
        "response_compression_resolution": (
            "Curved-manifold distance separates frontier policies that have similar aggregate exposure but different ordering."
        ),
        "scale_transfer_expectation": (
            "The manifold power and dimensionless regularization ratio should transfer for similar task hierarchy; integrated speed may vary with scale."
        ),
        "cheapest_falsification": (
            "Reject if m=1 wins globally on either StarCoder schedule. If active, require nested m>1 on both schedules, "
            "compatible powers/ratios, shape RMSE within 5% of frozen references, and raw optimum distance at most 0.15."
        ),
        "status": "active_frozen_round32",
        "status_evidence": "Frozen before CMSF fitting; no new Delphi heldout or adversarial evaluation was read.",
    }
    registry = registry.loc[~registry["id"].eq("CMSF")]
    registry = pd.concat([registry, pd.DataFrame([row], columns=registry.columns)], ignore_index=True)
    registry.to_csv(REGISTRY, index=False)

    ledger = pd.read_csv(LEDGER)
    ledger_row = {
        "timestamp": datetime.now(UTC).isoformat(),
        "round_id": "round_32_batch_preregistration",
        "candidate_id": "CMSF",
        "candidate_family": "Curved-manifold specialization flow",
        "hyperparameters": (
            "power={1,2,3}, broad_specialist_regularization={0.25,1,4}, rare_general_weight={0,0.3,1}, "
            "speed={0.5,2,8}, evaluation={0.2,0.5,0.8}, ridge={0.1,1}; 256 RK4 steps/unit time"
        ),
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": True,
        "observations_inspiring_mechanism": (
            "The StarCoder late-specialization optimum suggests a compositional prerequisite, but prior phenomenological gates and product bottlenecks were unstable."
        ),
        "novelty_class": "Conservative gradient flow toward a nonlinear general-specialist task manifold",
        "evaluation_status": "frozen before two-stage StarCoder evaluation",
        "evidence_path": "round32_curved_specialization_starcoder/report.md",
        "notes": "Derived independently of Round 31 results; no adversarial outcomes will be read during the StarCoder gate.",
    }
    identity = ["round_id", "candidate_id", "evaluation_status"]
    existing = set(map(tuple, ledger[identity].itertuples(index=False, name=None)))
    if tuple(ledger_row[column] for column in identity) not in existing:
        ledger = pd.concat([ledger, pd.DataFrame([ledger_row], columns=ledger.columns)], ignore_index=True)
        ledger.to_csv(LEDGER, index=False)


if __name__ == "__main__":
    main()
