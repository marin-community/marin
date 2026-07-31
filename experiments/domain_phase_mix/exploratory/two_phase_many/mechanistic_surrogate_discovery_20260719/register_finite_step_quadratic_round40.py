# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2"]
# ///
"""Preregister finite-step quadratic flow before StarCoder evaluation."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
REGISTRY = OUTPUT_ROOT / "approach_registry.csv"
LEDGER = OUTPUT_ROOT / "data_use_ledger.csv"


def registry_row() -> dict[str, str]:
    return {
        "id": "FSQF",
        "family": "Finite-step quadratic task flow",
        "relationship_to_prior": "Reopens NQGF and optimizer-time flow only through the discrete transition law. Prior routes retained only integrated LR mass, whereas finite-step GD depends on all power sums of the declared LR sequence.",
        "materially_new_mechanism": "The task transition retains second- through fourth-order learning-rate moments from the exact optimizer schedule; first order is the continuous-flow ablation.",
        "mechanistic_premise": "For a quadratic mode with curvature h, one optimizer step contracts by 1-eta h, not exp(-eta h). Products over a phase therefore depend on sum eta^2 and higher moments, which differ between cosine and WSD even after integrated LR time is matched.",
        "governing_equations": "theta_{k+1}=theta_k-s eta_k H_p(theta_k-mu_p); log P_p(h)=sum_k log(1-s eta_k h)=-sum_{r>=1}(s h)^r sum_k eta_k^r/r; Y=b+A||theta_T-c||^2, A>=0. Orders 1,2,4 are compared, with order 1 the exact continuous-flow ablation.",
        "latent_state": "A two-dimensional shared task representation theta. Broad and rare quadratic losses have differently oriented Hessians and optima.",
        "state_transition": "Exact affine phase equilibrium plus a fourth-order modified-equation approximation to the discrete LR sequence. The approximation is checked against the exact step product for selected configurations.",
        "response_link": "One nonnegative BPB amplitude on terminal squared distance to a fixed evaluation point plus an intercept.",
        "additional_degrees_of_freedom": "No free schedule parameter beyond task curvature, Hessian anisotropy/angle, total relaxation, and evaluation point. Expansion order is compared as a nested discrete-transition ablation.",
        "units_and_symmetries": "theta, task optima, and normalized LR are dimensionless; total relaxation is dimensionless integrated curvature; output amplitude and intercept have BPB units. Fixed task optima remove affine state symmetry.",
        "single_phase_restriction": "For tied phases the exact discrete products concatenate to the unsplit optimizer sequence; no artificial boundary remains. The restricted equation can be refit independently on tied data.",
        "starcoder_signature": "Higher-order LR moments must beat first-order flow on both schedules, preserve the cosine near-diagonal optimum, and explain the WSD late-rare optimum through the actual warmup/stable/decay sequence.",
        "catastrophic_optimism_resolution": "Finite-step stability prices high-curvature extreme mixtures more strongly than integrated-time flow; this is accepted only if raw optima remain stable and non-cornered.",
        "response_compression_resolution": "Schedule-specific higher moments can expand phase-order response without a free late multiplier or output correction.",
        "scale_transfer_expectation": "The law transfers through declared step count and LR schedule; dimensionless Hessian geometry should be comparable, while total relaxation may vary with model scale.",
        "cheapest_falsification": "Reject before multi-swarm work unless order 2 or 4 beats order 1 globally and in at least three folds on both schedules, clears both corrected shape references, locates both raw optima within 0.15, and agrees with the exact step product.",
        "status": "active_round40_preregistered",
        "status_evidence": "Frozen grid: curvature {0.5,1,2}; Hessian anisotropy {0.25,1,4}; angle {0,45,90}; total relaxation {1,4,16,64}; evaluation center {-0.2,0,0.2}; expansion order {1,2,4}; ridge {0,0.1,1}. No historical, exposed-adversarial, or sealed-confirmation outcome will be read unless the StarCoder gate passes.",
    }


def main() -> None:
    registry = pd.read_csv(REGISTRY)
    row = registry_row()
    registry = registry.loc[~registry["id"].eq(row["id"])]
    registry = pd.concat([registry, pd.DataFrame([row], columns=registry.columns)], ignore_index=True)
    registry.to_csv(REGISTRY, index=False)

    ledger = pd.read_csv(LEDGER)
    ledger_row = {
        "timestamp": datetime.now(UTC).isoformat(),
        "round_id": "round_40_finite_step_quadratic",
        "candidate_id": row["id"],
        "candidate_family": row["family"],
        "hyperparameters": row["status_evidence"].split("Frozen grid: ", maxsplit=1)[1],
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": True,
        "observations_inspiring_mechanism": "Corrected schedule provenance established exact LR sequences; all earlier task flows reduced them to integrated mass. No new adversarial target was inspected.",
        "novelty_class": "Higher optimizer-step moments in a discrete transition law",
        "evaluation_status": "preregistered before StarCoder evaluation",
        "evidence_path": "round40_finite_step_quadratic_starcoder/report.md",
        "notes": "The exposed development panel was already known. The sealed frontier phase-fiber panel remains untouched.",
    }
    identity = ["round_id", "candidate_id", "evaluation_status"]
    existing = set(map(tuple, ledger[identity].itertuples(index=False, name=None)))
    key = tuple(ledger_row[column] for column in identity)
    if key not in existing:
        ledger = pd.concat([ledger, pd.DataFrame([ledger_row], columns=ledger.columns)], ignore_index=True)
        ledger.to_csv(LEDGER, index=False)


if __name__ == "__main__":
    main()
