# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2"]
# ///
"""Preregister optimizer-momentum gradient flow before fitting."""

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
    ledger = pd.read_csv(LEDGER)
    row = {
        "id": "OMGF",
        "family": "Optimizer-momentum gradient flow",
        "relationship_to_prior": "Extends first-order scalar HWER with a materially new signed velocity state. It differs from fast/slow consolidation AL because the second state is optimizer momentum, can change sign, and enters a damped second-order transition rather than a second competence pool.",
        "materially_new_mechanism": "Optimizer momentum acquired under phase 0 persists across the distribution shift and changes the phase-1 trajectory, allowing overshoot and hysteresis even for quadratic task losses.",
        "mechanistic_premise": "Momentum-based optimizers carry a velocity-like update state. A phase boundary changes the gradient equilibrium immediately but not the carried velocity, so early data can accelerate or oppose late specialization.",
        "governing_equations": "d2z/dt2+2 xi omega dz/dt+omega^2 H(p)[z-m(p)]=0; H(p)=1-p+r p; m(p)=[-(1-p)+r p]/[2H(p)]; Y=b+A(z_T-nu)^2, A>=0. The exact ablation is dz/dt=-omega H(p)[z-m(p)].",
        "latent_state": "A dimensionless scalar specialization position z and signed optimizer velocity v=dz/dt, initialized at zero.",
        "state_transition": "Exact damped-oscillator flow through each constant-mixture phase. Position and velocity are continuous at the phase boundary; first-order relaxation is included as the exact no-momentum ablation.",
        "response_link": "One nonnegative BPB amplitude times squared distance from a target-specific evaluation center, plus an intercept. Velocity is not evaluated directly.",
        "additional_degrees_of_freedom": "Rare-to-broad curvature ratio r, integrated natural frequency omega, damping ratio xi, evaluation center nu, nonnegative amplitude, and intercept. The first-order ablation omits xi.",
        "units_and_symmetries": "z, p, r, xi, nu, and normalized time are dimensionless; omega is inverse normalized time; v has inverse-time units; amplitude and intercept carry BPB. Fixed task optima and initial state remove affine state symmetries.",
        "single_phase_restriction": "Tied phases compose the same autonomous second-order flow exactly. The same restricted law is also independently selected and fitted on tied policies.",
        "starcoder_signature": "A stable non-first-order damping regime should improve both schedules and shift WSD more strongly because its late phase is shorter. Raw optima should stay in the observed valleys rather than use unbounded momentum overshoot.",
        "catastrophic_optimism_resolution": "The quadratic terminal response grows when carried momentum drives the representation past the evaluation optimum; extreme schedules cannot receive an unbounded additive benefit.",
        "response_compression_resolution": "Under- or critically damped trajectories can expand response range through physically interpretable overshoot rather than output calibration.",
        "scale_transfer_expectation": "Damping ratio should be comparable for the same optimizer; integrated frequency may scale with training duration and tokens per parameter. If phase lengths dwarf optimizer memory, first-order dynamics should win.",
        "cheapest_falsification": "The first-order ablation wins globally or in most folds; damping regimes disagree across schedules; nested RMSE misses the shape frontier by over 5%; or raw optima miss observed minima by more than 0.15.",
        "status": "active_preregistered",
        "status_evidence": "Frozen before any OMGF fit. No historical or adversarial outcome may be evaluated until both StarCoder schedules support the momentum state.",
    }
    registry = registry.loc[~registry["id"].eq("OMGF")]
    registry = pd.concat([registry, pd.DataFrame([row], columns=registry.columns)], ignore_index=True)
    registry.to_csv(REGISTRY, index=False)

    ledger_row = {
        "timestamp": datetime.now(UTC).isoformat(),
        "round_id": "round_17_preregistration",
        "candidate_id": "OMGF",
        "candidate_family": "Optimizer-momentum gradient flow",
        "hyperparameters": "Frozen dynamics={first_order,momentum}; r={0.25,0.5,1,2,4}; omega={0.5,1,2,4,8}; xi={0.25,0.5,1,2}; nu={-0.4,-0.2,0,0.2,0.4}; ridge={0,0.1,1}",
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": True,
        "observations_inspiring_mechanism": "Scalar affine endpoint dynamics cannot create a strict phase advantage, while nonlinear task potentials and unconstrained bilinear interactions fail transfer. A signed optimizer state is a distinct path variable with a direct training interpretation.",
        "novelty_class": "Signed optimizer-velocity state with exact damped transition",
        "evaluation_status": "preregistered for algebraic and StarCoder gates; no OMGF adversarial evaluation",
        "evidence_path": "approach_registry.csv",
        "notes": "The first-order ablation is mandatory. Failure cannot be rescued by evaluating velocity, adding replay, or adding a phase output head.",
    }
    identity = ["round_id", "candidate_id", "evaluation_status"]
    existing = set(map(tuple, ledger[identity].itertuples(index=False, name=None)))
    if tuple(ledger_row[column] for column in identity) not in existing:
        ledger = pd.concat([ledger, pd.DataFrame([ledger_row], columns=ledger.columns)], ignore_index=True)
        ledger.to_csv(LEDGER, index=False)
    print(registry.tail(1)[["id", "family", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()
