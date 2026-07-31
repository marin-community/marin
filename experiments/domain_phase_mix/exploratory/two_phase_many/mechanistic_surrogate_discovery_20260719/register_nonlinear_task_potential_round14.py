# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2"]
# ///
"""Register nonlinear task-potential gradient flow before fitting it."""

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
        "id": "NTPGF",
        "family": "Nonlinear task-potential gradient flow",
        "relationship_to_prior": "Extends rejected reduced gradient-flow bowl AI, scalar HWER, and noncommuting quadratic NQGF through a materially new transition invariant: state-dependent curvature makes scalar task vector fields noncommute. It does not add a phase head or output interaction.",
        "materially_new_mechanism": "The marginal gradient supplied by a data family changes with representation state. Sequential mixtures therefore follow nonlinear vector fields whose Lie bracket is nonzero even in one latent dimension.",
        "mechanistic_premise": "Training on broad and specialist data pulls a shared representation toward different optima. Each task has diminishing gradients near its optimum and steeper gradients far away, so early training changes the plasticity available to late data.",
        "governing_equations": "V_b(z)=0.5(z+1/2)^2+h(z+1/2)^4/4; V_r(z)=r(z-1/2)^2/2+hs(z-1/2)^4/4; dz/dt=-k[(1-p)V_b'(z)+pV_r'(z)]; Y=b+A[(1-q)V_b(z_T)+qV_r(z_T)], A>=0.",
        "latent_state": "A dimensionless shared specialization coordinate z in [-1/2,1/2], initialized at zero. Broad and rare task optima are fixed at -1/2 and +1/2.",
        "state_transition": "Deterministic convex gradient flow through each constant-mixture phase. The quartic coefficient h controls state-dependent curvature; h=0 is the exact quadratic/affine-flow ablation.",
        "response_link": "Evaluation is the same convex task-potential family at a target-specific mixture q, scaled by one nonnegative BPB amplitude and shifted by an intercept.",
        "additional_degrees_of_freedom": "Rare quadratic curvature r, common quartic strength h, rare-to-broad quartic ratio s, integrated relaxation k, evaluation mixture q, nonnegative BPB amplitude A, and intercept b. All but A and b are frozen by nested selection in the shape audit.",
        "units_and_symmetries": "z, p, r, h, s, k, q, and normalized time are dimensionless; A and b have BPB units. Fixed task optima, broad quadratic curvature, and z(0)=0 remove translation, reflection, and state-scale symmetries.",
        "single_phase_restriction": "When p0=p1, the autonomous flow composes exactly across an artificial phase boundary. The same restricted law is also independently refit on tied policies.",
        "starcoder_signature": "Nonzero h should rotate the WSD and cosine valleys through a stable state-dependent-curvature mechanism. The schedule may change k, but h and s should remain comparable; the raw optimum should remain inside the observed valley.",
        "catastrophic_optimism_resolution": "Convex task potentials grow quartically when a representation is driven far from an evaluation optimum, preventing a bounded additive benefit from rewarding extreme specialization.",
        "response_compression_resolution": "The quartic potential expands poor-policy response range through distance in a mechanistic latent state rather than an output calibrator.",
        "scale_transfer_expectation": "Dimensionless curvature ratios h and s should be shared or comparable across schedules and scales, while integrated relaxation k may grow with tokens-per-parameter and optimizer plasticity.",
        "cheapest_falsification": "The exact tied semigroup fails numerically; h=0 wins; nonzero h is fold-unstable or differs by more than 4x across schedules; nested RMSE misses the existing shape frontier by over 5%; or either raw optimum misses the observed best by more than 0.15.",
        "status": "active_preregistered",
        "status_evidence": "Frozen before any NTPGF fit. Historical and adversarial evaluation is forbidden until both StarCoder schedules support the nonlinear transition.",
    }
    registry = registry.loc[~registry["id"].eq("NTPGF")]
    registry = pd.concat([registry, pd.DataFrame([row], columns=registry.columns)], ignore_index=True)
    registry.to_csv(REGISTRY, index=False)

    ledger_row = {
        "timestamp": datetime.now(UTC).isoformat(),
        "round_id": "round_14_preregistration",
        "candidate_id": "NTPGF",
        "candidate_family": "Nonlinear task-potential gradient flow",
        "hyperparameters": "Frozen r={0.25,0.5,1,2,4}; h={0,0.25,1,4}; s={0.25,1,4}; k={0.5,1,2,4,8}; q={0.1,0.3,0.5,0.7,0.9}; ridge={0,0.1,1}; 512 RK4 steps per unit time",
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": True,
        "observations_inspiring_mechanism": "Quadratic gradient flows miss WSD while an unconstrained bilinear response is schedule-specific. This tests whether nonlinear state evolution, rather than an output interaction, creates the missing order effect.",
        "novelty_class": "State-dependent-curvature Lie bracket in a shared representation coordinate",
        "evaluation_status": "preregistered for algebraic and StarCoder gates; no NTPGF adversarial evaluation",
        "evidence_path": "approach_registry.csv",
        "notes": "The h=0 exact ablation is mandatory. Failure cannot be rescued by adding replay, a phase head, or output calibration in this round.",
    }
    identity = ["round_id", "candidate_id", "evaluation_status"]
    existing = set(map(tuple, ledger[identity].itertuples(index=False, name=None)))
    if tuple(ledger_row[column] for column in identity) not in existing:
        ledger = pd.concat([ledger, pd.DataFrame([ledger_row], columns=ledger.columns)], ignore_index=True)
        ledger.to_csv(LEDGER, index=False)
    print(registry.tail(1)[["id", "family", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()
