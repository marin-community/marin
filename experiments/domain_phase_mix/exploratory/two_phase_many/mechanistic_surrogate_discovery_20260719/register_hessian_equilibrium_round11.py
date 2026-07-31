# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2"]
# ///
"""Register the frozen Hessian-weighted equilibrium relaxation candidate."""

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
        "id": "HWER",
        "family": "Hessian-weighted equilibrium relaxation",
        "relationship_to_prior": "Reopens the reduced gradient-flow bowl AI only through a materially new transition law. AI relaxed at a constant rate toward raw family mass. HWER is the exact gradient flow of two quadratic domain losses: both the equilibrium and convergence rate depend on mixture-weighted curvature.",
        "materially_new_mechanism": "Mixture composition changes the local training Hessian as well as the target parameter equilibrium. High-curvature data can move the model quickly even at modest mass, while phase order matters through noncommuting relaxation toward successive equilibria.",
        "mechanistic_premise": "Near a training trajectory, broad and specialist losses can be approximated by quadratics with different curvature and preferred parameter values. Gradient flow on their mixture has a closed-form piecewise-constant transition.",
        "governing_equations": "Fix broad/specialist optima at 0/1. H(p)=1-p+r p and m(p)=r p/H(p). For phase duration a: z'=m(p)+(z-m(p))exp[-k H(p)a]. Evaluation Y=b+A(z-nu)^2 with A>=0. The one-dimensional affine coordinate fixes the otherwise arbitrary optimum scale.",
        "latent_state": "z is the learned parameter coordinate along the broad-to-specialist optimum displacement.",
        "state_transition": "Exact scalar gradient-flow relaxation under each constant-mixture phase, with mixture-dependent Hessian H(p) and curvature-weighted equilibrium m(p).",
        "response_link": "A convex quadratic evaluation loss around target coordinate nu with nonnegative curvature A; no replay term in the initial falsification because the quadratic compromise itself must generate both arms.",
        "additional_degrees_of_freedom": "Curvature ratio r, global integrated relaxation k, initial coordinate z0, evaluation optimum nu, one nonnegative BPB curvature, and an intercept. The initial StarCoder screen freezes finite grids for the four nonlinear quantities.",
        "units_and_symmetries": "z, p, r, k, phase duration, and nu are dimensionless; A and b have BPB units. Broad/specialist optimum locations are fixed at 0/1, removing affine state symmetry. Swapping labels maps z->1-z, r->1/r, and nu->1-nu.",
        "single_phase_restriction": "For p0=p1, exact semigroup composition equals one uninterrupted relaxation at p. The same form is separately refit on tied policies.",
        "starcoder_signature": "A curved constant-loss valley follows policies that end near nu. WSD 80/20 should show stronger late leverage because its long first phase approaches the early equilibrium before the short terminal phase rotates the state.",
        "catastrophic_optimism_resolution": "Extreme concentration moves the latent model toward a domain-specific parameter optimum and away from the evaluation optimum; surplus in one domain cannot linearly cancel that displacement.",
        "response_compression_resolution": "Quadratic evaluation curvature expands differences among remote terminal states without post-hoc output calibration.",
        "scale_transfer_expectation": "Curvature ratios and target coordinates should be more stable than integrated relaxation k, which should increase with optimization progress. Schedule changes enter through declared phase durations.",
        "cheapest_falsification": "Nested StarCoder RMSE fails to approach the existing shape frontier, curvature/relaxation select unsupported boundaries, label-swap invariance or tied semigroup fails, or either raw optimum misses the observed valley by more than 0.15 policy distance.",
        "status": "active_preregistered",
        "status_evidence": "Frozen before any HWER fit. No HWER prediction will be evaluated on historical or adversarial outcomes unless both StarCoder surfaces pass.",
    }
    registry = registry.loc[~registry["id"].eq("HWER")]
    registry = pd.concat([registry, pd.DataFrame([row], columns=registry.columns)], ignore_index=True)
    registry.to_csv(REGISTRY, index=False)

    ledger_row = {
        "timestamp": datetime.now(UTC).isoformat(),
        "round_id": "round_11_preregistration",
        "candidate_id": "HWER",
        "candidate_family": "Hessian-weighted equilibrium relaxation",
        "hyperparameters": "Frozen r={0.1,0.25,0.5,1,2,4,10}; k={0.1,0.3,1,3,10,30}; z0={0,0.25,0.5,0.75,1}; nu={0.05,0.1,...,0.95}; ridge={0,0.1,1}",
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": True,
        "observations_inspiring_mechanism": "AI omitted mixture-dependent Hessian and curvature-weighted equilibrium; adversarial outcomes motivate the need for nonadditive range but do not set this law or its grid.",
        "novelty_class": "Exact quadratic multi-task gradient-flow transition",
        "evaluation_status": "preregistered for algebraic and StarCoder gates; no HWER adversarial evaluation",
        "evidence_path": "approach_registry.csv",
        "notes": "No replay response is allowed in the first gate. Failure cannot be rescued by adding a tail term.",
    }
    identity = ["round_id", "candidate_id", "evaluation_status"]
    existing = set(map(tuple, ledger[identity].itertuples(index=False, name=None)))
    key = tuple(ledger_row[column] for column in identity)
    if key not in existing:
        ledger = pd.concat([ledger, pd.DataFrame([ledger_row], columns=ledger.columns)], ignore_index=True)
        ledger.to_csv(LEDGER, index=False)
    print(registry.tail(1)[["id", "family", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()
