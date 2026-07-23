# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2"]
# ///
"""Preregister spherical projected flow before StarCoder evaluation."""

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
        "id": "SPTF",
        "family": "Spherical projected task flow",
        "relationship_to_prior": "Reopens conservative task flow through the actual constant-norm manifold imposed by MuonH. Prior Euclidean NQGF/NTPGF and the generic curved-specialization surface did not encode MuonH's norm-preserving state invariant.",
        "materially_new_mechanism": "Projected learning on a constant-norm parameter manifold; zero angular curvature is the exact Euclidean quadratic-flow ablation.",
        "mechanistic_premise": "MuonH renormalizes every linear weight to its initialization norm after each update. Broad and rare gradient fields therefore compose on a sphere rather than in Euclidean parameter space, which can make phase order matter through geodesic curvature.",
        "governing_equations": "psi_c(u)=(1-cos(cu))/c^2 with psi_0(u)=u^2/2; dx/dtau=-k[(1-p)psi_c'(x+1/2)+r p psi_c'(x-1/2)]; Y=b+A[(1-q)psi_c(x_T+1/2)+q r psi_c(x_T-1/2)], A>=0. c=0 is exact Euclidean flow.",
        "latent_state": "One dimensionless geodesic coordinate x on a fixed-norm task manifold, initialized midway between broad and rare task optima.",
        "state_transition": "Autonomous projected gradient flow through each phase in the fixed normalized integrated-learning-rate clock of the declared schedule.",
        "response_link": "One nonnegative BPB amplitude on terminal broad/rare chordal task debt plus an intercept; manifold curvature cannot directly recalibrate the output.",
        "additional_degrees_of_freedom": "One dimensionless angular-curvature c beyond the exact Euclidean ablation. Rare curvature, relaxation, evaluation mixture, and ridge use a frozen finite grid.",
        "units_and_symmetries": "x and phase weights are dimensionless; c is radians per unit x; k is inverse normalized optimizer time; A and b have BPB units. Fixed optima, initialization, and broad curvature remove affine state and task-gradient-scale symmetries.",
        "single_phase_restriction": "For phase-tied weights the autonomous projected flow composes exactly across the artificial boundary. The same restricted law can be independently fitted on tied data.",
        "starcoder_signature": "Interior nonzero curvature must beat c=0 on both schedules, retain the cosine near-diagonal optimum, and move the WSD raw optimum toward late rare enrichment near (0.1,0.5).",
        "catastrophic_optimism_resolution": "A bounded geodesic task distance prevents concentrated policies from claiming unbounded Euclidean capability displacement; this claim is accepted only if raw-optimum and later heldout gates pass.",
        "response_compression_resolution": "Curved task geometry can separate policies with the same aggregate exposure but different traversal paths without an output calibrator.",
        "scale_transfer_expectation": "Constant-norm projection is optimizer-defined and shared across these MuonH swarms. Dimensionless angular separation should be stable across schedules; relaxation may depend on total optimizer progress.",
        "cheapest_falsification": "Reject if c=0 wins either schedule, c selects a boundary or differs by over 4x, nested RMSE misses either corrected StarCoder reference by over 5%, or either raw optimum is over 0.15 from the observed optimum.",
        "status": "active_round36_preregistered",
        "status_evidence": "Frozen before evaluation. Grid: angular curvature {0,0.5,1,2,3}, rare curvature {0.25,0.5,1,2,4}, relaxation {0.5,1,2,4,8,16}, evaluation rare weight {0.1,0.2,0.5,0.8,1}, ridge {0,0.1,1}. No Delphi or adversarial outcome will be read unless the StarCoder gate passes.",
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
        "round_id": "round_36_spherical_projected_flow",
        "candidate_id": "SPTF",
        "candidate_family": row["family"],
        "hyperparameters": row["status_evidence"].split("Grid: ", maxsplit=1)[1],
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": True,
        "observations_inspiring_mechanism": "CTPF showed that scalar clipping is not the missing optimizer effect. Source audit showed that MuonH also enforces constant Frobenius norm, an invariant absent from prior Euclidean task-flow models.",
        "novelty_class": "Actual MuonH constant-norm projection as a geodesic state-transition invariant",
        "evaluation_status": "preregistered before StarCoder evaluation",
        "evidence_path": "round36_spherical_projected_flow_starcoder/report.md",
        "notes": "No new adversarial outcome was inspected for this proposal. The exposed panel was already known; the running phase-fiber panel remains sealed.",
    }
    identity = ["round_id", "candidate_id", "evaluation_status"]
    existing = set(map(tuple, ledger[identity].itertuples(index=False, name=None)))
    key = tuple(ledger_row[column] for column in identity)
    if key not in existing:
        ledger = pd.concat([ledger, pd.DataFrame([ledger_row], columns=ledger.columns)], ignore_index=True)
        ledger.to_csv(LEDGER, index=False)


if __name__ == "__main__":
    main()
