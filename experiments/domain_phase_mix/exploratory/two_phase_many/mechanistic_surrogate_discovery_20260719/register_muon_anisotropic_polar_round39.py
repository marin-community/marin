# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2"]
# ///
"""Preregister anisotropic matrix-polar flow before StarCoder evaluation."""

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
        "id": "MAPTF",
        "family": "Muon anisotropic polar task flow",
        "relationship_to_prior": "Reopens round-38 MPMTF because its isotropic target geometry confined every fitted trajectory to conformal matrices, making polar and vector-normalized updates identical. Domain-specific second-moment matrices are a new physical input invariant, not a response correction.",
        "materially_new_mechanism": "Broad and rare domains have equal-trace but differently oriented anisotropic input covariances, so mixed matrix gradients have unequal singular values that Muon's polar map explicitly equalizes.",
        "mechanistic_premise": "Linear-layer gradients are products of representation error and input covariance. Domain shifts change that covariance; Muon removes singular-value scale but preserves singular directions, creating phase-order effects unavailable to scalar clipping or isotropic task flow.",
        "governing_equations": "L_j(W)=tr[(W-T_j)C_j(W-T_j)^T]/2; G_p=(1-p)(W-T_B)C_B+r p(W-T_R)C_R; Phi_polar(G)=UV^T/sqrt(2); dW/dtau=-k[Phi-<W,Phi>_F W], ||W||_F=1; Y=b+A[(1-q)L_B(W_T)+q r L_R(W_T)], A>=0.",
        "latent_state": "One dimensionless 2x2 constant-norm representation W. Each data domain contributes an equal-trace SPD input-covariance matrix C_j and an equal-norm representation target T_j.",
        "state_transition": "Autonomous polar-matrix gradient flow in the corrected optimizer-time phase clock. C_j=I recovers round 38; Euclidean and vector-normalized gradient rules are exact update-geometry ablations.",
        "response_link": "One nonnegative BPB amplitude on terminal covariance-weighted task debt plus an intercept.",
        "additional_degrees_of_freedom": "One dimensionless input anisotropy beyond round 38; task angle, rare curvature, relaxation, evaluation mixture, update-rule ablations, and ridge use a frozen finite grid.",
        "units_and_symmetries": "W, T, C after unit-trace normalization, p, and q are dimensionless; k is inverse normalized optimizer time; A and b have BPB units. Unit trace fixes covariance scale, and fixed initialization removes global rotation symmetry.",
        "single_phase_restriction": "Autonomy gives an exact phase-tied semigroup in the continuous law. The identical restricted equation can be fit independently on tied data.",
        "starcoder_signature": "A nonunit covariance anisotropy and polar update must be selected on both schedules, beat isotropic, Euclidean, and vector-normalized ablations, retain the cosine near-diagonal optimum, and recover WSD late rare enrichment.",
        "catastrophic_optimism_resolution": "The bounded representation state and covariance-aware terminal debt price directions neglected by an extreme mixture; singular-value equalization prevents one high-variance direction from masquerading as broad capability.",
        "response_compression_resolution": "Different ordered covariance products can expand terminal debt variation among frontier policies without changing the output link.",
        "scale_transfer_expectation": "Unit-trace covariance anisotropy is a dimensionless data property and the polar map is optimizer-defined. Task angle should transfer more plausibly than a fitted late multiplier; relaxation can vary with optimizer progress.",
        "cheapest_falsification": "Reject before multi-swarm work unless polar with nonunit anisotropy beats all exact ablations globally and in at least three folds on both StarCoder schedules, clears both corrected shape references, and locates both raw optima within 0.15.",
        "status": "active_round39_preregistered",
        "status_evidence": "Frozen grid: task angle {30,60,90}; rare curvature {0.5,1,2}; input anisotropy {0.25,0.5,1,2,4}; relaxation {1,2,4,8}; evaluation rare weight {0.2,0.5,0.8}; update rule {euclidean,normalized,polar}; ridge {0,0.1,1}. No historical, exposed-adversarial, or sealed-confirmation outcome will be read unless the StarCoder gate passes.",
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
        "round_id": "round_39_muon_anisotropic_polar",
        "candidate_id": row["id"],
        "candidate_family": row["family"],
        "hyperparameters": row["status_evidence"].split("Frozen grid: ", maxsplit=1)[1],
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": True,
        "observations_inspiring_mechanism": "Round 38's StarCoder audit showed polar and normalized trajectories were exactly equal under isotropic conformal geometry. This algebraic non-activation, not an adversarial residual, motivated the input-covariance state.",
        "novelty_class": "Domain-specific anisotropic input covariance under Muon polar flow",
        "evaluation_status": "preregistered before StarCoder evaluation",
        "evidence_path": "round39_muon_anisotropic_polar_starcoder/report.md",
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
