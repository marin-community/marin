# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2"]
# ///
"""Preregister deep matrix-factorization dynamics before evaluation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
REGISTRY = OUTPUT_ROOT / "approach_registry.csv"
LEDGER = OUTPUT_ROOT / "data_use_ledger.csv"


def main() -> None:
    registry = pd.read_csv(REGISTRY)
    row = {
        "id": "DMFSB",
        "family": "Deep matrix-factorization spectral-bias flow",
        "relationship_to_prior": (
            "Distinct from scalar DLSF and direct-matrix NQGF/MPMTF. DLSF had one shared scalar factor and two scalar heads; direct-matrix routes evolved W itself. DMFSB evolves two matrix factors whose product is the representation, introducing the balanced-factor invariant and singular-mode learning delay."
        ),
        "materially_new_mechanism": ("Implicit low-rank spectral bias from a bilinear representation W=U V^T."),
        "mechanistic_premise": (
            "Deep linear and locally linearized neural networks learn singular modes at rates that depend on their current factor amplitudes. Early data can establish a shared singular direction, after which late data learns a compatible specialist direction faster; incompatible targets compete through finite factor rank."
        ),
        "governing_equations": (
            "W=UV^T; G(p,W)=(1-p)(W-T_B)+c p(W-T_R); dU/dtau=-k G V; dV/dtau=-k G^T U; Y=b+A[(1-q)||W-T_B||_F^2+q||W-T_R||_F^2]/2, A>=0. Direct dW/dtau=-kG is the exact no-factorization ablation."
        ),
        "latent_state": ("Two balanced 2x2 representation factors U and V; their product is the learned task matrix."),
        "state_transition": (
            "Autonomous bilinear gradient flow through each phase, with both factors continuous across the phase boundary."
        ),
        "response_link": (
            "One nonnegative BPB amplitude on terminal evaluation-weighted matrix task debt plus an intercept."
        ),
        "additional_degrees_of_freedom": (
            "No fitted factor depth or rank. Task angle, total relaxation, rare curvature, evaluation mix, and ridge use frozen finite grids; balanced initialization is fixed."
        ),
        "units_and_symmetries": (
            "Factors, target matrices, weights, and normalized time are dimensionless; A and b have BPB units. Equal positive diagonal initialization fixes U/V rescaling, sign, permutation, and basis symmetries."
        ),
        "single_phase_restriction": (
            "A tied mixture applies one autonomous factor flow for the whole schedule; the identical restricted law can be fit independently on tied observations."
        ),
        "starcoder_signature": (
            "Factorization must beat direct-W flow on both schedules, keep cosine near the tied valley, and permit WSD late-code enrichment by reusing an early shared singular direction."
        ),
        "catastrophic_optimism_resolution": (
            "A concentrated policy cannot claim independent task capabilities unless the required singular directions have actually grown in both factors; unresolved matrix debt remains explicit."
        ),
        "response_compression_resolution": (
            "Bilinear takeoff separates policies around singular-mode activation and preserves large terminal debt when a mode remains unformed."
        ),
        "scale_transfer_expectation": (
            "Balanced-factor spectral bias is architecture-generic; dimensionless task overlap and rare curvature should transfer, while total relaxation may scale with optimizer progress and tokens per parameter."
        ),
        "cheapest_falsification": (
            "Reject unless factorized flow beats direct-W globally and in >=3/5 folds on both StarCoder schedules, clears both shape references, has stable non-boundary relaxation, and places both raw optima within 0.15."
        ),
        "status": "active_preregistered",
        "status_evidence": "Frozen before any round-47 evaluation; no new adversarial outcome inspected.",
    }
    if not registry["id"].eq(row["id"]).any():
        registry = pd.concat([registry, pd.DataFrame([row], columns=registry.columns)], ignore_index=True)
        registry.to_csv(REGISTRY, index=False)
    ledger = pd.read_csv(LEDGER)
    ledger_row = {
        "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
        "round_id": "round_47_matrix_factorization_freeze",
        "candidate_id": row["id"],
        "candidate_family": row["family"],
        "hyperparameters": "Frozen rank-2 balanced factorization, task-geometry and clock grids, direct-W ablation, and immutable StarCoder gate",
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": True,
        "observations_inspiring_mechanism": "Scalar and direct-matrix flows failed both schedules; this route introduces factorized singular-mode dynamics rather than another scalar clock.",
        "novelty_class": "Bilinear matrix-factorization latent state",
        "evaluation_status": "preregistered_before_evaluation",
        "evidence_path": "approach_registry.csv#DMFSB",
        "notes": "No round-47 target values were read before freezing this row.",
    }
    identity = ["round_id", "candidate_id", "evaluation_status"]
    existing = set(map(tuple, ledger[identity].itertuples(index=False, name=None)))
    if tuple(ledger_row[key] for key in identity) not in existing:
        ledger = pd.concat([ledger, pd.DataFrame([ledger_row], columns=ledger.columns)], ignore_index=True)
        ledger.to_csv(LEDGER, index=False)


if __name__ == "__main__":
    main()
