# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2"]
# ///
"""Preregister the round-46 curvature-spectrum mechanism before evaluation."""

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
        "id": "CSNRF",
        "family": "Curvature-spectrum noncommuting residual flow",
        "relationship_to_prior": (
            "Reopens NQGF only through a materially new latent invariant: NQGF represented one curvature mode. "
            "CSNRF carries a fixed quadrature over a broad Hessian spectrum, so slow and fast residual modes "
            "coexist and cannot be replaced by retuning one relaxation rate."
        ),
        "materially_new_mechanism": (
            "A distribution of learning time constants induced by the network Hessian spectrum."
        ),
        "mechanistic_premise": (
            "Neural-network loss is a superposition of curvature modes. Fast modes saturate early while slow modes "
            "retain unresolved error; domain Hessians rotate these modes differently, so ordered phases apply "
            "noncommuting contractions at every timescale."
        ),
        "governing_equations": (
            "lambda_m in {s^-1/2,1,s^1/2}; pi_m proportional to lambda_m^nu; "
            "H_m(p)=lambda_m[(1-p)H_B+c p H_R]; dz_m/dtau=-k H_m(p)z_m; "
            "Y=b+A sum_m pi_m z_m(T)^T H_eval z_m(T), A>=0. s=1 is the exact single-mode ablation."
        ),
        "latent_state": ("Three two-dimensional unresolved-error vectors, one at each fixed geometric curvature node."),
        "state_transition": (
            "Exact path-ordered linear contraction under phase-specific mixtures; each phase uses a matrix exponential."
        ),
        "response_link": (
            "One nonnegative BPB amplitude on the spectral average of terminal evaluation-weighted residual energy, plus an intercept."
        ),
        "additional_degrees_of_freedom": (
            "Spectral span s and density tilt nu beyond single-mode NQGF. Angle, anisotropy, total relaxation, "
            "rare curvature, evaluation mix, and ridge use frozen finite grids."
        ),
        "units_and_symmetries": (
            "Residuals, unit-trace Hessians, weights, normalized time, spectral rates, span, and tilt are dimensionless; "
            "A and b have BPB units. Unit geometric-mean rate and fixed broad eigenbasis remove rate and rotation symmetries."
        ),
        "single_phase_restriction": (
            "A tied policy applies one autonomous contraction across the full schedule; the same spectral law can be refit independently on tied policies."
        ),
        "starcoder_signature": (
            "The spectrum must preserve a slow unresolved tail while fast modes create the Nike-swoosh valley; it must beat the exact single-mode law on both schedules and move the WSD optimum late without moving the cosine optimum off its diagonal valley."
        ),
        "catastrophic_optimism_resolution": (
            "Concentrated policies cannot erase slow residual modes merely by saturating the fastest mode, so unsupported corners retain explicit debt."
        ),
        "response_compression_resolution": (
            "Policies that look equivalent after fast-mode saturation remain separated by their slow-mode residual energy, widening the physical response range."
        ),
        "scale_transfer_expectation": (
            "Dimensionless spectral span and density tilt are architecture properties and should be comparable across schedules and related scales; total relaxation may vary with optimizer progress and tokens per parameter."
        ),
        "cheapest_falsification": (
            "Reject unless the three-node spectrum beats s=1 globally and in at least three folds on both StarCoder schedules, clears both shape references, selects stable non-boundary spectral parameters, and places both raw optima within 0.15."
        ),
        "status": "active_preregistered",
        "status_evidence": "Frozen before any round-46 evaluation; no new adversarial outcome inspected.",
    }
    if not registry["id"].eq(row["id"]).any():
        registry = pd.concat([registry, pd.DataFrame([row], columns=registry.columns)], ignore_index=True)
        registry.to_csv(REGISTRY, index=False)

    ledger = pd.read_csv(LEDGER)
    ledger_row = {
        "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
        "round_id": "round_46_spectral_portfolio_freeze",
        "candidate_id": row["id"],
        "candidate_family": row["family"],
        "hyperparameters": "Frozen three-node geometric spectrum, span/tilt and task-geometry grids, exact s=1 ablation, and immutable StarCoder gate",
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": True,
        "observations_inspiring_mechanism": (
            "Prior scalar-rate and finite-step mechanisms compressed both surfaces; the new hypothesis is a persistent distribution of curvature time constants, not a residual correction."
        ),
        "novelty_class": "Hessian-spectrum latent-state distribution",
        "evaluation_status": "preregistered_before_evaluation",
        "evidence_path": "approach_registry.csv#CSNRF",
        "notes": "No round-46 target values were read before freezing this row.",
    }
    identity = ["round_id", "candidate_id", "evaluation_status"]
    existing = set(map(tuple, ledger[identity].itertuples(index=False, name=None)))
    if tuple(ledger_row[key] for key in identity) not in existing:
        ledger = pd.concat([ledger, pd.DataFrame([ledger_row], columns=ledger.columns)], ignore_index=True)
        ledger.to_csv(LEDGER, index=False)


if __name__ == "__main__":
    main()
