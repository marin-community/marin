# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pandas>=2.2",
# ]
# ///
"""Preregister power-law gradient-Gram loss flow."""

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
    if not registry["id"].eq("GGLLF").any():
        row = {
            "id": "GGLLF",
            "family": "Gradient-Gram coupled loss flow",
            "relationship_to_prior": (
                "Unlike shared-margin LMCF and capability-state gradient flows, GGLLF evolves the two task excess losses "
                "themselves and constrains cross-task transfer by the exact symmetric gradient-Gram identity."
            ),
            "materially_new_mechanism": (
                "A positive-semidefinite task-gradient Gram matrix couples power-law learning curves at the loss-state level."
            ),
            "mechanistic_premise": (
                "Under gradient flow, dL_i/dt is the mixture-weighted inner product between task-i and sampled-task "
                "gradients. A stable gradient correlation therefore transfers learning between tasks while each gradient "
                "norm decays with its own unresolved loss."
            ),
            "governing_equations": (
                r"u_i=exp(-x_i), ||grad L_i||^2=k_i u_i^(1+zeta), "
                r"<grad L_b,grad L_r>=rho sqrt(k_b k_r)(u_b u_r)^((1+zeta)/2). "
                r"dx_i/dtau=-(1/u_i)dL_i/dtau under mixture p; "
                r"Y=b+A_b u_b(T)+A_r u_r(T), A_b,A_r>=0."
            ),
            "latent_state": "Two dimensionless log excess losses x_b and x_r, initialized at zero.",
            "state_transition": (
                "Autonomous gradient-flow evolution in token or source-derived optimizer time with a fixed symmetric PSD "
                "two-task gradient Gram."
            ),
            "response_link": "A nonnegative BPB-weighted sum of the two terminal excess losses plus an intercept.",
            "additional_degrees_of_freedom": (
                "One shared power-law exponent, gradient correlation, total learning rate, rare-to-broad rate ratio, clock, "
                "and ridge. rho=0 is the exact independent-learning-curve ablation."
            ),
            "units_and_symmetries": (
                "Normalized time, log losses, excess losses, and correlation are dimensionless; rates are inverse time; "
                "response amplitudes have BPB units. Fixed unit initial losses identify rate and amplitude scales."
            ),
            "single_phase_restriction": (
                "A tied mixture composes the same autonomous loss flow across the artificial boundary. The identical law "
                "must also be fit independently on tied-policy outcomes."
            ),
            "starcoder_signature": (
                "A stable positive gradient correlation should improve both schedules over rho=0, retain a near-diagonal "
                "cosine optimum, and permit greater late rare allocation on WSD through state-dependent marginal learning."
            ),
            "catastrophic_optimism_resolution": (
                "A concentrated policy cannot eliminate the omitted task's explicit positive excess loss, and cross-task "
                "credit is bounded by the PSD Gram constraint rho<=1."
            ),
            "response_compression_resolution": (
                "Coupled state-dependent gradient norms separate policies with equal aggregate exposure but different loss "
                "histories without fitting an output calibration layer."
            ),
            "scale_transfer_expectation": (
                "Gradient-correlation sign and power-law exponent should be more transferable than the total rate; the "
                "latter may vary with optimizer progress and tokens per parameter."
            ),
            "cheapest_falsification": (
                "rho>0 must beat rho=0 globally and in >=3/5 folds on both StarCoder schedules, stay within 5% of both "
                "shape frontiers, select stable non-boundary correlation, and locate both raw optima within 0.15."
            ),
            "status": "active_preregistered",
            "status_evidence": "Frozen before any GGLLF fit or StarCoder evaluation.",
        }
        registry = pd.concat([registry, pd.DataFrame([row], columns=registry.columns)], ignore_index=True)
        registry.to_csv(REGISTRY, index=False)

    ledger = pd.read_csv(LEDGER)
    row = {
        "timestamp": datetime.now(UTC).isoformat(),
        "round_id": "round_51_preregistration",
        "candidate_id": "GGLLF",
        "candidate_family": "Gradient-Gram coupled loss flow",
        "hyperparameters": "Frozen clock/rate/power/correlation/rare-scale/ridge grid with rho=0 ablation",
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": True,
        "observations_inspiring_mechanism": (
            "Positive task transfer helped LMCF but shared-margin geometry missed both optima. GGLLF tests the exact "
            "loss-derivative identity instead of another shared representation potential."
        ),
        "novelty_class": "Power-law excess-loss state coupled by a PSD task-gradient Gram",
        "evaluation_status": "preregistered for StarCoder gate; no new adversarial evaluation",
        "evidence_path": "round51_gradient_gram_loss_starcoder",
        "notes": "Running sealed phase-fiber panel remains untouched.",
    }
    identity = ["round_id", "candidate_id", "evaluation_status"]
    existing = set(map(tuple, ledger[identity].itertuples(index=False, name=None)))
    if tuple(row[column] for column in identity) not in existing:
        ledger = pd.concat([ledger, pd.DataFrame([row], columns=ledger.columns)], ignore_index=True)
        ledger.to_csv(LEDGER, index=False)


if __name__ == "__main__":
    main()
