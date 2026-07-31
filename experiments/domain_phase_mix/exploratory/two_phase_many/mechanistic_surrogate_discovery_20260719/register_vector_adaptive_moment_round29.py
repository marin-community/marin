# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2"]
# ///
"""Preregister vector adaptive-moment gradient flow before fitting."""

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
        "id": "VAMGF",
        "family": "Vector adaptive-moment gradient flow",
        "relationship_to_prior": (
            "Combines the exact two-dimensional task geometry of blocked NQGF with a materially new coordinatewise "
            "optimizer state. Scalar ASMGF cannot express cross-domain directional preconditioning; NQGF has no "
            "persistent optimizer metric."
        ),
        "materially_new_mechanism": (
            "A coordinatewise Adam/RMSProp second moment persists across the phase boundary. Because broad and rare "
            "gradients are non-collinear, the first phase changes the metric applied to the second phase."
        ),
        "mechanistic_premise": (
            "Adam carries direction-specific gradient-scale estimates. A distribution shift can therefore transiently "
            "suppress parameter directions that were active early and favor directions that were quiet."
        ),
        "governing_equations": (
            "g(z,p)=(1-p)H_b(z-mu_b)+pH_r(z-mu_r); dz/dt=-k g/(sqrt(v)+eps); "
            "dv/dt=kappa(g odot g-v); Y=b+A[(1-q)L_b(z_T)+qL_r(z_T)], A>=0. "
            "The exact ablation is dz/dt=-k g with the same H_b,H_r and response."
        ),
        "latent_state": "Two-dimensional shared capability coordinate z and positive coordinatewise second moment v.",
        "state_transition": (
            "Deterministic piecewise-constant vector RMSProp flow; z and v remain continuous across the phase boundary."
        ),
        "response_link": (
            "One nonnegative amplitude on the same broad/rare quadratic task potential used by the transition, plus an intercept."
        ),
        "additional_degrees_of_freedom": (
            "Task curvature ratio, Hessian anisotropy and angle, representation speed, second-moment memory, epsilon, "
            "evaluation mixture, one response amplitude, intercept, and ridge."
        ),
        "units_and_symmetries": (
            "z and task optima are dimensionless; v has squared-gradient units; normalized time makes k and kappa "
            "dimensionless rates; epsilon has gradient units; A and b have BPB units. Fixed optima and broad gradient "
            "scale remove affine state symmetries; coordinate signs are fixed by the Hessian-angle convention."
        ),
        "single_phase_restriction": (
            "For tied phases the autonomous (z,v) flow composes exactly across an artificial boundary. The same law is "
            "also selected and refit independently on tied observations."
        ),
        "starcoder_signature": (
            "Anisotropic memory should beat the exact vector gradient-flow ablation on both schedules, with a larger "
            "ordering effect under WSD; compatible Hessian orientation and memory regimes should be selected."
        ),
        "catastrophic_optimism_resolution": (
            "Remote policies terminate far from the evaluation compromise and pay the declared quadratic task potential; "
            "the optimizer state cannot grant direct output credit."
        ),
        "response_compression_resolution": (
            "Directional preconditioner mismatch expands terminal-state separation after a phase shift without an output calibrator."
        ),
        "scale_transfer_expectation": (
            "Hessian orientation and dimensionless moment-memory regime should transfer for a fixed optimizer family; "
            "integrated speed may change with tokens per parameter and schedule."
        ),
        "cheapest_falsification": (
            "The exact no-adaptive vector flow wins globally on either StarCoder schedule. If it passes, reject when "
            "nested selection removes adaptive memory, regimes disagree by over 4x, shape RMSE misses by over 5%, or "
            "a raw optimum misses by over 0.15."
        ),
        "status": "active_frozen_round29",
        "status_evidence": "Frozen before VAMGF fitting; no new Delphi heldout or adversarial evaluation was read.",
    }
    registry = registry.loc[~registry["id"].eq("VAMGF")]
    registry = pd.concat([registry, pd.DataFrame([row], columns=registry.columns)], ignore_index=True)
    registry.to_csv(REGISTRY, index=False)

    ledger = pd.read_csv(LEDGER)
    ledger_row = {
        "timestamp": datetime.now(UTC).isoformat(),
        "round_id": "round_29_batch_preregistration",
        "candidate_id": "VAMGF",
        "candidate_family": "Vector adaptive-moment gradient flow",
        "hyperparameters": (
            "geometry curvature={0.25,1,4}, anisotropy={0.5,2,4}, angle={30,75}; "
            "adaptive speed={1,4}, memory={0.5,2,8}, epsilon={0.1,0.3,1}, eval={0.2,0.5,0.8}, "
            "exact-gradient-flow speed={1,4,16}, eval={0.2,0.5,0.8}, "
            "ridge={0.1,1}; exact vector-gradient-flow ablation; 128 RK4 steps/unit time"
        ),
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": True,
        "observations_inspiring_mechanism": (
            "Scalar adaptive memory and noncommuting vector flow failed separately; neither represented a persistent "
            "direction-specific optimizer metric under a domain shift."
        ),
        "novelty_class": "Coordinatewise adaptive preconditioner over non-collinear domain gradients",
        "evaluation_status": "frozen before two-stage StarCoder evaluation",
        "evidence_path": "round29_vector_adaptive_moment_starcoder/report.md",
        "notes": (
            "Stage 1 is global OOF only; if the exact gradient-flow ablation wins on either schedule, no nested or Delphi evaluation is allowed."
        ),
    }
    identity = ["round_id", "candidate_id", "evaluation_status"]
    existing = set(map(tuple, ledger[identity].itertuples(index=False, name=None)))
    if tuple(ledger_row[column] for column in identity) not in existing:
        ledger = pd.concat([ledger, pd.DataFrame([ledger_row], columns=ledger.columns)], ignore_index=True)
        ledger.to_csv(LEDGER, index=False)


if __name__ == "__main__":
    main()
