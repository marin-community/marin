# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2"]
# ///
"""Preregister deep-linear shared-feature multitask flow before fitting."""

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
        "id": "DLSF",
        "family": "Deep-linear shared-feature multitask flow",
        "relationship_to_prior": (
            "FPCGF represents one general factor and one specialist factor, so broad data directly suppresses the only "
            "specialist coordinate. DLSF introduces two task-specific heads coupled only through a learned shared feature."
        ),
        "materially_new_mechanism": (
            "Broad and rare tasks have separate terminal heads but share a learned representation factor. Early broad "
            "training can grow the shared factor, making late rare-head adaptation faster without forcing the two heads to coincide."
        ),
        "mechanistic_premise": (
            "In a deep network, early data can train reusable features while the final phase rapidly adjusts task-specific "
            "readout directions. A two-layer deep-linear model is the minimal exact gradient-flow realization."
        ),
        "governing_equations": (
            "L_b=(r h_b-1)^2/2; L_r=(r h_r-1)^2/2; dr/dt=-gamma[(1-p)d_r L_b+p d_r L_r+lambda r]; "
            "dh_b/dt=-k[(1-p)d_hb L_b+lambda h_b]; dh_r/dt=-k[p d_hr L_r+lambda h_r]; "
            "Y=b0+A[(1-q)L_b+qL_r], A>=0. The exact frozen-feature ablation is gamma=0."
        ),
        "latent_state": "One shared feature amplitude r and two task-specific head amplitudes h_b and h_r.",
        "state_transition": "Exact gradient flow of a two-layer scalar multitask network with L2 weight decay under each phase mixture.",
        "response_link": "One nonnegative amplitude on the evaluation-weighted broad/rare task loss, plus intercept.",
        "additional_degrees_of_freedom": (
            "Shared-feature rate, head rate, weight decay, evaluation mixture, one response amplitude, intercept, and ridge."
        ),
        "units_and_symmetries": (
            "r and heads are dimensionless; fixed equal initialization and common weight decay break the deep-linear rescaling symmetry; "
            "rates are per normalized training time; A and b0 have BPB units."
        ),
        "single_phase_restriction": (
            "A tied mixture gives one autonomous three-state gradient flow for unit duration; the identical restricted law can be refit on tied policies."
        ),
        "starcoder_signature": (
            "A learned shared feature should beat gamma=0 on both schedules. WSD should favor broad feature learning early "
            "and rare-head adaptation late, while cosine can remain closer to phase-tied."
        ),
        "catastrophic_optimism_resolution": (
            "A policy must fit both task products; concentration in one head leaves explicit error in the other rather than receiving free pooled exposure credit."
        ),
        "response_compression_resolution": (
            "Fast task heads can separate frontier policies with similar aggregate weights but different final-phase allocation."
        ),
        "scale_transfer_expectation": (
            "The shared-to-head rate ratio and dimensionless decay should reflect feature-versus-readout learning and transfer more naturally than a free phase multiplier."
        ),
        "cheapest_falsification": (
            "Reject if gamma=0 wins globally on either StarCoder schedule. If active, require stable shared/head rate ratios, "
            "shape RMSE within 5% of frozen references, and raw optimum distance at most 0.15."
        ),
        "status": "active_frozen_round33",
        "status_evidence": "Frozen before DLSF fitting; no new Delphi heldout or adversarial evaluation was read.",
    }
    registry = registry.loc[~registry["id"].eq("DLSF")]
    registry = pd.concat([registry, pd.DataFrame([row], columns=registry.columns)], ignore_index=True)
    registry.to_csv(REGISTRY, index=False)

    ledger = pd.read_csv(LEDGER)
    ledger_row = {
        "timestamp": datetime.now(UTC).isoformat(),
        "round_id": "round_33_batch_preregistration",
        "candidate_id": "DLSF",
        "candidate_family": "Deep-linear shared-feature multitask flow",
        "hyperparameters": (
            "shared_rate={0,0.25,1,4}, head_rate={1,4,16}, weight_decay={0,0.1,1}, "
            "evaluation={0.2,0.5,0.8}, ridge={0.1,1}; fixed state initialization 0.1; 256 RK4 steps/unit time"
        ),
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": True,
        "observations_inspiring_mechanism": (
            "Single-specialist-state mechanisms repeatedly predict tied WSD optima; separate task heads with a shared feature are the minimal missing state."
        ),
        "novelty_class": "Exact deep-linear multitask gradient flow with shared representation and task-specific heads",
        "evaluation_status": "frozen before two-stage StarCoder evaluation",
        "evidence_path": "round33_deep_linear_multitask_starcoder/report.md",
        "notes": "No adversarial outcomes will be read during the StarCoder gate.",
    }
    identity = ["round_id", "candidate_id", "evaluation_status"]
    existing = set(map(tuple, ledger[identity].itertuples(index=False, name=None)))
    if tuple(ledger_row[column] for column in identity) not in existing:
        ledger = pd.concat([ledger, pd.DataFrame([ledger_row], columns=ledger.columns)], ignore_index=True)
        ledger.to_csv(LEDGER, index=False)


if __name__ == "__main__":
    main()
