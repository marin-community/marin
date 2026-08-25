# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2"]
# ///
"""Preregister bistable representation switching before fitting."""

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
        "id": "BRSF",
        "family": "Bistable representation-switch flow",
        "relationship_to_prior": (
            "AAGF models an LR-temperature activation barrier and NTPGF uses a smooth nonlinear task potential. BRSF "
            "instead introduces two stable representation basins with hysteretic switching under data-composition tilt."
        ),
        "materially_new_mechanism": (
            "A nonconvex latent representation has two stable basins. A curriculum can cross the separating barrier and "
            "retain the new representation after the data mixture changes."
        ),
        "mechanistic_premise": (
            "Feature circuits can exhibit thresholded formation and hysteresis: once a specialist representation is "
            "formed, it need not disappear immediately when broad data returns."
        ),
        "governing_equations": (
            "V(z,p)=z^4/4-a z^2/2-s(2p-1)z; dz/dt=-k dV/dz=k[a z+s(2p-1)-z^3]; "
            "Y=b0+A(z_T-z_eval)^2, A>=0. The exact no-bistability ablation is a=0."
        ),
        "latent_state": "One dimensionless representation coordinate z with broad and specialist basins.",
        "state_transition": "Autonomous gradient flow in a data-tilted quartic potential, continuous across phase boundaries.",
        "response_link": "One nonnegative amplitude on squared terminal distance to a fixed evaluation representation, plus intercept.",
        "additional_degrees_of_freedom": (
            "Barrier strength, data tilt, transition speed, evaluation coordinate, one response amplitude, intercept, and ridge."
        ),
        "units_and_symmetries": (
            "z,p and potential coefficients are dimensionless after fixing the quartic coefficient to one; k is per normalized time; "
            "A and b0 have BPB units. Fixing initial z=-1 and rare tilt positive removes sign symmetry."
        ),
        "single_phase_restriction": (
            "A tied mixture produces one autonomous flow for total duration one; the same quartic law can be independently refit on tied policies."
        ),
        "starcoder_signature": (
            "Active bistability should create a threshold ridge and asymmetric WSD basin switch while the cosine optimum can remain nearer the diagonal."
        ),
        "catastrophic_optimism_resolution": (
            "Remote policies left in the wrong representation basin pay a bounded terminal-distance cost rather than receiving smooth extrapolated credit."
        ),
        "response_compression_resolution": (
            "Crossing the barrier yields a finite response jump among frontier policies instead of compressing all terminal states."
        ),
        "scale_transfer_expectation": (
            "Dimensionless barrier-to-tilt ratio should transfer if feature formation is critical; speed may vary with tokens per parameter."
        ),
        "cheapest_falsification": (
            "Reject if a=0 wins globally on either StarCoder schedule. If active, require nested barrier selection, compatible "
            "barrier-to-tilt ratios, shape RMSE within 5% of frozen references, and raw optimum distance at most 0.15."
        ),
        "status": "active_frozen_round31",
        "status_evidence": "Frozen before BRSF fitting; no new Delphi heldout or adversarial evaluation was read.",
    }
    registry = registry.loc[~registry["id"].eq("BRSF")]
    registry = pd.concat([registry, pd.DataFrame([row], columns=registry.columns)], ignore_index=True)
    registry.to_csv(REGISTRY, index=False)

    ledger = pd.read_csv(LEDGER)
    ledger_row = {
        "timestamp": datetime.now(UTC).isoformat(),
        "round_id": "round_31_batch_preregistration",
        "candidate_id": "BRSF",
        "candidate_family": "Bistable representation-switch flow",
        "hyperparameters": (
            "barrier={0,0.5,1,2}, tilt={0.5,1,2,4}, speed={0.5,2,8}, "
            "evaluation={-0.5,0,0.5}, ridge={0.1,1}; 256 RK4 steps/unit time"
        ),
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": True,
        "observations_inspiring_mechanism": (
            "Smooth relaxation and optimizer-memory laws fail to produce a stable WSD phase-order effect; a genuine representation transition is underexplored."
        ),
        "novelty_class": "Nonconvex bistable representation state with hysteresis",
        "evaluation_status": "frozen before two-stage StarCoder evaluation",
        "evidence_path": "round31_bistable_representation_starcoder/report.md",
        "notes": "Derived independently of Round 30 results; no adversarial outcomes will be read during the StarCoder gate.",
    }
    identity = ["round_id", "candidate_id", "evaluation_status"]
    existing = set(map(tuple, ledger[identity].itertuples(index=False, name=None)))
    if tuple(ledger_row[column] for column in identity) not in existing:
        ledger = pd.concat([ledger, pd.DataFrame([ledger_row], columns=ledger.columns)], ignore_index=True)
        ledger.to_csv(LEDGER, index=False)


if __name__ == "__main__":
    main()
