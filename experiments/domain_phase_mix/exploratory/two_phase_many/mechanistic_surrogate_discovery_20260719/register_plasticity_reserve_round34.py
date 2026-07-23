# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2"]
# ///
"""Preregister replenishable specialist plasticity before fitting."""

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
        "id": "RSPR",
        "family": "Replenishable specialist-plasticity reserve",
        "relationship_to_prior": (
            "FSC gates specialist acquisition with accumulated foundation competence, while fast/slow memory and "
            "adaptive-moment routes carry learned or optimizer state. RSPR adds a distinct consumable reserve: broad "
            "updates replenish specialist plasticity and specialist updates spend it."
        ),
        "materially_new_mechanism": (
            "A bounded plasticity reserve is neither competence nor uncertainty. It changes only the acquisition "
            "hazard, has an exact depletion-free ablation, and never enters the response link directly."
        ),
        "mechanistic_premise": (
            "Broad training can build reusable feature capacity that makes subsequent specialist adaptation efficient, "
            "but sustained specialist optimization exhausts that reserve. A short terminal specialist phase can therefore "
            "outperform both a tied policy and a long specialist-heavy phase."
        ),
        "governing_equations": (
            "dg/dt=k_g[(1-p)+rho p](1-g); dr/dt=k_r(1-p)(1-r)-d p r; "
            "ds/dt=k_s p r(1-s); Y=b+A_g(1-g_T)+A_s(1-s_T)+H_b R_b+H_s R_s, all amplitudes nonnegative. "
            "d=0 is the exact nondepleting-reserve ablation."
        ),
        "latent_state": "Foundation competence g, specialist-plasticity reserve r, and specialist competence s, all in [0,1].",
        "state_transition": (
            "Exact piecewise-constant integration: g and r have closed-form linear hazards, and specialist acquisition "
            "uses the exact within-phase integral of r."
        ),
        "response_link": (
            "Nonnegative amplitudes on terminal foundation/specialist unresolved error and literal broad/specialist replay; "
            "the reserve is not an output feature."
        ),
        "additional_degrees_of_freedom": (
            "Three acquisition/recovery rates, rare-to-foundation efficiency, one reserve-depletion rate, four nonnegative "
            "response amplitudes, intercept, and ridge."
        ),
        "units_and_symmetries": (
            "States, weights, and normalized time are dimensionless; rates are per normalized training time; replay is in "
            "simulated epochs; response amplitudes have BPB units. Fixed zero states and unit endpoints remove affine state symmetries."
        ),
        "single_phase_restriction": (
            "A tied policy drives the same autonomous three-state system for unit time, so an artificial phase boundary "
            "cancels exactly. The identical restricted form must also be selected and refit on tied observations."
        ),
        "starcoder_signature": (
            "Positive depletion should beat d=0 on both schedules. WSD should favor broad early and specialist late because "
            "its short terminal phase spends accumulated reserve; cosine should remain nearer tied."
        ),
        "catastrophic_optimism_resolution": (
            "A specialist-heavy policy cannot claim unlimited specialist acquisition after its reserve is exhausted, while "
            "missing foundation and literal replay remain explicit costs."
        ),
        "response_compression_resolution": (
            "Reserve depletion separates frontier schedules with similar aggregate specialist exposure but different timing, "
            "without a post-hoc output expansion."
        ),
        "scale_transfer_expectation": (
            "The recovery-to-depletion ratio is dimensionless and should retain sign/order across schedules; integrated rates "
            "may follow tokens-per-parameter and must be audited before cross-scale use."
        ),
        "cheapest_falsification": (
            "Reject if d=0 wins globally on either StarCoder schedule. If active, require depletion in at least three of five "
            "folds on both schedules, compatible recovery/depletion ratios, nested RMSE within 5% of the frozen shape references, "
            "and raw-optimum distance at most 0.15."
        ),
        "status": "active_frozen_round34",
        "status_evidence": "Frozen before fitting; no new Delphi historical or adversarial outcomes were read.",
    }
    registry = registry.loc[~registry["id"].eq("RSPR")]
    registry = pd.concat([registry, pd.DataFrame([row], columns=registry.columns)], ignore_index=True)
    registry.to_csv(REGISTRY, index=False)

    ledger = pd.read_csv(LEDGER)
    ledger_row = {
        "timestamp": datetime.now(UTC).isoformat(),
        "round_id": "round_34_batch_preregistration",
        "candidate_id": "RSPR",
        "candidate_family": "Replenishable specialist-plasticity reserve",
        "hyperparameters": (
            "foundation_rate={0.5,2,8}, reserve_recovery={0.5,2,8}, specialist_rate={0.5,2,8}, "
            "rare_foundation={0,0.3,1}, depletion={0,0.5,2,8}, ridge={0.1,1}; exact piecewise transition"
        ),
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": True,
        "observations_inspiring_mechanism": (
            "Both StarCoder schedules share similar low-order aggregate/contrast coefficients, while fixed recency laws miss "
            "the WSD short-terminal-phase optimum; a consumable acquisition reserve predicts duration-dependent recency."
        ),
        "novelty_class": "Consumable and replenishable plasticity state, distinct from competence and optimizer moments",
        "evaluation_status": "frozen before two-stage StarCoder evaluation",
        "evidence_path": "round34_plasticity_reserve_starcoder/report.md",
        "notes": "No Delphi historical or adversarial outcomes will be read during the StarCoder gate.",
    }
    identity = ["round_id", "candidate_id", "evaluation_status"]
    existing = set(map(tuple, ledger[identity].itertuples(index=False, name=None)))
    if tuple(ledger_row[column] for column in identity) not in existing:
        ledger = pd.concat([ledger, pd.DataFrame([ledger_row], columns=ledger.columns)], ignore_index=True)
        ledger.to_csv(LEDGER, index=False)


if __name__ == "__main__":
    main()
