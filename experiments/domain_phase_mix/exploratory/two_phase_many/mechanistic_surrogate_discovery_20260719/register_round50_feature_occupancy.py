# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pandas>=2.2",
# ]
# ///
"""Preregister finite feature-occupancy Markov flow."""

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
    if not registry["id"].eq("FOMF").any():
        row = {
            "id": "FOMF",
            "family": "Finite feature-occupancy Markov flow",
            "relationship_to_prior": (
                "Materially differs from the replicator-capacity HRC and relaxing-capacity prior AG. Those continuously "
                "reweight an already allocated simplex. FOMF includes an explicit unallocated feature state, irreversible "
                "feature acquisition, and separately controlled cross-task overwrite."
            ),
            "materially_new_mechanism": (
                "A three-state continuous-time Markov population U/B/R that distinguishes acquiring unused representation "
                "from overwriting an already specialized feature."
            ),
            "mechanistic_premise": (
                "A finite population of feature slots begins unallocated. Broad or rare examples recruit unused slots; "
                "later examples may retune existing slots more slowly. Early evidence can therefore remain useful without "
                "requiring a scalar retention multiplier."
            ),
            "governing_equations": (
                r"du/dtau=-a[(1-p)+r p]u; db/dtau=a(1-p)u+h(1-p)r_state-h r p b; "
                r"dr_state/dtau=a r p u+h r p b-h(1-p)r_state. Unresolved broad mass is u+r_state and "
                r"unresolved rare mass is u+b; Y=b0+A_b(u+r_state)+A_r(u+b), A_b,A_r>=0."
            ),
            "latent_state": "Fractions of unallocated, broad-specialized, and rare-specialized feature slots, summing to one.",
            "state_transition": (
                "Exact linear CTMC evolution within each constant-mixture phase. Acquisition leaves the unallocated state; "
                "overwrite transfers mass between specializations."
            ),
            "response_link": "Nonnegative BPB amplitudes on unresolved broad and rare feature mass plus an intercept.",
            "additional_degrees_of_freedom": (
                "Acquisition rate, overwrite rate, rare-to-broad acquisition ratio, clock, and ridge. The independent "
                "infinite-capacity acquisition model is the exact no-competition ablation."
            ),
            "units_and_symmetries": (
                "State fractions and time are dimensionless; rates are inverse time; response amplitudes have BPB units. "
                "Fixed U/B/R labels, unit total mass, and nonnegative amplitudes remove scale and permutation symmetries."
            ),
            "single_phase_restriction": (
                "A tied policy composes the same CTMC for the full horizon exactly. The identical finite-occupancy form must "
                "also be refit independently on tied policies."
            ),
            "starcoder_signature": (
                "Finite occupancy should preserve early rare slots on cosine while allowing slower late overwrite to favor "
                "rare specialization on WSD. It must beat independent infinite-capacity acquisition on both schedules."
            ),
            "catastrophic_optimism_resolution": (
                "A concentrated policy can recruit only finite unallocated capacity and must explicitly overwrite useful "
                "features to specialize further, preventing unlimited additive credits."
            ),
            "response_compression_resolution": (
                "Policies with equal aggregate exposure but different acquisition/overwrite histories retain different "
                "unresolved feature masses, expanding phase-fiber response without calibration."
            ),
            "scale_transfer_expectation": (
                "The ordering acquisition faster than overwrite should transfer; rates may scale with training horizon, "
                "while the rare acquisition ratio should remain comparable."
            ),
            "cheapest_falsification": (
                "Finite occupancy must beat independent acquisition globally and in >=3/5 folds on both StarCoder surfaces, "
                "remain within 5% of both shape frontiers, select non-boundary overwrite, and locate both raw optima within 0.15."
            ),
            "status": "active_preregistered",
            "status_evidence": "Frozen before any FOMF fit or StarCoder evaluation.",
        }
        registry = pd.concat([registry, pd.DataFrame([row], columns=registry.columns)], ignore_index=True)
        registry.to_csv(REGISTRY, index=False)

    ledger = pd.read_csv(LEDGER)
    row = {
        "timestamp": datetime.now(UTC).isoformat(),
        "round_id": "round_50_preregistration",
        "candidate_id": "FOMF",
        "candidate_family": "Finite feature-occupancy Markov flow",
        "hyperparameters": "Frozen clock/acquisition/overwrite/rare-scale/ridge grid with independent-acquisition ablation",
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": True,
        "observations_inspiring_mechanism": (
            "Round 48 shows no value for deeper memory, while multiple state laws erase early rare allocation. FOMF tests "
            "whether explicit unallocated capacity and slower overwrite, rather than retention depth, preserve primacy."
        ),
        "novelty_class": "Finite U/B/R feature-slot CTMC with separate acquisition and overwrite",
        "evaluation_status": "preregistered for StarCoder gate; no new adversarial evaluation",
        "evidence_path": "round50_feature_occupancy_starcoder",
        "notes": "Running sealed phase-fiber panel remains untouched.",
    }
    identity = ["round_id", "candidate_id", "evaluation_status"]
    existing = set(map(tuple, ledger[identity].itertuples(index=False, name=None)))
    if tuple(row[column] for column in identity) not in existing:
        ledger = pd.concat([ledger, pd.DataFrame([row], columns=ledger.columns)], ignore_index=True)
        ledger.to_csv(LEDGER, index=False)


if __name__ == "__main__":
    main()
