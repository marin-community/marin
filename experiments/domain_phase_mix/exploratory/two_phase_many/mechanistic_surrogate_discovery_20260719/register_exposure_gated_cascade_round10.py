# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2"]
# ///
"""Register the frozen exposure-gated competence-cascade candidate."""

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
    ledger = pd.read_csv(LEDGER)
    row = {
        "id": "EGCC",
        "family": "Exposure-gated competence cascade",
        "relationship_to_prior": "Reopens the phase-0 primer and fast/slow routes only through a materially new "
        "boundary-free transition. EAGP froze late efficiency from phase-0 exposure and failed the tied restriction; "
        "fast/slow consolidation updated the slow state without requiring contemporaneous bucket exposure. EGCC is "
        "a continuous triangular acquisition cascade whose second state advances only when both current exposure and "
        "the first state are present.",
        "materially_new_mechanism": "Useful task competence requires two sequential exposure-dependent acquisitions: "
        "feature formation followed by exposure-gated conversion of those features into deployable competence.",
        "mechanistic_premise": "Early examples can build reusable features, but those features become useful for a "
        "bucket only when subsequent examples from that bucket train the corresponding readout or routing. The same "
        "data stream supplies both stages, making curriculum order matter without privileging an artificial phase boundary.",
        "governing_equations": "For normalized time t and bucket mass w_i(t): df_i/dt=k_f w_i(1-f_i), "
        "dc_i/dt=k_c w_i f_i(1-c_i). Y=b+sum_i A_i log[(delta+c_i^prop)/(delta+c_i)] "
        "+sum_g H_g(R_g-R_g^prop), with A_i,H_g>=0 and literal R_g=sum_{i in g}(E_i-1)_+.",
        "latent_state": "f_i in [0,1] is feature readiness and c_i in [0,1] is deployable competence for bucket i.",
        "state_transition": "Piecewise-constant policies admit an exact triangular update. Feature readiness follows "
        "first-order acquisition; competence survival is multiplied by exp[-k_c w_i integral f_i dt]. Both states "
        "compose exactly across arbitrary subdivisions of a tied schedule.",
        "response_link": "A nonnegative log competence-debt response relative to the proportional state plus literal "
        "physical replay harm. There is no output calibration or candidate-specific correction.",
        "additional_degrees_of_freedom": "Two global dimensionless rates k_f and k_c, one response offset delta, and "
        "ridge-selected nonnegative bucket/family BPB amplitudes. The zero-replay ablation is nested by H_g=0.",
        "units_and_symmetries": "Time, rates-times-time, f, c, epochs, and log debts are dimensionless; b, A, and H "
        "have BPB units. Fixing the training horizon to one removes time-rate scale symmetry. Swapping f and c is not "
        "a symmetry because only f gates acquisition of c.",
        "single_phase_restriction": "For w0=w1 the exact semigroup equals one uninterrupted constant-mixture update; "
        "the same restricted form is also refit independently on tied rows.",
        "starcoder_signature": "Early rare data builds rare features and late rare data converts them into competence, "
        "so the valley may move off diagonal. WSD 80/20 should show stronger late-placement leverage than cosine 50/50 "
        "without changing the tied curve under an artificial split.",
        "catastrophic_optimism_resolution": "A bucket receiving only a late burst cannot instantly achieve full competence "
        "because its feature state must be built first; concentration also leaves other bucket competences unresolved.",
        "response_compression_resolution": "The multiplicative conversion gate creates state-dependent curvature from "
        "input dynamics rather than an output rescaling, potentially preserving frontier response range.",
        "scale_transfer_expectation": "Dimensionless rates should vary monotonically with total optimization progress, "
        "while the triangular transition and sign constraints should transfer across schedules and swarms.",
        "cheapest_falsification": "Either StarCoder surface selects instantaneous feature/readout acquisition at a grid "
        "boundary, leave-region-out RMSE fails to improve over existing shape baselines, the off-diagonal optimum is "
        "misplaced, or the tied semigroup/restriction audit fails.",
        "status": "active_preregistered",
        "status_evidence": "Frozen before fitting EGCC. The adversarial panel has been inspected diagnostically, but no "
        "EGCC prediction or parameter is evaluated there unless it first passes algebraic, StarCoder, multi-swarm, and "
        "historical gates.",
    }
    registry = registry.loc[~registry["id"].eq("EGCC")]
    registry = pd.concat([registry, pd.DataFrame([row], columns=registry.columns)], ignore_index=True)

    now = datetime.now(UTC).isoformat()
    ledger_row = {
        "timestamp": now,
        "round_id": "round_10_preregistration",
        "candidate_id": "EGCC",
        "candidate_family": "Exposure-gated competence cascade",
        "hyperparameters": "Frozen k_feature,k_conversion={0.25,0.5,1,2,4,8,16}; offset={0.03,0.1,0.3,1}; ridge={0.1,1,10}; literal replay on/off",
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": True,
        "observations_inspiring_mechanism": "Existing phase-0 gates violate a clean tied restriction; fast/slow consolidation does not require contemporaneous target exposure; phase effects remain smooth but weakly identified.",
        "novelty_class": "Boundary-free triangular feature-to-competence acquisition cascade",
        "evaluation_status": "preregistered for algebraic and StarCoder gates; no EGCC adversarial evaluation",
        "evidence_path": "approach_registry.csv",
        "notes": "The exact tied semigroup and zero-replay nested ablation are mandatory. Rates must not be retained at unsupported boundaries.",
    }
    identity = ["round_id", "candidate_id", "evaluation_status"]
    existing = set(map(tuple, ledger[identity].itertuples(index=False, name=None)))
    key = tuple(ledger_row[column] for column in identity)
    if key not in existing:
        ledger = pd.concat([ledger, pd.DataFrame([ledger_row], columns=ledger.columns)], ignore_index=True)

    registry.to_csv(REGISTRY, index=False)
    ledger.to_csv(LEDGER, index=False)
    print(registry.tail(1)[["id", "family", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()
