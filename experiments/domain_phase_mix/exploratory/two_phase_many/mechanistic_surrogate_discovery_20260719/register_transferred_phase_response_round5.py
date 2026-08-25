# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2"]
# ///
"""Reject primed plasticity and preregister cross-scale phase response."""

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
    registry.loc[registry["id"].eq("EAGP"), ["status", "status_evidence"]] = [
        "blocked_before_adversarial",
        "Rejected before adversarial evaluation. Both StarCoder schedules select epsilon=1, the exact no-gating "
        "ablation. Cosine/WSD RMSE is 0.1677/0.1143; leave-region RMSE reaches 0.329/0.195, and the predicted "
        "optima miss the observed WSD asymmetric optimum. The independently refitted tied restriction is also "
        "unstable because the primer is unidentified once the phases are tied.",
    ]
    row = {
        "id": "TPRB",
        "family": "Transferred phase-response basis",
        "relationship_to_prior": "Reopens PMVT only through a materially new cross-scale identification argument. "
        "PMVT estimated a free phase-response head at each scale; TPRB learns a normalized signed response direction "
        "at 300M and permits Delphi to learn only its amplitude and a nonnegative contrast cost.",
        "materially_new_mechanism": "A target-specific terminal-utility gradient is treated as a transferable "
        "direction across model scale, while scale changes only the strength of that response and the cost of finite "
        "phase contrast.",
        "mechanistic_premise": "Moving a fixed aggregate token allocation from the early phase to the terminal phase "
        "perturbs terminal capability along a signed marginal-value direction. Bucket signs reflect which capabilities "
        "benefit from recency; model scale changes response magnitude more readily than it changes those signs.",
        "governing_equations": "a=alpha0*w0+alpha1*w1; d=alpha0*alpha1*(w1-w0); "
        "z_tau(a,d)=sum_f v_f sum_{i in f} d_i/(a_i+tau*p_i), ||v||_2=1; "
        "q_tau(a,d)=sum_i[d_i/(a_i+tau*p_i)]^2; Y_s=F_s(a)+lambda_s*z_tau+chi_s*q_tau, chi_s>=0.",
        "latent_state": "The independently fitted tied-policy capability state F_s(a), a normalized signed terminal-utility "
        "direction v learned at the source scale, and a dimensionless finite-contrast magnitude q.",
        "state_transition": "The terminal state is transported once along the source-identified recency direction by "
        "lambda_s and loses capability through a nonnegative second-order contrast cost chi_s*q.",
        "response_link": "An independently fitted one-phase spine plus one signed transferred response coordinate and "
        "one nonnegative quadratic phase-contrast cost.",
        "additional_degrees_of_freedom": "At the target scale, one signed amplitude lambda_s and one nonnegative "
        "contrast coefficient chi_s. The source direction has one signed coefficient per three predeclared families, "
        "normalized to remove amplitude symmetry. tau and ridge are selected only by source paired-difference CV.",
        "units_and_symmetries": "Weights, d/(a+tau*p), z, and q are dimensionless; lambda and chi have BPB units. "
        "The source vector has unit norm and a deterministic sign convention, removing v-lambda scale and sign "
        "symmetries. Phase reversal flips z and preserves q.",
        "single_phase_restriction": "When w0=w1, d=z=q=0 and the model is exactly F_s(w). F_s is independently fit "
        "on one-phase observations; this is distinct from tying a two-phase coefficient fit.",
        "starcoder_signature": "A response direction identified on the 50/50 cosine surface should predict the sign "
        "and orientation of the 80/20 WSD off-diagonal valley after fitting only one amplitude; the reverse transfer "
        "should behave analogously. Quadratic cost may raise both remote arms but cannot rotate the valley.",
        "catastrophic_optimism_resolution": "The one-phase spine must price aggregate shortage. The nonnegative q term "
        "prevents arbitrarily large unsupported phase contrasts from appearing beneficial, while the signed term is "
        "restricted to a source-identified direction rather than a flexible target head.",
        "response_compression_resolution": "A freely fitted target amplitude can preserve the target scale's measured "
        "phase-response range without relearning a high-dimensional direction from the sparse two-phase panel.",
        "scale_transfer_expectation": "Family-level signs should transfer between 300M and Delphi if the same benchmark "
        "capabilities value recency similarly. Amplitude may grow with optimization progress. Failure is expected for "
        "targets whose matched phase deltas have weak cross-scale correlation.",
        "cheapest_falsification": "Source-direction transfer fails to beat a zero phase correction in grouped Delphi CV, "
        "the fitted amplitude changes sign across folds, the quadratic coefficient collapses while raw optima remain "
        "pathological, or cosine-to-WSD transfer misses the valley orientation.",
        "status": "active_preregistered",
        "status_evidence": "Preregistered before any TPRB fit. Source grids, three-family direction, unit-norm convention, "
        "and two target degrees of freedom are frozen without reading adversarial outcomes.",
    }
    registry = registry.loc[~registry["id"].eq("TPRB")]
    registry = pd.concat([registry, pd.DataFrame([row], columns=registry.columns)], ignore_index=True)

    now = datetime.now(UTC).isoformat()
    rows = [
        {
            "timestamp": now,
            "round_id": "round_4_starcoder_rejection",
            "candidate_id": "EAGP",
            "candidate_family": "Early-allocation-gated plasticity",
            "hyperparameters": "Selected only by StarCoder OOF; both schedules chose epsilon=1",
            "adversarial_outcomes_available_before_proposal": True,
            "adversarial_outcomes_inspected_before_proposal": True,
            "observations_inspiring_mechanism": "WSD early support and historical joint-undercoverage failures.",
            "novelty_class": "Irreversible bucket-local primer gating final-phase plasticity",
            "evaluation_status": "rejected at StarCoder gate; adversarial panel not evaluated",
            "evidence_path": "round4_primed_plasticity_starcoder/report.md",
            "notes": "The exact no-gating ablation won on both schedules, so the new latent state is unsupported.",
        },
        {
            "timestamp": now,
            "round_id": "round_5_preregistration",
            "candidate_id": "TPRB",
            "candidate_family": "Transferred phase-response basis",
            "hyperparameters": "Not selected; tau and source ridge restricted to source paired-CV, target has one signed amplitude and one nonnegative contrast coefficient",
            "adversarial_outcomes_available_before_proposal": True,
            "adversarial_outcomes_inspected_before_proposal": True,
            "observations_inspiring_mechanism": "Matched 300M-to-Delphi phase deltas transfer moderately for Uncheatable but are compressed by free target heads; one-phase outcomes transfer more reliably than two-phase outcomes.",
            "novelty_class": "Cross-scale transfer of a normalized terminal-utility gradient with target-only amplitude identification",
            "evaluation_status": "preregistered; adversarial evaluation forbidden until a later batch freeze",
            "evidence_path": "approach_registry.csv",
            "notes": "This round tests an identification strategy, not another output link. The source direction is frozen before target fitting.",
        },
    ]
    identity = ["round_id", "candidate_id", "evaluation_status"]
    existing = set(map(tuple, ledger[identity].itertuples(index=False, name=None)))
    additions = pd.DataFrame(rows, columns=ledger.columns)
    additions = additions.loc[
        [tuple(value) not in existing for value in additions[identity].itertuples(index=False, name=None)]
    ]
    ledger = pd.concat([ledger, additions], ignore_index=True)
    registry.to_csv(REGISTRY, index=False)
    ledger.to_csv(LEDGER, index=False)
    print(registry.tail(3)[["id", "family", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()
