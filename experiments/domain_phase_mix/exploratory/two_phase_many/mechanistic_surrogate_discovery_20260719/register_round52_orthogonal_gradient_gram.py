# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pandas>=2.2",
# ]
# ///
"""Preregister aggregate-orthogonal gradient-Gram transport."""

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
    if not registry["id"].eq("OGGTR").any():
        row = {
            "id": "OGGTR",
            "family": "Orthogonal gradient-Gram transport",
            "relationship_to_prior": (
                "Reopens blocked GGLLF with a materially new identification strategy: a physical aggregate shortage/replay "
                "spine explains total exposure, while the gradient-Gram state is differenced against its exact tied-policy "
                "counterfactual and can explain only phase order. PMVT used an algebraic marginal-value direction rather than "
                "a constrained loss-state transition."
            ),
            "materially_new_mechanism": (
                "Counterfactual orthogonalization of a PSD gradient-Gram loss flow around the policy's clock-matched tied path."
            ),
            "mechanistic_premise": (
                "Finite-data shortage and repetition determine the aggregate learning curve. Conditional on that aggregate, "
                "noncommuting task-gradient transfer changes terminal excess losses according to the exact gradient-flow identity."
            ),
            "governing_equations": (
                r"a=alpha_0 w_0+alpha_1 w_1; F(a)=positive shortage plus literal replay spine. "
                r"u(w)=GGLLF terminal excess loss; Delta u=u(w_0,w_1)-u(a_tau,a_tau), where a_tau uses the selected "
                r"dynamics clock. Y=F(a)+A_b Delta u_b+A_r Delta u_r, A_b,A_r>=0."
            ),
            "latent_state": "Two log excess losses under a PSD task-gradient Gram, plus no additional aggregate latent state.",
            "state_transition": (
                "GGLLF power-law loss flow is integrated for the actual path and its clock-matched tied counterfactual."
            ),
            "response_link": (
                "Positive physical aggregate deficit/replay terms plus nonnegative BPB amplitudes on terminal excess-loss differences."
            ),
            "additional_degrees_of_freedom": (
                "Aggregate shortage power and offset; GGLLF clock, rate, power, gradient correlation, rare rate ratio, and one "
                "shared ridge. rho=0 makes Delta u numerically zero and is the exact aggregate-only ablation."
            ),
            "units_and_symmetries": (
                "Aggregate exposures are dimensionless simulated epochs; loss states and correlations are dimensionless; "
                "rates are inverse normalized time; all response amplitudes have BPB units. Counterfactual subtraction fixes "
                "the aggregate/phase intercept symmetry."
            ),
            "single_phase_restriction": (
                "For w0=w1, the actual and counterfactual state trajectories coincide exactly, so the model reduces to the "
                "aggregate one-phase spine. That restricted spine must also be refit independently on one-phase outcomes."
            ),
            "starcoder_signature": (
                "The aggregate spine must locate the rare-data exposure valley, while rho>0 must improve phase-fiber shape over "
                "rho=0 on both schedules and produce a larger late-rare contrast on WSD than cosine."
            ),
            "catastrophic_optimism_resolution": (
                "Phase transfer cannot cancel aggregate undercoverage or replay harm because it is a tied-counterfactual "
                "difference with bounded PSD cross-gradient credit."
            ),
            "response_compression_resolution": (
                "The aggregate spine preserves one-phase dynamic range while path-dependent loss-state differences add measured "
                "phase-fiber variation without output calibration."
            ),
            "scale_transfer_expectation": (
                "The aggregate response may be target-scale specific; the sign and dimensionless strength of gradient transfer "
                "should be comparable across related schedules and swarms."
            ),
            "cheapest_falsification": (
                "rho>0 must beat the exact rho=0 aggregate-only ablation globally and in >=3/5 folds on both StarCoder "
                "schedules, remain within 5% of both shape frontiers, select stable interior rho, and place raw optima within 0.15."
            ),
            "status": "active_preregistered",
            "status_evidence": "Frozen before any OGGTR fit or StarCoder evaluation.",
        }
        registry = pd.concat([registry, pd.DataFrame([row], columns=registry.columns)], ignore_index=True)
        registry.to_csv(REGISTRY, index=False)

    ledger = pd.read_csv(LEDGER)
    row = {
        "timestamp": datetime.now(UTC).isoformat(),
        "round_id": "round_52_preregistration",
        "candidate_id": "OGGTR",
        "candidate_family": "Orthogonal gradient-Gram transport",
        "hyperparameters": "Frozen aggregate-power/offset and clock/rate/power/correlation/rare-scale/ridge grid",
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": True,
        "observations_inspiring_mechanism": (
            "Round 51 shows loss-state dynamics underprice aggregate over-specialization; the frozen StarCoder frontiers all "
            "separate aggregate shortage/replay from phase correction. OGGTR tests the same separation with an exact PSD flow."
        ),
        "novelty_class": "Tied-counterfactual identification of gradient-Gram phase transport",
        "evaluation_status": "preregistered for StarCoder gate; no new adversarial evaluation",
        "evidence_path": "round52_orthogonal_gradient_gram_starcoder",
        "notes": "Running sealed phase-fiber panel remains untouched.",
    }
    identity = ["round_id", "candidate_id", "evaluation_status"]
    existing = set(map(tuple, ledger[identity].itertuples(index=False, name=None)))
    if tuple(row[column] for column in identity) not in existing:
        ledger = pd.concat([ledger, pd.DataFrame([row], columns=ledger.columns)], ignore_index=True)
        ledger.to_csv(LEDGER, index=False)


if __name__ == "__main__":
    main()
