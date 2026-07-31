# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2"]
# ///
"""Preregister optimizer-time task-potential flow before fitting."""

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
        "id": "OTTPF",
        "family": "Optimizer-time task-potential flow",
        "relationship_to_prior": "Reopens nonlinear task-potential flow NTPGF through a materially new transition invariant. NTPGF used token fraction as time. Learning-rate exposure AC multiplied an additive dose by phase-average LR but never evolved a state in optimizer time.",
        "materially_new_mechanism": "Representation dynamics evolve in integrated learning-rate time d tau proportional to eta(t) dt, while the data policy remains defined in physical token time. The cosine and WSD schedules therefore induce fixed, different phase-transition durations without a learned late multiplier.",
        "mechanistic_premise": "For gradient-based training, a token at near-zero learning rate cannot move the representation as much as a token at peak learning rate. State dynamics should compose in optimizer time, not raw token count, especially when the phase boundary and LR decay interact.",
        "governing_equations": "d tau=eta(t)dt/integral eta; dz/d tau=-k[(1-p)V_b'(z)+pV_r'(z)]; V_b=(z+1/2)^2/2+h(z+1/2)^4/4; V_r=r(z-1/2)^2/2+hs(z-1/2)^4/4; Y=b+A[(1-q)V_b(z_T)+qV_r(z_T)], A>=0. The exact ablation uses d tau=dt.",
        "latent_state": "One dimensionless shared specialization coordinate z initialized at zero. No phase-specific head or additional memory variable is introduced.",
        "state_transition": "Deterministic nonlinear gradient flow through each constant-mixture phase. Phase duration is either physical token mass or the fixed normalized integral of the declared LR schedule over that phase.",
        "response_link": "The same nonnegative terminal task-potential response as NTPGF; the clock changes only state evolution, not output calibration.",
        "additional_degrees_of_freedom": "No continuous parameter beyond NTPGF. One discrete clock choice is selected against the exact token-time ablation. Cosine and WSD optimizer-time masses are fixed from their schedules.",
        "units_and_symmetries": "t and tau are dimensionless normalized clocks; eta is normalized by total LR mass; z, p, r, h, s, k, and q are dimensionless; A and b have BPB units. Fixed task optima and initial state remove affine state symmetry.",
        "single_phase_restriction": "When phase weights are tied, the autonomous flow composes exactly under either clock. The same restricted form is independently selected and fitted on tied policies.",
        "starcoder_signature": "Optimizer time assigns about 82% of cosine update mass and 89% of WSD update mass before the phase boundary. It should improve both surfaces over token time, align dimensionless curvature regimes, and keep raw optima in the observed valleys.",
        "catastrophic_optimism_resolution": "The convex terminal task potential still penalizes remote specialization; the fixed physical clock prevents a learned late multiplier from assigning unsupported value to low-LR tokens.",
        "response_compression_resolution": "Schedule-dependent optimizer time can expand phase-order differences through the state trajectory rather than an output rescaling.",
        "scale_transfer_expectation": "The clock is computed from the declared LR schedule at every scale. Curvature ratios should transfer; total relaxation may scale with total integrated LR and tokens per parameter.",
        "cheapest_falsification": "Token time wins globally or in most folds; optimizer time improves only one schedule; nested RMSE misses the existing StarCoder frontier by over 5%; or raw optima miss either observed best by more than 0.15.",
        "status": "active_preregistered",
        "status_evidence": "Frozen before any OTTPF fit. No historical or adversarial outcome may be evaluated until both StarCoder schedules support the optimizer clock.",
    }
    registry = registry.loc[~registry["id"].eq("OTTPF")]
    registry = pd.concat([registry, pd.DataFrame([row], columns=registry.columns)], ignore_index=True)
    registry.to_csv(REGISTRY, index=False)

    ledger_row = {
        "timestamp": datetime.now(UTC).isoformat(),
        "round_id": "round_20_preregistration",
        "candidate_id": "OTTPF",
        "candidate_family": "Optimizer-time task-potential flow",
        "hyperparameters": "Frozen clock={token_time,optimizer_time}; optimizer clock fixed from full cosine or WSD plateau+cosine-decay LR; NTPGF grid r={0.25,0.5,1,2,4}, h={0,0.25,1,4}, s={0.25,1,4}, k={0.5,1,2,4,8}, q={0.1,0.3,0.5,0.7,0.9}, ridge={0,0.1,1}",
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": True,
        "observations_inspiring_mechanism": "The cosine and WSD StarCoder surfaces use different LR schedules, while all dynamical audits used raw token fractions as phase duration. Prior AC changed additive exposure but did not change a latent-state transition clock.",
        "novelty_class": "Separate physical-token and integrated-learning-rate clocks with no learned phase multiplier",
        "evaluation_status": "preregistered for StarCoder clock-ablation gate; no OTTPF adversarial evaluation",
        "evidence_path": "approach_registry.csv",
        "notes": "The token-time clock is mandatory. Failure cannot be rescued by tuning an LR exponent, adding phase heads, or fitting schedule-specific clock masses.",
    }
    identity = ["round_id", "candidate_id", "evaluation_status"]
    existing = set(map(tuple, ledger[identity].itertuples(index=False, name=None)))
    if tuple(ledger_row[column] for column in identity) not in existing:
        ledger = pd.concat([ledger, pd.DataFrame([ledger_row], columns=ledger.columns)], ignore_index=True)
        ledger.to_csv(LEDGER, index=False)
    print(registry.tail(1)[["id", "family", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()
