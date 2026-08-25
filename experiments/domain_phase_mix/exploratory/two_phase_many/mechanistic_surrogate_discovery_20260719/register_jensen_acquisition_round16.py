# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2"]
# ///
"""Preregister exposure-rate-curved acquisition before fitting."""

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
        "id": "JARA",
        "family": "Jensen acquisition-rate response",
        "relationship_to_prior": "Unlike AA importance ESS and AD global concentration discounts, this law curves each bucket's instantaneous acquisition rate before time integration. Unlike A/I/AJ, it is not a scalar linear memory kernel and therefore does not collapse to one effective phase multiplier.",
        "materially_new_mechanism": "Useful competence can scale nonlinearly with instantaneous bucket sampling probability. Temporal specialization then changes acquired evidence by Jensen's inequality even at fixed aggregate exposure.",
        "mechanistic_premise": "If representation transfer makes low-rate exposure disproportionately useful, acquisition is sublinear in sampling mass; if coherent repeated gradients are needed, it is superlinear. A single dimensionless exponent tests which regime the data support.",
        "governing_equations": "x_i=sum_t gamma_t (w_i^(t))^zeta; r_i=x_i/(p_i^prop)^zeta; Y=b+sum_i A_i[(r_i+delta)^(-beta)-(1+delta)^(-beta)]+sum_i H_i[R(E_i)-R(E_i^prop)], A_i,H_i>=0, R(E)=E-(1-exp(-E)).",
        "latent_state": "One nonnegative accumulated useful-acquisition dose x_i per bucket plus physical duplicate mass E_i-(1-exp(-E_i)).",
        "state_transition": "During a constant-mixture phase, dx_i/dt=(w_i)^zeta. The state is additive across time but nonlinear in instantaneous sampling rate; zeta=1 is the exact physical-exposure ablation.",
        "response_link": "Nonnegative inverse-power shortage debt plus nonnegative literal duplicate-mass harm, both centered at the proportional tied policy.",
        "additional_degrees_of_freedom": "One global rate exponent zeta; common shortage power beta and offset delta selected by nested CV; one nonnegative shortage and replay amplitude per bucket plus intercept.",
        "units_and_symmetries": "Weights, normalized time, x_i, r_i, zeta, beta, and delta are dimensionless. BPB amplitudes and intercept carry BPB units. Reference normalization fixes the dose scale; phase labels are not interchangeable when durations differ.",
        "single_phase_restriction": "For tied weights w, x_i=(w_i)^zeta. The same restricted form is independently selected and fitted on tied policies; zeta=1 recovers physical aggregate exposure exactly.",
        "starcoder_signature": "zeta<1 rewards distributing a bucket across time and predicts a tied preference; zeta>1 rewards concentrated acquisition phases and can create an off-diagonal optimum. The same side of one should be selected across cosine and WSD if the mechanism transfers.",
        "catastrophic_optimism_resolution": "Shortage debt diverges as useful acquisition vanishes, while physical replay prevents high-rate phases from earning unlimited benefit.",
        "response_compression_resolution": "The rate exponent expands or contracts the policy-induced dose before the response, so it can widen poor-policy predictions without an output calibrator.",
        "scale_transfer_expectation": "The direction of zeta-1 should transfer across schedules and scales; its magnitude may approach one as optimization becomes more nearly linear in data rate. Shortage amplitudes remain target-specific.",
        "cheapest_falsification": "zeta=1 wins globally or in most folds; cosine and WSD select opposite sides of one; nested RMSE misses the existing shape frontier by over 5%; or either raw optimum is more than 0.15 from the observed best.",
        "status": "active_preregistered",
        "status_evidence": "Frozen before any JARA fit. No historical or adversarial outcome may be evaluated until both StarCoder schedules pass the shape and transfer gates.",
    }
    registry = registry.loc[~registry["id"].eq("JARA")]
    registry = pd.concat([registry, pd.DataFrame([row], columns=registry.columns)], ignore_index=True)
    registry.to_csv(REGISTRY, index=False)

    ledger_row = {
        "timestamp": datetime.now(UTC).isoformat(),
        "round_id": "round_16_preregistration",
        "candidate_id": "JARA",
        "candidate_family": "Jensen acquisition-rate response",
        "hyperparameters": "Frozen zeta={0.25,0.5,0.75,1,1.25,1.5,2}; beta={0.25,0.5,1,2}; delta={0.03,0.1,0.3,1}; ridge={0,0.1,1,10}; nonnegative shortage/replay amplitudes",
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": True,
        "observations_inspiring_mechanism": "Linear-kernel equivalence and affine endpoint no-go results leave nonlinear instantaneous acquisition as a distinct minimal transition law. StarCoder shows a phase advantage and aggregate-by-recency interaction not explained by scalar memory.",
        "novelty_class": "Nonlinear instantaneous acquisition-rate law",
        "evaluation_status": "preregistered for algebraic and StarCoder gates; no JARA adversarial evaluation",
        "evidence_path": "approach_registry.csv",
        "notes": "The zeta=1 ablation is mandatory. Failure cannot be rescued by family-specific exponents, phase heads, or output calibration in this round.",
    }
    identity = ["round_id", "candidate_id", "evaluation_status"]
    existing = set(map(tuple, ledger[identity].itertuples(index=False, name=None)))
    if tuple(ledger_row[column] for column in identity) not in existing:
        ledger = pd.concat([ledger, pd.DataFrame([ledger_row], columns=ledger.columns)], ignore_index=True)
        ledger.to_csv(LEDGER, index=False)
    print(registry.tail(1)[["id", "family", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()
