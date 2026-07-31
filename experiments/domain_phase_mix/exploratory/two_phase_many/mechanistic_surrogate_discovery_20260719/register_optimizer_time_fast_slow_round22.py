# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2"]
# ///
"""Preregister optimizer-time fast/slow consolidation before fitting."""

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
        "id": "OTFSC",
        "family": "Optimizer-time fast/slow consolidation",
        "relationship_to_prior": "Reopens identified fast/slow consolidation IFSC using the optimizer-time invariant supported directionally in OTTPF. IFSC evolved in token-fraction time; OTTPF used optimizer time but a rejected scalar task-potential state.",
        "materially_new_mechanism": "Fast competence and slow consolidated competence evolve in integrated learning-rate time rather than token time. The phase durations are fixed by the declared learning-rate schedule, not learned from BPB.",
        "mechanistic_premise": "Acquisition, forgetting, and consolidation require parameter updates. Tokens seen at near-zero learning rate should advance these states less than tokens seen at peak learning rate, while bucket exposure and repetition remain physical-token quantities.",
        "governing_equations": "d tau=eta_lr(t)dt/integral eta_lr; df_i/d tau=q w_i(1-f_i)-h(1-w_i)f_i; ds_i/d tau=k(f_i-s_i); Y=F_phys(a)-sum_i A_i[(1-omega)(f_i-f_i^tie)+omega(s_i-s_i^tie)], A_i>=0. The exact ablation sets tau=t.",
        "latent_state": "Per declared family, a bounded fast competence f_i and a bounded slow consolidated competence s_i. Physical aggregate exposure remains a separate state in F_phys.",
        "state_transition": "Exact constant-input integration of the triangular fast/slow ODE through each phase. Token-clock and optimizer-clock transitions differ only in their fixed phase durations; tied policies obey the autonomous semigroup law under either clock.",
        "response_link": "A nonnegative capability benefit from terminal competence relative to the phase-tied aggregate counterfactual, added to the same physical-exposure shortage and replay spine. There is no output calibrator.",
        "additional_degrees_of_freedom": "No continuous degree of freedom beyond IFSC. The new choice is one discrete physical clock, compared against token time. Rates q,h,k, slow weight omega, and ridge are selected by nested StarCoder CV from a frozen grid.",
        "units_and_symmetries": "Weights, normalized clocks, f, s, and omega are dimensionless; q,h,k are inverse normalized clock; response amplitudes and intercept have BPB units. Fixed zero initialization and bounded states remove affine state symmetries. A common clock-rate scale is fixed by normalizing total clock mass to one.",
        "single_phase_restriction": "For phase-tied weights, exact autonomous composition equals an uninterrupted one-phase transition. The algebraic correction is zero by construction; the same tied physical-exposure response must also be independently fitted on tied data.",
        "starcoder_signature": "Relative to token time, optimizer time lengthens the effective early phase under both schedules, especially WSD. If consolidation is update-driven, the same rate regime should fit both surfaces better, preserve the WSD off-diagonal optimum, and reduce the IFSC boundary-rate pathology.",
        "catastrophic_optimism_resolution": "Slow competence prevents a short late specialization phase from erasing every early capability, while the physical aggregate spine still prices missing exposure. This claim is not accepted unless the shape and rate-identification gates pass.",
        "response_compression_resolution": "Two bounded state timescales can preserve wider phase-order variation; optimizer time changes the transition trajectory rather than rescaling outputs.",
        "scale_transfer_expectation": "The clock is computed from each declared LR schedule. Dimensionless rate ordering should transfer; absolute rates may depend on total integrated optimizer progress and tokens per parameter. The token-clock ablation must remain available at every scale.",
        "cheapest_falsification": "Optimizer time fails to beat token time on both StarCoder schedules; nested RMSE is over 5% above the corrected shape gate; raw optima miss observed minima by over 0.15; acquisition or consolidation still selects a screened boundary; or selected rate regimes disagree by more than 4x across schedules.",
        "status": "active_preregistered",
        "status_evidence": "Frozen before OTFSC fitting. The complete 107-policy WSD surface is available; historical and adversarial targets are prohibited until the StarCoder gate passes.",
    }
    registry = registry.loc[~registry["id"].eq(row["id"])]
    registry = pd.concat([registry, pd.DataFrame([row], columns=registry.columns)], ignore_index=True)
    registry.to_csv(REGISTRY, index=False)

    ledger_row = {
        "timestamp": datetime.now(UTC).isoformat(),
        "round_id": "round_22_preregistration",
        "candidate_id": row["id"],
        "candidate_family": row["family"],
        "hyperparameters": "Frozen clock={token,optimizer}; learn={0.25,1,4,16,64}; forget={0.125,0.5,2,8,32}; consolidate={0.0625,0.25,1,4,16,64}; slow={0.25,0.5,0.75}; l2={0.1,1}; aggregate spine fixed to prior IFSC schedule-specific selection",
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": True,
        "observations_inspiring_mechanism": "The corrected 107-point WSD surface makes IFSC the strongest WSD shape baseline, while OTTPF found a small consistent within-family benefit from optimizer time. OTFSC tests whether the physical clock resolves IFSC's boundary-rate identification rather than merely retuning a rejected task-potential model.",
        "novelty_class": "Fixed optimizer-time transition for a two-timescale competence state",
        "evaluation_status": "preregistered for corrected StarCoder gate; no OTFSC adversarial evaluation",
        "evidence_path": "approach_registry.csv",
        "notes": "Exact token-clock ablation mandatory. Failure cannot be repaired by a learned clock exponent, phase head, or adversarial tuning.",
    }
    identity = ["round_id", "candidate_id", "evaluation_status"]
    existing = set(map(tuple, ledger[identity].itertuples(index=False, name=None)))
    if tuple(ledger_row[column] for column in identity) not in existing:
        ledger = pd.concat([ledger, pd.DataFrame([ledger_row], columns=ledger.columns)], ignore_index=True)
        ledger.to_csv(LEDGER, index=False)
    print(registry.tail(1)[["id", "family", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()
