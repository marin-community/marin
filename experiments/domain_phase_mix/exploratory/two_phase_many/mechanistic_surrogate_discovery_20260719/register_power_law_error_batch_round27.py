# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Preregister the frozen power-law error-kinetics batch."""

from __future__ import annotations

import csv
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
REGISTRY = OUTPUT_ROOT / "approach_registry.csv"
LEDGER = OUTPUT_ROOT / "data_use_ledger.csv"


def append(path: Path, row: dict[str, str], key: str, value: str) -> None:
    with path.open(newline="") as source:
        reader = csv.DictReader(source)
        rows = list(reader)
        fields = reader.fieldnames
    if fields is None:
        raise RuntimeError(f"{path} has no header")
    if any(existing[key] == value for existing in rows):
        return
    with path.open("a", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writerow({name: row.get(name, "") for name in fields})


def common() -> dict[str, str]:
    return {
        "units_and_symmetries": (
            "Errors, competence, weights, and normalized time are dimensionless; E and replay are simulated epochs; "
            "rates are inverse normalized time and response amplitudes have BPB units. Unit initial errors and fixed "
            "power-law normalization remove state-scale symmetries."
        ),
        "single_phase_restriction": (
            "Tie both phase policies and integrate the same autonomous error kinetics; an artificial boundary has no "
            "effect. Independently select and refit the same restricted form on tied observations."
        ),
        "response_compression_resolution": (
            "Power-law error decay preserves a wider low-error dynamic range than exponential saturation, while literal "
            "replay expands the high-loss range without an output calibrator."
        ),
        "scale_transfer_expectation": (
            "Learning-curve powers and replay semantics should transfer; integrated rates may scale with tokens per "
            "parameter and must receive an explicit clock law before cross-scale deployment."
        ),
        "status": "active_frozen_round27",
        "status_evidence": "Frozen before either power-law candidate was fit; no Delphi heldout outcome was read.",
    }


def main() -> None:
    append(
        REGISTRY,
        {
            "id": "PLSC",
            "family": "Power-law foundation-specialization kinetics",
            "relationship_to_prior": (
                "Reopens frozen FSC/FSCR through a materially new scaling-law transition. Their exponential hazard forces "
                "fast saturation; PLSC makes marginal learning progress decay as a power of remaining error."
            ),
            "materially_new_mechanism": (
                "Foundation and specialist error obey power-law learning kinetics; specialist progress is gated by current "
                "foundation competence. The zero power is the exact exponential ablation."
            ),
            "mechanistic_premise": (
                "LLM learning curves exhibit diminishing power-law progress rather than memoryless exponential saturation. "
                "Specialist learning additionally requires already acquired foundation competence."
            ),
            "governing_equations": (
                "du/dt=-k_g[(1-p)+rho p]u^(1+zeta); dv/dt=-k_s p(1-u)^nu v^(1+zeta); "
                "R_i=(E_i-1)_+; Y=b+A_g u_T+A_s v_T+sum_i H_iR_i. zeta=0 is FSCR."
            ),
            "latent_state": "Positive foundation error u, specialist error v, cumulative exposure E, and literal replay R.",
            "state_transition": (
                "Analytic power-law error decay per phase, with the specialist hazard integrating current foundation "
                "competence along the phase trajectory."
            ),
            "response_link": "Nonnegative amplitudes on terminal errors and literal repeated traversals.",
            "additional_degrees_of_freedom": (
                "Two rates, rare foundation efficiency, prerequisite power, one shared learning-curve power, four "
                "nonnegative response amplitudes, intercept, and ridge."
            ),
            "starcoder_signature": (
                "A nonzero learning-curve power should avoid the maximum specialist-rate collapse, retain broad-first/rare-"
                "late WSD asymmetry, and price high-code arms through replay."
            ),
            "catastrophic_optimism_resolution": (
                "Foundation shortage and specialist shortage cannot vanish exponentially after a small exposure; power-law "
                "tails and replay jointly keep remote policies costly."
            ),
            "cheapest_falsification": (
                "zeta=0 or nu=0 wins, rates hit boundaries or disagree by over 4x, nested RMSE misses either StarCoder "
                "reference by over 5%, or raw optima miss by over 0.15."
            ),
            **common(),
        },
        "id",
        "PLSC",
    )
    append(
        REGISTRY,
        {
            "id": "PLAFK",
            "family": "Power-law acquisition-forgetting kinetics",
            "relationship_to_prior": (
                "Reopens retained-state route A with a materially new learning-curve law and explicit foundation/specialist "
                "error decomposition. It is not a retained-exposure feature transform."
            ),
            "materially_new_mechanism": (
                "Specialist error decays by power-law acquisition under rare data and relaxes back toward the unlearned "
                "state under broad updates. Zero forgetting is an exact nested ablation."
            ),
            "mechanistic_premise": (
                "Rare examples reduce specialist error with diminishing returns; subsequent broad-only updates can forget "
                "that private capability while continuing to improve foundation error."
            ),
            "governing_equations": (
                "du/dt=-k_g[(1-p)+rho p]u^(1+zeta); dv/dt=-k_s p v^(1+zeta)+h(1-p)(1-v); "
                "R_i=(E_i-1)_+; Y=b+A_g u_T+A_s v_T+sum_i H_iR_i. h=0 is no-forgetting."
            ),
            "latent_state": "Positive foundation and specialist errors plus cumulative exposure and literal replay.",
            "state_transition": (
                "Bounded nonlinear acquisition-forgetting ODE integrated continuously through each phase."
            ),
            "response_link": "Nonnegative amplitudes on terminal errors and literal repeated traversals.",
            "additional_degrees_of_freedom": (
                "Two acquisition rates, rare foundation efficiency, forgetting rate, shared learning-curve power, four "
                "nonnegative response amplitudes, intercept, and ridge."
            ),
            "starcoder_signature": (
                "Early-only code is forgotten by a later broad phase, late code persists, and the power-law learning curve "
                "keeps both cosine and WSD optima interior."
            ),
            "catastrophic_optimism_resolution": (
                "Terminal specialist error explicitly rises after broad-only continuation, while physical replay prevents "
                "all-code corners from becoming falsely favorable."
            ),
            "cheapest_falsification": (
                "h=0 or zeta=0 wins, rates hit boundaries or disagree by over 4x, nested RMSE misses either StarCoder "
                "reference by over 5%, or raw optima miss by over 0.15."
            ),
            **common(),
        },
        "id",
        "PLAFK",
    )
    append(
        LEDGER,
        {
            "round_id": "round27_power_law_error_batch_frozen",
            "timestamp_utc": "2026-07-19T00:00:00Z",
            "model_ids": "PLSC,PLAFK",
            "hyperparameters_frozen_before_adversarial": "true",
            "adversarial_outcomes_inspected": "none for this batch",
            "observations_inspiring_mechanism": (
                "Round-25/26 exponential competence fits selected maximum specialist rates; LLM scaling motivates a "
                "power-law rather than exponential learning transition."
            ),
            "genuinely_new_or_retuning": (
                "new transition law and exact nested exponential/no-forgetting ablations; no adversarial retuning"
            ),
            "data_used_for_selection": "algebraic checks and both StarCoder surfaces only",
            "data_explicitly_not_used": (
                "historical Delphi heldouts, all exposed adversarial outcomes, and sealed frontier phase-fiber outcomes"
            ),
            "decision": "frozen before first fit",
            "evidence_path": "round27_power_law_error_starcoder/report.md",
        },
        "round_id",
        "round27_power_law_error_batch_frozen",
    )


if __name__ == "__main__":
    main()
