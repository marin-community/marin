# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2"]
# ///
"""Preregister the frozen round-25 shared/private competence batch."""

from __future__ import annotations

import csv
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
REGISTRY = OUTPUT_ROOT / "approach_registry.csv"
LEDGER = OUTPUT_ROOT / "data_use_ledger.csv"


def append_registry(row: dict[str, str]) -> None:
    with REGISTRY.open(newline="") as source:
        reader = csv.DictReader(source)
        rows = list(reader)
        fieldnames = reader.fieldnames
    if fieldnames is None:
        raise RuntimeError("Registry has no header")
    if any(existing["id"] == row["id"] for existing in rows):
        return
    with REGISTRY.open("a", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writerow({name: row.get(name, "") for name in fieldnames})


def append_ledger() -> None:
    with LEDGER.open(newline="") as source:
        reader = csv.DictReader(source)
        rows = list(reader)
        fieldnames = reader.fieldnames
    if fieldnames is None:
        raise RuntimeError("Ledger has no header")
    round_id = "round25_shared_private_batch_frozen"
    if any(existing["round_id"] == round_id for existing in rows):
        return
    row = {
        "round_id": round_id,
        "timestamp_utc": "2026-07-19T00:00:00Z",
        "model_ids": "FSC,FPCGF",
        "hyperparameters_frozen_before_adversarial": "true",
        "adversarial_outcomes_inspected": "none for this batch",
        "observations_inspiring_mechanism": (
            "The refined WSD StarCoder optimum is broad-early and rare-late, while prior scalar retained-exposure, "
            "optimizer-memory, and output-penalty routes miss that geometry. The batch asks whether a separately "
            "represented foundation prerequisite and specialist state can create the ordering effect."
        ),
        "genuinely_new_or_retuning": (
            "genuinely new latent-state/response batch: FSC is a bounded cross-state acquisition cascade; FPCGF is "
            "gradient flow on an explicit multiplicative shared/private task potential"
        ),
        "data_used_for_selection": "algebraic checks and both StarCoder surfaces only",
        "data_explicitly_not_used": (
            "historical Delphi heldouts, all 120 exposed adversarial policies, and the sealed frontier phase-fiber panel"
        ),
        "decision": "frozen before first fit",
        "evidence_path": "round25_shared_private_starcoder/report.md",
    }
    with LEDGER.open("a", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writerow({name: row.get(name, "") for name in fieldnames})


def main() -> None:
    common = {
        "units_and_symmetries": (
            "Normalized training time, mixture weights, and competence states are dimensionless. Transition rates are "
            "inverse normalized time; response amplitudes and intercepts have BPB units. Fixed zero initialization and "
            "bounded/state-potential normalization remove affine state symmetries."
        ),
        "response_compression_resolution": (
            "Two explicit error-bearing states can expand the response range through state evolution rather than an "
            "output calibrator; this claim is not accepted unless both StarCoder and later heldout gates pass."
        ),
        "status": "active_frozen_round25",
        "status_evidence": "Frozen before either candidate was fit; no historical or adversarial outcome was read.",
    }
    append_registry(
        {
            "id": "FSC",
            "family": "Foundation-specialization competence cascade",
            "relationship_to_prior": (
                "Distinct from prior directional effective-exposure gate R and same-domain cascade EGCC: FSC carries two "
                "bounded cross-domain competence states and evaluates their terminal unresolved error directly."
            ),
            "materially_new_mechanism": (
                "A foundation state is a prerequisite for specialist acquisition; specialist competence is a separate "
                "latent state rather than a rescaled exposure feature."
            ),
            "mechanistic_premise": (
                "Broad data builds reusable foundation competence. Rare data can build the same foundation less "
                "efficiently, while its specialist gain is proportional to the foundation already available."
            ),
            "governing_equations": (
                "dg/dt=k_g[(1-p)+rho p](1-g); ds/dt=k_s p g^nu(1-s); "
                "Y=b+A_g(1-g_T)+A_s(1-s_T), A_g,A_s>=0. nu=0 is the exact no-prerequisite ablation."
            ),
            "latent_state": "Bounded foundation competence g and bounded specialist competence s, both initialized at zero.",
            "state_transition": (
                "Autonomous monotone acquisition within each constant-mixture phase; the specialist hazard integrates the "
                "current foundation state exactly."
            ),
            "response_link": "Nonnegative BPB amplitudes on terminal foundation and specialist unresolved-error masses.",
            "additional_degrees_of_freedom": (
                "Foundation and specialist rates, rare-to-broad foundation efficiency, prerequisite power, two nonnegative "
                "response amplitudes, intercept, and ridge."
            ),
            "single_phase_restriction": (
                "Tie both phase policies and integrate the same autonomous ODE; the artificial boundary cancels exactly. "
                "The same restricted equation must also be independently selected and refit on tied data."
            ),
            "starcoder_signature": (
                "Broad-first training raises g before rare specialist updates, rotating the WSD valley off diagonal; the "
                "cosine optimum should remain nearly tied. A positive prerequisite must beat nu=0 on both schedules."
            ),
            "catastrophic_optimism_resolution": (
                "A rare-heavy policy cannot claim specialist benefit without first acquiring foundation competence, so "
                "aggregate underexposure cannot be canceled by an independent surplus credit."
            ),
            "scale_transfer_expectation": (
                "Dimensionless rate ordering and rare foundation efficiency should transfer; integrated rates may vary "
                "with tokens per parameter, so scale transfer requires a declared clock law."
            ),
            "cheapest_falsification": (
                "nu=0 wins either StarCoder schedule, rates hit boundaries or differ by over 4x, nested RMSE misses the "
                "corrected shape references by over 5%, or either raw optimum misses the observed valley by over 0.15."
            ),
            **common,
        }
    )
    append_registry(
        {
            "id": "FPCGF",
            "family": "Factorized-capability gradient flow",
            "relationship_to_prior": (
                "Not the static bottleneck C, scalar nonlinear potential NTPGF, or quadratic noncommuting flow NQGF. FPCGF "
                "uses a multiplicative shared/private capability in the task potential itself."
            ),
            "materially_new_mechanism": (
                "Target error is a variational production bottleneck g*s; broad data builds g but suppresses private s, "
                "while rare gradients can improve both."
            ),
            "mechanistic_premise": (
                "Specialist performance requires both reusable foundation features and a private specialist factor. Broad "
                "updates improve foundation but wash out private specialization; rare updates optimize their product."
            ),
            "governing_equations": (
                "L_b=(g-1)^2/2+lambda s^2/2; L_r=(gs-1)^2/2+rho(g-1)^2/2; "
                "d(g,s)/dt=-k grad[(1-p)L_b+pL_r]; Y=b+A L_r(g_T,s_T), A>=0. lambda=0 removes broad forgetting."
            ),
            "latent_state": "Dimensionless shared foundation factor g and private specialist factor s, initialized at zero.",
            "state_transition": (
                "Deterministic gradient flow on the current mixture-weighted broad/specialist task potential through each "
                "piecewise-constant phase."
            ),
            "response_link": (
                "The same specialist task potential used to generate training gradients, followed only by a nonnegative "
                "BPB amplitude and intercept."
            ),
            "additional_degrees_of_freedom": (
                "One flow speed, broad-to-private forgetting lambda, rare foundation efficiency rho, nonnegative amplitude, "
                "intercept, and ridge."
            ),
            "single_phase_restriction": (
                "For tied policies the autonomous gradient flow composes exactly across the artificial boundary; the same "
                "restricted task potentials must also be independently selected and fitted on tied observations."
            ),
            "starcoder_signature": (
                "Multiplicative gating makes rare-only training initially inefficient, while broad suppression of s makes "
                "rare-late training valuable. It should recover both the near-diagonal cosine and off-diagonal WSD valleys."
            ),
            "catastrophic_optimism_resolution": (
                "The specialist potential remains high if either factor is missing; independent additive surplus credits "
                "cannot cancel the bottleneck."
            ),
            "scale_transfer_expectation": (
                "Task-potential ratios are dimensionless and should retain signs; flow speed may depend on optimizer progress "
                "and tokens per parameter and must be modeled explicitly before cross-scale use."
            ),
            "cheapest_falsification": (
                "lambda=0 wins both schedules, selected rho/lambda regimes are incompatible, nested RMSE misses the corrected "
                "shape references by over 5%, or either raw optimum misses the observed valley by over 0.15."
            ),
            **common,
        }
    )
    append_ledger()


if __name__ == "__main__":
    main()
