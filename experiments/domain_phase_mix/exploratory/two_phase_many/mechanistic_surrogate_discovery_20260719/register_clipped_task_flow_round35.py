# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2"]
# ///
"""Preregister the clipped task-potential flow before reading StarCoder outcomes."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
REGISTRY = OUTPUT_ROOT / "approach_registry.csv"
LEDGER = OUTPUT_ROOT / "data_use_ledger.csv"


def registry_row() -> dict[str, str]:
    return {
        "id": "CTPF",
        "family": "Clipped task-potential flow",
        "relationship_to_prior": "Adds the optimizer's actual global-gradient clipping nonlinearity to the rejected two-dimensional quadratic task flow NQGF. It is not another optimizer-memory or output-interaction term.",
        "materially_new_mechanism": "A state-dependent saturation of update magnitude at the declared optimizer clip threshold; the unclipped flow is an exact nested ablation.",
        "mechanistic_premise": "The StarCoder runs use global gradient clipping. Extreme mixtures can generate large task gradients, but their parameter displacement is capped, changing both acquisition speed and phase-order effects.",
        "governing_equations": "g(z,p)=(1-p)H_b(z-mu_b)+pH_r(z-mu_r); dz/dtau=-k g/max(1,||g||/c); Y=b+A[(1-q)L_b(z_T)+qL_r(z_T)], A>=0. tau is normalized integrated learning-rate time and c=infinity is the exact ablation.",
        "latent_state": "A two-dimensional shared capability coordinate z; clipping introduces no extra dynamic state.",
        "state_transition": "Deterministic gradient flow on two quadratic task potentials in normalized optimizer time, with the same global-norm clipping used in training.",
        "response_link": "One nonnegative BPB amplitude on the declared broad/rare terminal task potential plus an intercept; clipping cannot directly recalibrate outputs.",
        "additional_degrees_of_freedom": "One dimensionless clip threshold c beyond NQGF. Task curvature, anisotropy, angle, relaxation, evaluation rare weight, and ridge are selected from a frozen grid.",
        "units_and_symmetries": "z, task optima, and normalized time are dimensionless. c has units of the fixed task-gradient scale; k is inverse optimizer time; A and b have BPB units. Fixed task optima and broad curvature remove affine state and gradient-scale symmetries.",
        "single_phase_restriction": "When both phases have the same mixture, the autonomous clipped flow composes across the artificial boundary. The same restricted law can be independently fitted on tied policies.",
        "starcoder_signature": "Finite clipping must beat c=infinity on both schedules, especially on extreme policies, while preserving the cosine near-diagonal and WSD late-code valleys.",
        "catastrophic_optimism_resolution": "Concentrated policies cannot obtain arbitrarily fast apparent capability acquisition because high task gradients saturate at c.",
        "response_compression_resolution": "State-dependent saturation expands differences between moderate policies that remain unclipped and extreme policies that spend more optimizer time clipped.",
        "scale_transfer_expectation": "The physical max-gradient-norm is fixed by the optimizer; after fixing task-gradient scale, the active clipping regime should be stable across schedules and predictable from model gradient norms across scale.",
        "cheapest_falsification": "Reject if finite clipping does not beat c=infinity on both StarCoder surfaces, is selected at a grid boundary, misses either frozen shape reference by over 5%, or places either raw optimum over 0.15 from the observed optimum.",
        "status": "active_round35_preregistered",
        "status_evidence": "Frozen before StarCoder evaluation. Grid: curvature {0.5,1,2}, anisotropy {0.25,1,4}, angle {0,30,60,90}, relaxation {0.5,2,8}, evaluation rare weight {0.25,0.5,0.75,1}, clip {0.25,0.5,1,2,infinity}, ridge {0,0.1,1}. Historical, adversarial, and sealed-confirmation outcomes will not be read unless the StarCoder gate passes.",
    }


def main() -> None:
    registry = pd.read_csv(REGISTRY)
    row = registry_row()
    registry = registry.loc[~registry["id"].eq(row["id"])]
    registry = pd.concat([registry, pd.DataFrame([row], columns=registry.columns)], ignore_index=True)
    registry.to_csv(REGISTRY, index=False)

    ledger = pd.read_csv(LEDGER)
    ledger_row = {
        "timestamp": datetime.now(UTC).isoformat(),
        "round_id": "round_35_clipped_task_flow",
        "candidate_id": "CTPF",
        "candidate_family": row["family"],
        "hyperparameters": row["status_evidence"].split("Grid: ", maxsplit=1)[1],
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": True,
        "observations_inspiring_mechanism": "The actual StarCoder optimizer has max_grad_norm=1, while every prior optimizer-state model omitted clipping; previous NQGF and extreme-policy optimism establish the exact ablation and expected failure signature.",
        "novelty_class": "Actual optimizer global-gradient clipping as a state-dependent transition saturation",
        "evaluation_status": "preregistered before StarCoder evaluation",
        "evidence_path": "round35_clipped_task_flow_starcoder/report.md",
        "notes": "No new adversarial outcome was inspected for this proposal. The exposed development panel was already known before the round. The running phase-fiber panel remains sealed and was not inspected.",
    }
    identity = ["round_id", "candidate_id", "evaluation_status"]
    existing = set(map(tuple, ledger[identity].itertuples(index=False, name=None)))
    key = tuple(ledger_row[column] for column in identity)
    if key not in existing:
        ledger = pd.concat([ledger, pd.DataFrame([ledger_row], columns=ledger.columns)], ignore_index=True)
        ledger.to_csv(LEDGER, index=False)


if __name__ == "__main__":
    main()
