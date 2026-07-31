# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2"]
# ///
"""Preregister adaptive-second-moment gradient flow before fitting."""

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
        "id": "ASMGF",
        "family": "Adaptive-second-moment gradient flow",
        "relationship_to_prior": "Extends the rejected quadratic gradient-flow bowl AI with an optimizer preconditioner state. It is distinct from learning-rate exposure AC, global gradient-noise discount AD, Kalman uncertainty AK, and signed momentum OMGF: the persistent state is the optimizer's positive gradient second moment and it changes the transition denominator.",
        "materially_new_mechanism": "A phase shift changes the gradient immediately but the adaptive second-moment accumulator decays over its own timescale, transiently suppressing or amplifying parameter motion under the new mixture.",
        "mechanistic_premise": "Adam-like optimizers carry a positive gradient-scale estimate across phase boundaries. The same raw gradient can therefore produce a different update depending on the preceding phase, creating path dependence without a phase-specific response head.",
        "governing_equations": "g(z,p)=(1-p)(z+1/2)+r p(z-1/2); dz/dt=-k g/(sqrt(v)+epsilon); dv/dt=kappa(g^2-v); Y=b+A[(1-q)(z_T+1/2)^2/2+q r(z_T-1/2)^2/2], A>=0. The exact ablation is ordinary gradient flow dz/dt=-k g.",
        "latent_state": "A dimensionless scalar specialization coordinate z and a nonnegative squared-gradient accumulator v. Both start at zero; z and v are continuous at the phase boundary.",
        "state_transition": "Deterministic continuous-time RMSProp/Adam-second-moment flow under each piecewise-constant mixture. The second-moment memory rate kappa is separate from the representation speed k.",
        "response_link": "A nonnegative BPB amplitude times the same broad/rare quadratic task potential evaluated at the terminal representation, plus an intercept. The optimizer state v is not evaluated directly.",
        "additional_degrees_of_freedom": "Rare-to-broad task curvature r, representation speed k, second-moment memory rate kappa, optimizer stabilizer epsilon, evaluation mixture q, nonnegative amplitude, and intercept. Ordinary gradient flow omits kappa and epsilon.",
        "units_and_symmetries": "z, p, r, q, normalized time, and epsilon relative to the fixed task-gradient scale are dimensionless; v has squared-gradient units; k and kappa are inverse normalized time; A and b have BPB units. Fixed task optima, broad curvature, initialization, and gradient scale remove affine state symmetries.",
        "single_phase_restriction": "For tied phases the autonomous (z,v) flow composes exactly across the artificial boundary. The same restricted law is also independently selected and fitted on tied policies.",
        "starcoder_signature": "A transferable adaptive-memory regime should beat ordinary gradient flow on both schedules, with a larger order effect under WSD's short terminal phase. Memory rate and epsilon should select compatible regimes and raw optima should lie in both observed valleys.",
        "catastrophic_optimism_resolution": "The terminal task potential grows when adaptive preconditioning drives the representation away from the evaluation compromise; extreme phase shifts cannot earn an unbounded additive credit.",
        "response_compression_resolution": "Persistent preconditioner mismatch expands terminal-state separation after a phase shift through a declared optimizer state, rather than through an output calibrator.",
        "scale_transfer_expectation": "The dimensionless second-moment memory regime should be stable for the same optimizer and schedule family; integrated representation speed may increase with tokens per parameter. If optimizer memory is negligible at phase scale, the gradient-flow ablation should win.",
        "cheapest_falsification": "Ordinary gradient flow wins globally or in most folds; memory/epsilon regimes disagree across cosine and WSD; nested RMSE misses the existing StarCoder frontier by over 5%; or either raw optimum misses the observed best by more than 0.15 policy distance.",
        "status": "active_preregistered",
        "status_evidence": "Frozen before any ASMGF fit. No historical or adversarial outcome may be evaluated until both StarCoder schedules support the adaptive second-moment state.",
    }
    registry = registry.loc[~registry["id"].eq("ASMGF")]
    registry = pd.concat([registry, pd.DataFrame([row], columns=registry.columns)], ignore_index=True)
    registry.to_csv(REGISTRY, index=False)

    ledger_row = {
        "timestamp": datetime.now(UTC).isoformat(),
        "round_id": "round_19_preregistration",
        "candidate_id": "ASMGF",
        "candidate_family": "Adaptive-second-moment gradient flow",
        "hyperparameters": "Frozen dynamics={gradient_flow,adaptive_second_moment}; r={0.5,1,2}; adaptive speed={0.03,0.1,0.3,1}; kappa={0.25,1,4,16}; epsilon={0.03,0.1,0.3,1}; gradient-flow speed={0.3,1,3,10}; q={0.2,0.5,0.8}; ridge={0,0.1,1}; RK4=256 steps/unit time",
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": True,
        "observations_inspiring_mechanism": "Theoretical no-go results exclude scalar linear memory and affine endpoint laws; optimizer momentum and nonlinear task potentials failed. An adaptive optimizer's positive second moment is a distinct path state not tested by the prior learning-rate, gradient-noise, or uncertainty routes.",
        "novelty_class": "Persistent adaptive optimizer preconditioner with exact gradient-flow ablation",
        "evaluation_status": "preregistered for algebraic and StarCoder gates; no ASMGF adversarial evaluation",
        "evidence_path": "approach_registry.csv",
        "notes": "The gradient-flow ablation is mandatory. Failure cannot be rescued by adding momentum, direct v response, replay, phase heads, output calibration, or tuning on heldouts.",
    }
    identity = ["round_id", "candidate_id", "evaluation_status"]
    existing = set(map(tuple, ledger[identity].itertuples(index=False, name=None)))
    if tuple(ledger_row[column] for column in identity) not in existing:
        ledger = pd.concat([ledger, pd.DataFrame([ledger_row], columns=ledger.columns)], ignore_index=True)
        ledger.to_csv(LEDGER, index=False)
    print(registry.tail(1)[["id", "family", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()
