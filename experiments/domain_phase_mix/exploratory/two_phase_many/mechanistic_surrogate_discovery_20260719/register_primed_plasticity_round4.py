# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2"]
# ///
"""Reject MCR and preregister early-allocation-gated plasticity."""

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
    registry.loc[registry["id"].eq("MCR"), ["status", "status_evidence"]] = [
        "blocked_before_adversarial",
        "Rejected before adversarial evaluation. Delphi Table-9 Regret@1 is 0.0913 and family rates mostly collapse to a shared fast timescale. Cosine/WSD StarCoder OOF RMSE is 0.1749/0.2182; leave-region RMSE is 0.1256-0.2537. Foldwise rates are unstable and the WSD response is biased upward by 0.153 BPB.",
    ]
    row = {
        "id": "EAGP",
        "family": "Early-allocation-gated plasticity",
        "relationship_to_prior": "Reopens prior foundation-transfer route R and finite-capacity route AG only with a materially new bucket-local critical-period state. Unlike effective exposure, phase-1 efficiency is not a free scalar; unlike AG, the early allocation is irreversible over the final phase.",
        "materially_new_mechanism": "Phase-0 exposure builds a bounded bucket-specific representation primer that gates how efficiently phase-1 samples can update that capability.",
        "mechanistic_premise": "A short final phase can specialize efficiently only where pretraining has already allocated representation; late data with no early support must spend part of its budget establishing that representation.",
        "governing_equations": "u0_i=e0_i/e0_i^prop; r_i=1-exp(-kappa u0_i); q_i=epsilon+(1-epsilon)r_i; x_i=e0_i+q_i e1_i; Y=b+sum_i beta_i h(x_i/x_i^prop)+family debt+physical replay, beta>=0.",
        "latent_state": "A bounded phase-0 representation primer r_i in [0,1] and terminal effective evidence x_i measured in simulated epochs.",
        "state_transition": "Phase 0 builds r_i by first-order acquisition. During phase 1, evidence accumulates at efficiency q_i fixed by the phase-0 primer; physical replay remains separately conserved.",
        "response_link": "Nonnegative convex shortage debt in primer-gated effective evidence plus nonnegative physical duplicate-exposure harm.",
        "additional_degrees_of_freedom": "One global dimensionless primer rate kappa and one global residual-plasticity floor epsilon beyond the independently fitted nonnegative response head.",
        "units_and_symmetries": "u0, r, q, and evidence ratios are dimensionless; x is simulated epochs; response amplitudes have BPB units. Bucket relabeling within a predeclared family preserves the equation.",
        "single_phase_restriction": "Set w0=w1=w and fit the identical primer-gated form directly on one-phase outcomes. This is not equated with the algebraic restriction of a two-phase coefficient fit.",
        "starcoder_signature": "The WSD optimum should require nonzero early StarCoder weight to prime a larger late StarCoder allocation, while the 50/50 cosine optimum should remain nearer the diagonal. Epsilon=1 is the no-gating ablation.",
        "catastrophic_optimism_resolution": "A candidate that assigns a bucket only in the final phase receives less effective evidence than its physical exposure suggests, preventing late specialization from canceling severe early undercoverage.",
        "response_compression_resolution": "The gate expands differences among frontier schedules according to early support rather than applying an output calibrator.",
        "scale_transfer_expectation": "The primer rate should grow with effective optimization progress, while epsilon and the direction of early-support effects should be stable. Schedule changes act through physical phase exposure, not a fitted phase label.",
        "cheapest_falsification": "Epsilon selects the no-gating boundary, kappa is fold-unstable, or the model fails to improve both StarCoder optimum location and leave-region prediction relative to effective-exposure baselines.",
        "status": "active_preregistered",
        "status_evidence": "Preregistered before any EAGP fit or evaluation. No adversarial outcome will be read during form or hyperparameter selection.",
    }
    registry = registry.loc[~registry["id"].eq("EAGP")]
    registry = pd.concat([registry, pd.DataFrame([row], columns=registry.columns)], ignore_index=True)
    now = datetime.now(UTC).isoformat()
    rows = [
        {
            "timestamp": now,
            "round_id": "round_3_starcoder_rejection",
            "candidate_id": "MCR",
            "candidate_family": "Multi-rate component relaxation",
            "hyperparameters": "Family rates selected only by paired fit-panel and StarCoder OOF",
            "adversarial_outcomes_available_before_proposal": True,
            "adversarial_outcomes_inspected_before_proposal": True,
            "observations_inspiring_mechanism": "One-phase components were better identified than unconstrained phase heads.",
            "novelty_class": "Independent one-phase component states with an exact family-pooled relaxation semigroup",
            "evaluation_status": "rejected at fit-panel and StarCoder gates; adversarial panel not evaluated",
            "evidence_path": "round3_component_relaxation/report.md; round3_component_relaxation_starcoder/report.md",
            "notes": "Rates collapsed toward a shared fast state and were unstable across folds.",
        },
        {
            "timestamp": now,
            "round_id": "round_4_preregistration",
            "candidate_id": "EAGP",
            "candidate_family": "Early-allocation-gated plasticity",
            "hyperparameters": "Not selected; kappa, epsilon, shortage geometry, and ridge restricted to a preregistered non-adversarial grid",
            "adversarial_outcomes_available_before_proposal": True,
            "adversarial_outcomes_inspected_before_proposal": True,
            "observations_inspiring_mechanism": "The WSD StarCoder optimum has nonzero early rare-data support followed by a larger late allocation; radial heldout failures often combine early broad underexposure with late specialization.",
            "novelty_class": "Irreversible bucket-local early representation primer gating final-phase plasticity",
            "evaluation_status": "preregistered; adversarial evaluation forbidden until a later batch freeze",
            "evidence_path": "approach_registry.csv",
            "notes": "Epsilon=1 is the exact no-gating ablation.",
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
    print(registry.tail(5)[["id", "family", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()
