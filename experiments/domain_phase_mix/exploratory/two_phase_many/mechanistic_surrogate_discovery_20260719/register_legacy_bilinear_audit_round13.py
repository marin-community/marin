# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2"]
# ///
"""Preregister a rigorous audit of the legacy bilinear phase-state model."""

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
        "id": "LPSI",
        "family": "Legacy low-rank phase-state interaction",
        "relationship_to_prior": "This is an exact audit of benchmark_low_rank_phase_state_interaction.py, not a new candidate. It differs from rejected low-rank family competition by placing a signed bilinear term in the response, and from rejected JLPT by fitting one target at a time without a shared cross-output subspace.",
        "materially_new_mechanism": "None in this drive. The purpose is to determine whether an unresolved legacy result is a stable shape diagnostic before deriving any lower-dimensional mechanism from it.",
        "mechanistic_premise": "Useful early and late capabilities may interact multiplicatively: an early representation direction changes the marginal value of a late representation direction.",
        "governing_equations": "s_t=1-exp(-rho_t e_t); L=L_sep(e_0,e_1)+(s_0^T u)(s_1^T v) for rank one, or the sum of r such products. L_sep is the existing nonnegative separate-head exposure bowl.",
        "latent_state": "Per-bucket bounded learned-state coordinates s_0 and s_1. The invariant interaction is the matrix M=sum_j u_j v_j^T.",
        "state_transition": "No causal transition beyond independent saturating phase states; phase interaction occurs only in the terminal response. This is a mechanistic weakness that the audit must expose rather than conceal.",
        "response_link": "A nonnegative additive separate-head bowl plus one signed low-rank bilinear form in bounded early and late states.",
        "additional_degrees_of_freedom": "Rank r adds 2rm factor coefficients, equivalent to a rank-r m-by-m interaction matrix. Total nominal count is (4+2r)m+1; rank one therefore has 13 parameters for StarCoder and 235 for a 39-bucket swarm.",
        "units_and_symmetries": "Exposure and states are dimensionless; bowl and interaction coefficients have BPB units. u and v have reciprocal scale ambiguity and joint sign ambiguity; only M=uv^T is identified. Rank greater than one additionally has rotation ambiguity.",
        "single_phase_restriction": "Tying phase weights evaluates the same bilinear response on tied inputs, but does not collapse to the independently fitted one-phase separate-head model. A future candidate would need to justify or remove this mismatch.",
        "starcoder_signature": "A stable rank-one M should rotate the Nike-swoosh valley on both schedules, retain similar normalized interaction orientation across folds and starts, and place the raw optimum near the observed valley.",
        "catastrophic_optimism_resolution": "A signed cross-phase interaction can price joint exposure patterns that additive heads miss, but bounded states also cap its penalty; no optimism-resolution claim is accepted before shape and identifiability gates pass.",
        "response_compression_resolution": "The bilinear form expands response range only through a physical interaction of two learned states, not an output calibrator. It can nevertheless create unsupported rewards if M is unstable.",
        "scale_transfer_expectation": "If the interaction is physical, the normalized invariant M should be stable across cosine and WSD schedules even if its BPB scale changes. Per-bucket factors are not expected to transfer to 39 buckets without pooling.",
        "cheapest_falsification": "On both updated StarCoder surfaces: nested RMSE must be within 5% of the existing shape frontier; rank one must win globally and in at least three of five outer folds; normalized M must have pairwise fold cosine at least 0.8 and norm CV at most 0.5; near-optimal starts must agree at cosine at least 0.95; and the raw optimum must lie within 0.15 of the observed best coordinate.",
        "status": "legacy_diagnostic_preregistered",
        "status_evidence": "Frozen before the updated-surface audit. The old evidence used 116 cosine rows, one CV seed, and no WSD or invariant-factor stability test.",
    }
    registry = registry.loc[~registry["id"].eq("LPSI")]
    registry = pd.concat([registry, pd.DataFrame([row], columns=registry.columns)], ignore_index=True)
    registry.to_csv(REGISTRY, index=False)

    ledger_row = {
        "timestamp": datetime.now(UTC).isoformat(),
        "round_id": "round_13_legacy_audit_preregistration",
        "candidate_id": "LPSI",
        "candidate_family": "Legacy low-rank phase-state interaction",
        "hyperparameters": "Frozen rank={0,1,2}; interaction_l2={1e-4,1e-3,1e-2,1e-1,1,10}; base_l2=0.1; five outer and four inner folds; eight deterministic starts",
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": True,
        "observations_inspiring_mechanism": "An unresolved 2026-07-10 result reported StarCoder OOF RMSE 0.0536 on 116 rows, but lacked the current WSD panel, nested selection, and factor-identifiability diagnostics.",
        "novelty_class": "Legacy diagnostic only; no new mechanism claimed",
        "evaluation_status": "preregistered for updated StarCoder shape and identifiability gates; no multi-swarm or adversarial evaluation",
        "evidence_path": "approach_registry.csv",
        "notes": "Passing only licenses derivation of a parsimonious identified mechanism. It does not promote the 235-parameter 39-bucket form.",
    }
    identity = ["round_id", "candidate_id", "evaluation_status"]
    existing = set(map(tuple, ledger[identity].itertuples(index=False, name=None)))
    if tuple(ledger_row[column] for column in identity) not in existing:
        ledger = pd.concat([ledger, pd.DataFrame([ledger_row], columns=ledger.columns)], ignore_index=True)
        ledger.to_csv(LEDGER, index=False)
    print(registry.tail(1)[["id", "family", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()
