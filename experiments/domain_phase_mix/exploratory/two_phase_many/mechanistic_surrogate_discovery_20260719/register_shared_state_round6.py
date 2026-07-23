# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2"]
# ///
"""Reject transferred phase response and preregister shared-state replay."""

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
    registry.loc[registry["id"].eq("TPRB"), ["status", "status_evidence"]] = [
        "blocked_before_adversarial",
        "Rejected before historical or adversarial evaluation. On Delphi, the transferred correction improves the "
        "zero-correction tied-spine RMSE only from 0.01851 to 0.01838 on Uncheatable and 0.02849 to 0.02833 on "
        "Table-9. Phase-delta Spearman is 0.168/0.172, while target ridge selects the maximum screened value. The "
        "source direction therefore contains too little target-scale ranking information.",
    ]
    row = {
        "id": "DMSR",
        "family": "Dimensionless multi-panel state replay",
        "relationship_to_prior": "Reopens hierarchical phase replay through a materially new identification strategy, "
        "not a new feature. Earlier fits selected nonlinear retained-state shapes independently per target and swarm; "
        "DMSR requires one dimensionless transition and saturation law across 300M, Delphi 3e18, and production.",
        "materially_new_mechanism": "No new state variable. The new falsifiable claim is that normalized acquisition, "
        "retention, and replay onset are scale-free training dynamics and must therefore be shared across independent "
        "panels rather than re-estimated from each response surface.",
        "mechanistic_premise": "If retained exposure is a physical state rather than a flexible basis, its diminishing-"
        "returns exponent, normalized forgetting rate, late-evidence multiplier, and replay onset should be comparable "
        "across model scales and smooth targets. Only BPB response amplitudes may be target-specific.",
        "governing_equations": "x_i=exp[-lambda(1-w1_i)]e0_i+eta*e1_i; S_i=x_i^a; "
        "Y_{s,t}=b_{s,t}+A_{s,t}^T Phi(S,x), with shared (a,lambda,eta,tau) and panel-target-specific nonnegative A.",
        "latent_state": "Per-bucket retained effective exposure x_i and pooled family coverage/replay states Phi, all "
        "normalized in simulated epochs.",
        "state_transition": "Early exposure survives the late phase with absence-dependent exponential retention; late "
        "exposure enters with one global relative multiplier. The same transition parameters are used on every panel.",
        "response_link": "Hierarchically pooled nonnegative bucket/family benefit and family replay-harm amplitudes, with "
        "a target-specific intercept and ridge-selected amplitudes.",
        "additional_degrees_of_freedom": "No additional degrees of freedom over hierarchical phase replay. DMSR removes "
        "panel-specific nonlinear choices by sharing four dimensionless shape parameters; l2 and pooling shrinkage remain "
        "panel-target-specific nuisance choices selected without heldouts.",
        "units_and_symmetries": "x and replay onset are simulated epochs; a, lambda, and eta are dimensionless; response "
        "amplitudes have BPB units. Shared shape normalization removes state/amplitude rescaling as a panel-specific "
        "degree of freedom.",
        "single_phase_restriction": "For w0=w1, retention is evaluated on the same physical exposure path and the exact "
        "restricted design is refit on one-phase data with the shared shape. Algebraic restriction and independent "
        "one-phase refit are reported separately.",
        "starcoder_signature": "The same shape should recover both Nike-swoosh surfaces after refitting only nonnegative "
        "response amplitudes; schedule differences act through phase fractions and physical exposures, not a new shape.",
        "catastrophic_optimism_resolution": "Joint identification prevents a single fit panel from choosing an aggressive "
        "saturation or retention shape whose credits extrapolate beyond support. It does not add a distance penalty.",
        "response_compression_resolution": "Panel-specific amplitudes preserve each target's response scale while a shared "
        "state law tests whether compressed predictions came from target-specific shape equifinality.",
        "scale_transfer_expectation": "All four shape parameters should be stable across model scale and target if the state "
        "is mechanistic. Failure on either StarCoder schedule or a >5% core OOF regression falsifies that claim.",
        "cheapest_falsification": "No shared shape remains within 5% of every core panel's independently selected OOF "
        "RMSE, the shared shape misses either StarCoder schedule, or response coefficients change sign systematically "
        "across related panels.",
        "status": "active_preregistered",
        "status_evidence": "Preregistered before shared-shape screening. The shape pool, equal-panel normalized-RMSE "
        "objective, and panel-specific amplitude-only refits are frozen without historical or adversarial evaluation.",
    }
    registry = registry.loc[~registry["id"].eq("DMSR")]
    registry = pd.concat([registry, pd.DataFrame([row], columns=registry.columns)], ignore_index=True)

    now = datetime.now(UTC).isoformat()
    rows = [
        {
            "timestamp": now,
            "round_id": "round_5_fit_panel_rejection",
            "candidate_id": "TPRB",
            "candidate_family": "Transferred phase-response basis",
            "hyperparameters": "Source and target grids selected only by grouped fit-panel CV",
            "adversarial_outcomes_available_before_proposal": True,
            "adversarial_outcomes_inspected_before_proposal": True,
            "observations_inspiring_mechanism": "Matched-policy cross-scale phase effects and stronger one-phase transfer.",
            "novelty_class": "Cross-scale normalized terminal-utility-gradient identification",
            "evaluation_status": "rejected at cross-scale fit-panel gate; historical and adversarial panels not evaluated",
            "evidence_path": "round5_transferred_phase_response/report.md",
            "notes": "The correction barely beats a zero phase correction and does not rank phase deltas.",
        },
        {
            "timestamp": now,
            "round_id": "round_6_preregistration",
            "candidate_id": "DMSR",
            "candidate_family": "Dimensionless multi-panel state replay",
            "hyperparameters": "One shared 12-candidate retained-state shape; equal-panel normalized OOF RMSE; panel-specific l2 and pooling shrinkage",
            "adversarial_outcomes_available_before_proposal": True,
            "adversarial_outcomes_inspected_before_proposal": True,
            "observations_inspiring_mechanism": "Prior nonlinear shapes are equifinal and unstable across panels; independent target fits may be using transition parameters as response flexibility.",
            "novelty_class": "Cross-swarm joint identification of a dimensionless retained-state transition law",
            "evaluation_status": "preregistered; historical and adversarial evaluation forbidden until core and StarCoder gates pass",
            "evidence_path": "approach_registry.csv",
            "notes": "This route removes flexibility rather than adding a correction.",
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
    print(registry.tail(3)[["id", "family", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()
