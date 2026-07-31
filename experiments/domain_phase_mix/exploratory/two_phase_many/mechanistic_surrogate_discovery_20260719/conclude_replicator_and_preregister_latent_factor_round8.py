# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2"]
# ///
"""Reject replicator capacity and preregister joint latent phase transport."""

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
    registry.loc[registry["id"].eq("HRC"), ["status", "status_evidence"]] = [
        "blocked_before_adversarial",
        "Rejected at the StarCoder gate without multi-swarm, historical, or adversarial evaluation. The conserved "
        "replicator state improves the no-capacity surface RMSE by 15.7% on cosine and 1.8% on WSD, but absolute "
        "OOF RMSE remains 0.142/0.116 BPB. Leave-region RMSE reaches 0.249/0.181 and Regret@1 reaches 0.344/0.234; "
        "the predicted optima miss the observed optima by 0.252/0.148 in policy distance. The representation-budget "
        "transition therefore does not recover phase-order geometry.",
    ]
    row = {
        "id": "JLPT",
        "family": "Joint latent phase transport",
        "relationship_to_prior": "Reopens PMVT/TPRB only through a materially new identification argument. PMVT "
        "fit each target independently and TPRB transferred a direction estimated from one source. JLPT jointly "
        "identifies a low-rank phase-response subspace from four coordinate-matched scale-target outputs while holding "
        "the same policy coordinates out from every output.",
        "materially_new_mechanism": "A small set of latent capability-displacement directions is shared across smooth "
        "targets and model scales. Each target-scale panel observes a different BPB loading on those same terminal "
        "capability perturbations.",
        "mechanistic_premise": "At fixed aggregate exposure, moving data between phases perturbs a common terminal "
        "capability state. Evaluation suites and model scales weight that state differently, but should not require "
        "unrelated bucket-level phase directions if the perturbation is physical.",
        "governing_equations": "a=alpha0*w0+alpha1*w1; d=alpha0*alpha1*(w1-w0); "
        "z_f=sum_{i in f} p_i[d_i/p_i]/(tau+a_i/p_i); Delta Y_q=z^T U lambda_q+chi_q q(z), "
        "rank(U lambda^T)=r and chi_q>=0; Y_q=F_q(a)+Delta Y_q.",
        "latent_state": "A rank-r vector of dimensionless terminal capability displacements U^T z along the phase "
        "fiber, plus a nonnegative finite-contrast magnitude q(z).",
        "state_transition": "The phase contrast transports the aggregate state once along shared latent directions. "
        "Remaining learnability scales transport by inverse relative aggregate exposure; finite contrast incurs a "
        "nonnegative second-order loss.",
        "response_link": "Each target-scale output has signed BPB loadings on the shared transport coordinates and a "
        "nonnegative BPB loading on contrast cost. There is no free phase intercept or output calibration layer.",
        "additional_degrees_of_freedom": "For three predeclared families and four outputs, rank r contributes "
        "r(3+4-r) identifiable coefficient degrees of freedom; optional contrast cost contributes four nonnegative "
        "loadings. Tau, rank, and ridge are selected by coordinate-grouped nested CV.",
        "units_and_symmetries": "a, d, z, and q are dimensionless; target loadings have BPB units. The low-rank "
        "coefficient product is identifiable although factor rotations are not; diagnostics use its singular subspace, "
        "not arbitrary factor signs.",
        "single_phase_restriction": "When w0=w1, d=z=q=0 exactly, so the two-phase correction vanishes. F_q is fitted "
        "independently on one-phase outcomes; algebraically tying a two-phase fit is reported separately.",
        "starcoder_signature": "A shared transport direction should orient both StarCoder valleys while schedule-specific "
        "loadings change their magnitude. The second-order cost should raise remote phase-contrast arms without moving "
        "the tied spine.",
        "catastrophic_optimism_resolution": "Low-rank joint identification prevents one target from assigning a large "
        "benefit to a phase direction unsupported by related panels; nonnegative contrast cost prevents unbounded "
        "off-diagonal reward.",
        "response_compression_resolution": "Target-specific loadings preserve each output's phase-response scale while "
        "shared directions pool coordinate-level evidence. The model must expand held-out variation through input state, "
        "not output calibration.",
        "scale_transfer_expectation": "The shared latent subspace should be stable across 300M and Delphi. Output "
        "loadings may vary with optimization progress; direction signs and principal angles should remain stable.",
        "cheapest_falsification": "Coordinate-grouped nested CV fails to improve a zero phase correction on at least "
        "three of four outputs, the selected rank is unstable, the shared subspace has low fold agreement, or an "
        "independent per-output ridge materially dominates it.",
        "status": "active_preregistered",
        "status_evidence": "Preregistered after observing only matched fit-panel phase-delta correlations and before "
        "fitting JLPT. Historical and adversarial outcomes are forbidden until the coordinate-grouped identification "
        "and StarCoder gates pass.",
    }
    registry = registry.loc[~registry["id"].eq("JLPT")]
    registry = pd.concat([registry, pd.DataFrame([row], columns=registry.columns)], ignore_index=True)

    now = datetime.now(UTC).isoformat()
    rows = [
        {
            "timestamp": now,
            "round_id": "round_7_starcoder_rejection",
            "candidate_id": "HRC",
            "candidate_family": "Homeostatic replicator capacity",
            "hyperparameters": "Frozen selection/homeostasis/replay/ridge grid; selection=0 exact ablation",
            "adversarial_outcomes_available_before_proposal": True,
            "adversarial_outcomes_inspected_before_proposal": True,
            "observations_inspiring_mechanism": "Concentrated heldout failures and additive cancellation.",
            "novelty_class": "Simplex-constrained representation allocation",
            "evaluation_status": "rejected at StarCoder gate; no multi-swarm, historical, or adversarial evaluation",
            "evidence_path": "round7_replicator_starcoder/report.md",
            "notes": "The conserved state does not recover either surface or leave-region geometry.",
        },
        {
            "timestamp": now,
            "round_id": "round_8_preregistration",
            "candidate_id": "JLPT",
            "candidate_family": "Joint latent phase transport",
            "hyperparameters": "Frozen tau/rank/ridge/contrast-cost grid; coordinate-grouped nested CV",
            "adversarial_outcomes_available_before_proposal": True,
            "adversarial_outcomes_inspected_before_proposal": True,
            "observations_inspiring_mechanism": "Phase deltas correlate across targets (0.78/0.70) and Uncheatable transfers across scale (0.67), while source-only TPRB directions fail.",
            "novelty_class": "Joint multi-output identification of shared phase-displacement subspace",
            "evaluation_status": "preregistered; no historical or adversarial evaluation before identification and StarCoder gates",
            "evidence_path": "approach_registry.csv",
            "notes": "All outputs at a held-out coordinate are excluded together; phase features vanish for tied policies.",
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
