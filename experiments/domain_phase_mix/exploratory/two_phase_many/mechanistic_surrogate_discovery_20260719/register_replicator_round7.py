# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2"]
# ///
"""Reject shared-state replay and preregister homeostatic replicator capacity."""

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
    registry.loc[registry["id"].eq("DMSR"), ["status", "status_evidence"]] = [
        "blocked_before_adversarial",
        "Rejected at the frozen core OOF gate. No shared shape is within 5% of every panel-specific optimum. The "
        "best equal-panel shape worsens Delphi Uncheatable/Table-9 RMSE by 13.7%/12.2%, despite remaining within "
        "4.4% at 300M and being optimal on production. Nonlinear retained-state shapes are not scale-invariant under "
        "this law.",
    ]
    row = {
        "id": "HRC",
        "family": "Homeostatic replicator capacity",
        "relationship_to_prior": "Reopens finite-capacity route AG with a materially new simplex-constrained nonlinear "
        "transition. AG linearly relaxed independent capacities toward current family mass; HRC uses competitive "
        "replicator selection plus homeostatic mutation, so allocation is conserved and path-dependent.",
        "materially_new_mechanism": "A finite representation budget is redistributed multiplicatively toward currently "
        "sampled buckets while a homeostatic process restores the proportional foundation allocation.",
        "mechanistic_premise": "Training examples do not only add independent evidence; they allocate a conserved "
        "representational budget. Sustained oversampling compounds a bucket's allocation, while replay of broad data "
        "and homeostatic pressure preserve capabilities that would otherwise be displaced.",
        "governing_equations": "z in simplex; r_i(t)=w_i(t)/p_i; dz_i/dt=kappa*z_i[r_i-sum_j z_j r_j]+mu[p_i-z_i]; "
        "u_i=1-exp(-e_i); x_i=u_i*z_i/p_i; Y=b-sum_i A_i log(1+x_i/(delta*x_i^prop))+sum_i H_i R(e_i), A,H>=0.",
        "latent_state": "A normalized finite-capacity allocation z_i on the bucket simplex and cumulative unique "
        "coverage u_i. x_i is coverage weighted by terminal allocated capacity relative to proportional.",
        "state_transition": "Replicator selection amplifies capacity for above-proportional sampling rates; mutation at "
        "rate mu returns capacity toward p. The ODE is integrated exactly to numerical tolerance through each phase.",
        "response_link": "Nonnegative logarithmic diminishing benefit in capacity-weighted unique coverage plus "
        "nonnegative physical replay harm. The response link is shared with the kappa=0 ablation.",
        "additional_degrees_of_freedom": "Two global dimensionless rates, selection kappa and homeostasis mu, beyond the "
        "nested kappa=0 response. Bucket response amplitudes are nonnegative and ridge-regularized.",
        "units_and_symmetries": "z, p, r, u, and x/x_prop are dimensionless; normalized training time makes kappa and mu "
        "dimensionless; response amplitudes have BPB units. The simplex and fixed z(0)=p remove capacity scale symmetry.",
        "single_phase_restriction": "For w0=w1, integrating the same autonomous ODE across either one interval or the "
        "two declared phases gives the same terminal state. The restricted response is also refit independently on "
        "one-phase outcomes.",
        "starcoder_signature": "Early specialist allocation can persist into the late phase, while a late broad phase can "
        "restore foundation capacity. Nonlinear conserved competition should tilt the Nike swoosh without a free phase "
        "head and should change predictably between 50/50 and 80/20 schedules.",
        "catastrophic_optimism_resolution": "Extreme concentration necessarily displaces capacity from absent buckets, "
        "so surplus evidence in one family cannot linearly cancel the resulting broad capability loss.",
        "response_compression_resolution": "Multiplicative selection expands differences among concentrated frontier "
        "policies in latent-state space instead of applying an output calibration slope.",
        "scale_transfer_expectation": "If normalized representation competition is structural, kappa/mu signs and order "
        "should transfer; absolute rates may depend on optimization progress and are explicitly audited across panels.",
        "cheapest_falsification": "StarCoder selects kappa=0, rates hit incompatible boundaries across schedules, "
        "leave-region prediction remains outside the surface Pareto baseline, or the raw optimum becomes a corner.",
        "status": "active_preregistered",
        "status_evidence": "Preregistered before any HRC state integration. StarCoder is the first gate; no multi-swarm, "
        "historical, or adversarial outcome will be evaluated unless the capacity state beats its exact kappa=0 ablation.",
    }
    registry = registry.loc[~registry["id"].eq("HRC")]
    registry = pd.concat([registry, pd.DataFrame([row], columns=registry.columns)], ignore_index=True)

    now = datetime.now(UTC).isoformat()
    rows = [
        {
            "timestamp": now,
            "round_id": "round_6_core_rejection",
            "candidate_id": "DMSR",
            "candidate_family": "Dimensionless multi-panel state replay",
            "hyperparameters": "One shape selected by equal-panel normalized fit-OOF RMSE",
            "adversarial_outcomes_available_before_proposal": True,
            "adversarial_outcomes_inspected_before_proposal": True,
            "observations_inspiring_mechanism": "Panel-specific nonlinear shape equifinality.",
            "novelty_class": "Cross-swarm joint identification of an existing retained-state law",
            "evaluation_status": "rejected at core OOF gate; historical and adversarial panels not evaluated",
            "evidence_path": "round6_shared_state/report.md",
            "notes": "No candidate shape satisfies the frozen 5% OOF constraint on all five panels.",
        },
        {
            "timestamp": now,
            "round_id": "round_7_preregistration",
            "candidate_id": "HRC",
            "candidate_family": "Homeostatic replicator capacity",
            "hyperparameters": "Frozen kappa/mu/replay-onset/ridge grids; kappa=0 exact ablation; StarCoder first",
            "adversarial_outcomes_available_before_proposal": True,
            "adversarial_outcomes_inspected_before_proposal": True,
            "observations_inspiring_mechanism": "Extreme heldout policies combine multi-bucket underexposure with concentrated surplus, while additive models allow cancellation; linear capacity relaxation previously ran to a fast boundary.",
            "novelty_class": "Simplex-constrained replicator-mutator representation allocation",
            "evaluation_status": "preregistered; no historical or adversarial evaluation before StarCoder promotion",
            "evidence_path": "approach_registry.csv",
            "notes": "Conserved multiplicative capacity allocation is a new transition law, not a concentration surcharge.",
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
