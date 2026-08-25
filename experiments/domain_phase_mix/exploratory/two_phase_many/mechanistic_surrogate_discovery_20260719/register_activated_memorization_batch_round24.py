# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2"]
# ///
"""Preregister activated annealing and recoverable memorization as one frozen batch."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
REGISTRY = OUTPUT_ROOT / "approach_registry.csv"
LEDGER = OUTPUT_ROOT / "data_use_ledger.csv"


def candidate_rows() -> list[dict[str, object]]:
    return [
        {
            "id": "AAGF",
            "family": "Activation-barrier annealing gradient flow",
            "relationship_to_prior": "Reopens optimizer-time task-potential flow OTTPF through a Kramers-style activation law. OTTPF assumed mobility proportional to learning rate; AAGF posits metastable feature/basin transitions with mobility exp(-B/eta). B=0 is the exact token-time gradient-flow ablation.",
            "materially_new_mechanism": "A finite activation barrier causes representation transitions to freeze sharply as LR temperature falls, rather than slowing linearly with LR mass.",
            "mechanistic_premise": "If feature or basin changes require crossing an optimization barrier, the LR schedule controls transition mobility exponentially. WSD retains high mobility throughout its long stable phase and then freezes terminal specialization; cosine loses mobility continuously.",
            "governing_equations": "dz/dt=-k exp[-B/max(eta(t),epsilon)] [(1-p)(z+1/2)+r p(z-1/2)]; Y=b+A[(1-q)(z_T+1/2)^2/2+q r(z_T-1/2)^2/2], A>=0. B=0 is exact token-time gradient flow.",
            "latent_state": "One dimensionless shared specialization coordinate z initialized at zero; the activated mobility is a transition law, not an output feature.",
            "state_transition": "Exact scalar gradient-flow relaxation through each phase using the fixed integral of Kramers mobility under the declared cosine or WSD LR trajectory.",
            "response_link": "One nonnegative terminal quadratic task potential amplitude and intercept; no direct barrier or schedule output term.",
            "additional_degrees_of_freedom": "Rare curvature r, relaxation k, dimensionless barrier B relative to peak LR, evaluation mixture q, amplitude, intercept, and ridge. The exact ablation removes B.",
            "units_and_symmetries": "Normalized time, z, weights, r, q, eta, and B under peak-LR normalization are dimensionless; k is inverse normalized time; A and b carry BPB. Fixed task optima and initialization remove affine symmetry.",
            "single_phase_restriction": "Tie the phase policies and integrate the same LR trajectory; the artificial policy boundary is irrelevant. Independently refit the same restricted law on tied data.",
            "starcoder_signature": "A common interior barrier should beat B=0 on both schedules, preserve the near-tied cosine optimum, and allow the off-diagonal WSD optimum because stable-phase mobility differs sharply.",
            "catastrophic_optimism_resolution": "The terminal convex task potential still penalizes remote specialization; activated freeze-out prevents late low-LR exposure from receiving arbitrary effective mass.",
            "response_compression_resolution": "Schedule-dependent freeze-out changes terminal-state separation physically rather than rescaling outputs; it must expand both surface ranges relative to B=0.",
            "scale_transfer_expectation": "B normalized by peak LR should transfer only when optimizer and LR scaling are comparable; the dimensionless barrier regime should agree across the two StarCoder schedules before multi-swarm use.",
            "cheapest_falsification": "B=0 wins either schedule; barriers hit a boundary or differ by over 4x; nested RMSE misses the corrected reference by over 5%; or either raw optimum is over 0.15 from observed.",
            "status": "active_preregistered",
            "status_evidence": "Frozen jointly with RMR before either fit; no historical or adversarial evaluation is permitted before both StarCoder audits finish.",
        },
        {
            "id": "RMR",
            "family": "Recoverable replay memorization",
            "relationship_to_prior": "Reopens unique-coverage plus replay B and replay-induced forgetting W through a materially new state. B priced duplicate mass directly; W used duplicates to destroy competence. RMR tracks a separate bounded memorization/generalization-gap load that can be washed out by other-domain updates.",
            "materially_new_mechanism": "Repeated examples accumulate a domain-specific memorization load only after finite-corpus coverage, while training on other data decays that load. The load is distinct from useful competence.",
            "mechanistic_premise": "Once examples repeat, updates can narrow the solution toward finite-sample idiosyncrasies; subsequent diverse/out-of-domain training can regularize or overwrite that memorization. Late replay is therefore more harmful than equally repeated early data followed by broad training.",
            "governing_equations": "dE_i/dt=c_i w_i(t); u_i=1-exp(-E_i); dm_i/dt=a w_i(1-exp(-E_i))(1-m_i)-b(1-w_i)m_i. Y is a nonnegative linear combination of log unique-coverage debt and excess m_i relative to proportional, plus an intercept.",
            "latent_state": "Per domain, cumulative physical exposure E_i, literal expected unique coverage u_i, and a bounded memorization load m_i initialized at zero.",
            "state_transition": "Finite-corpus occupancy determines duplicate probability; duplicate current-domain tokens accumulate m_i and other-domain tokens recover it. The ODE is autonomous under a tied policy and composes across artificial boundaries.",
            "response_link": "Nonnegative bucket-specific coverage-debt and memorization amplitudes. Memorization has a separate physical state but no unconstrained nonlinear output calibration.",
            "additional_degrees_of_freedom": "One global accumulation rate a, one recovery rate b, a coverage floor, 2m nonnegative response amplitudes, intercept, and ridge. a=0 is the exact unique-coverage-only ablation.",
            "units_and_symmetries": "E, u, m, weights, a, b, and normalized time are dimensionless; epoch coefficients convert token share to E; response amplitudes and intercept have BPB units. Bounded states and fixed occupancy remove scale symmetry.",
            "single_phase_restriction": "Tie phase weights and integrate the same autonomous occupancy/memorization law. Independently refit its response on one-phase observations; no coefficient is removed.",
            "starcoder_signature": "The high-repetition arms should rise, but early repetition followed by broad data should be less harmful than late repetition. Rates should agree across cosine and WSD and raw optima should remain in both valleys.",
            "catastrophic_optimism_resolution": "A heavily repeated specialist bucket carries a persistent positive memorization cost even after unique coverage saturates, while omitted buckets still incur coverage debt.",
            "response_compression_resolution": "The bounded load selectively expands the high-repetition tail without changing ordinary low-repetition predictions through a global calibrator.",
            "scale_transfer_expectation": "Occupancy is determined by simulated epochs and should transfer across scale; recovery rates per normalized token time should be comparable under the same data and schedule semantics.",
            "cheapest_falsification": "a=0 wins either schedule; accumulation or recovery hits a boundary or changes by over 4x; nested RMSE misses the corrected reference by over 5%; or raw optima miss either observed valley by over 0.15.",
            "status": "active_preregistered",
            "status_evidence": "Frozen jointly with AAGF before either fit; no historical or adversarial evaluation is permitted before both StarCoder audits finish.",
        },
    ]


def main() -> None:
    registry = pd.read_csv(REGISTRY)
    ledger = pd.read_csv(LEDGER)
    rows = candidate_rows()
    ids = {str(row["id"]) for row in rows}
    registry = registry.loc[~registry["id"].isin(ids)]
    registry = pd.concat([registry, pd.DataFrame(rows, columns=registry.columns)], ignore_index=True)
    registry.to_csv(REGISTRY, index=False)

    ledger_rows = [
        {
            "timestamp": datetime.now(UTC).isoformat(),
            "round_id": "round_24_batch_preregistration",
            "candidate_id": "AAGF",
            "candidate_family": "Activation-barrier annealing gradient flow",
            "hyperparameters": "curvature={0.5,1,2,4}; speed={0.25,1,4,16}; barrier={0,0.03,0.1,0.3,1,3}; eval={0.2,0.5,0.8}; ridge={0.1,1}; exact cosine/WSD LR trajectories",
            "adversarial_outcomes_available_before_proposal": True,
            "adversarial_outcomes_inspected_before_proposal": True,
            "observations_inspiring_mechanism": "WSD requires a late-phase effect absent in cosine; linear optimizer time and fast/slow consolidation failed, suggesting an annealing freeze-out invariant rather than another memory state.",
            "novelty_class": "Kramers activation barrier in the transition clock",
            "evaluation_status": "batch frozen before StarCoder evaluation",
            "evidence_path": "approach_registry.csv",
            "notes": "No historical/adversarial tuning; B=0 mandatory ablation.",
        },
        {
            "timestamp": datetime.now(UTC).isoformat(),
            "round_id": "round_24_batch_preregistration",
            "candidate_id": "RMR",
            "candidate_family": "Recoverable replay memorization",
            "hyperparameters": "accumulation={0,0.25,1,4,16}; recovery={0.25,1,4,16}; offset={0.03,0.1,0.3}; ridge={0.1,1}; 512 integration steps",
            "adversarial_outcomes_available_before_proposal": True,
            "adversarial_outcomes_inspected_before_proposal": True,
            "observations_inspiring_mechanism": "Existing replay models either add duplicate mass statically or destroy competence. A distinct recoverable memorization/generalization-gap state may price high-repetition tails without conflating useful learning.",
            "novelty_class": "Bounded replay-memorization load with out-of-domain washout",
            "evaluation_status": "batch frozen before StarCoder evaluation",
            "evidence_path": "approach_registry.csv",
            "notes": "No historical/adversarial tuning; accumulation=0 mandatory ablation.",
        },
    ]
    identity = ["round_id", "candidate_id", "evaluation_status"]
    existing = set(map(tuple, ledger[identity].itertuples(index=False, name=None)))
    additions = [row for row in ledger_rows if tuple(row[column] for column in identity) not in existing]
    if additions:
        ledger = pd.concat([ledger, pd.DataFrame(additions, columns=ledger.columns)], ignore_index=True)
        ledger.to_csv(LEDGER, index=False)
    print(registry.loc[registry["id"].isin(ids), ["id", "family", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()
