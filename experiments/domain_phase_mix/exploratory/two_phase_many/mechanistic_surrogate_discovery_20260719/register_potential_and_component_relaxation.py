# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2"]
# ///
"""Record round-two rejection and preregister component relaxation."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
REGISTRY = OUTPUT_ROOT / "approach_registry.csv"
LEDGER = OUTPUT_ROOT / "data_use_ledger.csv"


def upsert_rows(frame: pd.DataFrame, rows: list[dict[str, object]], key: str) -> pd.DataFrame:
    additions = pd.DataFrame(rows, columns=frame.columns)
    frame = frame.loc[~frame[key].isin(additions[key])]
    return pd.concat([frame, additions], ignore_index=True)


def append_ledger(frame: pd.DataFrame, rows: list[dict[str, object]]) -> pd.DataFrame:
    additions = pd.DataFrame(rows, columns=frame.columns)
    identity = ["round_id", "candidate_id", "evaluation_status"]
    existing = set(map(tuple, frame[identity].itertuples(index=False, name=None)))
    additions = additions.loc[
        [tuple(row) not in existing for row in additions[identity].itertuples(index=False, name=None)]
    ]
    return pd.concat([frame, additions], ignore_index=True)


def main() -> None:
    registry = pd.read_csv(REGISTRY)
    ledger = pd.read_csv(LEDGER)
    now = datetime.now(UTC).isoformat()
    common = {
        "relationship_to_prior": "Uses the independently fitted tied convex loss potential from round two; unlike PMVT and TEA, the temporal law is derived from the potential geometry rather than a free bucketwise phase head.",
        "mechanistic_premise": "A phase shift transports the terminal policy along the tied loss geometry, so signed work or dissipated potential should explain the phase-order correction.",
        "latent_state": "The scalar tied loss potential and its local gradient on the mixture simplex.",
        "units_and_symmetries": "Mixture weights and Bregman divergence are dimensionless after the BPB-valued potential is fitted; response coefficients map the derived geometric quantities to BPB. Tied schedules have exactly zero phase correction.",
        "single_phase_restriction": "Set w0=w1; displacement and phase correction vanish exactly, leaving an independently fitted tied potential.",
        "catastrophic_optimism_resolution": "Convex debt was intended to make radial undercoverage increasingly costly rather than allowing additive surplus cancellation.",
        "response_compression_resolution": "The geometric phase correction was intended to recover signed ordering range without calibrating the output.",
        "scale_transfer_expectation": "Dimensionless geometry could transfer only if the tied potential and normalized phase transport describe the same capability state across scales.",
        "cheapest_falsification": "Miss either StarCoder optimum or exceed the strongest surface baseline in leave-region prediction.",
        "status": "blocked_before_adversarial",
    }
    registry_rows = [
        {
            "id": "PWD",
            "family": "Potential work and Bregman dissipation",
            **common,
            "materially_new_mechanism": "Signed first-order work plus nonnegative second-order Bregman dissipation under an aggregate-preserving phase displacement.",
            "governing_equations": "a=alpha0*w0+alpha1*w1; d=alpha0*alpha1*(w1-w0); DeltaY=theta gradF(a)^T d + chi D_F(a+d,a), chi>=0.",
            "state_transition": "One aggregate-preserving displacement from a to a+d; the response is the first- and second-order change in the tied potential.",
            "response_link": "Y_2p=F_1p(a)+DeltaY with a zero-intercept phase head.",
            "additional_degrees_of_freedom": "Two phase coefficients after the nonnegative convex potential is fitted independently on tied data.",
            "starcoder_signature": "A gradient-aligned signed tilt plus convex curvature around the phase-tied valley.",
            "status_evidence": "Rejected before adversarial evaluation. Cosine/WSD OOF RMSE is 0.1259/0.0908; WSD Regret@1 is 0.2514. Leave-region RMSE reaches 0.1699/0.1731, and the WSD optimum is predicted near tied at (0.265,0.270) rather than the observed (0.145,0.517).",
        },
        {
            "id": "ESR",
            "family": "Equilibrium stress relaxation",
            **common,
            "materially_new_mechanism": "A scalar stress state relaxes toward the tied potential of each phase, with an exact two-phase transition.",
            "governing_equations": "dq/dt=k(F(w)-q); DeltaY=A[e^(-k alpha1)(1-e^(-k alpha0))(F(w0)-F(a))+(1-e^(-k alpha1))(F(w1)-F(a))], A>=0.",
            "state_transition": "Exact constant-input relaxation in each phase.",
            "response_link": "Y_2p=F_1p(a)+DeltaY.",
            "additional_degrees_of_freedom": "One selected dimensionless rate and one nonnegative BPB response amplitude.",
            "starcoder_signature": "A recency-weighted tilt of a tied loss valley, stronger under an 80/20 schedule.",
            "status_evidence": "Rejected before adversarial evaluation. Cosine/WSD OOF RMSE is 0.1074/0.0695; leave-region RMSE is 0.0777-0.1071. Both raw optima are predicted near tied, and WSD misses the observed asymmetric optimum by Euclidean distance 0.279.",
        },
        {
            "id": "MCR",
            "family": "Multi-rate component relaxation",
            "relationship_to_prior": "Reopens phase-specific response route H and relaxation route AI only with a materially new latent decomposition: independently identified one-phase loss components obey an exact shared semigroup instead of receiving free phase heads or tracking mixture coordinates.",
            "materially_new_mechanism": "Each mechanistic one-phase loss component has its own family-pooled temporal relaxation rate. Different error components can preserve early information or follow late data, yielding two-phase synergy without unconstrained phase coefficients.",
            "mechanistic_premise": "Evaluation BPB decomposes into shortage and repetition error components whose equilibration times differ by semantic capability family.",
            "governing_equations": "F(w)=b+sum_j beta_j phi_j(w), beta_j>=0; ds_j/dt=kappa_g(phi_j(w)-s_j); Y_2p=b+sum_j beta_j[c_g phi_j(w0)+(1-c_g)phi_j(w1)], c_g=e^(-kappa_g alpha1)(1-e^(-kappa_g alpha0))/(1-e^(-kappa_g)).",
            "latent_state": "One dimensionless equilibrium-normalized loss-component state s_j per tied-response feature, with rates pooled by predeclared semantic family.",
            "state_transition": "Exact first-order relaxation toward the current phase's component equilibrium; no output calibration or candidate identity enters.",
            "response_link": "A nonnegative BPB-valued linear sum of terminal component states plus an intercept fitted only from one-phase outcomes.",
            "additional_degrees_of_freedom": "One nonnegative dimensionless relaxation rate per predeclared family; tied-response amplitudes are fitted independently and are not refit on phase outcomes.",
            "units_and_symmetries": "phi and s are dimensionless; beta has BPB units; kappa is per normalized training horizon. Rates are invariant to arbitrary subdivision of a constant policy. Components within a family share kappa.",
            "single_phase_restriction": "For w0=w1=w, c phi(w)+(1-c)phi(w)=phi(w), exactly recovering the independently fitted F(w). The algebraic restriction and one-phase refit are therefore explicit and distinct.",
            "starcoder_signature": "An interior Nike swoosh can arise when broad and rare error components have different rates: a slow broad component retains phase 0 while a fast rare component follows phase 1. Rates should change coherently between cosine 50/50 and WSD 80/20.",
            "catastrophic_optimism_resolution": "Radial shortage is inherited from the convex tied potential; late specialization cannot erase slow broad-family debt because each component remains separately charged.",
            "response_compression_resolution": "Different component time constants can expand phase-order response range while retaining the one-phase dynamic range exactly.",
            "scale_transfer_expectation": "Rates normalized by the training horizon should be comparable across swarms only when optimization progress is comparable; family rate ordering, not necessarily magnitude, is the transferable claim.",
            "cheapest_falsification": "Fail to recover both StarCoder surface optima with stable family-rate ordering, or fail to improve paired phase-delta prediction over the zero-correction tied model on at least two fit panels.",
            "status": "active_preregistered",
            "status_evidence": "Preregistered before any MCR fit or evaluation. Adversarial outcomes will not be read unless the form and hyperparameters survive non-adversarial gates and are frozen.",
        },
    ]
    registry = upsert_rows(registry, registry_rows, "id")
    ledger_rows = [
        {
            "timestamp": now,
            "round_id": "round_2_starcoder_rejection",
            "candidate_id": candidate,
            "candidate_family": family,
            "hyperparameters": "Selected exclusively by tied/paired fit-panel and StarCoder OOF",
            "adversarial_outcomes_available_before_proposal": True,
            "adversarial_outcomes_inspected_before_proposal": True,
            "observations_inspiring_mechanism": "Prior exposed failures motivated convex debt, but no adversarial targets were read during round-two selection.",
            "novelty_class": "Convex tied potential with a derived temporal law",
            "evaluation_status": "rejected at StarCoder gate; adversarial panel not evaluated",
            "evidence_path": "round2_potential_phase/report.md; round2_potential_starcoder/report.md",
            "notes": "The route missed WSD asymmetry and did not beat the existing surface Pareto baseline.",
        }
        for candidate, family in (
            ("PWD", "Potential work and Bregman dissipation"),
            ("ESR", "Equilibrium stress relaxation"),
        )
    ]
    ledger_rows.append(
        {
            "timestamp": now,
            "round_id": "round_3_preregistration",
            "candidate_id": "MCR",
            "candidate_family": "Multi-rate component relaxation",
            "hyperparameters": "Not selected; family rates restricted to a preregistered nonnegative grid and selected without adversarial outcomes",
            "adversarial_outcomes_available_before_proposal": True,
            "adversarial_outcomes_inspected_before_proposal": True,
            "observations_inspiring_mechanism": "PWD/ESR miss the WSD asymmetric optimum; existing free phase heads compress adversarial response; one-phase components are better identified than a full two-phase surface.",
            "novelty_class": "Independent one-phase component states with exact family-pooled relaxation semigroup",
            "evaluation_status": "preregistered; adversarial evaluation forbidden until a later batch freeze",
            "evidence_path": "approach_registry.csv",
            "notes": "No MCR fit or candidate-specific result existed at preregistration time.",
        }
    )
    ledger = append_ledger(ledger, ledger_rows)
    registry.to_csv(REGISTRY, index=False)
    ledger.to_csv(LEDGER, index=False)
    print(registry.tail(7)[["id", "family", "status"]].to_string(index=False))
    print(ledger.tail(5)[["round_id", "candidate_id", "evaluation_status"]].to_string(index=False))


if __name__ == "__main__":
    main()
