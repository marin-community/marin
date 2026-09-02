# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas"]
# ///
"""Initialize the append-only approach registry and adversarial data-use ledger."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = SCRIPT_DIR.parent
PRIOR_REGISTRY = (
    RESEARCH_DIR / "reference_outputs/mechanistic_surrogate_discovery_20260717/final_synthesis/approach_registry.csv"
)
DEFAULT_OUTPUT = RESEARCH_DIR / "reference_outputs/mechanistic_surrogate_discovery_20260719"
FROZEN_LEDGER = DEFAULT_OUTPUT / "frozen_gate/data_use_ledger.csv"

REGISTRY_COLUMNS = (
    "id",
    "family",
    "relationship_to_prior",
    "materially_new_mechanism",
    "mechanistic_premise",
    "governing_equations",
    "latent_state",
    "state_transition",
    "response_link",
    "additional_degrees_of_freedom",
    "units_and_symmetries",
    "single_phase_restriction",
    "starcoder_signature",
    "catastrophic_optimism_resolution",
    "response_compression_resolution",
    "scale_transfer_expectation",
    "cheapest_falsification",
    "status",
    "status_evidence",
)


def research_relative(path: Path) -> str:
    """Return a portable path relative to the two-phase research directory."""
    return Path(os.path.relpath(path.resolve(), RESEARCH_DIR.resolve())).as_posix()


def prior_rows() -> list[dict[str, object]]:
    prior = pd.read_csv(PRIOR_REGISTRY).fillna("")
    output: list[dict[str, object]] = []
    for row in prior.to_dict(orient="records"):
        output.append(
            {
                "id": f"prior_{row['id']}",
                "family": row["family"],
                "relationship_to_prior": f"Exact route {row['id']} from the 2026-07-17 registry.",
                "materially_new_mechanism": "None; imported to prevent accidental reinvention.",
                "mechanistic_premise": row["premise"],
                "governing_equations": f"Transition: {row['state_transition']} Response: {row['response']}",
                "latent_state": row["latent_state"],
                "state_transition": row["state_transition"],
                "response_link": row["response"],
                "additional_degrees_of_freedom": row["additional_degrees_of_freedom"],
                "units_and_symmetries": row["units_and_symmetries"],
                "single_phase_restriction": row["single_phase_restriction"],
                "starcoder_signature": row["starcoder_signature"],
                "catastrophic_optimism_resolution": row["optimism_resolution"],
                "response_compression_resolution": "No new claim; see the prior rejection evidence.",
                "scale_transfer_expectation": "Rejected by the prior cross-panel audit.",
                "cheapest_falsification": row["cheapest_falsification"],
                "status": "rejected_prior_drive",
                "status_evidence": row["status_evidence"],
            }
        )
    return output


def active_rows() -> list[dict[str, object]]:
    return [
        {
            "id": "PMVT",
            "family": "Paired marginal-value transport",
            "relationship_to_prior": (
                "Uses a new paired aggregate/contrast identification strategy unavailable to prior routes H, R, "
                "and AB; it does not add a free phase head or output-level transition penalty."
            ),
            "materially_new_mechanism": (
                "The one-phase aggregate response and the temporal phase correction are identified from different "
                "observations: tied outcomes identify the aggregate spine; paired two-minus-one-phase outcomes "
                "identify transport along the constant-total-exposure phase fiber."
            ),
            "mechanistic_premise": (
                "Moving a fixed token allocation later changes terminal capability in proportion to the marginal "
                "learnability remaining at the aggregate exposure; large contrasts also incur nonnegative curvature loss."
            ),
            "governing_equations": (
                "a=alpha0*w0+alpha1*w1; d=alpha0*alpha1*(w1-w0); m_i(a)=1/(tau_f+E_i(a)); "
                "Delta Y=sum_f theta_f sum_{i in f} m_i(a)d_i + sum_f chi_f sum_{i in f}(m_i(a)d_i)^2, chi_f>=0."
            ),
            "latent_state": (
                "Aggregate physical exposure E_i(a) and its dimensionless marginal-learnability state "
                "m_i(a)=(tau_f+E_i(a))^-1."
            ),
            "state_transition": (
                "The phase-1 allocation transports the terminal state from the tied aggregate counterfactual by "
                "the first two terms of a saturating acquisition expansion along d."
            ),
            "response_link": "Y_2p=F_1p(a)+Delta Y; the aggregate spine F_1p is fit only on tied policies.",
            "additional_degrees_of_freedom": (
                "Per declared family: one signed transport coefficient theta_f and one nonnegative curvature "
                "coefficient chi_f; tau_f is selected without adversarial outcomes."
            ),
            "units_and_symmetries": (
                "E and tau are simulated epochs; m*d is dimensionless after exposure scaling. Delta is exactly zero "
                "when w0=w1. Reversing phase order flips the linear term but preserves the quadratic term."
            ),
            "single_phase_restriction": (
                "d=0 gives Y=F_1p(a) exactly. The algebraic restriction and independently fit one-phase spine are "
                "reported separately; under the paired protocol they coincide by construction."
            ),
            "starcoder_signature": (
                "A tilted Nike-swoosh valley: phase reversal changes the signed slope near the diagonal, while "
                "quadratic contrast cost raises both far-off-diagonal arms."
            ),
            "catastrophic_optimism_resolution": (
                "The aggregate spine prices physical underexposure using tied data; phase surplus cannot cancel an "
                "aggregate shortage because the temporal correction is estimated only from paired differences."
            ),
            "response_compression_resolution": (
                "The tied spine retains the observed one-phase dynamic range, while the contrast term adds rather "
                "than compresses measured phase-order variation."
            ),
            "scale_transfer_expectation": (
                "Dimensionless transport signs should transfer; magnitudes may vary with schedule through alpha0*alpha1 "
                "and with scale through remaining marginal learnability."
            ),
            "cheapest_falsification": (
                "Paired-difference CV fails to beat the zero-correction null, theta signs are unstable, or the quadratic "
                "term runs to a boundary on either StarCoder schedule."
            ),
            "status": "active_preregistered",
            "status_evidence": "Frozen before candidate fitting and adversarial evaluation round 1.",
        },
        {
            "id": "FCF",
            "family": "Family commutator flow",
            "relationship_to_prior": (
                "Unlike prior symmetric competition routes F, N, AO, and AM, this models the signed noncommutativity "
                "of ordered family updates and is antisymmetric under phase reversal."
            ),
            "materially_new_mechanism": (
                "A Baker-Campbell-Hausdorff commutator term represents the second-order difference between applying "
                "family update vector fields in early-late versus late-early order."
            ),
            "mechanistic_premise": (
                "Gradient-flow updates from different data families need not commute; order changes the terminal "
                "representation even when aggregate exposure is conserved."
            ),
            "governing_equations": (
                "Delta Y=alpha0*alpha1*sum_{f<g} K_fg*(W0_f*W1_g-W0_g*W1_f)*r_fg(a), K_fg=-K_gf, "
                "where r_fg(a) is a fixed remaining-learnability scale from the tied aggregate spine."
            ),
            "latent_state": "Family capability state acted on by mixture-dependent update vector fields V_f.",
            "state_transition": (
                "exp(alpha1 V(w1))exp(alpha0 V(w0)) is approximated through second order; the new term is the "
                "commutator [V(w1),V(w0)]."
            ),
            "response_link": "Y_2p=F_1p(a)+Delta Y with no free output calibrator.",
            "additional_degrees_of_freedom": (
                "For three predeclared broad-text, tech-code, and reasoning families: three signed K_fg coefficients."
            ),
            "units_and_symmetries": (
                "Family masses and r_fg are dimensionless; K has BPB units. Delta is zero for tied policies and "
                "changes sign exactly when the two phases are swapped."
            ),
            "single_phase_restriction": "The commutator vanishes identically, leaving the independently fit tied spine.",
            "starcoder_signature": (
                "A schedule-dependent rotation of the valley with equal and opposite response under phase reversal; "
                "the diagonal is unchanged."
            ),
            "catastrophic_optimism_resolution": (
                "Only if catastrophic errors are caused by order-sensitive cross-family interference; it cannot hide "
                "aggregate undercoverage because the tied spine is separate."
            ),
            "response_compression_resolution": (
                "Adds signed order range without changing the tied response range, directly testing missing phase-order variation."
            ),
            "scale_transfer_expectation": (
                "Commutator signs should be schedule- and scale-stable if gradient alignment is structural; magnitudes "
                "should scale with alpha0*alpha1 and remaining learnability."
            ),
            "cheapest_falsification": (
                "The antisymmetric StarCoder signal is absent, K signs disagree between schedules, or paired-difference "
                "CV does not beat a zero commutator."
            ),
            "status": "active_preregistered",
            "status_evidence": "Frozen before candidate fitting and adversarial evaluation round 1.",
        },
        {
            "id": "IFSC",
            "family": "Identified fast-slow consolidation",
            "relationship_to_prior": (
                "Reopens prior route AL only under a materially new paired identification strategy: tied outcomes "
                "identify acquisition/response, and paired phase differences identify retention/consolidation."
            ),
            "materially_new_mechanism": (
                "No new latent state relative to AL; novelty is the experimental identification equation. The route "
                "is admissible only if paired fitting removes the prior equifinality."
            ),
            "mechanistic_premise": (
                "Recent evidence controls a fast state while earlier evidence can survive through a slower consolidated state."
            ),
            "governing_equations": (
                "dot f_f=q W_f(1-f_f)-h(1-W_f)f_f; dot s_f=k(f_f-s_f); "
                "Y=b-sum_f A_f[(1-omega)f_f+omega s_f]+physical replay."
            ),
            "latent_state": "Per-family fast competence f_f and consolidated competence s_f.",
            "state_transition": "Exact two-phase integration of the coupled acquisition, forgetting, and consolidation ODE.",
            "response_link": "Nonnegative family capability benefit plus physical repetition harm.",
            "additional_degrees_of_freedom": (
                "Global dimensionless q,h,k,omega selected using paired CV, plus nonnegative family response amplitudes."
            ),
            "units_and_symmetries": (
                "Normalized time makes q,h,k dimensionless; f,s,omega are dimensionless. Tied constant schedules obey "
                "the autonomous semigroup law."
            ),
            "single_phase_restriction": (
                "Tie the phase policy and integrate the same autonomous ODE; refit response amplitudes on tied data."
            ),
            "starcoder_signature": (
                "Early specialist exposure survives through s while late exposure controls f, producing two recency scales."
            ),
            "catastrophic_optimism_resolution": (
                "A narrow late phase cannot instantly erase consolidated broad competence, but missing families still "
                "lose fast competence."
            ),
            "response_compression_resolution": (
                "Two independently identified state timescales can preserve a larger phase-order response range than a scalar retention state."
            ),
            "scale_transfer_expectation": (
                "Dimensionless time constants should transfer across token budgets only if consolidation is tied to "
                "normalized training progress; failure is expected if they are token-count-specific."
            ),
            "cheapest_falsification": (
                "The paired likelihood still selects boundary rates, bootstrap time constants are unstable, or either "
                "StarCoder leave-region-out audit reproduces route AL's failure."
            ),
            "status": "active_preregistered_reopened",
            "status_evidence": "Reopened solely because 238 paired Delphi aggregate counterfactuals are now available.",
        },
    ]


def append_ledger(output_dir: Path) -> None:
    ledger_path = output_dir / "data_use_ledger.csv"
    ledger = pd.read_csv(FROZEN_LEDGER)
    timestamp = datetime.now(UTC).isoformat()
    additions = []
    for candidate in active_rows():
        additions.append(
            {
                "timestamp": timestamp,
                "round_id": "round_1_preregistration",
                "candidate_id": candidate["id"],
                "candidate_family": candidate["family"],
                "hyperparameters": "not selected; selection restricted to non-adversarial evidence",
                "adversarial_outcomes_available_before_proposal": True,
                "adversarial_outcomes_inspected_before_proposal": True,
                "observations_inspiring_mechanism": (
                    "Matched one-phase aggregate counterfactuals, prior response-compression diagnosis, and prior "
                    "cross-swarm/StarCoder failures; no individual adversarial target value used."
                ),
                "novelty_class": candidate["materially_new_mechanism"],
                "evaluation_status": "preregistered; adversarial evaluation forbidden until batch freeze",
                "evidence_path": "approach_registry.csv",
                "notes": "All exposed adversarial outcomes are development evidence; any survivor requires a new sealed panel.",
            }
        )
    combined = pd.concat([ledger, pd.DataFrame(additions)], ignore_index=True, sort=False)
    combined.to_csv(ledger_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    registry = pd.DataFrame([*prior_rows(), *active_rows()], columns=REGISTRY_COLUMNS)
    registry.to_csv(args.output_dir / "approach_registry.csv", index=False)
    append_ledger(args.output_dir)
    summary = {
        "prior_routes": len(registry.loc[registry["status"].eq("rejected_prior_drive")]),
        "active_routes": len(registry.loc[registry["status"].str.startswith("active")]),
        "registry": research_relative(args.output_dir / "approach_registry.csv"),
        "ledger": research_relative(args.output_dir / "data_use_ledger.csv"),
    }
    (args.output_dir / "registry_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
