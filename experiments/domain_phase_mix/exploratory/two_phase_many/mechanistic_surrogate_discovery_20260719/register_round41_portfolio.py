# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2"]
# ///
"""Preregister the round-41 mechanistic portfolio before evaluating it."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
REGISTRY = OUTPUT_ROOT / "approach_registry.csv"
LEDGER = OUTPUT_ROOT / "data_use_ledger.csv"


def _registry_rows() -> list[dict[str, object]]:
    return [
        {
            "id": "FNSMF",
            "family": "Finite Newton-Schulz Muon flow",
            "relationship_to_prior": (
                "Reopens blocked MAPTF only through the optimizer's actual transition map. MAPTF replaced Muon's "
                "five finite quintic Newton-Schulz iterations with an exact SVD polar factor; the finite map retains "
                "singular-value-ratio information that the exact polar ablation removes."
            ),
            "materially_new_mechanism": "Finite polynomial singular-value dynamics in the declared Muon update.",
            "mechanistic_premise": (
                "Muon does not compute an exact polar factor. Its fixed five-step Newton-Schulz polynomial maps "
                "different gradient singular spectra to different update spectra, so domain order can affect the "
                "representation even when exact polar flow erases that information."
            ),
            "governing_equations": (
                "G_p=(1-p)(W-T_B)C_B+r p(W-T_R)C_R; X_0=G_p/(||G_p||_F+eps); "
                "A_k=X_k X_k^T; X_{k+1}=a_k X_k+(b_k A_k+c_k A_k^2)X_k for the five source quintic "
                "coefficient triples; dW/dtau=-k[I-W W^T_F]X_5; "
                "Y=b+A[(1-q)L_B(W_T)+q r L_R(W_T)], A>=0."
            ),
            "latent_state": "One dimensionless 2x2 constant-Frobenius-norm representation matrix W.",
            "state_transition": (
                "Autonomous tangent flow driven by the exact source-defined finite Newton-Schulz polynomial. "
                "Exact polar, vector-normalized, and Euclidean maps are mandatory nested ablations."
            ),
            "response_link": "One nonnegative BPB amplitude on covariance-weighted terminal task debt plus an intercept.",
            "additional_degrees_of_freedom": (
                "No transition degree of freedom beyond MAPTF: Newton-Schulz coefficients, iteration count, and "
                "epsilon are fixed from MuonH source. Task angle, covariance anisotropy, relaxation, target mix, and "
                "ridge use a frozen finite grid."
            ),
            "units_and_symmetries": (
                "W, targets, unit-trace covariances, weights, and normalized optimizer time are dimensionless; k is "
                "inverse optimizer time; A and b have BPB units. Fixed initialization, covariance trace, and target "
                "orientation remove scale and rotation symmetries."
            ),
            "single_phase_restriction": (
                "For tied phase weights the autonomous finite-map flow composes across the artificial boundary. The "
                "identical restricted law can be selected and refit independently on tied policies."
            ),
            "starcoder_signature": (
                "The fixed finite map must beat exact polar and normalized ablations on both schedules, retain the "
                "cosine near-diagonal optimum, and recover WSD late-code enrichment."
            ),
            "catastrophic_optimism_resolution": (
                "Remote policies incur terminal task debt after a bounded representation trajectory; retained "
                "singular-spectrum information must price neglected task directions rather than calibrating outputs."
            ),
            "response_compression_resolution": (
                "Finite singular-value shaping can separate policies whose mean gradient directions are similar but "
                "whose spectra differ, expanding the physical terminal-state range."
            ),
            "scale_transfer_expectation": (
                "The Newton-Schulz map is optimizer-defined and transfers exactly when MuonH settings match. "
                "Dimensionless task geometry may transfer; total relaxation may vary with optimizer progress."
            ),
            "cheapest_falsification": (
                "Block before multi-swarm work unless the finite map beats all exact update-map ablations globally and "
                "in at least three folds on both StarCoder schedules, clears both shape references, and places both raw "
                "optima within 0.15."
            ),
            "status": "active_preregistered",
            "status_evidence": "Frozen before any round-41 evaluation.",
        },
        {
            "id": "BCNSF",
            "family": "Batch-composition Newton-Schulz flow",
            "relationship_to_prior": (
                "Reopens neither JARA nor SGDDD. JARA curved deterministic sampling rates and SGDDD propagated output-"
                "priced trajectory variance. BCNSF instead averages the nonlinear optimizer map over the source-derived "
                "within-step mixture-composition distribution; its exact ablation maps the mean gradient once."
            ),
            "materially_new_mechanism": "Jensen drift from stochastic batch composition before Muon's nonlinear map.",
            "mechanistic_premise": (
                "Levanter fixes source counts in each 2,048-sequence mixture block, permutes them, and consumes "
                "128-sequence global batches. Therefore each step has hypergeometric source composition. Since Muon's "
                "finite zeroth-power map is nonlinear, E[Psi(G_batch)] need not equal Psi(E[G_batch])."
            ),
            "governing_equations": (
                "K~Hypergeom(N=2048,M=M(p),B=128); G_K=(1-K/B)G_B+(K/B)G_R; "
                "F_B(W,p)=E_K[NS5(G_K)]; dW/dtau=-k[I-W W^T_F]F_B; "
                "Y=b+A[(1-q)L_B(W_T)+q r L_R(W_T)], A>=0."
            ),
            "latent_state": "One constant-norm 2x2 representation W; batch composition is integrated out from the transition.",
            "state_transition": (
                "Autonomous expected finite-Newton-Schulz tangent flow under the exact marginal hypergeometric batch law. "
                "Applying NS5 once to the mean composition is the exact nested ablation."
            ),
            "response_link": "One nonnegative amplitude on terminal task debt plus an intercept; no variance feature enters the output.",
            "additional_degrees_of_freedom": (
                "Zero stochastic-strength parameters. Batch size 128, block size 2,048, deterministic count rounding, "
                "Newton-Schulz coefficients, and iteration count are fixed by source."
            ),
            "units_and_symmetries": (
                "K and counts are sequences; K/B, W, targets, covariances, and normalized time are dimensionless. "
                "Response amplitude and intercept have BPB units. The exact loader law fixes stochastic scale."
            ),
            "single_phase_restriction": (
                "A tied policy supplies one stationary hypergeometric law throughout training; splitting its integration "
                "at a phase boundary has no effect. The same restricted law can be refit on tied observations."
            ),
            "starcoder_signature": (
                "The stochastic-composition law should matter most near opposing-gradient cancellation and at low rare "
                "counts, beat the mean-composition map on both schedules, and produce a larger ordering effect under WSD."
            ),
            "catastrophic_optimism_resolution": (
                "If mean-gradient cancellation made extreme policies appear too efficient, the expected nonlinear map "
                "retains finite stochastic updates and moves them to a higher-debt terminal state."
            ),
            "response_compression_resolution": (
                "Policy-dependent composition variance changes the transition itself and can widen terminal debt without "
                "an output variance coefficient."
            ),
            "scale_transfer_expectation": (
                "The law transfers through declared global batch and mixture-block sizes. Its magnitude should shrink with "
                "larger batches and must not be represented by a freely fitted strength."
            ),
            "cheapest_falsification": (
                "First reject algebraically if the expected-map correction is negligible across plausible states at "
                "B=128. Otherwise reject unless it beats the mean-composition ablation on both StarCoder schedules and "
                "fixes both shape and raw-optimum gates."
            ),
            "status": "active_preregistered",
            "status_evidence": "Frozen after source audit and before any batch-composition target evaluation.",
        },
        {
            "id": "DMACF",
            "family": "Dual Muon-Adam capability flow",
            "relationship_to_prior": (
                "Reopens DLSF through a materially different optimizer-channel transition. DLSF used ordinary gradient "
                "flow for a scalar shared feature and both task heads. DMACF places a shared vector representation on "
                "MuonH's constant-norm manifold while task readouts follow a separate Adam-like Euclidean channel."
            ),
            "materially_new_mechanism": "Coupled capability learning through optimizer-specific matrix and readout channels.",
            "mechanistic_premise": (
                "MuonH trains linear matrices with normalized constant-norm updates, but embeddings, biases, and the LM "
                "head use Adam/AdamH. Broad data can orient shared features while late specialist data adapts its readout; "
                "the two channels have different clocks and invariants."
            ),
            "governing_equations": (
                "L_j=(h_j u^T t_j-1)^2/2; g_u=(1-p)grad_u L_B+p grad_u L_R; "
                "du/dtau=-k_M(I-u u^T)g_u/(||g_u||+eps); "
                "dh_B/dtau=-k_A(1-p)partial_hB L_B; dh_R/dtau=-k_A p partial_hR L_R; "
                "Y=b+A[(1-q)L_B+qL_R], A>=0."
            ),
            "latent_state": "A unit-norm two-dimensional shared representation u and two scalar task readouts h_B,h_R.",
            "state_transition": (
                "Autonomous coupled deep-linear task flow with a Muon tangent-normalized feature channel and Euclidean "
                "readout channel. Freezing u and setting the channels equal are mandatory ablations."
            ),
            "response_link": "One nonnegative BPB amplitude on evaluation-weighted terminal broad/rare task loss plus intercept.",
            "additional_degrees_of_freedom": (
                "Task angle, total Muon relaxation, Adam-to-Muon rate ratio, evaluation mix, and ridge. Equal fixed initial "
                "readouts remove deep-linear rescaling; no task-specific response amplitudes are allowed."
            ),
            "units_and_symmetries": (
                "u, t_j, h_j, weights, rates in normalized time, and losses are dimensionless; A and b have BPB units. "
                "Unit norm and fixed positive initialization remove scale and sign symmetries."
            ),
            "single_phase_restriction": (
                "Tied weights produce one autonomous three-state law; an artificial boundary has no effect. The same "
                "restricted form must also be independently selected and fit on tied data."
            ),
            "starcoder_signature": (
                "WSD should use early broad exposure to orient u and late code exposure to adapt h_R, whereas cosine can "
                "remain closer to tied. The active optimizer split must beat frozen-feature and equal-channel ablations."
            ),
            "catastrophic_optimism_resolution": (
                "A concentrated policy leaves either shared orientation or one task readout unresolved and pays explicit "
                "terminal task loss."
            ),
            "response_compression_resolution": (
                "Feature/readout mismatch separates policies with similar aggregate exposure but different phase order "
                "through terminal task error, not output calibration."
            ),
            "scale_transfer_expectation": (
                "The optimizer-channel distinction is architecture- and optimizer-defined; the dimensionless relative "
                "clock should track the declared Adam/Muon LR ratio, while total relaxation may vary with scale."
            ),
            "cheapest_falsification": (
                "Reject unless the split channel beats both ablations on both StarCoder schedules, has a stable interior "
                "rate ratio, clears both shape references, and places both raw optima within 0.15."
            ),
            "status": "active_preregistered",
            "status_evidence": "Frozen before any round-41 evaluation.",
        },
        {
            "id": "MKBIF",
            "family": "Matrix Kalman-Bucy information flow",
            "relationship_to_prior": (
                "Reopens rejected scalar Kalman uncertainty AK only through a matrix-valued orientation state. Scalar AK "
                "cannot encode noncommuting information directions; MKBIF's SPD covariance and domain-specific Fisher "
                "and process matrices introduce an identifiable orientation invariant."
            ),
            "materially_new_mechanism": "Noncommuting matrix information acquisition and interference in posterior covariance.",
            "mechanistic_premise": (
                "Different data domains constrain different representation directions. Training accumulates information "
                "in observed directions while distribution-specific interference reintroduces uncertainty elsewhere. "
                "The Riccati flows need not commute, so phase order changes terminal uncertainty."
            ),
            "governing_equations": (
                "F(p)=(1-p)F_B+r p F_R; Q(p)=q[(1-p)Q_B+pQ_R]; "
                "dV/dtau=k[Q(p)-V F(p)V]; Y=b+A tr(F_eval V_T), A>=0, "
                "F_eval=(1-s)F_B+sF_R. q=0 and isotropic matrices are nested ablations."
            ),
            "latent_state": "A symmetric positive-definite 2x2 posterior covariance V over shared capabilities.",
            "state_transition": (
                "Autonomous matrix Riccati/Kalman-Bucy covariance flow through each phase, preserving positive definiteness. "
                "Information contracts uncertainty; process interference injects it in domain-oriented directions."
            ),
            "response_link": "One nonnegative BPB amplitude on evaluation-weighted posterior uncertainty plus intercept.",
            "additional_degrees_of_freedom": (
                "Information anisotropy and angle, dimensionless process-to-information ratio, total relaxation, evaluation "
                "mix, and ridge. Unit traces and fixed initial V remove scale non-identifiability."
            ),
            "units_and_symmetries": (
                "After unit-trace normalization, V, F, Q, weights, time, and rates are dimensionless; A and b have BPB "
                "units. Fixed broad eigenbasis and angle range remove rotation/reflection symmetry."
            ),
            "single_phase_restriction": (
                "A tied mixture defines one autonomous Riccati flow for the full duration, so the artificial boundary "
                "cancels exactly. The same matrix law can be selected and refit independently on tied data."
            ),
            "starcoder_signature": (
                "Nonzero interior process interference and anisotropy must beat q=0 and scalar/isotropic ablations on "
                "both schedules; WSD should favor late reduction of code-oriented uncertainty."
            ),
            "catastrophic_optimism_resolution": (
                "Unobserved or repeatedly disrupted capability directions retain large covariance and therefore explicit "
                "terminal debt under remote policies."
            ),
            "response_compression_resolution": (
                "Matrix orientation can expand uncertainty differences among frontier policies without a free output "
                "calibrator."
            ),
            "scale_transfer_expectation": (
                "Dimensionless information geometry should be comparable across schedules and scales; total information "
                "rate may grow with tokens per parameter, while process-to-information ratio must retain sign and order."
            ),
            "cheapest_falsification": (
                "Reject unless noncommuting matrix flow beats scalar/isotropic and zero-process ablations on both "
                "StarCoder schedules, has stable interior ratios, clears both shape references, and gives plausible raw optima."
            ),
            "status": "active_preregistered",
            "status_evidence": "Frozen before any round-41 evaluation.",
        },
    ]


def main() -> None:
    registry = pd.read_csv(REGISTRY)
    rows = _registry_rows()
    ids = {str(row["id"]) for row in rows}
    registry = registry.loc[~registry["id"].isin(ids)]
    registry = pd.concat([registry, pd.DataFrame(rows, columns=registry.columns)], ignore_index=True)
    registry.to_csv(REGISTRY, index=False)

    ledger = pd.read_csv(LEDGER)
    now = datetime.now(UTC).isoformat()
    ledger_rows = []
    for row in rows:
        ledger_rows.append(
            {
                "timestamp": now,
                "round_id": "round_41_portfolio_preregistration",
                "candidate_id": row["id"],
                "candidate_family": row["family"],
                "hyperparameters": "Frozen finite grids documented in the candidate-specific audit; exact source constants are not tuned.",
                "adversarial_outcomes_available_before_proposal": True,
                "adversarial_outcomes_inspected_before_proposal": True,
                "observations_inspiring_mechanism": (
                    "Rounds 38-40 showed that exact polar geometry and higher LR moments do not explain the StarCoder "
                    "surfaces. Source audit exposed finite Newton-Schulz, mixed optimizer channels, and 128-of-2048 "
                    "batch composition. Matrix uncertainty was derived independently as a non-optimizer portfolio arm."
                ),
                "novelty_class": row["materially_new_mechanism"],
                "evaluation_status": "preregistered before algebraic and StarCoder evaluation",
                "evidence_path": "approach_registry.csv",
                "notes": (
                    "No new exposed-adversarial outcome was inspected. The running frontier phase-fiber confirmation panel "
                    "remains sealed. No historical or adversarial evaluation is permitted until the StarCoder gate passes."
                ),
            }
        )
    identity = ["round_id", "candidate_id", "evaluation_status"]
    existing = set(map(tuple, ledger[identity].itertuples(index=False, name=None)))
    append = [row for row in ledger_rows if tuple(row[column] for column in identity) not in existing]
    if append:
        ledger = pd.concat([ledger, pd.DataFrame(append, columns=ledger.columns)], ignore_index=True)
        ledger.to_csv(LEDGER, index=False)
    print(registry.loc[registry["id"].isin(ids), ["id", "family", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()
