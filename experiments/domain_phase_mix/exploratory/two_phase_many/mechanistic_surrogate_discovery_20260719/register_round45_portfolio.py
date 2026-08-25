# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2"]
# ///
"""Preregister the round-45 mechanistic portfolio before evaluation."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
REGISTRY = OUTPUT_ROOT / "approach_registry.csv"
LEDGER = OUTPUT_ROOT / "data_use_ledger.csv"


def registry_rows() -> list[dict[str, object]]:
    return [
        {
            "id": "SDMMF",
            "family": "Source-discrete momentum Muon flow",
            "relationship_to_prior": (
                "Reopens FNSMF and scalar OMGF only through their missing joint transition. FNSMF had no optimizer "
                "memory and replaced exact projected steps by a tangent ODE; OMGF put scalar momentum outside an "
                "ordinary gradient flow. SDMMF carries the source matrix momentum through Nesterov and finite NS5."
            ),
            "materially_new_mechanism": "Matrix momentum changes the singular spectrum entering finite Newton-Schulz before each exact norm projection.",
            "mechanistic_premise": (
                "MuonH stores m_t=0.95m_{t-1}+g_t and applies NS5 to 0.95m_t+g_t before an exact constant-norm step. "
                "A phase boundary changes g_t but not m_t, so the optimizer traverses a source-defined transient absent "
                "from both prior models."
            ),
            "governing_equations": (
                "m_t=beta m_{t-1}+G(W_t,p_t); u_t=beta m_t+G(W_t,p_t); z_t=NS5(u_t); "
                "W~=W_t-eta_t z_t||W_t||_F/(||z_t||_F+eps); W_{t+1}=W~||W_t||_F/||W~||_F; "
                "Y=b+A[(1-q)L_B(W_T)+qrL_R(W_T)], A>=0. beta=0 is the exact memory ablation."
            ),
            "latent_state": "A constant-Frobenius-norm 2x2 representation W and an unconstrained 2x2 Muon momentum buffer m.",
            "state_transition": "The exact discrete source ordering: clipped task gradient, source-fixed Nesterov momentum, five quintic NS iterations, and norm-preserving projected update under the persisted LR schedule.",
            "response_link": "One nonnegative amplitude on terminal covariance-weighted task debt plus intercept.",
            "additional_degrees_of_freedom": "No momentum, NS, step-count, or peak-LR parameters are fitted; beta=0.95, Nesterov, five NS steps, epsilon, eta_peak=0.02, total steps, and schedules come from source. Task geometry and output ridge use a frozen finite grid.",
            "units_and_symmetries": "W, m after task-gradient normalization, targets, covariances, and weights are dimensionless; eta is a dimensionless relative norm step; A and b have BPB units. Fixed norm, basis, initialization, and source optimizer constants remove scale and rotation symmetries.",
            "single_phase_restriction": "A tied mixture runs the same discrete optimizer for all steps; the phase boundary changes no input. The same law must be independently selected and refit on tied policies.",
            "starcoder_signature": "Momentum must produce a material target-free boundary transient, beat beta=0 on both schedules and most folds, clear both shape references, and retain the cosine near-diagonal and WSD late-code optima.",
            "catastrophic_optimism_resolution": "Carried gradients oppose abrupt unsupported reallocations and can leave concentrated schedules with explicit terminal task debt rather than instantaneous adaptation.",
            "response_compression_resolution": "The momentum transient separates paths with equal exposure through a physical optimizer state rather than a free output rescaling.",
            "scale_transfer_expectation": "Optimizer constants transfer exactly under MuonH; total steps and LR schedules are declared inputs, while task geometry may be target-specific.",
            "cheapest_falsification": "Reject algebraically if beta=0.95 changes terminal state by under 0.001 median and 0.01 p95 across phase shifts; otherwise require global and >=3/5-fold wins over beta=0, both shape references, stable geometry, and raw-optimum distance <=0.15.",
            "status": "active_preregistered",
            "status_evidence": "Frozen before any round-45 evaluation.",
        },
        {
            "id": "BDLMTF",
            "family": "Balanced-depth linear multitask flow",
            "relationship_to_prior": (
                "Reopens DLSF only through an architecture-fixed balanced-depth invariant. DLSF used one scalar trunk "
                "factor and selected its shared rate at a boundary. BDLMTF represents the product of L balanced trunk "
                "factors, whose gradient-flow kinetics have a depth-dependent takeoff exponent."
            ),
            "materially_new_mechanism": "Architecture depth creates a balanced-product spectral-learning delay and rapid takeoff without a fitted activation gate.",
            "mechanistic_premise": (
                "In balanced deep-linear gradient flow, a mode's product grows slowly when its factors are small and "
                "accelerates once shared factors develop. Early broad data may grow a shared trunk that makes a late "
                "specialist readout learn rapidly."
            ),
            "governing_equations": (
                "o_j=a^L h_j; L_j=(o_j-1)^2/2; da/dt=-k[(1-p)partial_a L_B+p partial_a L_R]; "
                "dh_B/dt=-k_h(1-p)partial_hB L_B; dh_R/dt=-k_h p partial_hR L_R; "
                "Y=b+A[(1-q)L_B+qL_R], A>=0. L=1 and frozen a are exact ablations."
            ),
            "latent_state": "One nonnegative balanced trunk factor a and two task-specific scalar readouts h_B,h_R.",
            "state_transition": "Autonomous gradient flow of an L-layer balanced scalar trunk coupled to two task heads, continuous across phases.",
            "response_link": "One nonnegative amplitude on evaluation-weighted terminal broad/rare squared error plus intercept.",
            "additional_degrees_of_freedom": "Depth L is fixed to the declared transformer depth for the active model and compared with L=1; total trunk relaxation, head-to-trunk rate ratio, evaluation mix, and ridge use frozen grids.",
            "units_and_symmetries": "a, h_j, outputs, weights, and normalized time are dimensionless; A and b have BPB units. Equal positive initialization and balanced factors remove deep-linear rescaling and sign symmetries.",
            "single_phase_restriction": "Tied weights define one autonomous balanced flow; an artificial phase split has no effect. The restricted form can be fit independently on tied data.",
            "starcoder_signature": "Architecture depth must beat L=1 and frozen-trunk ablations on both schedules, delay specialist takeoff under cosine, and support late specialist acceleration under WSD.",
            "catastrophic_optimism_resolution": "Concentrated schedules that never establish the shared trunk cannot claim specialist capability from exposure alone and retain terminal error.",
            "response_compression_resolution": "Depth-induced takeoff expands terminal-error differences near the activation region without an output calibrator.",
            "scale_transfer_expectation": "Depth is architecture-declared; dimensionless trunk/head clock ratios should remain comparable at fixed optimizer family, while total relaxation may vary with tokens per parameter.",
            "cheapest_falsification": "Reject unless declared depth beats L=1 and frozen trunk globally and in >=3/5 folds on both schedules, clears both shape references, has stable interior clock ratios, and places raw optima within 0.15.",
            "status": "active_preregistered",
            "status_evidence": "Frozen before any round-45 evaluation.",
        },
        {
            "id": "ANTKF",
            "family": "Adaptive neural-tangent-kernel flow",
            "relationship_to_prior": (
                "Distinct from fixed-Hessian NQGF and competence cascades. Those evolve parameters or scalar competence "
                "under a fixed learning operator. ANTKF makes the task-space learning kernel itself a retained state, "
                "then uses it to transport residuals."
            ),
            "materially_new_mechanism": "A data-adapted task-space kernel persists across phases and controls subsequent residual-learning rates.",
            "mechanistic_premise": (
                "Feature learning changes the neural tangent kernel. Training on one domain can align the learning "
                "operator with shared or specialist residual directions, changing how efficiently later examples reduce "
                "error even when the aggregate token exposure is fixed."
            ),
            "governing_equations": (
                "K_star(p)=(1-p)K_B+pK_R; dK/dt=rho[K_star(p)-K]; D(p)=diag(1-p,cp); "
                "de/dt=-k K D(p)e; Y=b+A[(1-q)e_B^2+q e_R^2]/2, A>=0. rho=0 and rho=infinity are fixed-kernel and instantaneous-kernel ablations."
            ),
            "latent_state": "A symmetric positive-definite 2x2 task-space kernel K and a two-dimensional task-residual vector e.",
            "state_transition": "Kernel alignment relaxes toward a mixture-specific SPD target while that retained kernel transports residuals; both states remain continuous through the phase boundary.",
            "response_link": "One nonnegative amplitude on evaluation-weighted terminal squared residual plus intercept.",
            "additional_degrees_of_freedom": "Kernel angle and anisotropy, kernel-adaptation rate, residual rate, rare residual curvature, evaluation mix, and ridge on frozen finite grids; unit trace and fixed initial isotropic kernel identify scale.",
            "units_and_symmetries": "K, e, task targets, weights, and normalized time are dimensionless; k and rho are inverse normalized time; A and b have BPB units. Unit trace, fixed task basis, and positive initial residual remove scale, rotation, and sign symmetries.",
            "single_phase_restriction": "A tied mixture supplies one autonomous kernel target and residual flow for the full duration; the same law must also be refit independently on tied data.",
            "starcoder_signature": "Finite retained-kernel adaptation must beat both frozen and instantaneous kernels on both schedules, keep cosine close to tied, and rotate the WSD valley toward late rare enrichment.",
            "catastrophic_optimism_resolution": "A remote policy cannot instantly acquire the learning operator needed by its terminal data; unresolved residual remains explicit at the endpoint.",
            "response_compression_resolution": "Kernel alignment changes residual decay rates among frontier policies, widening physical response without a fitted calibration layer.",
            "scale_transfer_expectation": "Dimensionless kernel geometry should transfer across schedules and swarms; adaptation and residual clocks may scale with optimizer progress and tokens per parameter.",
            "cheapest_falsification": "Reject unless finite kernel adaptation beats frozen and instantaneous ablations globally and in >=3/5 folds on both schedules, selects interior stable rate ratios, clears both shape references, and places raw optima within 0.15.",
            "status": "active_preregistered",
            "status_evidence": "Frozen before any round-45 evaluation.",
        },
    ]


def main() -> None:
    registry = pd.read_csv(REGISTRY)
    rows = registry_rows()
    ids = {str(row["id"]) for row in rows}
    registry = registry.loc[~registry["id"].isin(ids)]
    registry = pd.concat([registry, pd.DataFrame(rows, columns=registry.columns)], ignore_index=True)
    registry.to_csv(REGISTRY, index=False)

    ledger = pd.read_csv(LEDGER)
    now = datetime.now(UTC).isoformat()
    entries = []
    for row in rows:
        entries.append(
            {
                "timestamp": now,
                "round_id": "round_45_portfolio_preregistration",
                "candidate_id": row["id"],
                "candidate_family": row["family"],
                "hyperparameters": "Frozen candidate-specific finite grids and mandatory nested ablations documented in the approach registry.",
                "adversarial_outcomes_available_before_proposal": True,
                "adversarial_outcomes_inspected_before_proposal": True,
                "observations_inspiring_mechanism": (
                    "Rounds 41-44 showed that active finite-map, optimizer-channel, and matrix-information mechanisms "
                    "remain insufficient. Source audit exposed the exact momentum-before-NS transition; balanced depth "
                    "and a retained learning operator were derived independently as architecture and feature-learning arms."
                ),
                "novelty_class": row["materially_new_mechanism"],
                "evaluation_status": "preregistered before algebraic and StarCoder evaluation",
                "evidence_path": "approach_registry.csv",
                "notes": "No new adversarial-development outcome was inspected. The running phase-fiber confirmation panel remains sealed.",
            }
        )
    identity = ["round_id", "candidate_id", "evaluation_status"]
    existing = set(map(tuple, ledger[identity].itertuples(index=False, name=None)))
    append = [row for row in entries if tuple(row[key] for key in identity) not in existing]
    if append:
        ledger = pd.concat([ledger, pd.DataFrame(append, columns=ledger.columns)], ignore_index=True)
        ledger.to_csv(LEDGER, index=False)
    print(registry.loc[registry["id"].isin(ids), ["id", "family", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()
