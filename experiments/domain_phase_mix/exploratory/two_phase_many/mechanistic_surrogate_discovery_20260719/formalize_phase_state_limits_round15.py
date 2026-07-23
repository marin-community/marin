# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2.0", "pandas>=2.2"]
# ///
"""Record algebraic limits that rule out two broad phase-model classes."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
OUTPUT_DIR = OUTPUT_ROOT / "round15_phase_state_no_go"
REGISTRY = OUTPUT_ROOT / "approach_registry.csv"
LEDGER = OUTPUT_ROOT / "data_use_ledger.csv"
SEED = 20260719


def exponential_kernel_coefficients(rate: float, alpha0: float) -> tuple[float, float]:
    """Integrate exp(-rate * age) over each phase."""
    alpha1 = 1.0 - alpha0
    if rate == 0.0:
        return alpha0, alpha1
    early = (np.exp(-rate * alpha1) - np.exp(-rate)) / rate
    late = (1.0 - np.exp(-rate * alpha1)) / rate
    return float(early), float(late)


def linear_kernel_audit() -> dict[str, float]:
    rng = np.random.default_rng(SEED)
    maximum_error = 0.0
    for alpha0 in (0.2, 0.5, 0.8):
        for rate in (0.0, 0.1, 1.0, 10.0):
            early, late = exponential_kernel_coefficients(rate, alpha0)
            for _ in range(100):
                phase0 = rng.dirichlet(np.ones(39))
                phase1 = rng.dirichlet(np.ones(39))
                direct = early * phase0 + late * phase1
                collapsed = early * (phase0 + (late / max(early, 1e-15)) * phase1)
                maximum_error = max(maximum_error, float(np.max(np.abs(direct - collapsed))))
    return {"maximum_linear_kernel_collapse_error": maximum_error}


def affine_endpoint_audit() -> dict[str, float]:
    rng = np.random.default_rng(SEED + 1)
    maximum_state_error = 0.0
    minimum_equivalent_policy = 1.0
    maximum_equivalent_policy = 0.0
    for alpha0 in (0.2, 0.5, 0.8):
        alpha1 = 1.0 - alpha0
        for rate in (0.1, 0.5, 1.0, 5.0, 20.0):
            full_decay = np.exp(-rate)
            early_decay = np.exp(-rate * alpha0)
            late_decay = np.exp(-rate * alpha1)
            for _ in range(1000):
                initial, phase0, phase1 = rng.uniform(size=3)
                after_early = early_decay * initial + (1.0 - early_decay) * phase0
                terminal = late_decay * after_early + (1.0 - late_decay) * phase1
                tied_policy = (terminal - full_decay * initial) / (1.0 - full_decay)
                tied_terminal = full_decay * initial + (1.0 - full_decay) * tied_policy
                maximum_state_error = max(maximum_state_error, abs(terminal - tied_terminal))
                minimum_equivalent_policy = min(minimum_equivalent_policy, tied_policy)
                maximum_equivalent_policy = max(maximum_equivalent_policy, tied_policy)
    return {
        "maximum_affine_endpoint_equivalence_error": maximum_state_error,
        "minimum_equivalent_tied_policy": minimum_equivalent_policy,
        "maximum_equivalent_tied_policy": maximum_equivalent_policy,
    }


def registry_rows() -> list[dict[str, str]]:
    return [
        {
            "id": "NG-LK",
            "family": "Scalar linear temporal kernels",
            "relationship_to_prior": "Formalizes why prior recency-kernel routes I and AJ and any scalar linear-kernel retuning cannot constitute a new mechanism.",
            "materially_new_mechanism": "None. This is an equivalence result, not a candidate.",
            "mechanistic_premise": "A scalar memory state is a linear convolution of each bucket's sampling history with a common temporal kernel.",
            "governing_equations": "x_i(T)=integral_0^T K(T-t)w_i(t)dt=a_0 w_i^(0)+a_1 w_i^(1)=a_0[w_i^(0)+eta w_i^(1)], eta=a_1/a_0.",
            "latent_state": "One scalar retained exposure per bucket.",
            "state_transition": "Linear time-invariant convolution with a common kernel.",
            "response_link": "Arbitrary downstream response of the scalar retained exposure.",
            "additional_degrees_of_freedom": "Kernel parameters change only the single effective phase multiplier eta for a two-phase piecewise-constant policy.",
            "units_and_symmetries": "K has inverse-time normalization; a common scale is absorbed by the downstream response. Kernel shapes with the same a1/a0 are observationally equivalent.",
            "single_phase_restriction": "For tied phases, the convolution reduces to a common scalar multiple of the tied policy.",
            "starcoder_signature": "Identical to an effective-exposure model with one phase multiplier; no kernel-shape signature remains with only two constant phases.",
            "catastrophic_optimism_resolution": "None beyond what effective-exposure already supplies.",
            "response_compression_resolution": "None beyond the downstream response link.",
            "scale_transfer_expectation": "Only eta could transfer; the kernel itself is not identified by a two-phase panel.",
            "cheapest_falsification": "Algebraic equality for arbitrary phase weights.",
            "status": "theoretical_equivalence_block",
            "status_evidence": "For every scalar common linear kernel, phasewise integration yields two coefficients and therefore exactly one effective-exposure multiplier. Numerical error is recorded in round15_phase_state_no_go.",
        },
        {
            "id": "NG-AES",
            "family": "Affine endpoint-only scalar dynamics",
            "relationship_to_prior": "Formalizes the limitation shared by reduced gradient-flow bowl AI, scalar HWER, and any one-coordinate affine relaxation with an endpoint-only response.",
            "materially_new_mechanism": "None. This is a reachable-set equivalence result, not a candidate.",
            "mechanistic_premise": "A scalar representation relaxes affinely toward the current task-mixture equilibrium and evaluation depends only on its terminal value.",
            "governing_equations": "dz/dt=k(p-z). For two phases, z_T=e^-k z_0+(1-e^-k)p_star, where p_star is a convex combination of p0 and p1 in [0,1].",
            "latent_state": "One scalar specialization coordinate z.",
            "state_transition": "Autonomous affine relaxation toward phase mixture p.",
            "response_link": "Any function ell(z_T) of terminal state only.",
            "additional_degrees_of_freedom": "Relaxation rate k and arbitrary terminal response do not enlarge the tied reachable set.",
            "units_and_symmetries": "z and p are dimensionless; normalized time absorbs k. Translation and scale of z can be fixed without changing the result.",
            "single_phase_restriction": "The equivalent tied policy p_star exactly reproduces every two-phase terminal state.",
            "starcoder_signature": "Cannot make the best two-phase terminal response strictly better than the best tied response on a two-domain scalar surface.",
            "catastrophic_optimism_resolution": "An endpoint bowl may bound extremes, but it cannot explain a strict phase-policy-class advantage.",
            "response_compression_resolution": "An arbitrary endpoint link can change calibration but not the reachable-set equivalence.",
            "scale_transfer_expectation": "The no-go holds at every scale unless another state, nonlinear transition, or path-dependent response is introduced.",
            "cheapest_falsification": "Construct the equivalent tied policy for arbitrary p0, p1, k, phase fractions, and z0.",
            "status": "theoretical_reachable_set_block",
            "status_evidence": "Every sampled two-phase affine endpoint was reproduced by a valid tied policy to numerical precision in round15_phase_state_no_go.",
        },
    ]


def update_registry_and_ledger() -> None:
    registry = pd.read_csv(REGISTRY)
    ids = {row["id"] for row in registry_rows()}
    registry = registry.loc[~registry["id"].isin(ids)]
    registry = pd.concat([registry, pd.DataFrame(registry_rows(), columns=registry.columns)], ignore_index=True)
    registry.to_csv(REGISTRY, index=False)

    ledger = pd.read_csv(LEDGER)
    timestamp = datetime.now(UTC).isoformat()
    rows = []
    for candidate_id, family, novelty in (
        ("NG-LK", "Scalar linear temporal kernels", "Algebraic observational-equivalence theorem"),
        ("NG-AES", "Affine endpoint-only scalar dynamics", "Reachable-set equivalence theorem"),
    ):
        rows.append(
            {
                "timestamp": timestamp,
                "round_id": "round_15_theoretical_limits",
                "candidate_id": candidate_id,
                "candidate_family": family,
                "hyperparameters": "None; exact algebraic audit",
                "adversarial_outcomes_available_before_proposal": True,
                "adversarial_outcomes_inspected_before_proposal": True,
                "observations_inspiring_mechanism": "Repeated kernel and scalar gradient-flow failures motivated checking whether those model classes are identifiable or capable of a strict phase advantage.",
                "novelty_class": novelty,
                "evaluation_status": "theoretical block; no candidate prediction evaluated",
                "evidence_path": "round15_phase_state_no_go/report.md",
                "notes": "No historical or adversarial target was predicted. The result prevents retuning an exhausted algebraic class under a new name.",
            }
        )
    identity = ["round_id", "candidate_id", "evaluation_status"]
    existing = set(map(tuple, ledger[identity].itertuples(index=False, name=None)))
    additions = [row for row in rows if tuple(row[column] for column in identity) not in existing]
    if additions:
        ledger = pd.concat([ledger, pd.DataFrame(additions, columns=ledger.columns)], ignore_index=True)
        ledger.to_csv(LEDGER, index=False)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    audit = {**linear_kernel_audit(), **affine_endpoint_audit()}
    (OUTPUT_DIR / "algebraic_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    report = [
        "# Round 15: phase-state identification limits",
        "",
        "## Scalar linear temporal kernels",
        "",
        r"For a two-phase piecewise-constant policy, every common scalar linear kernel satisfies",
        "",
        r"$$x_i(T)=\int_0^T K(T-t)w_i(t)\,dt=a_0w_i^{(0)}+a_1w_i^{(1)}=a_0\left(w_i^{(0)}+\eta w_i^{(1)}\right).$$",
        "",
        r"Thus the kernel is observationally equivalent to one effective-exposure multiplier $\eta=a_1/a_0$. Changing exponential to power-law or another scalar linear kernel does not introduce an identified mechanism with only two constant phases. Multiple latent modes observed separately, a state-dependent transition, or a nonlinear acquisition-rate law is required to escape this class.",
        "",
        f"Maximum numerical collapse error: `{audit['maximum_linear_kernel_collapse_error']:.3e}`.",
        "",
        "## Affine endpoint-only scalar dynamics",
        "",
        r"For $\dot z=k(p-z)$, sequential phase relaxation gives",
        "",
        r"$$z_T=e^{-k}z_0+(1-e^{-k})p_\star,$$",
        "",
        r"where $p_\star$ is a convex combination of $p^{(0)}$ and $p^{(1)}$. Therefore every two-phase terminal state is exactly reachable by a tied policy. Any response $\ell(z_T)$ has the same attainable optimum in the one- and two-phase policy classes. A strict two-phase advantage requires an additional path-dependent state, a nonlinear vector field, or a response that depends on the trajectory rather than only $z_T$.",
        "",
        f"Maximum endpoint equivalence error: `{audit['maximum_affine_endpoint_equivalence_error']:.3e}`.",
        f"Equivalent tied policies stayed in `[{audit['minimum_equivalent_tied_policy']:.6f}, {audit['maximum_equivalent_tied_policy']:.6f}]`.",
        "",
        "## Consequence",
        "",
        "These are theory blocks, not empirical candidates. They rule out further retuning of scalar common recency kernels and one-coordinate affine endpoint models. No historical or adversarial prediction was evaluated.",
    ]
    (OUTPUT_DIR / "report.md").write_text("\n".join(report) + "\n")
    update_registry_and_ledger()
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
