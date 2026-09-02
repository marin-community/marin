# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pandas>=2.2",
# ]
# ///
"""Preregister logistic margin-competition flow."""

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
    if not registry["id"].eq("LMCF").any():
        row = {
            "id": "LMCF",
            "family": "Logistic margin-competition flow",
            "relationship_to_prior": (
                "Materially differs from quadratic NQGF/HWER and quartic NTPGF: the state is trained and evaluated by "
                "the same finite-margin cross-entropy potential, whose gradient decays exponentially after mastery."
            ),
            "materially_new_mechanism": (
                "Finite-margin saturation under the actual log-loss geometry, with task transfer or conflict encoded by "
                "one identifiable angle between broad and rare gradient directions."
            ),
            "mechanistic_premise": (
                "Smooth BPB is a cross-entropy. Data families build margins along partially aligned representation "
                "directions; mastered directions yield vanishing gradients, while conflicting directions can overwrite margins."
            ),
            "governing_equations": (
                r"ell_c(z)=log(1+exp(-v_c^T z)); dz/d tau=k[(1-p)sigma(-v_b^T z)v_b+"
                r"r p sigma(-v_r^T z)v_r-lambda z]. The fixed unit vectors satisfy v_b^T v_r=cos(theta). "
                r"Y=b+A_b ell_b(z_T)+A_r ell_r(z_T), A_b,A_r>=0."
            ),
            "latent_state": "A two-dimensional shared representation margin vector.",
            "state_transition": (
                "Autonomous nonlinear gradient flow of the mixture-weighted logistic training loss plus isotropic "
                "representation decay, integrated in token or source-derived optimizer time."
            ),
            "response_link": "The same two task log losses used by training, with nonnegative BPB amplitudes and an intercept.",
            "additional_degrees_of_freedom": (
                "A task angle, representation-decay rate, acquisition rate, and rare-to-broad gradient scale; orthogonal "
                "theta=90 degrees is the exact no-transfer/no-conflict ablation."
            ),
            "units_and_symmetries": (
                "Margins and state are dimensionless; rates are inverse optimizer time; amplitudes have BPB units. Unit "
                "task vectors and a fixed broad direction remove scale/rotation symmetry; theta in [0,pi] removes sign ambiguity."
            ),
            "single_phase_restriction": (
                "With tied phase policies, autonomous flow composes exactly across the artificial boundary. The same "
                "logistic form must separately be fit to tied outcomes."
            ),
            "starcoder_signature": (
                "A stable non-orthogonal task angle must improve both schedules over theta=90 degrees. Cosine should retain "
                "a near-diagonal optimum; WSD may favor late rare margins because phase 1 receives more optimizer time."
            ),
            "catastrophic_optimism_resolution": (
                "Concentrated policies cannot earn unbounded additive benefit after their mastered margin saturates, while "
                "the omitted task retains explicit positive log-loss."
            ),
            "response_compression_resolution": (
                "Exponential margin gradients and task conflict can create wider frontier separation than a linear exposure "
                "state without an output calibrator."
            ),
            "scale_transfer_expectation": (
                "Gradient-alignment sign and task angle should transfer more strongly than the acquisition clock. Log-loss "
                "geometry is common across architectures and scales."
            ),
            "cheapest_falsification": (
                "A non-orthogonal model must beat theta=90 globally and in >=3/5 folds on both StarCoder schedules, remain "
                "within 5% of both shape frontiers, choose an interior stable angle/decay, and locate both raw optima within 0.15."
            ),
            "status": "active_preregistered",
            "status_evidence": "Frozen before any LMCF fit or StarCoder evaluation.",
        }
        registry = pd.concat([registry, pd.DataFrame([row], columns=registry.columns)], ignore_index=True)
        registry.to_csv(REGISTRY, index=False)

    ledger = pd.read_csv(LEDGER)
    row = {
        "timestamp": datetime.now(UTC).isoformat(),
        "round_id": "round_49_preregistration",
        "candidate_id": "LMCF",
        "candidate_family": "Logistic margin-competition flow",
        "hyperparameters": "Frozen clock/rate/decay/task-angle/rare-scale/ridge grid with theta=90 ablation",
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": True,
        "observations_inspiring_mechanism": (
            "Quadratic, quartic, clipping, and matrix-gradient models miss the two StarCoder optima. Actual pretraining and "
            "evaluation use cross-entropy, so finite-margin saturation is a distinct omitted response mechanism."
        ),
        "novelty_class": "Shared logistic-margin state with explicit gradient alignment",
        "evaluation_status": "preregistered for StarCoder gate; no new adversarial evaluation",
        "evidence_path": "round49_logistic_margin_starcoder",
        "notes": "Running sealed phase-fiber panel remains untouched.",
    }
    identity = ["round_id", "candidate_id", "evaluation_status"]
    existing = set(map(tuple, ledger[identity].itertuples(index=False, name=None)))
    if tuple(row[column] for column in identity) not in existing:
        ledger = pd.concat([ledger, pd.DataFrame([row], columns=ledger.columns)], ignore_index=True)
        ledger.to_csv(LEDGER, index=False)


if __name__ == "__main__":
    main()
