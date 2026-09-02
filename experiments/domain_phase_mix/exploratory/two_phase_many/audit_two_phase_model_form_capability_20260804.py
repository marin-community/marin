# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "scipy",
# ]
# ///
"""Audit whether a surrogate form can represent a genuine two-phase gain.

This is an algebraic implementation audit, not an empirical model comparison.
It separates three properties that fitting metrics can otherwise conflate:

1. Removing phase terms preserves every tied-policy prediction.
2. The phase-blind restriction is invariant along a fixed-aggregate fiber.
3. The phase-aware form admits at least one parameter setting whose global
   two-phase minimum is strictly better than its global tied minimum.

The third property is necessary for a model to solve a setting with a genuine
two-phase advantage. It is not sufficient for identification, generalization,
or plausible optimization; those remain empirical gates.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    retained_power_law_estimator_repair_20260731 as repaired,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    retained_power_law_model_20260728 as rpl,
)

DEFAULT_OUTPUT = SCRIPT_DIR / "reference_outputs" / "two_phase_model_form_capability_20260804" / "capability_audit.json"
GRID_SIZE = 201
PREDICTION_TOLERANCE = 1e-12
FIBER_TOLERANCE = 1e-12
STRICT_GAIN_FLOOR = 1e-4


class Predictor(Protocol):
    """Predict a scalar response from two-phase policies."""

    def predict(self, weights: np.ndarray) -> np.ndarray: ...


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def geometry() -> rpl.Geometry:
    """Return a two-bucket 80/20 geometry with aggregate-consistent epochs."""
    return rpl.Geometry(
        c0=np.asarray([0.8, 0.8]),
        c1=np.asarray([0.2, 0.2]),
        phase_0_fraction=0.8,
    )


def tied_policies(axis: np.ndarray) -> np.ndarray:
    phase = np.column_stack([1.0 - axis, axis])
    return np.stack([phase, phase], axis=1)


def full_policy_grid(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phase_0, phase_1 = np.meshgrid(axis, axis, indexing="ij")
    weights = np.stack(
        [
            np.column_stack([1.0 - phase_0.ravel(), phase_0.ravel()]),
            np.column_stack([1.0 - phase_1.ravel(), phase_1.ravel()]),
        ],
        axis=1,
    )
    return weights, phase_0.ravel(), phase_1.ravel()


def fixed_aggregate_fiber(aggregate: float, contrasts: np.ndarray) -> np.ndarray:
    """Parameterize policies with constant 0.8 w0 + 0.2 w1.

    ``contrasts`` is the late-minus-early weight of bucket 1. Feasibility is
    checked explicitly rather than repaired by clipping.
    """
    phase_0_bucket_1 = aggregate - 0.2 * contrasts
    phase_1_bucket_1 = aggregate + 0.8 * contrasts
    if np.any((phase_0_bucket_1 < 0.0) | (phase_0_bucket_1 > 1.0)):
        raise ValueError("phase-0 fiber coordinate is infeasible")
    if np.any((phase_1_bucket_1 < 0.0) | (phase_1_bucket_1 > 1.0)):
        raise ValueError("phase-1 fiber coordinate is infeasible")
    return np.stack(
        [
            np.column_stack([1.0 - phase_0_bucket_1, phase_0_bucket_1]),
            np.column_stack([1.0 - phase_1_bucket_1, phase_1_bucket_1]),
        ],
        axis=1,
    )


def parent_model(geometry_: rpl.Geometry) -> rpl.Fitted:
    """Construct a nontrivial admissible RPL parameterization for ablation."""
    shape = rpl.Shape(
        benefit_exponent=0.5,
        benefit_offset=0.1,
        damage_exponent=2.0,
        damage_threshold=0.0,
        retention=10.0,
        late_multiplier=4.0,
        ordering_channel=True,
    )
    probe = tied_policies(np.asarray([0.5]))
    columns = rpl.design_matrix(probe, geometry_, shape).shape[1]
    coefficients = np.linspace(0.05, 0.05 * columns, columns)
    return rpl.Fitted(
        shape=shape,
        ridge=1.0,
        intercept=0.73,
        coefficients=coefficients,
        geometry=geometry_,
    )


def empty_selection_summary() -> repaired.SelectionSummary:
    metrics = repaired.CandidateMetrics(
        shape_index=0,
        ridge_index=0,
        ridge=1.0,
        all_rmse=0.0,
        all_spearman=1.0,
        asymmetric_rmse=0.0,
        asymmetric_regret_at_1=0.0,
        asymmetric_lower_tail_rmse=0.0,
        pair_delta_rmse=0.0,
    )
    return repaired.SelectionSummary(
        candidate_count=1,
        selected_shape_index=0,
        selected_ridge_index=0,
        selected_metrics=metrics,
        minimum_all_rmse=0.0,
        maximum_eligible_all_rmse=0.0,
        minimum_eligible_regret_at_1=0.0,
        maximum_eligible_regret_at_1=0.0,
        rmse_eligible_count=1,
        regret_eligible_count=1,
    )


def repaired_model(geometry_: rpl.Geometry) -> repaired.Fitted:
    """Construct the same nontrivial shape under the maintained estimator."""
    shape = rpl.Shape(
        benefit_exponent=0.5,
        benefit_offset=0.1,
        damage_exponent=2.0,
        damage_threshold=0.0,
        retention=10.0,
        late_multiplier=4.0,
        ordering_channel=True,
    )
    design, layout = repaired.design_matrix(tied_policies(np.asarray([0.5])), geometry_, shape)
    if design.shape[1] != layout.total_count:
        raise AssertionError("repaired RPL layout does not match its design")
    return repaired.Fitted(
        shape=shape,
        ridge=1.0,
        intercept=0.73,
        aggregate_coefficients=np.linspace(0.05, 0.05 * layout.aggregate_count, layout.aggregate_count),
        phase_coefficients=np.linspace(-0.07, 0.07, layout.phase_count),
        phase_blind=False,
        geometry=geometry_,
        selection=empty_selection_summary(),
    )


def phase_gain_witness(geometry_: rpl.Geometry) -> tuple[rpl.Fitted, dict[str, float]]:
    """Return an admissible RPL witness with a strict two-phase optimum."""
    shape = rpl.Shape(
        benefit_exponent=0.5,
        benefit_offset=0.1,
        damage_exponent=2.0,
        damage_threshold=0.0,
        retention=10.0,
        late_multiplier=2.0,
        ordering_channel=False,
    )
    grid, phase_0, phase_1 = full_policy_grid(np.linspace(0.0, 1.0, GRID_SIZE))
    columns = rpl.design_matrix(grid[:1], geometry_, shape).shape[1]
    coefficients = np.zeros(columns)
    coefficients[:2] = 1.0
    model = rpl.Fitted(
        shape=shape,
        ridge=1.0,
        intercept=0.0,
        coefficients=coefficients,
        geometry=geometry_,
    )
    prediction = model.predict(grid)
    tied = np.isclose(phase_0, phase_1, atol=0.0, rtol=0.0)
    best = int(np.argmin(prediction))
    best_tied = int(np.flatnonzero(tied)[np.argmin(prediction[tied])])
    return model, {
        "global_minimum": float(prediction[best]),
        "tied_minimum": float(prediction[best_tied]),
        "strict_two_phase_gain": float(prediction[best_tied] - prediction[best]),
        "best_phase_0_bucket_1": float(phase_0[best]),
        "best_phase_1_bucket_1": float(phase_1[best]),
        "best_tied_bucket_1": float(phase_0[best_tied]),
    }


def ablation_audit(
    name: str,
    full: Predictor,
    phase_blind: Predictor,
) -> dict[str, float | str | bool]:
    tied = tied_policies(np.linspace(0.0, 1.0, GRID_SIZE))
    tied_difference = float(np.max(np.abs(full.predict(tied) - phase_blind.predict(tied))))

    fiber = fixed_aggregate_fiber(0.4, np.linspace(-0.4, 0.4, GRID_SIZE))
    fiber_prediction = phase_blind.predict(fiber)
    fiber_range = float(np.max(fiber_prediction) - np.min(fiber_prediction))

    passed = tied_difference <= PREDICTION_TOLERANCE and fiber_range <= FIBER_TOLERANCE
    if not passed:
        raise AssertionError(
            f"{name} phase ablation failed: tied difference={tied_difference:.3e}, " f"fiber range={fiber_range:.3e}"
        )
    return {
        "implementation": name,
        "tied_prediction_max_absolute_difference": tied_difference,
        "phase_blind_fixed_aggregate_prediction_range": fiber_range,
        "passed": passed,
    }


def main() -> None:
    args = parse_args()
    geometry_ = geometry()

    original = parent_model(geometry_)
    maintained = repaired_model(geometry_)
    ablations = [
        ablation_audit("original_rpl", original, rpl.without_phase_terms(original)),
        ablation_audit("repaired_rpl", maintained, repaired.without_phase_terms(maintained)),
    ]

    witness, gain = phase_gain_witness(geometry_)
    if gain["strict_two_phase_gain"] <= STRICT_GAIN_FLOOR:
        raise AssertionError(
            "phase-aware RPL failed its representability gate: " f"gain={gain['strict_two_phase_gain']:.6g}"
        )

    payload = {
        "audit": "two_phase_model_form_capability_v1",
        "interpretation": (
            "Passing is necessary for a phase-aware model but does not establish identification, "
            "generalization, or a plausible fitted optimum."
        ),
        "thresholds": {
            "tied_prediction_tolerance": PREDICTION_TOLERANCE,
            "fixed_aggregate_invariance_tolerance": FIBER_TOLERANCE,
            "strict_two_phase_gain_floor": STRICT_GAIN_FLOOR,
        },
        "phase_ablation": ablations,
        "phase_gain_witness": {
            **gain,
            "shape": asdict(witness.shape),
            "coefficients": witness.coefficients.tolist(),
            "passed": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
