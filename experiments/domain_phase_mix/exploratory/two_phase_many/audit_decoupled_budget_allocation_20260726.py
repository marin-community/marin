# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Split the response into aggregate and contrast parts, and split the budget to match.

Every candidate tried in this experiment improves fit and censored extrapolation while
degrading phase decision skill. The mixture-geometry exploration found the structural
reason: an exposure-matched pair shares an identical aggregate *and* identical per-bucket
epochs, so any aggregate-only column contributes exactly zero to the predicted
two-phase-versus-tied delta. Aggregate features cannot help the phase call, and they hurt
it only indirectly, by changing the shape selection and the coefficients on the few
phase-sensitive columns that a joint fit shares between both jobs.

That suggests the coupling is an artifact of fitting one head to two jobs. Write

    L(a, d) = L(a, 0) + Delta(a, d)

and the two terms are separately observable. The tied twin of every panel row gives
``L(a, 0)`` directly, and the difference within a pair gives ``Delta`` directly. Fitting
them as two heads on two row sets means an improvement to the aggregate head cannot move
the phase head at all, by construction rather than by tuning.

The cost is runs. A pair needs two trainings to yield one ``Delta``, while a tied row
needs one to yield one ``L(a, 0)``. Under a fixed 280-run budget,
``n_tied + 2 * n_pairs = 280``, so buying phase information means buying fewer aggregates.
This script prices that trade: it sweeps the allocation and asks where total quality peaks,
against the incumbent that spends all 280 runs on two-phase rows and fits one joint head.

Both arms are held to the same 280-run budget, so this is a fair comparison of how to
spend it rather than a comparison of budgets. Allocations are drawn several times to
separate the allocation effect from which particular rows were drawn.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dual_objective_harness_20260726 import build_benchmark  # noqa: E402
from phase_order_spine_20260725 import load_paired_panel  # noqa: E402
from proposal_metrics_20260726 import phase_decision_skill  # noqa: E402
from swarm39_harness_20260725 import TABLE9, UNCHEATABLE, Panel, fit_head  # noqa: E402
from swarm39_models_20260725 import _state_shapes, build_hierarchical_phase_replay  # noqa: E402

OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "decoupled_budget_allocation_20260726"

TARGETS = (UNCHEATABLE, TABLE9)
RUN_BUDGET = 280
# n_pairs per allocation; n_tied is whatever the budget leaves.
PAIR_COUNTS = (30, 50, 70, 90, 110, 140)
ALLOCATION_DRAWS = 6
CENSOR_FRACTION = 0.10
SEED = 20260726
L2 = 0.0
TIED_PREFIX = r"^singleavg_"


@dataclass(frozen=True)
class PairedData:
    """Exposure-matched pairs with both members observed."""

    phase0: np.ndarray
    phase1: np.ndarray
    aggregate: np.ndarray
    tied_bpb: dict[str, np.ndarray]
    delta: dict[str, np.ndarray]
    reference: Panel


def load_pairs(reference: Panel) -> PairedData:
    paired = load_paired_panel("300m", TIED_PREFIX)
    assert paired.buckets == reference.buckets, "bucket order mismatch"
    return PairedData(
        phase0=paired.phase0,
        phase1=paired.phase1,
        aggregate=paired.aggregate,
        tied_bpb=dict(paired.tied_bpb),
        delta=dict(paired.delta),
        reference=reference,
    )


def panel_of(data: PairedData, phase0: np.ndarray, phase1: np.ndarray, target: str, values: np.ndarray) -> Panel:
    rows = len(phase0)
    reference = data.reference
    return Panel(
        scale="300m",
        split="synthetic",
        alpha=reference.alpha,
        buckets=reference.buckets,
        c0=reference.c0,
        c1=reference.c1,
        family_index=reference.family_index,
        family_names=reference.family_names,
        phase0=phase0,
        phase1=phase1,
        targets={target: values},
        series=np.array(["synthetic"] * rows),
        policy_class=np.array(["two_phase"] * rows),
        group=np.arange(rows),
        row_id=np.array([f"row_{i}" for i in range(rows)]),
    )


def fit_aggregate_head(panel: Panel, target: str, shape: dict) -> tuple[float, np.ndarray]:
    """Fit L(a, 0) on tied rows, where phase0 equals phase1 so contrast columns vanish."""
    observed = panel.targets[target]
    design = build_hierarchical_phase_replay(panel, shape).matrix
    return fit_head(design, observed, L2)


def fit_contrast_head(data: PairedData, index: np.ndarray, target: str, shape: dict) -> tuple[float, np.ndarray]:
    """Fit Delta on observed pair differences, using the difference of designs.

    The design difference isolates exactly the columns that distinguish a two-phase policy
    from its tied twin. Aggregate-only columns cancel here, which is the property that makes
    the two heads independent.
    """
    two = panel_of(data, data.phase0[index], data.phase1[index], target, data.delta[target][index])
    tied = panel_of(data, data.aggregate[index], data.aggregate[index], target, data.delta[target][index])
    difference = build_hierarchical_phase_replay(two, shape).matrix - build_hierarchical_phase_replay(tied, shape).matrix
    # Delta is signed, so the sign-constrained head would be wrong here; use least squares
    # with a free intercept on the differenced design.
    centred = difference - difference.mean(axis=0, keepdims=True)
    target_values = data.delta[target][index]
    coefficients, *_ = np.linalg.lstsq(centred, target_values - target_values.mean(), rcond=None)
    intercept = float(target_values.mean() - difference.mean(axis=0) @ coefficients)
    return intercept, coefficients


def select_shape_for(panel: Panel, target: str) -> dict:
    """Pick the shape by in-sample RMSE of the aggregate head on the tied rows.

    In-sample rather than out-of-fold because the tied allocation can be as small as
    140 rows and a five-fold split at that size makes the selection noisier than the
    difference it is resolving. The comparison against the incumbent uses the same
    criterion on both arms, so the choice does not favour either.
    """
    observed = panel.targets[target]
    best: tuple[float, dict] | None = None
    for shape in _state_shapes(True):
        intercept, coefficients = fit_aggregate_head(panel, target, shape)
        prediction = intercept + build_hierarchical_phase_replay(panel, shape).matrix @ coefficients
        score = float(np.sqrt(np.mean((prediction - observed) ** 2)))
        if best is None or score < best[0]:
            best = (score, shape)
    assert best is not None
    return best[1]


def evaluate_decoupled(
    data: PairedData, target: str, tied_index: np.ndarray, pair_index: np.ndarray, censored: np.ndarray
) -> dict[str, float]:
    """Fit both heads on disjoint row sets and score all three arms."""
    tied_panel = panel_of(
        data, data.aggregate[tied_index], data.aggregate[tied_index], target, data.tied_bpb[target][tied_index]
    )
    shape = select_shape_for(tied_panel, target)
    aggregate_intercept, aggregate_coefficients = fit_aggregate_head(tied_panel, target, shape)
    contrast_intercept, contrast_coefficients = fit_contrast_head(data, pair_index, target, shape)

    # Censored arm: score the withheld best tied policies with the aggregate head only.
    censored_panel = panel_of(
        data, data.aggregate[censored], data.aggregate[censored], target, data.tied_bpb[target][censored]
    )
    censored_prediction = (
        aggregate_intercept + build_hierarchical_phase_replay(censored_panel, shape).matrix @ aggregate_coefficients
    )
    censored_truth = data.tied_bpb[target][censored]
    residual = censored_prediction - censored_truth
    ranks = lambda v: np.argsort(np.argsort(v))  # noqa: E731

    # Phase arm: predicted delta on every pair NOT used to fit the contrast head.
    holdout = np.setdiff1d(np.arange(len(data.phase0)), np.union1d(pair_index, np.flatnonzero(censored)))
    tied_hold = panel_of(data, data.aggregate[holdout], data.aggregate[holdout], target, data.delta[target][holdout])
    two_hold = panel_of(data, data.phase0[holdout], data.phase1[holdout], target, data.delta[target][holdout])
    difference_hold = (
        build_hierarchical_phase_replay(two_hold, shape).matrix
        - build_hierarchical_phase_replay(tied_hold, shape).matrix
    )
    predicted_delta = contrast_intercept + difference_hold @ contrast_coefficients
    skill = phase_decision_skill(predicted_delta, data.delta[target][holdout])

    return {
        "n_tied": len(tied_index),
        "n_pairs": len(pair_index),
        "runs": int(len(tied_index) + 2 * len(pair_index)),
        "cens_rmse": float(np.sqrt(np.mean(residual**2))),
        "cens_bias": float(np.mean(residual)),
        "cens_spearman": float(np.corrcoef(ranks(censored_prediction), ranks(censored_truth))[0, 1]),
        "phase_skill": float(skill["phase_skill_score"]),
        "phase_accuracy": float(skill["decision_accuracy"]),
        "n_phase_holdout": len(holdout),
    }


def evaluate_joint_control(
    data: PairedData, target: str, pool: np.ndarray, censored: np.ndarray, rng: np.random.Generator
) -> dict[str, float]:
    """One joint head on the same runs and the same observations as the decoupled arm.

    This is the control that isolates the decoupling rather than the budget. 140 pairs cost
    280 runs and yield 280 observations, because both members of a pair are trained and
    measured: 140 two-phase levels and 140 tied levels. The joint head is fitted on all 280
    of them, so it sees exactly the information the decoupled arm sees and differs only in
    using one head for both jobs instead of two heads on disjoint rows. Fitting it on the 140
    two-phase members alone would discard half the observations and flatter the decoupled arm.
    """
    pair_index = rng.choice(pool, size=min(140, len(pool)), replace=False)
    joint = panel_of(
        data,
        np.vstack([data.phase0[pair_index], data.aggregate[pair_index]]),
        np.vstack([data.phase1[pair_index], data.aggregate[pair_index]]),
        target,
        np.concatenate(
            [data.tied_bpb[target][pair_index] + data.delta[target][pair_index], data.tied_bpb[target][pair_index]]
        ),
    )
    shape = select_shape_for(joint, target)
    design = build_hierarchical_phase_replay(joint, shape).matrix
    intercept, coefficients = fit_head(design, joint.targets[target], L2)

    def predict(phase0: np.ndarray, phase1: np.ndarray, index: np.ndarray) -> np.ndarray:
        panel = panel_of(data, phase0, phase1, target, data.tied_bpb[target][index])
        return intercept + build_hierarchical_phase_replay(panel, shape).matrix @ coefficients

    censored_index = np.flatnonzero(censored)
    censored_prediction = predict(data.aggregate[censored_index], data.aggregate[censored_index], censored_index)
    censored_truth = data.tied_bpb[target][censored_index]
    residual = censored_prediction - censored_truth
    ranks = lambda v: np.argsort(np.argsort(v))  # noqa: E731

    holdout = np.setdiff1d(np.arange(len(data.phase0)), np.union1d(pair_index, censored_index))
    predicted_delta = predict(data.phase0[holdout], data.phase1[holdout], holdout) - predict(
        data.aggregate[holdout], data.aggregate[holdout], holdout
    )
    skill = phase_decision_skill(predicted_delta, data.delta[target][holdout])
    return {
        "n_tied": 0,
        "n_pairs": len(pair_index),
        "runs": int(2 * len(pair_index)),
        "cens_rmse": float(np.sqrt(np.mean(residual**2))),
        "cens_bias": float(np.mean(residual)),
        "cens_spearman": float(np.corrcoef(ranks(censored_prediction), ranks(censored_truth))[0, 1]),
        "phase_skill": float(skill["phase_skill_score"]),
        "phase_accuracy": float(skill["decision_accuracy"]),
        "n_phase_holdout": len(holdout),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    benchmark = build_benchmark()
    data = load_pairs(benchmark.fit_300m)
    n = len(data.phase0)
    rows: list[dict[str, Any]] = []

    for target in TARGETS:
        tied_values = data.tied_bpb[target]
        n_censored = max(1, int(CENSOR_FRACTION * n))
        censored = np.zeros(n, dtype=bool)
        censored[np.argsort(tied_values)[:n_censored]] = True
        pool = np.flatnonzero(~censored)

        for draw in range(ALLOCATION_DRAWS):
            rows.append(
                {
                    "target": target,
                    "arm": "joint_control",
                    "draw": draw,
                    **evaluate_joint_control(data, target, pool, censored, np.random.default_rng(SEED + draw)),
                }
            )

        for n_pairs in PAIR_COUNTS:
            n_tied = RUN_BUDGET - 2 * n_pairs
            if n_tied < 0 or n_pairs > len(pool):
                continue
            for draw in range(ALLOCATION_DRAWS):
                rng = np.random.default_rng(SEED + draw)
                pair_index = rng.choice(pool, size=n_pairs, replace=False)
                remaining = np.setdiff1d(pool, pair_index)
                # Tied rows may reuse pair aggregates only up to the pool size; beyond that
                # the allocation is capped by how many distinct aggregates exist.
                take = min(n_tied, len(remaining))
                tied_index = rng.choice(remaining, size=take, replace=False)
                rows.append(
                    {
                        "target": target,
                        "arm": "decoupled",
                        "draw": draw,
                        **evaluate_decoupled(data, target, tied_index, pair_index, censored),
                    }
                )
            print(f"  {target}: n_pairs={n_pairs} done")

    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT_DIR / "allocation_sweep.csv", index=False)
    print("\n=== decoupled fit: how to split a 280-run budget between tied rows and pairs ===")
    print(
        frame.groupby(["target", "arm", "n_pairs"])[
            ["n_tied", "runs", "cens_rmse", "cens_bias", "cens_spearman", "phase_skill", "n_phase_holdout"]
        ]
        .mean()
        .to_string(float_format=lambda v: f"{v:.5f}")
    )

    (OUTPUT_DIR / "provenance.json").write_text(
        json.dumps(
            {
                "run_budget": RUN_BUDGET,
                "pair_counts": list(PAIR_COUNTS),
                "allocation_draws": ALLOCATION_DRAWS,
                "censor_fraction": CENSOR_FRACTION,
                "note": (
                    "a pair costs two runs and yields one observed Delta; a tied row costs one run and "
                    "yields one observed L(a,0). The two heads are fitted on disjoint rows, so the aggregate "
                    "head cannot influence the phase head."
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
