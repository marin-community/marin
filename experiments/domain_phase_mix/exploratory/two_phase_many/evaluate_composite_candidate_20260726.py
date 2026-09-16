# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compose the three individually-validated improvements into one candidate and test it.

Each ingredient below beat the incumbent on one objective and was measured in isolation. None
of them has been combined, and they target different failure modes, so the composite is the
obvious untried candidate.

*Geometry-augmented design.* Adding a two-sided deviation from a target breadth and peak epoch
load gives the best fit anything in this experiment reached: OOF RMSE 0.00717 against the
incumbent's 0.00788 on Uncheatable, with censored Spearman up 0.100 and censored bias flipping
sign to conservative. Its weakness was phase skill, down 0.024.

*Band ensembling.* Averaging over the configurations out-of-fold RMSE cannot separate improves
censored-extrapolation RMSE with a paired interval excluding zero, and is phase-neutral rather
than phase-harmful. In the Observatory it cut low-tail RMSE by 22 percent on one 3e18 cell.

*Pairs-constructed training budget.* Spending the fixed 280-run budget on exposure-matched
pairs rather than entirely on two-phase policies lifts phase decision skill by +0.122 on
Uncheatable and +0.083 on Table-9, stable across every budget from 60 to 154 runs. This is a
swarm-construction change, which the brief permits provided the procedure is pre-specified
rather than a split chosen for its score: pairing every sampled policy with its
exposure-matched tied twin is specified before any outcome is read.

The composite is therefore: geometry-augmented hierarchical replay, band-ensembled, fitted on
140 exposure-matched pairs. The incumbent it must beat is single-configuration hierarchical
replay fitted on 280 two-phase rows. Both spend 280 training runs.

All three arms are scored: grouped out-of-fold fit, censored extrapolation on held-back best
rows, and phase decision skill on a held-out pair set. Paired bootstrap over shared draws.
A candidate has to win fit AND optimum quality to count, with optimum quality weighted higher,
so a fit gain paid for with phase skill is a failure and is reported as one.

The shape grid is screened in two stages rather than swept whole, because the geometry design
carries 1944 shapes and banding over all of them times four ridges is not affordable. The
screen is the same one the Observatory uses for its hierarchical model, and it is applied
identically to both arms so it cannot favour either.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import nnls

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from audit_mixture_geometry_features_20260726 import build_hpr_geometry_target, geometry_target_shapes  # noqa: E402
from audit_pairs_versus_all_two_phase_20260726 import load_pairs, panel_of  # noqa: E402
from dual_objective_harness_20260726 import RUN_SIGMA, build_benchmark  # noqa: E402
from proposal_metrics_20260726 import phase_decision_skill  # noqa: E402
from swarm39_harness_20260725 import TABLE9, UNCHEATABLE, Panel, fit_head, grouped_splits  # noqa: E402
from swarm39_models_20260725 import _state_shapes, build_hierarchical_phase_replay  # noqa: E402

OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "composite_candidate_20260726"

TARGETS = (UNCHEATABLE, TABLE9)
TARGET_LABEL = {UNCHEATABLE: "uncheatable", TABLE9: "table9"}
RUN_BUDGET = 280
CENSOR_FRACTION = 0.10
PHASE_HOLDOUT = 60
SCREEN_TOP_SHAPES = 24
L2_GRID = (0.0, 0.01, 0.1, 1.0)
N_SPLITS = 5
SPLIT_SEED = 0
DRAWS = 8
BOOTSTRAP_DRAWS = 80
SEED = 20260726
BAND_SIGMA = 1.0
MAX_MEMBERS = 20


@dataclass(frozen=True)
class Arm:
    label: str
    build: Any
    shapes: Any
    design: str  # "pairs" or "all_two_phase"
    ensemble: bool


ARMS = (
    Arm("incumbent", build_hierarchical_phase_replay, lambda: _state_shapes(True), "all_two_phase", False),
    Arm("composite", build_hpr_geometry_target, geometry_target_shapes, "pairs", True),
    # Ablations, so a win can be attributed rather than merely observed.
    Arm("geometry_only", build_hpr_geometry_target, geometry_target_shapes, "all_two_phase", False),
    Arm("pairs_only", build_hierarchical_phase_replay, lambda: _state_shapes(True), "pairs", False),
)


def training_panel(pairs, target: str, design: str, pool: np.ndarray, budget: int) -> Panel:
    """The panel a design buys with ``budget`` runs; a pair costs two runs, a row costs one."""
    if design == "all_two_phase":
        chosen = pool[:budget]
        return panel_of(pairs, pairs.phase0[chosen], pairs.phase1[chosen], target, pairs.two_phase[target][chosen])
    chosen = pool[: budget // 2]
    return panel_of(
        pairs,
        np.vstack([pairs.phase0[chosen], pairs.aggregate[chosen]]),
        np.vstack([pairs.phase1[chosen], pairs.aggregate[chosen]]),
        target,
        np.concatenate([pairs.two_phase[target][chosen], pairs.tied[target][chosen]]),
    )


def oof(panel: Panel, arm: Arm, target: str, shape: dict, l2: float) -> np.ndarray:
    observed = panel.targets[target]
    design = arm.build(panel, shape).matrix
    prediction = np.full(len(observed), np.nan)
    for train, test in grouped_splits(panel, N_SPLITS, SPLIT_SEED):
        if train.sum() < 2:
            continue
        intercept, coefficients = fit_head(design[train], observed[train], l2)
        prediction[test] = intercept + design[test] @ coefficients
    return prediction


def screen(panel: Panel, arm: Arm, target: str) -> list[tuple[float, dict, float]]:
    """Two-stage screen: rank shapes at zero ridge, then score the top shapes over the ridge grid."""
    observed = panel.targets[target]
    ranked = []
    for shape in arm.shapes():
        prediction = oof(panel, arm, target, shape, 0.0)
        finite = np.isfinite(prediction)
        ranked.append((float(np.sqrt(np.mean((prediction[finite] - observed[finite]) ** 2))), shape))
    ranked.sort(key=lambda item: item[0])
    scored = []
    for _, shape in ranked[:SCREEN_TOP_SHAPES]:
        for l2 in L2_GRID:
            prediction = oof(panel, arm, target, shape, l2)
            finite = np.isfinite(prediction)
            scored.append((float(np.sqrt(np.mean((prediction[finite] - observed[finite]) ** 2))), shape, l2))
    scored.sort(key=lambda item: item[0])
    return scored


def members_of(panel: Panel, arm: Arm, target: str, scored: list[tuple[float, dict, float]]) -> tuple[list, np.ndarray]:
    """Band members and stacked simplex weights, or the argmin alone when not ensembling."""
    if not arm.ensemble:
        return [scored[0]], np.ones(1)
    best = scored[0][0]
    inside = [item for item in scored if item[0] <= best + BAND_SIGMA * RUN_SIGMA[target]][:MAX_MEMBERS]
    if len(inside) == 1:
        return inside, np.ones(1)
    observed = panel.targets[target]
    stacked = np.column_stack([oof(panel, arm, target, shape, l2) for _, shape, l2 in inside])
    finite = np.isfinite(observed) & np.isfinite(stacked).all(axis=1)
    differences = stacked[finite, 1:] - stacked[finite, :1]
    coefficients, _ = nnls(differences, observed[finite] - stacked[finite, 0], maxiter=400)
    if coefficients.sum() > 1.0:
        coefficients = coefficients / coefficients.sum()
    weights = np.concatenate([[1.0 - coefficients.sum()], coefficients])
    return inside, np.maximum(weights, 0.0) / max(np.maximum(weights, 0.0).sum(), 1e-12)


def predictor(panel: Panel, arm: Arm, target: str, members: list, weights: np.ndarray, rows: np.ndarray | None):
    """Fit every member on ``rows`` and return a weighted-average prediction function."""
    observed = panel.targets[target]
    use = np.isfinite(observed) if rows is None else (rows & np.isfinite(observed))
    heads = []
    for (_, shape, l2), weight in zip(members, weights, strict=True):
        if weight <= 0.0:
            continue
        design = arm.build(panel, shape).matrix
        intercept, coefficients = fit_head(design[use], observed[use], l2)
        heads.append((weight, shape, intercept, coefficients))

    def predict(target_panel: Panel) -> np.ndarray:
        total = None
        for weight, shape, intercept, coefficients in heads:
            contribution = weight * (intercept + arm.build(target_panel, shape).matrix @ coefficients)
            total = contribution if total is None else total + contribution
        assert total is not None
        return total

    return predict


def ranks(values: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(values))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    benchmark = build_benchmark()
    pairs = load_pairs(benchmark.fit_300m)
    total = len(pairs)
    rows: list[dict[str, Any]] = []

    for target in TARGETS:
        label = TARGET_LABEL[target]
        n_censored = max(1, int(CENSOR_FRACTION * total))
        censored = np.argsort(pairs.tied[target])[:n_censored]
        remaining = np.setdiff1d(np.arange(total), censored)

        for draw in range(DRAWS):
            rng = np.random.default_rng(SEED + draw)
            shuffled = rng.permutation(remaining)
            holdout = shuffled[:PHASE_HOLDOUT]
            pool = shuffled[PHASE_HOLDOUT:]
            budget = min(RUN_BUDGET, (len(pool) // 2) * 2)
            truth_delta = pairs.delta[target][holdout]

            for arm in ARMS:
                panel = training_panel(pairs, target, arm.design, pool, budget)
                scored = screen(panel, arm, target)
                members, weights = members_of(panel, arm, target, scored)
                predict = predictor(panel, arm, target, members, weights, None)

                observed = panel.targets[target]
                in_panel = np.full(len(observed), np.nan)
                for _, shape, l2 in members[:1]:
                    in_panel = oof(panel, arm, target, shape, l2)
                finite = np.isfinite(in_panel)
                fit_rmse = float(np.sqrt(np.mean((in_panel[finite] - observed[finite]) ** 2)))
                fit_spearman = float(np.corrcoef(ranks(in_panel[finite]), ranks(observed[finite]))[0, 1])

                censored_panel = panel_of(
                    pairs, pairs.aggregate[censored], pairs.aggregate[censored], target, pairs.tied[target][censored]
                )
                prediction = predict(censored_panel)
                cens_truth = pairs.tied[target][censored]
                residual = prediction - cens_truth

                two = panel_of(pairs, pairs.phase0[holdout], pairs.phase1[holdout], target, truth_delta)
                tied = panel_of(pairs, pairs.aggregate[holdout], pairs.aggregate[holdout], target, truth_delta)
                skill = phase_decision_skill(predict(two) - predict(tied), truth_delta)

                rows.append(
                    {
                        "target": label,
                        "arm": arm.label,
                        "draw": draw,
                        "runs": budget,
                        "band_members": len(members),
                        "active_members": int((weights > 1e-6).sum()),
                        "oof_rmse": fit_rmse,
                        "oof_spearman": fit_spearman,
                        "cens_rmse": float(np.sqrt(np.mean(residual**2))),
                        "cens_bias": float(np.mean(residual)),
                        "cens_spearman": float(np.corrcoef(ranks(prediction), ranks(cens_truth))[0, 1]),
                        "phase_skill": float(skill["phase_skill_score"]),
                        "phase_accuracy": float(skill["decision_accuracy"]),
                    }
                )
            print(f"  {label} draw {draw + 1}/{DRAWS} done", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT_DIR / "composite_arms.csv", index=False)

    metrics = ["oof_rmse", "oof_spearman", "cens_rmse", "cens_bias", "cens_spearman", "phase_skill", "phase_accuracy"]
    print("\n=== mean over draws, identical 280-run budget, identical evaluation rows ===")
    print(
        frame.groupby(["target", "arm"])[["runs", "band_members", *metrics]]
        .mean()
        .to_string(float_format=lambda v: f"{v:.5f}")
    )

    print("\n=== paired over draws against the incumbent ===")
    summary = []
    for target, block in frame.groupby("target"):
        base = block[block.arm == "incumbent"].set_index("draw")
        for arm in ("composite", "geometry_only", "pairs_only"):
            other = block[block.arm == arm].set_index("draw")
            for metric in metrics:
                delta = (other[metric] - base[metric]).to_numpy()
                if metric in ("oof_spearman", "cens_spearman", "phase_skill", "phase_accuracy"):
                    better = other[metric] > base[metric]
                elif metric == "cens_bias":
                    better = other[metric].abs() < base[metric].abs()
                else:
                    better = other[metric] < base[metric]
                summary.append(
                    {
                        "target": target,
                        "arm": arm,
                        "metric": metric,
                        "mean_delta": float(delta.mean()),
                        "ci95_low": float(np.quantile(delta, 0.025)),
                        "ci95_high": float(np.quantile(delta, 0.975)),
                        "fraction_better": float(np.mean(better)),
                    }
                )
    summary_frame = pd.DataFrame(summary)
    summary_frame.to_csv(OUTPUT_DIR / "paired_summary.csv", index=False)
    print(summary_frame.to_string(index=False, float_format=lambda v: f"{v:+.5f}"))

    (OUTPUT_DIR / "provenance.json").write_text(
        json.dumps(
            {
                "run_budget": RUN_BUDGET,
                "draws": DRAWS,
                "screen_top_shapes": SCREEN_TOP_SHAPES,
                "band_sigma": BAND_SIGMA,
                "note": (
                    "both arms spend the same number of training runs; a pair costs two runs and yields "
                    "two observations, a two-phase row costs one run and yields one"
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
