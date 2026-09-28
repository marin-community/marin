# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Attribute the surrogate extrapolation bias to ridge shrinkage, and test directional ridge.

Every Observatory baseline under-rates good policies it has not seen: on top-censored
fits, prediction minus truth is positive for all five models at both targets. That is
the defect that matters for proposal, because a model that cannot believe in a policy
better than its training data will not propose one.

The mechanism tested here is shrinkage. Ridge pulls the nonnegative benefit
coefficients toward zero, which truncates how much improvement the model can express,
so rows better than anything in training are predicted too high. Forcing the ridge to
zero should remove the bias if that is the mechanism, at some cost in fit.

Directional ridge is the candidate fix that keeps both: leave the benefit block
unpenalized so the reachable improvement is not truncated, and penalize only the harm
and pooled blocks where shrinkage buys stability. The harness's ``penalty_scale`` hook
supplies per-column ridge multipliers, so this needs no new fitting machinery.

Bootstrap design. The censored set is a fixed property of the panel, so it is held
constant and only the training rows are resampled; that isolates estimation noise in
the fit rather than confounding it with which rows count as censored. Draws are shared
across variants so the reported statistic is the per-draw difference, which cancels the
component of variance common to both. Resampling is stratified by design series to
preserve the qsplit-to-domain-deletion composition. The unit is the row because each
row is an independent training run and the panel has only two series, too few to
resample as clusters.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dual_objective_harness_20260726 import (  # noqa: E402
    build_benchmark,
    fit_on,
    out_of_fold_predictions,
    select_by,
)
from swarm39_harness_20260725 import TABLE9, UNCHEATABLE, Model, Panel  # noqa: E402
from swarm39_models_20260725 import (  # noqa: E402
    _state_shapes,
    build_bucket_family_grp,
    build_compact_retained_state,
    build_hierarchical_phase_replay,
)

OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "shrinkage_extrapolation_bias_20260726"

TARGETS = (UNCHEATABLE, TABLE9)
CENSOR_FRACTION = 0.10
DEFAULT_L2_GRID = (0.0, 0.01, 0.1, 1.0)
DIRECTIONAL_L2_GRID = (0.01, 0.1, 1.0, 3.0)
BOOTSTRAP_DRAWS = 200
BOOTSTRAP_SEED = 20260726

DESIGNS = {
    "crs": (build_compact_retained_state, lambda: _state_shapes(False)),
    "bfgrp": (build_bucket_family_grp, lambda: _state_shapes(True)),
    "hpr": (build_hierarchical_phase_replay, lambda: _state_shapes(True)),
}


def benefit_columns(model: Model, panel: Panel, shape: dict) -> np.ndarray:
    """Boolean mask of the columns that carry improvement, classified by design sign.

    The head is nonnegative, so a column entering with non-positive values can only
    reduce predicted loss and a column entering non-negative can only increase it.
    Every design in this family splits cleanly on that test: benefit, coverage, and
    residual-utility blocks are non-positive, while replay, overexposure, and
    phase-shift blocks are non-negative. Classifying by sign rather than by column
    name keeps the mask correct when a design renames or adds a block, which
    name-matching silently got wrong for the hierarchical design.
    """
    matrix = model.build(panel, shape).matrix
    return matrix.max(axis=0) <= 1e-12


def directional_penalty(model: Model, panel: Panel, shape: dict) -> np.ndarray:
    """Ridge multipliers: zero on the improvement block, one on the harm block.

    Shrinking improvement coefficients truncates how much better than its training
    data the model can believe a policy is, which is the extrapolation bias. Shrinking
    harm coefficients costs nothing in that direction and still buys stability.
    """
    improvement = benefit_columns(model, panel, shape)
    assert improvement.any() and not improvement.all(), "sign split did not separate improvement from harm"
    return np.where(improvement, 0.0, 1.0)


def censored_split(observed: np.ndarray, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    available = np.isfinite(observed)
    n_censored = max(1, int(fraction * available.sum()))
    ordering = np.argsort(np.where(available, observed, np.inf))
    censored = np.zeros(len(observed), dtype=bool)
    censored[ordering[:n_censored]] = True
    return available & ~censored, censored


def censored_scores(
    panel: Panel, model: Model, target: str, shape: dict, l2: float, train: np.ndarray, censored: np.ndarray
) -> dict[str, float]:
    fitted = fit_on(panel, model, target, shape, l2, rows=train)
    prediction = fitted.predict(panel)[censored]
    truth = panel.targets[target][censored]
    residual = prediction - truth
    ranks = lambda values: np.argsort(np.argsort(values))  # noqa: E731
    return {
        "cens_rmse": float(np.sqrt(np.mean(residual**2))),
        "cens_bias": float(np.mean(residual)),
        "cens_spearman": float(np.corrcoef(ranks(prediction), ranks(truth))[0, 1]),
    }


def variants(design: str) -> dict[str, Model]:
    build, shapes = DESIGNS[design]
    return {
        "cv_ridge": Model(f"{design}_cv_ridge", build, shapes, l2_grid=DEFAULT_L2_GRID),
        "no_ridge": Model(f"{design}_no_ridge", build, shapes, l2_grid=(0.0,)),
        "directional_ridge": Model(
            f"{design}_directional",
            build,
            shapes,
            l2_grid=DIRECTIONAL_L2_GRID,
            penalty_scale=lambda panel, shape, _build=build, _shapes=shapes: directional_penalty(
                Model("probe", _build, _shapes), panel, shape
            ),
        ),
    }


def stratified_draw(series: np.ndarray, rows: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Resample training rows with replacement within each design series."""
    drawn: list[int] = []
    for name in np.unique(series[rows]):
        pool = np.flatnonzero(rows & (series == name))
        drawn.extend(rng.choice(pool, size=len(pool), replace=True).tolist())
    return np.asarray(drawn, dtype=int)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    benchmark = build_benchmark()
    panel = benchmark.fit_300m
    rows: list[dict[str, Any]] = []
    draws: list[dict[str, Any]] = []

    for target in TARGETS:
        observed = panel.targets[target]
        train, censored = censored_split(observed, CENSOR_FRACTION)
        for design in DESIGNS:
            chosen: dict[str, tuple[dict, float]] = {}
            for label, model in variants(design).items():
                # Selection sees training rows only, so the censored rows stay unseen.
                shape, l2, _ = select_by(panel, model, target, "rmse", rows=train)
                chosen[label] = (shape, l2)
                point = censored_scores(panel, model, target, shape, l2, train, censored)
                full_shape, full_l2, _ = select_by(panel, model, target, "rmse")
                oof = out_of_fold_predictions(panel, model, target, full_shape, full_l2)
                finite = np.isfinite(oof) & np.isfinite(observed)
                rows.append(
                    {
                        "target": target,
                        "design": design,
                        "variant": label,
                        "l2": l2,
                        "oof_rmse": float(np.sqrt(np.mean((oof[finite] - observed[finite]) ** 2))),
                        **point,
                    }
                )

            # Paired bootstrap: one shared set of resampled training rows per draw.
            rng = np.random.default_rng(BOOTSTRAP_SEED)
            for draw in range(BOOTSTRAP_DRAWS):
                indices = stratified_draw(panel.series, train, rng)
                resampled = np.zeros(len(observed), dtype=bool)
                resampled[np.unique(indices)] = True
                record = {"target": target, "design": design, "draw": draw}
                for label, model in variants(design).items():
                    shape, l2 = chosen[label]
                    scores = censored_scores(panel, model, target, shape, l2, resampled, censored)
                    for key, value in scores.items():
                        record[f"{label}:{key}"] = value
                draws.append(record)

    point_frame = pd.DataFrame(rows)
    draw_frame = pd.DataFrame(draws)
    point_frame.to_csv(OUTPUT_DIR / "point_estimates.csv", index=False)
    draw_frame.to_csv(OUTPUT_DIR / "bootstrap_draws.csv", index=False)

    print("=== point estimates (selection on training rows only) ===")
    print(point_frame.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    print("\n=== paired bootstrap differences vs cv_ridge (200 shared draws) ===")
    summary = []
    for (target, design), group in draw_frame.groupby(["target", "design"]):
        for label in ("no_ridge", "directional_ridge"):
            for metric in ("cens_bias", "cens_rmse", "cens_spearman"):
                delta = group[f"{label}:{metric}"].to_numpy() - group[f"cv_ridge:{metric}"].to_numpy()
                # Bias improves by moving toward zero, not by decreasing.
                better = np.abs(group[f"{label}:{metric}"]) < np.abs(group[f"cv_ridge:{metric}"])
                if metric == "cens_spearman":
                    better = group[f"{label}:{metric}"] > group[f"cv_ridge:{metric}"]
                elif metric == "cens_rmse":
                    better = group[f"{label}:{metric}"] < group[f"cv_ridge:{metric}"]
                summary.append(
                    {
                        "target": target,
                        "design": design,
                        "variant": label,
                        "metric": metric,
                        "mean_delta": float(delta.mean()),
                        "ci95_low": float(np.quantile(delta, 0.025)),
                        "ci95_high": float(np.quantile(delta, 0.975)),
                        "fraction_better": float(better.mean()),
                    }
                )
    summary_frame = pd.DataFrame(summary)
    summary_frame.to_csv(OUTPUT_DIR / "bootstrap_summary.csv", index=False)
    print(summary_frame.to_string(index=False, float_format=lambda v: f"{v:+.5f}"))

    (OUTPUT_DIR / "provenance.json").write_text(
        json.dumps(
            {
                "censor_fraction": CENSOR_FRACTION,
                "bootstrap_draws": BOOTSTRAP_DRAWS,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "n_fit_rows": len(panel.row_id),
                "resampling_unit": "row, stratified by design series",
                "censored_set": "fixed across draws; only training rows are resampled",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
