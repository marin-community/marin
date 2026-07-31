# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Interpolate between the additive and multiplicative links to buy fit without bias.

Two facts from this experiment pull in opposite directions. The multiplicative
log-deficit link improves in-panel fit and the ordering of good policies on the
hierarchical replay design, taking OOF RMSE from 0.00788 to 0.00769 and censored
Spearman from 0.5315 to 0.6388, both better than any Observatory baseline. But it
worsens censored bias, from +0.00029 under an unridged additive fit to +0.00561, so it
goes back to under-rating the policies a proposal would target.

A power link nests both. With ``d = y - floor`` the deficit and

    eta = log d                      for lambda = 0,
    eta = (d**lambda - 1) / lambda   otherwise,

lambda = 0 is the multiplicative link and lambda = 1 is affine in the deficit, hence
equivalent to the additive fit up to the free intercept. Sweeping lambda asks whether
the fit and ordering gains survive at a lambda whose extrapolation is still unbiased,
or whether the two properties are the same knob and cannot be separated.

The floor is swept alongside, because it sets how much of the response range the link
compresses. A floor far below the observed minimum leaves the deficit large and the
transform nearly affine over the observed range; a floor just under the minimum makes
the transform steep exactly where the censored rows sit. If the bias is a compression
artifact rather than intrinsic to curvature, a lower floor should relieve it.

The link is implemented locally rather than added to the shared harness, because two
concurrent explorations import that harness and a new branch there would change code
under them mid-run.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dual_objective_harness_20260726 import build_benchmark  # noqa: E402
from swarm39_harness_20260725 import TABLE9, UNCHEATABLE, Model, Panel, fit_head, grouped_splits  # noqa: E402
from swarm39_models_20260725 import (  # noqa: E402
    _state_shapes,
    build_compact_retained_state,
    build_hierarchical_phase_replay,
)

OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "power_deficit_link_20260726"

TARGETS = (UNCHEATABLE, TABLE9)
LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)
FLOOR_FRACTIONS = (0.70, 0.85, 0.95, 0.99)
L2_GRIDS = {"no_ridge": (0.0,), "cv_ridge": (0.0, 0.01, 0.1, 1.0)}
CENSOR_FRACTION = 0.10
N_SPLITS = 5
SPLIT_SEED = 0
BOOTSTRAP_DRAWS = 150
BOOTSTRAP_SEED = 20260726
# Guard on the inverse transform so a runaway linear predictor cannot produce a
# negative deficit or an overflow instead of an obviously wrong prediction.
DEFICIT_FLOOR_EPSILON = 1e-9

DESIGNS = {
    "hpr": (build_hierarchical_phase_replay, lambda: _state_shapes(True)),
    "crs": (build_compact_retained_state, lambda: _state_shapes(False)),
}


def censored_split(observed: np.ndarray, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Training rows and the censored best rows, by observed target."""
    available = np.isfinite(observed)
    n_censored = max(1, int(fraction * available.sum()))
    ordering = np.argsort(np.where(available, observed, np.inf))
    censored = np.zeros(len(observed), dtype=bool)
    censored[ordering[:n_censored]] = True
    return available & ~censored, censored


def to_link(deficit: np.ndarray, lam: float) -> np.ndarray:
    safe = np.maximum(deficit, DEFICIT_FLOOR_EPSILON)
    return np.log(safe) if lam == 0.0 else (safe**lam - 1.0) / lam


def from_link(eta: np.ndarray, lam: float) -> np.ndarray:
    if lam == 0.0:
        return np.exp(np.clip(eta, -30.0, 30.0))
    base = 1.0 + lam * eta
    return np.maximum(base, DEFICIT_FLOOR_EPSILON) ** (1.0 / lam)


def fit_predict(
    panel: Panel,
    model: Model,
    target: str,
    shape: dict,
    l2: float,
    lam: float,
    floor_fraction: float,
    train: np.ndarray,
) -> np.ndarray:
    """Fit on ``train`` under the power-deficit link and predict every row."""
    observed = panel.targets[target]
    rows = train & np.isfinite(observed)
    design = model.build(panel, shape).matrix
    floor = floor_fraction * float(np.min(observed[rows]))
    intercept, coefficients = fit_head(design[rows], to_link(observed[rows] - floor, lam), l2)
    return floor + from_link(intercept + design @ coefficients, lam)


def out_of_fold(
    panel: Panel, model: Model, target: str, shape: dict, l2: float, lam: float, floor_fraction: float
) -> np.ndarray:
    observed = panel.targets[target]
    prediction = np.full(len(observed), np.nan)
    for train, test in grouped_splits(panel, N_SPLITS, SPLIT_SEED):
        rows = train & np.isfinite(observed)
        if rows.sum() < 2:
            continue
        prediction[test] = fit_predict(panel, model, target, shape, l2, lam, floor_fraction, rows)[test]
    return prediction


def select(
    panel: Panel,
    model: Model,
    target: str,
    l2_grid: tuple[float, ...],
    lam: float,
    floor_fraction: float,
    rows: np.ndarray | None = None,
) -> tuple[dict, float]:
    """Choose shape and ridge by grouped OOF RMSE, on the supplied rows only."""
    observed = panel.targets[target]
    if rows is not None:
        observed = np.where(rows, observed, np.nan)
    scratch = dataclasses.replace(panel, targets={**panel.targets, target: observed})
    best: tuple[float, dict, float] | None = None
    for shape in model.shapes():
        for l2 in l2_grid:
            prediction = out_of_fold(scratch, model, target, shape, l2, lam, floor_fraction)
            finite = np.isfinite(prediction) & np.isfinite(observed)
            score = float(np.sqrt(np.mean((prediction[finite] - observed[finite]) ** 2)))
            if best is None or score < best[0]:
                best = (score, shape, l2)
    assert best is not None, "empty grid"
    return best[1], best[2]


def scores(panel: Panel, model: Model, target: str, shape: dict, l2: float, lam: float, floor: float) -> dict[str, Any]:
    observed = panel.targets[target]
    train, censored = censored_split(observed, CENSOR_FRACTION)
    oof = out_of_fold(panel, model, target, shape, l2, lam, floor)
    finite = np.isfinite(oof) & np.isfinite(observed)
    prediction = fit_predict(panel, model, target, shape, l2, lam, floor, train)[censored]
    truth = observed[censored]
    residual = prediction - truth
    ranks = lambda v: np.argsort(np.argsort(v))  # noqa: E731
    return {
        "oof_rmse": float(np.sqrt(np.mean((oof[finite] - observed[finite]) ** 2))),
        "oof_spearman": float(np.corrcoef(ranks(oof[finite]), ranks(observed[finite]))[0, 1]),
        "cens_rmse": float(np.sqrt(np.mean(residual**2))),
        "cens_bias": float(np.mean(residual)),
        "cens_spearman": float(np.corrcoef(ranks(prediction), ranks(truth))[0, 1]),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    benchmark = build_benchmark()
    panel = benchmark.fit_300m
    rows: list[dict[str, Any]] = []

    for target in TARGETS:
        train, _ = censored_split(panel.targets[target], CENSOR_FRACTION)
        for design, (build, shapes) in DESIGNS.items():
            model = Model(design, build, shapes)
            for ridge_label, l2_grid in L2_GRIDS.items():
                for lam in LAMBDAS:
                    for floor_fraction in FLOOR_FRACTIONS:
                        shape, l2 = select(panel, model, target, l2_grid, lam, floor_fraction, rows=train)
                        rows.append(
                            {
                                "target": target,
                                "design": design,
                                "ridge": ridge_label,
                                "lambda": lam,
                                "floor_fraction": floor_fraction,
                                "l2": l2,
                                **scores(panel, model, target, shape, l2, lam, floor_fraction),
                            }
                        )
                        print(f"  done {target[:11]:11s} {design} {ridge_label:8s} lam={lam} floor={floor_fraction}")

    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT_DIR / "link_sweep.csv", index=False)
    print("\n=== power-deficit link sweep ===")
    for target in TARGETS:
        for design in DESIGNS:
            block = frame[(frame.target == target) & (frame.design == design)]
            print(f"\n-- {target} / {design} --")
            print(block.drop(columns=["target", "design"]).to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    (OUTPUT_DIR / "provenance.json").write_text(
        json.dumps(
            {
                "lambdas": list(LAMBDAS),
                "floor_fractions": list(FLOOR_FRACTIONS),
                "censor_fraction": CENSOR_FRACTION,
                "note": (
                    "lambda 0 is the log-deficit link; lambda 1 is affine in the deficit and so equivalent "
                    "to the additive fit up to the intercept"
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
