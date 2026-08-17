# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy",
#   "fsspec",
#   "gcsfs",
#   "matplotlib",
#   "numpy",
#   "pandas",
#   "plotly",
#   "pyarrow",
#   "scikit-learn",
#   "scipy",
# ]
# ///
"""Verify whether the blocked-WSD80 Student-t failure is an optimizer artifact.

This follow-up was frozen after the head-only ablation and independent review.
It keeps the published repaired-RPL shape, ridge, and blocked folds fixed. The
Student-t scale is the Huber-fit training MAD, and each fold is solved from both
least-squares and Huber starts with an analytic gradient and explicit bounded
optimization diagnostics.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    audit_wsd80_cross_metric_rpl_20260730 as wsd_audit,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_noise_aware_rpl_heads_20260815 as ablation,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_repaired_rpl_wsd80_controls_20260731 as repaired_wsd,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    retained_power_law_estimator_repair_20260731 as repaired,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    retained_power_law_model_20260728 as rpl,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    starcoder_wsd80_panel_20260728 as wsd80,
)

OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "noise_aware_rpl_student_t_diagnostic_20260815"
PROTOCOL_VERSION = "noise-aware-rpl-student-t-diagnostic-v1"
MAX_ITERATIONS = 5_000
GRADIENT_TOLERANCE = 1e-10
OBJECTIVE_AGREEMENT_TOLERANCE = 1e-8
BLOCKED_RMSE_REJECTION_FLOOR = 0.030
BLOCKED_REGRET_REJECTION_FLOOR = 0.020


@dataclass(frozen=True)
class SolveResult:
    """One fixed-scale Student-t fit and its optimizer diagnostics."""

    head: ablation.Head
    start: str
    success: bool
    iterations: int
    objective: float
    gradient_inf_norm: float
    message: str


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def protocol_payload() -> dict[str, Any]:
    sources = (Path(__file__), Path(ablation.__file__), Path(repaired.__file__), Path(rpl.__file__))
    return {
        "version": PROTOCOL_VERSION,
        "frozen_before_diagnostic_results": True,
        "panel": "WSD80 Programming Languages BPB",
        "protocol": "blocked",
        "shape_and_ridge": "published fold-specific repaired-RPL selections",
        "student_df": ablation.STUDENT_DF,
        "scale": "training-fold Huber-residual MAD sigma, fixed during optimization",
        "starts": ["least_squares", "huber"],
        "optimizer": "L-BFGS-B with analytic gradient",
        "max_iterations": MAX_ITERATIONS,
        "gradient_tolerance": GRADIENT_TOLERANCE,
        "objective_agreement_tolerance": OBJECTIVE_AGREEMENT_TOLERANCE,
        "decision_rule": {
            "student_t_blocked_rmse_at_least": BLOCKED_RMSE_REJECTION_FLOOR,
            "student_t_regret_at_1_at_least": BLOCKED_REGRET_REJECTION_FLOOR,
            "both_starts_successful": True,
            "both_starts_agree_in_objective": True,
            "interpretation": "meeting every condition confirms that the blocked collapse is a loss property",
        },
        "source_hashes": {str(path.relative_to(REPO_ROOT)): ablation.file_hash(path) for path in sources},
    }


def scaled_coefficients(head: ablation.Head, scale: np.ndarray) -> np.ndarray:
    return np.concatenate([[head.intercept], head.coefficients * scale])


def fixed_student_objective(
    coefficients: np.ndarray,
    design: np.ndarray,
    target: np.ndarray,
    scale: float,
    ridge: float,
    multipliers: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Student-t robust loss scaled to have squared-error local curvature."""
    residual = design @ coefficients - target
    denominator = ablation.STUDENT_DF * scale**2
    loss = ablation.STUDENT_DF * scale**2 * np.log1p(residual**2 / denominator)
    penalty = ridge * float(np.sum(multipliers * coefficients[1:] ** 2))
    objective = float(loss.sum() + penalty)
    influence = 2.0 * residual / (1.0 + residual**2 / denominator)
    gradient = design.T @ influence
    gradient[1:] += 2.0 * ridge * multipliers * coefficients[1:]
    return objective, gradient


def solve_from_start(
    name: str,
    initial: np.ndarray,
    design: np.ndarray,
    target: np.ndarray,
    scale: float,
    ridge: float,
    multipliers: np.ndarray,
    layout: repaired.FeatureLayout,
    column_scale: np.ndarray,
) -> SolveResult:
    bounds = repaired._coefficient_bounds(layout)
    scipy_bounds = list(zip(bounds[0], bounds[1], strict=True))

    def objective(coefficients: np.ndarray) -> tuple[float, np.ndarray]:
        return fixed_student_objective(coefficients, design, target, scale, ridge, multipliers)

    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        bounds=scipy_bounds,
        options={"maxiter": MAX_ITERATIONS, "gtol": GRADIENT_TOLERANCE, "ftol": 1e-15, "maxls": 100},
    )
    final_objective, gradient = objective(np.asarray(result.x, dtype=float))
    unscaled = np.asarray(result.x[1:], dtype=float) / column_scale
    return SolveResult(
        head=ablation.Head(float(result.x[0]), unscaled),
        start=name,
        success=bool(result.success),
        iterations=int(result.nit),
        objective=final_objective,
        gradient_inf_norm=float(np.max(np.abs(gradient))),
        message=str(result.message),
    )


def fit_fold(
    design: np.ndarray,
    target: np.ndarray,
    ridge: float,
    multipliers: np.ndarray,
    layout: repaired.FeatureLayout,
) -> tuple[SolveResult, list[SolveResult], float]:
    least_squares = ablation.fit_head(
        "mse",
        design,
        target,
        ridge,
        multipliers,
        layout,
        np.ones(len(target)),
        ablation.PairIndex(np.asarray([], dtype=int), np.asarray([], dtype=int)),
    )
    huber = ablation.fit_head(
        "huber",
        design,
        target,
        ridge,
        multipliers,
        layout,
        np.ones(len(target)),
        ablation.PairIndex(np.asarray([], dtype=int), np.asarray([], dtype=int)),
    )
    residual = ablation.predict(huber, design) - target
    fixed_scale = rpl.MAD_TO_SIGMA * float(np.median(np.abs(residual - np.median(residual))))
    if fixed_scale <= 0.0:
        raise ValueError("Huber training residuals have zero MAD")
    column_scale = repaired._column_scale(design, layout)
    scaled_design = np.column_stack([np.ones(len(target)), design / column_scale])
    results = [
        solve_from_start(
            name,
            scaled_coefficients(head, column_scale),
            scaled_design,
            target,
            fixed_scale,
            ridge,
            multipliers,
            layout,
            column_scale,
        )
        for name, head in (("least_squares", least_squares), ("huber", huber))
    ]
    selected = min(results, key=lambda item: item.objective)
    return selected, results, fixed_scale


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT_DIR / "protocol.json", protocol_payload())
    panel, frame, available = wsd_audit.load_metric_panel()
    if ablation.PRIMARY_TARGET not in available:
        raise ValueError(f"missing target {ablation.PRIMARY_TARGET}")
    target = frame[ablation.PRIMARY_TARGET].to_numpy(dtype=float)
    geometry = rpl.Geometry(panel.c0, panel.c1, wsd80.REALIZED_PHASE_0_FRACTION)
    folds = repaired_wsd.fold_builder(
        "blocked",
        panel.weights,
        np.arange(len(target)),
        repaired_wsd.OUTER_SPLITS,
        repaired_wsd.OUTER_SEED,
    )
    selections = ablation.load_json(
        ablation.WSD_REFERENCE_DIR / "cells" / "blocked" / ablation.PRIMARY_HASH / "fold_selections.json"
    )
    prediction = np.full(len(target), np.nan)
    diagnostic_rows: list[dict[str, Any]] = []
    for fold, (train, test) in enumerate(folds):
        selected_shape = next(row for row in selections if int(row["outer_fold"]) == fold)
        if len(train) != int(selected_shape["train_rows"]) or len(test) != int(selected_shape["test_rows"]):
            raise ValueError(f"blocked fold {fold} no longer matches its frozen selection")
        shape = ablation.shape_from_dict(selected_shape["shape"])
        ridge = float(selected_shape["ridge"])
        design, layout = repaired.design_matrix(panel.weights, geometry, shape)
        multipliers = repaired.penalty_multipliers(geometry, layout)
        chosen, starts, fixed_scale = fit_fold(design[train], target[train], ridge, multipliers, layout)
        prediction[test] = ablation.predict(chosen.head, design[test])
        best_objective = min(result.objective for result in starts)
        for result in starts:
            diagnostic_rows.append(
                {
                    "outer_fold": fold,
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "ridge": ridge,
                    "shape": json.dumps(asdict(shape), sort_keys=True),
                    "fixed_scale": fixed_scale,
                    "start": result.start,
                    "selected": result is chosen,
                    "success": result.success,
                    "iterations": result.iterations,
                    "objective": result.objective,
                    "objective_excess": result.objective - best_objective,
                    "gradient_inf_norm": result.gradient_inf_norm,
                    "message": result.message,
                }
            )
    if not np.isfinite(prediction).all():
        raise RuntimeError("fixed-scale Student-t produced incomplete predictions")

    masks, _best = wsd_audit.subset_masks(panel, target)
    interior = masks["interior"]
    metric = ablation.metrics(
        "wsd80",
        "programming_languages",
        "blocked",
        "fixed_scale_student_t_two_start",
        target[interior],
        prediction[interior],
        ablation.restrict_folds(folds, interior),
        np.ones(int(interior.sum()), dtype=bool),
    )
    diagnostics = pd.DataFrame(diagnostic_rows)
    starts_agree = bool(
        diagnostics.groupby("outer_fold")["objective_excess"].max().max() <= OBJECTIVE_AGREEMENT_TOLERANCE
    )
    all_successful = bool(diagnostics["success"].all())
    collapse_confirmed = bool(
        metric["rmse"] >= BLOCKED_RMSE_REJECTION_FLOOR
        and metric["regret_at_1"] >= BLOCKED_REGRET_REJECTION_FLOOR
        and all_successful
        and starts_agree
    )
    decision = {
        "collapse_confirmed": collapse_confirmed,
        "all_starts_successful": all_successful,
        "all_starts_agree_in_objective": starts_agree,
        "maximum_objective_excess": float(diagnostics["objective_excess"].max()),
        "metrics": metric,
    }
    diagnostics.to_csv(OUTPUT_DIR / "optimizer_diagnostics.csv", index=False)
    pd.DataFrame(
        {
            "row_index": np.arange(len(target)),
            "observed": target,
            "predicted": prediction,
            "residual": prediction - target,
            "interior": interior,
        }
    ).to_csv(OUTPUT_DIR / "oof_predictions.csv", index=False)
    write_json(OUTPUT_DIR / "decision.json", decision)
    (OUTPUT_DIR / "report.md").write_text(
        "# Fixed-scale Student-t optimizer diagnostic\n\n"
        f"- Collapse confirmed under the frozen rule: **{collapse_confirmed}**\n"
        f"- Interior blocked RMSE: `{metric['rmse']:.6f}`\n"
        f"- Interior blocked Regret@1: `{metric['regret_at_1']:.6f}`\n"
        f"- Both starts successful: `{all_successful}`\n"
        f"- Both starts agree in objective: `{starts_agree}`\n"
        f"- Maximum objective excess: `{diagnostics['objective_excess'].max():.3e}`\n\n"
        "The nonlinear shape, ridge, folds, scale rule, starts, optimizer, and decision rule were fixed "
        "before this diagnostic was run.\n"
    )
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
