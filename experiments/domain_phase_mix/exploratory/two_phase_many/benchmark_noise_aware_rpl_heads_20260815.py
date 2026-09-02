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
#   "tabulate",
# ]
# ///
"""Frozen loss-function ablation for the repaired retained-power-law surrogate.

The nonlinear RPL shape and ridge are loaded from the published nested fits. Only
the linear-head likelihood changes. This makes the screen cheap and prevents a
shape search from compensating for a changed loss. A positive result must later
survive fully nested shape selection; a negative result blocks the claim that a
different residual loss alone repairs policy selection.

WSD80 supports an independently estimated relative variance shape from four-seed
calibration groups. The 300M panel does not, so heteroskedastic variants are not
reported there. Designed aggregate-matched policy differences enter only through
a fixed 50/50 auxiliary objective; they never replace the absolute observations.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    audit_wsd80_cross_metric_rpl_20260730 as wsd_audit,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_expanded_300m_pareto_baseline_20260731 as baseline_300m,
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

OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "noise_aware_rpl_head_ablation_20260815"
WSD_REFERENCE_DIR = SCRIPT_DIR / "reference_outputs" / "repaired_rpl_wsd80_controls_20260731"
RPL_300M_DIR = SCRIPT_DIR / "reference_outputs" / "repaired_rpl_300m_20260731"
VARIANCE_DIR = SCRIPT_DIR / "reference_outputs" / "starcoder_wsd80_dense_support_calibration_results_20260813"
PRIMARY_HASH = "f24e6168aead"
PRIMARY_TARGET = wsd_audit.PRIMARY_TARGET
PROTOCOL_VERSION = "noise-aware-rpl-head-ablation-v1"
STUDENT_DF = 4.0
PAIR_CHANNEL_SHARE = 0.5
IRLS_ITERATIONS = 80
IRLS_TOLERANCE = 1e-4
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260815
GRID_RESOLUTION = 401
LOWER_TAIL_FRACTION = 0.15
OPTIMISM_THRESHOLD = 0.05
EXPORT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}

WSD_VARIANTS = (
    "mse",
    "huber",
    "student_t",
    "hetero_student_t",
    "hetero_student_t_paired",
)
PANEL_300M_VARIANTS = ("mse", "huber", "student_t", "student_t_paired")


@dataclass(frozen=True)
class Head:
    """A fitted mixed-sign repaired-RPL linear head."""

    intercept: float
    coefficients: np.ndarray


@dataclass(frozen=True)
class PairIndex:
    """Absolute-row indices defining aggregate-matched contrasts."""

    tied: np.ndarray
    asymmetric: np.ndarray


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def shape_from_dict(values: dict[str, Any]) -> rpl.Shape:
    return rpl.Shape(
        benefit_exponent=float(values["benefit_exponent"]),
        benefit_offset=float(values["benefit_offset"]),
        damage_exponent=float(values["damage_exponent"]),
        damage_threshold=float(values["damage_threshold"]),
        retention=float(values["retention"]),
        late_multiplier=float(values["late_multiplier"]),
        ordering_channel=bool(values["ordering_channel"]),
    )


def protocol_payload() -> dict[str, Any]:
    sources = (
        Path(__file__),
        Path(repaired.__file__),
        Path(rpl.__file__),
        WSD_REFERENCE_DIR / "cells" / "random" / PRIMARY_HASH / "fold_selections.json",
        WSD_REFERENCE_DIR / "cells" / "blocked" / PRIMARY_HASH / "fold_selections.json",
        RPL_300M_DIR / "cells" / "uncheatable" / "retained_power_law_repaired" / "fold_selections.json",
        RPL_300M_DIR / "cells" / "table9" / "retained_power_law_repaired" / "fold_selections.json",
        VARIANCE_DIR / "variance_model_coefficients.csv",
    )
    return {
        "version": PROTOCOL_VERSION,
        "frozen_before_results": True,
        "student_df": STUDENT_DF,
        "pair_channel_share": PAIR_CHANNEL_SHARE,
        "irls_iterations": IRLS_ITERATIONS,
        "irls_tolerance": IRLS_TOLERANCE,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "wsd80_variants": WSD_VARIANTS,
        "300m_variants": PANEL_300M_VARIANTS,
        "shape_policy": "reuse published fold-specific and full-fit RPL shapes and ridge",
        "heteroskedasticity_policy": (
            "WSD80 uses only the four common variance-shape coefficients learned from the independent "
            "dense-support calibration; the block intercept is re-estimated as one training-fold scale. "
            "The 300M panel has no coordinate-resolved variance calibration, so no heteroskedastic arm is run."
        ),
        "paired_policy": (
            "absolute observations retain half the auxiliary-objective weight and designed aggregate-matched "
            "differences receive half; total row weight remains equal to the number of absolute rows"
        ),
        "evaluation_policy": (
            "WSD80 headline OOF and bootstrap metrics use the published interior mask from "
            "audit_wsd80_cross_metric_rpl_20260730; pair-delta diagnostics retain all exact pairs. "
            "The 300M headline OOF and bootstrap metrics use all rows."
        ),
        "source_hashes": {str(path.relative_to(REPO_ROOT)): file_hash(path) for path in sources},
    }


def wsd_relative_variance(weights: np.ndarray) -> np.ndarray:
    """Independent relative variance shape, with its geometric mean normalized to one."""
    coefficients = pd.read_csv(VARIANCE_DIR / "variance_model_coefficients.csv")
    shape = coefficients.loc[
        coefficients["parameter"].isin(("aggregate", "aggregate_squared", "absolute_contrast", "contrast_squared"))
    ].set_index("parameter")["coefficient"]
    phase_0 = weights[:, 0, 1]
    phase_1 = weights[:, 1, 1]
    aggregate = wsd80.PHASE_0_FRACTION * phase_0 + wsd80.PHASE_1_FRACTION * phase_1
    contrast = phase_1 - phase_0
    log_variance = (
        float(shape["aggregate"]) * aggregate
        + float(shape["aggregate_squared"]) * aggregate**2
        + float(shape["absolute_contrast"]) * np.abs(contrast)
        + float(shape["contrast_squared"]) * contrast**2
    )
    relative = np.exp(log_variance - np.mean(log_variance))
    return relative / np.exp(np.mean(np.log(relative)))


def wsd_pairs(weights: np.ndarray) -> PairIndex:
    phase_0 = weights[:, 0, 1]
    phase_1 = weights[:, 1, 1]
    tied_mask = np.isclose(phase_0, phase_1, atol=1e-9)
    aggregate = wsd80.PHASE_0_FRACTION * phase_0 + wsd80.PHASE_1_FRACTION * phase_1
    tied_by_key = {round(float(aggregate[index]), 9): int(index) for index in np.flatnonzero(tied_mask)}
    tied: list[int] = []
    asymmetric: list[int] = []
    for index in np.flatnonzero(~tied_mask):
        counterpart = tied_by_key.get(round(float(aggregate[index]), 9))
        if counterpart is not None:
            tied.append(counterpart)
            asymmetric.append(int(index))
    return PairIndex(np.asarray(tied, dtype=int), np.asarray(asymmetric, dtype=int))


def panel_300m_pairs(frame: pd.DataFrame, weights: np.ndarray) -> PairIndex:
    tied_mask = np.all(np.isclose(weights[:, 0, :], weights[:, 1, :], atol=1e-10, rtol=0.0), axis=1)
    tied_by_key: dict[str, int] = {}
    single_phase = frame["policy_family"].eq("single_phase").to_numpy()
    two_phase = frame["policy_family"].eq("two_phase").to_numpy()
    for index in np.flatnonzero(tied_mask & single_phase):
        key = str(frame.iloc[index]["phase_correspondence_key"])
        if key in tied_by_key:
            raise ValueError(f"duplicate single-phase row for correspondence key {key}")
        tied_by_key[key] = int(index)
    tied: list[int] = []
    asymmetric: list[int] = []
    for index in np.flatnonzero(~tied_mask & two_phase):
        key = str(frame.iloc[index]["phase_correspondence_key"])
        counterpart = tied_by_key.get(key)
        if counterpart is not None:
            tied.append(counterpart)
            asymmetric.append(int(index))
    return PairIndex(np.asarray(tied, dtype=int), np.asarray(asymmetric, dtype=int))


def subset_pairs(pairs: PairIndex, rows: np.ndarray) -> PairIndex:
    position = {int(row): index for index, row in enumerate(rows)}
    tied: list[int] = []
    asymmetric: list[int] = []
    for tied_row, asymmetric_row in zip(pairs.tied, pairs.asymmetric, strict=True):
        if int(tied_row) in position and int(asymmetric_row) in position:
            tied.append(position[int(tied_row)])
            asymmetric.append(position[int(asymmetric_row)])
    return PairIndex(np.asarray(tied, dtype=int), np.asarray(asymmetric, dtype=int))


def robust_scale(residual: np.ndarray, relative_sd: np.ndarray) -> float:
    standardized = residual / np.maximum(relative_sd, 1e-12)
    mad = rpl.MAD_TO_SIGMA * float(np.median(np.abs(standardized - np.median(standardized))))
    if mad > 1e-10:
        initial = mad
    else:
        initial = max(float(np.sqrt(np.mean(standardized**2))), 1e-6)
    upper = max(10.0 * initial, float(np.ptp(standardized)), 1e-4)

    def objective(log_scale: float) -> float:
        scale = math.exp(log_scale)
        z2 = (standardized / scale) ** 2
        return float(len(residual) * log_scale + 0.5 * (STUDENT_DF + 1.0) * np.log1p(z2 / STUDENT_DF).sum())

    result = minimize_scalar(objective, bounds=(math.log(1e-7), math.log(upper)), method="bounded")
    return float(math.exp(result.x))


def solve_student_head(
    design: np.ndarray,
    target: np.ndarray,
    ridge: float,
    multipliers: np.ndarray,
    layout: repaired.FeatureLayout,
    relative_variance: np.ndarray,
    pairs: PairIndex | None,
) -> Head:
    """Constrained Student-t IRLS, optionally with a paired-difference channel."""
    if len(relative_variance) != len(target):
        raise ValueError("one relative variance is required per absolute observation")
    scale = repaired._column_scale(design, layout)
    absolute_design = np.column_stack([np.ones(len(target)), design / scale])
    absolute_sd = np.sqrt(np.maximum(relative_variance, 1e-12))
    absolute_sd /= np.exp(np.mean(np.log(absolute_sd)))

    data_design = absolute_design
    data_target = target
    channel_slices = [slice(0, len(target))]
    channel_sd = [absolute_sd]
    base_weights = [np.ones(len(target))]
    if pairs is not None:
        if not len(pairs.tied):
            raise ValueError("paired Student-t requested without any complete training pairs")
        pair_design = absolute_design[pairs.asymmetric] - absolute_design[pairs.tied]
        pair_target = target[pairs.asymmetric] - target[pairs.tied]
        pair_sd = np.sqrt(relative_variance[pairs.asymmetric] + relative_variance[pairs.tied])
        pair_sd /= np.exp(np.mean(np.log(pair_sd)))
        data_design = np.vstack([absolute_design, pair_design])
        data_target = np.concatenate([target, pair_target])
        channel_slices.append(slice(len(target), len(target) + len(pair_target)))
        channel_sd.append(pair_sd)
        total = len(target)
        base_weights = [
            np.full(len(target), (1.0 - PAIR_CHANNEL_SHARE) * total / len(target)),
            np.full(len(pair_target), PAIR_CHANNEL_SHARE * total / len(pair_target)),
        ]

    penalty = np.diag(np.concatenate([[0.0], np.sqrt(ridge * multipliers)]))
    augmented = np.vstack([data_design, penalty])
    response = np.concatenate([data_target, np.zeros(len(penalty))])
    bounds = repaired._coefficient_bounds(layout)
    coefficients = repaired._bounded_solve(augmented, response, bounds, len(data_target), np.concatenate(base_weights))

    for _ in range(IRLS_ITERATIONS):
        residual = data_design @ coefficients - data_target
        updated_weights: list[np.ndarray] = []
        fitted_scales: list[float] = []
        for channel, relative_sd, base in zip(channel_slices, channel_sd, base_weights, strict=True):
            channel_residual = residual[channel]
            fitted_scale = robust_scale(channel_residual, relative_sd)
            fitted_scales.append(fitted_scale)
            standardized = channel_residual / (fitted_scale * relative_sd)
            likelihood_weight = (STUDENT_DF + 1.0) / (STUDENT_DF + standardized**2)
            channel_weight = likelihood_weight / relative_sd**2
            channel_weight /= np.mean(channel_weight)
            updated_weights.append(base * channel_weight)
        weights = np.concatenate(updated_weights)
        weights *= len(target) / weights.sum()
        updated = repaired._bounded_solve(augmented, response, bounds, len(data_target), weights)
        shift = float(np.max(np.abs(data_design @ (updated - coefficients))))
        coefficients = updated
        if shift < IRLS_TOLERANCE * max(min(fitted_scales), 1e-12):
            break

    return Head(float(coefficients[0]), coefficients[1:] / scale)


def fit_head(
    variant: str,
    design: np.ndarray,
    target: np.ndarray,
    ridge: float,
    multipliers: np.ndarray,
    layout: repaired.FeatureLayout,
    relative_variance: np.ndarray,
    pairs: PairIndex,
) -> Head:
    if variant == "mse":
        intercept, aggregate, phase = repaired.solve_head(design, target, ridge, multipliers, layout, huber_scale=None)
        return Head(intercept, np.concatenate([aggregate, phase]))
    if variant == "huber":
        intercept, aggregate, phase = repaired.solve_head(design, target, ridge, multipliers, layout)
        return Head(intercept, np.concatenate([aggregate, phase]))
    use_heteroskedastic = variant.startswith("hetero_")
    use_pairs = variant.endswith("_paired")
    variance = relative_variance if use_heteroskedastic else np.ones(len(target))
    return solve_student_head(design, target, ridge, multipliers, layout, variance, pairs if use_pairs else None)


def predict(head: Head, design: np.ndarray) -> np.ndarray:
    return head.intercept + design @ head.coefficients


def calibration_slope(observed: np.ndarray, predicted: np.ndarray) -> float:
    centered = predicted - predicted.mean()
    denominator = float(centered @ centered)
    return float(centered @ (observed - observed.mean()) / denominator) if denominator > 0 else float("nan")


def lower_tail_rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    count = max(5, math.ceil(LOWER_TAIL_FRACTION * len(observed)))
    selected = np.argsort(predicted)[:count]
    return float(np.sqrt(np.mean((predicted[selected] - observed[selected]) ** 2)))


def fold_regret(
    observed: np.ndarray,
    predicted: np.ndarray,
    folds: tuple[tuple[np.ndarray, np.ndarray], ...],
    eligible: np.ndarray,
    k: int,
) -> float:
    regrets: list[float] = []
    for _train, test in folds:
        candidates = test[eligible[test]]
        if not len(candidates):
            continue
        selected = candidates[np.argsort(predicted[candidates])[: min(k, len(candidates))]]
        regrets.append(float(np.min(observed[selected]) - np.min(observed[candidates])))
    return float(np.mean(regrets))


def metrics(
    panel: str,
    target: str,
    protocol: str,
    variant: str,
    observed: np.ndarray,
    predicted: np.ndarray,
    folds: tuple[tuple[np.ndarray, np.ndarray], ...],
    eligible: np.ndarray,
) -> dict[str, Any]:
    residual = predicted - observed
    return {
        "panel": panel,
        "target": target,
        "protocol": protocol,
        "variant": variant,
        "rows": len(observed),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "bias_predicted_minus_observed": float(np.mean(residual)),
        "calibration_slope_observed_on_predicted": calibration_slope(observed, predicted),
        "spearman": float(spearmanr(observed, predicted).statistic),
        "lower_tail_rmse": lower_tail_rmse(observed[eligible], predicted[eligible]),
        "optimism_gt_0p05": int(np.sum(observed - predicted > OPTIMISM_THRESHOLD)),
        "worst_optimism": float(np.max(observed - predicted)),
        "regret_at_1": fold_regret(observed, predicted, folds, eligible, 1),
        "regret_at_3": fold_regret(observed, predicted, folds, eligible, 3),
        "regret_at_5": fold_regret(observed, predicted, folds, eligible, 5),
    }


def pair_metrics(
    panel: str,
    target: str,
    protocol: str,
    variant: str,
    observed: np.ndarray,
    predicted: np.ndarray,
    pairs: PairIndex,
    fold_assignment: np.ndarray,
) -> dict[str, Any]:
    same_fold = fold_assignment[pairs.tied] == fold_assignment[pairs.asymmetric]
    tied = pairs.tied[same_fold]
    asymmetric = pairs.asymmetric[same_fold]
    observed_delta = observed[asymmetric] - observed[tied]
    predicted_delta = predicted[asymmetric] - predicted[tied]
    residual = predicted_delta - observed_delta
    sign = np.sign(observed_delta) == np.sign(predicted_delta)
    return {
        "panel": panel,
        "target": target,
        "protocol": protocol,
        "variant": variant,
        "same_fold_pairs": len(tied),
        "pair_delta_rmse": float(np.sqrt(np.mean(residual**2))) if len(residual) else float("nan"),
        "pair_delta_mae": float(np.mean(np.abs(residual))) if len(residual) else float("nan"),
        "pair_delta_bias": float(np.mean(residual)) if len(residual) else float("nan"),
        "pair_sign_accuracy": float(np.mean(sign)) if len(sign) else float("nan"),
    }


def predictions_frame(
    panel: str,
    target: str,
    protocol: str,
    variant: str,
    observed: np.ndarray,
    predicted: np.ndarray,
    groups: np.ndarray,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    result = metadata.reset_index(drop=True).copy()
    result.insert(0, "row_index", np.arange(len(observed)))
    result.insert(0, "group", groups.astype(str))
    result.insert(0, "variant", variant)
    result.insert(0, "protocol", protocol)
    result.insert(0, "target", target)
    result.insert(0, "panel", panel)
    result["observed"] = observed
    result["predicted"] = predicted
    result["residual"] = predicted - observed
    return result


def fold_assignment(folds: tuple[tuple[np.ndarray, np.ndarray], ...], rows: int) -> np.ndarray:
    assignment = np.full(rows, -1, dtype=int)
    for fold, (_train, test) in enumerate(folds):
        assignment[test] = fold
    if np.any(assignment < 0):
        raise ValueError("outer folds do not cover every row exactly once")
    return assignment


def restrict_folds(
    folds: tuple[tuple[np.ndarray, np.ndarray], ...],
    retained: np.ndarray,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Map global folds onto the positional indices of a retained row subset."""
    retained_rows = np.flatnonzero(retained)
    return tuple(
        (
            np.flatnonzero(np.isin(retained_rows, train)),
            np.flatnonzero(np.isin(retained_rows, test)),
        )
        for train, test in folds
    )


def evaluate_fixed_shapes(
    variants: tuple[str, ...],
    weights: np.ndarray,
    target: np.ndarray,
    geometry: rpl.Geometry,
    folds: tuple[tuple[np.ndarray, np.ndarray], ...],
    selections: list[dict[str, Any]],
    relative_variance: np.ndarray,
    pairs: PairIndex,
) -> dict[str, np.ndarray]:
    outputs = {variant: np.full(len(target), np.nan) for variant in variants}
    for fold, (train, test) in enumerate(folds):
        selected = next(row for row in selections if int(row["outer_fold"]) == fold)
        if len(train) != int(selected["train_rows"]) or len(test) != int(selected["test_rows"]):
            raise ValueError(f"fold {fold} no longer matches its frozen selection row counts")
        shape = shape_from_dict(selected["shape"])
        ridge = float(selected["ridge"])
        design, layout = repaired.design_matrix(weights, geometry, shape)
        multipliers = repaired.penalty_multipliers(geometry, layout)
        train_pairs = subset_pairs(pairs, train)
        for variant in variants:
            head = fit_head(
                variant,
                design[train],
                target[train],
                ridge,
                multipliers,
                layout,
                relative_variance[train],
                train_pairs,
            )
            outputs[variant][test] = predict(head, design[test])
    for variant, values in outputs.items():
        if not np.isfinite(values).all():
            raise RuntimeError(f"{variant} produced incomplete OOF predictions")
    return outputs


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def full_fit_wsd80(
    variants: tuple[str, ...],
    weights: np.ndarray,
    target: np.ndarray,
    geometry: rpl.Geometry,
    relative_variance: np.ndarray,
    pairs: PairIndex,
    interior: np.ndarray,
) -> list[dict[str, Any]]:
    selected = load_json(WSD_REFERENCE_DIR / "cells" / "random" / PRIMARY_HASH / "full_fit.json")
    shape = shape_from_dict(selected["shape"])
    ridge = float(selected["ridge"])
    design, layout = repaired.design_matrix(weights, geometry, shape)
    multipliers = repaired.penalty_multipliers(geometry, layout)
    axis = np.linspace(0.0, 1.0, GRID_RESOLUTION)
    phase_0, phase_1 = np.meshgrid(axis, axis, indexing="ij")
    grid_weights = np.stack(
        [
            np.column_stack([1.0 - phase_0.ravel(), phase_0.ravel()]),
            np.column_stack([1.0 - phase_1.ravel(), phase_1.ravel()]),
        ],
        axis=1,
    )
    grid_design, _ = repaired.design_matrix(grid_weights, geometry, shape)
    tied_weights = np.stack([np.column_stack([1.0 - axis, axis]), np.column_stack([1.0 - axis, axis])], axis=1)
    tied_design, _ = repaired.design_matrix(tied_weights, geometry, shape)
    interior_rows = np.flatnonzero(interior)
    observed_best = int(interior_rows[np.argmin(target[interior_rows])])
    observed_p0 = float(weights[observed_best, 0, 1])
    observed_p1 = float(weights[observed_best, 1, 1])
    rows: list[dict[str, Any]] = []
    for variant in variants:
        head = fit_head(variant, design, target, ridge, multipliers, layout, relative_variance, pairs)
        surface = predict(head, grid_design)
        tied = predict(head, tied_design)
        best = int(np.argmin(surface))
        p0 = float(phase_0.ravel()[best])
        p1 = float(phase_1.ravel()[best])
        rows.append(
            {
                "variant": variant,
                "predicted_p0": p0,
                "predicted_p1": p1,
                "observed_best_p0": observed_p0,
                "observed_best_p1": observed_p1,
                "optimum_distance": float(np.hypot(p0 - observed_p0, p1 - observed_p1)),
                "predicted_two_phase_gain": float(tied.min() - surface.min()),
                "gain_error": float(abs((tied.min() - surface.min()) - 0.009594)),
                "predicted_minimum": float(surface.min()),
            }
        )
    return rows


def run_wsd80() -> tuple[list[pd.DataFrame], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    panel, frame, available = wsd_audit.load_metric_panel()
    if PRIMARY_TARGET not in available:
        raise ValueError(f"missing WSD80 primary target {PRIMARY_TARGET}")
    target = frame[PRIMARY_TARGET].to_numpy(dtype=float)
    geometry = rpl.Geometry(
        c0=panel.c0,
        c1=panel.c1,
        phase_0_fraction=wsd80.REALIZED_PHASE_0_FRACTION,
    )
    relative_variance = wsd_relative_variance(panel.weights)
    pairs = wsd_pairs(panel.weights)
    phase_0 = panel.weights[:, 0, 1]
    phase_1 = panel.weights[:, 1, 1]
    aggregate = wsd80.PHASE_0_FRACTION * phase_0 + wsd80.PHASE_1_FRACTION * phase_1
    masks, _best_interior = wsd_audit.subset_masks(panel, target)
    interior = masks["interior"]
    metadata = pd.DataFrame(
        {
            "phase_0_starcoder": phase_0,
            "phase_1_starcoder": phase_1,
            "aggregate": aggregate,
            "evaluation_eligible": interior,
        }
    )
    groups = np.asarray([f"{value:.9f}" for value in aggregate])

    prediction_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for protocol in ("random", "blocked"):
        folds = repaired_wsd.fold_builder(
            protocol, panel.weights, np.arange(len(target)), repaired_wsd.OUTER_SPLITS, repaired_wsd.OUTER_SEED
        )
        selections = load_json(WSD_REFERENCE_DIR / "cells" / protocol / PRIMARY_HASH / "fold_selections.json")
        predictions = evaluate_fixed_shapes(
            WSD_VARIANTS,
            panel.weights,
            target,
            geometry,
            folds,
            selections,
            relative_variance,
            pairs,
        )
        assignment = fold_assignment(folds, len(target))
        for variant, predicted in predictions.items():
            prediction_frames.append(
                predictions_frame(
                    "wsd80", "programming_languages", protocol, variant, target, predicted, groups, metadata
                )
            )
            row = metrics(
                "wsd80",
                "programming_languages",
                protocol,
                variant,
                target[interior],
                predicted[interior],
                restrict_folds(folds, interior),
                np.ones(int(interior.sum()), dtype=bool),
            )
            metric_rows.append(row)
            pair_rows.append(
                pair_metrics("wsd80", "programming_languages", protocol, variant, target, predicted, pairs, assignment)
            )
    optimum_rows = full_fit_wsd80(
        WSD_VARIANTS,
        panel.weights,
        target,
        geometry,
        relative_variance,
        pairs,
        interior,
    )
    return prediction_frames, metric_rows, pair_rows, optimum_rows


def run_300m_target(
    target_name: str,
) -> tuple[list[pd.DataFrame], list[dict[str, Any]], list[dict[str, Any]]]:
    dataset, folds = baseline_300m.prepare_target(OUTPUT_DIR, target_name, baseline_300m.OUTER_SPLITS)
    geometry = baseline_300m.retained_geometry(dataset, dataset.family_index)
    pairs = panel_300m_pairs(dataset.frame, dataset.weights)
    selections = load_json(RPL_300M_DIR / "cells" / target_name / "retained_power_law_repaired" / "fold_selections.json")
    predictions = evaluate_fixed_shapes(
        PANEL_300M_VARIANTS,
        dataset.weights,
        dataset.y,
        geometry,
        folds,
        selections,
        np.ones(dataset.n),
        pairs,
    )
    assignment = fold_assignment(folds, dataset.n)
    tied = np.all(
        np.isclose(dataset.weights[:, 0, :], dataset.weights[:, 1, :], atol=1e-10, rtol=0.0),
        axis=1,
    )
    groups = dataset.frame["phase_correspondence_key"].astype(str).to_numpy()
    metadata = dataset.frame[["run_name", "policy_family", "phase_correspondence_key"]].copy()
    metadata["evaluation_eligible"] = True
    prediction_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for variant, predicted in predictions.items():
        prediction_frames.append(
            predictions_frame("300m", target_name, "correspondence", variant, dataset.y, predicted, groups, metadata)
        )
        metric_rows.append(metrics("300m", target_name, "correspondence", variant, dataset.y, predicted, folds, ~tied))
        pair_rows.append(
            pair_metrics("300m", target_name, "correspondence", variant, dataset.y, predicted, pairs, assignment)
        )
    return prediction_frames, metric_rows, pair_rows


def bootstrap_metric_differences(predictions: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows: list[dict[str, Any]] = []
    keys = ["panel", "target", "protocol"]
    for key, block in predictions.groupby(keys, sort=True):
        block = block.loc[block["evaluation_eligible"]].copy()
        wide = block.pivot(index="row_index", columns="variant", values="predicted")
        observed = block.drop_duplicates("row_index").set_index("row_index")["observed"].loc[wide.index].to_numpy()
        groups = block.drop_duplicates("row_index").set_index("row_index")["group"].loc[wide.index].to_numpy()
        unique = np.unique(groups)
        group_rows = {group: np.flatnonzero(groups == group) for group in unique}
        baseline = wide["huber"].to_numpy()
        for variant in wide.columns:
            if variant == "huber":
                continue
            candidate = wide[variant].to_numpy()
            rmse_differences = np.empty(BOOTSTRAP_SAMPLES)
            mae_differences = np.empty(BOOTSTRAP_SAMPLES)
            for draw in range(BOOTSTRAP_SAMPLES):
                sampled_groups = rng.choice(unique, size=len(unique), replace=True)
                index = np.concatenate([group_rows[group] for group in sampled_groups])
                baseline_residual = baseline[index] - observed[index]
                candidate_residual = candidate[index] - observed[index]
                rmse_differences[draw] = np.sqrt(np.mean(candidate_residual**2)) - np.sqrt(np.mean(baseline_residual**2))
                mae_differences[draw] = np.mean(np.abs(candidate_residual)) - np.mean(np.abs(baseline_residual))
            rows.append(
                {
                    "panel": key[0],
                    "target": key[1],
                    "protocol": key[2],
                    "variant": variant,
                    "comparator": "huber",
                    "rmse_difference": float(
                        np.sqrt(np.mean((candidate - observed) ** 2)) - np.sqrt(np.mean((baseline - observed) ** 2))
                    ),
                    "rmse_difference_ci_low": float(np.quantile(rmse_differences, 0.025)),
                    "rmse_difference_ci_high": float(np.quantile(rmse_differences, 0.975)),
                    "mae_difference": float(
                        np.mean(np.abs(candidate - observed)) - np.mean(np.abs(baseline - observed))
                    ),
                    "mae_difference_ci_low": float(np.quantile(mae_differences, 0.025)),
                    "mae_difference_ci_high": float(np.quantile(mae_differences, 0.975)),
                }
            )
    return pd.DataFrame(rows)


def write_dashboard(metrics_frame: pd.DataFrame, pair_frame: pd.DataFrame, optimum_frame: pd.DataFrame) -> None:
    panels = [
        ("wsd80", "programming_languages", "random", "WSD80 random-fold"),
        ("wsd80", "programming_languages", "blocked", "WSD80 blocked-region"),
        ("300m", "uncheatable", "correspondence", "300M Uncheatable"),
        ("300m", "table9", "correspondence", "300M Table-9"),
    ]
    figure = make_subplots(rows=2, cols=2, subplot_titles=[label for *_rest, label in panels])
    colors = {
        "mse": "#b2182b",
        "huber": "#ef8a62",
        "student_t": "#fddbc7",
        "hetero_student_t": "#a6dba0",
        "hetero_student_t_paired": "#1b7837",
        "student_t_paired": "#1b7837",
    }
    for index, (panel, target, protocol, _label) in enumerate(panels):
        row, column = index // 2 + 1, index % 2 + 1
        selected = metrics_frame.loc[
            metrics_frame["panel"].eq(panel)
            & metrics_frame["target"].eq(target)
            & metrics_frame["protocol"].eq(protocol)
        ]
        figure.add_trace(
            go.Bar(
                x=selected["variant"],
                y=selected["rmse"],
                marker_color=[colors[value] for value in selected["variant"]],
                customdata=np.column_stack(
                    [
                        selected["mae"],
                        selected["spearman"],
                        selected["regret_at_1"],
                        selected["calibration_slope_observed_on_predicted"],
                    ]
                ),
                hovertemplate=(
                    "%{x}<br>RMSE=%{y:.6f}<br>MAE=%{customdata[0]:.6f}<br>Spearman=%{customdata[1]:.4f}"
                    "<br>Regret@1=%{customdata[2]:.6f}<br>calibration slope=%{customdata[3]:.3f}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=row,
            col=column,
        )
    figure.update_layout(
        title="Noise-aware RPL head ablation: lower RMSE is better",
        template="plotly_white",
        height=900,
        width=1450,
        margin={"l": 70, "r": 35, "t": 95, "b": 120},
    )
    figure.update_xaxes(tickangle=-25)
    figure.write_html(
        OUTPUT_DIR / "noise_aware_head_metrics.html",
        include_plotlyjs=True,
        full_html=True,
        config=EXPORT_CONFIG,
    )

    optimum = go.Figure()
    optimum.add_trace(
        go.Scatter(
            x=optimum_frame["predicted_p0"],
            y=optimum_frame["predicted_p1"],
            mode="markers+text",
            text=optimum_frame["variant"],
            textposition="top center",
            marker={"size": 13, "color": [colors[value] for value in optimum_frame["variant"]]},
            customdata=np.column_stack(
                [
                    optimum_frame["optimum_distance"],
                    optimum_frame["predicted_two_phase_gain"],
                    optimum_frame["gain_error"],
                ]
            ),
            hovertemplate=(
                "%{text}<br>p0=%{x:.4f}<br>p1=%{y:.4f}<br>distance=%{customdata[0]:.5f}"
                "<br>predicted gain=%{customdata[1]:.6f}<br>gain error=%{customdata[2]:.6f}<extra></extra>"
            ),
        )
    )
    optimum.add_trace(
        go.Scatter(
            x=[optimum_frame["observed_best_p0"].iloc[0]],
            y=[optimum_frame["observed_best_p1"].iloc[0]],
            mode="markers",
            marker={"size": 18, "symbol": "star", "color": "#17324d"},
            name="observed best",
        )
    )
    optimum.update_layout(
        title="WSD80 full-fit raw optima under a fixed RPL shape",
        xaxis_title="Phase 0 StarCoder weight",
        yaxis_title="Phase 1 StarCoder weight",
        template="plotly_white",
        width=1000,
        height=800,
    )
    optimum.write_html(
        OUTPUT_DIR / "wsd80_noise_aware_optima.html",
        include_plotlyjs=True,
        full_html=True,
        config=EXPORT_CONFIG,
    )

    pair_summary = pair_frame.pivot_table(
        index=["panel", "target", "protocol"], columns="variant", values="pair_delta_rmse"
    )
    pair_summary.to_csv(OUTPUT_DIR / "pair_delta_rmse_pivot.csv")


def write_report(
    metrics_frame: pd.DataFrame,
    pair_frame: pd.DataFrame,
    optimum_frame: pd.DataFrame,
    bootstrap_frame: pd.DataFrame,
) -> None:
    def table(frame: pd.DataFrame, columns: list[str]) -> str:
        return frame[columns].to_markdown(index=False, floatfmt=".6f")

    primary_columns = [
        "panel",
        "target",
        "protocol",
        "variant",
        "rmse",
        "mae",
        "spearman",
        "calibration_slope_observed_on_predicted",
        "regret_at_1",
        "optimism_gt_0p05",
    ]
    pair_columns = ["panel", "target", "protocol", "variant", "same_fold_pairs", "pair_delta_rmse", "pair_sign_accuracy"]
    optimum_columns = [
        "variant",
        "predicted_p0",
        "predicted_p1",
        "optimum_distance",
        "predicted_two_phase_gain",
        "gain_error",
    ]
    bootstrap_columns = [
        "panel",
        "target",
        "protocol",
        "variant",
        "rmse_difference",
        "rmse_difference_ci_low",
        "rmse_difference_ci_high",
    ]

    report = f"""# Noise-aware RPL head ablation

## Protocol

- The RPL equation, nonlinear shape, ridge, folds, and target rows are frozen from the published
  repaired-RPL artifacts.
- The only changed object is the linear-head residual objective.
- Student-t degrees of freedom: `{STUDENT_DF:g}` (fixed, not selected).
- Paired augmentation: `{PAIR_CHANNEL_SHARE:.0%}` absolute-response and
  `{PAIR_CHANNEL_SHARE:.0%}` designed aggregate-matched-difference auxiliary objective.
- WSD80 heteroskedastic relative variance uses only the independently estimated aggregate/contrast
  shape from 188 four-seed calibration groups. Its block intercept is not transferred.
- No heteroskedastic 300M arm is reported because that panel lacks coordinate-resolved variance calibration.
- This is a fixed-shape screen. A positive result must survive nested shape selection; a negative
  result blocks changing the head alone.

## OOF metrics

{table(metrics_frame, primary_columns)}

## Aggregate-matched phase deltas

Only pairs whose two members occur in the same outer test fold are scored.

{table(pair_frame, pair_columns)}

## WSD80 raw optimum

{table(optimum_frame, optimum_columns)}

## Paired cluster bootstrap versus Huber

Negative differences favor the candidate. Clusters are fixed-aggregate fibers on WSD80 and correspondence groups on 300M.

{table(bootstrap_frame, bootstrap_columns)}

## Interpretation

**No noise-aware head is promoted.** The fixed-shape screen supports a narrower claim: changing only the
residual loss at the repaired RPL's published Huber-selected operating point does not repair extrapolation or
policy selection.

- Plain Student-t improves WSD80 random-fold interior RMSE by `0.000158` BPB versus Huber, but the
  aggregate-cluster 95% interval is `[-0.000683, +0.000314]`. Its MAE gain is clearer, but blocked-region
  RMSE worsens by `0.006644` BPB with interval `[+0.004339, +0.009114]`, Regret@1 rises from `0.001900`
  to `0.025108`, and optimism errors above 0.05 BPB rise from 9 to 21.
- A preregistered fixed-scale, two-start diagnostic independently confirmed that the Student-t blocked failure
  is not an optimizer basin: every solve converged, the two starts agreed within `3.1e-14` objective units,
  blocked RMSE remained `0.032428`, and Regret@1 remained `0.025108`. See
  `../noise_aware_rpl_student_t_diagnostic_20260815/report.md`.
- Neither 300M target improves in RMSE beyond correspondence-cluster bootstrap uncertainty. Paired fitting
  modestly improves some 300M phase-delta diagnostics, but worsens or preserves end-to-end RMSE and Regret@1.
- The WSD80 paired arm is not decision-valid: 255 contrasts reuse a small number of tied fiber anchors, so the
  diagonal pair-variance approximation double-counts shared anchor noise and concentrates leverage. It is
  retained as a failed diagnostic, not evidence against paired information generally.
- The heteroskedastic arm transfers seed-noise variance into a regime where blocked residuals are dominated by
  structural misspecification. Its near-equivalence to homoskedastic Student-t therefore does not falsify a
  better variance model; it shows that this particular variance instrument cannot repair the surface.

The strongest positive lead is instead **less robustness**: MSE beats Huber on the frozen blocked shape by
`0.004986` BPB RMSE with interval `[-0.008637, -0.002832]`, while also improving Spearman, Regret@1, and
optimism counts. Because the shape and ridge were selected under Huber and only one spatial partition was used,
this is not a promotion. The next high-value local test is fully nested MSE shape/ridge selection over multiple
blocked-region partitions. Quantile or rank loss should not replace mean-BPB regression: the deployment objective
is cardinal expected BPB, and this experiment shows that downweighting large residuals can directly remove the
rows needed to anchor extrapolation.

The independent review used Claude Code through the verified OAuth subscription with read-only repository access.

## Reproduce

```bash
uv run {Path(__file__).relative_to(REPO_ROOT)}
```
"""
    (OUTPUT_DIR / "report.md").write_text(report)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT_DIR / "protocol.json", protocol_payload())

    prediction_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    wsd_predictions, wsd_metrics, wsd_pairs_rows, optimum_rows = run_wsd80()
    prediction_frames.extend(wsd_predictions)
    metric_rows.extend(wsd_metrics)
    pair_rows.extend(wsd_pairs_rows)
    for target in ("uncheatable", "table9"):
        target_predictions, target_metrics, target_pairs = run_300m_target(target)
        prediction_frames.extend(target_predictions)
        metric_rows.extend(target_metrics)
        pair_rows.extend(target_pairs)

    predictions = pd.concat(prediction_frames, ignore_index=True, sort=False)
    metrics_frame = pd.DataFrame(metric_rows)
    pair_frame = pd.DataFrame(pair_rows)
    optimum_frame = pd.DataFrame(optimum_rows)
    bootstrap_frame = bootstrap_metric_differences(predictions)
    predictions.to_csv(OUTPUT_DIR / "oof_predictions.csv", index=False)
    metrics_frame.to_csv(OUTPUT_DIR / "metrics.csv", index=False)
    pair_frame.to_csv(OUTPUT_DIR / "pair_metrics.csv", index=False)
    optimum_frame.to_csv(OUTPUT_DIR / "wsd80_full_fit_optima.csv", index=False)
    bootstrap_frame.to_csv(OUTPUT_DIR / "bootstrap_vs_huber.csv", index=False)
    write_dashboard(metrics_frame, pair_frame, optimum_frame)
    write_report(metrics_frame, pair_frame, optimum_frame, bootstrap_frame)
    print(metrics_frame.to_string(index=False))
    print(f"\nWrote {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
