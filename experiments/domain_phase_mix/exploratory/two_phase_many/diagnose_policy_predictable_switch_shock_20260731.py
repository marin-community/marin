# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
#
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy",
#   "fsspec",
#   "gcsfs",
#   "numpy",
#   "pandas",
#   "plotly",
#   "pyarrow",
#   "scikit-learn",
#   "scipy",
#   "tabulate",
#   "wandb",
# ]
# ///
"""Identify a policy-computable phase-switch state from logged optimizer shock.

SUR-069 established that the asymmetric-minus-tied optimizer shock predicts a
common residual left by the frozen SUR-068 phase-relaxation law. This script
tests whether that shock can itself be predicted from policy inputs. It keeps
the structural target at the phase boundary: no final BPB outcome is used to
select features, nonlinear shapes, or ridge values.

The proposed cross-phase state is counterfactual late unfamiliarity. For each
predeclared family, it compares the phase-0 materialized exposure expected by a
tied late sample with the exposure expected by the actual phase-1 sample. The
state is zero for tied schedules, has units of materialized epochs, and adds no
per-bucket fitted parameter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import diagnose_switch_gradient_shock_20260731 as switch_shock
import fit_trajectory_identified_acquisition_forgetting_20260731 as trajectory
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import spearmanr
from sklearn.cluster import KMeans

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "policy_predictable_switch_shock_20260731"
PAIR_SHOCKS_PATH = SCRIPT_DIR / "reference_outputs" / "switch_gradient_shock_20260731" / "pair_shocks.csv"

CANDIDATE_ID = "WSD80-SUR-070"
OUTER_FOLDS = 5
INNER_FOLDS = 4
SPLIT_SEED = 20_260_731
BOOTSTRAP_SEED = 20_260_732
BOOTSTRAP_SAMPLES = 5_000
RIDGE_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
TRANSFER_STEPS = (19_000, 20_000, 21_000, 22_000, switch_shock.FINAL_STEP)

SHOCK_TARGETS = {
    "gradient_log_jump": "Log change in total gradient norm across the phase boundary",
    "training_loss_jump": "Change in training loss across the phase boundary",
}

FEATURE_BLOCKS = {
    "family_shift": ("shift",),
    "late_static": ("shift", "late_repetition"),
    "cross_phase": ("shift", "unfamiliarity"),
    "full": ("shift", "late_repetition", "unfamiliarity"),
}


@dataclass(frozen=True)
class FitResult:
    """One no-intercept ridge fit in physical feature units."""

    coefficients: np.ndarray
    effective_df: float
    condition_number: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preregister", "evaluate"), required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def protocol_payload() -> dict[str, object]:
    return {
        "candidate_id": CANDIDATE_ID,
        "title": "Counterfactual late-unfamiliarity switch map",
        "scope": "development identification diagnostic; not an endpoint surrogate",
        "data_use": {
            "exposed_before_freeze": [
                "SUR-069 observed boundary shocks and their correlations with SUR-068 residuals",
                "exploratory family directional-feature ranks against both shock targets",
                "SUR-068 residual trajectories, including step 22000 and final summaries",
            ],
            "not_used_for_structure_or_hyperparameters": [
                "final Uncheatable endpoint deltas",
                "Table-9 endpoint deltas",
                "WSD80 outcomes",
            ],
            "interpretation": (
                "All results are development evidence. The feature family was inspired by exposed shock outcomes; "
                "post-switch transfer was previously visible at the observed-shock level and is not confirmatory."
            ),
        },
        "policy_state": {
            "aggregate": "wbar = beta0*w0 + beta1*w1, represented by the exact tied counterpart",
            "phase0_exposure": "e0_i = c0_i*w0_i materialized epochs",
            "phase1_exposure": "e1_i = c1_i*w1_i materialized epochs",
            "family_shift": "M_f = sum_{i in f}(w1_i - wbar_i); omit one redundant family",
            "late_repetition": "R_f = sum_{i in f}(w1_i*e1_i - wbar_i*ebar1_i), materialized epochs",
            "late_unfamiliarity": "U_f = sum_{i in f}(wbar_i*ebar0_i - w1_i*e0_i), materialized epochs",
            "tied_invariant": "M_f = R_f = U_f = 0 when w0 = w1 = wbar",
        },
        "mechanism_tests": {
            "family_shift": "Static phase-1 family composition only",
            "late_static": "Family composition plus within-phase-1 repetition intensity",
            "cross_phase": "Family composition plus prior phase-0 familiarity of phase-1 samples",
            "full": "Late static state plus prior phase-0 familiarity",
        },
        "estimator": {
            "intercept": False,
            "centering": False,
            "scaling": "training-fold RMS only; zero policy state remains zero",
            "ridge_grid": RIDGE_GRID,
            "ridge_selection": "minimum inner mixture-blocked RMSE",
            "outer_folds": OUTER_FOLDS,
            "inner_folds": INNER_FOLDS,
            "blocking": "KMeans over concatenated physical phase-0 and phase-1 weights",
            "split_seed": SPLIT_SEED,
        },
        "primary_target": "gradient_log_jump",
        "secondary_target": "training_loss_jump",
        "uncertainty": {
            "paired_bootstrap_samples": BOOTSTRAP_SAMPLES,
            "paired_bootstrap_seed": BOOTSTRAP_SEED,
            "outer_fold_win_requirement": 4,
        },
        "transfer": {
            "predictor": "OOF full-model predicted gradient shock",
            "outcome": "SUR-068 common residual across seven smooth Uncheatable components",
            "steps": TRANSFER_STEPS,
            "fitted_transfer_amplitude": False,
            "purpose": "Test whether the policy state predicts independent post-switch target dynamics",
        },
        "gates": {
            "gradient_full_mean_improvement_min": 0.15,
            "gradient_full_spearman_min": 0.50,
            "gradient_full_vs_late_static_bootstrap_upper_max": 0.0,
            "gradient_full_vs_late_static_fold_wins_min": 4,
            "gradient_positive_unfamiliarity_families_min": 2,
            "gradient_unfamiliarity_sign_folds_min": 4,
            "training_full_mean_improvement_min": 0.25,
            "training_full_spearman_min": 0.75,
            "training_full_vs_late_static_bootstrap_upper_max": 0.0,
            "training_full_vs_late_static_fold_wins_min": 4,
            "transfer_step19000_spearman_min": 0.20,
            "transfer_step20000_spearman_min": 0.15,
        },
        "forbidden_repairs": [
            "intercept or output calibration",
            "per-bucket fitted shock coefficients",
            "outcome-selected family partition",
            "novelty temperature or saturation grid",
            "endpoint-fitted transfer amplitude or decay rate",
            "feature additions after inspecting this evaluation",
        ],
        "decision_boundary": (
            "A pass licenses a separately frozen bounded temporal-transition test. It does not license an endpoint "
            "correction, aggregate-spine change, or full-surrogate promotion."
        ),
    }


def wrapped_protocol() -> dict[str, object]:
    payload = protocol_payload()
    digest = hashlib.sha256(canonical_json(payload).encode()).hexdigest()
    return {"protocol_sha256": digest, "protocol": payload}


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def freeze_protocol(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    expected = wrapped_protocol()
    path = output_dir / "protocol.json"
    if path.exists():
        observed = json.loads(path.read_text())
        if canonical_json(observed) != canonical_json(expected):
            raise RuntimeError(f"Existing protocol differs from current code: {path}")
    else:
        write_json(path, expected)
    print(expected["protocol_sha256"])


def require_frozen_protocol(output_dir: Path) -> dict[str, object]:
    path = output_dir / "protocol.json"
    if not path.exists():
        raise RuntimeError("Run --mode preregister before evaluation")
    observed = json.loads(path.read_text())
    expected = wrapped_protocol()
    if canonical_json(observed) != canonical_json(expected):
        raise RuntimeError("Frozen protocol does not match the evaluation code")
    return expected


def mixture_blocks(coordinates: np.ndarray, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    labels = KMeans(n_clusters=n_splits, n_init=50, random_state=seed).fit_predict(coordinates)
    folds = []
    for label in sorted(np.unique(labels)):
        test = labels == label
        if not test.any() or test.all():
            continue
        folds.append((~test, test))
    if len(folds) != n_splits:
        raise RuntimeError(f"Expected {n_splits} blocked folds, found {len(folds)}")
    return folds


def feature_frame(data: trajectory.PairData) -> pd.DataFrame:
    w0 = np.asarray(data.asymmetric_weights[:, 0, :], dtype=float)
    w1 = np.asarray(data.asymmetric_weights[:, 1, :], dtype=float)
    wbar0 = np.asarray(data.tied_weights[:, 0, :], dtype=float)
    wbar1 = np.asarray(data.tied_weights[:, 1, :], dtype=float)
    if not np.allclose(wbar0, wbar1, atol=1e-12):
        raise RuntimeError("Tied counterparts are not physically tied")
    phase_fraction = float(np.median(data.c0 / np.maximum(data.c0 + data.c1, 1e-12)))
    reconstructed = phase_fraction * w0 + (1.0 - phase_fraction) * w1
    if not np.allclose(reconstructed, wbar0, atol=2e-9):
        raise RuntimeError("Asymmetric policies do not preserve the tied aggregate")

    e0 = w0 * data.c0[None, :]
    e1 = w1 * data.c1[None, :]
    ebar0 = wbar0 * data.c0[None, :]
    ebar1 = wbar1 * data.c1[None, :]
    rows: dict[str, object] = {"pair_id": data.keys}
    for family_index, (family, members) in enumerate(zip(data.family_names, data.family_members, strict=True)):
        rows[f"shift__{family}"] = np.sum(w1[:, members] - wbar1[:, members], axis=1)
        rows[f"late_repetition__{family}"] = np.sum(
            w1[:, members] * e1[:, members] - wbar1[:, members] * ebar1[:, members],
            axis=1,
        )
        rows[f"unfamiliarity__{family}"] = np.sum(
            wbar1[:, members] * ebar0[:, members] - w1[:, members] * e0[:, members],
            axis=1,
        )
        if family_index + 1 == len(data.family_names):
            continue
    frame = pd.DataFrame(rows)
    if frame.isna().any().any():
        raise RuntimeError("Policy feature frame contains missing values")
    return frame


def feature_columns(frame: pd.DataFrame, block: str, family_names: tuple[str, ...]) -> list[str]:
    mechanisms = FEATURE_BLOCKS[block]
    columns: list[str] = []
    if "shift" in mechanisms:
        columns.extend(f"shift__{family}" for family in family_names[:-1])
    if "late_repetition" in mechanisms:
        columns.extend(f"late_repetition__{family}" for family in family_names)
    if "unfamiliarity" in mechanisms:
        columns.extend(f"unfamiliarity__{family}" for family in family_names)
    missing = set(columns) - set(frame.columns)
    if missing:
        raise RuntimeError(f"Missing feature columns for {block}: {sorted(missing)}")
    return columns


def fit_ridge(features: np.ndarray, target: np.ndarray, alpha: float) -> FitResult:
    scale = np.sqrt(np.mean(np.square(features), axis=0))
    scale = np.where(scale > 1e-12, scale, 1.0)
    scaled = features / scale
    gram = scaled.T @ scaled
    system = gram + alpha * np.eye(scaled.shape[1])
    scaled_coefficients = np.linalg.solve(system, scaled.T @ target)
    coefficients = scaled_coefficients / scale
    effective_df = float(np.trace(np.linalg.solve(system, gram)))
    condition_number = float(np.linalg.cond(system))
    return FitResult(coefficients, effective_df, condition_number)


def rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(predicted) - np.asarray(observed)))))


def select_alpha(
    features: np.ndarray,
    target: np.ndarray,
    coordinates: np.ndarray,
    seed: int,
) -> float:
    folds = mixture_blocks(coordinates, INNER_FOLDS, seed)
    losses = []
    for alpha in RIDGE_GRID:
        prediction = np.full(len(target), np.nan, dtype=float)
        for train, test in folds:
            fit = fit_ridge(features[train], target[train], alpha)
            prediction[test] = features[test] @ fit.coefficients
        if not np.isfinite(prediction).all():
            raise RuntimeError("Nested ridge selection produced incomplete predictions")
        losses.append(rmse(target, prediction))
    return float(RIDGE_GRID[int(np.argmin(losses))])


def calibration_slope(observed: np.ndarray, predicted: np.ndarray) -> float:
    denominator = float(np.dot(predicted, predicted))
    if denominator <= 1e-15:
        return float("nan")
    return float(np.dot(predicted, observed) / denominator)


def paired_bootstrap_rmse_difference(
    observed: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
    seed: int,
) -> tuple[float, float, float]:
    generator = np.random.default_rng(seed)
    differences = np.empty(BOOTSTRAP_SAMPLES, dtype=float)
    for draw in range(BOOTSTRAP_SAMPLES):
        sample = generator.integers(0, len(observed), len(observed))
        differences[draw] = rmse(observed[sample], candidate[sample]) - rmse(observed[sample], baseline[sample])
    return (
        float(np.mean(differences)),
        float(np.quantile(differences, 0.025)),
        float(np.quantile(differences, 0.975)),
    )


def blocked_oof(
    feature_values: pd.DataFrame,
    coordinates: np.ndarray,
    target_name: str,
    family_names: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outer = mixture_blocks(coordinates, OUTER_FOLDS, SPLIT_SEED)
    predictions = pd.DataFrame({"pair_id": feature_values["pair_id"], "observed": feature_values[target_name]})
    predictions["outer_fold"] = -1
    parameters: list[dict[str, object]] = []
    fold_metrics: list[dict[str, object]] = []
    for block, columns_by_block in (
        (name, feature_columns(feature_values, name, family_names)) for name in FEATURE_BLOCKS
    ):
        prediction = np.full(len(feature_values), np.nan, dtype=float)
        for fold_index, (train, test) in enumerate(outer):
            alpha = select_alpha(
                feature_values.loc[train, columns_by_block].to_numpy(float),
                feature_values.loc[train, target_name].to_numpy(float),
                coordinates[train],
                SPLIT_SEED + 100 * fold_index + len(columns_by_block),
            )
            fit = fit_ridge(
                feature_values.loc[train, columns_by_block].to_numpy(float),
                feature_values.loc[train, target_name].to_numpy(float),
                alpha,
            )
            local_prediction = feature_values.loc[test, columns_by_block].to_numpy(float) @ fit.coefficients
            prediction[test] = local_prediction
            predictions.loc[test, "outer_fold"] = fold_index
            fold_metrics.append(
                {
                    "target": target_name,
                    "block": block,
                    "outer_fold": fold_index,
                    "rows": int(test.sum()),
                    "rmse": rmse(feature_values.loc[test, target_name].to_numpy(float), local_prediction),
                    "ridge": alpha,
                    "effective_df": fit.effective_df,
                    "condition_number": fit.condition_number,
                }
            )
            for column, coefficient in zip(columns_by_block, fit.coefficients, strict=True):
                parameters.append(
                    {
                        "target": target_name,
                        "block": block,
                        "outer_fold": fold_index,
                        "feature": column,
                        "coefficient": float(coefficient),
                        "ridge": alpha,
                    }
                )
        if not np.isfinite(prediction).all():
            raise RuntimeError(f"Incomplete OOF predictions for {target_name}/{block}")
        predictions[f"predicted__{block}"] = prediction

    mean_prediction = np.full(len(feature_values), np.nan, dtype=float)
    for train, test in outer:
        mean_prediction[test] = float(feature_values.loc[train, target_name].mean())
    predictions["predicted__mean"] = mean_prediction
    predictions["predicted__zero"] = 0.0
    return predictions, pd.DataFrame(parameters), pd.DataFrame(fold_metrics)


def aggregate_metrics(predictions: pd.DataFrame, target_name: str) -> pd.DataFrame:
    observed = predictions["observed"].to_numpy(float)
    mean_rmse = rmse(observed, predictions["predicted__mean"].to_numpy(float))
    rows = []
    for block in (*FEATURE_BLOCKS, "mean", "zero"):
        predicted = predictions[f"predicted__{block}"].to_numpy(float)
        rho, p_value = spearmanr(predicted, observed)
        candidate_rmse = rmse(observed, predicted)
        rows.append(
            {
                "target": target_name,
                "block": block,
                "rows": len(observed),
                "rmse": candidate_rmse,
                "mean_improvement": 1.0 - candidate_rmse / mean_rmse,
                "spearman": float(rho),
                "spearman_p": float(p_value),
                "calibration_slope": calibration_slope(observed, predicted),
                "bias": float(np.mean(predicted - observed)),
                "amplitude_ratio": float(np.std(predicted) / np.std(observed)),
            }
        )
    return pd.DataFrame(rows)


def incremental_metrics(
    predictions: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    target_name: str,
) -> dict[str, float | int]:
    observed = predictions["observed"].to_numpy(float)
    full = predictions["predicted__full"].to_numpy(float)
    late_static = predictions["predicted__late_static"].to_numpy(float)
    mean, low, high = paired_bootstrap_rmse_difference(
        observed,
        full,
        late_static,
        BOOTSTRAP_SEED + (0 if target_name == "gradient_log_jump" else 1),
    )
    pivot = fold_metrics.loc[fold_metrics["target"].eq(target_name)].pivot(
        index="outer_fold", columns="block", values="rmse"
    )
    fold_wins = int((pivot["full"] < pivot["late_static"]).sum())
    return {
        "target": target_name,
        "full_minus_late_static_rmse_bootstrap_mean": mean,
        "full_minus_late_static_rmse_bootstrap_low": low,
        "full_minus_late_static_rmse_bootstrap_high": high,
        "full_vs_late_static_fold_wins": fold_wins,
    }


def familiarity_sign_metrics(parameters: pd.DataFrame, target_name: str) -> pd.DataFrame:
    block = parameters.loc[
        parameters["target"].eq(target_name)
        & parameters["block"].eq("full")
        & parameters["feature"].str.startswith("unfamiliarity__")
    ].copy()
    rows = []
    for feature, local in block.groupby("feature", sort=True):
        rows.append(
            {
                "target": target_name,
                "feature": feature,
                "median_coefficient": float(local["coefficient"].median()),
                "positive_folds": int((local["coefficient"] > 0.0).sum()),
                "folds": len(local),
            }
        )
    return pd.DataFrame(rows)


def transfer_metrics(
    gradient_predictions: pd.DataFrame,
    pair_shocks: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    residuals = switch_shock.relaxation_residuals()
    common = residuals.groupby(["pair_id", "global_step"], as_index=False).agg(
        common_residual=("residual", "mean"),
        component_residual_sd=("residual", "std"),
    )
    predictor = gradient_predictions[["pair_id", "predicted__full", "outer_fold"]].rename(
        columns={"predicted__full": "predicted_gradient_shock"}
    )
    joined = common.merge(predictor, on="pair_id", how="inner", validate="many_to_one").merge(
        pair_shocks[["pair_id", "gradient_log_jump"]],
        on="pair_id",
        how="inner",
        validate="many_to_one",
    )
    joined = joined.loc[joined["global_step"].isin(TRANSFER_STEPS)].copy()
    rows = []
    for step, block in joined.groupby("global_step", sort=True):
        predicted_rho, predicted_p = spearmanr(block["predicted_gradient_shock"], block["common_residual"])
        observed_rho, observed_p = spearmanr(block["gradient_log_jump"], block["common_residual"])
        rows.append(
            {
                "global_step": int(step),
                "pairs": len(block),
                "predicted_shock_spearman": float(predicted_rho),
                "predicted_shock_p": float(predicted_p),
                "observed_shock_spearman": float(observed_rho),
                "observed_shock_p": float(observed_p),
                "rank_retention": float(predicted_rho / observed_rho) if abs(observed_rho) > 1e-12 else float("nan"),
            }
        )
    return pd.DataFrame(rows), joined


def decide(
    metrics: pd.DataFrame,
    incremental: pd.DataFrame,
    signs: pd.DataFrame,
    transfer: pd.DataFrame,
) -> dict[str, object]:
    indexed = metrics.set_index(["target", "block"])
    incremental_indexed = incremental.set_index("target")
    transfer_indexed = transfer.set_index("global_step")
    gradient_signs = signs.loc[signs["target"].eq("gradient_log_jump")]
    stable_positive = int(((gradient_signs["median_coefficient"] > 0.0) & (gradient_signs["positive_folds"] >= 4)).sum())
    checks = {
        "gradient_full_mean_improvement": float(indexed.loc[("gradient_log_jump", "full"), "mean_improvement"]) >= 0.15,
        "gradient_full_spearman": float(indexed.loc[("gradient_log_jump", "full"), "spearman"]) >= 0.50,
        "gradient_increment_bootstrap": (
            float(incremental_indexed.loc["gradient_log_jump", "full_minus_late_static_rmse_bootstrap_high"]) < 0.0
        ),
        "gradient_increment_fold_wins": (
            int(incremental_indexed.loc["gradient_log_jump", "full_vs_late_static_fold_wins"]) >= 4
        ),
        "gradient_unfamiliarity_signs": stable_positive >= 2,
        "training_full_mean_improvement": float(indexed.loc[("training_loss_jump", "full"), "mean_improvement"]) >= 0.25,
        "training_full_spearman": float(indexed.loc[("training_loss_jump", "full"), "spearman"]) >= 0.75,
        "training_increment_bootstrap": (
            float(incremental_indexed.loc["training_loss_jump", "full_minus_late_static_rmse_bootstrap_high"]) < 0.0
        ),
        "training_increment_fold_wins": (
            int(incremental_indexed.loc["training_loss_jump", "full_vs_late_static_fold_wins"]) >= 4
        ),
        "transfer_step19000": float(transfer_indexed.loc[19_000, "predicted_shock_spearman"]) >= 0.20,
        "transfer_step20000": float(transfer_indexed.loc[20_000, "predicted_shock_spearman"]) >= 0.15,
    }
    passed = all(checks.values())
    return {
        "candidate_id": CANDIDATE_ID,
        "passed": passed,
        "decision": (
            "PASS: policy-computable switch state licensed for a frozen transition test"
            if passed
            else "FAIL: policy-computable switch state not identified beyond late-static controls"
        ),
        "checks": checks,
        "stable_positive_gradient_unfamiliarity_families": stable_positive,
        "scope": "diagnostic_only_not_endpoint_surrogate",
    }


def render_plot(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    transfer: pd.DataFrame,
    path: Path,
) -> None:
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Gradient shock: blocked OOF",
            "Training-loss shock: blocked OOF",
            "Nested mechanism comparison",
            "Transfer to smooth-target relaxation residual",
        ),
    )
    for column, target, row, col in (
        ("gradient_log_jump", "gradient_log_jump", 1, 1),
        ("training_loss_jump", "training_loss_jump", 1, 2),
    ):
        block = predictions.loc[predictions["target"].eq(target)]
        figure.add_trace(
            go.Scatter(
                x=block[column],
                y=block["predicted__full"],
                mode="markers",
                marker={
                    "color": block["outer_fold"],
                    "colorscale": "RdYlGn_r",
                    "size": 7,
                    "showscale": row == 1 and col == 2,
                    "colorbar": {"title": "fold"},
                },
                text=block["pair_id"],
                name=target,
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        limits = [
            float(min(block[column].min(), block["predicted__full"].min())),
            float(max(block[column].max(), block["predicted__full"].max())),
        ]
        figure.add_trace(
            go.Scatter(x=limits, y=limits, mode="lines", line={"dash": "dash", "color": "#718096"}, showlegend=False),
            row=row,
            col=col,
        )

    candidate_metrics = metrics.loc[metrics["block"].isin(FEATURE_BLOCKS)].copy()
    for target, color in (("gradient_log_jump", "#1f6f8b"), ("training_loss_jump", "#d95f0e")):
        block = candidate_metrics.loc[candidate_metrics["target"].eq(target)]
        figure.add_trace(
            go.Bar(x=block["block"], y=block["mean_improvement"], name=target, marker_color=color),
            row=2,
            col=1,
        )
    figure.add_trace(
        go.Scatter(
            x=transfer["global_step"],
            y=transfer["observed_shock_spearman"],
            mode="lines+markers",
            name="observed shock",
            line={"color": "#d95f0e"},
        ),
        row=2,
        col=2,
    )
    figure.add_trace(
        go.Scatter(
            x=transfer["global_step"],
            y=transfer["predicted_shock_spearman"],
            mode="lines+markers",
            name="policy-predicted shock",
            line={"color": "#1f6f8b"},
        ),
        row=2,
        col=2,
    )
    figure.update_xaxes(title_text="Observed shock", row=1, col=1)
    figure.update_xaxes(title_text="Observed shock", row=1, col=2)
    figure.update_yaxes(title_text="Predicted shock", row=1, col=1)
    figure.update_yaxes(title_text="Predicted shock", row=1, col=2)
    figure.update_yaxes(title_text="RMSE improvement over train-fold mean", row=2, col=1)
    figure.update_xaxes(title_text="Global step", row=2, col=2)
    figure.update_yaxes(title_text="Spearman with common residual", row=2, col=2)
    figure.update_layout(
        title="Counterfactual late familiarity as a policy-computable switch state",
        template="plotly_white",
        barmode="group",
        height=900,
    )
    figure.write_html(path, include_plotlyjs="cdn")


def render_report(
    protocol: dict[str, object],
    metrics: pd.DataFrame,
    incremental: pd.DataFrame,
    signs: pd.DataFrame,
    transfer: pd.DataFrame,
    decision: dict[str, object],
    path: Path,
) -> None:
    path.write_text(
        "\n".join(
            [
                "# Counterfactual late-unfamiliarity switch map",
                "",
                f"**Decision: {decision['decision']}**",
                "",
                f"Frozen protocol: `{protocol['protocol_sha256']}`.",
                "",
                "## Mechanism",
                "",
                "For each predeclared family, the state compares the prior phase-0 materialized exposure "
                "expected under the tied late distribution with the exposure expected under the actual "
                "phase-1 distribution. Positive unfamiliarity means late training samples buckets that "
                "received less phase-0 exposure than their tied counterfactual. The state is exactly zero "
                "for tied policies and introduces no fitted bucket parameter.",
                "",
                "The `late_static` baseline contains family mass shift and phase-1 repetition but no "
                "phase-0 state. The cross-phase mechanism is identified only if adding unfamiliarity "
                "improves that baseline under blocked OOF.",
                "",
                "## Blocked OOF metrics",
                "",
                metrics.to_markdown(index=False),
                "",
                "## Incremental cross-phase evidence",
                "",
                incremental.to_markdown(index=False),
                "",
                "## Unfamiliarity coefficient stability",
                "",
                signs.to_markdown(index=False),
                "",
                "## Independent post-switch transfer",
                "",
                transfer.to_markdown(index=False),
                "",
                "## Interpretation boundary",
                "",
                "This is exposed development evidence, not confirmation. A pass licenses only a frozen "
                "bounded transition test. It does not license an endpoint correction or a full surrogate.",
                "",
            ]
        )
    )


def evaluate(output_dir: Path) -> None:
    protocol = require_frozen_protocol(output_dir)
    data = trajectory.load_pair_data()
    features = feature_frame(data)
    pair_shocks = pd.read_csv(PAIR_SHOCKS_PATH)
    pair_shocks["pair_id"] = pair_shocks["pair_id"].astype(str)
    features["pair_id"] = features["pair_id"].astype(str)
    values = features.merge(
        pair_shocks[["pair_id", *SHOCK_TARGETS]],
        on="pair_id",
        how="inner",
        validate="one_to_one",
    )
    if len(values) < 180:
        raise RuntimeError(f"Only {len(values)} exact pairs have policy features and switch shock")
    key_to_index = {key: index for index, key in enumerate(data.keys)}
    row_indices = np.asarray([key_to_index[key] for key in values["pair_id"]], dtype=int)
    coordinates = np.column_stack(
        [
            data.asymmetric_weights[row_indices, 0, :],
            data.asymmetric_weights[row_indices, 1, :],
        ]
    )

    prediction_blocks = []
    parameter_blocks = []
    fold_metric_blocks = []
    metric_blocks = []
    incremental_rows = []
    sign_blocks = []
    for target_name in SHOCK_TARGETS:
        predictions, parameters, fold_metrics = blocked_oof(values, coordinates, target_name, data.family_names)
        predictions["target"] = target_name
        predictions[target_name] = predictions["observed"]
        prediction_blocks.append(predictions)
        parameter_blocks.append(parameters)
        fold_metric_blocks.append(fold_metrics)
        metric_blocks.append(aggregate_metrics(predictions, target_name))
        incremental_rows.append(incremental_metrics(predictions, fold_metrics, target_name))
        sign_blocks.append(familiarity_sign_metrics(parameters, target_name))

    predictions = pd.concat(prediction_blocks, ignore_index=True)
    parameters = pd.concat(parameter_blocks, ignore_index=True)
    fold_metrics = pd.concat(fold_metric_blocks, ignore_index=True)
    metrics = pd.concat(metric_blocks, ignore_index=True)
    incremental = pd.DataFrame(incremental_rows)
    signs = pd.concat(sign_blocks, ignore_index=True)
    gradient_predictions = predictions.loc[predictions["target"].eq("gradient_log_jump")].copy()
    transfer, transfer_predictions = transfer_metrics(gradient_predictions, pair_shocks)
    decision = decide(metrics, incremental, signs, transfer)

    features.to_csv(output_dir / "policy_features.csv", index=False)
    predictions.to_csv(output_dir / "oof_predictions.csv", index=False)
    parameters.to_csv(output_dir / "fold_parameters.csv", index=False)
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    incremental.to_csv(output_dir / "incremental_metrics.csv", index=False)
    signs.to_csv(output_dir / "coefficient_signs.csv", index=False)
    transfer.to_csv(output_dir / "transfer_metrics.csv", index=False)
    transfer_predictions.to_csv(output_dir / "transfer_predictions.csv", index=False)
    write_json(output_dir / "decision.json", decision)
    render_plot(predictions, metrics, transfer, output_dir / "policy_predictable_switch_shock.html")
    render_report(protocol, metrics, incremental, signs, transfer, decision, output_dir / "report.md")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "preregister":
        freeze_protocol(args.output_dir)
        return
    evaluate(args.output_dir)


if __name__ == "__main__":
    main()
