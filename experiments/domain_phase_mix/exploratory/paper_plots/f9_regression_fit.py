# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["kaleido==0.2.1", "numpy", "pandas", "plotly", "scikit-learn"]
# ///
"""Render Olmix-style regression-fit diagnostics for the 60M subset study."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import KFold

from experiments.domain_phase_mix.exploratory.paper_plots.paper_plot_style import (
    PAPER_MUTED,
    configure_static_layout,
    configure_interactive_layout,
    method_color,
    write_static_images,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    benchmark_grp_power_family_penalty_no_l2_raw_subset_optima as no_l2_subset,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.benchmark_grp_power_family_penalty_no_l2_retune import (
    CV_SEED,
    VARIANT_NAME,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.dataset_metadata import (
    load_two_phase_many_candidate_summary_spec,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_followup import (
    load_generic_family_packet,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_penalty_calibration import (
    build_penalty_calibration_surrogate,
)
from experiments.domain_phase_mix.static_batch_selection import (
    build_dataset_spec_from_frame,
    retrospective_generic_selection,
)
from experiments.domain_phase_mix.two_phase_many_genericfamily_observed_only_trustblend_subset_optima import (
    OBJECTIVE_METRIC,
)
from experiments.domain_phase_mix.two_phase_many_genericfamily_penalty_raw_optima_baselines import (
    genericfamily_penalty_raw_optimum_summary,
)
from experiments.domain_phase_mix.two_phase_many_genericfamily_retuned_subset_optima import CSV_PATH, _subset_packet
from experiments.domain_phase_mix.two_phase_many_olmix_loglinear_sl_verb import fit_olmix_loglinear_model

SCRIPT_DIR = Path(__file__).resolve().parent
IMG_DIR = SCRIPT_DIR / "img"
OUTPUT_STEM = IMG_DIR / "f9_regression_fit"
PAPER_OUTPUT_STEM = IMG_DIR / "f9_regression_fit_oof_pearson"
OUTPUT_POINTS_CSV = IMG_DIR / "f9_regression_fit_points.csv"
OUTPUT_REPLICATES_CSV = IMG_DIR / "f9_regression_fit_replicates.csv"
OUTPUT_REPLICATES_PARTIAL_CSV = IMG_DIR / "f9_regression_fit_replicates.partial.csv"
OUTPUT_BANDS_CSV = IMG_DIR / "f9_regression_fit_bands.csv"
OUTPUT_SUMMARY_JSON = IMG_DIR / "f9_regression_fit_summary.json"
SUBSET_SIZES = (20, 40, 60, 80, 100, 140, 180, 220, 242)
KFOLD_SPLITS = 5
KFOLD_SEED = 0
DEFAULT_RANDOM_REPLICATES = 10
DEFAULT_FIXED_SEED_REPLICATES = 3
RANDOM_BASE_SEED = 20260427
FEATURE_BAYES_LINEAR_POLICY = "feature_bayes_linear"
RANDOM_POLICY = "random"
BEST_VARIANT = no_l2_subset.BEST_VARIANT
SUBSET_COARSE_TOP_K = no_l2_subset.SUBSET_COARSE_TOP_K


@dataclass(frozen=True)
class MethodSpec:
    """Regression-fit method to evaluate."""

    method_id: str
    label: str


@dataclass(frozen=True)
class MetricSpec:
    """One metric available in the regression-fit diagnostic."""

    column: str
    label: str
    y_title: str


METHODS = (
    MethodSpec("grp_no_l2", "GRP no-L2"),
    MethodSpec("olmix", "Olmix log-linear"),
)

METRICS = (
    MetricSpec("oof_pearson", "OOF Pearson fit", "Pearson correlation"),
    MetricSpec("complement_pearson", "Complement Pearson fit", "Pearson correlation"),
    MetricSpec("oof_spearman", "OOF Spearman fit", "Spearman correlation"),
    MetricSpec("oof_r2", "OOF R²", "R²"),
    MetricSpec("oof_rmse", "OOF RMSE", "RMSE"),
    MetricSpec("complement_rmse", "Complement RMSE", "RMSE"),
)


def _safe_pearson(predictions: np.ndarray, actuals: np.ndarray) -> float:
    predictions = np.asarray(predictions, dtype=float)
    actuals = np.asarray(actuals, dtype=float)
    mask = np.isfinite(predictions) & np.isfinite(actuals)
    if int(mask.sum()) < 2:
        return float("nan")
    pred = predictions[mask]
    y = actuals[mask]
    if float(np.std(pred)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    return float(np.corrcoef(pred, y)[0, 1])


def _safe_spearman(predictions: np.ndarray, actuals: np.ndarray) -> float:
    predictions = np.asarray(predictions, dtype=float)
    actuals = np.asarray(actuals, dtype=float)
    mask = np.isfinite(predictions) & np.isfinite(actuals)
    if int(mask.sum()) < 2:
        return float("nan")
    pred_rank = pd.Series(predictions[mask]).rank(method="average").to_numpy(dtype=float)
    actual_rank = pd.Series(actuals[mask]).rank(method="average").to_numpy(dtype=float)
    return _safe_pearson(pred_rank, actual_rank)


def _safe_r2(predictions: np.ndarray, actuals: np.ndarray) -> float:
    predictions = np.asarray(predictions, dtype=float)
    actuals = np.asarray(actuals, dtype=float)
    mask = np.isfinite(predictions) & np.isfinite(actuals)
    if int(mask.sum()) < 2:
        return float("nan")
    pred = predictions[mask]
    y = actuals[mask]
    denom = float(np.sum((y - float(np.mean(y))) ** 2))
    if denom == 0.0:
        return float("nan")
    return float(1.0 - np.sum((pred - y) ** 2) / denom)


def _safe_rmse(predictions: np.ndarray, actuals: np.ndarray) -> float:
    predictions = np.asarray(predictions, dtype=float)
    actuals = np.asarray(actuals, dtype=float)
    mask = np.isfinite(predictions) & np.isfinite(actuals)
    if int(mask.sum()) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((predictions[mask] - actuals[mask]) ** 2)))


def _metrics_from_predictions(prefix: str, predictions: np.ndarray, actuals: np.ndarray) -> dict[str, float]:
    return {
        f"{prefix}_pearson": _safe_pearson(predictions, actuals),
        f"{prefix}_spearman": _safe_spearman(predictions, actuals),
        f"{prefix}_r2": _safe_r2(predictions, actuals),
        f"{prefix}_rmse": _safe_rmse(predictions, actuals),
    }


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a plot color from hex to Plotly rgba."""
    color = hex_color.lstrip("#")
    if len(color) != 6:
        raise ValueError(f"Expected #RRGGBB color, got {hex_color}")
    red = int(color[0:2], 16)
    green = int(color[2:4], 16)
    blue = int(color[4:6], 16)
    return f"rgba({red},{green},{blue},{alpha})"


def _subset_indices(subset_size: int, *, policy: str, seed: int) -> np.ndarray:
    _, spec, _ = load_two_phase_many_candidate_summary_spec(
        CSV_PATH,
        objective_metric=OBJECTIVE_METRIC,
        name="f9_regression_fit",
    )
    packet = load_generic_family_packet(target=OBJECTIVE_METRIC)
    if subset_size == len(packet.base.y):
        return np.arange(len(packet.base.y), dtype=int)
    if policy == FEATURE_BAYES_LINEAR_POLICY:
        selection = retrospective_generic_selection(spec, method="feature_bayes_linear", k=subset_size, seed=seed)
        return np.asarray(selection.selected_indices, dtype=int)
    if policy == RANDOM_POLICY:
        rng = np.random.default_rng(seed)
        return np.asarray(sorted(rng.choice(len(packet.base.y), size=subset_size, replace=False).tolist()), dtype=int)
    raise ValueError(f"Unknown subset policy: {policy}")


def _fit_grp_point(
    subset_size: int,
    *,
    subset_policy: str = FEATURE_BAYES_LINEAR_POLICY,
    subset_seed: int = KFOLD_SEED,
    fit_seed: int = 0,
    replicate_kind: str = "fixed_main",
    replicate_id: int = 0,
) -> dict[str, object]:
    packet = load_generic_family_packet(target=OBJECTIVE_METRIC)
    subset_indices = _subset_indices(subset_size, policy=subset_policy, seed=subset_seed)
    train_packet = _subset_packet(packet, subset_indices)
    if subset_size == len(packet.base.y):
        # The full-swarm no-L2 retune is already a canonical artifact. Reusing it avoids spending most of
        # this plotting script re-running the same expensive top-3 Powell search.
        full_summary = genericfamily_penalty_raw_optimum_summary(BEST_VARIANT)
        best_params = {key: float(value) for key, value in full_summary.tuned_params.items()}
        tuning_metrics: dict[str, float] = {}
    else:
        best_params, tuning_metrics, _ = no_l2_subset._optimize_no_l2_subset(
            train_packet,
            coarse_top_k=SUBSET_COARSE_TOP_K,
        )

    y = np.asarray(train_packet.base.y, dtype=float)
    oof = np.full_like(y, np.nan, dtype=float)
    kf = KFold(n_splits=KFOLD_SPLITS, shuffle=True, random_state=CV_SEED + fit_seed)
    for fold_idx, (tr, te) in enumerate(kf.split(train_packet.base.w)):
        del fold_idx
        fold_packet = _subset_packet(train_packet, np.asarray(tr, dtype=int))
        model = build_penalty_calibration_surrogate(
            fold_packet,
            params=best_params,
            variant_name=VARIANT_NAME,
        ).fit(fold_packet.base.w, fold_packet.base.y)
        oof[te] = model.predict(train_packet.base.w[te])

    full_model = build_penalty_calibration_surrogate(
        train_packet,
        params=best_params,
        variant_name=VARIANT_NAME,
    ).fit(train_packet.base.w, train_packet.base.y)
    full_predictions = full_model.predict(packet.base.w)
    return _row_from_predictions(
        method_id="grp_no_l2",
        subset_size=subset_size,
        subset_indices=subset_indices,
        subset_policy=subset_policy,
        subset_seed=subset_seed,
        fit_seed=fit_seed,
        replicate_kind=replicate_kind,
        replicate_id=replicate_id,
        oof_predictions=oof,
        oof_actuals=y,
        full_predictions=full_predictions,
        full_actuals=packet.base.y,
        extra_metrics={
            f"tuning_{key}": float(value) for key, value in tuning_metrics.items() if isinstance(value, float)
        },
    )


def _fit_olmix_point(
    subset_size: int,
    *,
    subset_policy: str = FEATURE_BAYES_LINEAR_POLICY,
    subset_seed: int = KFOLD_SEED,
    fit_seed: int = 0,
    replicate_kind: str = "fixed_main",
    replicate_id: int = 0,
) -> dict[str, object]:
    packet = load_generic_family_packet(target=OBJECTIVE_METRIC)
    subset_indices = _subset_indices(subset_size, policy=subset_policy, seed=subset_seed)
    train_frame = packet.base.frame.iloc[subset_indices].reset_index(drop=True)
    train_spec = build_dataset_spec_from_frame(
        train_frame,
        objective_metric=OBJECTIVE_METRIC,
        name=f"f9_olmix_regression_fit_k{subset_size}",
    )
    weights = np.asarray(train_spec.weights, dtype=float)
    y = np.asarray(train_spec.y, dtype=float)
    oof = np.full_like(y, np.nan, dtype=float)
    kf = KFold(n_splits=KFOLD_SPLITS, shuffle=True, random_state=KFOLD_SEED + fit_seed)
    for fold_idx, (tr, te) in enumerate(kf.split(weights)):
        fit = fit_olmix_loglinear_model(weights[tr], y[tr], seed=fit_seed * 1000 + fold_idx)
        oof[te] = fit.predict(weights[te])

    full_fit = fit_olmix_loglinear_model(weights, y, seed=fit_seed)
    full_predictions = full_fit.predict(packet.base.w)
    return _row_from_predictions(
        method_id="olmix",
        subset_size=subset_size,
        subset_indices=subset_indices,
        subset_policy=subset_policy,
        subset_seed=subset_seed,
        fit_seed=fit_seed,
        replicate_kind=replicate_kind,
        replicate_id=replicate_id,
        oof_predictions=oof,
        oof_actuals=y,
        full_predictions=full_predictions,
        full_actuals=packet.base.y,
        extra_metrics={"huber_loss": float(full_fit.huber_loss)},
    )


def _row_from_predictions(
    *,
    method_id: str,
    subset_size: int,
    subset_indices: np.ndarray,
    subset_policy: str,
    subset_seed: int,
    fit_seed: int,
    replicate_kind: str,
    replicate_id: int,
    oof_predictions: np.ndarray,
    oof_actuals: np.ndarray,
    full_predictions: np.ndarray,
    full_actuals: np.ndarray,
    extra_metrics: dict[str, float],
) -> dict[str, object]:
    complement_mask = np.ones(len(full_actuals), dtype=bool)
    complement_mask[subset_indices] = False
    row: dict[str, object] = {
        "method_id": method_id,
        "method": next(method.label for method in METHODS if method.method_id == method_id),
        "subset_size": subset_size,
        "objective_metric": OBJECTIVE_METRIC,
        "subset_policy": subset_policy,
        "subset_seed": int(subset_seed),
        "fit_seed": int(fit_seed),
        "replicate_kind": replicate_kind,
        "replicate_id": int(replicate_id),
        "subset_indices": json.dumps([int(index) for index in subset_indices.tolist()]),
        **_metrics_from_predictions("oof", oof_predictions, oof_actuals),
        **_metrics_from_predictions("fullswarm", full_predictions, full_actuals),
        **extra_metrics,
    }
    if complement_mask.any():
        row.update(
            _metrics_from_predictions(
                "complement",
                full_predictions[complement_mask],
                np.asarray(full_actuals, dtype=float)[complement_mask],
            )
        )
    else:
        row.update({f"complement_{suffix}": float("nan") for suffix in ("pearson", "spearman", "r2", "rmse")})
    return row


def _fit_point(payload: tuple[str, int]) -> dict[str, object]:
    method_id, subset_size = payload
    if method_id == "grp_no_l2":
        return _fit_grp_point(subset_size)
    if method_id == "olmix":
        return _fit_olmix_point(subset_size)
    raise ValueError(f"Unknown method_id: {method_id}")


def _fit_replicate_point(payload: tuple[str, int, str, int, int, str, int]) -> dict[str, object]:
    method_id, subset_size, subset_policy, subset_seed, fit_seed, replicate_kind, replicate_id = payload
    kwargs = {
        "subset_policy": subset_policy,
        "subset_seed": subset_seed,
        "fit_seed": fit_seed,
        "replicate_kind": replicate_kind,
        "replicate_id": replicate_id,
    }
    if method_id == "grp_no_l2":
        return _fit_grp_point(subset_size, **kwargs)
    if method_id == "olmix":
        return _fit_olmix_point(subset_size, **kwargs)
    raise ValueError(f"Unknown method_id: {method_id}")


def _compute_points(max_workers: int) -> pd.DataFrame:
    tasks = [(method.method_id, subset_size) for method in METHODS for subset_size in SUBSET_SIZES]
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fit_point, task): task for task in tasks}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"Finished {row['method']} subset_size={row['subset_size']} "
                f"oof_pearson={float(row['oof_pearson']):.4f}",
                flush=True,
            )
    return pd.DataFrame(rows).sort_values(["method_id", "subset_size"]).reset_index(drop=True)


def _compute_replicates(
    *,
    max_workers: int,
    random_replicates: int,
    fixed_seed_replicates: int,
) -> pd.DataFrame:
    packet = load_generic_family_packet(target=OBJECTIVE_METRIC)
    full_size = len(packet.base.y)
    tasks: list[tuple[str, int, str, int, int, str, int]] = []
    existing = pd.DataFrame()
    completed_keys: set[tuple[str, int, str, int]] = set()
    if OUTPUT_REPLICATES_PARTIAL_CSV.exists():
        existing = pd.read_csv(OUTPUT_REPLICATES_PARTIAL_CSV)
        if not existing.empty:
            completed_keys = {
                (str(row.method_id), int(row.subset_size), str(row.replicate_kind), int(row.replicate_id))
                for row in existing.itertuples(index=False)
            }
    for method in METHODS:
        for subset_size in SUBSET_SIZES:
            for replicate_id in range(fixed_seed_replicates):
                key = (method.method_id, subset_size, "fixed_seed", replicate_id)
                if key in completed_keys:
                    continue
                tasks.append(
                    (
                        method.method_id,
                        subset_size,
                        FEATURE_BAYES_LINEAR_POLICY,
                        KFOLD_SEED,
                        replicate_id,
                        "fixed_seed",
                        replicate_id,
                    )
                )
            if subset_size == full_size:
                continue
            for replicate_id in range(random_replicates):
                key = (method.method_id, subset_size, "random_subset", replicate_id)
                if key in completed_keys:
                    continue
                subset_seed = RANDOM_BASE_SEED + 1009 * subset_size + replicate_id
                tasks.append(
                    (
                        method.method_id,
                        subset_size,
                        RANDOM_POLICY,
                        subset_seed,
                        replicate_id,
                        "random_subset",
                        replicate_id,
                    )
                )

    rows: list[dict[str, object]] = existing.to_dict(orient="records") if not existing.empty else []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fit_replicate_point, task): task for task in tasks}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            pd.DataFrame(rows).to_csv(OUTPUT_REPLICATES_PARTIAL_CSV, index=False)
            print(
                "Finished "
                f"{row['method']} {row['replicate_kind']} subset_size={row['subset_size']} "
                f"replicate={row['replicate_id']} oof_pearson={float(row['oof_pearson']):.4f}",
                flush=True,
            )
    return (
        pd.DataFrame(rows)
        .sort_values(["replicate_kind", "method_id", "subset_size", "replicate_id"])
        .reset_index(drop=True)
    )


def _summarize_bands(replicates: pd.DataFrame) -> pd.DataFrame:
    metric_columns = sorted(
        {metric.column for metric in METRICS}
        | {f"complement_{suffix}" for suffix in ("pearson", "spearman", "r2", "rmse")}
    )
    rows: list[dict[str, object]] = []
    group_columns = ["replicate_kind", "method_id", "method", "subset_policy", "subset_size"]
    for group_values, group in replicates.groupby(group_columns, dropna=False):
        row = dict(zip(group_columns, group_values, strict=True))
        row["n_replicates"] = len(group)
        for metric_column in metric_columns:
            if metric_column not in group.columns:
                continue
            values = pd.to_numeric(group[metric_column], errors="coerce").dropna().to_numpy(dtype=float)
            if len(values) == 0:
                for suffix in ("mean", "std", "min", "q10", "q25", "median", "q75", "q90", "max"):
                    row[f"{metric_column}_{suffix}"] = float("nan")
                continue
            row[f"{metric_column}_mean"] = float(np.mean(values))
            row[f"{metric_column}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric_column}_min"] = float(np.min(values))
            row[f"{metric_column}_q10"] = float(np.quantile(values, 0.10))
            row[f"{metric_column}_q25"] = float(np.quantile(values, 0.25))
            row[f"{metric_column}_median"] = float(np.quantile(values, 0.50))
            row[f"{metric_column}_q75"] = float(np.quantile(values, 0.75))
            row[f"{metric_column}_q90"] = float(np.quantile(values, 0.90))
            row[f"{metric_column}_max"] = float(np.max(values))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["replicate_kind", "method_id", "subset_size"]).reset_index(drop=True)


def _hover_text(frame: pd.DataFrame, metric: MetricSpec) -> list[str]:
    text: list[str] = []
    for row in frame.to_dict(orient="records"):
        value = row.get(metric.column)
        value_text = "nan" if value is None or pd.isna(value) else f"{float(value):.6f}"
        text.append(
            "<br>".join(
                [
                    f"Method: {row['method']}",
                    f"Subset size: {int(row['subset_size'])}",
                    f"{metric.label}: {value_text}",
                    f"Subset policy: {row.get('subset_policy', FEATURE_BAYES_LINEAR_POLICY)}",
                    f"Fit seed: {int(row.get('fit_seed', 0))}",
                ]
            )
        )
    return text


def _add_metric_traces(
    fig: go.Figure,
    points: pd.DataFrame,
    metric: MetricSpec,
    *,
    bands: pd.DataFrame | None = None,
    row: int | None = None,
    col: int | None = None,
    visible: bool | None = None,
    showlegend: bool = True,
) -> list[int]:
    trace_indices: list[int] = []
    for method in METHODS:
        color = method_color(method.method_id)
        if bands is not None and not bands.empty:
            band_frame = bands[
                (bands["method_id"] == method.method_id) & (bands["replicate_kind"] == "random_subset")
            ].sort_values("subset_size")
            lower_column = f"{metric.column}_min"
            upper_column = f"{metric.column}_max"
            if not band_frame.empty and lower_column in band_frame.columns and upper_column in band_frame.columns:
                valid = band_frame[["subset_size", lower_column, upper_column]].dropna()
                if not valid.empty:
                    x_values = valid["subset_size"].to_numpy(dtype=float)
                    lower = valid[lower_column].to_numpy(dtype=float)
                    upper = valid[upper_column].to_numpy(dtype=float)
                    fig.add_trace(
                        go.Scatter(
                            x=np.concatenate([x_values, x_values[::-1]]),
                            y=np.concatenate([upper, lower[::-1]]),
                            mode="lines",
                            line={"color": "rgba(0,0,0,0)", "width": 0},
                            fill="toself",
                            fillcolor=_hex_to_rgba(color, 0.10),
                            legendgroup=f"{method.method_id}_random_subset",
                            name=f"{method.label} random-subset range",
                            showlegend=False,
                            visible=visible,
                            hoverinfo="skip",
                        ),
                        row=row,
                        col=col,
                    )
                    trace_indices.append(len(fig.data) - 1)

        random_frame = pd.DataFrame()
        if bands is not None and not bands.empty:
            random_frame = bands[
                (bands["method_id"] == method.method_id) & (bands["replicate_kind"] == "random_subset")
            ].sort_values("subset_size")
        if not random_frame.empty and f"{metric.column}_median" in random_frame.columns:
            fig.add_trace(
                go.Scatter(
                    x=random_frame["subset_size"],
                    y=random_frame[f"{metric.column}_median"],
                    mode="lines+markers",
                    name=f"{method.label} random subsets",
                    legendgroup=f"{method.method_id}_random_subset",
                    showlegend=showlegend,
                    visible=visible,
                    line={"color": color, "width": 2.5, "dash": "dash"},
                    marker={"color": "white", "size": 8, "line": {"color": color, "width": 2}},
                    hovertemplate=(
                        "Method: %{customdata[0]}<br>"
                        "Subset size: %{x}<br>"
                        f"{metric.label} median: "
                        "%{y:.6f}<br>"
                        "Subset policy: random<extra></extra>"
                    ),
                    customdata=np.asarray([[method.label]] * len(random_frame), dtype=object),
                ),
                row=row,
                col=col,
            )
            trace_indices.append(len(fig.data) - 1)

        point_frame = points[points["method_id"] == method.method_id].sort_values("subset_size")
        frame = point_frame
        error_y: dict[str, object] | None = None
        if bands is not None and not bands.empty:
            fixed_frame = bands[
                (bands["method_id"] == method.method_id) & (bands["replicate_kind"] == "fixed_seed")
            ].sort_values("subset_size")
            lower_column = f"{metric.column}_min"
            median_column = f"{metric.column}_median"
            upper_column = f"{metric.column}_max"
            if (
                not fixed_frame.empty
                and lower_column in fixed_frame.columns
                and median_column in fixed_frame.columns
                and upper_column in fixed_frame.columns
            ):
                frame = fixed_frame.rename(columns={median_column: metric.column})
                y_values = frame[metric.column].to_numpy(dtype=float)
                upper = frame[upper_column].to_numpy(dtype=float)
                lower = frame[lower_column].to_numpy(dtype=float)
                error_y = {
                    "type": "data",
                    "array": np.maximum(0.0, upper - y_values),
                    "arrayminus": np.maximum(0.0, y_values - lower),
                    "thickness": 1.4,
                    "width": 4,
                    "color": color,
                    "visible": True,
                }
        trace = go.Scatter(
            x=frame["subset_size"],
            y=frame[metric.column],
            mode="lines+markers",
            name=f"{method.label} D-optimal subset",
            legendgroup=f"{method.method_id}_fixed",
            showlegend=showlegend,
            visible=visible,
            line={"color": color, "width": 3},
            marker={"color": color, "size": 9, "line": {"color": "white", "width": 1}},
            error_y=error_y,
            hovertemplate=(
                "Method: %{customdata[0]}<br>"
                "Subset size: %{x}<br>"
                f"{metric.label} seed median: "
                "%{y:.6f}<br>"
                "Subset policy: D-optimal<extra></extra>"
            ),
            customdata=np.asarray([[method.label]] * len(frame), dtype=object),
        )
        fig.add_trace(trace, row=row, col=col)
        trace_indices.append(len(fig.data) - 1)
    return trace_indices


def _build_grid_figure(points: pd.DataFrame, bands: pd.DataFrame) -> go.Figure:
    metrics = METRICS[:4]
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[metric.label for metric in metrics],
        horizontal_spacing=0.09,
        vertical_spacing=0.16,
    )
    for idx, metric in enumerate(metrics):
        row = idx // 2 + 1
        col = idx % 2 + 1
        _add_metric_traces(fig, points, metric, bands=bands, row=row, col=col, showlegend=(idx == 0))
        fig.update_xaxes(title_text="subset size", row=row, col=col)
        fig.update_yaxes(title_text=metric.y_title, row=row, col=col)
    configure_interactive_layout(
        fig,
        title="Olmix-style regression fit on 60M subset curves",
        y_title="",
        x_title="",
    )
    fig.update_layout(
        width=1350,
        height=900,
        margin={"l": 86, "r": 190, "t": 130, "b": 90},
    )
    fig.update_annotations(font={"size": 22, "color": PAPER_MUTED})
    return fig


def _build_paper_figure(points: pd.DataFrame, bands: pd.DataFrame) -> go.Figure:
    metric = METRICS[0]
    fig = go.Figure()
    _add_metric_traces(fig, points, metric, bands=bands, showlegend=True)
    random_replicates = 0
    fixed_replicates = 0
    if not bands.empty and "n_replicates" in bands.columns:
        random_values = bands[bands["replicate_kind"] == "random_subset"]["n_replicates"]
        fixed_values = bands[bands["replicate_kind"] == "fixed_seed"]["n_replicates"]
        random_replicates = int(random_values.max()) if not random_values.empty else 0
        fixed_replicates = int(fixed_values.max()) if not fixed_values.empty else 0
    short_names = {
        "GRP no-L2 random subsets": f"GRP random subsets (n={random_replicates})",
        "GRP no-L2 D-optimal subset": f"GRP D-optimal subset seeds (n={fixed_replicates})",
        "Olmix log-linear random subsets": f"Olmix random subsets (n={random_replicates})",
        "Olmix log-linear D-optimal subset": f"Olmix D-optimal subset seeds (n={fixed_replicates})",
    }
    for trace in fig.data:
        if trace.name in short_names:
            trace.name = short_names[trace.name]
    configure_static_layout(
        fig,
        y_title=metric.y_title,
        x_title="60M swarm runs used for fitting",
    )
    fig.update_layout(
        width=780,
        height=520,
        legend={
            "orientation": "h",
            "x": 0.5,
            "y": 1.18,
            "xanchor": "center",
            "yanchor": "bottom",
            "entrywidth": 0.48,
            "entrywidthmode": "fraction",
            "font": {"size": 15},
        },
        margin={"l": 76, "r": 22, "t": 112, "b": 78},
    )
    fig.update_xaxes(tickmode="array", tickvals=list(SUBSET_SIZES), tickangle=0)
    fig.update_yaxes(range=[-0.45, 1.0])
    return fig


def _build_picker_figure(points: pd.DataFrame, bands: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    traces_by_metric: list[list[int]] = []
    for metric_idx, metric in enumerate(METRICS):
        traces = _add_metric_traces(
            fig,
            points,
            metric,
            bands=bands,
            visible=(True if metric_idx == 0 else False),
            showlegend=(metric_idx == 0),
        )
        traces_by_metric.append(traces)

    buttons = []
    for metric_idx, metric in enumerate(METRICS):
        visible = [False] * len(fig.data)
        for trace_idx in traces_by_metric[metric_idx]:
            visible[trace_idx] = True
        buttons.append(
            {
                "label": metric.label,
                "method": "update",
                "args": [
                    {"visible": visible},
                    {
                        "title": {"text": f"Olmix-style regression fit: {metric.label}"},
                        "yaxis": {"title": {"text": metric.y_title}},
                    },
                ],
            }
        )
    configure_interactive_layout(
        fig,
        title=f"Olmix-style regression fit: {METRICS[0].label}",
        y_title=METRICS[0].y_title,
        x_title="subset size",
    )
    fig.update_layout(
        updatemenus=[
            {
                "type": "dropdown",
                "buttons": buttons,
                "direction": "down",
                "x": 0.02,
                "xanchor": "left",
                "y": 1.08,
                "yanchor": "top",
            }
        ],
    )
    return fig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="Parallel workers for subset fits.",
    )
    parser.add_argument(
        "--random-replicates",
        type=int,
        default=DEFAULT_RANDOM_REPLICATES,
        help="Random-subset replicates per method and subset size.",
    )
    parser.add_argument(
        "--fixed-seed-replicates",
        type=int,
        default=DEFAULT_FIXED_SEED_REPLICATES,
        help="Fixed-subset refit/CV seeds per method and subset size.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute cached deterministic and replicate CSVs.",
    )
    parser.add_argument(
        "--skip-ci",
        action="store_true",
        help="Only render deterministic fixed-subset curves.",
    )
    return parser.parse_args()


def _load_or_compute_points(*, max_workers: int, force: bool) -> pd.DataFrame:
    if OUTPUT_POINTS_CSV.exists() and not force:
        return pd.read_csv(OUTPUT_POINTS_CSV)
    task_count = len(METHODS) * len(SUBSET_SIZES)
    return _compute_points(max_workers=max(1, min(task_count, max_workers)))


def _load_or_compute_replicates(
    *,
    max_workers: int,
    random_replicates: int,
    fixed_seed_replicates: int,
    force: bool,
    skip_ci: bool,
) -> pd.DataFrame:
    if skip_ci:
        return pd.DataFrame()
    if OUTPUT_REPLICATES_CSV.exists() and not force:
        cached = pd.read_csv(OUTPUT_REPLICATES_CSV)
        if _replicate_cache_satisfies(
            cached,
            random_replicates=random_replicates,
            fixed_seed_replicates=fixed_seed_replicates,
        ):
            return cached
    if random_replicates <= 0 and fixed_seed_replicates <= 0:
        return pd.DataFrame()
    if force and OUTPUT_REPLICATES_PARTIAL_CSV.exists():
        OUTPUT_REPLICATES_PARTIAL_CSV.unlink()
    if not force and OUTPUT_REPLICATES_CSV.exists() and not OUTPUT_REPLICATES_PARTIAL_CSV.exists():
        cached = pd.read_csv(OUTPUT_REPLICATES_CSV)
        if not cached.empty:
            cached.to_csv(OUTPUT_REPLICATES_PARTIAL_CSV, index=False)
    task_count = len(METHODS) * len(SUBSET_SIZES) * max(1, random_replicates + fixed_seed_replicates)
    return _compute_replicates(
        max_workers=max(1, min(task_count, max_workers)),
        random_replicates=max(0, random_replicates),
        fixed_seed_replicates=max(0, fixed_seed_replicates),
    )


def _replicate_cache_satisfies(
    replicates: pd.DataFrame,
    *,
    random_replicates: int,
    fixed_seed_replicates: int,
) -> bool:
    if replicates.empty:
        return random_replicates <= 0 and fixed_seed_replicates <= 0
    required = {
        "method_id",
        "subset_size",
        "replicate_kind",
        "replicate_id",
    }
    if not required.issubset(replicates.columns):
        return False
    full_size = int(max(SUBSET_SIZES))
    for method in METHODS:
        for subset_size in SUBSET_SIZES:
            fixed_count = replicates[
                (replicates["method_id"] == method.method_id)
                & (replicates["subset_size"] == subset_size)
                & (replicates["replicate_kind"] == "fixed_seed")
            ]["replicate_id"].nunique()
            if fixed_count < fixed_seed_replicates:
                return False
            if subset_size == full_size:
                continue
            random_count = replicates[
                (replicates["method_id"] == method.method_id)
                & (replicates["subset_size"] == subset_size)
                & (replicates["replicate_kind"] == "random_subset")
            ]["replicate_id"].nunique()
            if random_count < random_replicates:
                return False
    return True


def main() -> None:
    args = _parse_args()
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    points = _load_or_compute_points(max_workers=int(args.max_workers), force=bool(args.force))
    points.to_csv(OUTPUT_POINTS_CSV, index=False)
    replicates = _load_or_compute_replicates(
        max_workers=int(args.max_workers),
        random_replicates=int(args.random_replicates),
        fixed_seed_replicates=int(args.fixed_seed_replicates),
        force=bool(args.force),
        skip_ci=bool(args.skip_ci),
    )
    if replicates.empty:
        bands = pd.DataFrame()
    else:
        replicates.to_csv(OUTPUT_REPLICATES_CSV, index=False)
        bands = _summarize_bands(replicates)
        bands.to_csv(OUTPUT_BANDS_CSV, index=False)
    OUTPUT_SUMMARY_JSON.write_text(
        json.dumps(
            {
                "objective_metric": OBJECTIVE_METRIC,
                "subset_sizes": list(SUBSET_SIZES),
                "methods": [method.__dict__ for method in METHODS],
                "metrics": [metric.__dict__ for metric in METRICS],
                "definition": (
                    "Olmix regression fit is Pearson correlation between predicted and true BPB on held-out mixtures."
                ),
                "points_csv": str(OUTPUT_POINTS_CSV),
                "replicates_csv": str(OUTPUT_REPLICATES_CSV) if not replicates.empty else None,
                "bands_csv": str(OUTPUT_BANDS_CSV) if not bands.empty else None,
                "random_replicates": int(args.random_replicates),
                "fixed_seed_replicates": int(args.fixed_seed_replicates),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    grid = _build_grid_figure(points, bands)
    grid.write_html(OUTPUT_STEM.with_name(f"{OUTPUT_STEM.name}_grid.html"))
    write_static_images(grid, OUTPUT_STEM.with_name(f"{OUTPUT_STEM.name}_grid"))

    paper = _build_paper_figure(points, bands)
    paper.write_html(PAPER_OUTPUT_STEM.with_suffix(".html"))
    write_static_images(paper, PAPER_OUTPUT_STEM)

    picker = _build_picker_figure(points, bands)
    picker.write_html(OUTPUT_STEM.with_name(f"{OUTPUT_STEM.name}_picker.html"))
    print(f"Wrote {OUTPUT_POINTS_CSV}")
    if not replicates.empty:
        print(f"Wrote {OUTPUT_REPLICATES_CSV}")
    if not bands.empty:
        print(f"Wrote {OUTPUT_BANDS_CSV}")
    print(f"Wrote {OUTPUT_SUMMARY_JSON}")
    print(f"Wrote {OUTPUT_STEM.with_name(f'{OUTPUT_STEM.name}_grid.png')}")
    print(f"Wrote {PAPER_OUTPUT_STEM.with_suffix('.png')}")
    print(f"Wrote {OUTPUT_STEM.with_name(f'{OUTPUT_STEM.name}_picker.html')}")


if __name__ == "__main__":
    main()
