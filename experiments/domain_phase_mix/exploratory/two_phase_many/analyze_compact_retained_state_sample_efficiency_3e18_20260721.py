# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy>=1.7",
#   "fsspec>=2025.7",
#   "gcsfs>=2025.7",
#   "numpy>=2.0",
#   "pandas>=2.2",
#   "plotly>=6.0",
#   "scikit-learn>=1.6",
#   "scipy>=1.15",
#   "tabulate>=0.9",
# ]
# ///
"""Audit Compact retained-state sample efficiency below 280 Delphi fit rows.

This is a local development diagnostic. It never trains checkpoints and never
uses development-heldout outcomes to select model form or hyperparameters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    analyze_delphi_phase_policy_sample_efficiency_20260721 as sample_eff,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_retained_weibull_replay_20260713 as compact_retained,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    export_mixture_fit_observatory as observatory,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/compact_retained_state_sample_efficiency_3e18_20260721"
TARGETS = ("uncheatable", "table9")
TARGET_COLUMNS = sample_eff.TARGET_COLUMNS
SAMPLE_SIZES = (48, 64, 80, 112, 144, 184, 232, 280)
SAMPLING_DESIGNS = ("panel_stratified", "intervention_core")
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
L2_GRID = tuple(float(value) for value in observatory.COMPACT_L2_GRID)
INNER_CV_SPLITS = 3
RAW_OPTIMIZER_STARTS = 8
BASELINE_NAMES = ("baseline_proportional", "baseline_unimax", "baseline_stratified")
MODEL_ID = "compact_retained_state"
POLICY_CLASS = observatory.TWO_PHASE
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}, "responsive": True}
DESIGN_LABELS = {
    "panel_stratified": "Panel-proportional sample",
    "intervention_core": "Deletion + anchor core",
}
DESIGN_COLORS = {
    "panel_stratified": "#d73027",
    "intervention_core": "#1a9850",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-sizes", default=",".join(str(value) for value in SAMPLE_SIZES))
    parser.add_argument("--seeds", default=",".join(str(value) for value in DEFAULT_SEEDS))
    parser.add_argument("--targets", default=",".join(TARGETS))
    parser.add_argument("--optimizer-starts", type=int, default=RAW_OPTIMIZER_STARTS)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-optima", action="store_true")
    parser.add_argument("--rebuild-only", action="store_true")
    return parser.parse_args()


def parse_values(raw: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in raw.split(",") if value.strip())
    if not values:
        raise ValueError("At least one value is required")
    return values


def parse_ints(raw: str) -> tuple[int, ...]:
    return tuple(int(value) for value in parse_values(raw))


def protocol_payload(sample_sizes: tuple[int, ...], seeds: tuple[int, ...], targets: tuple[str, ...]) -> dict[str, Any]:
    return {
        "version": 1,
        "model": MODEL_ID,
        "policy_class": POLICY_CLASS,
        "targets": list(targets),
        "sample_sizes": list(sample_sizes),
        "sampling_designs": list(SAMPLING_DESIGNS),
        "seeds": list(seeds),
        "l2_grid": list(L2_GRID),
        "inner_cv_splits": INNER_CV_SPLITS,
        "raw_optimizer_starts": RAW_OPTIMIZER_STARTS,
        "minimum_sample_size_reason": "48 exceeds the 45 nominal parameters and the 42-row intervention core",
        "intervention_core": {
            "domain_deletion_rows": 39,
            "anchor_rows": list(BASELINE_NAMES),
        },
        "subsampling_rule": "nested random priorities within frozen design strata",
        "hyperparameter_rule": "select L2 by three-fold panel-stratified CV inside each selected subset",
        "heldout_rule": "all complete coordinate-disjoint two-phase policies, collapsed by coordinate",
        "heldout_use": "evaluation only; never used for fitting, model selection, or L2 selection",
        "data_status": "exposed local development diagnostic; not confirmatory",
    }


def persist_protocol(output_dir: Path, protocol: dict[str, Any], overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "protocol.json"
    if path.exists() and not overwrite:
        existing = json.loads(path.read_text())
        if existing != protocol:
            raise ValueError(f"Existing protocol differs at {path}; choose a new output directory or use --overwrite")
        return
    path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")


def row_digest(indices: np.ndarray) -> str:
    payload = ",".join(str(int(value)) for value in np.sort(indices))
    return hashlib.sha256(payload.encode()).hexdigest()


def panel_stratified_quotas(frame: pd.DataFrame, total: int) -> dict[str, int]:
    counts = frame["panel_source"].value_counts(sort=False)
    return sample_eff.largest_remainder_quotas(counts, total)


def nested_subsets(
    frame: pd.DataFrame,
    sample_sizes: tuple[int, ...],
    design: str,
    seed: int,
) -> dict[int, np.ndarray]:
    rng = np.random.default_rng(seed)
    by_panel = {
        str(panel): rng.permutation(np.asarray(indices, dtype=int))
        for panel, indices in frame.groupby("panel_source", sort=True).indices.items()
    }
    all_indices = np.arange(len(frame), dtype=int)
    if design == "intervention_core":
        deletion = np.flatnonzero(frame["panel_source"].eq("domain_deletion").to_numpy())
        anchors = np.flatnonzero(frame["run_name"].isin(BASELINE_NAMES).to_numpy())
        core = np.unique(np.concatenate([deletion, anchors]))
        if len(core) != 42:
            raise ValueError(f"Expected 42 intervention-core rows, found {len(core)}")
        remainder = rng.permutation(np.setdiff1d(all_indices, core, assume_unique=True))

    result: dict[int, np.ndarray] = {}
    previous: set[int] = set()
    for sample_size in sample_sizes:
        if sample_size == len(frame):
            selected = all_indices.copy()
        elif design == "panel_stratified":
            quotas = panel_stratified_quotas(frame, sample_size)
            selected = np.sort(np.concatenate([by_panel[panel][:quota] for panel, quota in sorted(quotas.items())]))
        elif design == "intervention_core":
            if sample_size < len(core):
                raise ValueError(f"Sample size {sample_size} is smaller than the {len(core)}-row intervention core")
            selected = np.sort(np.concatenate([core, remainder[: sample_size - len(core)]]))
        else:
            raise ValueError(f"Unknown sampling design {design!r}")
        current = set(int(value) for value in selected)
        if len(selected) != sample_size or not previous.issubset(current):
            raise AssertionError(f"Non-nested {design} subset for seed={seed}, n={sample_size}")
        result[sample_size] = selected
        previous = current
    return result


def fixed_heldout_pool(reference: sample_eff.pooled.Dataset) -> sample_eff.CoordinatePool:
    frame, weights = observatory.load_delphi_3e18_heldouts(reference)
    mask = (
        frame["policy_class"].eq("two_phase")
        & frame["fit_panel_overlap"].eq("coordinate_disjoint")
        & frame["training_state"].eq("finished")
        & frame["checkpoint_declared_complete"].eq(1)
    )
    for column in TARGET_COLUMNS.values():
        mask &= pd.to_numeric(frame[column], errors="coerce").notna()
    pool = sample_eff.coordinate_pool(frame.loc[mask].reset_index(drop=True), weights[mask.to_numpy()])
    if len(pool.frame) < 900:
        raise ValueError(f"Expected at least 900 coordinate-distinct two-phase heldouts, found {len(pool.frame)}")
    return pool


def subset_dataset(
    reference: sample_eff.pooled.Dataset,
    indices: np.ndarray,
    target: str,
    name: str,
) -> sample_eff.pooled.Dataset:
    frame = reference.frame.iloc[indices].reset_index(drop=True).copy()
    frame[TARGET_COLUMNS[target]] = reference.y[indices]
    return sample_eff.target_dataset(reference, frame, reference.weights[indices], target, name)


def metric_prefix(prefix: str, values: dict[str, float | int]) -> dict[str, float | int]:
    return {f"{prefix}_{name}": value for name, value in values.items()}


def empty_metrics(prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_{name}": float("nan")
        for name in (
            "n_eval",
            "rmse",
            "normalized_rmse",
            "spearman",
            "bias",
            "calibration_slope",
            "calibration_error",
            "regret_at_1",
            "regret_at_3",
            "regret_at_5",
            "lower_tail_optimism",
            "low_tail_rmse",
            "optimism_gt_0p05",
            "worst_optimism",
            "selected_optimism",
            "selected_observed",
            "selected_predicted",
            "frontier_observed",
        )
    }


def select_l2(dataset: sample_eff.pooled.Dataset, seed: int) -> tuple[float, pd.DataFrame]:
    labels = dataset.frame["panel_source"].astype(str).to_numpy()
    splitter = StratifiedKFold(n_splits=INNER_CV_SPLITS, shuffle=True, random_state=seed)
    rows: list[dict[str, float]] = []
    for l2 in L2_GRID:
        prediction = np.full(dataset.n, np.nan, dtype=float)
        for train, test in splitter.split(np.arange(dataset.n), labels):
            model = observatory.compact_fit(dataset, train, l2, POLICY_CLASS)
            prediction[test] = model.predict(dataset.weights[test])
        if not np.isfinite(prediction).all():
            raise ValueError(f"Incomplete inner-CV prediction for n={dataset.n}, L2={l2}")
        values = sample_eff.metrics(dataset.y, prediction)
        rows.append({"l2": l2, **{name: float(value) for name, value in values.items()}})
    frame = pd.DataFrame(rows)
    selected = min(
        rows,
        key=lambda row: (row["rmse"], -row["spearman"], row["l2"]),
    )
    return float(selected["l2"]), frame


def shape_record(
    model: compact_retained.FittedModel,
    full_model: compact_retained.FittedModel,
) -> dict[str, float | int]:
    shape = model.shape
    full_shape = full_model.shape
    coefficient_delta = np.linalg.norm(model.signal_coef - full_model.signal_coef)
    coefficient_scale = max(np.linalg.norm(full_model.signal_coef), 1e-12)
    return {
        "shape_rate": shape.rate,
        "shape_power": shape.power,
        "shape_late_multiplier": shape.late_multiplier,
        "shape_forgetting_rate": shape.forgetting_rate,
        "shape_log_rate_delta_to_full": math.log(max(shape.rate, 1e-12) / max(full_shape.rate, 1e-12)),
        "shape_power_delta_to_full": shape.power - full_shape.power,
        "shape_log_late_multiplier_delta_to_full": math.log(
            max(shape.late_multiplier, 1e-12) / max(full_shape.late_multiplier, 1e-12)
        ),
        "shape_forgetting_rate_delta_to_full": shape.forgetting_rate - full_shape.forgetting_rate,
        "signal_coefficient_relative_l2_to_full": float(coefficient_delta / coefficient_scale),
        "active_signal_coefficients": int(np.sum(model.signal_coef > 1e-10)),
        "active_replay_coefficients": int(np.sum(model.replay_coef > 1e-10)),
    }


def fit_record_key(record: dict[str, Any]) -> tuple[str, str, int, int]:
    return str(record["target"]), str(record["sampling_design"]), int(record["sample_size"]), int(record["seed"])


def load_rows(path: Path, overwrite: bool) -> list[dict[str, Any]]:
    if overwrite or not path.exists():
        return []
    return pd.read_csv(path).to_dict(orient="records")


def persist_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows).sort_values(["target", "sampling_design", "sample_size", "seed"]).to_csv(path, index=False)


def target_matched_mask(frame: pd.DataFrame, target: str) -> np.ndarray:
    if "proposal_target" not in frame:
        return np.zeros(len(frame), dtype=bool)
    proposal = frame["proposal_target"].fillna("").astype(str)
    return proposal.eq(target).to_numpy()


def raw_optimum_metrics(
    optimum: sample_eff.RawOptimum,
    train: sample_eff.pooled.Dataset,
    full_optimum: sample_eff.RawOptimum,
) -> dict[str, float | int | bool]:
    alpha0, alpha1 = observatory.phase_fractions(train)
    aggregate = alpha0 * optimum.weights[0] + alpha1 * optimum.weights[1]
    proportional = observatory.natural_weights(train, alpha0)
    exposure = optimum.weights[0] * train.c0 + optimum.weights[1] * train.c1
    return {
        "raw_predicted_bpb": optimum.predicted_bpb,
        "raw_optimizer_converged": optimum.optimizer_converged,
        "raw_successful_starts": optimum.successful_starts,
        "raw_finite_starts": optimum.finite_starts,
        "raw_max_bucket_weight": float(optimum.weights.max()),
        "raw_max_simulated_epochs": float(exposure.max()),
        "raw_phase_total_variation": float(0.5 * np.abs(optimum.weights[0] - optimum.weights[1]).sum()),
        "raw_aggregate_tv_to_proportional": float(0.5 * np.abs(aggregate - proportional).sum()),
        "raw_standardized_fit_support_distance": sample_eff.standardized_support_distance(train, optimum.weights),
        "raw_policy_tv_to_full_280_optimum": sample_eff.weighted_policy_tv(
            optimum.weights,
            full_optimum.weights,
            alpha0,
            alpha1,
        ),
        "raw_phase_0_weights_json": json.dumps(optimum.weights[0].tolist(), separators=(",", ":")),
        "raw_phase_1_weights_json": json.dumps(optimum.weights[1].tolist(), separators=(",", ":")),
    }


def run_analysis(
    output_dir: Path,
    targets: tuple[str, ...],
    sample_sizes: tuple[int, ...],
    seeds: tuple[int, ...],
    optimizer_starts: int,
    overwrite: bool,
    skip_optima: bool,
) -> pd.DataFrame:
    path = output_dir / "learning_curve_runs.csv"
    rows = load_rows(path, overwrite)
    complete = {fit_record_key(row) for row in rows}

    for target in targets:
        reference = observatory.load_delphi_3e18_fit_dataset(target)
        heldout = fixed_heldout_pool(reference)
        heldout_dataset = sample_eff.target_dataset(
            reference,
            heldout.frame,
            heldout.weights,
            target,
            f"compact_sample_efficiency_heldout_{target}",
        )
        full_spec = sample_eff.frozen_spec(target, POLICY_CLASS, MODEL_ID)
        full_l2 = float(full_spec.tuning["l2"])
        full_model = observatory.compact_fit(reference, np.arange(reference.n), full_l2, POLICY_CLASS)
        full_heldout_prediction = full_model.predict(heldout.weights)
        full_optimum = sample_eff.optimize_raw_model(
            reference,
            full_model,
            full_spec,
            seed=0,
            count=optimizer_starts,
            previous=None,
        )
        matched = target_matched_mask(heldout.frame, target)

        for design in SAMPLING_DESIGNS:
            for seed in seeds:
                subsets = nested_subsets(reference.frame, sample_sizes, design, seed)
                for sample_size in sample_sizes:
                    effective_seed = 0 if sample_size == reference.n else seed
                    key = (target, design, sample_size, effective_seed)
                    if key in complete or (sample_size == reference.n and seed != seeds[0]):
                        continue
                    selected = subsets[sample_size]
                    train = subset_dataset(
                        reference,
                        selected,
                        target,
                        f"compact_sample_efficiency_{target}_{design}_{sample_size}_{effective_seed}",
                    )
                    selected_l2, cv_rows = select_l2(train, effective_seed)
                    model = observatory.compact_fit(train, np.arange(train.n), selected_l2, POLICY_CLASS)
                    train_prediction = model.predict(train.weights)
                    heldout_prediction = model.predict(heldout.weights)
                    complement = np.setdiff1d(np.arange(reference.n), selected, assume_unique=True)

                    record: dict[str, Any] = {
                        "target": target,
                        "sampling_design": design,
                        "sampling_design_label": DESIGN_LABELS[design],
                        "sample_size": sample_size,
                        "seed": effective_seed,
                        "subset_sha256": row_digest(selected),
                        "domain_deletion_rows": int(
                            reference.frame.iloc[selected]["panel_source"].eq("domain_deletion").sum()
                        ),
                        "qsplit_rows": int(reference.frame.iloc[selected]["panel_source"].eq("qsplit_signal").sum()),
                        "anchor_rows": int(reference.frame.iloc[selected]["run_name"].isin(BASELINE_NAMES).sum()),
                        "nominal_parameter_count": compact_retained.nominal_parameter_count(
                            train,
                            observatory.compact_config(POLICY_CLASS),
                        ),
                        "selected_l2": selected_l2,
                        "full_280_l2": full_l2,
                        "inner_cv_selected_rmse": float(cv_rows.loc[cv_rows["l2"].eq(selected_l2), "rmse"].iloc[0]),
                        "inner_cv_selected_spearman": float(
                            cv_rows.loc[cv_rows["l2"].eq(selected_l2), "spearman"].iloc[0]
                        ),
                        **shape_record(model, full_model),
                        **metric_prefix("train", sample_eff.metrics(train.y, train_prediction)),
                        **metric_prefix("heldout", sample_eff.metrics(heldout_dataset.y, heldout_prediction)),
                        "heldout_prediction_rmse_to_full_280_model": float(
                            np.sqrt(np.mean(np.square(heldout_prediction - full_heldout_prediction)))
                        ),
                    }
                    if matched.sum() >= 5:
                        record.update(
                            metric_prefix(
                                "target_matched_heldout",
                                sample_eff.metrics(heldout_dataset.y[matched], heldout_prediction[matched]),
                            )
                        )
                    else:
                        record.update(empty_metrics("target_matched_heldout"))
                    if len(complement):
                        complement_prediction = model.predict(reference.weights[complement])
                        record.update(
                            metric_prefix(
                                "fit_complement",
                                sample_eff.metrics(reference.y[complement], complement_prediction),
                            )
                        )
                    else:
                        record.update(empty_metrics("fit_complement"))
                    if not skip_optima:
                        optimum = sample_eff.optimize_raw_model(
                            train,
                            model,
                            full_spec,
                            seed=effective_seed,
                            count=optimizer_starts,
                            previous=full_optimum.weights,
                        )
                        record.update(raw_optimum_metrics(optimum, train, full_optimum))
                    rows.append(record)
                    complete.add(key)
                    persist_rows(path, rows)
                    print(
                        f"[{target}] {design} n={sample_size} seed={effective_seed} "
                        f"L2={selected_l2:g} heldout_rmse={record['heldout_rmse']:.6f} "
                        f"spearman={record['heldout_spearman']:.4f}",
                        flush=True,
                    )
    return pd.DataFrame(rows).sort_values(["target", "sampling_design", "sample_size", "seed"]).reset_index(drop=True)


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        column
        for column in runs.columns
        if column
        not in {
            "target",
            "sampling_design",
            "sampling_design_label",
            "sample_size",
            "seed",
            "subset_sha256",
            "raw_phase_0_weights_json",
            "raw_phase_1_weights_json",
        }
        and pd.api.types.is_numeric_dtype(runs[column])
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in runs.groupby(["target", "sampling_design", "sampling_design_label", "sample_size"], sort=True):
        target, design, label, sample_size = keys
        row: dict[str, Any] = {
            "target": target,
            "sampling_design": design,
            "sampling_design_label": label,
            "sample_size": sample_size,
            "replicates": len(group),
        }
        for column in metric_columns:
            values = pd.to_numeric(group[column], errors="coerce")
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_std"] = float(values.std(ddof=1)) if values.notna().sum() > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def add_band(
    figure: go.Figure,
    frame: pd.DataFrame,
    metric: str,
    design: str,
    row: int,
    col: int,
) -> None:
    values = frame[frame["sampling_design"].eq(design)].sort_values("sample_size")
    x = values["sample_size"].to_numpy()
    mean = values[f"{metric}_mean"].to_numpy(dtype=float)
    std = values[f"{metric}_std"].fillna(0.0).to_numpy(dtype=float)
    color = DESIGN_COLORS[design]
    red, green, blue = (int(color[offset : offset + 2], 16) for offset in (1, 3, 5))
    figure.add_trace(
        go.Scatter(
            x=x,
            y=mean,
            mode="lines+markers",
            name=DESIGN_LABELS[design],
            legendgroup=design,
            showlegend=row == 1 and col == 1,
            line={"color": color, "width": 3},
            marker={"size": 8},
            customdata=np.stack([std], axis=1),
            hovertemplate="n=%{x}<br>mean=%{y:.6f}<br>seed SD=%{customdata[0]:.6f}<extra></extra>",
        ),
        row=row,
        col=col,
    )
    if np.isfinite(std).any() and np.nanmax(std) > 0.0:
        figure.add_trace(
            go.Scatter(
                x=np.concatenate([x, x[::-1]]),
                y=np.concatenate([mean - std, (mean + std)[::-1]]),
                fill="toself",
                fillcolor=f"rgba({red}, {green}, {blue}, 0.13)",
                line={"width": 0},
                hoverinfo="skip",
                legendgroup=design,
                showlegend=False,
            ),
            row=row,
            col=col,
        )


def plot_metric_panel(
    summary: pd.DataFrame,
    output_dir: Path,
    metric_prefix: str,
    title: str,
    filename: str,
) -> None:
    targets = tuple(target for target in TARGETS if target in set(summary["target"]))
    metrics = (
        (f"{metric_prefix}_rmse", "RMSE"),
        (f"{metric_prefix}_spearman", "Spearman"),
        (f"{metric_prefix}_regret_at_1", "Regret@1"),
        (f"{metric_prefix}_calibration_slope", "Calibration slope"),
    )
    figure = make_subplots(
        rows=len(targets),
        cols=4,
        subplot_titles=[f"{target.title()} · {label}" for target in targets for _metric, label in metrics],
        horizontal_spacing=0.06,
        vertical_spacing=0.15,
    )
    for row, target in enumerate(targets, start=1):
        target_frame = summary[summary["target"].eq(target)]
        for col, (metric, _label) in enumerate(metrics, start=1):
            for design in SAMPLING_DESIGNS:
                add_band(figure, target_frame, metric, design, row, col)
            figure.update_xaxes(title_text="Fit rows", row=row, col=col)
    for row in range(1, len(targets) + 1):
        figure.update_yaxes(title_text="BPB", row=row, col=1)
        figure.update_yaxes(title_text="BPB", row=row, col=3)
    figure.update_layout(
        title=title,
        template="plotly_white",
        height=440 * len(targets),
        width=1760,
        hovermode="x unified",
        legend={"orientation": "h", "y": -0.09, "x": 0.5, "xanchor": "center"},
        margin={"l": 70, "r": 30, "t": 100, "b": 90},
    )
    figure.write_html(output_dir / filename, include_plotlyjs=True, config=PLOT_CONFIG)


def plot_metrics(summary: pd.DataFrame, output_dir: Path) -> None:
    plot_metric_panel(
        summary,
        output_dir,
        "fit_complement",
        "Compact retained-state sample efficiency: withheld original-swarm rows",
        "fit_complement_learning_curves.html",
    )
    plot_metric_panel(
        summary,
        output_dir,
        "heldout",
        "Compact retained-state sample efficiency: all fixed 3e18 development heldouts",
        "heldout_learning_curves.html",
    )
    plot_metric_panel(
        summary,
        output_dir,
        "target_matched_heldout",
        "Compact retained-state sample efficiency: target-matched proposed frontier heldouts",
        "target_matched_heldout_learning_curves.html",
    )


def plot_optima(summary: pd.DataFrame, output_dir: Path) -> None:
    if "raw_predicted_bpb_mean" not in summary:
        return
    targets = tuple(target for target in TARGETS if target in set(summary["target"]))
    metrics = (
        ("raw_policy_tv_to_full_280_optimum", "Policy TV to 280-row optimum"),
        ("raw_predicted_bpb", "Predicted raw-optimum BPB"),
        ("raw_max_simulated_epochs", "Maximum simulated epochs"),
        ("raw_standardized_fit_support_distance", "Fit-support distance"),
    )
    figure = make_subplots(
        rows=len(targets),
        cols=4,
        subplot_titles=[f"{target.title()} · {label}" for target in targets for _metric, label in metrics],
        horizontal_spacing=0.06,
        vertical_spacing=0.15,
    )
    for row, target in enumerate(targets, start=1):
        target_frame = summary[summary["target"].eq(target)]
        for col, (metric, _label) in enumerate(metrics, start=1):
            for design in SAMPLING_DESIGNS:
                add_band(figure, target_frame, metric, design, row, col)
            figure.update_xaxes(title_text="Fit rows", row=row, col=col)
    figure.update_layout(
        title="Compact retained-state raw optimum: convergence is not validation",
        template="plotly_white",
        height=440 * len(targets),
        width=1760,
        hovermode="x unified",
        legend={"orientation": "h", "y": -0.09, "x": 0.5, "xanchor": "center"},
        margin={"l": 70, "r": 30, "t": 100, "b": 90},
    )
    figure.write_html(output_dir / "raw_optimum_stability.html", include_plotlyjs=True, config=PLOT_CONFIG)


def convergence_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (target, design), group in summary.groupby(["target", "sampling_design"], sort=True):
        group = group.sort_values("sample_size")
        endpoint = group[group["sample_size"].eq(280)].iloc[0]
        rmse_limit = 1.05 * float(endpoint["heldout_rmse_mean"])
        spearman_limit = float(endpoint["heldout_spearman_mean"]) - 0.02
        regret_limit = float(endpoint["heldout_regret_at_1_mean"]) + 0.002
        eligible = group[
            group["heldout_rmse_mean"].le(rmse_limit)
            & group["heldout_spearman_mean"].ge(spearman_limit)
            & group["heldout_regret_at_1_mean"].le(regret_limit)
        ]
        earliest = int(eligible["sample_size"].min()) if len(eligible) else -1
        rows.append(
            {
                "target": target,
                "sampling_design": design,
                "earliest_fit_metric_convergence_rows": earliest,
                "endpoint_heldout_rmse": endpoint["heldout_rmse_mean"],
                "endpoint_heldout_spearman": endpoint["heldout_spearman_mean"],
                "endpoint_heldout_regret_at_1": endpoint["heldout_regret_at_1_mean"],
                "endpoint_calibration_slope": endpoint["heldout_calibration_slope_mean"],
                "endpoint_raw_support_distance": endpoint.get(
                    "raw_standardized_fit_support_distance_mean",
                    float("nan"),
                ),
            }
        )
    return pd.DataFrame(rows)


def write_report(
    output_dir: Path,
    runs: pd.DataFrame,
    summary: pd.DataFrame,
    convergence: pd.DataFrame,
) -> None:
    best_rows: list[dict[str, Any]] = []
    targets = tuple(target for target in TARGETS if target in set(summary["target"]))
    for target in targets:
        target_rows = summary[summary["target"].eq(target)]
        best = target_rows.loc[target_rows["heldout_rmse_mean"].idxmin()]
        best_rows.append(
            {
                "target": target,
                "best design": best["sampling_design"],
                "fit rows": int(best["sample_size"]),
                "heldout RMSE": best["heldout_rmse_mean"],
                "Spearman": best["heldout_spearman_mean"],
                "Regret@1": best["heldout_regret_at_1_mean"],
                "calibration slope": best["heldout_calibration_slope_mean"],
            }
        )
    endpoint = summary[summary["sample_size"].eq(280)].drop_duplicates(["target"])
    endpoint_columns = [
        "target",
        "heldout_rmse_mean",
        "heldout_spearman_mean",
        "heldout_regret_at_1_mean",
        "heldout_calibration_slope_mean",
        "heldout_optimism_gt_0p05_mean",
    ]
    endpoint_columns.extend(
        column
        for column in (
            "raw_predicted_bpb_mean",
            "raw_max_simulated_epochs_mean",
            "raw_standardized_fit_support_distance_mean",
        )
        if column in endpoint
    )
    target_matched_endpoint = endpoint[
        [
            "target",
            "target_matched_heldout_n_eval_mean",
            "target_matched_heldout_rmse_mean",
            "target_matched_heldout_spearman_mean",
            "target_matched_heldout_regret_at_1_mean",
            "target_matched_heldout_regret_at_3_mean",
            "target_matched_heldout_regret_at_5_mean",
            "target_matched_heldout_calibration_slope_mean",
            "target_matched_heldout_bias_mean",
        ]
    ]
    key_sizes = (80, 112, 184, 232, 280)
    key_rows = summary[summary["sample_size"].isin(key_sizes)][
        [
            "target",
            "sampling_design",
            "sample_size",
            "heldout_rmse_mean",
            "heldout_rmse_std",
            "heldout_spearman_mean",
            "heldout_spearman_std",
            "heldout_regret_at_1_mean",
            "heldout_regret_at_3_mean",
            "heldout_regret_at_5_mean",
            "raw_policy_tv_to_full_280_optimum_mean",
            "raw_standardized_fit_support_distance_mean",
        ]
    ]
    fit_complement_rows = summary[summary["sample_size"].isin((80, 112, 144, 184, 232))][
        [
            "target",
            "sampling_design",
            "sample_size",
            "fit_complement_rmse_mean",
            "fit_complement_spearman_mean",
            "fit_complement_regret_at_1_mean",
            "fit_complement_calibration_slope_mean",
        ]
    ]
    prior_validation_path = (
        SCRIPT_DIR / "reference_outputs/delphi_compact_optimum_path_validation_results_20260721/path_summary.csv"
    )
    prior_validation = pd.read_csv(prior_validation_path)[
        [
            "target",
            "best_observed_fit_rows",
            "best_observed_target_bpb",
            "predicted_best_fit_rows",
            "predicted_best_bpb",
            "predicted_best_observed_bpb",
            "predicted_best_optimism_bpb",
        ]
    ]
    lines = [
        "# Compact retained-state sample efficiency below 280 fit rows",
        "",
        "## Scope",
        "",
        "This frozen local audit asks how quickly the Compact retained-state fit approaches its 280-row behavior. "
        "It does not rehabilitate the raw optimum: the independently trained optimum-path panel already showed that "
        "the 280-row raw surface is substantially optimistic at deployment.",
        "",
        "Two nested sample designs are compared over five seeds. `panel_stratified` preserves the original 241:39 "
        "qsplit/deletion ratio. `intervention_core` always retains all 39 deletion interventions and the three "
        "proportional/UniMax/stratified anchors. L2 is selected from {0.1, 1.0} by inner CV on the selected rows only.",
        "",
        "## Verdict",
        "",
        "Compact retained state is moderately sample-efficient for interpolation inside the original 280-row "
        "swarm, but it is not sample-efficient for trustworthy optimum selection. Withheld original-swarm "
        "metrics become useful around 112--144 rows. Broad external-heldout metrics require roughly 232--280 "
        "rows, and the target-matched proposed-frontier subsets remain poorly ranked even at 280 rows.",
        "",
        "The fitted form has 45 nominal parameters, so the tested row-to-parameter ratios range from 1.07 at "
        "48 rows to 6.22 at 280 rows. Inner CV still switches between L2=0.1 and L2=1.0 across some 232-row "
        "subsamples, which is additional evidence that the finite-sample fit has not fully stabilized.",
        "",
        "Raw policies become visually closer to the 280-row endpoint near 232 rows, but this is convergence to "
        "an already falsified surface. The endpoint raw optima remain far outside empirical fit support, and "
        "independent 3e18 validation observed large optimism. Subsampling does not provide a reason to validate "
        "another unregularized raw-optimum path.",
        "",
        "## Lowest broad-heldout RMSE",
        "",
        "This table is descriptive, not a winner selection: a low pooled RMSE can coexist with poor frontier "
        "ranking, unstable seeds, or bad regret.",
        "",
        pd.DataFrame(best_rows).to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Convergence gate",
        "",
        "The descriptive fit-metric gate is the earliest row count within 5% of endpoint heldout RMSE, 0.02 of "
        "endpoint Spearman, and 0.002 BPB of endpoint Regret@1. It measures convergence to the 280-row fit, not "
        "truth of the raw optimum.",
        "",
        convergence.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## 280-row endpoint",
        "",
        endpoint[endpoint_columns].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Target-matched frontier heldouts at 280 rows",
        "",
        target_matched_endpoint.to_markdown(index=False, floatfmt=".6f"),
        "",
        "The negative target-matched Spearman values show that the strong pooled heldout ranks are driven by "
        "easy separation across the archive's wide response range. They do not demonstrate reliable ordering "
        "among mixtures proposed near the target frontier.",
        "",
        "## Selected learning-curve rows",
        "",
        key_rows.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Original-swarm fit complements",
        "",
        fit_complement_rows.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Independent raw-optimum validation",
        "",
        prior_validation.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Interpretation rule",
        "",
        "A smaller panel is sample-efficient only if fit metrics, calibration, and decision regret stabilize together. "
        "Raw-policy TV convergence is necessary but insufficient because every path can converge to the same biased "
        "surface. Another 3e18 raw-optimum validation is justified only by a materially new model or a predeclared "
        "deployment rule, not merely by finding the earliest subset that reproduces the failed 280-row optimum.",
        "",
        "## Artifacts",
        "",
        "- `learning_curve_runs.csv`: seed-level fits, parameters, decisions, and all metrics.",
        "- `learning_curve_summary.csv`: mean and standard deviation by target, design, and fit-row count.",
        "- `fit_complement_learning_curves.html`: interpolation within the original swarm.",
        "- `heldout_learning_curves.html`: broad external-heldout RMSE, rank, regret, and calibration.",
        "- `target_matched_heldout_learning_curves.html`: proposed-frontier heldouts for the fitted target.",
        "- `raw_optimum_stability.html`: policy, predicted value, exposure, and support convergence.",
        "- `protocol.json`: frozen data-use and fitting protocol.",
        "",
        f"The fixed heldout pool contains {int(runs['heldout_n_eval'].max())} coordinate-distinct two-phase policies.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    sample_sizes = parse_ints(args.sample_sizes)
    seeds = parse_ints(args.seeds)
    targets = parse_values(args.targets)
    if sample_sizes[-1] != 280 or min(sample_sizes) < 48 or tuple(sorted(set(sample_sizes))) != sample_sizes:
        raise ValueError("Sample sizes must be unique, increasing, at least 48, and end at 280")
    if not set(targets).issubset(TARGETS):
        raise ValueError(f"Targets must be a subset of {TARGETS}")
    protocol = protocol_payload(sample_sizes, seeds, targets)
    persist_protocol(args.output_dir, protocol, args.overwrite)
    if args.rebuild_only:
        runs = pd.read_csv(args.output_dir / "learning_curve_runs.csv")
    else:
        runs = run_analysis(
            args.output_dir,
            targets,
            sample_sizes,
            seeds,
            args.optimizer_starts,
            args.overwrite,
            args.skip_optima,
        )
    summary = summarize(runs)
    summary.to_csv(args.output_dir / "learning_curve_summary.csv", index=False)
    convergence = convergence_table(summary)
    convergence.to_csv(args.output_dir / "convergence_summary.csv", index=False)
    plot_metrics(summary, args.output_dir)
    plot_optima(summary, args.output_dir)
    write_report(args.output_dir, runs, summary, convergence)
    print(f"Wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
