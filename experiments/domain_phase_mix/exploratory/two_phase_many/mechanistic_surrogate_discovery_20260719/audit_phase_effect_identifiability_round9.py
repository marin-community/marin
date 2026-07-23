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
"""Quantify how much phase-effect structure is identifiable from matched fit panels."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    audit_joint_latent_phase_transport_round8 as round8,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719/round9_phase_identifiability"
)
SEED = 20260719
N_SPLITS = 5
INNER_SPLITS = 4
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


@dataclass(frozen=True)
class ModelSpec:
    family: str
    hyperparameter: float

    @property
    def key(self) -> str:
        return f"{self.family}:{self.hyperparameter:g}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def feature_sets() -> tuple[dict[str, np.ndarray], np.ndarray, tuple[str, ...]]:
    panel, indices, target, _panels = round8.aligned_data()
    weights = panel.weights[indices]
    aggregate = panel.alpha0 * weights[:, 0, :] + panel.alpha1 * weights[:, 1, :]
    prior = np.maximum(panel.proportional_weights[None, :], 1e-12)
    relative_aggregate = aggregate / prior
    relative_contrast = panel.alpha0 * panel.alpha1 * (weights[:, 1, :] - weights[:, 0, :]) / prior
    transported = relative_contrast / (1.0 + relative_aggregate)

    family_aggregate = []
    family_contrast = []
    family_energy = []
    for members in panel.family_members:
        mass = panel.proportional_weights[members]
        mass = mass / mass.sum()
        family_aggregate.append(relative_aggregate[:, members] @ mass)
        family_contrast.append(relative_contrast[:, members] @ mass)
        family_energy.append((relative_contrast[:, members] ** 2) @ mass)

    phase_tv = 0.5 * np.sum(np.abs(weights[:, 1, :] - weights[:, 0, :]), axis=1)
    aggregate_hhi = np.sum(aggregate**2, axis=1)
    max_relative_aggregate = np.max(relative_aggregate, axis=1)
    max_relative_contrast = np.max(np.abs(relative_contrast), axis=1)
    physical_summary = np.column_stack(
        [
            *family_aggregate,
            *family_contrast,
            *family_energy,
            phase_tv,
            aggregate_hhi,
            max_relative_aggregate,
            max_relative_contrast,
        ]
    )
    raw = np.column_stack([relative_aggregate, relative_contrast])
    second_order = np.column_stack(
        [
            relative_aggregate,
            relative_contrast,
            transported,
            relative_aggregate * relative_contrast,
            relative_contrast**2,
        ]
    )
    return (
        {
            "family_physics": physical_summary,
            "raw_aggregate_contrast": raw,
            "second_order_physics": second_order,
        },
        target,
        panel.domain_names,
    )


def model_specs(family: str) -> list[ModelSpec]:
    if family == "ridge":
        return [ModelSpec(family, value) for value in (0.01, 0.1, 1.0, 10.0, 100.0)]
    if family == "pls":
        return [ModelSpec(family, float(value)) for value in (1, 2, 3, 4, 6, 8)]
    if family == "extra_trees":
        return [ModelSpec(family, float(value)) for value in (2, 4, 8, 16)]
    raise ValueError(f"Unknown diagnostic family {family}")


def fit_predict(spec: ModelSpec, train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    if spec.family == "ridge":
        model = make_pipeline(StandardScaler(), Ridge(alpha=spec.hyperparameter))
    elif spec.family == "pls":
        components = min(int(spec.hyperparameter), train_x.shape[1], train_y.shape[1], len(train_x) - 1)
        model = make_pipeline(StandardScaler(), PLSRegression(n_components=components, scale=False, max_iter=1000))
    elif spec.family == "extra_trees":
        model = ExtraTreesRegressor(
            n_estimators=400,
            min_samples_leaf=int(spec.hyperparameter),
            max_features=0.7,
            random_state=SEED,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown diagnostic family {spec.family}")
    model.fit(train_x, train_y)
    return np.asarray(model.predict(test_x), dtype=float)


def normalized_rmse(observed: np.ndarray, predicted: np.ndarray, scale: np.ndarray) -> float:
    error = np.sqrt(np.mean((predicted - observed) ** 2, axis=0))
    return float(np.mean(error / np.maximum(scale, 1e-12)))


def select_spec(
    family: str,
    features: np.ndarray,
    target: np.ndarray,
    outer_train: np.ndarray,
    seed: int,
) -> tuple[ModelSpec, float]:
    scale = np.std(target[outer_train], axis=0, ddof=1)
    folds = KFold(INNER_SPLITS, shuffle=True, random_state=seed).split(outer_train)
    fold_indices = list(folds)
    rows = []
    for spec in model_specs(family):
        prediction = np.full_like(target, np.nan)
        for train_local, test_local in fold_indices:
            train = outer_train[train_local]
            test = outer_train[test_local]
            prediction[test] = fit_predict(spec, features[train], target[train], features[test])
        score = normalized_rmse(target[outer_train], prediction[outer_train], scale)
        rows.append((score, spec.key, spec))
    score, _key, selected = min(rows)
    return selected, score


def scalar_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = predicted - observed
    slope = float(np.polyfit(predicted, observed, 1)[0]) if np.std(predicted) > 1e-12 else np.nan
    return {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "normalized_rmse": float(np.sqrt(np.mean(residual**2)) / np.std(observed, ddof=1)),
        "r2": float(1.0 - np.sum(residual**2) / np.sum((observed - observed.mean()) ** 2)),
        "spearman": float(spearmanr(observed, predicted).statistic),
        "sign_accuracy": float(np.mean(np.sign(observed) == np.sign(predicted))),
        "calibration_slope": slope,
        "bias": float(np.mean(residual)),
    }


def nested_coordinate_oof(
    feature_name: str,
    family: str,
    features: np.ndarray,
    target: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction = np.full_like(target, np.nan)
    selection_rows = []
    for fold, (train, test) in enumerate(KFold(N_SPLITS, shuffle=True, random_state=SEED).split(features)):
        selected, score = select_spec(family, features, target, train, SEED + fold + 1)
        prediction[test] = fit_predict(selected, features[train], target[train], features[test])
        selection_rows.append(
            {
                "feature_set": feature_name,
                "diagnostic_family": family,
                "fold": fold,
                "selected_spec": selected.key,
                "inner_normalized_rmse": score,
            }
        )
    rows = []
    prediction_rows = []
    for column, output in enumerate(round8.OUTPUT_LABELS):
        rows.append(
            {
                "feature_set": feature_name,
                "diagnostic_family": family,
                "output": output,
                **scalar_metrics(target[:, column], prediction[:, column]),
            }
        )
        for index in range(len(target)):
            prediction_rows.append(
                {
                    "feature_set": feature_name,
                    "diagnostic_family": family,
                    "coordinate": index,
                    "output": output,
                    "observed_delta": target[index, column],
                    "predicted_delta": prediction[index, column],
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(selection_rows), pd.DataFrame(prediction_rows)


def leave_region_out(
    feature_name: str,
    family: str,
    features: np.ndarray,
    target: np.ndarray,
    selected_key: str,
) -> pd.DataFrame:
    spec = next(spec for spec in model_specs(family) if spec.key == selected_key)
    standardized = StandardScaler().fit_transform(features)
    regions = KMeans(n_clusters=5, random_state=SEED, n_init=50).fit_predict(standardized)
    prediction = np.full_like(target, np.nan)
    for region in range(5):
        test = np.flatnonzero(regions == region)
        train = np.flatnonzero(regions != region)
        prediction[test] = fit_predict(spec, features[train], target[train], features[test])
    rows = []
    for column, output in enumerate(round8.OUTPUT_LABELS):
        rows.append(
            {
                "feature_set": feature_name,
                "diagnostic_family": family,
                "output": output,
                **scalar_metrics(target[:, column], prediction[:, column]),
            }
        )
    return pd.DataFrame(rows)


def learning_curves(
    feature_name: str,
    family: str,
    features: np.ndarray,
    target: np.ndarray,
    selected_key: str,
) -> pd.DataFrame:
    spec = next(spec for spec in model_specs(family) if spec.key == selected_key)
    rng = np.random.default_rng(SEED)
    rows = []
    for train_size in (40, 80, 120, 160, 200):
        for repeat in range(20):
            permutation = rng.permutation(len(features))
            train = permutation[:train_size]
            test = permutation[train_size:]
            prediction = fit_predict(spec, features[train], target[train], features[test])
            for column, output in enumerate(round8.OUTPUT_LABELS):
                metrics = scalar_metrics(target[test, column], prediction[:, column])
                rows.append(
                    {
                        "feature_set": feature_name,
                        "diagnostic_family": family,
                        "train_size": train_size,
                        "repeat": repeat,
                        "output": output,
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def render_learning_curves(frame: pd.DataFrame, output: Path) -> None:
    summary = frame.groupby(["diagnostic_family", "output", "train_size"], as_index=False).agg(
        rmse=("rmse", "mean"), rmse_sd=("rmse", "std")
    )
    fig = make_subplots(rows=2, cols=2, subplot_titles=round8.OUTPUT_LABELS)
    colors = {"ridge": "#2166ac", "pls": "#fdae61", "extra_trees": "#1a9850"}
    for index, output_name in enumerate(round8.OUTPUT_LABELS):
        row = index // 2 + 1
        col = index % 2 + 1
        subset = summary.loc[summary["output"].eq(output_name)]
        for family, group in subset.groupby("diagnostic_family"):
            fig.add_trace(
                go.Scatter(
                    x=group["train_size"],
                    y=group["rmse"],
                    error_y={"type": "data", "array": group["rmse_sd"]},
                    mode="lines+markers",
                    name=family,
                    legendgroup=family,
                    showlegend=index == 0,
                    line={"color": colors[family]},
                ),
                row=row,
                col=col,
            )
        fig.update_xaxes(title_text="Training coordinates", row=row, col=col)
        fig.update_yaxes(title_text="Phase-delta RMSE", row=row, col=col)
    fig.update_layout(title="Matched phase-effect learning curves", template="plotly_white", height=900, width=1400)
    fig.write_html(output, include_plotlyjs="cdn", config=PLOT_CONFIG)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_map, target, _domain_names = feature_sets()

    metric_tables = []
    selection_tables = []
    prediction_tables = []
    region_tables = []
    learning_tables = []
    selected_specs: dict[tuple[str, str], str] = {}
    combinations = (
        ("family_physics", "ridge"),
        ("raw_aggregate_contrast", "pls"),
        ("second_order_physics", "ridge"),
        ("second_order_physics", "extra_trees"),
    )
    for feature_name, family in combinations:
        metrics, selections, predictions = nested_coordinate_oof(
            feature_name,
            family,
            feature_map[feature_name],
            target,
        )
        metric_tables.append(metrics)
        selection_tables.append(selections)
        prediction_tables.append(predictions)
        selected_key = Counter(selections["selected_spec"]).most_common(1)[0][0]
        selected_specs[(feature_name, family)] = selected_key
        region_tables.append(leave_region_out(feature_name, family, feature_map[feature_name], target, selected_key))
        learning_tables.append(learning_curves(feature_name, family, feature_map[feature_name], target, selected_key))

    metrics = pd.concat(metric_tables, ignore_index=True)
    selections = pd.concat(selection_tables, ignore_index=True)
    predictions = pd.concat(prediction_tables, ignore_index=True)
    regions = pd.concat(region_tables, ignore_index=True)
    learning = pd.concat(learning_tables, ignore_index=True)
    metrics.to_csv(args.output_dir / "nested_oof_metrics.csv", index=False)
    selections.to_csv(args.output_dir / "nested_selections.csv", index=False)
    predictions.to_csv(args.output_dir / "nested_oof_predictions.csv", index=False)
    regions.to_csv(args.output_dir / "leave_region_out_metrics.csv", index=False)
    learning.to_csv(args.output_dir / "learning_curve_runs.csv", index=False)
    render_learning_curves(learning, args.output_dir / "learning_curves.html")

    learning_summary = learning.groupby(
        ["feature_set", "diagnostic_family", "output", "train_size"], as_index=False
    ).agg(rmse_mean=("rmse", "mean"), rmse_sd=("rmse", "std"), spearman_mean=("spearman", "mean"))
    learning_summary.to_csv(args.output_dir / "learning_curve_summary.csv", index=False)
    selected_table = pd.DataFrame(
        [
            {"feature_set": feature, "diagnostic_family": family, "modal_selected_spec": key}
            for (feature, family), key in selected_specs.items()
        ]
    )
    selected_table.to_csv(args.output_dir / "modal_selected_specs.csv", index=False)

    report = [
        "# Phase-effect identifiability audit",
        "",
        "This is a diagnostic upper-bound study, not a surrogate-model proposal. It reads only the 238 coordinate-matched 300M/Delphi fit-panel pairs and never reads historical or adversarial heldout targets.",
        "",
        "All four target-scale phase deltas at a coordinate are held out together. Flexible trees are included only to test whether a smooth nonlinear signal is recoverable; they are inadmissible as the requested headline surrogate.",
        "",
        "## Nested coordinate OOF",
        "",
        metrics.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Leave-region-out",
        "",
        regions.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Modal diagnostic hyperparameters",
        "",
        selected_table.to_markdown(index=False),
        "",
        "## Learning curves",
        "",
        learning_summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "The gap between random-coordinate OOF and leave-region-out is the key identification diagnostic. A flexible learner that succeeds only under interpolation does not justify a more elaborate mechanistic extrapolator.",
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report))
    (args.output_dir / "audit_manifest.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "coordinate_count": len(target),
                "outputs": round8.OUTPUT_LABELS,
                "adversarial_targets_read": False,
                "selected_specs": {f"{key[0]}::{key[1]}": value for key, value in selected_specs.items()},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
