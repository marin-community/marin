# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "pandas",
#     "plotly",
#     "scikit-learn",
#     "scipy",
# ]
# ///
"""Test whether auxiliary smooth metrics improve heldout DCLM prediction.

This is a diagnostic follow-up to the DCLM noise-floor audit. It does not
optimize a mixture. It compares DCLM-only smooth proxies, auxiliary smooth
metrics, and a general-factor decoy under row-heldout regression against DCLM
hard targets.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import pearsonr, spearmanr
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MATRIX_CSV = (
    SCRIPT_DIR
    / "reference_outputs"
    / "raw_metric_matrix_300m_dclm_updated_20260615"
    / "raw_metric_matrix_300m_with_proportional_noise.csv"
)
DEFAULT_COMPONENT_SUMMARY_CSV = (
    SCRIPT_DIR
    / "reference_outputs"
    / "raw_metric_matrix_300m_dclm_updated_20260615"
    / "dclm_component_smooth_proxy_summary.csv"
)
DEFAULT_NOISE_AUDIT_CSV = (
    SCRIPT_DIR
    / "reference_outputs"
    / "dclm_calibrated_auxiliary_noise_floor_20260616"
    / "component_noise_reliability.csv"
)
DEFAULT_OUTPUT_DIR = (
    SCRIPT_DIR
    / "reference_outputs"
    / "dclm_calibrated_auxiliary_anchored_regression_20260616"
)
MACRO_COLUMN = "lm_eval/dclm_core/centered_accuracy_macro"
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}
RIDGE_ALPHAS = np.logspace(-2, 8, 31)
SMOOTH_SUFFIXES = (
    "/bpb",
    "/choice_logprob",
    "/choice_logprob_norm",
    "/choice_prob_norm",
    "/native_gold_bpb",
    "/native_gold_logprob_per_byte",
    "/native_margin_per_byte",
    "/native_choice_prob",
)


@dataclass(frozen=True)
class FeatureGroup:
    """Feature matrix and names."""

    name: str
    values: np.ndarray
    columns: list[str]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-csv", type=Path, default=DEFAULT_MATRIX_CSV)
    parser.add_argument("--component-summary-csv", type=Path, default=DEFAULT_COMPONENT_SUMMARY_CSV)
    parser.add_argument("--noise-audit-csv", type=Path, default=DEFAULT_NOISE_AUDIT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--exclude-run-name", action="append", default=["baseline_stratified"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260616)
    return parser.parse_args()


def utility_from_column(frame: pd.DataFrame, column: str, direction: str | None = None) -> pd.Series:
    """Convert a smooth metric column into a higher-is-better utility."""
    values = pd.to_numeric(frame[column], errors="coerce")
    if direction is not None:
        if direction == "lower":
            return -values
        if direction == "higher":
            return values
        raise ValueError(f"Unknown smooth direction: {direction}")
    if column.endswith(("/bpb", "/loss", "/perplexity", "/native_gold_bpb")):
        return -values
    return values


def is_complete_smooth_column(frame: pd.DataFrame, column: str) -> bool:
    """Return whether a column is a complete, nonconstant smooth-like metric."""
    if column.startswith(("phase_0_", "phase_1_")):
        return False
    if column.startswith("lm_eval/dclm_core/"):
        return False
    if not column.endswith(SMOOTH_SUFFIXES):
        return False
    values = pd.to_numeric(frame[column], errors="coerce")
    return bool(values.notna().all() and float(values.std(ddof=0)) > 0.0)


def dclm_task_prefixes(components: pd.DataFrame) -> tuple[str, ...]:
    """Return column prefixes for DCLM component-specific smooth metrics."""
    aliases = [str(alias) for alias in components["component_alias"].tolist()]
    return tuple(f"lm_eval/{alias}/" for alias in aliases)


def complete_smooth_columns(frame: pd.DataFrame) -> list[str]:
    """Return complete, nonconstant smooth-like columns."""
    columns: list[str] = []
    for column in frame.columns:
        if is_complete_smooth_column(frame, column):
            columns.append(column)
    return columns


def build_feature_groups(
    frame: pd.DataFrame,
    components: pd.DataFrame,
) -> list[FeatureGroup]:
    """Build DCLM-only, auxiliary-only, and combined smooth feature groups."""
    dclm_values = []
    dclm_columns = []
    for row in components.itertuples(index=False):
        column = str(row.selected_smooth_proxy_column)
        direction = str(row.selected_smooth_direction)
        dclm_values.append(utility_from_column(frame, column, direction=direction).to_numpy(dtype=float))
        dclm_columns.append(column)
    dclm_matrix = np.column_stack(dclm_values)
    all_smooth_columns = complete_smooth_columns(frame)
    dclm_prefixes = dclm_task_prefixes(components)
    dclm_all_columns = [column for column in all_smooth_columns if column.startswith(dclm_prefixes)]
    non_dclm_auxiliary_columns = [column for column in all_smooth_columns if not column.startswith(dclm_prefixes)]
    dclm_all_values = [utility_from_column(frame, column).to_numpy(dtype=float) for column in dclm_all_columns]
    non_dclm_auxiliary_values = [
        utility_from_column(frame, column).to_numpy(dtype=float) for column in non_dclm_auxiliary_columns
    ]
    dclm_all_matrix = np.column_stack(dclm_all_values) if dclm_all_values else np.empty((len(frame), 0))
    non_dclm_auxiliary_matrix = (
        np.column_stack(non_dclm_auxiliary_values) if non_dclm_auxiliary_values else np.empty((len(frame), 0))
    )
    return [
        FeatureGroup("dclm_selected_smooth_22", dclm_matrix, dclm_columns),
        FeatureGroup("dclm_all_complete_smooth", dclm_all_matrix, dclm_all_columns),
        FeatureGroup("non_dclm_auxiliary_smooth", non_dclm_auxiliary_matrix, non_dclm_auxiliary_columns),
        FeatureGroup(
            "dclm_selected_plus_non_dclm_auxiliary",
            np.column_stack([dclm_matrix, non_dclm_auxiliary_matrix]),
            dclm_columns + non_dclm_auxiliary_columns,
        ),
        FeatureGroup(
            "dclm_all_plus_non_dclm_auxiliary",
            np.column_stack([dclm_all_matrix, non_dclm_auxiliary_matrix]),
            dclm_all_columns + non_dclm_auxiliary_columns,
        ),
    ]


def zscore_columns(values: np.ndarray) -> np.ndarray:
    """Z-score columns with finite safeguards."""
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0)
    std = np.where(std > 0.0, std, 1.0)
    return (values - mean) / std


def target_columns(components: pd.DataFrame) -> list[str]:
    """Return DCLM hard component columns in component order."""
    return [str(column) for column in components["hard_centered_accuracy_column"].tolist()]


def build_targets(frame: pd.DataFrame, components: pd.DataFrame, noise_audit: pd.DataFrame) -> dict[str, np.ndarray]:
    """Build official and reliability-aware hard DCLM targets."""
    hard_columns = target_columns(components)
    hard = frame[hard_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    reliability_by_label = {
        str(row.label): float(row.reliability_proxy)
        for row in noise_audit.itertuples(index=False)
        if np.isfinite(float(row.reliability_proxy))
    }
    labels = [str(label) for label in components["component_alias"].tolist()]
    reliability = np.asarray([reliability_by_label.get(label, 0.0) for label in labels], dtype=float)
    reliability = np.where(np.isfinite(reliability) & (reliability > 0.0), reliability, 0.0)
    reliable_mask = reliability >= 0.5
    if not reliable_mask.any():
        raise ValueError("No reliability-filtered components available")
    weighted = np.average(hard[:, reliability > 0.0], axis=1, weights=reliability[reliability > 0.0])
    return {
        "hard_macro_official": pd.to_numeric(frame[MACRO_COLUMN], errors="coerce").to_numpy(dtype=float),
        "hard_macro_reliability_weighted": weighted,
        "hard_macro_reliability_ge_0p5": np.nanmean(hard[:, reliable_mask], axis=1),
    }


def finite_metric(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    """Compute a prediction metric with finite guards."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(mask.sum()) < 3:
        return math.nan
    true = y_true[mask]
    pred = y_pred[mask]
    if np.std(true) == 0.0 or np.std(pred) == 0.0:
        if metric in {"pearson", "spearman"}:
            return math.nan
    if metric == "pearson":
        return float(pearsonr(true, pred).statistic)
    if metric == "spearman":
        return float(spearmanr(true, pred).statistic)
    if metric == "r2":
        return float(r2_score(true, pred))
    if metric == "rmse":
        return float(math.sqrt(mean_squared_error(true, pred)))
    raise ValueError(f"Unknown metric: {metric}")


def fit_oof_predictions(model: Pipeline, x: np.ndarray, y: np.ndarray, folds: int, repeats: int, seed: int) -> list[dict[str, float]]:
    """Fit repeated K-fold predictions and return per-repeat metrics."""
    results = []
    for repeat in range(repeats):
        splitter = KFold(n_splits=folds, shuffle=True, random_state=seed + repeat)
        predictions = np.full(len(y), np.nan, dtype=float)
        for train_idx, test_idx in splitter.split(x):
            fold_model = clone(model)
            fold_model.fit(x[train_idx], y[train_idx])
            pred = fold_model.predict(x[test_idx])
            predictions[test_idx] = np.asarray(pred).reshape(-1)
        results.append(
            {
                "repeat": float(repeat),
                "oof_spearman": finite_metric(y, predictions, "spearman"),
                "oof_pearson": finite_metric(y, predictions, "pearson"),
                "oof_r2": finite_metric(y, predictions, "r2"),
                "oof_rmse": finite_metric(y, predictions, "rmse"),
            }
        )
    return results


def model_specs(feature_group: FeatureGroup) -> list[tuple[str, Pipeline]]:
    """Return model specs appropriate for a feature group."""
    if feature_group.values.shape[1] == 0:
        return []
    max_components = min(feature_group.values.shape[1], 10)
    specs: list[tuple[str, Pipeline]] = [
        (
            "ridge",
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("ridge", RidgeCV(alphas=RIDGE_ALPHAS)),
                ]
            ),
        ),
    ]
    for components in [1, 3, 5, 10]:
        if components <= max_components:
            specs.append(
                (
                    f"pca{components}_ridge",
                    Pipeline(
                        [
                            ("scale", StandardScaler()),
                            ("pca", PCA(n_components=components, random_state=0)),
                            ("ridge", RidgeCV(alphas=RIDGE_ALPHAS)),
                        ]
                    ),
                )
            )
            specs.append(
                (
                    f"pls{components}",
                    Pipeline(
                        [
                            ("scale", StandardScaler()),
                            ("pls", PLSRegression(n_components=components, scale=True)),
                        ]
                    ),
                )
            )
    return specs


def direct_macro_scores(feature_groups: list[FeatureGroup], targets: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    """Score unsupervised smooth macro baselines against targets."""
    rows = []
    dclm = next(group for group in feature_groups if group.name == "dclm_selected_smooth_22")
    dclm_macro = np.nanmean(zscore_columns(dclm.values), axis=1)
    for target_name, target in targets.items():
        rows.append(
            {
                "target": target_name,
                "feature_group": "dclm_selected_smooth_22",
                "model": "unsupervised_z_macro",
                "repeat": 0.0,
                "oof_spearman": finite_metric(target, dclm_macro, "spearman"),
                "oof_pearson": finite_metric(target, dclm_macro, "pearson"),
                "oof_r2": finite_metric(target, dclm_macro, "r2"),
                "oof_rmse": finite_metric(target, dclm_macro, "rmse"),
                "feature_count": int(dclm.values.shape[1]),
            }
        )
    return rows


def summarize_results(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Aggregate repeated-CV metric rows."""
    frame = pd.DataFrame(rows)
    group_cols = ["target", "feature_group", "model", "feature_count"]
    metric_cols = ["oof_spearman", "oof_pearson", "oof_r2", "oof_rmse"]
    summaries = []
    for key, group in frame.groupby(group_cols, dropna=False):
        item = dict(zip(group_cols, key, strict=True))
        for metric in metric_cols:
            item[f"{metric}_mean"] = float(group[metric].mean())
            item[f"{metric}_std"] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
        item["repeat_count"] = int(len(group))
        summaries.append(item)
    return pd.DataFrame(summaries)


def write_outputs(output_dir: Path, summary: pd.DataFrame, metadata: dict[str, Any]) -> None:
    """Write CSV, plots, README, and JSON metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summary.sort_values(["target", "oof_spearman_mean"], ascending=[True, False])
    summary.to_csv(output_dir / "anchored_regression_summary.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    fig = px.bar(
        summary,
        x="model",
        y="oof_spearman_mean",
        error_y="oof_spearman_std",
        color="feature_group",
        facet_row="target",
        hover_data=["oof_pearson_mean", "oof_r2_mean", "oof_rmse_mean", "feature_count"],
        title="Heldout DCLM target prediction: DCLM-only vs auxiliary smooth features",
        labels={"oof_spearman_mean": "Repeated-CV OOF Spearman", "model": "model"},
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    fig.update_layout(template="plotly_white", height=900)
    fig.write_html(output_dir / "anchored_regression_oof_spearman.html", include_plotlyjs="cdn", config=PLOT_CONFIG)
    best = summary.sort_values("oof_spearman_mean", ascending=False).groupby("target").head(6)
    readme = [
        "# DCLM-Calibrated Auxiliary Regression Diagnostic",
        "",
        "This diagnostic compares DCLM-only smooth proxies, auxiliary smooth metrics, and low-rank latent decoys for heldout prediction of hard DCLM targets. It is not a mixture optimizer.",
        "",
        "## Best Models By Target",
        "",
        best[
            [
                "target",
                "feature_group",
                "model",
                "feature_count",
                "oof_spearman_mean",
                "oof_spearman_std",
                "oof_r2_mean",
            ]
        ].to_markdown(index=False),
        "",
        "## Interpretation Rules",
        "",
        "- If auxiliary features do not beat DCLM-only features on heldout hard-DCLM targets, do not use them as an optimization objective.",
        "- If PC1/PCA decoys match or beat supervised DCLM models, the result is likely a generic capability factor rather than DCLM-specific signal.",
        "- These repeated K-fold numbers are still diagnostics, not confirmatory validation; future DSP fitting must refit nonlinear parameters inside folds.",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(readme))


def main() -> None:
    """Run anchored regression diagnostics."""
    args = parse_args()
    matrix = pd.read_csv(args.matrix_csv, low_memory=False)
    components = pd.read_csv(args.component_summary_csv)
    noise_audit = pd.read_csv(args.noise_audit_csv)
    signal = matrix.loc[matrix["row_kind"].eq("signal") & ~matrix["run_name"].isin(args.exclude_run_name)].copy()
    feature_groups = build_feature_groups(signal, components)
    targets = build_targets(signal, components, noise_audit)
    rows: list[dict[str, Any]] = []
    rows.extend(direct_macro_scores(feature_groups, targets))
    for target_name, y in targets.items():
        finite_target = np.isfinite(y)
        for group in feature_groups:
            x = group.values[finite_target]
            y_subset = y[finite_target]
            for model_name, model in model_specs(group):
                repeat_rows = fit_oof_predictions(model, x, y_subset, args.folds, args.repeats, args.seed)
                for row in repeat_rows:
                    row.update(
                        {
                            "target": target_name,
                            "feature_group": group.name,
                            "model": model_name,
                            "feature_count": int(group.values.shape[1]),
                        }
                    )
                    rows.append(row)
    summary = summarize_results(rows)
    metadata = {
        "matrix_csv": str(args.matrix_csv),
        "component_summary_csv": str(args.component_summary_csv),
        "noise_audit_csv": str(args.noise_audit_csv),
        "output_dir": str(args.output_dir),
        "signal_rows": int(len(signal)),
        "folds": int(args.folds),
        "repeats": int(args.repeats),
        "feature_groups": {group.name: len(group.columns) for group in feature_groups},
        "targets": sorted(targets),
    }
    write_outputs(args.output_dir, summary, metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    print(summary.sort_values(["target", "oof_spearman_mean"], ascending=[True, False]).head(30).to_string(index=False))


if __name__ == "__main__":
    main()
