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
"""Leave-task-out DCLM component prediction diagnostic.

This tests whether DCLM smooth-hard coupling survives when predicting a
component's hard score without any smooth features from that same task. It is a
fallback for the cleaner eval-example-split test when per-example logs are not
available locally.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px

from experiments.domain_phase_mix.exploratory.two_phase_many.analyze_dclm_auxiliary_anchored_regression import (
    FeatureGroup,
    PLOT_CONFIG,
    complete_smooth_columns,
    dclm_task_prefixes,
    fit_oof_predictions,
    model_specs,
    utility_from_column,
)

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
DEFAULT_OUTPUT_DIR = (
    SCRIPT_DIR
    / "reference_outputs"
    / "dclm_leave_task_out_prediction_20260616"
)
MODELS_TO_RUN = {"ridge", "pca5_ridge", "pls3"}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-csv", type=Path, default=DEFAULT_MATRIX_CSV)
    parser.add_argument("--component-summary-csv", type=Path, default=DEFAULT_COMPONENT_SUMMARY_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--exclude-run-name", action="append", default=["baseline_stratified"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260616)
    return parser.parse_args()


def matrix_for_columns(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Build a higher-is-better utility matrix from smooth metric columns."""
    if not columns:
        return np.empty((len(frame), 0))
    values = [utility_from_column(frame, column).to_numpy(dtype=float) for column in columns]
    return np.column_stack(values)


def component_feature_groups(
    frame: pd.DataFrame,
    components: pd.DataFrame,
    target_alias: str,
) -> dict[str, tuple[np.ndarray, list[str]]]:
    """Build same-task, other-task, non-DCLM, and combined feature groups."""
    all_smooth = complete_smooth_columns(frame)
    dclm_prefixes = dclm_task_prefixes(components)
    target_prefix = f"lm_eval/{target_alias}/"
    same_task = [column for column in all_smooth if column.startswith(target_prefix)]
    other_dclm = [
        column for column in all_smooth if column.startswith(dclm_prefixes) and not column.startswith(target_prefix)
    ]
    non_dclm = [column for column in all_smooth if not column.startswith(dclm_prefixes)]
    return {
        "same_task_smooth_upper_bound": (matrix_for_columns(frame, same_task), same_task),
        "other_dclm_task_smooth": (matrix_for_columns(frame, other_dclm), other_dclm),
        "non_dclm_auxiliary_smooth": (matrix_for_columns(frame, non_dclm), non_dclm),
        "other_dclm_plus_non_dclm": (matrix_for_columns(frame, other_dclm + non_dclm), other_dclm + non_dclm),
    }


def run_component_models(
    *,
    frame: pd.DataFrame,
    components: pd.DataFrame,
    folds: int,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    """Run repeated-CV component prediction models."""
    rows: list[dict[str, Any]] = []
    for component in components.itertuples(index=False):
        alias = str(component.component_alias)
        target_column = str(component.hard_centered_accuracy_column)
        y = pd.to_numeric(frame[target_column], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(y)
        if int(finite.sum()) < folds:
            continue
        feature_groups = component_feature_groups(frame, components, alias)
        for group_name, (x_all, columns) in feature_groups.items():
            if x_all.shape[1] == 0:
                continue
            x = x_all[finite]
            y_subset = y[finite]
            for model_name, model in model_specs(FeatureGroup(group_name, x, columns)):
                if model_name not in MODELS_TO_RUN:
                    continue
                repeat_rows = fit_oof_predictions(model, x, y_subset, folds, repeats, seed)
                for row in repeat_rows:
                    row.update(
                        {
                            "component_alias": alias,
                            "target_column": target_column,
                            "feature_group": group_name,
                            "model": model_name,
                            "feature_count": int(x.shape[1]),
                        }
                    )
                    rows.append(row)
    return pd.DataFrame(rows)


def summarize(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize per-component and aggregate repeated-CV metrics."""
    metric_cols = ["oof_spearman", "oof_pearson", "oof_r2", "oof_rmse"]
    per_component = (
        rows.groupby(["component_alias", "feature_group", "model", "feature_count"], dropna=False)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    per_component.columns = [
        "_".join(part for part in column if part) if isinstance(column, tuple) else column
        for column in per_component.columns
    ]
    aggregate = (
        per_component.groupby(["feature_group", "model"], dropna=False)
        .agg(
            component_count=("component_alias", "count"),
            median_spearman=("oof_spearman_mean", "median"),
            mean_spearman=("oof_spearman_mean", "mean"),
            median_r2=("oof_r2_mean", "median"),
            mean_r2=("oof_r2_mean", "mean"),
            median_feature_count=("feature_count", "median"),
        )
        .reset_index()
        .sort_values("median_spearman", ascending=False)
    )
    return per_component, aggregate


def write_outputs(output_dir: Path, per_component: pd.DataFrame, aggregate: pd.DataFrame, metadata: dict[str, Any]) -> None:
    """Write artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    per_component.to_csv(output_dir / "leave_task_out_component_summary.csv", index=False)
    aggregate.to_csv(output_dir / "leave_task_out_aggregate_summary.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    fig = px.bar(
        aggregate,
        x="feature_group",
        y="median_spearman",
        color="model",
        hover_data=["component_count", "mean_spearman", "median_r2", "median_feature_count"],
        title="Leave-task-out DCLM component prediction",
        labels={"median_spearman": "median component OOF Spearman"},
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    fig.update_layout(template="plotly_white", xaxis_tickangle=-25)
    fig.write_html(output_dir / "leave_task_out_aggregate.html", include_plotlyjs="cdn", config=PLOT_CONFIG)
    top = aggregate.head(12)
    readme = [
        "# DCLM Leave-Task-Out Prediction Diagnostic",
        "",
        "This diagnostic predicts each DCLM component hard score while excluding smooth features from the same DCLM task. The same-task feature group is retained as an upper-bound / mechanical-reconstruction reference.",
        "",
        "## Aggregate Results",
        "",
        top.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "- If `same_task_smooth_upper_bound` is strong but `other_dclm_task_smooth` collapses, the T1 fit is mostly same-task smooth-hard reconstruction.",
        "- If `other_dclm_task_smooth` remains strong, there is cross-task DCLM latent signal that may support a denoised DCLM constraint.",
        "- Non-DCLM auxiliary metrics should not become objectives unless they beat DCLM-only transfer under this stricter test.",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(readme))


def main() -> None:
    """Run the diagnostic."""
    args = parse_args()
    matrix = pd.read_csv(args.matrix_csv, low_memory=False)
    components = pd.read_csv(args.component_summary_csv)
    signal = matrix.loc[matrix["row_kind"].eq("signal") & ~matrix["run_name"].isin(args.exclude_run_name)].copy()
    rows = run_component_models(frame=signal, components=components, folds=args.folds, repeats=args.repeats, seed=args.seed)
    per_component, aggregate = summarize(rows)
    metadata = {
        "matrix_csv": str(args.matrix_csv),
        "component_summary_csv": str(args.component_summary_csv),
        "output_dir": str(args.output_dir),
        "signal_rows": int(len(signal)),
        "components": int(len(components)),
        "folds": int(args.folds),
        "repeats": int(args.repeats),
        "models": sorted(MODELS_TO_RUN),
    }
    write_outputs(args.output_dir, per_component, aggregate, metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
