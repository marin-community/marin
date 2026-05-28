# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Analyze Grug-MoE v4 path-test task response.

The input dashboard table already orients task deltas so positive values mean
better than proportional at the same scale. This script focuses on the
one-dimensional path coordinate t in w(t) = (1 - t) p + t w_v4.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REPO_ROOT = Path(__file__).resolve().parents[4]
DASHBOARD_DIR = (
    REPO_ROOT / "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/"
    "grug_moe_mix_dashboard_20260517"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/"
    "grug_moe_path_response_analysis_20260525"
)
PATH_DELTAS_CSV = DASHBOARD_DIR / "grug_moe_path_task_deltas.csv"
METRIC_SCALE_SOURCE_CSV = DASHBOARD_DIR / "grug_moe_mix_preferred_task_metrics.csv"
REPRESENTABILITY_NOTE = Path("/Users/calvinxu/Downloads/Representability in Stratified Sampling.md")

EXPECTED_T_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)
EXCLUDED_HEADLINE_TASKS = {"mmlu_sl_0shot", "mmlu_sl_5shot"}
CLASSIFICATION_ORDER = [
    "endpoint_improves",
    "interior_peak",
    "mixed_or_flat",
    "worsens_with_t",
]
CLASSIFICATION_COLORS = {
    "endpoint_improves": "#1a9850",
    "interior_peak": "#91cf60",
    "mixed_or_flat": "#fee08b",
    "worsens_with_t": "#d73027",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path-deltas-csv", type=Path, default=PATH_DELTAS_CSV)
    parser.add_argument("--metric-scale-source-csv", type=Path, default=METRIC_SCALE_SOURCE_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--include-incomplete-mmlu-sl", action="store_true")
    return parser.parse_args()


def read_path_deltas(path: Path, include_incomplete_mmlu_sl: bool) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = {
        "task_alias",
        "task_group",
        "preferred_metric",
        "hidden_dim",
        "budget",
        "t",
        "delta_oriented",
        "baseline_oriented_value",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
    df = df.copy()
    df["t"] = df["t"].astype(float)
    df["hidden_dim"] = df["hidden_dim"].astype(int)
    df["delta_oriented"] = df["delta_oriented"].astype(float)
    if not include_incomplete_mmlu_sl:
        df = df[~df["task_alias"].isin(EXCLUDED_HEADLINE_TASKS)].copy()
    return df


def task_metric_scales(path_df: pd.DataFrame, reference_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Estimate per-task metric scales for cross-task effect-size comparisons.

    The scale is empirical variation in oriented metric value, not repeated-seed
    noise. It is useful for comparing rough effect sizes across native metric
    units while preserving the caveat that this is a dashboard-scale diagnostic.
    """
    required = {"task_alias", "preferred_metric", "oriented_value"}
    frames = []
    for source_name, frame in (("path", path_df), ("reference", reference_df)):
        if frame is None or frame.empty:
            continue
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Missing required columns in {source_name} metric-scale frame: {sorted(missing)}")
        source = frame.loc[:, sorted(required)].copy()
        source["scale_source"] = source_name
        frames.append(source)
    if not frames:
        raise ValueError("No metric-scale source rows provided.")

    values = pd.concat(frames, ignore_index=True)
    values["oriented_value"] = pd.to_numeric(values["oriented_value"], errors="coerce")
    values = values[np.isfinite(values["oriented_value"].to_numpy(dtype=float))].copy()
    rows = []
    for (task_alias, preferred_metric), group in values.groupby(["task_alias", "preferred_metric"], sort=True):
        observed = group["oriented_value"].to_numpy(dtype=float)
        if len(observed) >= 2:
            std = float(np.std(observed, ddof=1))
            median = float(np.median(observed))
            mad = float(1.4826 * np.median(np.abs(observed - median)))
        else:
            std = float("nan")
            mad = float("nan")
        rows.append(
            {
                "task_alias": task_alias,
                "preferred_metric": preferred_metric,
                "metric_scale_n": int(len(observed)),
                "metric_scale_std": std if std > 0.0 else float("nan"),
                "metric_scale_mad": mad if mad > 0.0 else float("nan"),
                "metric_scale_min": float(np.min(observed)) if len(observed) else float("nan"),
                "metric_scale_max": float(np.max(observed)) if len(observed) else float("nan"),
                "metric_scale_source_rows": ",".join(sorted(group["scale_source"].unique())),
            }
        )
    return pd.DataFrame(rows)


def add_standardized_path_deltas(path_df: pd.DataFrame, scales_df: pd.DataFrame) -> pd.DataFrame:
    """Attach empirical-standardized path deltas to the native-unit path table."""
    required = {"task_alias", "preferred_metric", "metric_scale_std", "metric_scale_mad", "metric_scale_n"}
    missing = required - set(scales_df.columns)
    if missing:
        raise ValueError(f"Missing required columns in metric scales: {sorted(missing)}")
    out = path_df.merge(scales_df, on=["task_alias", "preferred_metric"], how="left", validate="many_to_one")
    for scale_col, delta_col in (("metric_scale_std", "delta_std"), ("metric_scale_mad", "delta_mad")):
        scale = pd.to_numeric(out[scale_col], errors="coerce").to_numpy(dtype=float)
        delta = pd.to_numeric(out["delta_oriented"], errors="coerce").to_numpy(dtype=float)
        out[delta_col] = np.where(np.isfinite(scale) & (scale > 0.0), delta / scale, np.nan)
    return out


def coverage_summary(df: pd.DataFrame) -> pd.DataFrame:
    all_t = set(EXPECTED_T_VALUES)
    rows = []
    for (task, hidden_dim), group in df.groupby(["task_alias", "hidden_dim"], sort=True):
        observed_t = sorted(float(value) for value in group["t"].unique())
        missing_t = sorted(all_t - set(observed_t))
        rows.append(
            {
                "task_alias": task,
                "task_group": group["task_group"].mode().iloc[0],
                "preferred_metric": group["preferred_metric"].mode().iloc[0],
                "hidden_dim": hidden_dim,
                "n_t": len(observed_t),
                "observed_t": ",".join(f"{value:.2f}" for value in observed_t),
                "missing_t": ",".join(f"{value:.2f}" for value in missing_t),
                "complete_path": not missing_t,
            }
        )
    return pd.DataFrame(rows)


def strict_common_hidden_dims(coverage_df: pd.DataFrame) -> list[int]:
    complete = coverage_df[coverage_df["complete_path"]]
    if complete.empty:
        return []
    complete_by_task = complete.groupby("task_alias")["hidden_dim"].apply(set)
    common = set.intersection(*complete_by_task.tolist())
    return sorted(common)


def task_complete_hidden_dims(coverage_df: pd.DataFrame, task_alias: str) -> list[int]:
    task_cov = coverage_df[coverage_df["task_alias"].eq(task_alias)]
    return sorted(task_cov[task_cov["complete_path"]]["hidden_dim"].astype(int).tolist())


def pearson(x_values: pd.Series, y_values: pd.Series) -> float:
    if len(x_values) < 2 or float(y_values.std(ddof=0)) == 0.0:
        return float("nan")
    return float(x_values.corr(y_values, method="pearson"))


def spearman(x_values: pd.Series, y_values: pd.Series) -> float:
    if len(x_values) < 2 or float(y_values.std(ddof=0)) == 0.0:
        return float("nan")
    return float(x_values.corr(y_values, method="spearman"))


def linear_slope(x_values: pd.Series, y_values: pd.Series) -> float:
    if len(x_values) < 2 or float(x_values.std(ddof=0)) == 0.0:
        return float("nan")
    x = x_values.to_numpy(dtype=float)
    y = y_values.to_numpy(dtype=float)
    return float(np.polyfit(x, y, deg=1)[0])


def scale_fixed_slope(df: pd.DataFrame) -> float:
    if df["hidden_dim"].nunique() < 1 or len(df) < 2:
        return float("nan")
    centered_t = df["t"] - df.groupby("hidden_dim")["t"].transform("mean")
    centered_y = df["delta_oriented"] - df.groupby("hidden_dim")["delta_oriented"].transform("mean")
    denom = float(np.dot(centered_t, centered_t))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(centered_t, centered_y) / denom)


def classify_response(row: pd.Series) -> str:
    slope = float(row["scale_fixed_slope"])
    endpoint = float(row["endpoint_delta_mean"])
    best_t = float(row["best_t_mean"])
    best_delta = float(row["best_delta_mean"])
    positive_fraction = float(row["positive_scale_slope_fraction"])
    negative_fraction = float(row["negative_scale_slope_fraction"])
    if 0.0 < best_t < 1.0 and best_delta > max(endpoint, 0.0):
        return "interior_peak"
    if positive_fraction >= 2.0 / 3.0 and slope > 0.0 and endpoint > 0.0:
        return "endpoint_improves"
    if negative_fraction >= 2.0 / 3.0 and slope < 0.0 and endpoint < 0.0:
        return "worsens_with_t"
    return "mixed_or_flat"


def summarize_scope(
    path_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    scope: str,
    hidden_dims: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    by_scale_rows = []
    mean_delta_rows = []
    for task, task_df_all in path_df.groupby("task_alias", sort=True):
        dims = hidden_dims if scope == "strict_common" else task_complete_hidden_dims(coverage_df, task)
        task_df = task_df_all[task_df_all["hidden_dim"].isin(dims)].copy()
        if task_df.empty:
            continue

        per_scale = []
        for hidden_dim, scale_df in task_df.groupby("hidden_dim", sort=True):
            if set(float(value) for value in scale_df["t"].unique()) != set(EXPECTED_T_VALUES):
                continue
            scale_df = scale_df.sort_values("t")
            slope = linear_slope(scale_df["t"], scale_df["delta_oriented"])
            r_pearson = pearson(scale_df["t"], scale_df["delta_oriented"])
            r_spearman = spearman(scale_df["t"], scale_df["delta_oriented"])
            best_row = scale_df.loc[scale_df["delta_oriented"].idxmax()]
            endpoint_delta = float(scale_df[scale_df["t"].eq(1.0)]["delta_oriented"].iloc[0])
            row = {
                "analysis_scope": scope,
                "task_alias": task,
                "task_group": scale_df["task_group"].mode().iloc[0],
                "preferred_metric": scale_df["preferred_metric"].mode().iloc[0],
                "hidden_dim": int(hidden_dim),
                "n_points": len(scale_df),
                "pearson_r": r_pearson,
                "spearman_r": r_spearman,
                "slope": slope,
                "endpoint_delta": endpoint_delta,
                "best_t": float(best_row["t"]),
                "best_delta": float(best_row["delta_oriented"]),
            }
            per_scale.append(row)
            by_scale_rows.append(row)

        if not per_scale:
            continue

        complete_task_df = task_df[task_df["hidden_dim"].isin([row["hidden_dim"] for row in per_scale])].copy()
        mean_by_t = (
            complete_task_df.groupby("t", as_index=False)
            .agg(
                delta_mean=("delta_oriented", "mean"),
                delta_median=("delta_oriented", "median"),
                delta_min=("delta_oriented", "min"),
                delta_max=("delta_oriented", "max"),
                n_scales=("hidden_dim", "nunique"),
            )
            .sort_values("t")
        )
        for _, mean_row in mean_by_t.iterrows():
            mean_delta_rows.append(
                {
                    "analysis_scope": scope,
                    "task_alias": task,
                    "task_group": complete_task_df["task_group"].mode().iloc[0],
                    "preferred_metric": complete_task_df["preferred_metric"].mode().iloc[0],
                    "t": float(mean_row["t"]),
                    "delta_mean": float(mean_row["delta_mean"]),
                    "delta_median": float(mean_row["delta_median"]),
                    "delta_min": float(mean_row["delta_min"]),
                    "delta_max": float(mean_row["delta_max"]),
                    "n_scales": int(mean_row["n_scales"]),
                }
            )

        slopes = np.array([row["slope"] for row in per_scale], dtype=float)
        positive_fraction = float(np.mean(slopes > 0.0))
        negative_fraction = float(np.mean(slopes < 0.0))
        sign_agreement = max(positive_fraction, negative_fraction)
        best_mean_row = mean_by_t.loc[mean_by_t["delta_mean"].idxmax()]
        endpoint_mean = float(mean_by_t[mean_by_t["t"].eq(1.0)]["delta_mean"].iloc[0])
        t025_mean = float(mean_by_t[mean_by_t["t"].eq(0.25)]["delta_mean"].iloc[0])
        t050_mean = float(mean_by_t[mean_by_t["t"].eq(0.50)]["delta_mean"].iloc[0])
        t075_mean = float(mean_by_t[mean_by_t["t"].eq(0.75)]["delta_mean"].iloc[0])
        summary_rows.append(
            {
                "analysis_scope": scope,
                "task_alias": task,
                "task_group": complete_task_df["task_group"].mode().iloc[0],
                "preferred_metric": complete_task_df["preferred_metric"].mode().iloc[0],
                "n_points": len(complete_task_df),
                "n_scales": len(per_scale),
                "hidden_dims": ",".join(str(row["hidden_dim"]) for row in per_scale),
                "pooled_pearson_r": pearson(complete_task_df["t"], complete_task_df["delta_oriented"]),
                "pooled_spearman_r": spearman(complete_task_df["t"], complete_task_df["delta_oriented"]),
                "scale_fixed_slope": scale_fixed_slope(complete_task_df),
                "mean_scale_slope": float(np.mean(slopes)),
                "median_scale_slope": float(np.median(slopes)),
                "mean_scale_pearson_r": float(np.nanmean([row["pearson_r"] for row in per_scale])),
                "mean_scale_spearman_r": float(np.nanmean([row["spearman_r"] for row in per_scale])),
                "positive_scale_slope_fraction": positive_fraction,
                "negative_scale_slope_fraction": negative_fraction,
                "scale_slope_sign_agreement": sign_agreement,
                "delta_t025_mean": t025_mean,
                "delta_t050_mean": t050_mean,
                "delta_t075_mean": t075_mean,
                "endpoint_delta_mean": endpoint_mean,
                "best_t_mean": float(best_mean_row["t"]),
                "best_delta_mean": float(best_mean_row["delta_mean"]),
                "worst_delta_mean": float(mean_by_t["delta_mean"].min()),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df["classification"] = summary_df.apply(classify_response, axis=1)
        summary_df["classification"] = pd.Categorical(
            summary_df["classification"], categories=CLASSIFICATION_ORDER, ordered=True
        )
        summary_df = summary_df.sort_values(["classification", "pooled_pearson_r"], ascending=[True, False])
    return summary_df, pd.DataFrame(by_scale_rows), pd.DataFrame(mean_delta_rows)


def write_summary_markdown(
    output_dir: Path,
    summary_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    strict_hidden_dims: list[int],
    standardized_summary_df: pd.DataFrame | None = None,
) -> None:
    strict = summary_df[summary_df["analysis_scope"].eq("strict_common")].copy()
    improves = strict[strict["classification"].astype(str).eq("endpoint_improves")]
    worsens = strict[strict["classification"].astype(str).eq("worsens_with_t")]
    interior = strict[strict["classification"].astype(str).eq("interior_peak")]
    missing_coverage = coverage_df[~coverage_df["complete_path"]].sort_values(["hidden_dim", "task_alias"])
    if missing_coverage.empty:
        coverage_note = "- No missing task/scale paths remain in the filtered input."
    else:
        missing_bits = [
            f"d{int(row.hidden_dim)} `{row.task_alias}` missing t=`{row.missing_t}`"
            for row in missing_coverage.itertuples(index=False)
        ]
        coverage_note = "- Remaining missing task/scale paths: " + "; ".join(missing_bits) + "."

    lines = [
        "# Grug-MoE v4 Path Response Analysis",
        "",
        (
            "This analysis uses the one-dimensional path \\(w(t)=(1-t)p+t w_{v4}\\), "
            "with task deltas oriented so positive means better than proportional at the same scale."
        ),
        "",
        (
            "The representability hypothesis predicts three broad empirical regimes: aligned tasks improve "
            "along the controllable path, coverage-sensitive tasks worsen as the path moves away from "
            "proportional, and weakly controllable or noisy tasks show flat or inconsistent response."
        ),
        "",
        "## Coverage",
        "",
        f"- Headline strict-common analysis uses hidden dimensions: `{', '.join(map(str, strict_hidden_dims))}`.",
        (
            f"- Headline task count: `{strict['task_alias'].nunique()}` after excluding incomplete "
            "non-verb MMLU-SL aliases."
        ),
        (
            f"- Complete task/scale paths in the filtered input: "
            f"`{int(coverage_df['complete_path'].sum())}` out of `{len(coverage_df)}` task-scale cells."
        ),
        coverage_note,
        "",
        "## Headline Classification",
        "",
        f"- Endpoint improves: `{len(improves)}` tasks.",
        f"- Interior peak: `{len(interior)}` tasks.",
        f"- Worsens with t: `{len(worsens)}` tasks.",
        f"- Mixed or flat: `{len(strict) - len(improves) - len(interior) - len(worsens)}` tasks.",
        "",
        "## Strongest Positive t-Response",
        "",
    ]
    for _, row in strict.sort_values("pooled_pearson_r", ascending=False).head(8).iterrows():
        lines.append(
            f"- `{row['task_alias']}`: Pearson `{row['pooled_pearson_r']:.3f}`, "
            f"endpoint delta `{row['endpoint_delta_mean']:.4g}`, best t `{row['best_t_mean']:.2f}`."
        )
    lines.extend(["", "## Strongest Negative t-Response", ""])
    for _, row in strict.sort_values("pooled_pearson_r", ascending=True).head(8).iterrows():
        lines.append(
            f"- `{row['task_alias']}`: Pearson `{row['pooled_pearson_r']:.3f}`, "
            f"endpoint delta `{row['endpoint_delta_mean']:.4g}`, best t `{row['best_t_mean']:.2f}`."
        )
    if standardized_summary_df is not None and not standardized_summary_df.empty:
        endpoint = standardized_summary_df.dropna(subset=["endpoint_delta_std_mean"]).copy()
        endpoint_positive = endpoint[endpoint["endpoint_delta_std_mean"] > 0.0]
        endpoint_negative = endpoint[endpoint["endpoint_delta_std_mean"] < 0.0]
        lines.extend(
            [
                "",
                "## Standardized Effect-Size View",
                "",
                (
                    "- Standardization divides each task's oriented delta by the empirical standard deviation "
                    "of that task's oriented metric values across the Grug-MoE dashboard/path cells. This is "
                    "a native-unit effect-size diagnostic, not a repeated-seed noise standard deviation."
                ),
                (
                    f"- At t=1, `{len(endpoint_positive)}` tasks are positive and `{len(endpoint_negative)}` "
                    "tasks are negative in standardized units."
                ),
                (
                    f"- Mean positive endpoint standardized delta: "
                    f"`{endpoint_positive['endpoint_delta_std_mean'].mean():.3f}`; mean absolute negative "
                    f"endpoint standardized delta: "
                    f"`{(-endpoint_negative['endpoint_delta_std_mean']).mean():.3f}`."
                ),
                "",
                "### Largest Standardized Endpoint Gains",
                "",
            ]
        )
        for _, row in endpoint.sort_values("endpoint_delta_std_mean", ascending=False).head(6).iterrows():
            lines.append(
                f"- `{row['task_alias']}`: endpoint z-delta `{row['endpoint_delta_std_mean']:.3f}`, "
                f"best t `{row['best_t_std_mean']:.2f}`."
            )
        lines.extend(["", "### Largest Standardized Endpoint Deteriorations", ""])
        for _, row in endpoint.sort_values("endpoint_delta_std_mean", ascending=True).head(6).iterrows():
            lines.append(
                f"- `{row['task_alias']}`: endpoint z-delta `{row['endpoint_delta_std_mean']:.3f}`, "
                f"best t `{row['best_t_std_mean']:.2f}`."
            )
    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            (
                "- Correlation is useful for sign and monotonicity, but not effect size across tasks because "
                "task metrics have different native units."
            ),
            (
                "- This is not a repeated-seed uncertainty analysis. Flat or mixed response should be treated "
                "as weak evidence until paired with SNR/noise estimates."
            ),
            (
                "- A positive endpoint and consistent positive slope supports controllability along the v4 "
                "direction. A negative endpoint and consistent negative slope supports a real path tradeoff "
                "against proportional coverage. An interior best t suggests a trust-region interpolation may "
                "dominate the endpoint."
            ),
            "",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines))


def standardized_task_frames(
    standardized_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    strict_hidden_dims: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate empirical-standardized path deltas over strict-common scales."""
    if not strict_hidden_dims:
        return pd.DataFrame(), pd.DataFrame()
    complete_task_counts = (
        coverage_df[
            coverage_df["complete_path"] & coverage_df["hidden_dim"].isin(strict_hidden_dims)
        ]
        .groupby("task_alias")["hidden_dim"]
        .nunique()
    )
    complete_tasks = set(complete_task_counts[complete_task_counts.eq(len(strict_hidden_dims))].index)
    scoped = standardized_df[
        standardized_df["task_alias"].isin(complete_tasks) & standardized_df["hidden_dim"].isin(strict_hidden_dims)
    ].copy()
    scoped = scoped[np.isfinite(scoped["delta_std"].to_numpy(dtype=float))].copy()
    if scoped.empty:
        return pd.DataFrame(), pd.DataFrame()

    mean_rows = []
    summary_rows = []
    for task_alias, task_df in scoped.groupby("task_alias", sort=True):
        mean_by_t = (
            task_df.groupby("t", as_index=False)
            .agg(
                delta_std_mean=("delta_std", "mean"),
                delta_std_median=("delta_std", "median"),
                delta_std_min=("delta_std", "min"),
                delta_std_max=("delta_std", "max"),
                delta_mad_mean=("delta_mad", "mean"),
                n_scales=("hidden_dim", "nunique"),
                metric_scale_std=("metric_scale_std", "first"),
                metric_scale_mad=("metric_scale_mad", "first"),
                metric_scale_n=("metric_scale_n", "first"),
            )
            .sort_values("t")
        )
        if set(float(value) for value in mean_by_t["t"]) != set(EXPECTED_T_VALUES):
            continue
        task_group = task_df["task_group"].mode().iloc[0]
        preferred_metric = task_df["preferred_metric"].mode().iloc[0]
        for _, row in mean_by_t.iterrows():
            mean_rows.append(
                {
                    "task_alias": task_alias,
                    "task_group": task_group,
                    "preferred_metric": preferred_metric,
                    "t": float(row["t"]),
                    "delta_std_mean": float(row["delta_std_mean"]),
                    "delta_std_median": float(row["delta_std_median"]),
                    "delta_std_min": float(row["delta_std_min"]),
                    "delta_std_max": float(row["delta_std_max"]),
                    "delta_mad_mean": float(row["delta_mad_mean"]),
                    "n_scales": int(row["n_scales"]),
                    "metric_scale_std": float(row["metric_scale_std"]),
                    "metric_scale_mad": float(row["metric_scale_mad"]),
                    "metric_scale_n": int(row["metric_scale_n"]),
                }
            )
        endpoint = mean_by_t[mean_by_t["t"].eq(1.0)].iloc[0]
        best = mean_by_t.loc[mean_by_t["delta_std_mean"].idxmax()]
        worst = mean_by_t.loc[mean_by_t["delta_std_mean"].idxmin()]
        summary_rows.append(
            {
                "task_alias": task_alias,
                "task_group": task_group,
                "preferred_metric": preferred_metric,
                "endpoint_delta_std_mean": float(endpoint["delta_std_mean"]),
                "endpoint_delta_mad_mean": float(endpoint["delta_mad_mean"]),
                "best_t_std_mean": float(best["t"]),
                "best_delta_std_mean": float(best["delta_std_mean"]),
                "worst_t_std_mean": float(worst["t"]),
                "worst_delta_std_mean": float(worst["delta_std_mean"]),
                "metric_scale_std": float(endpoint["metric_scale_std"]),
                "metric_scale_mad": float(endpoint["metric_scale_mad"]),
                "metric_scale_n": int(endpoint["metric_scale_n"]),
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(mean_rows)


def task_order_by_summary(summary_df: pd.DataFrame) -> list[str]:
    strict = summary_df[summary_df["analysis_scope"].eq("strict_common")].copy()
    strict = strict.sort_values("pooled_pearson_r", ascending=False)
    return strict["task_alias"].tolist()


def write_plots(output_dir: Path, summary_df: pd.DataFrame, mean_delta_df: pd.DataFrame) -> None:
    strict_summary = summary_df[summary_df["analysis_scope"].eq("strict_common")].copy()
    strict_mean = mean_delta_df[mean_delta_df["analysis_scope"].eq("strict_common")].copy()
    if strict_summary.empty or strict_mean.empty:
        return

    order = task_order_by_summary(summary_df)
    strict_summary["task_alias"] = pd.Categorical(strict_summary["task_alias"], categories=order, ordered=True)
    strict_mean["task_alias"] = pd.Categorical(strict_mean["task_alias"], categories=order, ordered=True)
    strict_summary = strict_summary.sort_values("task_alias")
    strict_mean = strict_mean.sort_values(["task_alias", "t"])

    heatmap_df = strict_mean.pivot(index="task_alias", columns="t", values="delta_mean").loc[order]
    fig_heatmap = px.imshow(
        heatmap_df,
        aspect="auto",
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0.0,
        labels={"x": "path t", "y": "task", "color": "mean oriented delta"},
        title="Grug-MoE v4 path: mean task delta vs proportional over strict-common scales",
    )
    fig_heatmap.update_layout(height=max(620, 28 * len(order)), margin={"l": 160, "r": 40, "t": 70, "b": 60})
    fig_heatmap.write_html(output_dir / "task_t_delta_heatmap.html", include_plotlyjs="cdn")

    fig_bar = px.bar(
        strict_summary.sort_values("pooled_pearson_r"),
        x="pooled_pearson_r",
        y="task_alias",
        color="endpoint_delta_mean",
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0.0,
        orientation="h",
        hover_data=[
            "classification",
            "task_group",
            "preferred_metric",
            "scale_fixed_slope",
            "endpoint_delta_mean",
            "best_t_mean",
            "best_delta_mean",
            "hidden_dims",
        ],
        title="Path response ranking: correlation between t and oriented task delta",
        labels={"pooled_pearson_r": "Pearson r(t, delta)", "task_alias": "task"},
    )
    fig_bar.update_layout(height=max(620, 28 * len(order)), margin={"l": 180, "r": 40, "t": 70, "b": 60})
    fig_bar.add_vline(x=0.0, line={"color": "#555", "dash": "dot", "width": 1})
    fig_bar.write_html(output_dir / "task_t_correlation_ranking.html", include_plotlyjs="cdn")

    fig_scatter = px.scatter(
        strict_summary,
        x="endpoint_delta_mean",
        y="pooled_pearson_r",
        color="classification",
        color_discrete_map=CLASSIFICATION_COLORS,
        hover_name="task_alias",
        hover_data=[
            "task_group",
            "preferred_metric",
            "scale_fixed_slope",
            "mean_scale_pearson_r",
            "best_t_mean",
            "best_delta_mean",
        ],
        title="Path endpoint gain vs monotonic t-response",
        labels={"endpoint_delta_mean": "mean delta at t=1", "pooled_pearson_r": "Pearson r(t, delta)"},
    )
    fig_scatter.add_hline(y=0.0, line={"color": "#555", "dash": "dot", "width": 1})
    fig_scatter.add_vline(x=0.0, line={"color": "#555", "dash": "dot", "width": 1})
    fig_scatter.write_html(output_dir / "task_endpoint_vs_correlation.html", include_plotlyjs="cdn")

    n_cols = 4
    n_rows = math.ceil(len(order) / n_cols)
    subplot_titles = []
    for task in order:
        row = strict_summary[strict_summary["task_alias"].astype(str).eq(task)].iloc[0]
        subplot_titles.append(f"{task}<br><sup>r={row['pooled_pearson_r']:.2f}, {row['classification']}</sup>")
    fig_facets = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=subplot_titles, horizontal_spacing=0.055)
    for task_idx, task in enumerate(order):
        row_idx = task_idx // n_cols + 1
        col_idx = task_idx % n_cols + 1
        task_mean = strict_mean[strict_mean["task_alias"].astype(str).eq(task)].sort_values("t")
        fig_facets.add_trace(
            go.Scatter(
                x=task_mean["t"],
                y=task_mean["delta_mean"],
                mode="lines+markers",
                marker={"size": 7, "color": task_mean["t"], "colorscale": "RdYlGn", "cmin": 0.0, "cmax": 1.0},
                line={"color": "#2b5c8a", "width": 1.5},
                customdata=np.stack(
                    [
                        task_mean["delta_min"],
                        task_mean["delta_max"],
                        task_mean["n_scales"],
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "t=%{x:.2f}<br>"
                    "mean delta=%{y:.4g}<br>"
                    "min=%{customdata[0]:.4g}<br>"
                    "max=%{customdata[1]:.4g}<br>"
                    "n_scales=%{customdata[2]}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=row_idx,
            col=col_idx,
        )
        fig_facets.add_hline(y=0.0, line={"color": "#777", "dash": "dot", "width": 1}, row=row_idx, col=col_idx)
        fig_facets.update_xaxes(title_text="t", row=row_idx, col=col_idx)
    fig_facets.update_layout(
        title="Mean oriented task delta along v4 path, strict-common scales",
        height=max(800, 255 * n_rows),
        margin={"l": 45, "r": 25, "t": 95, "b": 60},
    )
    fig_facets.update_annotations(font_size=11)
    fig_facets.write_html(output_dir / "task_t_mean_delta_facets.html", include_plotlyjs="cdn")


def write_standardized_plots(
    output_dir: Path,
    standardized_summary_df: pd.DataFrame,
    standardized_mean_df: pd.DataFrame,
) -> None:
    """Write plots where task deltas are normalized by empirical task scale."""
    if standardized_summary_df.empty or standardized_mean_df.empty:
        return
    order = (
        standardized_summary_df.sort_values("endpoint_delta_std_mean", ascending=False)["task_alias"]
        .drop_duplicates()
        .tolist()
    )
    summary = standardized_summary_df.copy()
    mean_df = standardized_mean_df.copy()
    summary["task_alias"] = pd.Categorical(summary["task_alias"], categories=order, ordered=True)
    mean_df["task_alias"] = pd.Categorical(mean_df["task_alias"], categories=order, ordered=True)
    summary = summary.sort_values("task_alias")
    mean_df = mean_df.sort_values(["task_alias", "t"])

    heatmap_df = mean_df.pivot(index="task_alias", columns="t", values="delta_std_mean").loc[order]
    fig_heatmap = px.imshow(
        heatmap_df,
        aspect="auto",
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0.0,
        labels={"x": "path t", "y": "task", "color": "standardized delta"},
        title=(
            "Grug-MoE v4 path: task deltas divided by empirical task metric std "
            "(positive is better)"
        ),
    )
    fig_heatmap.update_layout(height=max(620, 28 * len(order)), margin={"l": 170, "r": 40, "t": 80, "b": 60})
    fig_heatmap.write_html(output_dir / "task_t_standardized_delta_heatmap.html", include_plotlyjs="cdn")

    fig_bar = px.bar(
        summary.sort_values("endpoint_delta_std_mean"),
        x="endpoint_delta_std_mean",
        y="task_alias",
        color="endpoint_delta_std_mean",
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0.0,
        orientation="h",
        hover_data=[
            "task_group",
            "preferred_metric",
            "best_t_std_mean",
            "best_delta_std_mean",
            "worst_t_std_mean",
            "worst_delta_std_mean",
            "metric_scale_std",
            "metric_scale_n",
        ],
        title=(
            "Endpoint effect-size ranking: t=1 delta divided by empirical task metric std "
            "(positive is better)"
        ),
        labels={"endpoint_delta_std_mean": "endpoint standardized delta", "task_alias": "task"},
    )
    fig_bar.add_vline(x=0.0, line={"color": "#555", "dash": "dot", "width": 1})
    fig_bar.update_layout(height=max(620, 28 * len(order)), margin={"l": 190, "r": 40, "t": 80, "b": 60})
    fig_bar.write_html(output_dir / "task_endpoint_standardized_delta_ranking.html", include_plotlyjs="cdn")

    n_cols = 4
    n_rows = math.ceil(len(order) / n_cols)
    subplot_titles = []
    for task in order:
        row = summary[summary["task_alias"].astype(str).eq(task)].iloc[0]
        subplot_titles.append(f"{task}<br><sup>endpoint z={row['endpoint_delta_std_mean']:.2f}</sup>")
    fig_facets = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=subplot_titles, horizontal_spacing=0.055)
    for task_idx, task in enumerate(order):
        row_idx = task_idx // n_cols + 1
        col_idx = task_idx % n_cols + 1
        task_mean = mean_df[mean_df["task_alias"].astype(str).eq(task)].sort_values("t")
        fig_facets.add_trace(
            go.Scatter(
                x=task_mean["t"],
                y=task_mean["delta_std_mean"],
                mode="lines+markers",
                marker={"size": 7, "color": task_mean["t"], "colorscale": "RdYlGn", "cmin": 0.0, "cmax": 1.0},
                line={"color": "#2b5c8a", "width": 1.5},
                customdata=np.stack(
                    [
                        task_mean["delta_std_min"],
                        task_mean["delta_std_max"],
                        task_mean["metric_scale_std"],
                        task_mean["metric_scale_n"],
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "t=%{x:.2f}<br>"
                    "mean standardized delta=%{y:.3f}<br>"
                    "min=%{customdata[0]:.3f}<br>"
                    "max=%{customdata[1]:.3f}<br>"
                    "metric std=%{customdata[2]:.4g}<br>"
                    "scale n=%{customdata[3]}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=row_idx,
            col=col_idx,
        )
        fig_facets.add_hline(y=0.0, line={"color": "#777", "dash": "dot", "width": 1}, row=row_idx, col=col_idx)
        fig_facets.update_xaxes(title_text="t", row=row_idx, col=col_idx)
    fig_facets.update_layout(
        title="Mean standardized task delta along v4 path, strict-common scales",
        height=max(800, 255 * n_rows),
        margin={"l": 45, "r": 25, "t": 95, "b": 60},
    )
    fig_facets.update_annotations(font_size=11)
    fig_facets.write_html(output_dir / "task_t_standardized_mean_delta_facets.html", include_plotlyjs="cdn")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    path_df = read_path_deltas(args.path_deltas_csv, args.include_incomplete_mmlu_sl)
    metric_reference_df = pd.read_csv(args.metric_scale_source_csv) if args.metric_scale_source_csv.exists() else None
    metric_scales_df = task_metric_scales(path_df, metric_reference_df)
    standardized_path_df = add_standardized_path_deltas(path_df, metric_scales_df)
    cov_df = coverage_summary(path_df)
    strict_dims = strict_common_hidden_dims(cov_df)
    if not strict_dims:
        raise ValueError("No strict-common hidden dimensions with complete t paths.")

    strict_summary, strict_by_scale, strict_mean = summarize_scope(path_df, cov_df, "strict_common", strict_dims)
    task_summary, task_by_scale, task_mean = summarize_scope(path_df, cov_df, "task_complete", [])

    summary_df = pd.concat([strict_summary, task_summary], ignore_index=True)
    by_scale_df = pd.concat([strict_by_scale, task_by_scale], ignore_index=True)
    mean_delta_df = pd.concat([strict_mean, task_mean], ignore_index=True)
    standardized_summary_df, standardized_mean_df = standardized_task_frames(standardized_path_df, cov_df, strict_dims)

    cov_df.to_csv(output_dir / "coverage_summary.csv", index=False)
    summary_df.to_csv(output_dir / "task_t_response_summary.csv", index=False)
    by_scale_df.to_csv(output_dir / "task_t_response_by_scale.csv", index=False)
    mean_delta_df.to_csv(output_dir / "task_t_mean_deltas.csv", index=False)
    metric_scales_df.to_csv(output_dir / "task_metric_scales.csv", index=False)
    standardized_path_df.to_csv(output_dir / "standardized_path_task_deltas.csv", index=False)
    standardized_summary_df.to_csv(output_dir / "task_endpoint_standardized_delta_summary.csv", index=False)
    standardized_mean_df.to_csv(output_dir / "task_t_standardized_mean_deltas.csv", index=False)
    write_plots(output_dir, summary_df, mean_delta_df)
    write_standardized_plots(output_dir, standardized_summary_df, standardized_mean_df)
    write_summary_markdown(output_dir, summary_df, cov_df, strict_dims, standardized_summary_df)

    print(f"Wrote analysis outputs to {output_dir}")
    print(f"Representability note consulted: {REPRESENTABILITY_NOTE}")
    print(f"Strict-common hidden dims: {strict_dims}")
    print(
        summary_df[summary_df["analysis_scope"].eq("strict_common")]
        .groupby("classification", observed=False)
        .size()
        .to_string()
    )
    if not standardized_summary_df.empty:
        endpoint = standardized_summary_df["endpoint_delta_std_mean"].dropna()
        print("Standardized endpoint deltas:")
        print(endpoint.describe().to_string())


if __name__ == "__main__":
    main()
