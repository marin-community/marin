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
) -> None:
    strict = summary_df[summary_df["analysis_scope"].eq("strict_common")].copy()
    improves = strict[strict["classification"].astype(str).eq("endpoint_improves")]
    worsens = strict[strict["classification"].astype(str).eq("worsens_with_t")]
    interior = strict[strict["classification"].astype(str).eq("interior_peak")]

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
        (
            "- The current strict-common cut intentionally excludes d1280 for most tasks and d1536 for all "
            "intermediate path points; rerun this script after pending eval/training completion."
        ),
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


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    path_df = read_path_deltas(args.path_deltas_csv, args.include_incomplete_mmlu_sl)
    cov_df = coverage_summary(path_df)
    strict_dims = strict_common_hidden_dims(cov_df)
    if not strict_dims:
        raise ValueError("No strict-common hidden dimensions with complete t paths.")

    strict_summary, strict_by_scale, strict_mean = summarize_scope(path_df, cov_df, "strict_common", strict_dims)
    task_summary, task_by_scale, task_mean = summarize_scope(path_df, cov_df, "task_complete", [])

    summary_df = pd.concat([strict_summary, task_summary], ignore_index=True)
    by_scale_df = pd.concat([strict_by_scale, task_by_scale], ignore_index=True)
    mean_delta_df = pd.concat([strict_mean, task_mean], ignore_index=True)

    cov_df.to_csv(output_dir / "coverage_summary.csv", index=False)
    summary_df.to_csv(output_dir / "task_t_response_summary.csv", index=False)
    by_scale_df.to_csv(output_dir / "task_t_response_by_scale.csv", index=False)
    mean_delta_df.to_csv(output_dir / "task_t_mean_deltas.csv", index=False)
    write_plots(output_dir, summary_df, mean_delta_df)
    write_summary_markdown(output_dir, summary_df, cov_df, strict_dims)

    print(f"Wrote analysis outputs to {output_dir}")
    print(f"Representability note consulted: {REPRESENTABILITY_NOTE}")
    print(f"Strict-common hidden dims: {strict_dims}")
    print(
        summary_df[summary_df["analysis_scope"].eq("strict_common")]
        .groupby("classification", observed=False)
        .size()
        .to_string()
    )


if __name__ == "__main__":
    main()
