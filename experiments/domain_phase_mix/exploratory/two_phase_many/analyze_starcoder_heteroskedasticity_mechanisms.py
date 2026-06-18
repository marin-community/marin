# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "plotly", "scipy"]
# ///
"""Diagnose mechanisms behind StarCoder mixture-dependent noise.

This script uses existing repeated-anchor StarCoder/Nemotron data only. It is a
follow-up to the noise-shape diagnostics: it tests whether the observed
heteroskedasticity is better explained by variance-scale coupling,
concentration/effective support, or outlier-heavy repeats.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REPEATS = (
    SCRIPT_DIR.parent / "reference_outputs" / "starcoder_heteroskedastic_snr_20260523" / "collected_train_only_metrics_live.csv"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "starcoder_heteroskedasticity_mechanisms_20260617"
TO_IMAGE_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}

METRIC_PREFIXES = ("eval/",)
COUNT_SUFFIXES = ("/bytes", "/documents", "/example_count", "/loading_time", "/total_time")
KEY_METRICS = [
    "eval/bpb",
    "eval/loss",
    "eval/uncheatable_eval/bpb",
    "eval/paloma/dolma_100_programing_languages/bpb",
    "eval/paloma/macro_bpb",
    "eval/uncheatable_eval/github_python/bpb",
    "eval/uncheatable_eval/github_cpp/bpb",
]


@dataclass(frozen=True)
class RegressionResult:
    """Simple OLS summary for one metric/predictor pair."""

    metric: str
    target: str
    predictor: str
    n: int
    slope: float
    intercept: float
    r2: float
    p_value: float


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=Path, default=DEFAULT_REPEATS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def is_metric_column(column: str) -> bool:
    """Return whether a column is a scalar metric rather than metadata."""
    return column.startswith(METRIC_PREFIXES) and not column.endswith(COUNT_SUFFIXES)


def is_primary_bpb_metric(column: str) -> bool:
    """Return whether a metric is a primary BPB-like scalar for residual-correlation checks."""
    return column.endswith("/bpb") or column.endswith("_bpb")


def phase_hhi(phase_0_starcoder: float, phase_1_starcoder: float) -> float:
    """Average two-domain Herfindahl index across phases."""
    h0 = phase_0_starcoder**2 + (1.0 - phase_0_starcoder) ** 2
    h1 = phase_1_starcoder**2 + (1.0 - phase_1_starcoder) ** 2
    return float(0.5 * (h0 + h1))


def binary_entropy(p: np.ndarray) -> np.ndarray:
    """Binary entropy with natural logs."""
    p = np.clip(p.astype(float), 1e-12, 1.0 - 1e-12)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def phase_entropy(phase_0_starcoder: float, phase_1_starcoder: float) -> float:
    """Average two-domain entropy across phases."""
    return float(0.5 * (binary_entropy(np.array([phase_0_starcoder]))[0] + binary_entropy(np.array([phase_1_starcoder]))[0]))


def add_anchor_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add concentration and effective-support features."""
    out = frame.copy()
    out["phase_hhi"] = [
        phase_hhi(float(p0), float(p1)) for p0, p1 in zip(out["phase_0_starcoder"], out["phase_1_starcoder"], strict=True)
    ]
    out["phase_entropy"] = [
        phase_entropy(float(p0), float(p1))
        for p0, p1 in zip(out["phase_0_starcoder"], out["phase_1_starcoder"], strict=True)
    ]
    out["phase_effective_support"] = np.exp(out["phase_entropy"])
    out["max_phase_weight"] = np.maximum.reduce(
        [
            out["phase_0_starcoder"].to_numpy(dtype=float),
            out["phase_1_starcoder"].to_numpy(dtype=float),
            1.0 - out["phase_0_starcoder"].to_numpy(dtype=float),
            1.0 - out["phase_1_starcoder"].to_numpy(dtype=float),
        ]
    )
    out["phase_imbalance"] = np.abs(out["phase_0_starcoder"] - out["phase_1_starcoder"])
    return out


def anchor_summary(frame: pd.DataFrame, metric_columns: list[str]) -> pd.DataFrame:
    """Compute per-anchor metric summary with concentration features."""
    rows: list[dict[str, Any]] = []
    group_cols = ["anchor_index", "anchor_id", "phase_0_starcoder", "phase_1_starcoder", "total_starcoder_epochs"]
    for group_values, group in frame.groupby(group_cols, dropna=False):
        base = dict(zip(group_cols, group_values, strict=True))
        for metric in metric_columns:
            values = pd.to_numeric(group[metric], errors="coerce").to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if len(values) < 3:
                continue
            std = float(np.std(values, ddof=1))
            if not np.isfinite(std) or std <= 0.0:
                continue
            row = base.copy()
            row.update(
                {
                    "metric": metric,
                    "count": int(len(values)),
                    "mean": float(np.mean(values)),
                    "std": std,
                    "variance": std**2,
                    "log_std": float(np.log(std)),
                    "log_variance": float(np.log(std**2)),
                    "log_mean": float(np.log(max(abs(float(np.mean(values))), 1e-12))),
                    "median": float(np.median(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "jackknife_std_min": float("nan"),
                    "jackknife_std_max": float("nan"),
                    "jackknife_std_ratio": float("nan"),
                }
            )
            if len(values) > 3:
                jack = np.array([np.std(np.delete(values, i), ddof=1) for i in range(len(values))], dtype=float)
                finite = jack[np.isfinite(jack) & (jack > 0.0)]
                if len(finite):
                    row["jackknife_std_min"] = float(np.min(finite))
                    row["jackknife_std_max"] = float(np.max(finite))
                    row["jackknife_std_ratio"] = float(np.max(finite) / max(std, 1e-12))
            rows.append(row)
    return add_anchor_features(pd.DataFrame(rows))


def brown_forsythe_tests(frame: pd.DataFrame, metric_columns: list[str]) -> pd.DataFrame:
    """Run Brown-Forsythe robust equal-variance tests per metric."""
    rows = []
    for metric in metric_columns:
        groups = []
        for _, group in frame.groupby("anchor_id", dropna=False):
            values = pd.to_numeric(group[metric], errors="coerce").to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if len(values) >= 2 and np.std(values) > 0.0:
                groups.append(values)
        if len(groups) < 2:
            continue
        stat = stats.levene(*groups, center="median")
        rows.append({"metric": metric, "anchor_count": len(groups), "statistic": float(stat.statistic), "p_value": float(stat.pvalue)})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["reject_p05"] = out["p_value"] < 0.05
        out["reject_p10"] = out["p_value"] < 0.10
    return out


def fit_simple_ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """Fit y = intercept + slope*x and return slope/intercept/R2/p."""
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask].astype(float)
    y = y[mask].astype(float)
    if len(x) < 4 or np.std(x) <= 0.0 or np.std(y) <= 0.0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    slope, intercept, r_value, p_value, _ = stats.linregress(x, y)
    return float(slope), float(intercept), float(r_value**2), float(p_value)


def regression_summaries(summary: pd.DataFrame) -> pd.DataFrame:
    """Fit log-std association regressions for each metric."""
    rows: list[RegressionResult] = []
    predictors = ["log_mean", "phase_hhi", "phase_effective_support", "max_phase_weight", "phase_imbalance", "total_starcoder_epochs"]
    for metric, group in summary.groupby("metric"):
        for predictor in predictors:
            slope, intercept, r2, p_value = fit_simple_ols(
                pd.to_numeric(group[predictor], errors="coerce").to_numpy(dtype=float),
                pd.to_numeric(group["log_std"], errors="coerce").to_numpy(dtype=float),
            )
            rows.append(
                RegressionResult(
                    metric=metric,
                    target="log_std",
                    predictor=predictor,
                    n=int(len(group)),
                    slope=slope,
                    intercept=intercept,
                    r2=r2,
                    p_value=p_value,
                )
            )
    return pd.DataFrame([row.__dict__ for row in rows])


def transform_summary(frame: pd.DataFrame, metric_columns: list[str]) -> pd.DataFrame:
    """Compare raw and log-scale heteroskedasticity ratios."""
    rows = []
    for metric in metric_columns:
        work = frame[["anchor_id", metric]].copy()
        work[metric] = pd.to_numeric(work[metric], errors="coerce")
        if work[metric].min(skipna=True) <= 0.0:
            continue
        raw_stds = work.groupby("anchor_id")[metric].std(ddof=1).replace(0.0, np.nan).dropna()
        log_stds = np.log(work[metric]).groupby(work["anchor_id"]).std(ddof=1).replace(0.0, np.nan).dropna()
        for exclude_starcoder_only in [False, True]:
            raw = raw_stds.copy()
            logv = log_stds.copy()
            if exclude_starcoder_only:
                raw = raw.drop(index="starcoder_only", errors="ignore")
                logv = logv.drop(index="starcoder_only", errors="ignore")
            if len(raw) < 3 or len(logv) < 3:
                continue
            rows.append(
                {
                    "metric": metric,
                    "exclude_starcoder_only": exclude_starcoder_only,
                    "raw_std_max_over_min": float(raw.max() / raw.min()),
                    "log_std_max_over_min": float(logv.max() / logv.min()),
                    "ratio_reduction": float((raw.max() / raw.min()) / (logv.max() / logv.min())),
                }
            )
    return pd.DataFrame(rows)


def residual_correlation(frame: pd.DataFrame, metric_columns: list[str]) -> pd.DataFrame:
    """Compute residual correlations after subtracting anchor means."""
    residuals = pd.DataFrame({"anchor_id": frame["anchor_id"], "repeat_index": frame["repeat_index"]})
    usable_metrics = []
    for metric in metric_columns:
        values = pd.to_numeric(frame[metric], errors="coerce")
        if values.notna().sum() < 10:
            continue
        residuals[metric] = values - values.groupby(frame["anchor_id"]).transform("mean")
        if residuals[metric].std(skipna=True) > 0.0:
            usable_metrics.append(metric)
    corr = residuals[usable_metrics].corr()
    pairs = []
    for i, left in enumerate(usable_metrics):
        for right in usable_metrics[i + 1 :]:
            value = corr.loc[left, right]
            if np.isfinite(value):
                pairs.append({"left_metric": left, "right_metric": right, "residual_correlation": float(value)})
    return pd.DataFrame(pairs).sort_values("residual_correlation", ascending=False)


def write_plots(output_dir: Path, summary: pd.DataFrame, tests: pd.DataFrame, regressions: pd.DataFrame, transforms: pd.DataFrame) -> None:
    """Write diagnostic plots."""
    key = summary[summary["metric"].isin(KEY_METRICS)].copy()
    if not key.empty:
        fig = px.scatter(
            key,
            x="mean",
            y="std",
            color="metric",
            hover_name="anchor_id",
            size="phase_hhi",
            log_y=True,
            title="StarCoder repeated anchors: local std vs anchor mean",
        )
        fig.update_layout(width=1100, height=750)
        fig.write_html(output_dir / "key_metric_std_vs_mean.html", config=TO_IMAGE_CONFIG)

        fig = px.scatter(
            key,
            x="phase_hhi",
            y="std",
            color="metric",
            hover_name="anchor_id",
            size="mean",
            log_y=True,
            title="StarCoder repeated anchors: local std vs phase concentration",
        )
        fig.update_layout(width=1100, height=750)
        fig.write_html(output_dir / "key_metric_std_vs_phase_hhi.html", config=TO_IMAGE_CONFIG)

    if not tests.empty:
        plot_tests = tests.sort_values("p_value").head(40)
        fig = px.bar(
            plot_tests,
            x="-log10_p_value",
            y="metric",
            orientation="h",
            color="-log10_p_value",
            color_continuous_scale="RdYlGn_r",
            title="Brown-Forsythe equal-variance evidence across StarCoder anchors",
        )
        fig.update_layout(width=1200, height=1000, yaxis={"categoryorder": "total ascending"})
        fig.write_html(output_dir / "brown_forsythe_top_metrics.html", config=TO_IMAGE_CONFIG)

    if not regressions.empty:
        plot_reg = regressions[regressions["predictor"].isin(["log_mean", "phase_hhi", "phase_effective_support"])].copy()
        plot_reg = plot_reg[plot_reg["metric"].isin(KEY_METRICS)]
        fig = px.bar(
            plot_reg,
            x="r2",
            y="metric",
            color="predictor",
            orientation="h",
            barmode="group",
            title="Key metrics: explanatory power for log local std",
        )
        fig.update_layout(width=1200, height=800, yaxis={"categoryorder": "total ascending"})
        fig.write_html(output_dir / "key_metric_log_std_regression_r2.html", config=TO_IMAGE_CONFIG)

    if not transforms.empty:
        plot_transforms = transforms[transforms["metric"].isin(KEY_METRICS)].copy()
        fig = go.Figure()
        for exclude, group in plot_transforms.groupby("exclude_starcoder_only"):
            suffix = "excluding starcoder_only" if exclude else "all anchors"
            fig.add_trace(
                go.Bar(
                    x=group["metric"],
                    y=group["raw_std_max_over_min"],
                    name=f"raw ratio, {suffix}",
                )
            )
            fig.add_trace(
                go.Bar(
                    x=group["metric"],
                    y=group["log_std_max_over_min"],
                    name=f"log ratio, {suffix}",
                )
            )
        fig.update_layout(
            width=1250,
            height=700,
            barmode="group",
            yaxis_type="log",
            title="Variance-stabilizing transform check: raw vs log metric local-std ratios",
        )
        fig.write_html(output_dir / "key_metric_transform_std_ratios.html", config=TO_IMAGE_CONFIG)


def write_report(
    output_dir: Path,
    tests: pd.DataFrame,
    regressions: pd.DataFrame,
    transforms: pd.DataFrame,
    residual_pairs: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    """Write a compact Markdown report."""
    key_summary = summary[summary["metric"].isin(KEY_METRICS)].copy()
    key_ratios = []
    for metric, group in key_summary.groupby("metric"):
        raw_ratio = group["std"].max() / group["std"].min()
        no_vertex = group[~group["anchor_id"].eq("starcoder_only")]
        key_ratios.append(
            {
                "metric": metric,
                "std_max_over_min": float(raw_ratio),
                "std_max_over_min_excluding_starcoder_only": float(no_vertex["std"].max() / no_vertex["std"].min()),
                "min_std_anchor": str(group.loc[group["std"].idxmin(), "anchor_id"]),
                "max_std_anchor": str(group.loc[group["std"].idxmax(), "anchor_id"]),
            }
        )
    key_ratio_frame = pd.DataFrame(key_ratios)
    bf_summary = {
        "metric_count": int(len(tests)),
        "reject_p05_share": float(tests["reject_p05"].mean()) if "reject_p05" in tests else float("nan"),
        "reject_p10_share": float(tests["reject_p10"].mean()) if "reject_p10" in tests else float("nan"),
    }
    key_reg = regressions[regressions["metric"].isin(KEY_METRICS)].copy()
    best_predictors = key_reg.sort_values(["metric", "r2"], ascending=[True, False]).groupby("metric").head(2)
    key_transforms = transforms[transforms["metric"].isin(KEY_METRICS)].copy()
    top_residual = residual_pairs.head(15)
    lines = [
        "# StarCoder Heteroskedasticity Mechanism Diagnostics",
        "",
        "This analysis uses the existing 10-anchor x 5-repeat StarCoder/Nemotron panel. It tests mechanism-level explanations for mixture-dependent endpoint variance without new training runs.",
        "",
        "## Local Variance Ratios",
        "",
        key_ratio_frame.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Brown-Forsythe Robust Equal-Variance Tests",
        "",
        pd.DataFrame([bf_summary]).to_markdown(index=False, floatfmt=".4f"),
        "",
        "Top metrics by equal-variance evidence:",
        "",
        tests.sort_values("p_value").head(12)[["metric", "anchor_count", "statistic", "p_value"]].to_markdown(
            index=False, floatfmt=".4g"
        ),
        "",
        "## Predictors of Log Local Std",
        "",
        best_predictors[["metric", "predictor", "slope", "r2", "p_value"]].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Variance-Stabilizing Transform Check",
        "",
        key_transforms[
            ["metric", "exclude_starcoder_only", "raw_std_max_over_min", "log_std_max_over_min", "ratio_reduction"]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Strongest Cross-Metric BPB Repeat Residual Correlations",
        "",
        top_residual.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Interpretation",
        "",
        "- StarCoder heteroskedasticity is a second-moment effect: local repeat variance changes strongly with mixture.",
        "- The high-variance vertex behavior is not explained by mixture-composition randomness, because simplex vertices have no between-domain interleaving uncertainty.",
        "- The leading mechanisms supported by this panel are effective sample size/diversification, repetition/support concentration, mean-variance coupling on the BPB scale, and local trajectory instability.",
        "- Rare valuable chunk inclusion remains plausible when simulated epoching uses variable subset membership, but it is not a sufficient global explanation and should not imply a globally skewed likelihood.",
        "- For optimization, use heteroskedastic weighting, variance-stabilizing transforms where they help, local noise estimates near competitive mixtures, and risk-adjusted candidate selection.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines))


def main() -> None:
    """Run mechanism diagnostics."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.repeats, low_memory=False)
    metric_columns = [column for column in frame.columns if is_metric_column(column)]
    frame = add_anchor_features(frame)

    summary = anchor_summary(frame, metric_columns)
    tests = brown_forsythe_tests(frame, metric_columns)
    if not tests.empty:
        tests["-log10_p_value"] = -np.log10(np.clip(tests["p_value"].to_numpy(dtype=float), 1e-300, None))
    regressions = regression_summaries(summary)
    transforms = transform_summary(frame, metric_columns)
    primary_residual_metrics = [metric for metric in metric_columns if is_primary_bpb_metric(metric)]
    residual_pairs = residual_correlation(frame, primary_residual_metrics)

    summary.to_csv(args.output_dir / "anchor_metric_noise_features.csv", index=False)
    tests.to_csv(args.output_dir / "brown_forsythe_equal_variance_tests.csv", index=False)
    regressions.to_csv(args.output_dir / "log_std_regression_summaries.csv", index=False)
    transforms.to_csv(args.output_dir / "variance_stabilizing_transform_summary.csv", index=False)
    residual_pairs.to_csv(args.output_dir / "within_anchor_residual_correlations.csv", index=False)
    write_plots(args.output_dir, summary, tests, regressions, transforms)
    write_report(args.output_dir, tests, regressions, transforms, residual_pairs, summary)

    key = summary[summary["metric"].isin(KEY_METRICS)].copy()
    key_ratio_max = float(
        key.groupby("metric")
        .apply(lambda group: group["std"].max() / group["std"].min(), include_groups=False)
        .max()
    )
    no_vertex = key[~key["anchor_id"].eq("starcoder_only")]
    key_ratio_no_vertex_max = float(
        no_vertex.groupby("metric")
        .apply(lambda group: group["std"].max() / group["std"].min(), include_groups=False)
        .max()
    )
    summary_json = {
        "output_dir": str(args.output_dir),
        "repeat_rows": int(len(frame)),
        "anchor_count": int(frame["anchor_id"].nunique()),
        "metric_count": int(len(metric_columns)),
        "key_metric_max_std_ratio": key_ratio_max,
        "key_metric_max_std_ratio_excluding_starcoder_only": key_ratio_no_vertex_max,
        "brown_forsythe_reject_p05_share": float(tests["reject_p05"].mean()) if "reject_p05" in tests else None,
        "brown_forsythe_reject_p10_share": float(tests["reject_p10"].mean()) if "reject_p10" in tests else None,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary_json, indent=2))
    print(json.dumps(summary_json, indent=2), flush=True)


if __name__ == "__main__":
    main()
