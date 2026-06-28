# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "plotly"]
# ///
"""OLMoBaseEval Easy domain-deletion effect matrix for the 300M pctrl panel.

This script is intentionally an effect-size diagnostic, not a p-value
diagnostic. The SC OLMoBaseEval Easy run contains complete 300M parity and
proportional-controllability rows, but not repeated proportional OLMo rows.
Without repeated proportional OLMo measurements, we should not report the same
training-noise p-values used by the older pctrl matrix artifacts.
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
import plotly.graph_objects as go
from plotly.subplots import make_subplots

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = (
    SCRIPT_DIR / "reference_outputs" / "olmo_base_eval_sc_716_20260620" / "olmo_base_eval_sc_716_manifest.csv"
)
DEFAULT_METRICS_ROOT = SCRIPT_DIR / "reference_outputs" / "olmo_base_eval_sc_300m_metrics_20260623"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "olmo_base_easy_domain_ablation_effects_20260623"
DEFAULT_DOMAIN_METADATA = (
    SCRIPT_DIR
    / "reference_outputs"
    / "ppert_bump_vs_log_tilt_comparison_20260614"
    / "domain_ablation_vs_local_gradient_domain_comparison.csv"
)

BASELINE_RUN_NAME = "baseline_proportional"
PCTRL_PANEL = "proportional_controllability"
SCALE = "300m_6b"
KEY_PREFIX = "olmo_base_eval/easy_bpb"
TO_IMAGE_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


@dataclass(frozen=True)
class MetricValue:
    """One scalar OLMo metric extracted from ``metrics.json`` summary."""

    benchmark_key: str
    olmo_task: str
    value: float
    metric_path: str
    metric_group: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--metrics-root", type=Path, default=DEFAULT_METRICS_ROOT)
    parser.add_argument("--domain-metadata", type=Path, default=DEFAULT_DOMAIN_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def safe_metric_segment(value: str) -> str:
    """Return a stable display/key segment for an OLMo task name."""
    return value.replace(":", "_").replace("/", "_")


def finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def metric_group(task_name: str, metric_path: str) -> str:
    if metric_path == "primary_score:average":
        return "aggregate"
    if task_name.startswith("mmlu_"):
        return "mmlu_leaf"
    if task_name.startswith("mt_mbpp_") or task_name.startswith("mbpp:"):
        return "code_leaf"
    if task_name.startswith("minerva_math_"):
        return "math_leaf"
    return "qa_leaf"


def metric_values_from_json(path: Path) -> list[MetricValue]:
    data = json.loads(path.read_text())
    summary = data.get("summary")
    if not isinstance(summary, dict):
        raise ValueError(f"Missing summary object in {path}")

    values: list[MetricValue] = []
    for task_name, row in sorted(summary.items()):
        if not isinstance(row, dict):
            continue
        metric_path = str(row.get("metric", ""))
        score = finite_float(row.get("score"))
        if score is None:
            continue
        # Keep true BPB leaves plus suite/group BPB aggregates. Exclude the
        # known CSQA accuracy/logprob leaf in this OLMo-Eval output format.
        if metric_path not in {"bits_per_byte:bits_per_byte", "primary_score:average"}:
            continue
        if ":bpb" not in task_name and not task_name.endswith(":bpb") and task_name != "mmlu:bpb":
            continue
        benchmark_key = f"{KEY_PREFIX}/{safe_metric_segment(task_name)}/bpb"
        values.append(
            MetricValue(
                benchmark_key=benchmark_key,
                olmo_task=task_name,
                value=score,
                metric_path=metric_path,
                metric_group=metric_group(task_name, metric_path),
            )
        )
    if not values:
        raise ValueError(f"No OLMo BPB values found in {path}")
    return values


def load_run_metrics(manifest: pd.DataFrame, metrics_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in manifest.itertuples(index=False):
        metrics_path = metrics_root / "outputs" / str(row.output_name) / "metrics.json"
        if not metrics_path.is_file():
            continue
        for metric in metric_values_from_json(metrics_path):
            rows.append(
                {
                    "index": int(row.index),
                    "scale": str(row.scale),
                    "panel": str(row.panel),
                    "run_name": str(row.run_name),
                    "source_experiment": str(row.source_experiment),
                    "wandb_run_id": str(row.wandb_run_id),
                    "output_name": str(row.output_name),
                    "benchmark_key": metric.benchmark_key,
                    "olmo_task": metric.olmo_task,
                    "metric_path": metric.metric_path,
                    "metric_group": metric.metric_group,
                    "value_bpb": metric.value,
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"No local metrics found under {metrics_root}")
    return frame


def domain_slug(domain: str) -> str:
    return domain.replace("/", "_")


def load_domain_metadata(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=["target_domain", "base_mass"])
    frame = frame.drop_duplicates().copy()
    frame["domain_slug"] = frame["target_domain"].map(domain_slug)
    if frame["domain_slug"].duplicated().any():
        duplicated = frame.loc[frame["domain_slug"].duplicated(), "domain_slug"].tolist()
        raise ValueError(f"Duplicate domain slugs: {duplicated}")
    return frame


def build_effect_table(metrics: pd.DataFrame, domain_metadata: pd.DataFrame) -> pd.DataFrame:
    baseline = metrics[
        metrics["scale"].eq(SCALE) & metrics["panel"].eq("parity") & metrics["run_name"].eq(BASELINE_RUN_NAME)
    ].copy()
    if baseline.empty:
        raise ValueError("Missing 300M baseline_proportional OLMo metrics")
    baseline = baseline.rename(columns={"value_bpb": "baseline_bpb"})
    baseline = baseline[["benchmark_key", "olmo_task", "metric_path", "metric_group", "baseline_bpb"]].drop_duplicates()

    deletion = metrics[
        metrics["scale"].eq(SCALE)
        & metrics["panel"].eq(PCTRL_PANEL)
        & metrics["run_name"].str.startswith("pctrl_del_", na=False)
    ].copy()
    if deletion.empty:
        raise ValueError("Missing 300M pctrl domain-deletion OLMo metrics")
    deletion["domain_slug"] = deletion["run_name"].str.removeprefix("pctrl_del_")
    deletion = deletion.merge(domain_metadata, on="domain_slug", how="left", validate="many_to_one")
    if deletion["target_domain"].isna().any():
        missing = sorted(deletion.loc[deletion["target_domain"].isna(), "domain_slug"].unique())
        raise ValueError(f"Could not map deletion run names to domains: {missing[:20]}")

    merged = deletion.merge(
        baseline,
        on=["benchmark_key", "olmo_task", "metric_path", "metric_group"],
        how="inner",
        validate="many_to_one",
    )
    merged = merged.rename(columns={"value_bpb": "deletion_bpb"})
    merged["delta_bpb"] = merged["deletion_bpb"] - merged["baseline_bpb"]
    merged["utility_delta"] = -merged["delta_bpb"]
    merged["relative_delta_bpb_pct"] = 100.0 * merged["delta_bpb"] / merged["baseline_bpb"].replace(0.0, np.nan)
    merged["deletion_hurts"] = merged["delta_bpb"] > 0.0
    keep = [
        "benchmark_key",
        "olmo_task",
        "metric_path",
        "metric_group",
        "target_domain",
        "base_mass",
        "baseline_bpb",
        "deletion_bpb",
        "delta_bpb",
        "utility_delta",
        "relative_delta_bpb_pct",
        "deletion_hurts",
        "run_name",
        "wandb_run_id",
        "output_name",
    ]
    return merged[keep].sort_values(["metric_group", "benchmark_key", "target_domain"])


def summarize_metrics(cell: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, group in cell.groupby("benchmark_key", sort=False):
        worst = group.loc[group["delta_bpb"].idxmax()]
        best = group.loc[group["delta_bpb"].idxmin()]
        rows.append(
            {
                "benchmark_key": key,
                "olmo_task": str(worst["olmo_task"]),
                "metric_group": str(worst["metric_group"]),
                "baseline_bpb": float(worst["baseline_bpb"]),
                "max_harm_delta_bpb": float(worst["delta_bpb"]),
                "max_harm_domain": str(worst["target_domain"]),
                "max_improve_delta_bpb": float(best["delta_bpb"]),
                "max_improve_domain": str(best["target_domain"]),
                "mean_delta_bpb": float(group["delta_bpb"].mean()),
                "median_delta_bpb": float(group["delta_bpb"].median()),
                "fraction_domains_harm": float(group["deletion_hurts"].mean()),
                "n_domains": int(len(group)),
            }
        )
    return pd.DataFrame(rows).sort_values(["metric_group", "max_harm_delta_bpb"], ascending=[True, False])


def summarize_domains(cell: pd.DataFrame) -> pd.DataFrame:
    grouped = cell.groupby("target_domain", as_index=False).agg(
        base_mass=("base_mass", "first"),
        mean_delta_bpb=("delta_bpb", "mean"),
        median_delta_bpb=("delta_bpb", "median"),
        max_delta_bpb=("delta_bpb", "max"),
        min_delta_bpb=("delta_bpb", "min"),
        fraction_metrics_harm=("deletion_hurts", "mean"),
    )
    return grouped.sort_values(["mean_delta_bpb", "fraction_metrics_harm"], ascending=[False, False])


def short_label(benchmark_key: str) -> str:
    label = benchmark_key.removeprefix(f"{KEY_PREFIX}/").removesuffix("/bpb")
    label = label.replace("_bpb_olmo3base", "").replace("_olmo3base_bpb", "")
    label = label.replace("_rc_bpb", "").replace("_bpb", "")
    return label


def write_effect_heatmap(
    cell: pd.DataFrame, metric_summary: pd.DataFrame, domain_summary: pd.DataFrame, output_dir: Path
) -> Path:
    order_metrics = metric_summary["benchmark_key"].tolist()
    order_domains = domain_summary["target_domain"].tolist()
    z = cell.pivot(index="benchmark_key", columns="target_domain", values="delta_bpb").reindex(
        index=order_metrics, columns=order_domains
    )
    hover_frame = cell.copy()
    hover_frame["hover"] = (
        "<b>"
        + hover_frame["olmo_task"].astype(str)
        + "</b><br>"
        + "deleted domain: "
        + hover_frame["target_domain"].astype(str)
        + "<br>"
        + "baseline BPB: "
        + hover_frame["baseline_bpb"].map("{:.5g}".format)
        + "<br>"
        + "deletion BPB: "
        + hover_frame["deletion_bpb"].map("{:.5g}".format)
        + "<br>"
        + "delta BPB: "
        + hover_frame["delta_bpb"].map("{:+.5g}".format)
        + "<br>"
        + "utility delta: "
        + hover_frame["utility_delta"].map("{:+.5g}".format)
        + "<br>"
        + "deleted mass: "
        + hover_frame["base_mass"].map("{:.4g}".format)
    )
    hover = hover_frame.pivot(index="benchmark_key", columns="target_domain", values="hover").reindex(
        index=order_metrics, columns=order_domains
    )
    y_labels = [short_label(metric) for metric in z.index]
    max_abs = float(np.nanquantile(np.abs(z.to_numpy(dtype=float)), 0.995))
    max_abs = max(max_abs, 1e-6)

    fig = go.Figure(
        go.Heatmap(
            z=z.to_numpy(dtype=float),
            x=z.columns.tolist(),
            y=y_labels,
            text=hover.to_numpy(),
            hovertemplate="%{text}<extra></extra>",
            colorscale="RdYlGn_r",
            zmin=-max_abs,
            zmax=max_abs,
            colorbar={"title": "ΔBPB<br>deletion - proportional"},
        )
    )
    fig.update_layout(
        title=(
            "OLMoBaseEval Easy 300M domain-deletion effects<br>"
            "<sup>Positive ΔBPB means deleting the bucket worsened BPB. "
            "This is not a p-value matrix: OLMo proportional repeats are unavailable.</sup>"
        ),
        xaxis={"title": "deleted bucket", "tickangle": 45, "side": "top"},
        yaxis={"title": "OLMoBaseEval Easy BPB metric", "automargin": True},
        width=1850,
        height=max(900, 18 * len(y_labels) + 260),
        margin={"l": 360, "r": 90, "t": 280, "b": 80},
        template="plotly_white",
    )
    output_path = output_dir / "olmo_base_easy_domain_deletion_effect_matrix.html"
    fig.write_html(output_path, include_plotlyjs="cdn", config=TO_IMAGE_CONFIG)
    maybe_write_image(fig, output_dir / "olmo_base_easy_domain_deletion_effect_matrix.png")
    return output_path


def write_histograms(
    cell: pd.DataFrame, metric_summary: pd.DataFrame, domain_summary: pd.DataFrame, output_dir: Path
) -> Path:
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "All deleted-domain × OLMo metric ΔBPB cells",
            "Per-metric worst deletion ΔBPB",
            "Per-domain mean ΔBPB across OLMo metrics",
            "Per-domain fraction of OLMo metrics harmed",
        ),
    )
    fig.add_trace(
        go.Histogram(x=cell["delta_bpb"], nbinsx=80, marker_color="#4575b4", name="cell ΔBPB"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Histogram(x=metric_summary["max_harm_delta_bpb"], nbinsx=50, marker_color="#d73027", name="metric max harm"),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Histogram(x=domain_summary["mean_delta_bpb"], nbinsx=39, marker_color="#fdae61", name="domain mean ΔBPB"),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Histogram(
            x=domain_summary["fraction_metrics_harm"],
            nbinsx=20,
            marker_color="#1a9850",
            name="domain harm fraction",
        ),
        row=2,
        col=2,
    )
    for row in (1, 2):
        for col in (1, 2):
            fig.update_xaxes(zeroline=True, zerolinecolor="black", row=row, col=col)
    fig.update_layout(
        title=(
            "OLMoBaseEval Easy 300M domain-deletion effect histograms<br>"
            "<sup>ΔBPB = BPB(deleted bucket) - BPB(proportional); positive is worse.</sup>"
        ),
        width=1300,
        height=850,
        template="plotly_white",
        showlegend=False,
    )
    output_path = output_dir / "olmo_base_easy_domain_deletion_effect_histograms.html"
    fig.write_html(output_path, include_plotlyjs="cdn", config=TO_IMAGE_CONFIG)
    maybe_write_image(fig, output_dir / "olmo_base_easy_domain_deletion_effect_histograms.png")
    return output_path


def maybe_write_image(fig: go.Figure, output_path: Path) -> None:
    try:
        fig.write_image(output_path, scale=4)
    except Exception as exc:
        print(f"Skipping static image {output_path}: {exc}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest)
    metrics = load_run_metrics(manifest, args.metrics_root)
    metrics.to_csv(args.output_dir / "olmo_base_easy_300m_metrics_wide_source_long.csv", index=False)

    domain_metadata = load_domain_metadata(args.domain_metadata)
    cell = build_effect_table(metrics, domain_metadata)
    metric_summary = summarize_metrics(cell)
    domain_summary = summarize_domains(cell)

    cell.to_csv(args.output_dir / "olmo_base_easy_domain_deletion_cell_effects.csv", index=False)
    metric_summary.to_csv(args.output_dir / "olmo_base_easy_domain_deletion_metric_summary.csv", index=False)
    domain_summary.to_csv(args.output_dir / "olmo_base_easy_domain_deletion_domain_summary.csv", index=False)
    heatmap = write_effect_heatmap(cell, metric_summary, domain_summary, args.output_dir)
    histograms = write_histograms(cell, metric_summary, domain_summary, args.output_dir)

    summary = {
        "scale": SCALE,
        "baseline_run_name": BASELINE_RUN_NAME,
        "metric_count": int(cell["benchmark_key"].nunique()),
        "deleted_domain_count": int(cell["target_domain"].nunique()),
        "cell_count": int(len(cell)),
        "metric_groups": {
            str(k): int(v)
            for k, v in cell[["benchmark_key", "metric_group"]].drop_duplicates()["metric_group"].value_counts().items()
        },
        "excluded_note": (
            "CSQA leaf is excluded because this OLMo-Eval output reports accuracy/logprob instead of BPB for that leaf."
        ),
        "pvalue_note": (
            "No OLMoBaseEval Easy p-values are reported because the 300M SC run did not include repeated proportional OLMo measurements."
        ),
        "max_harm": metric_summary.head(20).to_dict(orient="records"),
        "most_harmful_domains_by_mean_delta_bpb": domain_summary.head(20).to_dict(orient="records"),
        "outputs": {
            "cell_effects_csv": str(args.output_dir / "olmo_base_easy_domain_deletion_cell_effects.csv"),
            "metric_summary_csv": str(args.output_dir / "olmo_base_easy_domain_deletion_metric_summary.csv"),
            "domain_summary_csv": str(args.output_dir / "olmo_base_easy_domain_deletion_domain_summary.csv"),
            "heatmap_html": str(heatmap),
            "histograms_html": str(histograms),
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
