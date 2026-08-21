# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "pandas",
#     "plotly",
#     "scipy",
#     "scikit-learn",
# ]
# ///
"""Fit DSP to DCLM-aligned smooth proxy targets available in the 300M matrix.

The completed DCLM Core matrix contains hard ``raw_score`` and
``centered_accuracy`` only. This script builds diagnostic smooth proxies from
overlapping non-DCLM lm-eval metrics in the raw 300M metric matrix. These
proxies are not exact DCLM Core: several tasks use different shot counts and
many generation/BigBench tasks have no smooth proxy.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import pearsonr, spearmanr

from experiments.domain_phase_mix.exploratory.two_phase_many.fit_dclm_core_dsp_300m import (
    DCLM_MATRIX_CSV,
    METADATA_CSV,
    RAW_MATRIX_CSV,
    TARGET_COLUMN,
    prediction_metrics,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.standalone_code import dsp_exact as dsp

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "dclm_smooth_proxy_dsp_300m_20260614_repeatcopy128"
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}
PHASE_EPS = 1e-12


@dataclass(frozen=True)
class ProxyComponent:
    """One DCLM-aligned smooth proxy component."""

    dclm_task: str
    proxy_task: str
    metric_family: str
    metric_column: str
    utility_transform: str
    match_quality: str


BPB_PROXY_COMPONENTS = [
    ProxyComponent("arc_challenge_10shot", "arc_challenge_5shot", "bpb", "lm_eval/arc_challenge_5shot/bpb", "negate", "shot_mismatch"),
    ProxyComponent("arc_easy_10shot", "arc_easy_5shot", "bpb", "lm_eval/arc_easy_5shot/bpb", "negate", "shot_mismatch"),
    ProxyComponent("boolq_10shot", "boolq_10shot", "bpb", "lm_eval/boolq_10shot/bpb", "negate", "exact"),
    ProxyComponent("commonsense_qa_10shot", "csqa_5shot", "bpb", "lm_eval/csqa_5shot/bpb", "negate", "shot_mismatch_alias"),
    ProxyComponent("copa_0shot", "copa_0shot", "bpb", "lm_eval/copa_0shot/bpb", "negate", "exact"),
    ProxyComponent("hellaswag_0shot", "hellaswag_0shot", "bpb", "lm_eval/hellaswag_0shot/bpb", "negate", "exact"),
    ProxyComponent("hellaswag_10shot", "hellaswag_5shot", "bpb", "lm_eval/hellaswag_5shot/bpb", "negate", "shot_mismatch"),
    ProxyComponent("lambada_0shot", "lambada_0shot", "perplexity", "lm_eval/lambada_0shot/perplexity", "neg_log", "exact"),
    ProxyComponent("openbookqa_0shot", "openbookqa_0shot", "bpb", "lm_eval/openbookqa_0shot/bpb", "negate", "exact"),
    ProxyComponent("piqa_10shot", "piqa_5shot", "bpb", "lm_eval/piqa_5shot/bpb", "negate", "shot_mismatch"),
    ProxyComponent("winogrande_0shot", "winogrande_5shot", "bpb", "lm_eval/winogrande_5shot/bpb", "negate", "shot_mismatch"),
]

CHOICE_NORM_PROXY_COMPONENTS = [
    ProxyComponent("arc_challenge_10shot", "arc_challenge_5shot", "choice_logprob_norm", "lm_eval/arc_challenge_5shot/choice_logprob_norm", "identity", "shot_mismatch"),
    ProxyComponent("arc_easy_10shot", "arc_easy_5shot", "choice_logprob_norm", "lm_eval/arc_easy_5shot/choice_logprob_norm", "identity", "shot_mismatch"),
    ProxyComponent("boolq_10shot", "boolq_10shot", "choice_logprob_norm", "lm_eval/boolq_10shot/choice_logprob_norm", "identity", "exact"),
    ProxyComponent("commonsense_qa_10shot", "csqa_5shot", "choice_logprob_norm", "lm_eval/csqa_5shot/choice_logprob_norm", "identity", "shot_mismatch_alias"),
    ProxyComponent("copa_0shot", "copa_0shot", "choice_logprob_norm", "lm_eval/copa_0shot/choice_logprob_norm", "identity", "exact"),
    ProxyComponent("hellaswag_0shot", "hellaswag_0shot", "choice_logprob_norm", "lm_eval/hellaswag_0shot/choice_logprob_norm", "identity", "exact"),
    ProxyComponent("hellaswag_10shot", "hellaswag_5shot", "choice_logprob_norm", "lm_eval/hellaswag_5shot/choice_logprob_norm", "identity", "shot_mismatch"),
    ProxyComponent("openbookqa_0shot", "openbookqa_0shot", "choice_logprob_norm", "lm_eval/openbookqa_0shot/choice_logprob_norm", "identity", "exact"),
    ProxyComponent("piqa_10shot", "piqa_5shot", "choice_logprob_norm", "lm_eval/piqa_5shot/choice_logprob_norm", "identity", "shot_mismatch"),
    ProxyComponent("winogrande_0shot", "winogrande_5shot", "choice_logprob_norm", "lm_eval/winogrande_5shot/choice_logprob_norm", "identity", "shot_mismatch"),
]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-matrix-csv", type=Path, default=RAW_MATRIX_CSV)
    parser.add_argument("--dclm-matrix-csv", type=Path, default=DCLM_MATRIX_CSV)
    parser.add_argument("--metadata-csv", type=Path, default=METADATA_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--variant", choices=sorted(dsp.VARIANTS), default="no_penalty")
    parser.add_argument("--maxiter", type=int, default=80)
    parser.add_argument("--coarse-top-k", type=int, default=3)
    parser.add_argument("--basin-hopping-iters", type=int, default=0)
    parser.add_argument("--optimum-starts", type=int, default=240)
    parser.add_argument("--local-tv-threshold", type=float, default=0.2)
    parser.add_argument("--positive-coupling-threshold", type=float, default=0.4)
    return parser.parse_args()


def finite_zscore(values: pd.Series) -> pd.Series:
    """Return z-scored values, preserving NaNs."""
    numeric = pd.to_numeric(values, errors="coerce")
    std = float(numeric.std(ddof=1))
    if not np.isfinite(std) or std <= 0.0:
        return numeric * np.nan
    return (numeric - float(numeric.mean())) / std


def safe_corr(left: np.ndarray, right: np.ndarray, method: str) -> float:
    """Return correlation or NaN for degenerate inputs."""
    mask = np.isfinite(left) & np.isfinite(right)
    if mask.sum() < 3:
        return float("nan")
    left = left[mask]
    right = right[mask]
    if np.std(left) <= 0.0 or np.std(right) <= 0.0:
        return float("nan")
    if method == "spearman":
        return float(spearmanr(left, right).statistic)
    if method == "pearson":
        return float(pearsonr(left, right).statistic)
    raise ValueError(f"Unknown correlation method {method!r}")


def utility_from_component(frame: pd.DataFrame, component: ProxyComponent) -> pd.Series:
    """Transform a raw smooth metric into higher-is-better utility."""
    values = pd.to_numeric(frame[component.metric_column], errors="coerce")
    if component.utility_transform == "identity":
        return values
    if component.utility_transform == "negate":
        return -values
    if component.utility_transform == "neg_log":
        return -np.log(np.clip(values, PHASE_EPS, None))
    raise ValueError(f"Unknown utility transform {component.utility_transform!r}")


def merge_inputs(raw_matrix_csv: Path, dclm_matrix_csv: Path) -> pd.DataFrame:
    """Merge raw smooth metrics with hard DCLM targets."""
    raw = pd.read_csv(raw_matrix_csv, low_memory=False)
    dclm = pd.read_csv(dclm_matrix_csv, low_memory=False)
    dclm_cols = ["run_name", TARGET_COLUMN]
    dclm_cols.extend(
        column
        for column in dclm.columns
        if column.startswith("lm_eval/dclm_core/") and column.endswith("/centered_accuracy")
    )
    return raw.merge(dclm[dclm_cols], on="run_name", how="left", validate="one_to_one")


def add_proportional_tv(frame: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Annotate average phase TV from the proportional baseline."""
    out = frame.copy()
    packet_frame = out.copy()
    packet_frame["objective_metric"] = 0.0
    packet = dsp.packet_from_frame(packet_frame, metadata)
    proportional_mask = packet.frame["run_name"].eq("baseline_proportional").to_numpy()
    if not proportional_mask.any():
        out["tv_to_proportional"] = np.nan
        return out
    proportional_weights = packet.w[proportional_mask][0]
    out["tv_to_proportional"] = dsp.average_phase_tv_distance(packet.w, proportional_weights[None, :, :])
    return out


def add_proxy_columns(frame: pd.DataFrame, components: list[ProxyComponent], proxy_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add component utility/z-score columns and aggregate proxy columns."""
    out = frame.copy()
    rows = []
    z_columns = []
    for component in components:
        if component.metric_column not in out.columns:
            rows.append({**component.__dict__, "available": False, "nonnull_count": 0})
            continue
        utility_column = f"{proxy_name}/{component.dclm_task}/utility"
        z_column = f"{proxy_name}/{component.dclm_task}/z_utility"
        utility = utility_from_component(out, component)
        out[utility_column] = utility
        out[z_column] = finite_zscore(utility)
        z_columns.append(z_column)
        hard_column = f"lm_eval/dclm_core/{component.dclm_task}/centered_accuracy"
        rows.append(
            {
                **component.__dict__,
                "available": True,
                "nonnull_count": int(utility.notna().sum()),
                "utility_mean": float(utility.mean()),
                "utility_std": float(utility.std(ddof=1)),
                "hard_centered_accuracy_column": hard_column if hard_column in out.columns else None,
                "utility_vs_hard_spearman": (
                    safe_corr(utility.to_numpy(dtype=float), out[hard_column].to_numpy(dtype=float), "spearman")
                    if hard_column in out.columns
                    else np.nan
                ),
                "utility_vs_hard_pearson": (
                    safe_corr(utility.to_numpy(dtype=float), out[hard_column].to_numpy(dtype=float), "pearson")
                    if hard_column in out.columns
                    else np.nan
                ),
            }
        )
    out[f"{proxy_name}/zscore_macro"] = out[z_columns].mean(axis=1)
    return out, pd.DataFrame.from_records(rows)


def fit_target(
    frame: pd.DataFrame,
    metadata: pd.DataFrame,
    target_column: str,
    variant: dsp.DSPVariant,
    *,
    maxiter: int,
    coarse_top_k: int,
    basin_hopping_iters: int,
    optimum_starts: int,
    optimize_raw_target: bool,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Fit DSP to a higher-is-better target and optimize the raw surrogate."""
    fit_frame = frame.loc[
        frame["row_kind"].eq("signal") & frame["status"].eq("completed") & frame[target_column].notna()
    ].copy()
    fit_frame["objective_metric"] = -pd.to_numeric(fit_frame[target_column], errors="raise")
    packet = dsp.packet_from_frame(fit_frame.reset_index(drop=True), metadata)
    model, _trace = dsp.fit_variant(
        packet,
        variant,
        maxiter=maxiter,
        coarse_top_k=coarse_top_k,
        basin_hopping_iters=basin_hopping_iters,
    )
    train = -dsp.predict(model, packet.w)
    oof = -dsp.oof_predictions(packet, model)
    actual = -packet.y
    prop_mask = packet.frame["run_name"].eq("baseline_proportional").to_numpy()
    best_idx = int(np.argmax(actual))
    raw_predicted_score = np.nan
    raw_nearest_observed_run_name = None
    raw_nearest_observed_score = np.nan
    raw_nearest_observed_tv = np.nan
    weights = pd.DataFrame()
    if optimize_raw_target:
        raw_result, raw_weights = dsp.optimize_raw(
            model,
            num_starts=optimum_starts,
            observed_start_weights=packet.w,
            max_observed_starts=len(packet.w),
        )
        distances = dsp.average_phase_tv_distance(packet.w, raw_weights[None, :, :])
        nearest_idx = int(np.argmin(distances))
        raw_predicted_score = float(-raw_result.fun)
        raw_nearest_observed_run_name = str(packet.frame.iloc[nearest_idx]["run_name"])
        raw_nearest_observed_score = float(actual[nearest_idx])
        raw_nearest_observed_tv = float(distances[nearest_idx])
        weights = dsp.weights_to_frame(model, raw_weights)
        weights.insert(0, "target_column", target_column)
    row = {
        "target_column": target_column,
        "variant": variant.name,
        "fit_row_count": int(len(packet.y)),
        "target_mean": float(actual.mean()),
        "target_std": float(actual.std(ddof=1)),
        "target_range": float(actual.max() - actual.min()),
        "best_observed_run_name": str(packet.frame.iloc[best_idx]["run_name"]),
        "best_observed_score": float(actual[best_idx]),
        "proportional_score": float(actual[prop_mask][0]) if prop_mask.any() else np.nan,
        "best_minus_proportional": (
            float(actual[best_idx] - actual[prop_mask][0]) if prop_mask.any() else np.nan
        ),
        "raw_predicted_score": raw_predicted_score,
        "raw_nearest_observed_run_name": raw_nearest_observed_run_name,
        "raw_nearest_observed_score": raw_nearest_observed_score,
        "raw_nearest_observed_tv": raw_nearest_observed_tv,
        **prediction_metrics(actual, train, "train"),
        **prediction_metrics(actual, oof, "oof"),
    }
    predictions = packet.frame[["run_name"]].copy()
    predictions["target_column"] = target_column
    predictions["actual"] = actual
    predictions["train_pred"] = train
    predictions["oof_pred"] = oof
    return row, predictions, weights


def add_hard_dclm_coupling(
    row: dict[str, Any],
    frame: pd.DataFrame,
    proxy_column: str,
    *,
    local_tv_threshold: float,
) -> None:
    """Add hard-DCLM macro coupling and proxy-selection hard outcomes."""
    hard_macro = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    proxy_macro = pd.to_numeric(frame[proxy_column], errors="coerce")
    row["hard_dclm_macro_spearman"] = safe_corr(
        proxy_macro.to_numpy(dtype=float),
        hard_macro.to_numpy(dtype=float),
        "spearman",
    )
    row["hard_dclm_macro_pearson"] = safe_corr(
        proxy_macro.to_numpy(dtype=float),
        hard_macro.to_numpy(dtype=float),
        "pearson",
    )
    local_mask = frame["tv_to_proportional"].le(local_tv_threshold).fillna(False)
    local_frame = frame.loc[local_mask].copy()
    row["local_tv_threshold"] = local_tv_threshold
    row["local_row_count"] = int(local_frame[proxy_column].notna().sum())
    row["local_hard_dclm_macro_spearman"] = safe_corr(
        pd.to_numeric(local_frame[proxy_column], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(local_frame[TARGET_COLUMN], errors="coerce").to_numpy(dtype=float),
        "spearman",
    )
    row["local_hard_dclm_macro_pearson"] = safe_corr(
        pd.to_numeric(local_frame[proxy_column], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(local_frame[TARGET_COLUMN], errors="coerce").to_numpy(dtype=float),
        "pearson",
    )
    best_hard_rows = frame.loc[frame["run_name"].eq(row["best_observed_run_name"]), TARGET_COLUMN]
    prop_hard_rows = frame.loc[frame["run_name"].eq("baseline_proportional"), TARGET_COLUMN]
    nearest_hard_rows = frame.loc[frame["run_name"].eq(str(row["raw_nearest_observed_run_name"])), TARGET_COLUMN]
    proportional_hard = float(prop_hard_rows.iloc[0]) if not prop_hard_rows.empty else np.nan
    row["best_observed_hard_dclm_macro"] = float(best_hard_rows.iloc[0]) if not best_hard_rows.empty else np.nan
    row["proportional_hard_dclm_macro"] = proportional_hard
    row["best_observed_hard_minus_proportional"] = row["best_observed_hard_dclm_macro"] - proportional_hard
    row["raw_nearest_observed_hard_dclm_macro"] = (
        float(nearest_hard_rows.iloc[0]) if not nearest_hard_rows.empty else np.nan
    )
    row["raw_nearest_observed_hard_minus_proportional"] = (
        row["raw_nearest_observed_hard_dclm_macro"] - proportional_hard
    )
    if local_frame[proxy_column].notna().any():
        local_best_index = pd.to_numeric(local_frame[proxy_column], errors="coerce").idxmax()
        local_best = frame.loc[local_best_index]
        row["local_best_proxy_run_name"] = local_best["run_name"]
        row["local_best_proxy_score"] = float(local_best[proxy_column])
        row["local_best_proxy_tv"] = float(local_best["tv_to_proportional"])
        row["local_best_proxy_hard_dclm_macro"] = float(local_best[TARGET_COLUMN])
        row["local_best_proxy_hard_minus_proportional"] = float(local_best[TARGET_COLUMN]) - proportional_hard
    else:
        row["local_best_proxy_run_name"] = None
        row["local_best_proxy_score"] = np.nan
        row["local_best_proxy_tv"] = np.nan
        row["local_best_proxy_hard_dclm_macro"] = np.nan
        row["local_best_proxy_hard_minus_proportional"] = np.nan


def proxy_sets() -> dict[str, list[ProxyComponent]]:
    """Return all proxy definitions."""
    return {
        "dclm_bpb_proxy": BPB_PROXY_COMPONENTS,
        "dclm_choice_norm_proxy": CHOICE_NORM_PROXY_COMPONENTS,
    }


def write_plot(
    aggregate_summary: pd.DataFrame,
    component_summary: pd.DataFrame,
    proxy_component_map: pd.DataFrame,
    output_path: Path,
) -> None:
    """Write proxy fit diagnostics."""
    agg = aggregate_summary.sort_values("oof_spearman", ascending=True)
    comp = component_summary.sort_values("oof_spearman", ascending=True)
    map_frame = proxy_component_map.copy()
    map_frame["label"] = map_frame["proxy_name"] + "/" + map_frame["dclm_task"]
    map_frame = map_frame.sort_values("utility_vs_hard_spearman", ascending=True)
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("proxy aggregate fit", "proxy component fit", "smooth utility vs hard DCLM"),
        horizontal_spacing=0.12,
        column_widths=[0.28, 0.36, 0.36],
    )
    fig.add_trace(
        go.Bar(x=agg["oof_spearman"], y=agg["proxy_name"], orientation="h", name="OOF Spearman"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=agg["oof_r2"], y=agg["proxy_name"], orientation="h", name="OOF R2"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=comp["oof_spearman"],
            y=comp["proxy_name"] + "/" + comp["component_task"],
            orientation="h",
            marker={"color": comp["oof_spearman"], "colorscale": "RdYlGn_r"},
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            x=map_frame["utility_vs_hard_spearman"],
            y=map_frame["label"],
            orientation="h",
            marker={"color": map_frame["utility_vs_hard_spearman"], "colorscale": "RdYlGn_r"},
            showlegend=False,
        ),
        row=1,
        col=3,
    )
    for col in (1, 2, 3):
        fig.add_vline(x=0.0, line_dash="dash", line_color="black", row=1, col=col)
    fig.update_xaxes(title_text="OOF metric", row=1, col=1)
    fig.update_xaxes(title_text="OOF Spearman", row=1, col=2)
    fig.update_xaxes(title_text="Spearman", row=1, col=3)
    fig.update_layout(
        template="plotly_white",
        barmode="group",
        width=1750,
        height=920,
        margin={"l": 330, "r": 40, "t": 80, "b": 90},
        title={"text": "DCLM-aligned smooth proxy DSP diagnostic", "x": 0.5},
        legend={"orientation": "h", "x": 0.18, "xanchor": "center", "y": -0.08},
    )
    fig.write_html(output_path, include_plotlyjs="cdn", config=PLOT_CONFIG)


def write_report(
    output_dir: Path,
    aggregate_summary: pd.DataFrame,
    component_summary: pd.DataFrame,
    proxy_component_map: pd.DataFrame,
) -> None:
    """Write a Markdown report."""
    lines = [
        "# DCLM-Aligned Smooth Proxy DSP Diagnostic",
        "",
        "This uses available smooth metrics from the broader 300M matrix. It is not exact DCLM Core because the DCLM eval output has no smooth metrics and several task/shot settings only have approximate overlaps.",
        "",
        "## Aggregate Fits",
        "",
        "| proxy | components | OOF Spearman | OOF R2 | hard-DCLM Spearman | local hard-DCLM Spearman | best observed | best hard Δ vs prop | local best hard Δ vs prop | raw nearest TV |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for _, row in aggregate_summary.sort_values("oof_spearman", ascending=False).iterrows():
        lines.append(
            f"| `{row['proxy_name']}` | {int(row['component_count'])} | `{row['oof_spearman']:.4f}` | `{row['oof_r2']:.4f}` | `{row['hard_dclm_macro_spearman']:.4f}` | `{row['local_hard_dclm_macro_spearman']:.4f}` | `{row['best_observed_run_name']}` | `{row['best_observed_hard_minus_proportional']:.6f}` | `{row['local_best_proxy_hard_minus_proportional']:.6f}` | `{row['raw_nearest_observed_tv']:.4f}` |"
        )
    lines.extend(["", "## Proxy Coverage", ""])
    for proxy_name, group in proxy_component_map.groupby("proxy_name", sort=True):
        exact_count = int(group["match_quality"].eq("exact").sum())
        lines.append(
            f"- `{proxy_name}`: {len(group)} components, {exact_count} exact task/shot matches, {len(group) - exact_count} approximate shot/alias matches."
        )
    lines.extend(
        [
            "",
            "## Locality Caveat",
            "",
            f"- The requested local coupling diagnostic uses `TV <= {aggregate_summary['local_tv_threshold'].iloc[0]:.3f}` around proportional.",
            "- In this 300M swarm this is not estimable: the neighborhood contains only `baseline_proportional`. The nearest non-proportional baselines/candidates are already relatively far from proportional in average phase TV.",
            "",
            "## Interpretation",
            "",
            "- A smooth proxy can be substantially more modelable than the hard DCLM macro if its components have stable logprob/BPB signal, but modelability of the proxy is not sufficient.",
            "- The all-component smooth proxies fail the submission gate: their best observed proxy rows are directionally worse than proportional on hard DCLM macro.",
            "- These proxies should not be reported as DCLM Core and should not drive a live submission unless they also predict or improve held-out hard DCLM outcomes.",
            "- The posthoc positive-coupling subset rows are diagnostic only; their component filters use the same 300M hard-DCLM data and are not pre-registered targets.",
            "- Raw optima are extrapolative if nearest-observed TV is large; treat them as diagnostics unless validated by a trust-region/path experiment.",
            "",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run smooth-proxy DSP diagnostics."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(args.metadata_csv)
    frame = merge_inputs(args.raw_matrix_csv, args.dclm_matrix_csv)
    frame = add_proportional_tv(frame, metadata)
    variant = dsp.VARIANTS[args.variant]

    map_frames = []
    aggregate_rows = []
    component_rows = []
    prediction_frames = []
    weight_frames = []
    for proxy_name, components in proxy_sets().items():
        print(f"Fitting aggregate proxy {proxy_name}", flush=True)
        frame, map_frame = add_proxy_columns(frame, components, proxy_name)
        map_frame.insert(0, "proxy_name", proxy_name)
        map_frames.append(map_frame)
        aggregate_column = f"{proxy_name}/zscore_macro"
        aggregate_row, aggregate_predictions, aggregate_weights = fit_target(
            frame,
            metadata,
            aggregate_column,
            variant,
            maxiter=args.maxiter,
            coarse_top_k=args.coarse_top_k,
            basin_hopping_iters=args.basin_hopping_iters,
            optimum_starts=args.optimum_starts,
            optimize_raw_target=True,
        )
        aggregate_row["proxy_name"] = proxy_name
        aggregate_row["component_count"] = len(components)
        aggregate_row["proxy_kind"] = "all_components"
        add_hard_dclm_coupling(
            aggregate_row,
            frame,
            aggregate_column,
            local_tv_threshold=args.local_tv_threshold,
        )
        aggregate_rows.append(aggregate_row)
        aggregate_predictions["proxy_name"] = proxy_name
        aggregate_predictions["component_task"] = "zscore_macro"
        prediction_frames.append(aggregate_predictions)
        if not aggregate_weights.empty:
            aggregate_weights["proxy_name"] = proxy_name
            aggregate_weights["component_task"] = "zscore_macro"
            weight_frames.append(aggregate_weights)

        positive_map = map_frame.loc[
            pd.to_numeric(map_frame["utility_vs_hard_spearman"], errors="coerce").ge(args.positive_coupling_threshold)
        ].copy()
        if len(positive_map) >= 2:
            subset_proxy_name = f"{proxy_name}_positive_coupling_ge{str(args.positive_coupling_threshold).replace('.', 'p')}"
            subset_z_columns = [
                f"{proxy_name}/{task}/z_utility"
                for task in positive_map["dclm_task"].tolist()
            ]
            subset_aggregate_column = f"{subset_proxy_name}/zscore_macro"
            frame[subset_aggregate_column] = frame[subset_z_columns].mean(axis=1)
            subset_row, subset_predictions, subset_weights = fit_target(
                frame,
                metadata,
                subset_aggregate_column,
                variant,
                maxiter=args.maxiter,
                coarse_top_k=args.coarse_top_k,
                basin_hopping_iters=args.basin_hopping_iters,
                optimum_starts=args.optimum_starts,
                optimize_raw_target=True,
            )
            subset_row["proxy_name"] = subset_proxy_name
            subset_row["source_proxy_name"] = proxy_name
            subset_row["proxy_kind"] = "posthoc_positive_coupling_subset"
            subset_row["component_count"] = len(positive_map)
            subset_row["positive_coupling_threshold"] = args.positive_coupling_threshold
            subset_row["included_components"] = ",".join(positive_map["dclm_task"].tolist())
            add_hard_dclm_coupling(
                subset_row,
                frame,
                subset_aggregate_column,
                local_tv_threshold=args.local_tv_threshold,
            )
            aggregate_rows.append(subset_row)
            subset_predictions["proxy_name"] = subset_proxy_name
            subset_predictions["component_task"] = "zscore_macro"
            prediction_frames.append(subset_predictions)
            if not subset_weights.empty:
                subset_weights["proxy_name"] = subset_proxy_name
                subset_weights["component_task"] = "zscore_macro"
                weight_frames.append(subset_weights)

        for index, component in enumerate(components, start=1):
            print(f"[{proxy_name} {index}/{len(components)}] fitting {component.dclm_task}", flush=True)
            component_column = f"{proxy_name}/{component.dclm_task}/z_utility"
            row, predictions, weights = fit_target(
                frame,
                metadata,
                component_column,
                variant,
                maxiter=args.maxiter,
                coarse_top_k=args.coarse_top_k,
                basin_hopping_iters=args.basin_hopping_iters,
                optimum_starts=max(20, args.optimum_starts // 4),
                optimize_raw_target=False,
            )
            row["proxy_name"] = proxy_name
            row["component_task"] = component.dclm_task
            row["proxy_task"] = component.proxy_task
            row["metric_column"] = component.metric_column
            row["match_quality"] = component.match_quality
            component_map_row = map_frame.loc[map_frame["dclm_task"].eq(component.dclm_task)].iloc[0]
            row["utility_vs_hard_spearman"] = component_map_row["utility_vs_hard_spearman"]
            row["utility_vs_hard_pearson"] = component_map_row["utility_vs_hard_pearson"]
            component_rows.append(row)
            predictions["proxy_name"] = proxy_name
            predictions["component_task"] = component.dclm_task
            prediction_frames.append(predictions)
            if not weights.empty:
                weights["proxy_name"] = proxy_name
                weights["component_task"] = component.dclm_task
                weight_frames.append(weights)

    proxy_component_map = pd.concat(map_frames, ignore_index=True)
    aggregate_summary = pd.DataFrame.from_records(aggregate_rows)
    component_summary = pd.DataFrame.from_records(component_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    weights = pd.concat(weight_frames, ignore_index=True) if weight_frames else pd.DataFrame()
    proxy_component_map.to_csv(args.output_dir / "proxy_component_map.csv", index=False)
    aggregate_summary.to_csv(args.output_dir / "proxy_aggregate_fit_summary.csv", index=False)
    component_summary.to_csv(args.output_dir / "proxy_component_fit_summary.csv", index=False)
    predictions.to_csv(args.output_dir / "proxy_predictions_long.csv", index=False)
    weights.to_csv(args.output_dir / "proxy_raw_optimum_weights.csv", index=False)
    summary = {
        "variant": variant.name,
        "proxy_count": int(len(aggregate_summary)),
        "proxies": aggregate_summary[
            [
                "proxy_name",
                "component_count",
                "oof_spearman",
                "oof_r2",
                "hard_dclm_macro_spearman",
                "best_observed_run_name",
                "best_observed_hard_dclm_macro",
                "proportional_hard_dclm_macro",
                "raw_nearest_observed_tv",
            ]
        ].to_dict(orient="records"),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_plot(aggregate_summary, component_summary, proxy_component_map, args.output_dir / "smooth_proxy_fit_diagnostics.html")
    write_report(args.output_dir, aggregate_summary, component_summary, proxy_component_map)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
