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
"""Fit DCLM Core v2 latent-factor targets and DSP surrogates.

This tests whether a v4-style latent factor target helps DCLM Core modeling
relative to the direct all-component smooth macros. Factor targets are
diagnostic unless explicitly marked pre-registered: the hard-aligned blend uses
hard DCLM outcomes to select factor weights on the same 300M panel.
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
from sklearn.decomposition import FactorAnalysis, PCA

from experiments.domain_phase_mix.exploratory.two_phase_many.analyze_dclm_all22_smooth_dsp_300m import (
    OUTPUT_DIR as SMOOTH_OUTPUT_DIR,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.analyze_dclm_all22_smooth_dsp_300m import (
    add_hard_coupling,
    add_smooth_targets,
    hard_component_variance_audit,
    load_joined_frame,
    safe_corr,
    write_weights_plot,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.fit_dclm_core_dsp_300m import (
    METADATA_CSV,
    RAW_MATRIX_CSV,
    TARGET_COLUMN,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.analyze_dclm_all22_smooth_dsp_300m import (
    DEFAULT_DCLM_MATRIX_CSV,
    fit_target,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.standalone_code import dsp_exact as dsp

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "dclm_latent_factor_dsp_300m_20260614_repeatcopy128"
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


@dataclass(frozen=True)
class ComponentGroup:
    """A component set used to fit latent factors."""

    name: str
    aliases: list[str]
    note: str


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-matrix-csv", type=Path, default=RAW_MATRIX_CSV)
    parser.add_argument("--dclm-matrix-csv", type=Path, default=DEFAULT_DCLM_MATRIX_CSV)
    parser.add_argument("--metadata-csv", type=Path, default=METADATA_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--variant", choices=sorted(dsp.VARIANTS), default="effective_exposure")
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument("--coarse-top-k", type=int, default=4)
    parser.add_argument("--basin-hopping-iters", type=int, default=0)
    parser.add_argument("--optimum-starts", type=int, default=160)
    parser.add_argument("--max-observed-starts", type=int, default=241)
    parser.add_argument("--factor-count", type=int, default=5)
    return parser.parse_args()


def finite_zscore_array(values: np.ndarray) -> np.ndarray:
    """Return a z-scored array with NaNs preserved."""
    out = np.asarray(values, dtype=float).copy()
    mask = np.isfinite(out)
    if int(mask.sum()) < 2:
        return out * np.nan
    std = float(np.std(out[mask], ddof=1))
    if not np.isfinite(std) or std <= 0.0:
        return out * np.nan
    out[mask] = (out[mask] - float(np.mean(out[mask]))) / std
    return out


def component_matrix(frame: pd.DataFrame, aliases: list[str]) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Return complete-case z-utility matrix for the selected aliases."""
    columns = [f"dclm_smooth/{alias}/z_utility" for alias in aliases]
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing z-utility columns: {missing}")
    values = frame[columns].to_numpy(dtype=float)
    complete_mask = np.isfinite(values).all(axis=1)
    if int(complete_mask.sum()) < 3:
        raise ValueError(f"Component group {aliases!r} has fewer than 3 complete rows")
    return values[complete_mask], columns, complete_mask


def orient_columns(scores: np.ndarray, loadings: np.ndarray, anchor: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orient factor score columns to have nonnegative correlation with anchor."""
    oriented_scores = np.asarray(scores, dtype=float).copy()
    oriented_loadings = np.asarray(loadings, dtype=float).copy()
    signs = np.ones(oriented_scores.shape[1], dtype=float)
    for index in range(oriented_scores.shape[1]):
        corr = safe_corr(oriented_scores[:, index], anchor, "pearson")
        if np.isfinite(corr) and corr < 0.0:
            oriented_scores[:, index] *= -1.0
            oriented_loadings[:, index] *= -1.0
            signs[index] = -1.0
    return oriented_scores, oriented_loadings, signs


def factor_count_for(values: np.ndarray, requested: int) -> int:
    """Return a safe factor count."""
    return max(1, min(requested, values.shape[1] - 1, values.shape[0] - 2))


def add_group_targets(
    frame: pd.DataFrame,
    group: ComponentGroup,
    *,
    factor_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Add PCA/FA targets for one component group."""
    out = frame.copy()
    values, _columns, complete_mask = component_matrix(out, group.aliases)
    row_indices = np.flatnonzero(complete_mask)
    mean_anchor = values.mean(axis=1)
    hard = pd.to_numeric(out.loc[complete_mask, TARGET_COLUMN], errors="coerce").to_numpy(dtype=float)
    k = factor_count_for(values, factor_count)

    target_rows: list[dict[str, Any]] = []
    loading_frames: list[pd.DataFrame] = []
    score_frames: list[pd.DataFrame] = []

    pca = PCA(n_components=k, random_state=0)
    pca_scores = pca.fit_transform(values)
    pca_loadings = pca.components_.T
    pca_scores, pca_loadings, pca_signs = orient_columns(pca_scores, pca_loadings, mean_anchor)
    pca_pc1_column = f"dclm_factor/{group.name}/pca_pc1"
    out[pca_pc1_column] = np.nan
    out.loc[out.index[row_indices], pca_pc1_column] = finite_zscore_array(pca_scores[:, 0])
    target_rows.append(
        {
            "target_name": pca_pc1_column,
            "group": group.name,
            "method": "pca_pc1",
            "factor_count": k,
            "component_count": len(group.aliases),
            "component_group_note": group.note,
            "posthoc_hard_aligned": False,
        }
    )

    fa = FactorAnalysis(n_components=k, rotation="varimax", random_state=0)
    fa_scores = fa.fit_transform(values)
    fa_loadings = fa.components_.T.copy()
    fa_scores, fa_loadings, fa_signs = orient_columns(fa_scores, fa_loadings, mean_anchor)
    fa_mean = fa_scores.mean(axis=1)
    fa_mean_column = f"dclm_factor/{group.name}/fa_varimax_mean"
    out[fa_mean_column] = np.nan
    out.loc[out.index[row_indices], fa_mean_column] = finite_zscore_array(fa_mean)
    target_rows.append(
        {
            "target_name": fa_mean_column,
            "group": group.name,
            "method": "fa_varimax_mean",
            "factor_count": k,
            "component_count": len(group.aliases),
            "component_group_note": group.note,
            "posthoc_hard_aligned": False,
        }
    )

    hard_corr = np.asarray([safe_corr(fa_scores[:, index], hard, "spearman") for index in range(k)], dtype=float)
    hard_weights = np.clip(np.nan_to_num(hard_corr, nan=0.0), 0.0, None)
    if float(hard_weights.sum()) > 0.0:
        hard_weights = hard_weights / hard_weights.sum()
        hard_blend = fa_scores @ hard_weights
        hard_blend_column = f"dclm_factor/{group.name}/fa_varimax_posthoc_hard_positive_blend"
        out[hard_blend_column] = np.nan
        out.loc[out.index[row_indices], hard_blend_column] = finite_zscore_array(hard_blend)
        target_rows.append(
            {
                "target_name": hard_blend_column,
                "group": group.name,
                "method": "fa_varimax_posthoc_hard_positive_blend",
                "factor_count": k,
                "component_count": len(group.aliases),
                "component_group_note": group.note,
                "posthoc_hard_aligned": True,
                "hard_positive_factor_count": int(np.sum(hard_weights > 0.0)),
            }
        )

    for method, loadings, signs, extra in (
        ("pca", pca_loadings, pca_signs, {"explained_variance_ratio": pca.explained_variance_ratio_}),
        ("fa_varimax", fa_loadings, fa_signs, {"hard_spearman": hard_corr}),
    ):
        rows = []
        for component_idx, alias in enumerate(group.aliases):
            row: dict[str, Any] = {"group": group.name, "method": method, "alias": alias}
            for factor_idx in range(k):
                row[f"F{factor_idx + 1}"] = float(loadings[component_idx, factor_idx])
            rows.append(row)
        loading_frames.append(pd.DataFrame.from_records(rows))
        factor_rows = []
        for factor_idx in range(k):
            row = {
                "group": group.name,
                "method": method,
                "factor": f"F{factor_idx + 1}",
                "orientation_sign": float(signs[factor_idx]),
                "mean_anchor_spearman": safe_corr(
                    pca_scores[:, factor_idx] if method == "pca" else fa_scores[:, factor_idx],
                    mean_anchor,
                    "spearman",
                ),
                "hard_macro_spearman": safe_corr(
                    pca_scores[:, factor_idx] if method == "pca" else fa_scores[:, factor_idx],
                    hard,
                    "spearman",
                ),
            }
            for key, values_extra in extra.items():
                row[key] = float(values_extra[factor_idx])
            factor_rows.append(row)
        score_frames.append(pd.DataFrame.from_records(factor_rows))

    return out, pd.DataFrame.from_records(target_rows), pd.concat(loading_frames, ignore_index=True), pd.concat(
        score_frames,
        ignore_index=True,
    )


def build_component_groups(component_map: pd.DataFrame, hard_audit: pd.DataFrame) -> list[ComponentGroup]:
    """Return the DCLM factor component groups."""
    all_aliases = component_map["alias"].astype(str).tolist()
    anchor_aliases = component_map.loc[component_map["proportional_available"], "alias"].astype(str).tolist()
    merged = component_map.merge(hard_audit[["alias", "hard_unique"]], on="alias", how="left")
    hard_positive = merged.loc[
        merged["proportional_available"]
        & merged["hard_unique"].gt(1)
        & pd.to_numeric(merged["utility_vs_hard_spearman"], errors="coerce").gt(0.0),
        "alias",
    ].astype(str).tolist()
    return [
        ComponentGroup("proportional_anchor20", anchor_aliases, "20 components with proportional smooth coverage"),
        ComponentGroup("all22_complete", all_aliases, "all 22 components; complete-case rows only"),
        ComponentGroup(
            "posthoc_hard_signal_positive12",
            hard_positive,
            "diagnostic-only hard-signal-positive subset selected on this 300M panel",
        ),
    ]


def write_factor_loadings_plot(loadings: pd.DataFrame, output_path: Path) -> None:
    """Write an HTML heatmap of factor loadings."""
    factor_columns = [column for column in loadings.columns if column.startswith("F")]
    plot_frame = loadings.loc[loadings["method"].eq("fa_varimax")].copy()
    labels = plot_frame["group"] + " / " + plot_frame["alias"]
    fig = go.Figure(
        go.Heatmap(
            z=plot_frame[factor_columns],
            x=factor_columns,
            y=labels,
            colorscale="RdBu",
            zmid=0.0,
            colorbar={"title": "loading"},
            customdata=np.stack([plot_frame["group"], plot_frame["alias"]], axis=-1),
            hovertemplate="group=%{customdata[0]}<br>alias=%{customdata[1]}<br>%{x}=%{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_white",
        width=1150,
        height=max(820, 20 * len(plot_frame)),
        margin={"l": 360, "r": 40, "t": 70, "b": 80},
        title={"text": "DCLM varimax factor loadings", "x": 0.5},
    )
    fig.write_html(output_path, include_plotlyjs="cdn", config=PLOT_CONFIG)


def write_fit_plot(fit_summary: pd.DataFrame, output_path: Path) -> None:
    """Write a fit/coupling summary plot for factor targets."""
    plot_frame = fit_summary.sort_values("hard_macro_spearman", ascending=True).copy()
    labels = plot_frame["target_name"].str.replace("dclm_factor/", "", regex=False)
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("DSP predictability", "hard-DCLM coupling", "raw optimum locality"),
        horizontal_spacing=0.14,
        column_widths=[0.34, 0.34, 0.32],
    )
    fig.add_trace(
        go.Bar(
            x=plot_frame["oof_spearman"],
            y=labels,
            orientation="h",
            name="OOF Spearman",
            marker={"color": plot_frame["oof_spearman"], "colorscale": "RdYlGn_r", "cmin": -1.0, "cmax": 1.0},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=plot_frame["oof_r2"], y=labels, orientation="h", name="OOF R2"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=plot_frame["hard_macro_spearman"],
            y=labels,
            orientation="h",
            showlegend=False,
            marker={"color": plot_frame["hard_macro_spearman"], "colorscale": "RdYlGn_r", "cmin": -1.0, "cmax": 1.0},
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            x=plot_frame["raw_tv_to_proportional"],
            y=labels,
            orientation="h",
            showlegend=False,
            marker={"color": plot_frame["raw_tv_to_proportional"], "colorscale": "RdYlGn_r"},
        ),
        row=1,
        col=3,
    )
    for col in (1, 2):
        fig.add_vline(x=0.0, line_dash="dash", line_color="black", row=1, col=col)
    fig.update_xaxes(title_text="OOF metric", row=1, col=1)
    fig.update_xaxes(title_text="Spearman with hard macro", row=1, col=2)
    fig.update_xaxes(title_text="TV to proportional", row=1, col=3)
    fig.update_layout(
        template="plotly_white",
        barmode="group",
        width=1700,
        height=max(760, 34 * len(plot_frame)),
        margin={"l": 430, "r": 40, "t": 75, "b": 90},
        title={"text": "DCLM latent-factor target diagnostics", "x": 0.5},
        legend={"orientation": "h", "x": 0.2, "xanchor": "center", "y": -0.1},
    )
    fig.write_html(output_path, include_plotlyjs="cdn", config=PLOT_CONFIG)


def smooth_reference_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Return direct smooth-macro references for factor-target interpretation."""
    target_columns = [
        "dclm_smooth/proportional_anchor_zscore_macro",
        "dclm_smooth/all22_complete_zscore_macro",
        "dclm_smooth/posthoc_hard_signal_positive_zscore_macro",
    ]
    rows = []
    proportional_hard_rows = frame.loc[frame["run_name"].eq("baseline_proportional"), TARGET_COLUMN]
    proportional_hard = float(proportional_hard_rows.iloc[0]) if not proportional_hard_rows.empty else np.nan
    previous_summary_path = SMOOTH_OUTPUT_DIR / "aggregate_fit_summary.csv"
    previous = pd.DataFrame()
    if previous_summary_path.exists():
        previous = pd.read_csv(previous_summary_path)
    for target_column in target_columns:
        if target_column not in frame.columns:
            continue
        target = pd.to_numeric(frame[target_column], errors="coerce")
        hard = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
        mask = frame["row_kind"].eq("signal") & frame["status"].eq("completed") & target.notna() & hard.notna()
        if not mask.any():
            continue
        best_index = target[mask].idxmax()
        previous_rows = previous.loc[previous["target_name"].eq(target_column)] if not previous.empty else pd.DataFrame()
        rows.append(
            {
                "target_name": target_column,
                "fit_row_count": int(mask.sum()),
                "hard_macro_spearman": safe_corr(target[mask], hard[mask], "spearman"),
                "hard_macro_pearson": safe_corr(target[mask], hard[mask], "pearson"),
                "best_observed_run_name": str(frame.loc[best_index, "run_name"]),
                "best_observed_hard_dclm_macro": float(frame.loc[best_index, TARGET_COLUMN]),
                "best_observed_hard_minus_proportional": float(frame.loc[best_index, TARGET_COLUMN] - proportional_hard),
                "oof_spearman": (
                    float(previous_rows["oof_spearman"].iloc[0]) if not previous_rows.empty else np.nan
                ),
                "oof_r2": float(previous_rows["oof_r2"].iloc[0]) if not previous_rows.empty else np.nan,
                "raw_tv_to_proportional": (
                    float(previous_rows["raw_tv_to_proportional"].iloc[0]) if not previous_rows.empty else np.nan
                ),
                "raw_nearest_observed_run_name": (
                    str(previous_rows["raw_nearest_observed_run_name"].iloc[0]) if not previous_rows.empty else ""
                ),
                "raw_nearest_observed_hard_minus_proportional": (
                    float(previous_rows["raw_nearest_observed_hard_minus_proportional"].iloc[0])
                    if not previous_rows.empty
                    else np.nan
                ),
                "posthoc_hard_aligned": target_column.endswith("posthoc_hard_signal_positive_zscore_macro"),
            }
        )
    return pd.DataFrame.from_records(rows)


def write_report(
    output_dir: Path,
    fit_summary: pd.DataFrame,
    smooth_reference: pd.DataFrame,
    factor_targets: pd.DataFrame,
    factor_scores: pd.DataFrame,
) -> None:
    """Write the Markdown report."""
    lines = [
        "# DCLM Core v2 Latent-Factor DSP Analysis",
        "",
        "This tests whether v4-style latent factors over exact DCLM smooth component utilities produce a better DCLM optimization target than the direct smooth macro.",
        "",
        "## Direct Smooth References",
        "",
        "| target | rows | OOF Spearman | OOF R2 | hard Spearman | best observed | best hard Δ vs prop | raw TV to prop | raw nearest | raw nearest hard Δ | posthoc? |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | ---: | --- |",
    ]
    for _, row in smooth_reference.sort_values("hard_macro_spearman", ascending=False).iterrows():
        lines.append(
            f"| `{row['target_name']}` | {int(row['fit_row_count'])} | `{row['oof_spearman']:.4f}` | `{row['oof_r2']:.4f}` | `{row['hard_macro_spearman']:.4f}` | `{row['best_observed_run_name']}` | `{row['best_observed_hard_minus_proportional']:.6f}` | `{row['raw_tv_to_proportional']:.4f}` | `{row['raw_nearest_observed_run_name']}` | `{row['raw_nearest_observed_hard_minus_proportional']:.6f}` | `{bool(row['posthoc_hard_aligned'])}` |"
        )
    lines.extend(
        [
            "",
            "## Aggregate Factor Fits",
            "",
            "| target | rows | OOF Spearman | OOF R2 | hard Spearman | best observed | best hard Δ vs prop | raw TV to prop | raw nearest | raw nearest hard Δ | posthoc? |",
            "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | ---: | --- |",
        ]
    )
    for _, row in fit_summary.sort_values("hard_macro_spearman", ascending=False).iterrows():
        lines.append(
            f"| `{row['target_name']}` | {int(row['fit_row_count'])} | `{row['oof_spearman']:.4f}` | `{row['oof_r2']:.4f}` | `{row['hard_macro_spearman']:.4f}` | `{row['best_observed_run_name']}` | `{row['best_observed_hard_minus_proportional']:.6f}` | `{row['raw_tv_to_proportional']:.4f}` | `{row['raw_nearest_observed_run_name']}` | `{row['raw_nearest_observed_hard_minus_proportional']:.6f}` | `{bool(row['posthoc_hard_aligned'])}` |"
        )
    lines.extend(
        [
            "",
            "## Construction Notes",
            "",
            "- `pca_pc1` and `fa_varimax_mean` targets are unsupervised with signs oriented toward the within-group smooth mean.",
            "- `fa_varimax_posthoc_hard_positive_blend` targets are diagnostic only: factor weights are selected from positive same-panel hard-DCLM correlations.",
            "- DSP OOF metrics are optimistic diagnostics: the shared DSP helper refits only the linear head per fold while reusing nonlinear parameters tuned on the full panel.",
            "- Raw optimum rows are model extrapolations and should not be read as deployable mixtures without a validation run.",
            "- A factor target is useful only if it beats the corresponding direct smooth reference on hard-DCLM coupling and does not move observed/nearest rows below proportional.",
            "",
            "## Factor-Hard Coupling",
            "",
        ]
    )
    for _, row in factor_scores.sort_values("hard_macro_spearman", ascending=False).head(12).iterrows():
        lines.append(
            f"- `{row['group']}` `{row['method']}` `{row['factor']}`: hard Spearman `{row['hard_macro_spearman']:.3f}`, mean-anchor Spearman `{row['mean_anchor_spearman']:.3f}`."
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Unsupervised latent factors do not produce a credible DCLM submission target at 300M. Posthoc hard-aligned factors modestly increase same-panel hard coupling over the direct all-component smooth references, but remain in-sample diagnostics, and their raw optima are far from proportional with no hard-DCLM validation support.",
            "",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run DCLM latent-factor diagnostics."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(args.metadata_csv)
    frame = load_joined_frame(args.raw_matrix_csv, args.dclm_matrix_csv)
    frame, component_map = add_smooth_targets(frame)
    hard_audit = hard_component_variance_audit(frame, component_map)
    component_groups = build_component_groups(component_map, hard_audit)
    posthoc_group = next(
        (group for group in component_groups if group.name == "posthoc_hard_signal_positive12"),
        None,
    )
    posthoc_z_columns = (
        [f"dclm_smooth/{alias}/z_utility" for alias in posthoc_group.aliases]
        if posthoc_group is not None
        else []
    )
    if len(posthoc_z_columns) >= 2:
        frame["dclm_smooth/posthoc_hard_signal_positive_zscore_macro"] = frame[posthoc_z_columns].mean(
            axis=1,
            skipna=False,
        )
    else:
        frame["dclm_smooth/posthoc_hard_signal_positive_zscore_macro"] = np.nan

    target_frames: list[pd.DataFrame] = []
    loading_frames: list[pd.DataFrame] = []
    score_frames: list[pd.DataFrame] = []
    for group in component_groups:
        if len(group.aliases) < 2:
            continue
        print(f"Building factor targets for {group.name} with {len(group.aliases)} components", flush=True)
        frame, targets, loadings, scores = add_group_targets(frame, group, factor_count=args.factor_count)
        target_frames.append(targets)
        loading_frames.append(loadings)
        score_frames.append(scores)

    factor_targets = pd.concat(target_frames, ignore_index=True)
    factor_loadings = pd.concat(loading_frames, ignore_index=True)
    factor_scores = pd.concat(score_frames, ignore_index=True)
    factor_targets.to_csv(args.output_dir / "factor_targets.csv", index=False)
    factor_loadings.to_csv(args.output_dir / "factor_loadings.csv", index=False)
    factor_scores.to_csv(args.output_dir / "factor_score_diagnostics.csv", index=False)

    variant = dsp.VARIANTS[args.variant]
    fit_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    weight_frames: list[pd.DataFrame] = []
    for index, target_column in enumerate(factor_targets["target_name"].tolist(), start=1):
        print(f"[target {index}/{len(factor_targets)}] fitting {target_column}", flush=True)
        row, predictions, weights = fit_target(
            frame,
            metadata,
            target_column,
            variant,
            maxiter=args.maxiter,
            coarse_top_k=args.coarse_top_k,
            basin_hopping_iters=args.basin_hopping_iters,
            optimize_raw=True,
            optimum_starts=args.optimum_starts,
            max_observed_starts=args.max_observed_starts,
        )
        add_hard_coupling(row, frame, target_column)
        row.update(factor_targets.loc[factor_targets["target_name"].eq(target_column)].iloc[0].to_dict())
        fit_rows.append(row)
        prediction_frames.append(predictions)
        if not weights.empty:
            weight_frames.append(weights)

    fit_summary = pd.DataFrame.from_records(fit_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    weights = pd.concat(weight_frames, ignore_index=True) if weight_frames else pd.DataFrame()
    fit_summary.to_csv(args.output_dir / "factor_fit_summary.csv", index=False)
    predictions.to_csv(args.output_dir / "factor_predictions_long.csv", index=False)
    weights.to_csv(args.output_dir / "factor_raw_optimum_weights.csv", index=False)
    frame[["run_name", TARGET_COLUMN, *factor_targets["target_name"].tolist()]].to_csv(
        args.output_dir / "factor_targets_wide.csv",
        index=False,
    )
    if not weights.empty:
        best_target = fit_summary.sort_values(["hard_macro_spearman", "oof_spearman"], ascending=False)[
            "target_name"
        ].iloc[0]
        best_weights = weights.loc[weights["target_name"].eq(best_target)].copy()
        write_weights_plot(best_weights.drop(columns=["target_name"]), args.output_dir / "best_factor_raw_weights.html")
    smooth_reference = smooth_reference_rows(frame)
    smooth_reference.to_csv(args.output_dir / "smooth_reference_fit_summary.csv", index=False)
    write_factor_loadings_plot(factor_loadings, args.output_dir / "factor_loadings.html")
    write_fit_plot(fit_summary, args.output_dir / "factor_fit_diagnostics.html")
    write_report(args.output_dir, fit_summary, smooth_reference, factor_targets, factor_scores)

    summary = {
        "dclm_matrix_csv": str(args.dclm_matrix_csv),
        "smooth_reference_output_dir": str(SMOOTH_OUTPUT_DIR),
        "output_dir": str(args.output_dir),
        "target_count": int(len(factor_targets)),
        "smooth_reference_count": int(len(smooth_reference)),
        "best_by_hard_spearman": fit_summary.sort_values("hard_macro_spearman", ascending=False)
        .head(5)
        .to_dict(orient="records"),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
