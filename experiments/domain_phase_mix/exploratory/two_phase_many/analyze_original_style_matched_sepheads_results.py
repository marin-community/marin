# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "kaleido==0.2.1",
#   "numpy",
#   "pandas",
#   "plotly",
#   "scipy",
#   "wandb",
# ]
# ///
"""Collect and analyze the matched one-/two-phase separate-heads panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import wandb
from plotly.subplots import make_subplots
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PANEL_DIR = SCRIPT_DIR / "reference_outputs" / "original_style_matched_sepheads_ablation_20260712"
TRAIN_PROJECT = "marin-community/marin"
EVAL_PROJECT = "marin-community/marin-eval"
TRAIN_TAG = "delphi-original-style-matched-sepheads-ablation"
EVAL_GROUP = "olmo_base_eval_table9_scaling_validation"
UNCHEATABLE_METRIC = "eval/uncheatable_eval/bpb"
TABLE9_METRIC = "olmo_base_easy/table9_51_component_macro_bpb"
RUN_SUFFIX = "_3e18"

# From ten independent proportional-mixture runs at 3e18. Difference SD is
# sqrt(2) times the run-level SD and is conservative for the shared-seed panel.
NOISE = {
    "uncheatable": {"run_sd": 0.00091299968961728, "difference_sd": 0.001291176543909418},
    "table9": {"run_sd": 0.003771768091801164, "difference_sd": 0.0053340855895512955},
}

PRIOR_FRONTIERS = {
    "uncheatable": {
        "prior separate-heads 2p": 0.9887123108,
        "best controlled 2p": 0.985661,
    },
    "table9": {
        "prior separate-heads 2p": 1.0676900654,
        "one-phase eff-exp DSP": 1.070728,
    },
}

POLICY_COLORS = {"1p": "#1A9850", "2p": "#D73027"}
POLICY_LABELS = {"1p": "Independently fitted one-phase", "2p": "Independently fitted two-phase"}
OBJECTIVE_SYMBOLS = {"uncheatable": "circle", "table9": "diamond"}
EXPORT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}
PHASE_FRACTIONS = np.array([0.8, 0.2])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-dir", type=Path, default=DEFAULT_PANEL_DIR)
    return parser.parse_args()


def candidate_training_run(runs: list[wandb.apis.public.Run], candidate: str) -> wandb.apis.public.Run:
    prefix = f"{candidate}{RUN_SUFFIX}-"
    matches = [run for run in runs if run.name.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"Expected one training run with prefix {prefix!r}, got {len(matches)}")
    return matches[0]


def candidate_eval_run(runs: list[wandb.apis.public.Run], candidate: str) -> wandb.apis.public.Run:
    expected_name = f"t9_{candidate}{RUN_SUFFIX}"
    matches = [run for run in runs if run.name == expected_name]
    if len(matches) != 1:
        raise ValueError(f"Expected one eval run named {expected_name!r}, got {len(matches)}")
    return matches[0]


def categorical_kl(weights: np.ndarray, reference: np.ndarray) -> float:
    values = np.clip(np.asarray(weights, dtype=float), 1e-12, 1.0)
    baseline = np.clip(np.asarray(reference, dtype=float), 1e-12, 1.0)
    return float(np.sum(values * (np.log(values) - np.log(baseline))))


def kl_decomposition(panel_dir: Path, manifest: pd.DataFrame) -> pd.DataFrame:
    """Decompose phase-conditioned KL into aggregate shift and phase information."""
    rows = []
    for record in manifest.to_dict(orient="records"):
        candidate = str(record["candidate"])
        mixture = pd.read_csv(panel_dir / "mixtures" / f"{candidate}.csv")
        proportional = mixture["proportional"].to_numpy(float)
        phase_weights = np.stack(
            [
                mixture["phase_0_weight"].to_numpy(float),
                mixture["phase_1_weight"].to_numpy(float),
            ]
        )
        aggregate = PHASE_FRACTIONS @ phase_weights
        weighted_phase_kl = sum(
            PHASE_FRACTIONS[phase] * categorical_kl(phase_weights[phase], proportional)
            for phase in range(len(PHASE_FRACTIONS))
        )
        aggregate_kl = categorical_kl(aggregate, proportional)
        phase_information = sum(
            PHASE_FRACTIONS[phase] * categorical_kl(phase_weights[phase], aggregate)
            for phase in range(len(PHASE_FRACTIONS))
        )
        identity_error = weighted_phase_kl - aggregate_kl - phase_information
        if not np.isclose(weighted_phase_kl, record["weighted_kl_to_proportional"], atol=1e-10):
            raise ValueError(f"Manifest KL mismatch for {candidate}")
        if abs(identity_error) > 1e-10:
            raise ValueError(f"KL decomposition failed for {candidate}: {identity_error}")
        rows.append(
            {
                "candidate": candidate,
                "aggregate_kl_to_proportional": aggregate_kl,
                "phase_domain_information": phase_information,
                "phase_information_fraction": phase_information / weighted_phase_kl if weighted_phase_kl else 0.0,
                "kl_identity_error": identity_error,
            }
        )
    return pd.DataFrame(rows)


def collect_results(manifest: pd.DataFrame) -> pd.DataFrame:
    api = wandb.Api(timeout=120)
    training_runs = list(api.runs(TRAIN_PROJECT, filters={"tags": {"$in": [TRAIN_TAG]}}, per_page=200))
    eval_runs = list(api.runs(EVAL_PROJECT, filters={"group": EVAL_GROUP}, per_page=1000))

    rows: list[dict[str, object]] = []
    for record in manifest.to_dict(orient="records"):
        candidate = str(record["candidate"])
        training_run = candidate_training_run(training_runs, candidate)
        eval_run = candidate_eval_run(eval_runs, candidate)
        uncheatable = training_run.summary.get(UNCHEATABLE_METRIC)
        table9 = eval_run.summary.get(TABLE9_METRIC)
        if training_run.state != "finished" or eval_run.state != "finished":
            raise ValueError(f"Incomplete candidate {candidate}: train={training_run.state}, eval={eval_run.state}")
        if uncheatable is None or table9 is None:
            raise ValueError(f"Missing headline metric for {candidate}: uncheatable={uncheatable}, table9={table9}")

        objective = str(record["objective"])
        target_value = float(uncheatable if objective == "uncheatable" else table9)
        rows.append(
            {
                **record,
                "observed_uncheatable_bpb": float(uncheatable),
                "observed_table9_macro_bpb": float(table9),
                "observed_target_bpb": target_value,
                "training_wandb_name": training_run.name,
                "training_wandb_url": training_run.url,
                "eval_wandb_name": eval_run.name,
                "eval_wandb_url": eval_run.url,
            }
        )

    results = pd.DataFrame(rows)
    if len(results) != 24 or results["candidate"].duplicated().any():
        raise ValueError(f"Expected 24 unique completed candidates, got {len(results)}")
    return results.sort_values(["objective", "policy", "kl_reg"]).reset_index(drop=True)


def path_diagnostics(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (objective, policy), group in results.groupby(["objective", "policy"]):
        spearman = stats.spearmanr(group["predicted_bpb"], group["observed_target_bpb"])
        best = group.loc[group["observed_target_bpb"].idxmin()]
        rows.append(
            {
                "objective": objective,
                "policy": policy,
                "n": len(group),
                "predicted_vs_observed_spearman": float(spearman.statistic),
                "best_candidate": best["candidate"],
                "best_kl": float(best["kl_reg"]),
                "best_observed_bpb": float(best["observed_target_bpb"]),
                "best_predicted_300m_bpb": float(best["predicted_bpb"]),
                "best_max_simulated_epoch": float(best["max_simulated_epoch"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["objective", "policy"]).reset_index(drop=True)


def paired_policy_gaps(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for objective, group in results.groupby("objective"):
        pivot = group.pivot(index="kl_reg", columns="policy", values="observed_target_bpb")
        if set(pivot.columns) != {"1p", "2p"} or len(pivot) != 6:
            raise ValueError(f"Incomplete one-/two-phase KL pairing for {objective}")
        for kl_reg, record in pivot.iterrows():
            gap = float(record["2p"] - record["1p"])
            rows.append(
                {
                    "objective": objective,
                    "kl_reg": float(kl_reg),
                    "one_phase_bpb": float(record["1p"]),
                    "two_phase_bpb": float(record["2p"]),
                    "two_minus_one_bpb": gap,
                    "two_phase_gain_bpb": -gap,
                    "two_minus_one_difference_sd": gap / NOISE[objective]["difference_sd"],
                }
            )
    return pd.DataFrame(rows).sort_values(["objective", "kl_reg"]).reset_index(drop=True)


def best_policy_comparison(paths: pd.DataFrame) -> list[dict[str, object]]:
    comparisons: list[dict[str, object]] = []
    for objective in ("uncheatable", "table9"):
        rows = paths[paths["objective"].eq(objective)].set_index("policy")
        one = rows.loc["1p"]
        two = rows.loc["2p"]
        gap = float(two["best_observed_bpb"] - one["best_observed_bpb"])
        comparisons.append(
            {
                "objective": objective,
                "best_one_phase_candidate": one["best_candidate"],
                "best_one_phase_kl": float(one["best_kl"]),
                "best_one_phase_bpb": float(one["best_observed_bpb"]),
                "best_two_phase_candidate": two["best_candidate"],
                "best_two_phase_kl": float(two["best_kl"]),
                "best_two_phase_bpb": float(two["best_observed_bpb"]),
                "best_two_minus_one_bpb": gap,
                "best_two_phase_gain_bpb": -gap,
                "best_two_minus_one_difference_sd": gap / NOISE[objective]["difference_sd"],
            }
        )
    return comparisons


def matched_distance_gaps(results: pd.DataFrame) -> pd.DataFrame:
    """Linearly compare policy paths over their overlapping realized distances."""
    rows: list[dict[str, object]] = []
    for objective, objective_rows in results.groupby("objective"):
        for distance in ("weighted_kl_to_proportional", "aggregate_tv_to_proportional"):
            one = objective_rows[objective_rows["policy"].eq("1p")].sort_values(distance)
            two = objective_rows[objective_rows["policy"].eq("2p")].sort_values(distance)
            overlap_low = max(float(one[distance].min()), float(two[distance].min()))
            overlap_high = min(float(one[distance].max()), float(two[distance].max()))
            support = np.unique(
                np.concatenate(
                    [
                        one.loc[one[distance].between(overlap_low, overlap_high), distance].to_numpy(float),
                        two.loc[two[distance].between(overlap_low, overlap_high), distance].to_numpy(float),
                        np.array([overlap_low, overlap_high]),
                    ]
                )
            )
            one_interp = np.interp(support, one[distance], one["observed_target_bpb"])
            two_interp = np.interp(support, two[distance], two["observed_target_bpb"])
            for value, one_bpb, two_bpb in zip(support, one_interp, two_interp, strict=True):
                rows.append(
                    {
                        "objective": objective,
                        "distance": distance,
                        "distance_value": float(value),
                        "interpolated_one_phase_bpb": float(one_bpb),
                        "interpolated_two_phase_bpb": float(two_bpb),
                        "interpolated_two_minus_one_bpb": float(two_bpb - one_bpb),
                        "interpolated_two_minus_one_difference_sd": float(
                            (two_bpb - one_bpb) / NOISE[objective]["difference_sd"]
                        ),
                    }
                )
    return pd.DataFrame(rows).sort_values(["objective", "distance", "distance_value"]).reset_index(drop=True)


def pareto_mask(results: pd.DataFrame) -> np.ndarray:
    values = results[["observed_uncheatable_bpb", "observed_table9_macro_bpb"]].to_numpy(float)
    mask = np.ones(len(values), dtype=bool)
    for index, value in enumerate(values):
        dominated = np.any(np.all(values <= value, axis=1) & np.any(values < value, axis=1))
        mask[index] = not dominated
    return mask


def render_primary_plot(results: pd.DataFrame, output_dir: Path) -> None:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Uncheatable BPB target", "Table-9 51-component macro BPB target"],
        horizontal_spacing=0.1,
    )
    for col, objective in enumerate(("uncheatable", "table9"), start=1):
        objective_rows = results[results["objective"].eq(objective)]
        for policy in ("1p", "2p"):
            rows = objective_rows[objective_rows["policy"].eq(policy)].sort_values("kl_reg")
            fig.add_trace(
                go.Scatter(
                    x=rows["kl_reg"],
                    y=rows["observed_target_bpb"],
                    mode="lines+markers",
                    name=POLICY_LABELS[policy],
                    legendgroup=policy,
                    showlegend=col == 1,
                    line={"color": POLICY_COLORS[policy], "width": 2.5},
                    marker={"color": POLICY_COLORS[policy], "size": 10},
                    customdata=np.column_stack(
                        [
                            rows["candidate"],
                            rows["predicted_bpb"],
                            rows["max_simulated_epoch"],
                            rows["aggregate_tv_to_proportional"],
                            rows["phase_tv"],
                        ]
                    ),
                    hovertemplate=(
                        "%{customdata[0]}<br>KL=%{x:g}<br>observed=%{y:.6f}<br>"
                        "predicted@300M=%{customdata[1]:.6f}<br>max epoch=%{customdata[2]:.3f}<br>"
                        "aggregate TV=%{customdata[3]:.3f}<br>phase TV=%{customdata[4]:.3f}<extra></extra>"
                    ),
                ),
                row=1,
                col=col,
            )
        for index, (label, value) in enumerate(PRIOR_FRONTIERS[objective].items()):
            fig.add_hline(
                y=value,
                line={"color": "#23395D", "width": 1.4, "dash": "dash" if index == 0 else "dot"},
                annotation_text=label,
                annotation_position="top left" if index == 0 else "bottom right",
                row=1,
                col=col,
            )
        fig.update_xaxes(title_text="Deployment KL coefficient", row=1, col=col)
        fig.update_yaxes(title_text="Observed BPB (lower is better)", row=1, col=col)

    fig.update_layout(
        template="plotly_white",
        width=1500,
        height=650,
        title={"text": "Matched separate-heads policy ablation at 3e18", "x": 0.5},
        font={"family": "Times New Roman, Times, serif", "size": 16, "color": "#23395D"},
        legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.18},
        margin={"l": 80, "r": 55, "t": 90, "b": 120},
    )
    stem = output_dir / "observed_primary_kl_sweeps"
    fig.write_html(stem.with_suffix(".html"), include_plotlyjs="cdn", config=EXPORT_CONFIG)
    fig.write_image(stem.with_suffix(".png"), scale=2)


def render_gap_plot(gaps: pd.DataFrame, output_dir: Path) -> None:
    fig = go.Figure()
    for objective, color in (("uncheatable", "#1A9850"), ("table9", "#D73027")):
        rows = gaps[gaps["objective"].eq(objective)].sort_values("kl_reg")
        fig.add_trace(
            go.Scatter(
                x=rows["kl_reg"],
                y=rows["two_minus_one_difference_sd"],
                mode="lines+markers",
                name=objective,
                line={"color": color, "width": 2.5},
                marker={"color": color, "size": 10},
                customdata=np.column_stack([rows["one_phase_bpb"], rows["two_phase_bpb"], rows["two_minus_one_bpb"]]),
                hovertemplate=(
                    "KL=%{x:g}<br>2p - 1p=%{customdata[2]:+.6f} BPB<br>"
                    "difference-noise SDs=%{y:+.2f}<br>1p=%{customdata[0]:.6f}<br>"
                    "2p=%{customdata[1]:.6f}<extra></extra>"
                ),
            )
        )
    fig.add_hline(y=0.0, line={"color": "#23395D", "width": 1.5})
    fig.add_hrect(y0=-1.0, y1=1.0, fillcolor="#FEE08B", opacity=0.18, line_width=0)
    fig.update_layout(
        template="plotly_white",
        width=1000,
        height=600,
        title={"text": "Matched policy gap: two-phase minus one-phase", "x": 0.5},
        xaxis_title="Deployment KL coefficient",
        yaxis_title="2p - 1p (proportional difference-noise SDs)",
        font={"family": "Times New Roman, Times, serif", "size": 16, "color": "#23395D"},
        legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.2},
        margin={"l": 100, "r": 50, "t": 90, "b": 120},
    )
    stem = output_dir / "paired_policy_gap"
    fig.write_html(stem.with_suffix(".html"), include_plotlyjs="cdn", config=EXPORT_CONFIG)
    fig.write_image(stem.with_suffix(".png"), scale=2)


def render_tradeoff_plot(results: pd.DataFrame, output_dir: Path) -> None:
    plotted = results.copy()
    plotted["pareto"] = pareto_mask(plotted)
    fig = go.Figure()
    for (objective, policy), rows in plotted.groupby(["objective", "policy"]):
        fig.add_trace(
            go.Scatter(
                x=rows["observed_uncheatable_bpb"],
                y=rows["observed_table9_macro_bpb"],
                mode="markers",
                name=f"{objective} target, {policy}",
                marker={
                    "color": POLICY_COLORS[policy],
                    "symbol": OBJECTIVE_SYMBOLS[objective],
                    "size": np.where(rows["pareto"], 15, 10),
                    "line": {"color": "#111111", "width": np.where(rows["pareto"], 2, 0.5)},
                },
                customdata=np.column_stack([rows["candidate"], rows["kl_reg"], rows["pareto"]]),
                hovertemplate=(
                    "%{customdata[0]}<br>KL=%{customdata[1]:g}<br>Uncheatable=%{x:.6f}<br>"
                    "Table-9=%{y:.6f}<br>Pareto=%{customdata[2]}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        template="plotly_white",
        width=1000,
        height=700,
        title={"text": "Cross-objective tradeoff for matched separate-heads candidates", "x": 0.5},
        xaxis_title="Uncheatable BPB (lower is better)",
        yaxis_title="Table-9 macro BPB (lower is better)",
        font={"family": "Times New Roman, Times, serif", "size": 16, "color": "#23395D"},
        legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.2},
        margin={"l": 100, "r": 50, "t": 90, "b": 140},
    )
    stem = output_dir / "cross_objective_tradeoff"
    fig.write_html(stem.with_suffix(".html"), include_plotlyjs="cdn", config=EXPORT_CONFIG)
    fig.write_image(stem.with_suffix(".png"), scale=2)


def render_distance_plot(results: pd.DataFrame, output_dir: Path) -> None:
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Uncheatable: weighted KL",
            "Table-9: weighted KL",
            "Uncheatable: aggregate TV",
            "Table-9: aggregate TV",
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )
    for row, distance in enumerate(("weighted_kl_to_proportional", "aggregate_tv_to_proportional"), start=1):
        for col, objective in enumerate(("uncheatable", "table9"), start=1):
            objective_rows = results[results["objective"].eq(objective)]
            for policy in ("1p", "2p"):
                policy_rows = objective_rows[objective_rows["policy"].eq(policy)].sort_values(distance)
                fig.add_trace(
                    go.Scatter(
                        x=policy_rows[distance],
                        y=policy_rows["observed_target_bpb"],
                        mode="lines+markers",
                        name=POLICY_LABELS[policy],
                        legendgroup=policy,
                        showlegend=row == 1 and col == 1,
                        line={"color": POLICY_COLORS[policy], "width": 2.3},
                        marker={"color": POLICY_COLORS[policy], "size": 9},
                        customdata=np.column_stack([policy_rows["candidate"], policy_rows["kl_reg"]]),
                        hovertemplate=(
                            "%{customdata[0]}<br>deployment KL=%{customdata[1]:g}<br>"
                            "realized distance=%{x:.4f}<br>observed=%{y:.6f}<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=col,
                )
            fig.update_xaxes(
                title_text="Weighted KL to proportional" if row == 1 else "Aggregate TV to proportional",
                row=row,
                col=col,
            )
            fig.update_yaxes(title_text="Observed BPB", row=row, col=col)
    fig.update_layout(
        template="plotly_white",
        width=1500,
        height=1000,
        title={"text": "Policy paths versus realized distance from proportional", "x": 0.5},
        font={"family": "Times New Roman, Times, serif", "size": 15, "color": "#23395D"},
        legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.08},
        margin={"l": 85, "r": 45, "t": 100, "b": 110},
    )
    stem = output_dir / "observed_realized_distance_paths"
    fig.write_html(stem.with_suffix(".html"), include_plotlyjs="cdn", config=EXPORT_CONFIG)
    fig.write_image(stem.with_suffix(".png"), scale=2)


def markdown_table(frame: pd.DataFrame, columns: list[str], formats: dict[str, str]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for record in frame[columns].to_dict(orient="records"):
        values = []
        for column in columns:
            value = record[column]
            values.append(formats[column].format(value) if column in formats else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_report(
    results: pd.DataFrame,
    paths: pd.DataFrame,
    gaps: pd.DataFrame,
    distance_gaps: pd.DataFrame,
    comparisons: list[dict[str, object]],
    output_dir: Path,
) -> None:
    comparison_by_objective = {str(row["objective"]): row for row in comparisons}
    lines = [
        "# Original-style matched separate-heads policy ablation: 3e18 results",
        "",
        "## Coverage",
        "",
        "- 24/24 training runs finished successfully.",
        "- 24/24 Marin-native Table-9 evals finished successfully.",
        "- Every result joins one-to-one to the reviewed candidate manifest.",
        (
            "- All candidates use data seed `690300`; this reduces seed confounding "
            "across the panel but does not make different mixtures consume identical tokens."
        ),
        "",
        "## Policy comparison",
        "",
    ]
    for objective in ("uncheatable", "table9"):
        row = comparison_by_objective[objective]
        sign = "better" if float(row["best_two_minus_one_bpb"]) < 0 else "worse"
        lines.extend(
            [
                (
                    f"- **{objective}:** best 1p is `{row['best_one_phase_candidate']}` at "
                    f"{float(row['best_one_phase_bpb']):.6f}; best 2p is `{row['best_two_phase_candidate']}` at "
                    f"{float(row['best_two_phase_bpb']):.6f}. The best 2p value is "
                    f"{abs(float(row['best_two_minus_one_bpb'])):.6f} BPB {sign} than best 1p "
                    f"({float(row['best_two_minus_one_difference_sd']):+.2f} conservative difference-noise SDs)."
                )
            ]
        )
    lines.extend(
        [
            "",
            (
                "The best-of-path comparison includes six KL choices per policy and therefore has "
                "selection noise. "
                "Use the same-KL contrasts below to diagnose policy effects; repeat the selected "
                "winner before a final scaling decision when its margin is below about two "
                "difference-noise SDs."
            ),
            (
                "The deployment KL coefficient does not induce the same realized distance for 1p "
                "and 2p. In particular, the low-KL 1p candidates move farther from proportional "
                "and repeat more data. The best-of-grid result therefore does not establish that "
                "the 1p policy class has the better attainable optimum."
            ),
            (
                "Two paths are left-edge censored rather than bracketed: Uncheatable 1p and "
                "Table-9 2p attain their best observed value at KL=0.05. The Table-9 1p winner at "
                "KL=0.075 is also a sharp single-run dip between worse KL=0.05 and KL=0.1 points."
            ),
            "",
            "## Best path points",
            "",
            *markdown_table(
                paths,
                [
                    "objective",
                    "policy",
                    "best_candidate",
                    "best_kl",
                    "best_observed_bpb",
                    "best_max_simulated_epoch",
                    "predicted_vs_observed_spearman",
                ],
                {
                    "best_kl": "{:.3g}",
                    "best_observed_bpb": "{:.6f}",
                    "best_max_simulated_epoch": "{:.3f}",
                    "predicted_vs_observed_spearman": "{:.3f}",
                },
            ),
            "",
            "## Same-KL policy contrasts",
            "",
            (
                "Negative `2p - 1p` means two-phase is better. Difference-noise units use the "
                "independent-repeat baseline and are conservative because this panel shares a "
                "data-seed label."
            ),
            "",
            *markdown_table(
                gaps,
                [
                    "objective",
                    "kl_reg",
                    "one_phase_bpb",
                    "two_phase_bpb",
                    "two_minus_one_bpb",
                    "two_minus_one_difference_sd",
                ],
                {
                    "kl_reg": "{:.3g}",
                    "one_phase_bpb": "{:.6f}",
                    "two_phase_bpb": "{:.6f}",
                    "two_minus_one_bpb": "{:+.6f}",
                    "two_minus_one_difference_sd": "{:+.2f}",
                },
            ),
            "",
            "## Realized-distance comparison",
            "",
            (
                "Linear interpolation over the overlapping observed support shows how the policy "
                "paths compare at approximately matched realized distance. This is diagnostic "
                "rather than a causal estimate, but it separates policy efficiency from how far "
                "each KL grid traveled."
            ),
            (
                "This interpolation does not replace candidates materialized under an exact shared "
                "deployment constraint: nonlinear paths, selection noise, and heteroskedasticity "
                "remain. Maximum simulated epoch is the most operationally relevant next match."
            ),
            "",
            *markdown_table(
                distance_gaps.groupby(["objective", "distance"], as_index=False).agg(
                    min_two_minus_one_bpb=("interpolated_two_minus_one_bpb", "min"),
                    mean_two_minus_one_bpb=("interpolated_two_minus_one_bpb", "mean"),
                    max_two_minus_one_bpb=("interpolated_two_minus_one_bpb", "max"),
                ),
                [
                    "objective",
                    "distance",
                    "min_two_minus_one_bpb",
                    "mean_two_minus_one_bpb",
                    "max_two_minus_one_bpb",
                ],
                {
                    "min_two_minus_one_bpb": "{:+.6f}",
                    "mean_two_minus_one_bpb": "{:+.6f}",
                    "max_two_minus_one_bpb": "{:+.6f}",
                },
            ),
            "",
            (
                "At matched aggregate TV, 2p is better through most of the common support; the "
                "advantage disappears only at the most aggressive observed end. The weighted-KL "
                "comparison is mixed because 2p spends part of that divergence on phase "
                "reordering rather than aggregate movement."
            ),
            "",
            "## KL decomposition",
            "",
            (
                "For phase fractions gamma and aggregate mixture w_bar, the exact identity is "
                "sum_t gamma_t KL(w_t || p) = KL(w_bar || p) + "
                "sum_t gamma_t KL(w_t || w_bar). The second term is phase-domain mutual "
                "information, or weighted Jensen-Shannon divergence. It is zero for 1p."
            ),
            (
                "Across the validated 2p paths, phase information consumes 23-48% of total KL. "
                "At the same penalty coefficient, 2p aggregate KL is only 70-89% of 1p aggregate "
                "KL. Therefore the observed conservative 2p path is partly induced by the "
                "regularizer geometry and is not specific to effective-exposure DSP."
            ),
            "",
            "## Interpretation limits",
            "",
            (
                "- The 1p and 2p models were fit independently on the same 279 matched coordinates "
                "and selected their own nested-CV ridge penalty; this is the intended full-method "
                "policy ablation."
            ),
            (
                "- The deployment KL coefficient is not directly comparable across policy classes "
                "because the fitted response surfaces differ. The matched-KL rows are a useful "
                "regularization-path diagnostic, not a claim of equal realized policy distance."
            ),
            (
                "- Predicted BPB is at the 300M fitting scale, while observed BPB is at the 3e18 "
                "Delphi scale. Only path ordering, not absolute calibration, should be compared "
                "across these columns."
            ),
            (
                "- The noise normalization comes from proportional runs near one simulated epoch. "
                "It may understate variance for the selected 1p candidates near 12-13 maximum "
                "simulated epochs."
            ),
            "",
            "## Decision",
            "",
            (
                "Nothing in this panel is ready to scale. First repeat all four current winners at "
                "fresh seeds, then compare Uncheatable 1p and 2p under a shared maximum simulated "
                "epoch near 6.7. Bracket the censored paths with KL in {0.025, 0.0375}."
            ),
        ]
    )
    (output_dir / "results_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    panel_dir = args.panel_dir.resolve()
    manifest = pd.read_csv(panel_dir / "candidate_manifest.csv")
    decomposition = kl_decomposition(panel_dir, manifest)
    manifest = manifest.merge(decomposition, on="candidate", validate="one_to_one")
    results = collect_results(manifest)
    paths = path_diagnostics(results)
    gaps = paired_policy_gaps(results)
    distance_gaps = matched_distance_gaps(results)
    comparisons = best_policy_comparison(paths)

    results.to_csv(panel_dir / "observed_results.csv", index=False)
    paths.to_csv(panel_dir / "observed_path_summary.csv", index=False)
    gaps.to_csv(panel_dir / "observed_paired_policy_gaps.csv", index=False)
    distance_gaps.to_csv(panel_dir / "observed_matched_distance_gaps.csv", index=False)
    decomposition.to_csv(panel_dir / "observed_kl_decomposition.csv", index=False)
    summary = {
        "coverage": {"training_runs": 24, "native_table9_evals": 24},
        "noise": NOISE,
        "prior_frontiers": PRIOR_FRONTIERS,
        "best_policy_comparisons": comparisons,
        "path_diagnostics": paths.to_dict(orient="records"),
    }
    (panel_dir / "observed_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    render_primary_plot(results, panel_dir)
    render_gap_plot(gaps, panel_dir)
    render_tradeoff_plot(results, panel_dir)
    render_distance_plot(results, panel_dir)
    write_report(results, paths, gaps, distance_gaps, comparisons, panel_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
