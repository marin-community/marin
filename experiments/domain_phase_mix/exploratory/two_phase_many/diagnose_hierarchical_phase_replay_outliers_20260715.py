# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy",
#   "fsspec",
#   "gcsfs",
#   "numpy",
#   "pandas",
#   "plotly",
#   "scikit-learn",
#   "scipy",
# ]
# ///
"""Diagnose optimistic outliers of the hierarchical phase-replay surrogate.

The Observatory already contains the exact OOF and heldout predictions shown
in its scatter plots. This script treats those predictions as immutable input
and asks which policy-geometry statistics explain the largest optimistic
errors. Historical validation rows remain diagnostic heldouts; they are not
used to refit or select a surrogate.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_hierarchical_coverage_grp_20260715 as hierarchical_grp,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    export_mixture_fit_observatory as observatory,
)

DEFAULT_DASHBOARD = SCRIPT_DIR / "mixture_fit_debugger/src/generated/dashboard_data.json"
DEFAULT_HELDOUT_REGISTRY = SCRIPT_DIR / "reference_outputs/delphi_3e18_append_only_heldouts_20260714/heldout_current.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/hierarchical_phase_replay_outlier_diagnostics_20260715"
SWARM_ID = "delphi_3e18"
MODEL_ID = "hierarchical_phase_bucket_replay"
TARGET_IDS = ("uncheatable", "table9")
KEY_OUTLIERS = (
    "bfgrp_unch_eta_2p_raw_3e18-09505d",
    "bfgrp_t9_eta_2p_raw_3e18-debf41",
    "bfgrp_unch_separate_heads_2p_raw_3e18-124148",
    "bfgrp_t9_separate_heads_2p_raw_3e18-625eee",
    "symsep_t9_1p_kl0p05_3e18-5160c6",
)
TECH_CODE_DOMAINS = frozenset(
    {
        "dolma3_arxiv",
        "dolma3_finemath_3plus",
        "dolma3_stack_edu",
        "dolmino_stack_edu_fim",
        "dolmino_synth_code",
        "dolmino_synth_math",
    }
)
REASONING_DOMAINS = frozenset({"dolmino_synth_instruction", "dolmino_synth_thinking"})
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dashboard", type=Path, default=DEFAULT_DASHBOARD)
    parser.add_argument("--heldout-registry", type=Path, default=DEFAULT_HELDOUT_REGISTRY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(denominator, 1e-12))


def effective_bucket_count(weights: np.ndarray) -> float:
    positive = weights[weights > 1e-12]
    return float(np.exp(-np.sum(positive * np.log(positive))))


def family_indices(domain_ids: tuple[str, ...]) -> dict[str, np.ndarray]:
    groups = {"broad_text": [], "tech_code": [], "reasoning": []}
    for index, domain in enumerate(domain_ids):
        if domain in TECH_CODE_DOMAINS:
            groups["tech_code"].append(index)
        elif domain in REASONING_DOMAINS:
            groups["reasoning"].append(index)
        else:
            groups["broad_text"].append(index)
    return {name: np.asarray(indices, dtype=int) for name, indices in groups.items()}


def geometry_record(
    row: dict[str, Any],
    proportional: np.ndarray,
    groups: dict[str, np.ndarray],
) -> dict[str, float | int | str | bool]:
    aggregate = np.asarray(row["aggregate"], dtype=float)
    ratios = aggregate / np.maximum(proportional, 1e-12)
    top = np.sort(aggregate)[::-1]
    record: dict[str, float | int | str | bool] = {
        "row_name": row["name"],
        "row_id": row["id"],
        "split": row["split"],
        "policy_family": row["policyFamily"],
        "panel": row["panel"],
        "candidate_target": row.get("candidateTarget") or "",
        "phase_tv": float(row["diagnostics"]["phaseTv"]),
        "aggregate_tv_to_proportional": float(row["diagnostics"]["aggregateTvToProportional"]),
        "aggregate_kl_to_proportional": float(row["diagnostics"]["aggregateKlToProportional"]),
        "support_distance": float(row["diagnostics"]["supportDistance"]),
        "max_epoch": float(row["diagnostics"]["maxEpoch"]),
        "top1_mass": float(top[0]),
        "top3_mass": float(top[:3].sum()),
        "effective_bucket_count": effective_bucket_count(aggregate),
        "below_quarter_proportional_count": int(np.sum(ratios < 0.25)),
        "below_quarter_proportional_mass": float(proportional[ratios < 0.25].sum()),
        "weighted_undercoverage": float(np.sum(proportional * np.maximum(1.0 - ratios, 0.0))),
        "nearest_fit_id": row["diagnostics"]["nearestFitId"],
        "wandb_url": row.get("wandbUrl") or "",
    }
    family_ratios = []
    for family, indices in groups.items():
        ratio = safe_ratio(float(aggregate[indices].sum()), float(proportional[indices].sum()))
        record[f"{family}_coverage_ratio"] = ratio
        family_ratios.append(ratio)
    record["minimum_family_coverage_ratio"] = min(family_ratios)
    record["geometric_family_coverage_ratio"] = float(np.exp(np.mean(np.log(np.maximum(family_ratios, 1e-12)))))
    return record


def prediction_frame(bundle: dict[str, Any], registry: pd.DataFrame) -> pd.DataFrame:
    swarm = bundle["swarms"][SWARM_ID]
    rows = swarm["rows"]
    domains = tuple(domain["id"] for domain in swarm["domains"])
    proportional = np.asarray([domain["proportionalWeight"] for domain in swarm["domains"]], dtype=float)
    groups = family_indices(domains)
    records = [geometry_record(row, proportional, groups) for row in rows]
    frame = pd.DataFrame(records)
    registry_columns = [
        "wandb_run_name",
        "training_state",
        "global_step",
        "num_train_steps",
        "train_loss",
        "table9_eval_run_id",
        "table9_eval_state",
        "table9_eval_url",
    ]
    available = [column for column in registry_columns if column in registry.columns]
    if available:
        frame = frame.merge(
            registry[available].rename(columns={"wandb_run_name": "row_name"}),
            on="row_name",
            how="left",
        )
    for target in TARGET_IDS:
        observed = np.asarray(
            [float(row["observed"][target]) if row["observed"].get(target) is not None else np.nan for row in rows],
            dtype=float,
        )
        prediction = np.asarray(
            swarm["predictions"][target]["two_phase"][MODEL_ID]["prediction"],
            dtype=float,
        )
        difference_sd = float(swarm["targets"][target]["noiseReference"]["differenceStandardDeviation"])
        frame[f"{target}_observed"] = observed
        frame[f"{target}_predicted"] = prediction
        frame[f"{target}_optimism"] = observed - prediction
        frame[f"{target}_optimism_noise_sd"] = (observed - prediction) / difference_sd
    return frame


def model_comparison(bundle: dict[str, Any], frame: pd.DataFrame) -> pd.DataFrame:
    swarm = bundle["swarms"][SWARM_ID]
    row_index = {row["name"]: index for index, row in enumerate(swarm["rows"])}
    records = []
    for target in TARGET_IDS:
        model_predictions = swarm["predictions"][target]["two_phase"]
        for row_name in KEY_OUTLIERS:
            index = row_index[row_name]
            observed = float(frame.loc[frame["row_name"].eq(row_name), f"{target}_observed"].iloc[0])
            for model_id, values in model_predictions.items():
                predicted = float(values["prediction"][index])
                records.append(
                    {
                        "target": target,
                        "row_name": row_name,
                        "model_id": model_id,
                        "observed": observed,
                        "predicted": predicted,
                        "optimism": observed - predicted,
                    }
                )
    return pd.DataFrame(records)


def contribution_frame() -> pd.DataFrame:
    records = []
    for target in TARGET_IDS:
        raw = observatory.load_delphi_3e18_fit_dataset(target)
        config, _sweep = observatory.select_hierarchical_phase_replay_config(raw, "two_phase")
        model = observatory.hierarchical_phase_replay_fit(raw, np.arange(raw.n), config)
        heldout, weights = observatory.load_delphi_3e18_heldouts(raw)
        heldout_index = {name: index for index, name in enumerate(heldout["wandb_run_name"])}
        for row_name in KEY_OUTLIERS:
            index = heldout_index[row_name]
            candidate = replace(
                model.dataset,
                weights=weights[[index]],
                target=np.zeros(1, dtype=float),
            )
            design = hierarchical_grp.build_design(candidate, config)
            contribution = design.values[0] * model.coefficients
            records.append(
                {
                    "target": target,
                    "row_name": row_name,
                    "category": "intercept",
                    "contribution": model.intercept,
                }
            )
            categories: dict[str, float] = {}
            for name, value in zip(design.names, contribution, strict=True):
                category = name.split(":", 1)[0]
                categories[category] = categories.get(category, 0.0) + float(value)
            for category, value in categories.items():
                records.append(
                    {
                        "target": target,
                        "row_name": row_name,
                        "category": category,
                        "contribution": value,
                    }
                )
    return pd.DataFrame(records)


def correlation_frame(frame: pd.DataFrame) -> pd.DataFrame:
    geometry = [
        "phase_tv",
        "aggregate_tv_to_proportional",
        "aggregate_kl_to_proportional",
        "support_distance",
        "max_epoch",
        "top1_mass",
        "top3_mass",
        "effective_bucket_count",
        "below_quarter_proportional_count",
        "below_quarter_proportional_mass",
        "weighted_undercoverage",
        "broad_text_coverage_ratio",
        "tech_code_coverage_ratio",
        "reasoning_coverage_ratio",
        "minimum_family_coverage_ratio",
        "geometric_family_coverage_ratio",
    ]
    records = []
    for target in TARGET_IDS:
        for split_name, mask in {
            "fit": frame["split"].eq("fit"),
            "heldout_two_phase": frame["split"].eq("heldout") & frame["policy_family"].eq("two_phase"),
            "heldout_all": frame["split"].eq("heldout"),
        }.items():
            selected = frame.loc[mask, [f"{target}_optimism", *geometry]].dropna()
            for feature in geometry:
                correlation = float("nan")
                if selected[f"{target}_optimism"].nunique() > 1 and selected[feature].nunique() > 1:
                    correlation = float(selected[f"{target}_optimism"].corr(selected[feature], method="spearman"))
                records.append(
                    {
                        "target": target,
                        "split": split_name,
                        "feature": feature,
                        "n": len(selected),
                        "spearman_with_optimism": correlation,
                    }
                )
    return pd.DataFrame(records)


def support_bins(frame: pd.DataFrame) -> pd.DataFrame:
    records = []
    heldout = frame.loc[frame["split"].eq("heldout") & frame["policy_family"].eq("two_phase")].copy()
    heldout["support_bin"] = pd.qcut(heldout["support_distance"], q=4, duplicates="drop")
    for target in TARGET_IDS:
        for interval, group in heldout.groupby("support_bin", observed=True):
            error = group[f"{target}_predicted"] - group[f"{target}_observed"]
            optimism = -error
            records.append(
                {
                    "target": target,
                    "support_bin": str(interval),
                    "n": len(group),
                    "mean_support_distance": float(group["support_distance"].mean()),
                    "rmse": float(np.sqrt(np.mean(error**2))),
                    "mean_optimism": float(optimism.mean()),
                    "p90_optimism": float(optimism.quantile(0.9)),
                }
            )
    return pd.DataFrame(records)


def render(frame: pd.DataFrame, output_path: Path) -> None:
    heldout = frame.loc[frame["split"].eq("heldout") & frame["policy_family"].eq("two_phase")].copy()
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Uncheatable: optimism vs broad-text coverage",
            "Table-9: optimism vs broad-text coverage",
            "Uncheatable: optimism vs support distance",
            "Table-9: optimism vs support distance",
        ),
        horizontal_spacing=0.1,
        vertical_spacing=0.14,
    )
    for column, target in enumerate(TARGET_IDS, start=1):
        custom = np.column_stack(
            [
                heldout["row_name"],
                heldout["top3_mass"],
                heldout["max_epoch"],
                heldout[f"{target}_optimism_noise_sd"],
            ]
        )
        marker = {
            "size": 8,
            "color": heldout["top3_mass"],
            "colorscale": "RdYlGn_r",
            "cmin": 0.0,
            "cmax": 1.0,
            "line": {"width": 0.7, "color": "#17324d"},
            "showscale": column == 2,
            "colorbar": {"title": "Top-3<br>mass", "x": 1.01},
        }
        hover = (
            "%{customdata[0]}<br>optimism=%{y:.4f}<br>top-3 mass=%{customdata[1]:.3f}"
            "<br>max epoch=%{customdata[2]:.1f}<br>optimism / noise SD=%{customdata[3]:.1f}<extra></extra>"
        )
        figure.add_trace(
            go.Scatter(
                x=heldout["broad_text_coverage_ratio"],
                y=heldout[f"{target}_optimism"],
                mode="markers",
                marker=marker,
                customdata=custom,
                hovertemplate=hover,
                showlegend=False,
            ),
            row=1,
            col=column,
        )
        figure.add_trace(
            go.Scatter(
                x=heldout["support_distance"],
                y=heldout[f"{target}_optimism"],
                mode="markers",
                marker={**marker, "showscale": False},
                customdata=custom,
                hovertemplate=hover,
                showlegend=False,
            ),
            row=2,
            col=column,
        )
        figure.add_hline(y=0.0, line_dash="dash", line_color="#6f7f8c", row=1, col=column)
        figure.add_hline(y=0.0, line_dash="dash", line_color="#6f7f8c", row=2, col=column)
    figure.update_xaxes(title_text="Broad-text aggregate mass / proportional mass", row=1)
    figure.update_xaxes(title_text="Nearest-fit L1 distance", row=2)
    figure.update_yaxes(title_text="Observed - predicted BPB", col=1)
    figure.update_layout(
        title="Hierarchical phase-replay outlier anatomy (two-phase heldouts only)",
        template="plotly_white",
        width=1450,
        height=950,
        margin={"l": 90, "r": 120, "t": 110, "b": 80},
    )
    output_path.write_text(figure.to_html(include_plotlyjs="cdn", full_html=True, config=PLOT_CONFIG))


def markdown_table(frame: pd.DataFrame, columns: list[str], formats: dict[str, str]) -> list[str]:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [header, divider]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            values.append(formats.get(column, "{}").format(value))
        rows.append("| " + " | ".join(values) + " |")
    return rows


def write_report(
    frame: pd.DataFrame,
    comparison: pd.DataFrame,
    contributions: pd.DataFrame,
    correlations: pd.DataFrame,
    bins: pd.DataFrame,
    output_path: Path,
) -> None:
    key = frame.loc[frame["row_name"].isin(KEY_OUTLIERS)].copy()
    key = key.sort_values("uncheatable_optimism", ascending=False)
    two_phase = frame.loc[frame["split"].eq("heldout") & frame["policy_family"].eq("two_phase")]
    cross_correlation = float(two_phase["uncheatable_optimism"].corr(two_phase["table9_optimism"], method="spearman"))
    circled = "bfgrp_unch_eta_2p_raw_3e18-09505d"
    circled_row = frame.loc[frame["row_name"].eq(circled)].iloc[0]
    nearest_row = frame.loc[frame["row_id"].eq(circled_row["nearest_fit_id"])].iloc[0]
    contribution_summary = contributions.loc[contributions["row_name"].eq(circled)].copy()
    benefit_categories = {"bucket_excess_signal", "pooled_base_signal", "family_coverage_signal"}
    harm_categories = {"family_member_replay", "family_overexposure", "phase_shift_tv"}
    contribution_totals = {}
    for target in TARGET_IDS:
        selected = contribution_summary.loc[contribution_summary["target"].eq(target)]
        contribution_totals[target] = {
            "intercept": float(selected.loc[selected["category"].eq("intercept"), "contribution"].sum()),
            "benefit": float(selected.loc[selected["category"].isin(benefit_categories), "contribution"].sum()),
            "harm": float(selected.loc[selected["category"].isin(harm_categories), "contribution"].sum()),
        }
    model_check = comparison.loc[comparison["row_name"].eq(circled)].copy()
    best_by_target = {
        target: model_check.loc[model_check["target"].eq(target)].sort_values("optimism").iloc[0]
        for target in TARGET_IDS
    }
    lines = [
        "# Hierarchical phase-replay outlier diagnosis",
        "",
        "The historical 3e18 validation archive is used only as a diagnostic heldout. No heldout value is used "
        "to refit or select the surrogate.",
        "",
        "## Main finding",
        "",
        "The large optimistic errors are real and structurally coherent. The worst rows concentrate most mixture "
        "mass in a few small code/math buckets while jointly deleting broad coverage. The promoted surrogate's "
        "extrapolated utility overwhelms its replay and overexposure harms. Phase modeling cannot explain the "
        "phase-tied outlier because its phase TV is exactly zero.",
        "",
        "Across two-phase heldouts, Uncheatable and Table-9 optimism have Spearman correlation "
        f"{cross_correlation:.3f}; "
        "the shared sign across independently evaluated targets argues against a metric-export artifact.",
        "",
        "## Key rows",
        "",
        *markdown_table(
            key,
            [
                "row_name",
                "policy_family",
                "top3_mass",
                "broad_text_coverage_ratio",
                "max_epoch",
                "uncheatable_optimism",
                "uncheatable_optimism_noise_sd",
                "table9_optimism",
                "table9_optimism_noise_sd",
            ],
            {
                "top3_mass": "{:.3f}",
                "broad_text_coverage_ratio": "{:.3f}",
                "max_epoch": "{:.1f}",
                "uncheatable_optimism": "{:.4f}",
                "uncheatable_optimism_noise_sd": "{:.1f}",
                "table9_optimism": "{:.4f}",
                "table9_optimism_noise_sd": "{:.1f}",
            },
        ),
        "",
        "## Contribution balance on the circled raw optimum",
        "",
        f"For Uncheatable, the intercept is {contribution_totals['uncheatable']['intercept']:.4f}, utility channels "
        f"contribute {contribution_totals['uncheatable']['benefit']:+.4f}, and all modeled harms contribute "
        f"{contribution_totals['uncheatable']['harm']:+.4f}. For Table-9 the corresponding values are "
        f"{contribution_totals['table9']['intercept']:.4f}, {contribution_totals['table9']['benefit']:+.4f}, and "
        f"{contribution_totals['table9']['harm']:+.4f}. The imbalance is therefore inside the mechanistic head, not "
        "a plotting transform.",
        "",
        f"This failure is not universal across data plumbing: the least-optimistic current Uncheatable model on the "
        f"same row is `{best_by_target['uncheatable']['model_id']}` (miss "
        f"{best_by_target['uncheatable']['optimism']:.4f}), while `{best_by_target['table9']['model_id']}` misses "
        f"Table-9 by only {best_by_target['table9']['optimism']:.4f}. Different forms can recognize the harm from "
        "the identical checkpoint.",
        "",
        "## Matched-neighbor diagnosis",
        "",
        f"The circled raw optimum and its nearest fit row (`{nearest_row['row_name']}`) have nearly identical maximum "
        f"exposure ({circled_row['max_epoch']:.1f} versus {nearest_row['max_epoch']:.1f} epochs). What changes is "
        f"composition: top-3 mass rises from {nearest_row['top3_mass']:.3f} to {circled_row['top3_mass']:.3f}, while "
        f"broad-text coverage falls from {nearest_row['broad_text_coverage_ratio']:.3f} to "
        f"{circled_row['broad_text_coverage_ratio']:.3f} of proportional. The model predicts an Uncheatable change of "
        f"{circled_row['uncheatable_predicted'] - nearest_row['uncheatable_predicted']:+.4f} BPB and a Table-9 change "
        f"of {circled_row['table9_predicted'] - nearest_row['table9_predicted']:+.4f}; the observed changes are "
        f"{circled_row['uncheatable_observed'] - nearest_row['uncheatable_observed']:+.4f} and "
        f"{circled_row['table9_observed'] - nearest_row['table9_observed']:+.4f}. This rules out maximum replay count "
        "as the sufficient statistic and points to concentration or missing cross-family complementarity.",
        "",
        "## Geometry correlations",
        "",
        "Positive correlation means the feature increases dangerous optimism (observed BPB exceeds predicted BPB).",
        "",
    ]
    selected_correlations = correlations.loc[
        correlations["split"].eq("heldout_two_phase")
        & correlations["feature"].isin(
            [
                "support_distance",
                "top3_mass",
                "effective_bucket_count",
                "weighted_undercoverage",
                "broad_text_coverage_ratio",
                "phase_tv",
                "max_epoch",
            ]
        )
    ]
    lines.extend(
        markdown_table(
            selected_correlations,
            ["target", "feature", "n", "spearman_with_optimism"],
            {"n": "{:.0f}", "spearman_with_optimism": "{:.3f}"},
        )
    )
    lines.extend(["", "## Support-distance bins", ""])
    lines.extend(
        markdown_table(
            bins,
            ["target", "support_bin", "n", "rmse", "mean_optimism", "p90_optimism"],
            {"n": "{:.0f}", "rmse": "{:.4f}", "mean_optimism": "{:.4f}", "p90_optimism": "{:.4f}"},
        )
    )
    lines.extend(
        [
            "",
            "## Cross-model check",
            "",
            "`model_comparison.csv` evaluates every current Observatory surrogate on the same key rows. A failure "
            "shared by models with different phase parameterizations indicates missing coverage/complementarity or "
            "unsupported proposal geometry, rather than a single bad late-phase multiplier.",
            "",
            "## Interpretation",
            "",
            "1. The raw proposals are far outside the fit panel in concentration and coverage geometry. Separately, the "
            "fit panel also contains high-loss qsplit rows that are underpredicted, so there is both extrapolation "
            "failure "
            "and lower-severity in-support calibration compression; the two sets are not geometrically identical.",
            "2. Maximum epochs alone is insufficient. The 238-epoch Synth Math proposal is recognized as harmful much "
            "better than the 74%-on-three-code-buckets proposal; joint broad undercoverage is the distinguishing "
            "signal.",
            "3. Generic undercoverage and coverage-gate variants do not solve the problem. A within-family "
            "concentration channel improves heldout RMSE, but does not uniformly calibrate the raw optima or improve "
            "selection regret. The next clean model test should change how related-bucket utility saturates or how "
            "member replay aggregates, "
            "not merely add a generic distance penalty.",
            "4. Until that mechanism transfers, raw optimized mixtures from this surrogate are not deployment-safe. "
            "This is a surrogate-form diagnosis, not a reason to tune a generic support penalty on heldouts.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    bundle = json.loads(args.dashboard.read_text())
    registry = pd.read_csv(args.heldout_registry)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = prediction_frame(bundle, registry)
    comparison = model_comparison(bundle, frame)
    contributions = contribution_frame()
    correlations = correlation_frame(frame)
    bins = support_bins(frame)
    top = pd.concat(
        [frame.nlargest(20, f"{target}_optimism").assign(ranked_target=target) for target in TARGET_IDS],
        ignore_index=True,
    )

    frame.to_csv(args.output_dir / "row_diagnostics.csv", index=False)
    top.to_csv(args.output_dir / "top_optimistic_outliers.csv", index=False)
    comparison.to_csv(args.output_dir / "model_comparison.csv", index=False)
    contributions.to_csv(args.output_dir / "contribution_decomposition.csv", index=False)
    correlations.to_csv(args.output_dir / "geometry_correlations.csv", index=False)
    bins.to_csv(args.output_dir / "support_bins.csv", index=False)
    render(frame, args.output_dir / "outlier_geometry.html")
    write_report(frame, comparison, contributions, correlations, bins, args.output_dir / "report.md")

    key = frame.loc[frame["row_name"].isin(KEY_OUTLIERS)]
    print(
        key[
            [
                "row_name",
                "policy_family",
                "top3_mass",
                "broad_text_coverage_ratio",
                "max_epoch",
                "uncheatable_optimism",
                "uncheatable_optimism_noise_sd",
                "table9_optimism",
                "table9_optimism_noise_sd",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
