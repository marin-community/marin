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
#   "tabulate",
# ]
# ///
"""Trace frozen surrogate value and support along proportional-to-optimum paths."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    benchmark_deficit_output_link_20260716 as output_link,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    benchmark_hierarchical_coverage_grp_20260715 as base,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    benchmark_hierarchical_deficit_response_20260716 as deficit,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260717 import (
    audit_convex_support as convex,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260717 import (
    freeze_baseline_gate as gate,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260717 import (
    screen_kish_collision_invariant as collision,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260717 import (
    screen_nested_support_invariants as support,
)

SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = SCRIPT_DIR.parent
ARTIFACT_ROOT = RESEARCH_DIR / "reference_outputs/mechanistic_surrogate_discovery_20260717"
DEFAULT_OUTPUT = ARTIFACT_ROOT / "raw_optimum_support_path_audit"
SOURCE_METRICS = RESEARCH_DIR / "reference_outputs/hierarchical_deficit_response_20260716/metrics.csv"
LINK_METRICS = RESEARCH_DIR / "reference_outputs/deficit_output_link_asymmetric_20260716/metrics.csv"
RAW_OPTIMUM_WEIGHTS = ARTIFACT_ROOT / "raw_optimum_audit/raw_optimum_weights.csv"
FIT_SUPPORT = ARTIFACT_ROOT / "convex_support_audit/fit_loo_convex_support.csv"
TARGETS = (
    base.DatasetId.DELPHI_3E18_UNCHEATABLE,
    base.DatasetId.DELPHI_3E18_TABLE9,
)
POLICIES = ("single_phase", "two_phase")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--path-points", type=int, default=101)
    return parser.parse_args()


def candidate_dataset(dataset: base.Dataset, weights: np.ndarray) -> base.Dataset:
    return replace(dataset, weights=weights, target=np.zeros(len(weights), dtype=float))


def optimum_weights(
    raw_weights: pd.DataFrame,
    dataset_id: base.DatasetId,
    dataset: base.Dataset,
    policy: str,
) -> np.ndarray:
    selected = raw_weights.loc[
        raw_weights["dataset"].eq(dataset_id.value)
        & raw_weights["model"].eq("baseline")
        & raw_weights["policy"].eq(policy)
    ]
    ordered = selected.set_index("domain").loc[list(dataset.domains)]
    return np.stack([ordered["phase0_weight"].to_numpy(dtype=float), ordered["phase1_weight"].to_numpy(dtype=float)])


def target_paths(
    dataset_id: base.DatasetId,
    source_metrics: pd.DataFrame,
    link_metrics: pd.DataFrame,
    raw_weights: pd.DataFrame,
    fit_support: pd.DataFrame,
    path_points: int,
) -> pd.DataFrame:
    dataset = base.load_dataset(dataset_id)
    config = output_link.selected_deficit_config(dataset_id, collision.DEFICIT_VARIANT, source_metrics)
    link_config = support.selected_link_config(dataset_id, link_metrics)
    model = collision.fit_model(
        dataset,
        config,
        link_config,
        collision.Config(collision.Mechanism.BASELINE),
        np.arange(dataset.n),
    )
    fit_design = deficit.build_design(dataset, config).values
    scale = fit_design.std(axis=0)
    active = scale > 1e-10
    mean = fit_design[:, active].mean(axis=0)
    standardized_fit = (fit_design[:, active] - mean) / scale[active]
    projector = convex.ConvexProjector(standardized_fit)
    fit_pairwise = np.linalg.norm(standardized_fit[:, None, :] - standardized_fit[None, :, :], axis=2)
    np.fill_diagonal(fit_pairwise, np.inf)
    fit_nearest = np.min(fit_pairwise, axis=1)
    fit_nearest_quantiles = {quantile: float(np.quantile(fit_nearest, quantile)) for quantile in (0.5, 0.75, 0.9, 0.95)}
    p95 = float(
        fit_support.loc[fit_support["dataset"].eq(dataset_id.value), "fit_loo_convex_hull_distance"].quantile(0.95)
    )
    proportional = base.proportional_weights(dataset)
    reference = np.stack([proportional, proportional])
    reference_prediction = float(model.predict(reference[None, :, :])[0])
    rows: list[dict[str, object]] = []
    for policy in POLICIES:
        optimum = optimum_weights(raw_weights, dataset_id, dataset, policy)
        fractions = np.linspace(0.0, 1.0, path_points)
        weights = np.asarray([(1.0 - fraction) * reference + fraction * optimum for fraction in fractions])
        predictions = model.predict(weights)
        designs = deficit.build_design(candidate_dataset(dataset, weights), config).values[:, active]
        standardized = (designs - mean) / scale[active]
        for fraction, prediction, design in zip(fractions, predictions, standardized, strict=True):
            distance, effective_support, _status = projector.project(design)
            nearest = int(np.argmin(np.linalg.norm(standardized_fit - design[None, :], axis=1)))
            rows.append(
                {
                    "dataset": dataset_id.value,
                    "policy": policy,
                    "path_fraction": fraction,
                    "predicted_bpb": float(prediction),
                    "predicted_gain_vs_proportional": float(prediction - reference_prediction),
                    "convex_hull_distance": distance,
                    "distance_over_fit_p95": distance / max(p95, 1e-12),
                    "inside_fit_loo_p95": bool(distance <= p95),
                    "convex_effective_support": effective_support,
                    "nearest_fit_row": str(dataset.frame.iloc[nearest]["run_name"]),
                    "nearest_fit_observed_bpb": float(dataset.target[nearest]),
                    "nearest_fit_design_distance": float(np.linalg.norm(standardized_fit[nearest] - design)),
                }
            )
            nearest_distance = float(np.linalg.norm(standardized_fit[nearest] - design))
            for quantile, threshold in fit_nearest_quantiles.items():
                suffix = f"p{int(100 * quantile)}"
                rows[-1][f"nearest_distance_over_fit_{suffix}"] = nearest_distance / max(threshold, 1e-12)
                rows[-1][f"inside_fit_nearest_{suffix}"] = bool(nearest_distance <= threshold)
    return pd.DataFrame(rows)


def summarize(paths: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (dataset, policy), local in paths.groupby(["dataset", "policy"], sort=False):
        ordered = local.sort_values("path_fraction")
        inside = ordered.loc[ordered["inside_fit_loo_p95"]]
        last_inside = inside.iloc[-1] if len(inside) else ordered.iloc[0]
        full = ordered.iloc[-1]
        record: dict[str, object] = {
            "dataset": dataset,
            "policy": policy,
            "last_in_support_path_fraction": float(last_inside["path_fraction"]),
            "predicted_gain_at_last_in_support": float(last_inside["predicted_gain_vs_proportional"]),
            "predicted_gain_at_raw_optimum": float(full["predicted_gain_vs_proportional"]),
            "fraction_of_predicted_gain_claimed_in_support": float(
                last_inside["predicted_gain_vs_proportional"]
                / min(float(full["predicted_gain_vs_proportional"]), -1e-12)
            ),
            "raw_optimum_distance_over_fit_p95": float(full["distance_over_fit_p95"]),
            "nearest_fit_observed_bpb_at_boundary": float(last_inside["nearest_fit_observed_bpb"]),
            "nearest_fit_design_distance_at_boundary": float(last_inside["nearest_fit_design_distance"]),
        }
        for quantile in (50, 75, 90, 95):
            local_inside = ordered.loc[ordered[f"inside_fit_nearest_p{quantile}"]]
            last_local = local_inside.iloc[-1] if len(local_inside) else ordered.iloc[0]
            record[f"last_local_p{quantile}_path_fraction"] = float(last_local["path_fraction"])
            record[f"predicted_gain_at_last_local_p{quantile}"] = float(last_local["predicted_gain_vs_proportional"])
            record[f"fraction_of_predicted_gain_in_local_p{quantile}"] = float(
                last_local["predicted_gain_vs_proportional"] / min(float(full["predicted_gain_vs_proportional"]), -1e-12)
            )
            record[f"nearest_fit_observed_bpb_at_local_p{quantile}"] = float(last_local["nearest_fit_observed_bpb"])
        record["raw_optimum_nearest_distance_over_fit_p95"] = float(full["nearest_distance_over_fit_p95"])
        rows.append(record)
    return pd.DataFrame(rows)


def render(paths: pd.DataFrame, output: Path) -> None:
    plot = paths.copy()
    plot["target"] = plot["dataset"].str.replace("delphi_3e18_", "", regex=False)
    figure = px.line(
        plot,
        x="path_fraction",
        y="predicted_gain_vs_proportional",
        color="policy",
        facet_col="target",
        markers=True,
        color_discrete_sequence=px.colors.diverging.RdYlGn_r,
        hover_data=[
            "distance_over_fit_p95",
            "inside_fit_loo_p95",
            "nearest_fit_row",
            "nearest_fit_observed_bpb",
        ],
        title="Frozen raw-optimum path: predicted gain and empirical-support exit",
    )
    outside = plot.loc[~plot["inside_fit_nearest_p95"]]
    figure.add_scatter(
        x=outside["path_fraction"],
        y=outside["predicted_gain_vs_proportional"],
        mode="markers",
        marker={"symbol": "x", "size": 7, "color": "#b91c1c"},
        name="outside local fit-support p95",
    )
    figure.add_hline(y=0.0, line_dash="dash", line_color="#475569")
    figure.update_layout(template="plotly_white")
    figure.write_html(output, include_plotlyjs="cdn")


def main() -> None:
    args = parse_args()
    for path in (SOURCE_METRICS, LINK_METRICS, RAW_OPTIMUM_WEIGHTS, FIT_SUPPORT):
        gate.assert_sealed_absent(path)
    source_metrics = pd.read_csv(SOURCE_METRICS)
    link_metrics = pd.read_csv(LINK_METRICS)
    raw_weights = pd.read_csv(RAW_OPTIMUM_WEIGHTS)
    fit_support = pd.read_csv(FIT_SUPPORT)
    paths = pd.concat(
        [
            target_paths(
                dataset_id,
                source_metrics,
                link_metrics,
                raw_weights,
                fit_support,
                args.path_points,
            )
            for dataset_id in TARGETS
        ],
        ignore_index=True,
    )
    summary = summarize(paths)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths.to_csv(args.output_dir / "raw_optimum_support_paths.csv", index=False)
    summary.to_csv(args.output_dir / "raw_optimum_support_path_summary.csv", index=False)
    render(paths, args.output_dir / "raw_optimum_support_paths.html")
    report = [
        "# Raw-optimum support-path audit",
        "",
        "Each path linearly interpolates policy weights from proportional to the frozen unregularized optimum. Support "
        "is audited in the mechanistic state, not used as a prediction feature or regularizer.",
        "",
        summary.to_markdown(index=False, floatfmt=".6f"),
    ]
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
