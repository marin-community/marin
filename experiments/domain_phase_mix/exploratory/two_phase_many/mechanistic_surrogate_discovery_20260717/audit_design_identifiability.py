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
"""Audit weakly excited directions in the frozen mechanistic design."""

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
DEFAULT_OUTPUT = ARTIFACT_ROOT / "design_identifiability_audit"
SOURCE_METRICS = RESEARCH_DIR / "reference_outputs/hierarchical_deficit_response_20260716/metrics.csv"
LINK_METRICS = RESEARCH_DIR / "reference_outputs/deficit_output_link_asymmetric_20260716/metrics.csv"
FAILURE_ATLAS = ARTIFACT_ROOT / "failure_atlas/heldout_failure_atlas.csv"
RAW_OPTIMUM_WEIGHTS = ARTIFACT_ROOT / "raw_optimum_audit/raw_optimum_weights.csv"
TARGETS = (
    base.DatasetId.DELPHI_3E18_UNCHEATABLE,
    base.DatasetId.DELPHI_3E18_TABLE9,
)
THRESHOLDS = (1e-2, 1e-3, 1e-4, 1e-6)
OPTIMISM_THRESHOLD = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def candidate_dataset(dataset: base.Dataset, weights: np.ndarray) -> base.Dataset:
    return replace(dataset, weights=weights, target=np.zeros(len(weights), dtype=float))


def weak_energy(coordinates: np.ndarray, relative_singular_values: np.ndarray, threshold: float) -> float:
    denominator = float(np.square(coordinates).sum())
    if denominator <= 1e-20:
        return 0.0
    return float(np.square(coordinates[relative_singular_values <= threshold]).sum() / denominator)


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


def audit_target(
    dataset_id: base.DatasetId,
    source_metrics: pd.DataFrame,
    link_metrics: pd.DataFrame,
    atlas: pd.DataFrame,
    raw_weights: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    design = deficit.build_design(dataset, config)
    scale = design.values.std(axis=0)
    active = scale > 1e-10
    mean = design.values[:, active].mean(axis=0)
    standardized = (design.values[:, active] - mean) / scale[active]
    _u, singular_values, right = np.linalg.svd(standardized, full_matrices=False)
    relative = singular_values / singular_values[0]
    active_names = np.asarray(design.names, dtype=object)[active]
    coefficient_standardized = model.coefficients[active] * scale[active]
    coefficient_coordinates = right @ coefficient_standardized

    spectrum_rows = [
        {
            "dataset": dataset_id.value,
            "direction": index + 1,
            "singular_value": singular_value,
            "relative_singular_value": relative[index],
            "coefficient_coordinate": coefficient_coordinates[index],
            "coefficient_energy_fraction": float(
                coefficient_coordinates[index] ** 2 / max(np.square(coefficient_coordinates).sum(), 1e-20)
            ),
        }
        for index, singular_value in enumerate(singular_values)
    ]
    loading_rows: list[dict[str, object]] = []
    for direction in np.argsort(relative)[: min(8, len(relative))]:
        order = np.argsort(-np.abs(right[direction]))[:8]
        for rank, feature_index in enumerate(order, start=1):
            loading_rows.append(
                {
                    "dataset": dataset_id.value,
                    "direction": int(direction + 1),
                    "relative_singular_value": relative[direction],
                    "loading_rank": rank,
                    "feature": str(active_names[feature_index]),
                    "loading": right[direction, feature_index],
                }
            )

    heldout = base.heldout_data(dataset_id, dataset)
    if heldout is None:
        raise ValueError(dataset_id)
    heldout_frame, heldout_weights, _heldout_target = heldout
    baseline = atlas.loc[atlas["dataset"].eq(dataset_id.value) & atlas["mechanism"].eq("baseline")].copy()
    retained = heldout_frame["wandb_run_name"].astype(str).isin(baseline["row_id"].astype(str)).to_numpy()
    heldout_frame = heldout_frame.loc[retained].reset_index(drop=True)
    heldout_design = deficit.build_design(candidate_dataset(dataset, heldout_weights[retained]), config)
    heldout_standardized = (heldout_design.values[:, active] - mean) / scale[active]
    baseline = baseline.set_index("row_id")
    policy_rows: list[dict[str, object]] = []
    for index, row in heldout_frame.iterrows():
        row_id = str(row["wandb_run_name"])
        atlas_row = baseline.loc[row_id]
        coordinates = heldout_standardized[index] @ right.T
        record: dict[str, object] = {
            "dataset": dataset_id.value,
            "row_id": row_id,
            "kind": "heldout",
            "training_series": atlas_row["training_series"],
            "optimism": float(atlas_row["optimism"]),
            "severe_optimism": bool(float(atlas_row["optimism"]) > OPTIMISM_THRESHOLD),
            "standardized_state_norm": float(np.linalg.norm(heldout_standardized[index])),
        }
        for threshold in THRESHOLDS:
            record[f"weak_energy_le_{threshold:g}"] = weak_energy(coordinates, relative, threshold)
        policy_rows.append(record)

    for policy in ("single_phase", "two_phase"):
        weights = optimum_weights(raw_weights, dataset_id, dataset, policy)
        optimum_design = deficit.build_design(candidate_dataset(dataset, weights[None, :, :]), config)
        standardized_optimum = (optimum_design.values[0, active] - mean) / scale[active]
        coordinates = standardized_optimum @ right.T
        record = {
            "dataset": dataset_id.value,
            "row_id": f"raw_optimum::{policy}",
            "kind": "raw_optimum",
            "training_series": policy,
            "optimism": float("nan"),
            "severe_optimism": False,
            "standardized_state_norm": float(np.linalg.norm(standardized_optimum)),
        }
        for threshold in THRESHOLDS:
            record[f"weak_energy_le_{threshold:g}"] = weak_energy(coordinates, relative, threshold)
        policy_rows.append(record)

    rank_rows = [
        {
            "dataset": dataset_id.value,
            "active_features": int(active.sum()),
            "numerical_rank": int((relative > threshold).sum()),
            "relative_threshold": threshold,
            "condition_number_above_threshold": float(
                singular_values[0] / singular_values[relative > threshold][-1]
                if np.any(relative > threshold)
                else float("inf")
            ),
            "coefficient_energy_in_weak_subspace": float(
                np.square(coefficient_coordinates[relative <= threshold]).sum()
                / max(np.square(coefficient_coordinates).sum(), 1e-20)
            ),
        }
        for threshold in THRESHOLDS
    ]
    return (
        pd.DataFrame(spectrum_rows),
        pd.DataFrame(loading_rows),
        pd.DataFrame(policy_rows),
        pd.DataFrame(rank_rows),
    )


def summarize_policies(policies: pd.DataFrame) -> pd.DataFrame:
    heldout = policies.loc[policies["kind"].eq("heldout")].copy()
    metrics = [column for column in policies.columns if column.startswith("weak_energy_")]
    rows: list[dict[str, object]] = []
    for (dataset, severe), local in heldout.groupby(["dataset", "severe_optimism"], sort=False):
        record: dict[str, object] = {
            "dataset": dataset,
            "severe_optimism": severe,
            "policies": len(local),
            "median_state_norm": float(local["standardized_state_norm"].median()),
        }
        for metric in metrics:
            record[f"median_{metric}"] = float(local[metric].median())
            record[f"p90_{metric}"] = float(local[metric].quantile(0.9))
        rows.append(record)
    return pd.DataFrame(rows)


def render(policies: pd.DataFrame, output: Path) -> None:
    plot = policies.copy()
    plot["category"] = np.where(
        plot["kind"].eq("raw_optimum"),
        plot["training_series"],
        np.where(plot["severe_optimism"], "severe heldout", "ordinary heldout"),
    )
    figure = px.box(
        plot,
        x="category",
        y="weak_energy_le_0.001",
        color="category",
        facet_col="dataset",
        points="all",
        color_discrete_sequence=px.colors.diverging.RdYlGn_r,
        title="Policy-state energy in fit-design directions with relative singular value <= 1e-3",
    )
    figure.update_layout(template="plotly_white", showlegend=False)
    figure.write_html(output, include_plotlyjs="cdn")


def main() -> None:
    args = parse_args()
    for path in (SOURCE_METRICS, LINK_METRICS, FAILURE_ATLAS, RAW_OPTIMUM_WEIGHTS):
        gate.assert_sealed_absent(path)
    source_metrics = pd.read_csv(SOURCE_METRICS)
    link_metrics = pd.read_csv(LINK_METRICS)
    atlas = pd.read_csv(FAILURE_ATLAS)
    raw_weights = pd.read_csv(RAW_OPTIMUM_WEIGHTS)
    outputs = [audit_target(dataset_id, source_metrics, link_metrics, atlas, raw_weights) for dataset_id in TARGETS]
    spectrum = pd.concat([output[0] for output in outputs], ignore_index=True)
    loadings = pd.concat([output[1] for output in outputs], ignore_index=True)
    policies = pd.concat([output[2] for output in outputs], ignore_index=True)
    ranks = pd.concat([output[3] for output in outputs], ignore_index=True)
    summary = summarize_policies(policies)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    spectrum.to_csv(args.output_dir / "design_singular_spectrum.csv", index=False)
    loadings.to_csv(args.output_dir / "weak_direction_loadings.csv", index=False)
    policies.to_csv(args.output_dir / "policy_weak_direction_energy.csv", index=False)
    ranks.to_csv(args.output_dir / "design_numerical_rank.csv", index=False)
    summary.to_csv(args.output_dir / "policy_weak_direction_summary.csv", index=False)
    render(policies, args.output_dir / "design_identifiability.html")
    report = [
        "# Frozen-design identifiability audit",
        "",
        "Weak directions are diagnosed from the singular spectrum of the centered, standardized fit design. They "
        "are not added as model features or deployment penalties.",
        "",
        "## Numerical rank and fitted response energy",
        "",
        ranks.to_markdown(index=False, floatfmt=".6g"),
        "",
        "## Heldout weak-direction energy",
        "",
        summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Raw optima",
        "",
        policies.loc[policies["kind"].eq("raw_optimum")].to_markdown(index=False, floatfmt=".6f"),
    ]
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n")
    print(ranks.to_string(index=False))
    print(summary.to_string(index=False))
    print(policies.loc[policies["kind"].eq("raw_optimum")].to_string(index=False))


if __name__ == "__main__":
    main()
