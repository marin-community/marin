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
"""Measure whether independently refit folds agree on frozen raw optima.

This audit does not optimize a new policy. It evaluates the already frozen
full-fit raw optima under models refit on grouped folds. It distinguishes
uncertain extrapolation from a stable, shared extrapolation error.
"""

from __future__ import annotations

import argparse
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
DEFAULT_OUTPUT = ARTIFACT_ROOT / "raw_optimum_crossfit_audit"
SOURCE_METRICS = RESEARCH_DIR / "reference_outputs/hierarchical_deficit_response_20260716/metrics.csv"
LINK_METRICS = RESEARCH_DIR / "reference_outputs/deficit_output_link_asymmetric_20260716/metrics.csv"
RAW_OPTIMUM_WEIGHTS = ARTIFACT_ROOT / "raw_optimum_audit/raw_optimum_weights.csv"
TARGETS = (
    base.DatasetId.DELPHI_3E18_UNCHEATABLE,
    base.DatasetId.DELPHI_3E18_TABLE9,
)
POLICIES = ("single_phase", "two_phase")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repeats", type=int, default=25)
    return parser.parse_args()


def optimum_weights(
    weights: pd.DataFrame,
    dataset_id: base.DatasetId,
    dataset: base.Dataset,
    policy: str,
) -> np.ndarray:
    selected = weights.loc[
        weights["dataset"].eq(dataset_id.value) & weights["model"].eq("baseline") & weights["policy"].eq(policy)
    ]
    ordered = selected.set_index("domain").loc[list(dataset.domains)]
    return np.stack([ordered["phase0_weight"].to_numpy(dtype=float), ordered["phase1_weight"].to_numpy(dtype=float)])


def fit_baseline(
    dataset_id: base.DatasetId,
    dataset: base.Dataset,
    source_metrics: pd.DataFrame,
    link_metrics: pd.DataFrame,
    indices: np.ndarray,
) -> collision.Model:
    deficit_config = output_link.selected_deficit_config(dataset_id, collision.DEFICIT_VARIANT, source_metrics)
    link_config = support.selected_link_config(dataset_id, link_metrics)
    return collision.fit_model(
        dataset,
        deficit_config,
        link_config,
        collision.Config(collision.Mechanism.BASELINE),
        indices,
    )


def audit_target(
    dataset_id: base.DatasetId,
    source_metrics: pd.DataFrame,
    link_metrics: pd.DataFrame,
    raw_weights: pd.DataFrame,
    repeats: int,
) -> pd.DataFrame:
    dataset = base.load_dataset(dataset_id)
    all_indices = np.arange(dataset.n)
    proportional = base.proportional_weights(dataset)
    proportional_policy = np.stack([proportional, proportional])
    best_index = int(np.argmin(dataset.target))
    best_observed_policy = dataset.weights[best_index]
    best_observed_bpb = float(dataset.target[best_index])
    optima = {policy: optimum_weights(raw_weights, dataset_id, dataset, policy) for policy in POLICIES}
    rows: list[dict[str, object]] = []
    for seed in range(repeats):
        splits = base.split_indices(dataset, dataset_id, all_indices, seed)
        train, test = splits[seed % len(splits)]
        model = fit_baseline(dataset_id, dataset, source_metrics, link_metrics, train)
        reference_predictions = model.predict(np.stack([proportional_policy, best_observed_policy]))
        for policy, optimum in optima.items():
            predicted = float(model.predict(optimum[None, :, :])[0])
            rows.append(
                {
                    "dataset": dataset_id.value,
                    "seed": seed,
                    "fold": seed % len(splits),
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "policy": policy,
                    "predicted_optimum_bpb": predicted,
                    "predicted_proportional_bpb": float(reference_predictions[0]),
                    "predicted_best_observed_bpb": float(reference_predictions[1]),
                    "observed_fit_frontier_bpb": best_observed_bpb,
                    "predicted_gain_vs_proportional": predicted - float(reference_predictions[0]),
                    "predicted_gain_vs_best_observed_policy": predicted - float(reference_predictions[1]),
                    "predicted_gap_below_observed_frontier": predicted - best_observed_bpb,
                }
            )
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby(["dataset", "policy"], as_index=False).agg(
        refits=("seed", "size"),
        mean_predicted_bpb=("predicted_optimum_bpb", "mean"),
        sd_predicted_bpb=("predicted_optimum_bpb", "std"),
        minimum_predicted_bpb=("predicted_optimum_bpb", "min"),
        maximum_predicted_bpb=("predicted_optimum_bpb", "max"),
        mean_gain_vs_proportional=("predicted_gain_vs_proportional", "mean"),
        sd_gain_vs_proportional=("predicted_gain_vs_proportional", "std"),
        fraction_predicts_gain_vs_proportional=("predicted_gain_vs_proportional", lambda x: float((x < 0).mean())),
        mean_gain_vs_best_observed=("predicted_gain_vs_best_observed_policy", "mean"),
        fraction_predicts_gain_vs_best_observed=(
            "predicted_gain_vs_best_observed_policy",
            lambda x: float((x < 0).mean()),
        ),
        mean_gap_below_observed_frontier=("predicted_gap_below_observed_frontier", "mean"),
        fraction_below_observed_frontier=("predicted_gap_below_observed_frontier", lambda x: float((x < 0).mean())),
    )


def render(frame: pd.DataFrame, output: Path) -> None:
    plot = frame.copy()
    plot["target"] = plot["dataset"].str.replace("delphi_3e18_", "", regex=False)
    figure = px.box(
        plot,
        x="policy",
        y="predicted_gain_vs_best_observed_policy",
        color="policy",
        facet_col="target",
        points="all",
        color_discrete_sequence=px.colors.diverging.RdYlGn_r,
        title="Cross-fit prediction at frozen raw optima relative to the best observed fit policy",
    )
    figure.add_hline(y=0.0, line_dash="dash", line_color="#475569")
    figure.update_yaxes(title="Predicted BPB difference (raw optimum - best observed policy)")
    figure.update_layout(template="plotly_white", showlegend=False)
    figure.write_html(output, include_plotlyjs="cdn")


def main() -> None:
    args = parse_args()
    for path in (SOURCE_METRICS, LINK_METRICS, RAW_OPTIMUM_WEIGHTS):
        gate.assert_sealed_absent(path)
    source_metrics = pd.read_csv(SOURCE_METRICS)
    link_metrics = pd.read_csv(LINK_METRICS)
    raw_weights = pd.read_csv(RAW_OPTIMUM_WEIGHTS)
    frames = [
        audit_target(dataset_id, source_metrics, link_metrics, raw_weights, args.repeats) for dataset_id in TARGETS
    ]
    frame = pd.concat(frames, ignore_index=True)
    summary = summarize(frame)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "crossfit_optimum_predictions.csv", index=False)
    summary.to_csv(args.output_dir / "crossfit_optimum_summary.csv", index=False)
    render(frame, args.output_dir / "crossfit_optimum_predictions.html")
    report = [
        "# Cross-fit raw-optimum audit",
        "",
        "The frozen full-fit raw optima are evaluated under independently refit grouped-fold models. No policy is "
        "re-optimized, no heldout target enters fitting, and no deployment regularizer is used.",
        "",
        summary.to_markdown(index=False, floatfmt=".6f"),
    ]
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
