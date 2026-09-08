# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy",
#   "gcsfs",
#   "numpy",
#   "pandas",
#   "plotly",
#   "pyarrow",
#   "scikit-learn",
#   "scipy",
#   "tabulate",
# ]
# ///
"""Quantify uncertainty for exploratory fixed-budget aggregate/phase survivors.

The candidate forms and allocations in this file were chosen after inspecting
the fixed-budget development sweep. Results are therefore exploratory, not a
new confirmation test. Bootstrap resampling preserves each proposal series'
row count so large panels cannot erase between-series failure modes.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_fixed_budget_aggregate_phase_20260724 as fixed_budget,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_orthogonal_aggregate_phase_identification_20260724 as orthogonal,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_tied_backbone_phase_order_20260724 as backbone_benchmark,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    export_mixture_fit_observatory as observatory,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "fixed_budget_aggregate_phase_survivor_uncertainty_20260724"
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20260724
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


@dataclass(frozen=True)
class Survivor:
    """One post-sweep exploratory configuration."""

    target: str
    tied_count: int
    phase_model: str
    label: str


SURVIVORS = (
    Survivor("uncheatable", 80, "two_group_retention_hellinger_h0.001", "Unch 80 tied / 192 treatments"),
    Survivor("uncheatable", 160, "two_group_retention_hellinger_h0.001", "Unch 160 tied / 112 treatments"),
    Survivor("uncheatable", 200, "two_group_retention_hellinger_h0.001", "Unch 200 tied / 72 treatments"),
    Survivor("table9", 240, "phase_null_hellinger_h0.002", "Table-9 240 tied / 32 treatments"),
    Survivor("table9", 120, "global_retention_hellinger_h0.002", "Table-9 120 tied / 152 treatments"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    return parser.parse_args()


def proposal_strata(
    frame: pd.DataFrame,
    positions: np.ndarray,
    reference: Any,
    heldout_frame: pd.DataFrame,
) -> np.ndarray:
    """Return stable source strata for paired resampling."""
    strata = []
    for source, position in zip(frame["source"], positions, strict=True):
        if source == "original_two_phase_swarm":
            reference_row = reference.frame.iloc[int(position)]
            panel = reference_row.get("panel_source", reference_row.get("run_group", "fit_swarm"))
            strata.append(f"original_two_phase_swarm::{panel}")
            continue
        heldout_position = int(position) - reference.n
        heldout_row = heldout_frame.iloc[heldout_position]
        strata.append(f"archive::{heldout_row['training_series']}")
    return np.asarray(strata, dtype=object)


def prediction_frame(
    survivor: Survivor,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    target = survivor.target
    reference = observatory.load_delphi_3e18_fit_dataset(target)
    heldout_frame, heldout_weights = observatory.load_delphi_3e18_heldouts(reference)
    single, _single_evaluation_indices = observatory.load_delphi_3e18_single_phase_dataset(
        target,
        reference,
        heldout_frame,
        heldout_weights,
    )
    controls = fixed_budget.fiber_control_dataset(target, single)
    families = orthogonal.family_partition(single.domain_names)
    phase_rows = orthogonal.load_phase_rows(
        target,
        single.domain_names,
        float(np.mean(single.c0 / (single.c0 + single.c1))),
    )
    treatment_count = fixed_budget.TOTAL_CHECKPOINT_BUDGET - fixed_budget.CONTROL_COUNT - survivor.tied_count
    phase_indices = fixed_budget.phase_training_indices(phase_rows, treatment_count, seed)
    training = fixed_budget.aggregate_training_dataset(
        target,
        single,
        controls,
        survivor.tied_count,
        seed,
    )
    backbone = fixed_budget.aggregate_backbones(
        target,
        training,
        families,
        include_canonical=False,
    )[0]
    config = next(config for config in fixed_budget.phase_configs(target) if config.name == survivor.phase_model)
    phase_model = backbone_benchmark.fit_phase(
        phase_rows,
        phase_indices,
        backbone,
        config,
    )
    frame, weights, observed, positions = backbone_benchmark.coordinate_disjoint_combined_rows(
        target,
        reference,
        single,
        heldout_frame,
        heldout_weights,
    )
    aggregate_prediction = backbone.aggregate_predictor.predict(weights)
    combined_prediction = aggregate_prediction + phase_model.predict_delta(weights)

    all_tied_backbone = fixed_budget.aggregate_backbones(
        target,
        single,
        families,
        include_canonical=False,
    )[0]
    all_tied_prediction = all_tied_backbone.aggregate_predictor.predict(weights)

    cache_path = orthogonal.OBSERVATORY_CACHE / target / "two_phase" / "compact_retained_state.json"
    observatory_prediction = np.asarray(json.loads(cache_path.read_text())["prediction"], dtype=float)[positions]

    output = frame.copy()
    output["target"] = target
    output["seed"] = seed
    output["survivor"] = survivor.label
    output["proposal_stratum"] = proposal_strata(output, positions, reference, heldout_frame)
    output["observed"] = observed
    return output, {
        "mixed_aggregate_only": aggregate_prediction,
        "mixed_plus_phase": combined_prediction,
        "all_tied_physical": all_tied_prediction,
        "observatory_compact_retained_state": observatory_prediction,
    }


def metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    return orthogonal.regression_metrics(observed, predicted)


def paired_bootstrap(
    observed: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    strata: np.ndarray,
    draws: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    candidate_squared_error = (candidate - observed) ** 2
    reference_squared_error = (reference - observed) ** 2
    candidate_sums = np.zeros(draws, dtype=float)
    reference_sums = np.zeros(draws, dtype=float)
    for value in np.unique(strata):
        group = np.flatnonzero(strata == value)
        sampled = rng.integers(0, len(group), size=(draws, len(group)))
        candidate_sums += candidate_squared_error[group][sampled].sum(axis=1)
        reference_sums += reference_squared_error[group][sampled].sum(axis=1)
    candidate_mse = candidate_sums / len(observed)
    reference_mse = reference_sums / len(observed)
    mse_delta = candidate_mse - reference_mse
    rmse_delta = np.sqrt(candidate_mse) - np.sqrt(reference_mse)
    point_candidate_rmse = float(np.sqrt(np.mean((candidate - observed) ** 2)))
    point_reference_rmse = float(np.sqrt(np.mean((reference - observed) ** 2)))
    return {
        "candidate_rmse": point_candidate_rmse,
        "reference_rmse": point_reference_rmse,
        "rmse_delta": point_candidate_rmse - point_reference_rmse,
        "rmse_delta_ci_low": float(np.quantile(rmse_delta, 0.025)),
        "rmse_delta_ci_high": float(np.quantile(rmse_delta, 0.975)),
        "mse_delta": float(np.mean((candidate - observed) ** 2 - (reference - observed) ** 2)),
        "mse_delta_ci_low": float(np.quantile(mse_delta, 0.025)),
        "mse_delta_ci_high": float(np.quantile(mse_delta, 0.975)),
        "probability_candidate_better": float(np.mean(rmse_delta < 0.0)),
    }


def plot_series_error_delta(frame: pd.DataFrame, output_path: Path) -> None:
    figure = px.scatter(
        frame,
        x="reference",
        y="mean_squared_error_delta",
        color="target",
        facet_col="survivor",
        facet_col_wrap=2,
        hover_name="proposal_stratum",
        hover_data=["n", "seed"],
        title="Fixed-budget survivors: paired squared-error change by source series",
        color_discrete_sequence=px.colors.diverging.RdYlGn_r,
    )
    figure.add_hline(y=0.0, line_dash="dash", line_color="#183447")
    figure.update_layout(template="plotly_white", width=1400, height=1050)
    figure.write_html(output_path, include_plotlyjs="cdn", config=PLOT_CONFIG)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    series_rows: list[dict[str, Any]] = []
    for survivor_index, survivor in enumerate(SURVIVORS):
        for seed_index, seed in enumerate(fixed_budget.DEFAULT_SEEDS):
            frame, predictions = prediction_frame(survivor, seed)
            observed = frame["observed"].to_numpy(dtype=float)
            for model, prediction in predictions.items():
                local = frame.copy()
                local["model"] = model
                local["predicted"] = prediction
                local["residual"] = prediction - observed
                prediction_rows.append(local)
                metric_rows.append(
                    {
                        "target": survivor.target,
                        "survivor": survivor.label,
                        "seed": seed,
                        "model": model,
                        **metrics(observed, prediction),
                    }
                )
            comparisons = (
                ("mixed_plus_phase", "observatory_compact_retained_state"),
                ("mixed_aggregate_only", "observatory_compact_retained_state"),
                ("mixed_plus_phase", "mixed_aggregate_only"),
                ("mixed_plus_phase", "all_tied_physical"),
            )
            for comparison_index, (candidate_name, reference_name) in enumerate(comparisons):
                comparison_rows.append(
                    {
                        "target": survivor.target,
                        "survivor": survivor.label,
                        "seed": seed,
                        "candidate": candidate_name,
                        "reference": reference_name,
                        **paired_bootstrap(
                            observed,
                            predictions[candidate_name],
                            predictions[reference_name],
                            frame["proposal_stratum"].to_numpy(),
                            int(args.bootstrap_draws),
                            BOOTSTRAP_SEED + 100 * survivor_index + 10 * seed_index + comparison_index,
                        ),
                    }
                )
            for stratum, indices in frame.groupby("proposal_stratum", sort=True).indices.items():
                group = np.asarray(indices, dtype=int)
                candidate_error = predictions["mixed_plus_phase"][group] - observed[group]
                for reference_name in ("observatory_compact_retained_state", "mixed_aggregate_only"):
                    reference_error = predictions[reference_name][group] - observed[group]
                    series_rows.append(
                        {
                            "target": survivor.target,
                            "survivor": survivor.label,
                            "seed": seed,
                            "proposal_stratum": stratum,
                            "reference": reference_name,
                            "n": len(group),
                            "mean_squared_error_delta": float(np.mean(candidate_error**2 - reference_error**2)),
                        }
                    )
    comparisons = pd.DataFrame(comparison_rows)
    all_metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    series = pd.DataFrame(series_rows)
    comparisons.to_csv(args.output_dir / "paired_bootstrap.csv", index=False)
    all_metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    series.to_csv(args.output_dir / "series_error_delta.csv", index=False)
    plot_series_error_delta(series, args.output_dir / "series_error_delta.html")

    phase_rows = comparisons[
        comparisons["candidate"].eq("mixed_plus_phase") & comparisons["reference"].eq("mixed_aggregate_only")
    ]
    baseline_rows = comparisons[
        comparisons["candidate"].eq("mixed_plus_phase")
        & comparisons["reference"].eq("observatory_compact_retained_state")
    ]
    lines = [
        "# Fixed-budget aggregate/phase survivor uncertainty",
        "",
        "This is a post-sweep exploratory audit. Configuration selection is not covered by the intervals.",
        "Bootstrap draws preserve the row count of every proposal-series stratum.",
        "",
        "## Against Observatory Compact Retained State",
        "",
        baseline_rows.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Increment from the phase term",
        "",
        phase_rows.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Point metrics",
        "",
        all_metrics.to_markdown(index=False, floatfmt=".6f"),
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
