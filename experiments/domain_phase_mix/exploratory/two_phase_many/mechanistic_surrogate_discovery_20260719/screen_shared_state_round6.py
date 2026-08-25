# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy>=1.7",
#   "fsspec>=2025.7",
#   "gcsfs>=2025.7",
#   "numpy>=2.0",
#   "pandas>=2.2",
#   "plotly>=6.0",
#   "scikit-learn>=1.6",
#   "scipy>=1.15",
#   "tabulate>=0.9",
# ]
# ///
"""Test whether one retained-state shape transfers across core panels."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_hierarchical_coverage_grp_20260715 as hierarchical,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_production_grp_quality_variants as family_grp,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719/round6_shared_state"
SHAPE_SAMPLE_COUNT = 10
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}
DATASET_IDS = (
    hierarchical.DatasetId.THREE_HUNDRED_M_UNCHEATABLE,
    hierarchical.DatasetId.THREE_HUNDRED_M_TABLE9,
    hierarchical.DatasetId.DELPHI_3E18_UNCHEATABLE,
    hierarchical.DatasetId.DELPHI_3E18_TABLE9,
    hierarchical.DatasetId.PRODUCTION_UNCHEATABLE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def configs_for_shape(shape_index: int, shape: family_grp.Shape) -> list[hierarchical.Config]:
    return [
        hierarchical.Config(
            variant=hierarchical.Variant.HIERARCHICAL_PHASE_BUCKET_REPLAY,
            shape_index=shape_index,
            shape=shape,
            l2=l2,
            residual_shrink=residual_shrink,
            undercoverage_fraction=0.0,
            coverage_gate_ratio=0.0,
        )
        for l2 in hierarchical.L2_GRID
        for residual_shrink in hierarchical.RESIDUAL_SHRINK_GRID
    ]


def screen_dataset(
    dataset_id: hierarchical.DatasetId,
    shapes: tuple[family_grp.Shape, ...],
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    dataset = hierarchical.load_dataset(dataset_id)
    splits = hierarchical.split_indices(
        dataset,
        dataset_id,
        np.arange(dataset.n),
        hierarchical.SCREEN_SEED,
    )
    scale = float(np.std(dataset.target, ddof=1))
    rows: list[dict[str, Any]] = []
    predictions: dict[int, np.ndarray] = {}
    for shape_index, shape in enumerate(shapes):
        config, prediction, scores = hierarchical.score_configs(
            dataset,
            configs_for_shape(shape_index, shape),
            splits,
        )
        selected = min(scores, key=lambda row: (float(row["rmse"]), -float(row["spearman"])))
        rows.append(
            {
                "dataset": dataset_id.value,
                "shape_index": shape_index,
                **asdict(shape),
                "selected_l2": config.l2,
                "selected_residual_shrink": config.residual_shrink,
                "rmse": float(selected["rmse"]),
                "normalized_rmse": float(selected["rmse"]) / max(scale, 1e-12),
                "spearman": float(selected["spearman"]),
                "regret_at_1": float(selected["regret_at_1"]),
                "lower_tail_optimism": float(selected["lower_tail_optimism"]),
                "low_tail_rmse": float(selected["low_tail_rmse"]),
                "config_json": json.dumps(
                    {
                        "variant": config.variant.value,
                        "shape_index": config.shape_index,
                        "shape": asdict(config.shape),
                        "l2": config.l2,
                        "residual_shrink": config.residual_shrink,
                        "undercoverage_fraction": 0.0,
                        "coverage_gate_ratio": 0.0,
                    },
                    sort_keys=True,
                ),
            }
        )
        predictions[shape_index] = prediction
    return pd.DataFrame(rows), predictions


def plot_profiles(profiles: pd.DataFrame, shared_shape: int, output_path: Path) -> None:
    fig = go.Figure()
    for dataset, local in profiles.groupby("dataset", sort=False):
        fig.add_trace(
            go.Scatter(
                x=local["shape_index"],
                y=local["normalized_rmse"],
                mode="lines+markers",
                name=dataset,
            )
        )
    fig.add_vline(x=shared_shape, line={"color": "#b23a2b", "dash": "dash"})
    fig.update_layout(
        title="Shared retained-state shape profile",
        xaxis_title="Frozen shape candidate",
        yaxis_title="OOF RMSE / target standard deviation",
        template="plotly_white",
        width=1200,
        height=650,
    )
    fig.write_html(output_path, include_plotlyjs=True, config=PLOT_CONFIG)


def full_coefficients(
    dataset_id: hierarchical.DatasetId,
    config: hierarchical.Config,
) -> pd.DataFrame:
    dataset = hierarchical.load_dataset(dataset_id)
    model = hierarchical.fit_model(dataset, config, np.arange(dataset.n))
    design = hierarchical.build_design(dataset, config)
    return pd.DataFrame(
        {
            "dataset": dataset_id.value,
            "feature": design.names,
            "coefficient": model.coefficients,
            "active": model.coefficients > 1e-10,
        }
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shapes = family_grp.shape_candidates(family_grp.Variant.BUCKET_RESOLVED, SHAPE_SAMPLE_COUNT)
    if len(shapes) != 12:
        raise ValueError(f"Expected 12 frozen shape candidates, found {len(shapes)}")

    profiles = []
    prediction_by_dataset: dict[str, dict[int, np.ndarray]] = {}
    for dataset_id in DATASET_IDS:
        frame, predictions = screen_dataset(dataset_id, shapes)
        profiles.append(frame)
        prediction_by_dataset[dataset_id.value] = predictions
    profile = pd.concat(profiles, ignore_index=True)
    global_score = profile.groupby("shape_index", as_index=False)["normalized_rmse"].mean()
    shared_shape = int(global_score.sort_values(["normalized_rmse", "shape_index"]).iloc[0]["shape_index"])

    selected_rows = []
    coefficient_frames = []
    prediction_rows = []
    for dataset_id in DATASET_IDS:
        local = profile.loc[profile["dataset"].eq(dataset_id.value)]
        independent = local.sort_values(["rmse", "regret_at_1"]).iloc[0]
        shared = local.loc[local["shape_index"].eq(shared_shape)].iloc[0]
        selected_rows.append(
            {
                "dataset": dataset_id.value,
                "independent_shape_index": int(independent["shape_index"]),
                "independent_rmse": float(independent["rmse"]),
                "shared_shape_index": shared_shape,
                "shared_rmse": float(shared["rmse"]),
                "relative_rmse_change": float(shared["rmse"] / independent["rmse"] - 1.0),
                "shared_spearman": float(shared["spearman"]),
                "shared_regret_at_1": float(shared["regret_at_1"]),
                "shared_lower_tail_optimism": float(shared["lower_tail_optimism"]),
                "shared_low_tail_rmse": float(shared["low_tail_rmse"]),
                "shared_config_json": shared["config_json"],
            }
        )
        payload = json.loads(shared["config_json"])
        payload["variant"] = hierarchical.Variant(payload["variant"])
        payload["shape"] = family_grp.Shape(**payload["shape"])
        config = hierarchical.Config(**payload)
        coefficient_frames.append(full_coefficients(dataset_id, config))
        dataset = hierarchical.load_dataset(dataset_id)
        prediction = prediction_by_dataset[dataset_id.value][shared_shape]
        prediction_rows.extend(
            {
                "dataset": dataset_id.value,
                "row_index": index,
                "observed": float(observed),
                "predicted": float(predicted),
            }
            for index, (observed, predicted) in enumerate(zip(dataset.target, prediction, strict=True))
        )

    selected = pd.DataFrame(selected_rows)
    coefficients = pd.concat(coefficient_frames, ignore_index=True)
    predictions = pd.DataFrame(prediction_rows)
    profile.to_csv(args.output_dir / "shape_profiles.csv", index=False)
    global_score.to_csv(args.output_dir / "global_shape_scores.csv", index=False)
    selected.to_csv(args.output_dir / "shared_shape_metrics.csv", index=False)
    coefficients.to_csv(args.output_dir / "full_fit_coefficients.csv", index=False)
    predictions.to_csv(args.output_dir / "shared_shape_oof_predictions.csv", index=False)
    plot_profiles(profile, shared_shape, args.output_dir / "shared_shape_profiles.html")

    shared_shape_record = {"shape_index": shared_shape, **asdict(shapes[shared_shape])}
    report = [
        "# Round-six dimensionless multi-panel state replay",
        "",
        "One retained-state shape was selected by equal-panel mean normalized grouped-OOF RMSE across the five core panels. Every panel retained its own nonnegative response amplitudes, ridge, and hierarchical residual shrinkage.",
        "",
        "## Shared shape",
        "",
        "```json",
        json.dumps(shared_shape_record, indent=2, sort_keys=True),
        "```",
        "",
        "## Core gate",
        "",
        selected.to_markdown(index=False, floatfmt=".6f"),
        "",
        "Historical and adversarial outcomes were not read. The route proceeds only if every relative RMSE change is at most 5%.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n")
    print(selected.to_string(index=False))


if __name__ == "__main__":
    main()
